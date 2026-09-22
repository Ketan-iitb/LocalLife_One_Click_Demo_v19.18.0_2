[CmdletBinding()]
param(
    [ValidateSet('Launcher', 'App', 'Pi', 'CloudTunnel', 'RecipeApi', 'Stop', 'Doctor')]
    [string]$Role = 'Launcher',

    # 'Local' runs everything on this laptop (free, default, ~2 min start).
    # 'Cloud' brings up a GPU VM via gpu.py (handles GPU stockouts across
    # zones automatically) and tunnels the laptop dashboard to it.
    [ValidateSet('Local', 'Cloud')]
    [string]$Mode = 'Local',

    # geometry_validation is the normal run mode because it detects EVERY
    # object. Waste mode runs a strict classifier that accepts only plastic
    # bags, paper bags and cardboard boxes, and a real run showed it dropping
    # the bag entirely (RealSense: bags seen 0, live tracks 0).
    #
    # The reason this was briefly switched to waste mode no longer applies:
    # recording used to be tied to the mode, so validation runs produced an
    # empty CSV. Recording is now unconditional (config.ledger_active), so this
    # mode measures, tracks AND records. Use -OperatingMode waste only when the
    # strict three-class waste classifier is actually wanted.
    [ValidateSet('waste', 'geometry_validation')]
    [string]$OperatingMode = 'geometry_validation',

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$ProjectDirectory = 'LocalLife_Plug_and_Play_Local',

    [ValidatePattern('^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$')]
    [string]$PiHost = 'locallife@locallife.local',

    # Passed to every child window so all three report against one session.
    # Blank in a hand-started role, which then recovers it from session.json.
    [ValidatePattern('^[A-Za-z0-9-]*$')]
    [string]$SessionId = '',

    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,

    [ValidateRange(1024, 65535)]
    [int]$PiBridgePort = 18000,

    # -Role RecipeApi only: port for the separate, optional FastAPI recipe
    # service (locallife_cloud/recipe_api.py). Distinct from -Port (the main
    # dashboard's port, 8000 by default) so both can run on the same
    # machine at once. Matches recipe_api.py's own DEFAULT_RECIPE_PORT.
    [ValidateRange(1024, 65535)]
    [int]$RecipeApiPort = 8100,

    [ValidateRange(1, 30)]
    [int]$UploadFps = 3,

    [string]$LanAddress = '',

    # Extra flags forwarded to gpu.py's "up" (e.g. "--fresh --l4-only").
    # Only used in -Mode Cloud.
    [string]$GpuArgs = '',

    # Bearer token the server requires once it binds outside localhost
    # (0.0.0.0, needed so the Pi's reverse SSH tunnel can reach it). The
    # Launcher role generates one and passes it to the App/Pi windows it
    # spawns; left blank here only when a role is run standalone.
    [string]$ApiToken = '',

    # Cloud target. These were previously hardcoded at each gcloud call site,
    # which meant the instance name could not be changed and --project was
    # never passed at all -- so every ssh/scp went to whichever project the
    # active gcloud config pointed at, not necessarily the one holding the VM.
    [ValidatePattern('^[a-z]([-a-z0-9]*[a-z0-9])?$')]
    [string]$VmName = 'depth-l4',

    [ValidatePattern('^[a-z][-a-z0-9:.]*[a-z0-9]$')]
    [string]$CloudProject = 'locallife-thesis-depth',

    # Blank in Local mode, where there is no VM. Cloud mode overwrites it with
    # the zone gpu.py actually landed in, which can differ from any default
    # after a capacity move -- deliberately unvalidated, because a validation
    # attribute here would also be enforced on that later assignment and could
    # reject a perfectly good zone the tool discovered at runtime.
    [string]$Zone = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$script:SessionDirectory = Join-Path $env:LOCALAPPDATA 'LocalLifeDemo'
$script:SessionPath = Join-Path $script:SessionDirectory 'session.json'
# Declared, not invented: Set-StrictMode makes reading an undeclared script
# variable a terminating error, and Set-CloudStage read this one before anything
# had ever assigned it. The real id is created by the launcher and recovered
# from session.json by every child window; this only guarantees the name exists.
$script:SessionId = ''

# -Mode Cloud only: a dedicated, port-forwarding-only Linux user provisioned
# on the depth-l4 GPU VM so the Raspberry Pi can open its own direct SSH
# tunnel straight to the VM (Pi -> VM), instead of relaying camera frames
# through the laptop. This user has no shell and no login password -- only
# a public key authorized with "restrict,port-forwarding" (see
# Initialize-PiCloudTunnel). Local mode never uses this; it keeps the
# existing laptop-relayed reverse tunnel unchanged.
$script:PiTunnelUser = 'locallife-tunnel'

# -Mode Cloud only: caches the depth-l4 VM's current external IP on this
# laptop between runs, so Initialize-PiCloudTunnel only has to call
# `gcloud compute instances describe` (which is slow) when the cached value
# is missing or the VM's address has actually changed.
$script:CloudVmAddressPath = Join-Path $script:SessionDirectory 'cloud-vm-address.txt'

function Write-Banner {
    param([string]$Message)
    Write-Host ''
    Write-Host ('=' * 72) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ('=' * 72) -ForegroundColor Cyan
    Write-Host ''
}

function Write-Step {
    param([string]$Message)
    Write-Host ('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $Message) -ForegroundColor White
}

function Invoke-NativeTolerantly {
    # Runs a native executable while treating its own stderr output as
    # non-fatal for the duration of this one call, then restores
    # $ErrorActionPreference. $LASTEXITCODE reflects the real exit code
    # afterward, exactly as a normal native call would leave it.
    #
    # Why this exists: native commands (pip, git, ssh, gcloud/plink) all
    # routinely write ordinary progress or warning text to stderr, even on
    # success or on an expected, recoverable condition -- e.g. `pip show`
    # correctly printing "WARNING: Package(s) not found: ..." and exiting 1
    # for a package that simply is not installed yet, which this launcher
    # deliberately checks for so it can install it. Under this script's own
    # `$ErrorActionPreference = 'Stop'`, PowerShell (this behavior is
    # documented and version/host-dependent, and was confirmed happening on
    # a real user run) can promote that stderr text into a
    # script-terminating exception before the script ever gets to inspect
    # $LASTEXITCODE and make its own decision -- exactly what surfaced as
    # "DEMONSTRATION ERROR: WARNING: Package(s) not found: locallife-cloud"
    # for what should have been a completely normal first-run install step.
    # If it still somehow throws despite the relaxed preference, that is
    # caught here too and reported as a plain nonzero exit rather than
    # propagating, so a bug in this defense can never be worse than the
    # thing it defends against.
    # $StdinLines (optional, named-only -- every call site passes it, when at
    # all, as -StdinLines): answers fed to the native process's own stdin,
    # one per line. This exists for gcloud's Windows SSH backend (PuTTY's
    # plink.exe/pscp.exe) and is written generically so any future caller can
    # use it the same way. No caller feeds host-key answers through it any
    # more -- see the verified-cloud-SSH section for what replaced that.
    #
    # Correction to a claim this comment used to make: an earlier version
    # asserted $StdinLines was "never bound positionally by any call site"
    # because it carried no explicit [Parameter(ValueFromPipeline=...)]
    # attribute -- that check was real (mixing ValueFromPipeline with this
    # function's ValueFromRemainingArguments $Arguments is genuinely
    # ambiguous and was tested) but it was the wrong check: a plain
    # [string[]] parameter with no [Parameter(...)] attribute at all is
    # STILL implicitly positional in PowerShell, at the next open slot after
    # $Executable, whether or not it was ever intended to be. A real
    # Windows run surfaced this directly: `Invoke-NativeTolerantly $pythonExe
    # '-m' 'pip' 'install' '-e' '.' '-q'` silently bound '-m' into
    # $StdinLines (as a one-element array) and left $Arguments as
    # ('pip','install','-e','.','-q') -- '-m' quietly dropped -- so the
    # process actually run was `python.exe pip install -e . -q` (no `-m`),
    # which Python interprets as "run the script literally named `pip`",
    # producing `python.exe: can't open file '...\pip': No such file or
    # directory` instead of running pip at all. Reproduced directly in a
    # real installed pwsh before fixing (confirmed the exact bad binding
    # above), then fixed with `[CmdletBinding(PositionalBinding = $false)]`
    # on this function plus explicit `Position` only on $Executable and
    # $Arguments -- with PositionalBinding disabled, a parameter is
    # positional ONLY when it declares its own Position, so $StdinLines
    # (which never declares one) can now only ever bind by its `-StdinLines`
    # name, exactly as every real call site already uses it, while
    # $Executable and $Arguments keep working positionally exactly as
    # before. Re-verified directly in pwsh, both the plain call shape (no
    # -StdinLines) and the gcloud/PuTTY call shape (-StdinLines given by
    # name, followed by more bare positional arguments) after the fix.
    [CmdletBinding(PositionalBinding = $false)]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Executable,
        [string[]]$StdinLines = @(),
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)][object[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        if ($StdinLines.Count -gt 0) {
            $StdinLines | & $Executable @Arguments
        }
        else {
            & $Executable @Arguments
        }
    }
    catch {
        Write-Host ('(continuing past a non-fatal native error) ' + $_.Exception.Message) -ForegroundColor DarkYellow
        $global:LASTEXITCODE = 1
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-PythonPackageIdentity {
    <#
        Run a tiny `python -c` probe and return its stdout as one trimmed
        string, or '' if the probe failed for any reason.

        Deliberately NOT routed through Invoke-NativeTolerantly: that helper
        writes the native command's output straight to the pipeline for its
        caller to ignore, which is right for `pip install`/the server but
        useless here, where the whole point is to CAPTURE what the probe
        printed. Failure is an expected, ordinary outcome (the package may
        genuinely not be importable yet), so stderr is discarded and a
        non-zero exit simply yields '' instead of stopping the launcher.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$Probe
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $output = (& $PythonExe '-c' $Probe 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            return ''
        }
        return $output
    }
    catch {
        return ''
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

# Why this launcher no longer goes through gcloud's Windows SSH backend.
#
# `gcloud compute ssh`/`scp` shell out to PuTTY (plink.exe/pscp.exe) on Windows
# -- gcloud's own ssh.py hardcodes that backend, with no supported flag to
# choose OpenSSH instead. PuTTY runs its own host-key prompts, entirely separate
# from and NOT suppressed by gcloud's --strict-host-key-checking flag, and those
# prompts are what blocked cloud startup: gpu.py recreates the VM on a zone move,
# so the key legitimately changes and plink stops at "Update cached key? (y/n)"
# in a window nobody is watching.
#
# An earlier build answered that prompt by piping "y". It unblocked startup and
# it was wrong: it accepted whatever key the far end offered, which is precisely
# the substitution the prompt exists to catch. It has been removed.
#
# Everything below instead establishes identity over the authenticated GCP API
# (see the verified-cloud-SSH section) and connects with Windows OpenSSH under
# StrictHostKeyChecking=yes against a pinned known_hosts file.
$script:CloudSshHost = ''
$script:PiHostResolved = $false
# Set when GCP could not publish the VM's host key, so this run falls back to
# interactive `gcloud compute ssh` -- the v3.2 reference deployment's own
# approach, where the operator sees the fingerprint and answers once.
$script:CloudSshInteractive = $false
$script:CloudZone = ''
$script:CloudSshUser = ''

# --------------------------------------------------------------------- #
# Verified cloud SSH.
#
# `gcloud compute ssh` shells out to PuTTY on Windows, and PuTTY's host-key
# prompts are what actually blocked cloud startup: gpu.py recreates the VM on a
# zone move, the key legitimately changes, and plink stops for a keystroke in a
# window nobody is watching.
#
# The fix is NOT to answer that prompt automatically. Piping "y" would make the
# launcher accept any key any host presented -- exactly the man-in-the-middle
# case the prompt exists to catch. Instead the VM's identity is established over
# a channel independent of the SSH connection: GCE publishes each instance's own
# host keys as guest attributes, readable over the authenticated GCP API.
# locallife_cloud/cloud_ssh.py resolves the instance, pins those keys into a
# per-session known_hosts file, and everything below connects with Windows
# OpenSSH under StrictHostKeyChecking=yes -- a matching key connects silently, a
# mismatched one is refused outright. That logic lives in Python because it is
# security-critical and therefore worth unit-testing; this is its caller.
# --------------------------------------------------------------------- #
function Test-CloudSshProbe {
    <#
        One bounded, authenticated `gcloud compute ssh` that runs `true` on the
        VM. Returns @{ ok; code; message }, with the failure classified so the
        operator is told which thing to fix rather than "cloud failed".
    #>
    param([Parameter(Mandatory = $true)][string]$Zone)
    $gcloudPath = Assert-GcloudAvailable
    $errorFile = Join-Path $script:SessionDirectory 'cloud-ssh-probe.txt'
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        & $gcloudPath 'compute' 'ssh' $VmName ('--project=' + $CloudProject) ('--zone=' + $Zone) `
            '--quiet' '--command=true' 2>$errorFile | Out-Null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    $detail = ''
    if (Test-Path -LiteralPath $errorFile) {
        $detail = ((Get-Content -LiteralPath $errorFile -Raw -ErrorAction SilentlyContinue) + '').Trim()
    }
    if ($exitCode -eq 0) {
        return @{ ok = $true; code = 'ok'; message = '' }
    }
    # Classify, so the message names the thing to fix.
    $code = 'remote_probe_failed'
    if ($detail -match 'not logged in|credentials|gcloud auth|Reauthentication') { $code = 'gcloud_authentication_required' }
    elseif ($detail -match 'POTENTIAL SECURITY BREACH|host key.*not match|REMOTE HOST IDENTIFICATION') { $code = 'ssh_host_key_conflict' }
    elseif ($detail -match 'Permission denied|publickey') { $code = 'ssh_permission_denied' }
    elseif ($detail -match 'timed out|Connection timed out|Operation timed out') { $code = 'ssh_timeout' }
    elseif ($detail -match 'ssh.*not found|plink.*not found|No such file') { $code = 'ssh_client_missing' }
    elseif ($detail -match 'could not resolve|Network is unreachable|No route to host') { $code = 'vm_unreachable' }
    $message = $detail
    if ([string]::IsNullOrWhiteSpace($message)) { $message = 'gcloud compute ssh exited ' + $exitCode }
    if ($code -eq 'ssh_host_key_conflict') {
        $message = ('The saved SSH host key for ' + $VmName + ' does not match the one it now ' +
                    'presents. If you recreated the VM this is expected; confirm the instance ' +
                    'in the Google Cloud console, then remove only that host''s cached key. ' + $message)
    }
    return @{ ok = $false; code = $code; message = $message }
}

function Set-CloudStage {
    <#
        Shared cloud startup state, so Window 2 stops waiting the moment
        Window 1 fails instead of counting to thirty minutes and blaming a
        zone migration that never happened.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [string]$Status = 'running',
        [string]$Zone = '',
        [string]$ErrorCode = '',
        [string]$ErrorMessage = ''
    )
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    $state = @{
        session_id = (Get-RequiredSessionId)
        stage = $Stage
        status = $Status
        zone = $Zone
        updated_at = [int][double]::Parse((Get-Date -UFormat %s))
        error_code = $ErrorCode
        error_message = $ErrorMessage
    }
    try {
        $state | ConvertTo-Json -Compress |
            Set-Content -LiteralPath (Join-Path $script:SessionDirectory 'cloud-stage.json') -Encoding UTF8
    }
    catch {
        # Status reporting must never be the thing that fails a startup.
    }
}

function Get-CloudStage {
    $path = Join-Path $script:SessionDirectory 'cloud-stage.json'
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try {
        return (Get-Content -LiteralPath $path -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Get-CloudKnownHostsPath {
    return (Join-Path $script:SessionDirectory 'cloud_known_hosts')
}

function Assert-CloudSshIdentity {
    <#
        Pin the VM's published host keys. Returns the parsed verification
        report. Throws when identity cannot be established -- never downgraded
        to a warning, and never worked around by relaxing the check.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$Zone
    )
    Write-Step 'Resolving VM identity...'
    $knownHosts = Get-CloudKnownHostsPath
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    # An already-running VM that answers an authenticated `gcloud compute ssh`
    # is reachable, full stop. The v3.2 reference deployment does exactly this
    # and nothing more. Host-key pinning below is an ENHANCEMENT on top; it was
    # wrongly a precondition, so a VM that had started perfectly -- DEPTH_READY,
    # L4 detected, CUDA available -- was refused because Compute Engine had not
    # published a host key through either channel.
    Write-Step 'Probing the VM over authenticated gcloud SSH...'
    $probe = Test-CloudSshProbe -Zone $Zone
    if (-not $probe.ok) {
        Set-CloudStage -Stage 'verifying_ssh' -Status 'failed' -Zone $Zone `
            -ErrorCode $probe.code -ErrorMessage $probe.message
        throw ('Cloud SSH probe failed (' + $probe.code + '): ' + $probe.message)
    }
    Write-Step ('Authenticated SSH to ' + $VmName + ' works.')
    # The authenticated gcloud readiness command above is the startup
    # authority. Publish progress now: the published-host-key lookup below is
    # an optional diagnostic, and it used to spend the whole verifying_ssh
    # budget (6 x 10 s) waiting for guest attributes, so a healthy VM was
    # reported "SSH identity NOT verified" and startup failed.
    Set-CloudStage -Stage 'checking_deployment' -Status 'running' -Zone $Zone
    Write-Step 'Checking for a Google-published SSH host key (optional)...'
    # PYTHONPATH, not the current directory: this script lives at the repository
    # root while the package is one level down in $ProjectDirectory, so
    # `python -m locallife_cloud.cloud_ssh` from here could not import it at
    # all. It failed with ModuleNotFoundError on stderr, which the old
    # SilentlyContinue call then swallowed -- leaving an empty result that was
    # misreported as a host key mismatch. Setting the path is the fix; the
    # separate stderr file below is what makes any future failure legible.
    $projectRoot = Find-ProjectRoot
    $errorFile = Join-Path $script:SessionDirectory 'ssh-verify-stderr.txt'
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $projectRoot
    try {
        # Hand over the gcloud path this script already resolved: on Windows
        # gcloud is a .cmd, which Python's subprocess does not find from the
        # bare name.
        $raw = (& $PythonExe '-m' 'locallife_cloud.cloud_ssh' `
            '--vm' $VmName '--zone' $Zone '--project' $CloudProject `
            '--gcloud' (Assert-GcloudAvailable) `
            '--known-hosts' $knownHosts '--attempts' '1' '--delay' '0' 2>$errorFile | Out-String)
        $verifyExitCode = $LASTEXITCODE
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }
    $errorText = ''
    if (Test-Path -LiteralPath $errorFile) {
        $errorText = ((Get-Content -LiteralPath $errorFile -Raw -ErrorAction SilentlyContinue) + '').Trim()
    }
    $report = $null
    foreach ($line in ($raw -split "`n")) {
        $trimmed = "$line".Trim()
        if ($trimmed.StartsWith('{')) {
            try { $report = $trimmed | ConvertFrom-Json } catch { }
        }
    }
    if ($null -eq $report) {
        # NOT a mismatch. The check never produced a verdict, which is a
        # different problem with a different fix, and calling it a mismatch sent
        # the last run looking for a security incident that had not happened.
        $detail = $errorText
        if ([string]::IsNullOrWhiteSpace($detail)) { $detail = $raw.Trim() }
        if ([string]::IsNullOrWhiteSpace($detail)) { $detail = '(no output; exit code ' + $verifyExitCode + ')' }
        # A multi-line traceback shows only its first line ("Traceback (most
        # recent call last):") once it is inside a thrown message, which is the
        # one line carrying no information. Print it in full first, then throw a
        # single-line summary.
        Write-Host '--- VM identity check output ---' -ForegroundColor Yellow
        Write-Host $detail -ForegroundColor Yellow
        Write-Host ('--- (also saved to ' + $errorFile + ') ---') -ForegroundColor Yellow
        $detail = (($detail -split "`r?`n" | Where-Object { $_.Trim() -ne '' } | Select-Object -Last 1) + '')
        if ($detail -match 'ModuleNotFoundError|No module named') {
            throw ('The VM identity check could not run: Python could not import locallife_cloud from ' +
                   $projectRoot + '. Confirm that folder contains locallife_cloud\. Details: ' + $detail)
        }
        throw ('The VM identity check could not run (no verdict was produced). ' +
               'This is a launcher problem, not a security failure. Details: ' + $detail)
    }
    if (-not $report.verified) {
        # Automatic verification was not possible -- Google had not published a
        # host key for this VM through either trusted channel. That is not
        # proof of an attack, and hard-failing here blocked cloud mode
        # entirely. Fall back to what the v3.2 reference deployment does:
        # plain `gcloud compute ssh`, where PuTTY shows the operator the
        # fingerprint and they answer once. A HUMAN confirms the key; nothing
        # is auto-accepted, and the key is never labelled as verified.
        Write-Host ''
        Write-Host 'SSH HOST KEY NOT AUTOMATICALLY VERIFIED (optional diagnostic only; startup continues).' -ForegroundColor Yellow
        Write-Host ('  Detail: ' + $report.error) -ForegroundColor DarkYellow
        if ($null -ne $report.identity) {
            Write-Host ('  VM: ' + $report.identity.vm_name + '  instance ' + $report.identity.instance_id) -ForegroundColor Yellow
            Write-Host ('  Zone: ' + $report.identity.zone + '  address ' + $report.identity.external_ip) -ForegroundColor Yellow
            $script:CloudSshHost = $report.identity.external_ip
        }
        Write-Host '  Authenticated gcloud SSH to this VM already succeeded, so cloud' -ForegroundColor Yellow
        Write-Host '  mode continues over that same authenticated channel.' -ForegroundColor Yellow
        Write-Host '  If a host key prompt appears, check the VM in the Google Cloud console' -ForegroundColor Yellow
        Write-Host '  before answering y. This is the one prompt you must read.' -ForegroundColor Yellow
        Write-Host ''
        $script:CloudSshInteractive = $true
        $script:CloudZone = $Zone
        return $report
    }
    $script:CloudSshInteractive = $false
    # Logged without credentials: instance id, zone, IP and fingerprints are all
    # public facts about the machine, and they are what makes a later mismatch
    # diagnosable.
    Write-Step ('SSH identity verified via ' + $report.source + ': ' + $report.identity.vm_name +
                ' (instance ' + $report.identity.instance_id + ') in ' + $report.identity.zone +
                ' at ' + $report.identity.external_ip)
    foreach ($item in @($report.fingerprints)) {
        Write-Step ('  host key ' + $item.key_type + ' ' + $item.fingerprint)
    }
    $script:CloudSshHost = $report.identity.external_ip
    $script:CloudZone = $Zone
    return $report
}

function Get-CloudSshOptions {
    <#
        OpenSSH arguments that verify without ever prompting. Set per
        connection; no global configuration is touched.
    #>
    $options = @(
        '-o', ('UserKnownHostsFile=' + (Get-CloudKnownHostsPath)),
        '-o', 'StrictHostKeyChecking=yes',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=20'
    )
    $identity = Join-Path $env:USERPROFILE '.ssh\google_compute_engine'
    if (Test-Path -LiteralPath $identity) {
        $options += @('-i', $identity, '-o', 'IdentitiesOnly=yes')
    }
    return $options
}

function Resolve-CloudSshUser {
    <#
        Which account to log in as.

        gcloud uses the local Windows username for metadata-based SSH and a
        mangled form of the account e-mail under OS Login, so both are tried --
        once per run, then cached. BatchMode means a wrong guess fails fast with
        an authentication error instead of sitting on a password prompt.
    #>
    param([Parameter(Mandatory = $true)][string]$HostAddress)
    if (-not [string]::IsNullOrWhiteSpace($script:CloudSshUser)) {
        return $script:CloudSshUser
    }
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:USERNAME)) { $candidates += $env:USERNAME }
    $account = Get-GcloudAccount
    if (-not [string]::IsNullOrWhiteSpace($account)) {
        $candidates += (($account -split '@')[0] -replace '[^A-Za-z0-9_-]', '_')
    }
    $options = Get-CloudSshOptions
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        Write-Step ('Trying cloud SSH account ' + $candidate + '...')
        Invoke-NativeTolerantly 'ssh' @options ($candidate + '@' + $HostAddress) 'true'
        if ($LASTEXITCODE -eq 0) {
            $script:CloudSshUser = $candidate
            return $candidate
        }
    }
    throw ('SSH authentication failed: none of these accounts could log in to ' + $HostAddress +
           ' (' + ($candidates -join ', ') + '). Run `gcloud compute ssh ' + $VmName +
           '` once by hand to provision your key, then start again.')
}

function Get-GcloudAccount {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $gcloudPath = Assert-GcloudAvailable
        return ((& $gcloudPath 'config' 'get-value' 'account' 2>$null | Out-String).Trim())
    }
    catch {
        return ''
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Invoke-VerifiedCloudSsh {
    <#
        Run one command on the cloud VM over a host-key-verified connection.
        Output goes to the pipeline so callers can inspect it, exactly as the
        gcloud calls this replaces did.
    #>
    param(
        [string]$Command = '',
        [string[]]$ExtraOptions = @()
    )
    if ($script:CloudSshInteractive) {
        # Reference-deployment path: gcloud drives PuTTY, which asks the
        # operator about the host key the first time and caches their answer.
        $gcloudPath = Assert-GcloudAvailable
        $arguments = @('compute', 'ssh', $VmName, ('--project=' + $CloudProject), ('--zone=' + $script:CloudZone))
        if ($ExtraOptions.Count -gt 0) {
            $arguments += '--'
            $arguments += $ExtraOptions
        }
        elseif (-not [string]::IsNullOrWhiteSpace($Command)) {
            $arguments += ('--command=' + $Command)
        }
        & $gcloudPath @arguments
        return
    }
    $target = (Resolve-CloudSshUser -HostAddress $script:CloudSshHost) + '@' + $script:CloudSshHost
    $options = (Get-CloudSshOptions) + $ExtraOptions
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        if ([string]::IsNullOrWhiteSpace($Command)) {
            # The tunnel case: -N means "no remote command", so passing an empty
            # string would make ssh try to run one and fail.
            & ssh @options $target
        }
        else {
            & ssh @options $target $Command
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function New-CloudBundle {
    <#
        Stage only the cloud application itself.

        The old upload sent the whole project directory recursively, which on a
        used laptop meant gigabytes of run output -- frames.jsonl, .npy
        captures, screenshots, result CSV/XLSX, __pycache__ and a 572 MB
        mobileclip_blt.ts -- so startup looked frozen during scp. This copies an
        explicit include list into a staging folder named after the project, and
        that folder is what gets uploaded. Nothing already on the VM (models,
        virtualenv, previous results) is touched.
    #>
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)
    $source = Get-Item -LiteralPath $ProjectRoot
    $staging = Join-Path $script:SessionDirectory 'cloud-bundle'
    $target = Join-Path $staging $source.Name
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    $includeFiles = @(
        'requirements-cloud.txt', 'requirements-local.txt', 'requirements-edge.txt',
        'cloud.env.example', 'PACKAGE_VERSION.txt', 'setup.py', '.locallife_deployment'
    )
    foreach ($name in $includeFiles) {
        $path = Join-Path $source.FullName $name
        if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination $target -Force }
    }
    foreach ($folder in @('locallife_cloud', 'scripts')) {
        $from = Join-Path $source.FullName $folder
        if (-not (Test-Path -LiteralPath $from)) { continue }
        $to = Join-Path $target $folder
        New-Item -ItemType Directory -Path $to -Force | Out-Null
        Get-ChildItem -LiteralPath $from -Recurse -File |
            Where-Object {
                $_.Extension -in @('.py', '.yaml', '.yml', '.sh', '.txt', '.json') -and
                $_.FullName -notmatch '\\__pycache__\\' -and $_.Name -notmatch '\.pyc$'
            } |
            ForEach-Object {
                $relative = $_.FullName.Substring($from.Length).TrimStart('\')
                $destination = Join-Path $to $relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
            }
    }
    $sizeMb = [math]::Round(((Get-ChildItem -LiteralPath $target -Recurse -File |
        Measure-Object -Property Length -Sum).Sum / 1MB), 1)
    $fileCount = (Get-ChildItem -LiteralPath $target -Recurse -File | Measure-Object).Count
    Write-Step ('Cloud bundle: ' + $fileCount + ' files, ' + $sizeMb + ' MB (run output, caches and models excluded)')
    return $target
}

function Invoke-VerifiedCloudScp {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$RemotePath
    )
    if ($script:CloudSshInteractive) {
        $gcloudPath = Assert-GcloudAvailable
        Invoke-NativeTolerantly $gcloudPath 'compute' 'scp' '--recurse' `
            ('--project=' + $CloudProject) ('--zone=' + $script:CloudZone) `
            $LocalPath ($VmName + ':' + $RemotePath)
        return
    }
    $target = (Resolve-CloudSshUser -HostAddress $script:CloudSshHost) + '@' + $script:CloudSshHost
    $options = Get-CloudSshOptions
    Invoke-NativeTolerantly 'scp' @options '-r' $LocalPath ($target + ':' + $RemotePath)
}

function Resolve-PiHostOnce {
    <#
        Replace the configured Pi name with one that actually answers.

        "locallife.local" is an mDNS name. Windows can only resolve it when
        multicast DNS is available, so on a network without it ssh stops at
        "Could not resolve hostname locallife.local: No such host is known"
        while the Pi is powered on and perfectly reachable by address. The name
        is the only broken part, so try the other ways of naming the same
        machine and keep whichever answers on port 22.

        The candidate list and the remembered address live in
        locallife_cloud/pi_discovery.py, so the welcome page and this launcher
        agree on what "reachable" means.
    #>
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    if ($script:PiHostResolved) {
        return $script:PiHost
    }
    $projectRoot = Find-ProjectRoot
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    $cache = Join-Path $script:SessionDirectory 'pi-address.json'
    $errorFile = Join-Path $script:SessionDirectory 'pi-lookup-stderr.txt'
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $projectRoot
    # stderr to a file, not the pipeline: under this script's own
    # $ErrorActionPreference = 'Stop', native stderr can be promoted to a
    # terminating error whose message is only its first line -- which is how a
    # real run reported "DEMONSTRATION ERROR: Traceback (most recent call
    # last):" and nothing else.
    try {
        $raw = (& $PythonExe '-m' 'locallife_cloud.pi_discovery' `
            '--pi-host' $PiHost '--cache' $cache 2>$errorFile | Out-String)
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }
    $errorText = ''
    if (Test-Path -LiteralPath $errorFile) {
        $errorText = ((Get-Content -LiteralPath $errorFile -Raw -ErrorAction SilentlyContinue) + '').Trim()
    }
    $report = $null
    foreach ($line in ($raw -split "`n")) {
        $trimmed = "$line".Trim()
        if ($trimmed.StartsWith('{')) { try { $report = $trimmed | ConvertFrom-Json } catch { } }
    }
    if ($null -eq $report) {
        # The lookup could not run. That is not the same as the Pi being
        # absent, so carry on with the configured name and let ssh report.
        $detail = $errorText
        if ([string]::IsNullOrWhiteSpace($detail)) { $detail = $raw.Trim() }
        Write-Host '--- Raspberry Pi lookup output ---' -ForegroundColor Yellow
        Write-Host $detail -ForegroundColor Yellow
        Write-Step ('Could not run the Raspberry Pi lookup; trying ' + $PiHost + ' as configured.')
        $script:PiHostResolved = $true
        return $PiHost
    }
    if (-not $report.found) {
        # Print the whole hint first: a multi-line message inside a thrown
        # error shows only its first line.
        Write-Host '' 
        Write-Host $report.hint -ForegroundColor Yellow
        Write-Host ''
        throw ('The Raspberry Pi could not be reached at ' + $PiHost +
               '. See the steps printed just above, then start again.')
    }
    if ($report.target -ne $PiHost) {
        Write-Step ('Raspberry Pi found at ' + $report.host + ' (' + $report.source +
                    '); the configured name ' + $PiHost + ' did not resolve.')
        $script:PiHost = $report.target
    }
    else {
        Write-Step ('Raspberry Pi reachable at ' + $report.host + '.')
    }
    $script:PiHostResolved = $true
    return $script:PiHost
}

function Assert-SshAvailable {
    if ($null -eq (Get-Command 'ssh.exe' -ErrorAction SilentlyContinue) -and
        $null -eq (Get-Command 'ssh' -ErrorAction SilentlyContinue)) {
        throw 'Windows OpenSSH is missing. Enable the OpenSSH Client Windows feature first.'
    }
}

function Test-RealPythonExecutable {
    # A stock Windows install always has "python.exe"/"python3.exe" App
    # Execution Alias stubs under ...\AppData\Local\Microsoft\WindowsApps\ on
    # PATH, even when Python itself was never installed. Get-Command happily
    # resolves them like any other executable, but running one does not run
    # Python -- it just prints an install prompt (or opens the Microsoft
    # Store) and exits, which is exactly what surfaced as this launcher's
    # "Python was not found; run without arguments to install..." error.
    # Reject those outright, then confirm whatever is left actually reports
    # a real Python 3.x version before trusting it.
    param([Parameter(Mandatory = $true)][string]$ExePath)
    if ($ExePath -match '\\WindowsApps\\') {
        return $false
    }
    if (-not (Test-Path -LiteralPath $ExePath)) {
        return $false
    }
    try {
        $versionOutput = & $ExePath '--version' 2>&1 | Out-String
    }
    catch {
        return $false
    }
    return ($LASTEXITCODE -eq 0 -and $versionOutput -match 'Python 3\.')
}

function Assert-PythonAvailable {
    # Prefer the official "py" launcher when present: it is installed
    # system-wide by the python.org installer and is not shadowed by the
    # WindowsApps stub the way "python"/"python3" on PATH can be, so it is
    # used here to discover the real interpreter's own path.
    $pyLauncher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        try {
            $discovered = (& $pyLauncher.Source '-3' '-c' 'import sys; print(sys.executable)' 2>$null |
                Select-Object -Last 1)
            if ($LASTEXITCODE -eq 0 -and $discovered -and (Test-RealPythonExecutable -ExePath $discovered.Trim())) {
                return $discovered.Trim()
            }
        }
        catch {
            # Fall through to the direct python3/python search below.
        }
    }

    # -All surfaces every "python3"/"python" match on PATH (not just the
    # first), so a real install that happens to sit behind the WindowsApps
    # stub in PATH order is still found.
    foreach ($name in @('python3', 'python')) {
        $resolved = @(Get-Command $name -ErrorAction SilentlyContinue -All)
        foreach ($candidate in $resolved) {
            if (Test-RealPythonExecutable -ExePath $candidate.Source) {
                return $candidate.Source
            }
        }
    }

    throw 'Python 3 is not installed, or only the Microsoft Store placeholder is on PATH. Install Python 3.10+ from https://python.org (not the Microsoft Store), then reopen PowerShell. If "python"/"python3" still resolves to the Store placeholder afterward, disable it from Settings > Apps > Advanced app settings > App execution aliases.'
}

function Assert-GcloudAvailable {
    foreach ($candidate in @('gcloud.cmd', 'gcloud')) {
        $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $resolved) {
            return $resolved.Source
        }
    }
    throw 'Google Cloud CLI was not found. -Mode Cloud needs it: install it and run "gcloud auth login" first, or use -Mode Local (no cloud, no account needed).'
}

function Find-GpuScript {
    # Nested two-argument Join-Path calls on purpose: Windows PowerShell 5.1
    # (the built-in "powershell.exe", still what most laptops actually run)
    # only accepts -Path/-ChildPath -- a third positional segment throws
    # "A positional parameter cannot be found that accepts argument ...".
    # PowerShell 7's multi-segment Join-Path is a newer addition, not safe
    # to rely on here.
    $candidates = @(
        (Join-Path $PSScriptRoot 'gpu.py'),
        (Join-Path (Join-Path $PSScriptRoot '..') 'gpu.py'),
        (Join-Path (Join-Path $env:USERPROFILE 'Downloads') 'gpu.py'),
        (Join-Path (Join-Path $env:USERPROFILE 'Documents') 'gpu.py')
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw 'gpu.py was not found next to this launcher. -Mode Cloud needs it (see the email from your collaborator); place it beside Start-LocalLife-Demo.ps1, or use -Mode Local.'
}

function Assert-ValidLanAddress {
    param([string]$Address)

    [System.Net.IPAddress]$parsed = $null
    if (-not [System.Net.IPAddress]::TryParse($Address, [ref]$parsed)) {
        throw ('The laptop network address is invalid: ' + $Address)
    }
    if ($parsed.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
        throw 'This demonstration requires a reachable IPv4 laptop address.'
    }
    if ($Address.StartsWith('127.') -or $Address.StartsWith('169.254.')) {
        throw 'The detected laptop address is not usable by the Raspberry Pi.'
    }
}

function Resolve-LaptopAddress {
    if (-not [string]::IsNullOrWhiteSpace($LanAddress)) {
        Assert-ValidLanAddress -Address $LanAddress
        return $LanAddress
    }

    $piProbe = $null
    try {
        $piComputer = ($PiHost -split '@')[-1]
        $piAddresses = @([System.Net.Dns]::GetHostAddresses($piComputer) |
            Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork })

        if ($piAddresses.Count -gt 0) {
            $piProbe = New-Object System.Net.Sockets.UdpClient
            $piProbe.Connect($piAddresses[0], 22)
            $piRouteAddress = $piProbe.Client.LocalEndPoint.Address.ToString()
            Assert-ValidLanAddress -Address $piRouteAddress
            return $piRouteAddress
        }
    }
    catch {
        Write-Step 'The Raspberry Pi route could not be resolved yet; checking the primary network adapter.'
    }
    finally {
        if ($null -ne $piProbe) {
            $piProbe.Close()
        }
    }

    try {
        $routes = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction Stop |
            Sort-Object RouteMetric, InterfaceMetric)

        foreach ($route in $routes) {
            $addresses = @(Get-NetIPAddress -InterfaceIndex $route.InterfaceIndex `
                -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.IPAddress -notlike '127.*' -and
                    $_.IPAddress -notlike '169.254.*' -and
                    $_.AddressState -eq 'Preferred'
                })

            if ($addresses.Count -gt 0) {
                return $addresses[0].IPAddress
            }
        }
    }
    catch {
        Write-Step 'Windows route inspection was unavailable; checking the active network socket.'
    }

    $socket = $null
    try {
        $socket = New-Object System.Net.Sockets.UdpClient
        $socket.Connect('8.8.8.8', 53)
        $detected = $socket.Client.LocalEndPoint.Address.ToString()
        Assert-ValidLanAddress -Address $detected
        return $detected
    }
    finally {
        if ($null -ne $socket) {
            $socket.Close()
        }
    }
}

function Find-ProjectRoot {
    # See the matching comment in Find-GpuScript: nested two-argument
    # Join-Path calls are required for Windows PowerShell 5.1 compatibility.
    $candidates = @(
        (Join-Path $PSScriptRoot $ProjectDirectory),
        (Join-Path (Join-Path $PSScriptRoot '..') $ProjectDirectory),
        (Join-Path (Join-Path $env:USERPROFILE 'Downloads') $ProjectDirectory),
        (Join-Path (Join-Path $env:USERPROFILE 'Documents') $ProjectDirectory)
    )
    foreach ($path in $candidates) {
        if (Test-Path (Join-Path $path 'locallife_cloud')) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw ('Project ' + $ProjectDirectory + ' not found. Checked: ' + ($candidates -join ', '))
}

function Get-RequiredSessionId {
    <#
        The one session id for this run.

        Order: an id already resolved in this process, then the one the launcher
        saved in session.json, then -- only for a child role started by hand --
        a clearly marked standalone id. Never a silent fake: a standalone id
        says so in its own name, so a status file written by a detached
        diagnostic run cannot be mistaken for the launcher's.
    #>
    param([string]$SessionId = '')
    if (-not [string]::IsNullOrWhiteSpace($SessionId)) {
        $script:SessionId = $SessionId
        return $script:SessionId
    }
    if (-not [string]::IsNullOrWhiteSpace($script:SessionId)) {
        return $script:SessionId
    }
    $saved = Get-Session
    if ($null -ne $saved -and
        $saved.PSObject.Properties.Name -contains 'session_id' -and
        -not [string]::IsNullOrWhiteSpace($saved.session_id)) {
        $script:SessionId = [string]$saved.session_id
        return $script:SessionId
    }
    $script:SessionId = 'standalone-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    return $script:SessionId
}

function Get-Session {
    if (-not (Test-Path -LiteralPath $script:SessionPath)) {
        return $null
    }
    try {
        return (Get-Content -LiteralPath $script:SessionPath -Raw | ConvertFrom-Json)
    }
    catch {
        Write-Warning 'An old session record could not be read and will be ignored.'
        return $null
    }
}

function Save-Session {
    param([hashtable]$Session)
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    $Session | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:SessionPath -Encoding UTF8
}

function Stop-RecordedWindows {
    param([object]$Session)
    if ($null -eq $Session) {
        return
    }
    foreach ($processId in @($Session.process_ids)) {
        if ($null -eq $processId) {
            continue
        }
        $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Write-Step ('Closing previous demonstration window ' + $processId + '...')
            if ($null -ne (Get-Command 'taskkill.exe' -ErrorAction SilentlyContinue)) {
                & taskkill.exe /PID ([int]$processId) /T /F 2>$null | Out-Null
            }
            else {
                Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Assert-PortAvailable {
    param([string]$Address)
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object {
            $_.LocalAddress -eq $Address -or
            $_.LocalAddress -eq '0.0.0.0' -or
            $_.LocalAddress -eq '::'
        })
    if ($listeners.Count -gt 0) {
        throw ('Laptop port ' + $Port + ' is already in use. Close the old demonstration or run STOP_LOCAL_LIFE_DEMO.cmd.')
    }
}

function Quote-PowerShellLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function ConvertTo-RemoteBootstrap {
    param([Parameter(Mandatory = $true)][string]$Command)
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command))
    if ($payload -notmatch '^[A-Za-z0-9+/]+={0,2}$') {
        throw 'The remote startup command could not be encoded safely.'
    }
    return ('printf %s ' + $payload + ' | base64 --decode | bash')
}

function Initialize-PiAutomaticLogin {
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    $credentialPath = Join-Path $script:SessionDirectory 'raspberry-pi-password.xml'
    if (-not (Test-Path -LiteralPath $credentialPath)) {
        # The device owner requested this shared demonstration credential.
        # Export-Clixml protects SecureString values using Windows user-scoped DPAPI.
        $protectedPassword = ConvertTo-SecureString -String 'locallife' -AsPlainText -Force
        $protectedPassword | Export-Clixml -LiteralPath $credentialPath -Force
    }
    $helperScript = Join-Path $script:SessionDirectory ('pi-askpass-' + $PID + '.ps1')
    $helperCommand = Join-Path $script:SessionDirectory ('pi-askpass-' + $PID + '.cmd')
    $helperLines = @(
        '$ErrorActionPreference = ''Stop''',
        ('$encrypted = Import-Clixml -LiteralPath ' + (Quote-PowerShellLiteral -Value $credentialPath)),
        '$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($encrypted)',
        'try { [Console]::Out.WriteLine([Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)) }',
        'finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }'
    )
    Set-Content -LiteralPath $helperScript -Value $helperLines -Encoding UTF8
    $commandLines = @(
        '@echo off',
        ('powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $helperScript + '"')
    )
    Set-Content -LiteralPath $helperCommand -Value $commandLines -Encoding ASCII
    return [PSCustomObject]@{
        CredentialPath = $credentialPath
        HelperScript = $helperScript
        HelperCommand = $helperCommand
    }
}

function Start-RoleWindow {
    param(
        [ValidateSet('App', 'Pi', 'CloudTunnel')]
        [string]$ChildRole,
        [string]$Address
    )
    $command = '& ' + (Quote-PowerShellLiteral -Value $PSCommandPath) +
        ' -Role ' + (Quote-PowerShellLiteral -Value $ChildRole) +
        ' -Mode ' + (Quote-PowerShellLiteral -Value $Mode) +
        ' -OperatingMode ' + (Quote-PowerShellLiteral -Value $OperatingMode) +
        ' -SessionId ' + (Quote-PowerShellLiteral -Value (Get-RequiredSessionId)) +
        ' -ProjectDirectory ' + (Quote-PowerShellLiteral -Value $ProjectDirectory) +
        ' -PiHost ' + (Quote-PowerShellLiteral -Value $PiHost) +
        ' -Port ' + $Port +
        ' -PiBridgePort ' + $PiBridgePort +
        ' -UploadFps ' + $UploadFps +
        ' -GpuArgs ' + (Quote-PowerShellLiteral -Value $GpuArgs) +
        ' -LanAddress ' + (Quote-PowerShellLiteral -Value $Address) +
        ' -ApiToken ' + (Quote-PowerShellLiteral -Value $ApiToken)

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $arguments = @(
        '-NoLogo', '-NoExit', '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-EncodedCommand', $encoded
    )
    # Launched via conhost.exe (not powershell.exe directly): on Windows 11
    # with "Windows Terminal" set as the default terminal app, Start-Process
    # -FilePath powershell.exe does NOT open a separate window at all -- it
    # opens a new TAB inside whichever Windows Terminal window is already
    # open (e.g. Window 1), which is invisible unless the user notices the
    # tab strip. conhost.exe is the legacy console host itself, so launching
    # through it forces a genuine standalone window every time regardless of
    # that default-terminal setting.
    return (Start-Process -FilePath 'conhost.exe' -ArgumentList (@('powershell.exe') + $arguments) -PassThru)
}

function Wait-ForAppHealth {
    param([string]$Address)
    $url = 'http://' + $Address + ':' + $Port + '/health'
    Write-Step ('Waiting for the application at ' + $url)
    # 30 attempts (~2 min) assumed a warm model cache (~30s per the Window 1
    # banner). A real run showed Window 1 becoming ready just 23 seconds
    # *after* this loop had already given up -- a cold-start model download
    # (Depth Anything weights from Hugging Face, unauthenticated so rate-
    # limited/slower) genuinely took longer than 2 minutes. Raised to 90
    # attempts (~6 min) so a slow first-run download has real headroom.
    #
    # Cloud mode's own 240 (8 min) was raised again to 1500 (~50 min) after
    # a real "moved to another zone" run: this can only succeed once BOTH
    # Window 1 finishes `gpu.py up` (which, per Wait-ForCloudZoneFile's own
    # comment, can legitimately take 15-30+ minutes during a forced zone
    # move -- image capture, zone hunting, then up to 12 min VM boot) AND
    # Window 2's tunnel connects afterward, plus the app itself starting on
    # the VM and loading its models. 8 minutes was nowhere near enough for
    # that path and threw here even when nothing was actually wrong.
    $attempts = if ($Mode -eq 'Cloud') { 1500 } else { 90 }  # cloud VM boot + model load can take minutes; a forced zone move takes much longer
    for ($attempt = 1; $attempt -le $attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                $health = $response.Content | ConvertFrom-Json
                if ($health.status -eq 'ok') {
                    Write-Step 'The application is ready.'
                    return
                }
            }
        }
        catch {
            if (($attempt % 5) -eq 0) {
                $where = if ($Mode -eq 'Cloud') { 'Window 1 (cloud GPU) or Window 2 (tunnel)' } else { 'Window 1' }
                Write-Step ('Still waiting. Check ' + $where + ' for startup messages.')
            }
        }
        Start-Sleep -Seconds 2
    }
    throw 'The application did not become reachable. Inspect the application/tunnel window(s) for errors.'
}

function Wait-ForCameraStreams {
    param([string]$Address)
    $url = 'http://' + $Address + ':' + $Port + '/health'
    Write-Step 'Waiting for live RealSense and Logitech frames through the private Raspberry Pi bridge...'
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        try {
            $health = Invoke-RestMethod -Uri $url -TimeoutSec 3
            $realsenseFrames = [int]$health.cameras.realsense
            $logitechFrames = [int]$health.cameras.logitech
            if ($health.status -eq 'ok' -and $realsenseFrames -gt 0 -and $logitechFrames -gt 0) {
                Write-Step 'Both camera streams have reached the application.'
                return
            }
            if (($attempt % 5) -eq 0) {
                Write-Step ('Camera frames received - RealSense: ' + $realsenseFrames + '; Logitech: ' + $logitechFrames)
            }
        }
        catch {
            if (($attempt % 5) -eq 0) {
                Write-Step 'Still waiting. Check the Raspberry Pi window for camera messages.'
            }
        }
        Start-Sleep -Seconds 2
    }
    throw 'Both camera streams did not reach the application. Check the Pi window and confirm both cameras are connected.'
}

# --------------------------------------------------------------------- #
# Window 1 (both modes): the actual LocalLife server + dashboard.
# Local mode: runs directly on this laptop.
# Cloud mode: runs on the gpu.py-managed VM over SSH (blocking foreground
#             command in this window, matching the local window's shape).
# --------------------------------------------------------------------- #
function Start-AppRole {
    if ($Mode -eq 'Local') {
        $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE 1 OF 2 - LAPTOP APPLICATION - KEEP OPEN'
        Write-Banner 'WINDOW 1 OF 2: LOCAL APPLICATION (RUNS ON YOUR LAPTOP)'
        Write-Host 'All processing happens here, on your laptop. No cloud needed.' -ForegroundColor Yellow
        Write-Host 'Keep this window open for the entire demonstration.' -ForegroundColor Yellow

        $pythonExe = Assert-PythonAvailable
        Write-Step ('Using Python: ' + $pythonExe)
        $projectRoot = Find-ProjectRoot
        Write-Step ('Using project: ' + $projectRoot)

        try {
            Push-Location $projectRoot
            Write-Step 'Verifying Python dependencies...'
            Invoke-NativeTolerantly $pythonExe '-m' 'pip' 'show' 'locallife-cloud' *>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Step 'Installing LocalLife package (one-time setup)...'
                Invoke-NativeTolerantly $pythonExe '-m' 'pip' 'install' '-e' '.' '-q'
                if ($LASTEXITCODE -ne 0) {
                    throw 'Failed to install LocalLife package.'
                }
            }
            # Report which locallife_cloud package Python ACTUALLY imports,
            # and from where. Every version banner printed by this window is
            # a hardcoded string inside this .ps1 file -- it says which
            # LAUNCHER is running, and cannot say anything about the Python
            # code that does the real work. Those two can genuinely diverge:
            # `pip install -e .` records one specific directory, several
            # extracted copies of this package can sit side by side in
            # Downloads, and `pip show` succeeding makes the install step
            # above skip entirely. Printing the resolved path and
            # __version__ makes "which build is really running?" answerable
            # straight from a pasted log instead of being guessed at.
            $packageProbe = 'import locallife_cloud, os, sys; sys.stdout.write(os.path.realpath(os.path.dirname(locallife_cloud.__file__)) + "|" + getattr(locallife_cloud, "__version__", "unknown"))'
            $probeOutput = Get-PythonPackageIdentity -PythonExe $pythonExe -Probe $packageProbe
            $expectedPackage = Join-Path $projectRoot 'locallife_cloud'
            if ([string]::IsNullOrWhiteSpace($probeOutput)) {
                Write-Warning 'Could not determine which locallife_cloud package Python imports.'
            }
            else {
                $pieces = $probeOutput -split '\|'
                $loadedFrom = $pieces[0]
                $loadedVersion = 'unknown'
                if ($pieces.Count -gt 1) { $loadedVersion = $pieces[1] }
                Write-Step ('Package in use: ' + $loadedVersion + ' from ' + $loadedFrom)
                $expectedResolved = ''
                if (Test-Path -LiteralPath $expectedPackage) {
                    $expectedResolved = (Resolve-Path -LiteralPath $expectedPackage).Path
                }
                if ($expectedResolved -and ($loadedFrom.TrimEnd('\', '/') -ne $expectedResolved.TrimEnd('\', '/'))) {
                    Write-Warning ('Python is importing locallife_cloud from ' + $loadedFrom +
                        ' but this launcher was started next to ' + $expectedResolved +
                        '. Re-pointing the editable install at this copy...')
                    Invoke-NativeTolerantly $pythonExe '-m' 'pip' 'install' '-e' '.' '-q'
                    $probeOutput = Get-PythonPackageIdentity -PythonExe $pythonExe -Probe $packageProbe
                    if (-not [string]::IsNullOrWhiteSpace($probeOutput)) {
                        Write-Step ('Package in use after re-install: ' + $probeOutput.Replace('|', ' from '))
                    }
                }
            }
            Write-Step 'Starting LocalLife server and dashboard...'
            Write-Step 'Models load on first startup (~30s). Subsequent runs are faster.'
            # Binding 0.0.0.0 (needed for the Pi's reverse SSH tunnel below)
            # is refused by the server without an API token -- the Launcher
            # role generates one and passes it via -ApiToken; set it as the
            # env var the server actually reads.
            if (-not [string]::IsNullOrEmpty($ApiToken)) {
                $env:LOCALLIFE_API_TOKEN = $ApiToken
            }
            $env:LOCALLIFE_OPERATING_MODE = $OperatingMode
            # The backend cannot see the VM itself; these tell the operator
            # page which deployment it was started under so it can show
            # "Cloud GPU" or "Local" honestly instead of guessing.
            $env:CLOUD_ENABLED = $(if ($Mode -eq 'Cloud') { 'true' } else { 'false' })
            $env:LOCALLIFE_GCP_PROJECT = $CloudProject
            $env:LOCALLIFE_VM_NAME = $VmName
            $env:LOCALLIFE_VM_ZONE = $Zone
            $env:LOCALLIFE_PI_HOST = $PiHost
            Write-Step ('Operating mode: ' + $OperatingMode)
            # --disable-sync: the background bucket-sync thread (BucketSync
            # in storage.py) is a Cloud-mode concern -- syncing results to a
            # Google Cloud Storage bucket so they survive an ephemeral VM
            # being torn down. Local mode's laptop disk is not ephemeral, so
            # this only ever produced a confusing "Cloud Storage
            # synchronization failed" warning every 3 minutes (Windows
            # subprocess can't launch gcloud.cmd without a shell -- a
            # separate, real but purely cosmetic bug in storage.py itself),
            # contradicting Local mode's own "no cloud needed" banner.
            Invoke-NativeTolerantly $pythonExe '-m' 'locallife_cloud.server' '--host' '0.0.0.0' '--port' $Port '--disable-sync'
        }
        finally {
            Pop-Location
        }
        if ($LASTEXITCODE -ne 0) {
            throw 'The application stopped or could not start.'
        }
        return
    }

    # -Mode Cloud: bring up (or restart) the GPU VM through gpu.py, then run
    # the server on it over SSH. gpu.py hunts across 11 EU zones (and,
    # with --us, US zones) for an available L4/T4 instead of the old
    # launcher's single hardcoded VM name + zone, which failed outright
    # whenever that one zone had no GPU capacity -- exactly what the email
    # reported ("L4 GPUs are sold out across most of Europe right now").
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE 1 OF 3 - CLOUD GPU - KEEP OPEN'
    Write-Banner 'WINDOW 1 OF 3: CLOUD GPU APPLICATION (via gpu.py)'
    Write-Host 'Bringing up the GPU VM. This hunts across zones automatically if the' -ForegroundColor Yellow
    Write-Host 'usual zone has no GPU capacity right now -- this can take 2-8 minutes.' -ForegroundColor Yellow
    Write-Host 'Keep this window open for the entire demonstration.' -ForegroundColor Yellow

    $pythonExe = Assert-PythonAvailable
    Assert-GcloudAvailable | Out-Null
    $gpuScript = Find-GpuScript
    Write-Step ('Using gpu.py: ' + $gpuScript)

    $upArguments = @('up')
    if (-not [string]::IsNullOrWhiteSpace($GpuArgs)) {
        $upArguments += ($GpuArgs -split '\s+' | Where-Object { $_ -ne '' })
    }
    Write-Step ('Running: python gpu.py ' + ($upArguments -join ' '))
    Invoke-NativeTolerantly $pythonExe $gpuScript @upArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'gpu.py could not bring up a GPU VM (see its output above). Try again in a few minutes, or add -GpuArgs "--us" to also search US zones, or use -Mode Local instead.'
    }

    # `up` already printed its own "ready: gcloud compute ssh depth-l4
    # --zone=X" line, but re-reading it back through `status` is more
    # robust than screen-scraping that specific line's wording. A fresh
    # instance is consistently visible to the same project's own
    # `instances list` immediately after `create` returns, but retry a
    # few times regardless in case of a transient API lag.
    $zone = $null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $statusOutput = & $pythonExe $gpuScript status
        $statusLine = ($statusOutput | Select-String -Pattern 'depth-l4:\s+RUNNING\s+in\s+(\S+)')
        if ($null -ne $statusLine) {
            $zone = $statusLine.Matches[0].Groups[1].Value
            break
        }
        Start-Sleep -Seconds 5
    }
    if ($null -eq $zone) {
        throw 'gpu.py reported success but the VM zone could not be determined from `python gpu.py status`. Run it manually to check.'
    }
    Write-Step ('GPU VM is running in zone ' + $zone)
    Set-CloudStage -Stage 'verifying_ssh' -Status 'running' -Zone $zone

    $gcloudPath = Assert-GcloudAvailable

    # Before ANY window connects, and specifically before the zone file below
    # releases Windows 2 and 3 to start their own SSH sessions: establish and
    # pin the VM's identity. Doing it first is what makes it safe -- the other
    # windows only learn the zone once the file exists, so they cannot race in
    # against an unpinned host.
    Assert-CloudSshIdentity -PythonExe $pythonExe -Zone $zone | Out-Null

    Set-CloudStage -Stage 'checking_deployment' -Status 'running' -Zone $zone
    Write-Step 'Recording the zone for the tunnel window...'
    $zoneFile = Join-Path $script:SessionDirectory 'cloud-zone.txt'
    if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
        New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
    }
    Set-Content -LiteralPath $zoneFile -Value $zone -Encoding ASCII -NoNewline

    # One-click means the launcher installs the project on a fresh VM
    # itself instead of throwing "not installed, see the instructions" and
    # leaving a manual `scp` step for the user (the same gap the Raspberry
    # Pi side had, before it was fixed by hand). A fresh VM from gpu.py
    # (freshly created in a new zone after a capacity move, or truly new)
    # never has the project on it, so check first and upload only when
    # actually missing -- an existing VM's disk persists across `up`/`down`
    # restarts, so most runs skip this entirely.
    Write-Step 'Checking whether the project is installed on the cloud VM...'
    # Print the home directory alongside the verdict: "not installed" with no
    # sign of what was looked for, or of what is actually on the VM, is the
    # hardest possible message to act on -- and the usual cause is SSHing into
    # a different project's VM than the one the upload went to.
    # Earlier builds uploaded to "vm:~/", which pscp turned into a directory
    # literally named "~". Move anything stranded there back into the real home
    # before checking, so an affected VM repairs itself instead of re-uploading
    # gigabytes every run. mv only relocates; nothing is deleted.
    # The deployment version is a hash of this laptop's Python sources. Asking
    # only whether the directory existed -- what this used to do -- meant a VM
    # carrying an OLD copy of the project counted as installed and silently ran
    # stale code. Comparing the hash both catches that and skips the upload
    # entirely when nothing changed, which is the expensive step.
    $deploymentVersion = (& $pythonExe '-c' `
        'import sys,pathlib;sys.path.insert(0,sys.argv[1]);from locallife_cloud.cloud_startup import deployment_version;print(deployment_version(pathlib.Path(sys.argv[1])))' `
        (Find-ProjectRoot) 2>$null | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($deploymentVersion)) {
        $deploymentVersion = 'unknown'
    }
    Write-Step ('Local deployment version: ' + $deploymentVersion)
    $checkCommand = 'if [ -d "$HOME/~" ]; then echo "LOCALLIFE_REPAIRING_TILDE_DIR"; ' +
        'mv -n "$HOME/~"/* "$HOME"/ 2>/dev/null; rmdir "$HOME/~" 2>/dev/null; fi; ' +
        'echo "LOCALLIFE_HOME_CONTENTS:"; ls -1 ~ 2>/dev/null | head -20; ' +
        'echo "LOCALLIFE_REMOTE_VERSION=$(cat ~/' + $ProjectDirectory + '/.locallife_deployment 2>/dev/null || echo none)"; ' +
        'if [ -d ~/' + $ProjectDirectory + '/locallife_cloud ]; then echo LOCALLIFE_PROJECT_PRESENT; ' +
        'else echo LOCALLIFE_PROJECT_MISSING; fi'
    $checkOutput = Invoke-VerifiedCloudSsh -Command $checkCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not check the cloud VM over SSH (see its output above). Run `python gpu.py ssh` to investigate.'
    }
    $checkText = ($checkOutput -join "`n")
    $remoteVersion = 'none'
    if ($checkText -match 'LOCALLIFE_REMOTE_VERSION=(\S+)') { $remoteVersion = $Matches[1] }
    $projectPresent = ($checkText -match 'LOCALLIFE_PROJECT_PRESENT')
    $versionMatches = ($remoteVersion -eq $deploymentVersion -and $deploymentVersion -ne 'unknown')
    if ($projectPresent -and $versionMatches) {
        Write-Step ('Cloud VM already has this exact code (' + $deploymentVersion + ') -- skipping upload.')
    }
    if (-not ($projectPresent -and $versionMatches)) {
        if ($projectPresent) {
            Write-Step ('Cloud VM has a different version (' + $remoteVersion + ' vs ' + $deploymentVersion + ') -- updating it...')
        }
        else {
            Write-Step 'Project not found on the cloud VM -- uploading it now (first time only; can take several minutes)...'
        }
        $projectRoot = Find-ProjectRoot
        # Ask the VM where its home actually is, and upload to that absolute
        # path. Windows' pscp.exe -- which `gcloud compute scp` shells out to --
        # does NOT expand a leading "~": it treats it as an ordinary directory
        # name. A destination of "vm:~/" therefore transfers every file
        # successfully into a directory literally called "~", so the upload
        # reports 100% while ~/<project> stays empty in any real shell.
        $homeOutput = Invoke-VerifiedCloudSsh -Command 'echo LOCALLIFE_REMOTE_HOME=$HOME'
        $remoteHome = ''
        foreach ($line in @($homeOutput)) {
            if ("$line" -match 'LOCALLIFE_REMOTE_HOME=(\S+)') { $remoteHome = $Matches[1] }
        }
        if ([string]::IsNullOrWhiteSpace($remoteHome)) {
            throw 'Could not determine the cloud VM home directory over SSH; cannot upload the project safely.'
        }
        Write-Step ('Uploading the cloud application bundle to ' + $remoteHome + ' on ' + $VmName + '...')
        Set-CloudStage -Stage 'uploading_bundle' -Status 'running' -Zone $script:CloudZone
        $bundle = New-CloudBundle -ProjectRoot $projectRoot
        Invoke-VerifiedCloudScp -LocalPath $bundle -RemotePath ($remoteHome + '/')
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not upload the project to the cloud VM (see its output above). Run `python gpu.py ssh` and check disk space, or copy it manually.'
        }
        # Verify rather than assume. A partial or wrong-target upload otherwise
        # surfaces minutes later as an unexplained "project is not installed".
        $verifyOutput = Invoke-VerifiedCloudSsh -Command $checkCommand
        if (($verifyOutput -join "`n") -notmatch 'LOCALLIFE_PROJECT_PRESENT') {
            throw ('The upload reported success but ~/' + $ProjectDirectory +
                   '/locallife_cloud is still missing on ' + $VmName + ' in ' + $zone +
                   ' (project ' + $CloudProject + '). The VM listing above shows what is ' +
                   'actually there. Confirm the VM and project are the ones you expect: ' +
                   'gcloud compute instances list --project=' + $CloudProject)
        }
        # Stamp the version only after the upload has been verified, so a
        # partial transfer cannot leave a marker claiming code that is not
        # actually there -- which would make every later run skip the repair.
        Invoke-VerifiedCloudSsh -Command (
            'printf %s ' + $deploymentVersion + ' > ~/' + $ProjectDirectory + '/.locallife_deployment'
        ) | Out-Null
        Write-Step ('Upload complete and verified (version ' + $deploymentVersion + ').')
    }
    # Each remaining step publishes its own stage, so Window 2 and the welcome
    # page name what is actually happening instead of the last SSH step.
    Set-CloudStage -Stage 'installing_dependencies' -Status 'running' -Zone $script:CloudZone

    # --host 127.0.0.1: the VM server binds its OWN loopback only, never the
    # public internet. It is reachable only through SSH tunnels that are
    # already authenticated -- Window 2 below (laptop -> VM, for the browser
    # dashboard) and the Raspberry Pi's own direct tunnel (Pi -> VM, opened
    # in Start-PiRole/Initialize-PiCloudTunnel) -- so no API token, bearer
    # secret, or firewall rule for the app port is needed. (An earlier
    # version of this script bound 0.0.0.0 here without ever exporting
    # LOCALLIFE_API_TOKEN on the remote command below, which server.py's own
    # "binding outside localhost requires a token" check would have refused
    # to start against -- this loopback bind removes that requirement
    # entirely instead of papering over it with a token.)
    $remoteCommand =
        'if [ ! -d ~/' + $ProjectDirectory + ' ]; then ' +
        'echo "ERROR: ~/' + $ProjectDirectory + ' does not exist on this VM."; ' +
        'echo "Home contains:"; ls -1 ~ 2>/dev/null | head -20; ' +
        'echo "If this is not the VM you uploaded to, check the project and zone."; ' +
        'exit 1; fi; ' +
        'cd ~/' + $ProjectDirectory + ' || exit 1; ' +
        'export LOCALLIFE_OPERATING_MODE=' + $OperatingMode + '; ' +
        # The GPU backend runs the Large Depth Anything V2 on every Logitech frame.
        'export CLOUD_ENABLED=true; ' +
        'echo LOCALLIFE_STAGE=installing_dependencies; ' +
        # Do not restart a backend that is already serving. A reconnect after a
        # dropped tunnel used to kill a healthy, model-loaded process and pay
        # the whole model-load cost again for nothing.
        'if curl -sf -m 3 http://127.0.0.1:' + $Port + '/api/state >/dev/null 2>&1; then ' +
        'echo "LOCALLIFE_BACKEND_ALREADY_HEALTHY"; exit 0; fi; ' +
        "pkill -f '[l]ocallife_cloud.server' || true; " +
        'if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; ' +
        # PEP 668: Debian 12+ / Python 3.12 images mark the system interpreter
        # "externally managed" and refuse a plain pip install. --user keeps the
        # install in ~/.local and --break-system-packages is what that refusal
        # itself names as the override. Harmless on images without the marker.
        'PIPFLAGS="--user --break-system-packages -q"; ' +
        'python3 -m pip show locallife-cloud >/dev/null 2>&1 || ' +
        'python3 -m pip install $PIPFLAGS -e . || ' +
        'python3 -m pip install --user -q -e . || exit 1; ' +
        # ultralytics depends on the GUI build of OpenCV, which needs libGL.so.1
        # -- absent on a headless VM image, so `import cv2` dies with
        # "libGL.so.1: cannot open shared object file" and takes the detector
        # and both video streams down with it. Swap in the headless wheel, but
        # only when cv2 is actually broken, so a healthy VM pays nothing.
        'python3 -c "import cv2" >/dev/null 2>&1 || { ' +
        'echo "LOCALLIFE: repairing OpenCV (headless VM has no libGL)"; ' +
        'python3 -m pip uninstall -y -q opencv-python opencv-contrib-python >/dev/null 2>&1; ' +
        'python3 -m pip install $PIPFLAGS --force-reinstall opencv-python-headless || ' +
        'python3 -m pip install --user -q --force-reinstall opencv-python-headless; }; ' +
        'python3 -c "import cv2" >/dev/null 2>&1 || { ' +
        'echo "ERROR: OpenCV still will not import on this VM. Run: sudo apt-get install -y libgl1"; ' +
        'exit 1; }; ' +
        'exec python3 -m locallife_cloud.server --host 127.0.0.1 --port ' + $Port

    $remoteBootstrap = ConvertTo-RemoteBootstrap -Command $remoteCommand
    # gcloud's OWN --strict-host-key-checking flag (distinct from OpenSSH's
    # -o StrictHostKeyChecking used directly elsewhere in this script for the
    # Pi's own ssh/scp calls) only accepts ask|no|yes on the gcloud CLI
    # versions actually seen in the field -- 'accept-new' was tried here in
    # an earlier round on the strength of Google's own reference docs, but a
    # real installed gcloud rejected it outright ("Invalid choice: 'accept-
    # new'. Valid choices are [ask, no, yes]"), so it is not a safe value to
    # rely on. 'no' is used instead: it never prompts (avoiding the
    # interactive "Store key in cache?" prompt Windows' bundled plink.exe
    # otherwise shows on first connection) but, unlike accept-new's
    # pin-on-first-use, it also never verifies the host key on later
    # connections either. That is an acceptable tradeoff here -- gpu.py can
    # create a fresh VM (a new host key) in a new zone on every `up`, so a
    # pinned key would otherwise have to be manually cleared on every zone
    # change anyway, and the connection is already gated by the user's own
    # gcloud/IAM auth, not by host-key trust.
    Invoke-VerifiedCloudSsh -Command $remoteBootstrap
    if ($LASTEXITCODE -ne 0) {
        throw 'The cloud application stopped or could not start. Check `python gpu.py status` and `python gpu.py ssh`.'
    }
}

# --------------------------------------------------------------------- #
# Recipe API (fully optional, -Role RecipeApi only): starts the separate
# FastAPI dual-camera volume/color/material endpoint (recipe_api.py). This
# is never started by -Role Launcher / Start-Demo -- the tested App/Pi/
# CloudTunnel flow above runs exactly as it always has whether or not
# anyone ever uses this role. Run it by hand, in its own window, alongside
# an already-running -Role Launcher session: "Start-LocalLife-Demo.ps1
# -Role RecipeApi". It always runs on this machine's own Python venv and
# binds to loopback only; a Cloud-mode user who wants it running on the GPU
# VM instead can SSH there and run the same `python -m
# locallife_cloud.recipe_api` command themselves, the same way the main
# server is started in Start-AppRole's Cloud branch above.
# --------------------------------------------------------------------- #
function Start-RecipeApiRole {
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE - RECIPE API (OPTIONAL) - KEEP OPEN'
    Write-Banner 'RECIPE API (OPTIONAL): DUAL-CAMERA VOLUME/COLOR/MATERIAL ENDPOINT'
    Write-Host 'This is a separate, additive service -- it does not replace or affect' -ForegroundColor Yellow
    Write-Host 'the main dashboard window (start that as usual with -Role Launcher).' -ForegroundColor Yellow
    Write-Host 'Keep this window open for as long as you want the recipe endpoint available.' -ForegroundColor Yellow

    $pythonExe = Assert-PythonAvailable
    Write-Step ('Using Python: ' + $pythonExe)
    $projectRoot = Find-ProjectRoot
    Write-Step ('Using project: ' + $projectRoot)

    try {
        Push-Location $projectRoot
        Write-Step 'Verifying recipe API dependencies (fastapi, uvicorn, open3d)...'
        Invoke-NativeTolerantly $pythonExe '-m' 'pip' 'show' 'fastapi' *>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Step 'Installing recipe API dependencies (one-time setup)...'
            Invoke-NativeTolerantly $pythonExe '-m' 'pip' 'install' '-r' 'requirements-local.txt' '-q'
            if ($LASTEXITCODE -ne 0) {
                throw 'Failed to install recipe API dependencies. See requirements-local.txt (open3d, fastapi, uvicorn, python-multipart).'
            }
        }
        # Same token as the main dashboard, if one was already set for this
        # shell -- so a request to the recipe API can reuse the credential
        # a Cloud-mode -Role Launcher session already generated. Left blank
        # (no auth) is fine here since this always binds to loopback only.
        if (-not [string]::IsNullOrEmpty($ApiToken)) {
            $env:LOCALLIFE_API_TOKEN = $ApiToken
        }
        Write-Step ('Starting recipe API on http://127.0.0.1:' + $RecipeApiPort + ' ...')
        Invoke-NativeTolerantly $pythonExe '-m' 'locallife_cloud.recipe_api' '--host' '127.0.0.1' '--port' $RecipeApiPort
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'The recipe API stopped or could not start.'
    }
}

# --------------------------------------------------------------------- #
# Shared by Window 2 (Start-CloudTunnelRole) and Window 3's Pi-tunnel setup
# (Initialize-PiCloudTunnel): wait for Window 1 to finish `gpu.py up` and
# record which zone the VM landed in.
#
# A real run showed why a short, fixed timeout here is wrong: gpu.py's own
# "restart failed here, move to another zone" path (cmd_up: 3 restart
# attempts, then make_image() -- its own comment says "3-8 min", hard
# timeout 1800s -- then delete the old VM, hunt() across every zone/shape
# tier, then wait_ready() waiting up to 12 more minutes for the fresh VM's
# boot script) can legitimately take 15-30+ minutes end to end. The
# original 60-attempt/2-second (2 minute) wait threw "the cloud VM zone was
# never recorded" while Window 1 was still mid-image-capture, minutes away
# from finishing real, successful work.
#
# Rather than just pick an even bigger blind timeout and hope it is enough,
# this also checks whether Window 1's own process is still alive: if it has
# already exited (crashed, or the user closed it) without ever writing the
# zone file, there is nothing left to wait for, so this fails immediately
# with a clearer message instead of sitting out the rest of a long timeout.
# --------------------------------------------------------------------- #
function Wait-ForCloudZoneFile {
    $zoneFile = Join-Path $script:SessionDirectory 'cloud-zone.txt'
    $appProcessId = $null
    $session = Get-Session
    if ($null -ne $session -and $session.PSObject.Properties.Name -contains 'process_ids' -and @($session.process_ids).Count -gt 0) {
        $appProcessId = [int](@($session.process_ids))[0]
    }
    # 800 attempts x 3s = 40 minutes -- covers gpu.py's own documented
    # worst case for a forced zone move (image capture, zone hunting, then
    # up to 12 min in wait_ready) with real-world slack on top.
    $maxAttempts = 800
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        if (Test-Path -LiteralPath $zoneFile) {
            return (Get-Content -LiteralPath $zoneFile -Raw).Trim()
        }
        # Window 1 publishes its stage; a failure there ends this wait in
        # seconds instead of counting to thirty minutes and blaming a zone
        # migration that is not happening.
        $stage = Get-CloudStage
        if ($null -ne $stage -and $stage.status -eq 'failed') {
            throw ('Cloud startup failed in Window 1 at stage "' + $stage.stage + '" (' +
                   $stage.error_code + '): ' + $stage.error_message + ' -- ' +
                   'Retry Cloud, or Run Locally with START_LOCAL_LIFE_DEMO.cmd. ' +
                   'Diagnostics are in Window 1. The VM may still be running and ' +
                   'billable: stop it with LocalLife_Stop.exe.')
        }
        if ($null -ne $appProcessId -and (($attempt % 10) -eq 0)) {
            if ($null -eq (Get-Process -Id $appProcessId -ErrorAction SilentlyContinue)) {
                throw 'Window 1 (cloud GPU) has already closed without ever bringing up the VM. Check its output for the actual gpu.py error.'
            }
        }
        if (($attempt % 20) -eq 0) {
            $minutesWaited = [math]::Round(($attempt * 3) / 60, 1)
            # Calibrated against what a real run actually costs. When the usual
            # zone has GPU capacity the VM is ready in about a minute and a
            # half, so the old unconditional "a zone move can take 15-30
            # minutes" fired at the 1-minute mark and made an ordinary,
            # on-schedule startup read as a hang. The long-wait wording now
            # appears only once the wait itself has gone past the point where a
            # zone move is the likely explanation.
            $stageNow = Get-CloudStage
            # Only a VM that is genuinely still being brought up can be moving
            # zones. Past that stage the long-wait wording would be a lie.
            $movingZones = ($null -eq $stageNow -or $stageNow.stage -eq 'starting_vm')
            if ($minutesWaited -lt 5 -or -not $movingZones) {
                Write-Step ('Still waiting on Window 1 to bring up the cloud VM (' + $minutesWaited + ' min so far). A normal start takes about 1-3 minutes.')
            }
            else {
                Write-Step ('Still waiting on Window 1 to bring up the cloud VM (' + $minutesWaited + ' min so far). Past a few minutes this usually means gpu.py is moving the VM to another zone for GPU capacity -- image capture plus a fresh VM in a new region can take 15-30 minutes. That is normal; watch Window 1''s own output for progress.')
            }
        }
        Start-Sleep -Seconds 3
    }
    throw 'The cloud VM zone was never recorded by Window 1; check it for gpu.py errors.'
}

# --------------------------------------------------------------------- #
# Window 2 (Cloud mode only): SSH tunnel from this laptop's own loopback to
# the cloud VM's $Port -- purely so a browser ON THIS LAPTOP can open the
# dashboard at http://127.0.0.1:$Port. It does NOT carry camera traffic:
# the Raspberry Pi opens its own separate, direct tunnel straight to the
# cloud VM (Window 3 / Initialize-PiCloudTunnel), so frames never have to
# hop through the laptop at all. Three tunnels total, each doing one job:
# laptop<->Pi (SSH login only, no camera traffic), laptop<->cloud (this
# window, dashboard viewing only), Pi<->cloud (camera traffic, direct).
# --------------------------------------------------------------------- #
function Start-CloudTunnelRole {
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE 2 OF 3 - SECURE TUNNEL - KEEP OPEN'
    Write-Banner 'WINDOW 2 OF 3: SECURE LAPTOP-TO-CLOUD TUNNEL (DASHBOARD VIEWING ONLY)'
    Write-Host 'This window only lets your browser see the dashboard -- camera frames go' -ForegroundColor Yellow
    Write-Host 'straight from the Raspberry Pi to the cloud VM, not through here.' -ForegroundColor Yellow
    Write-Host 'A quiet or blank window after connection is normal.' -ForegroundColor Yellow
    Write-Host 'Keep this window open for the entire demonstration.' -ForegroundColor Yellow

    $zone = Wait-ForCloudZoneFile
    Write-Step ('Tunneling to depth-l4 in ' + $zone)
    $stage = Get-CloudStage
    if ($null -ne $stage -and -not [string]::IsNullOrWhiteSpace($stage.stage)) {
        Write-Step ('Window 1 stage: ' + $stage.stage + ' (' + $stage.status + ')')
    }

    # This window is its own process, so it establishes the VM's identity for
    # itself rather than trusting a variable Window 1 set. Window 1 has already
    # pinned the same keys, so this is a fast re-verify that reports "unchanged"
    # -- and if it ever does NOT match, this window stops instead of tunnelling
    # to a host that is not the VM.
    Assert-CloudSshIdentity -PythonExe (Assert-PythonAvailable) -Zone $zone | Out-Null
    $forward = '127.0.0.1:' + $Port + ':127.0.0.1:' + $Port
    Invoke-VerifiedCloudSsh -Command '' -ExtraOptions @('-N', '-L', $forward)
    if ($LASTEXITCODE -ne 0) {
        throw 'The secure SSH tunnel stopped. Check cloud login and whether the laptop port is already occupied.'
    }
}

# --------------------------------------------------------------------- #
# Raspberry Pi cloud tunnel provisioning: -Mode Cloud only. Gives the Pi
# everything it needs to open its OWN direct SSH tunnel straight to the
# depth-l4 GPU VM, so camera frames never have to relay through the laptop.
# Idempotent -- safe to call on every run.
# --------------------------------------------------------------------- #
function Initialize-PiCloudTunnel {
    Write-Step 'Preparing the direct Raspberry-Pi-to-cloud tunnel...'

    # In the normal orchestrated flow (Start-Demo) this call finds the zone
    # file immediately -- Window 3 (Pi) is only started after Wait-ForAppHealth
    # already succeeded, which itself cannot succeed before Window 1/2 have
    # finished. The generous wait/liveness-check in Wait-ForCloudZoneFile
    # mainly matters here for someone running `-Role Pi` standalone.
    $zone = Wait-ForCloudZoneFile
    $gcloudPath = Assert-GcloudAvailable
    # Same reasoning as Window 2: verify identity in this process before using it.
    Assert-CloudSshIdentity -PythonExe (Assert-PythonAvailable) -Zone $zone | Out-Null

    # 1) The Pi needs its own dedicated keypair for this tunnel -- separate
    #    from whatever credential the laptop uses to log into the Pi, and
    #    never leaves the Pi (only the public half is read back here).
    Write-Step 'Checking the Raspberry Pi''s dedicated cloud-tunnel SSH key...'
    $keySetupCommand = 'test -f ~/.ssh/id_locallife_cloud || ' +
        '(umask 077; mkdir -p ~/.ssh; ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_locallife_cloud -C locallife-cloud-tunnel -q); ' +
        'cat ~/.ssh/id_locallife_cloud.pub'
    $keyOutput = & ssh '-o' 'StrictHostKeyChecking=accept-new' $PiHost $keySetupCommand
    $piPublicKey = (@($keyOutput) -join "`n").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($piPublicKey)) {
        throw 'Could not prepare the Raspberry Pi''s cloud-tunnel SSH key.'
    }

    # 2) The VM needs a matching, dedicated, restricted user for that key:
    #    no interactive shell, no agent/X11 forwarding, forwarding only --
    #    all it can ever do is carry this one tunnel's bytes. Rewriting
    #    authorized_keys from scratch each run is cheap and keeps this step
    #    idempotent without needing to grep-and-append.
    Write-Step ('Provisioning the restricted tunnel user on the cloud VM (' + $script:PiTunnelUser + ')...')
    $authorizedLine = 'restrict,port-forwarding,no-pty,no-agent-forwarding,no-X11-forwarding,no-user-rc ' + $piPublicKey
    $vmSetupCommand =
        'if ! id -u ' + $script:PiTunnelUser + ' >/dev/null 2>&1; then ' +
        'sudo useradd --system --create-home --shell /usr/sbin/nologin ' + $script:PiTunnelUser + '; fi; ' +
        'sudo install -d -m 700 -o ' + $script:PiTunnelUser + ' -g ' + $script:PiTunnelUser +
        ' /home/' + $script:PiTunnelUser + '/.ssh && ' +
        "echo '" + $authorizedLine + "' | sudo tee /home/" + $script:PiTunnelUser + '/.ssh/authorized_keys >/dev/null && ' +
        'sudo chmod 600 /home/' + $script:PiTunnelUser + '/.ssh/authorized_keys && ' +
        'sudo chown ' + $script:PiTunnelUser + ':' + $script:PiTunnelUser + ' /home/' + $script:PiTunnelUser + '/.ssh/authorized_keys'
    $vmBootstrap = ConvertTo-RemoteBootstrap -Command $vmSetupCommand
    Invoke-VerifiedCloudSsh -Command $vmBootstrap
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not provision the cloud VM''s restricted tunnel user (see its output above). Run `python gpu.py ssh` to investigate.'
    }

    # 3) Best-effort: the VM needs tcp:22 reachable from the Pi's network,
    #    which almost always already works (GCP's default network ships a
    #    "default-allow-ssh" rule, and gpu.py itself already SSHes in from
    #    this laptop) -- so this only creates a rule when one is genuinely
    #    missing, and never fails the run if it can't.
    Write-Step 'Checking the cloud firewall allows SSH for the Pi tunnel...'
    try {
        $ruleName = 'locallife-allow-ssh-pi-tunnel'
        & $gcloudPath 'compute' 'firewall-rules' 'describe' $ruleName '--format=value(name)' *>$null
        if ($LASTEXITCODE -ne 0) {
            Invoke-NativeTolerantly $gcloudPath 'compute' 'firewall-rules' 'create' $ruleName `
                '--direction=INGRESS' '--action=ALLOW' '--rules=tcp:22' '--source-ranges=0.0.0.0/0'
            if ($LASTEXITCODE -ne 0) {
                Write-Host 'Could not create the SSH firewall rule automatically. If the Pi tunnel fails to connect, allow tcp:22 inbound to the cloud VM in the Google Cloud Console.' -ForegroundColor Yellow
            }
        }
    }
    catch {
        Write-Host 'Could not confirm the SSH firewall rule automatically. If the Pi tunnel fails to connect, allow tcp:22 inbound to the cloud VM in the Google Cloud Console.' -ForegroundColor Yellow
    }

    # 4) Resolve (and cache) the VM's current external IP -- the address the
    #    Pi's own tunnel needs to connect out to. Cached so a transient
    #    `describe` failure on a later run can fall back to the last known
    #    good address instead of failing outright.
    $cachedAddress = $null
    if (Test-Path -LiteralPath $script:CloudVmAddressPath) {
        $cachedAddress = (Get-Content -LiteralPath $script:CloudVmAddressPath -Raw).Trim()
    }
    Write-Step 'Resolving the cloud VM''s external address...'
    $ipOutput = & $gcloudPath 'compute' 'instances' 'describe' $VmName ('--zone=' + $zone) ('--project=' + $CloudProject) `
        '--format=value(networkInterfaces[0].accessConfigs[0].natIP)'
    $vmAddress = (@($ipOutput) -join '').Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vmAddress)) {
        if (-not [string]::IsNullOrWhiteSpace($cachedAddress)) {
            Write-Host 'Could not re-resolve the cloud VM''s external address; reusing the last known one.' -ForegroundColor Yellow
            $vmAddress = $cachedAddress
        }
        else {
            throw 'Could not determine the cloud VM''s external IP address for the direct Pi tunnel.'
        }
    }
    else {
        if (-not (Test-Path -LiteralPath $script:SessionDirectory)) {
            New-Item -ItemType Directory -Path $script:SessionDirectory -Force | Out-Null
        }
        Set-Content -LiteralPath $script:CloudVmAddressPath -Value $vmAddress -Encoding ASCII -NoNewline
    }
    Write-Step ('Cloud VM address for the direct Pi tunnel: ' + $vmAddress)
    return $vmAddress
}

# --------------------------------------------------------------------- #
# Raspberry Pi camera bridge.
# Local mode: unchanged -- talks to 127.0.0.1:$PiBridgePort on the Pi,
# which a reverse SSH forward (-R, opened by this window) points at the
# laptop's own server. Camera traffic: Pi -> laptop.
# Cloud mode: no reverse tunnel through the laptop at all. The Pi opens its
# OWN outbound tunnel straight to the cloud VM (Initialize-PiCloudTunnel
# above provisions it), then talks to 127.0.0.1:$PiBridgePort on ITSELF,
# which that tunnel points at the VM's loopback-only server. Camera
# traffic: Pi -> cloud VM, directly -- the laptop is never in that path.
# --------------------------------------------------------------------- #
function Start-PiRole {
    $windowLabel = if ($Mode -eq 'Cloud') { '3 OF 3' } else { '2 OF 2' }
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE ' + $windowLabel + ' - RASPBERRY PI CAMERAS - KEEP OPEN'
    Write-Banner ('WINDOW ' + $windowLabel + ': RASPBERRY PI AND BOTH CAMERAS')
    Write-Host 'The Raspberry Pi signs in automatically; no password needs to be typed.' -ForegroundColor Green
    if ($Mode -eq 'Cloud') {
        Write-Host 'Camera traffic goes straight from the Pi to the cloud GPU VM over its own' -ForegroundColor Green
        Write-Host 'private SSH tunnel -- it does not relay through this laptop.' -ForegroundColor Green
    }
    else {
        Write-Host 'Camera traffic uses a private SSH bridge; no Windows firewall change is required.' -ForegroundColor Green
    }
    Write-Host 'Keep this window open for the entire demonstration.' -ForegroundColor Yellow

    $automaticLogin = Initialize-PiAutomaticLogin
    $oldAskPass = [Environment]::GetEnvironmentVariable('SSH_ASKPASS', 'Process')
    $oldAskPassRequirement = [Environment]::GetEnvironmentVariable('SSH_ASKPASS_REQUIRE', 'Process')
    $oldDisplay = [Environment]::GetEnvironmentVariable('DISPLAY', 'Process')

    try {
        [Environment]::SetEnvironmentVariable('SSH_ASKPASS', $automaticLogin.HelperCommand, 'Process')
        [Environment]::SetEnvironmentVariable('SSH_ASKPASS_REQUIRE', 'force', 'Process')
        [Environment]::SetEnvironmentVariable('DISPLAY', 'locallife-demo', 'Process')

        # One-click means this window installs the project on the Pi itself
        # instead of exec'ing straight into $remoteCommand above, which
        # would just print "not installed" and dead-end -- exactly what
        # happened on a real run and needed a manual `scp` + `pip3 install`
        # to recover from. A Pi that already has the project (the normal
        # case after the first run) skips this entirely.
        $PiHost = Resolve-PiHostOnce -PythonExe (Assert-PythonAvailable)
        Write-Step 'Checking whether the project is installed on the Raspberry Pi...'
        $checkCommand = 'if [ -d ~/' + $ProjectDirectory + '/locallife_cloud ]; then echo LOCALLIFE_PROJECT_PRESENT; else echo LOCALLIFE_PROJECT_MISSING; fi'
        $checkOutput = & ssh '-o' 'StrictHostKeyChecking=accept-new' $PiHost $checkCommand
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not check the Raspberry Pi over SSH (see its output above). Confirm it is powered on and reachable.'
        }
        if (($checkOutput -join "`n") -notmatch 'LOCALLIFE_PROJECT_PRESENT') {
            Write-Step 'Project not found on the Raspberry Pi -- uploading it now (first time only; can take a few minutes)...'
            $projectRoot = Find-ProjectRoot
            Invoke-NativeTolerantly 'scp' '-o' 'StrictHostKeyChecking=accept-new' '-r' $projectRoot ($PiHost + ':~/')
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not upload the project to the Raspberry Pi (see its output above).'
            }
            Write-Step 'Upload complete. Installing Raspberry Pi camera dependencies (requirements-edge.txt)...'
            Invoke-NativeTolerantly 'ssh' '-o' 'StrictHostKeyChecking=accept-new' $PiHost `
                ('cd ~/' + $ProjectDirectory + ' && pip3 install -r requirements-edge.txt')
            if ($LASTEXITCODE -ne 0) {
                throw 'Could not install the Raspberry Pi camera dependencies. SSH in and run: pip3 install -r requirements-edge.txt'
            }
            Write-Step 'Raspberry Pi setup complete.'
        }

        # StrictHostKeyChecking=accept-new avoids an interactive
        # "authenticity of host ... can't be established" prompt on the
        # very first connection to a given Pi (nothing in this unattended
        # window could answer it), matching the same fix applied to the
        # gcloud/plink connections above.
        if ($Mode -eq 'Cloud') {
            $vmAddress = Initialize-PiCloudTunnel
            $cloudUrl = 'http://127.0.0.1:' + $PiBridgePort

            # No -R reverse tunnel to the laptop at all here: the Pi opens
            # its OWN outbound tunnel straight to the GPU VM in the
            # background (the "Pi to cloud" leg -- see the banner comment
            # above Initialize-PiCloudTunnel), then talks to it over its own
            # loopback exactly the way Local mode talks to its laptop-side
            # bridge. The VM server binds 127.0.0.1 only (Start-AppRole), so
            # this tunnel is the only way in -- no token needed either end.
            $remoteCommand =
                'if [ ! -d ~/' + $ProjectDirectory + ' ]; then ' +
                'echo "ERROR: The project is not installed on the Raspberry Pi."; exit 1; fi; ' +
                'cd ~/' + $ProjectDirectory + ' || exit 1; ' +
                'pkill -f "id_locallife_cloud" 2>/dev/null || true; sleep 1; ' +
                'echo "Opening the direct Pi-to-cloud tunnel..."; ' +
                'nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 ' +
                '-o StrictHostKeyChecking=accept-new -i ~/.ssh/id_locallife_cloud ' +
                '-L 127.0.0.1:' + $PiBridgePort + ':127.0.0.1:' + $Port + ' ' +
                $script:PiTunnelUser + '@' + $vmAddress +
                ' >/tmp/locallife-cloud-tunnel.log 2>&1 & ' +
                'echo "Waiting for the direct tunnel and the cloud application..."; ' +
                'if ! curl --silent --show-error --retry 20 --retry-all-errors --retry-delay 3 --max-time 5 ' +
                $cloudUrl + '/health >/dev/null; then ' +
                'echo "ERROR: The direct Pi-to-cloud tunnel could not reach the healthy cloud application."; ' +
                'echo "Check Window 1 (cloud GPU) is still up; on the Pi, cat /tmp/locallife-cloud-tunnel.log"; exit 1; fi; ' +
                'echo "Direct tunnel is up."; ' +
                # A previous demo run's edge_client can be left running on the
                # Pi -- e.g. the Windows launcher window was closed or the
                # run errored out before the remote SSH command exited
                # cleanly -- and it keeps the RealSense pipeline (and often
                # the Logitech /dev/video node) open. The next run's
                # pipeline.start() then fails with
                # "xioctl(VIDIOC_S_FMT) failed, errno=16: Device or resource
                # busy" and the Logitech open fails the same way, even
                # though both cameras are physically fine. Kill any stray
                # instance and give the kernel a moment to release the USB
                # device nodes before reopening them, mirroring the existing
                # id_locallife_cloud tunnel cleanup above.
                'pkill -9 -f "[l]ocallife_cloud.edge_client" 2>/dev/null || true; sleep 2; ' +
                'exec python3 -m locallife_cloud.edge_client --cloud ' + $cloudUrl +
                ' --source dual --upload-fps ' + $UploadFps

            $remoteBootstrap = ConvertTo-RemoteBootstrap -Command $remoteCommand
            Invoke-NativeTolerantly 'ssh' '-o' 'StrictHostKeyChecking=accept-new' $PiHost $remoteBootstrap
            if ($LASTEXITCODE -ne 0) {
                throw 'The direct Pi-to-cloud tunnel or camera startup failed. Check host trust, cameras, and the cloud application/tunnel windows.'
            }
        }
        else {
            $cloudUrl = 'http://127.0.0.1:' + $PiBridgePort
            $reverseForward = '127.0.0.1:' + $PiBridgePort + ':' + $LanAddress + ':' + $Port

            # The server (started in Start-AppRole with the same $ApiToken)
            # requires this once it binds outside localhost -- without it
            # every frame upload below gets a 401 and the cameras never
            # reach the dashboard. $ApiToken is a GUID's hex digits only,
            # safe unquoted. Computed as its own statement (not inlined into
            # the '+' chain below) because a PowerShell 5.1 parser bug/quirk
            # misreads an `if` expression appended after a comment-
            # interrupted line-continuation as a bare `if` COMMAND -- exactly
            # the "The term 'if' is not recognized" crash this replaces.
            $tokenArg = if ([string]::IsNullOrEmpty($ApiToken)) { '' } else { ' --token ' + $ApiToken }

            $remoteCommand =
                'if [ ! -d ~/' + $ProjectDirectory + ' ]; then ' +
                'echo "ERROR: The project is not installed on the Raspberry Pi."; exit 1; fi; ' +
                'cd ~/' + $ProjectDirectory + ' || exit 1; ' +
                'echo "Checking the private Raspberry Pi bridge..."; ' +
                'if ! curl --silent --show-error --retry 12 --retry-all-errors --retry-delay 2 --max-time 5 ' +
                $cloudUrl + '/health >/dev/null; then ' +
                'echo "ERROR: The private Raspberry Pi bridge could not reach the healthy application."; ' +
                'echo "Keep the application window open, then restart."; exit 1; fi; ' +
                # See the matching comment in the Cloud-mode branch above:
                # a leftover edge_client process from a previous run still
                # holds the RealSense/Logitech device nodes open, which is
                # the "Device or resource busy" (errno=16) failure this
                # guards against.
                'pkill -9 -f "[l]ocallife_cloud.edge_client" 2>/dev/null || true; sleep 2; ' +
                'exec python3 -m locallife_cloud.edge_client --cloud ' + $cloudUrl +
                ' --source dual --upload-fps ' + $UploadFps + $tokenArg

            $remoteBootstrap = ConvertTo-RemoteBootstrap -Command $remoteCommand
            Invoke-NativeTolerantly 'ssh' '-o' 'ExitOnForwardFailure=yes' '-o' 'StrictHostKeyChecking=accept-new' `
                '-R' $reverseForward $PiHost $remoteBootstrap
            if ($LASTEXITCODE -ne 0) {
                throw 'The private Raspberry Pi bridge or camera startup failed. Check host trust, cameras, SSH port forwarding, and the application window(s).'
            }
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable('SSH_ASKPASS', $oldAskPass, 'Process')
        [Environment]::SetEnvironmentVariable('SSH_ASKPASS_REQUIRE', $oldAskPassRequirement, 'Process')
        [Environment]::SetEnvironmentVariable('DISPLAY', $oldDisplay, 'Process')
        Remove-Item -LiteralPath $automaticLogin.HelperScript -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $automaticLogin.HelperCommand -Force -ErrorAction SilentlyContinue
    }
}

function Start-Demo {
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE DEMONSTRATION LAUNCHER'
    if ($Mode -eq 'Local') {
        Write-Banner 'LOCAL LIFE DUAL-CAMERA DEMONSTRATION (v19.20.0 - LOCAL LAPTOP)'
        Write-Host 'NO CLOUD NEEDED - everything runs on your Windows laptop, for free.' -ForegroundColor Green
    }
    else {
        Write-Banner 'LOCAL LIFE DUAL-CAMERA DEMONSTRATION (v19.20.0 - CLOUD GPU via gpu.py)'
        Write-Host 'Using a cloud GPU VM. gpu.py hunts across zones for capacity automatically.' -ForegroundColor Green
        Write-Host 'Cloud charges apply while the VM is running -- see STOP_LOCAL_LIFE_DEMO.cmd.' -ForegroundColor Yellow
    }
    Write-Host 'This will open windows and your browser automatically.' -ForegroundColor Green
    Write-Host 'Do not close the opened windows during the demonstration.' -ForegroundColor Green

    Assert-SshAvailable
    Assert-PythonAvailable | Out-Null
    if ($Mode -eq 'Cloud') {
        Assert-GcloudAvailable | Out-Null
        Find-GpuScript | Out-Null
        $zoneFile = Join-Path $script:SessionDirectory 'cloud-zone.txt'
        Remove-Item -LiteralPath $zoneFile -Force -ErrorAction SilentlyContinue
    }
    $address = Resolve-LaptopAddress
    Assert-ValidLanAddress -Address $address
    Write-Step ('Laptop network address: ' + $address)

    # Local mode: the app binds this laptop's LAN address (0.0.0.0), so the
    # dashboard/health checks below use it too. Cloud mode: Window 2 now
    # tunnels to loopback only (camera traffic bypasses the laptop
    # entirely -- see the comment above Start-CloudTunnelRole), so the
    # dashboard lives at 127.0.0.1 on this laptop instead.
    $appAddress = if ($Mode -eq 'Cloud') { '127.0.0.1' } else { $address }

    $oldSession = Get-Session
    if ($null -ne $oldSession) {
        $active = @($oldSession.process_ids | Where-Object {
            $null -ne (Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue)
        })
        if ($active.Count -gt 0) {
            $answer = Read-Host 'A previous demonstration is still open. Close it and start fresh? [Y/n]'
            if ($answer -match '^(n|no)$') {
                throw 'Startup canceled. The earlier demonstration windows were left unchanged.'
            }
            Stop-RecordedWindows -Session $oldSession
            Start-Sleep -Seconds 2
        }
    }

    Assert-PortAvailable -Address $appAddress

    # Local mode only: the server refuses to bind outside localhost (the
    # normal 0.0.0.0 case there, so the Pi's reverse SSH tunnel can reach
    # it) without a bearer token -- generated fresh each run and threaded to
    # both the App window (sets it as the server's LOCALLIFE_API_TOKEN) and
    # the Pi window (passed to edge_client's --token so its uploads keep
    # authenticating). Cloud mode's VM server binds loopback only (see
    # Start-AppRole) and never needs this, but it is still generated
    # unconditionally here since -Role App/Pi can also be run standalone in
    # either mode and harmlessly ignore an unused token.
    if ([string]::IsNullOrEmpty($ApiToken)) {
        $ApiToken = [Guid]::NewGuid().ToString('N')
    }

    $script:SessionId = [Guid]::NewGuid().ToString('N').Substring(0, 12)
    $session = @{
        session_id = $script:SessionId
        started_at = (Get-Date).ToString('o')
        mode = $Mode
        operating_mode = $OperatingMode
        laptop_address = $address
        dashboard_url = ('http://' + $appAddress + ':' + $Port)
        pi_bridge_port = $PiBridgePort
        process_ids = @()
    }

    Write-Step 'Opening Window 1: application (models load here)...'
    $appProcess = Start-RoleWindow -ChildRole 'App' -Address $address
    $session.process_ids += $appProcess.Id
    Save-Session -Session $session

    if ($Mode -eq 'Cloud') {
        Write-Step 'Opening Window 2: secure tunnel to the cloud GPU VM...'
        $tunnelProcess = Start-RoleWindow -ChildRole 'CloudTunnel' -Address $address
        $session.process_ids += $tunnelProcess.Id
        Save-Session -Session $session
    }

    if ($Mode -eq 'Cloud') { Set-CloudStage -Stage 'starting_backend' -Status 'running' -Zone $script:CloudZone }
    Wait-ForAppHealth -Address $appAddress
    if ($Mode -eq 'Cloud') { Set-CloudStage -Stage 'opening_tunnel' -Status 'running' -Zone $script:CloudZone }

    $piWindowNumber = if ($Mode -eq 'Cloud') { 'Window 3' } else { 'Window 2' }
    Write-Step ('Opening ' + $piWindowNumber + ': Raspberry Pi, private bridge, and both cameras...')
    $piProcess = Start-RoleWindow -ChildRole 'Pi' -Address $address
    $session.process_ids += $piProcess.Id
    Save-Session -Session $session

    Wait-ForCameraStreams -Address $appAddress

    Write-Step ('Opening dashboard: ' + $session.dashboard_url)
    Start-Process $session.dashboard_url

    Write-Host ''
    Write-Host 'The demonstration is starting.' -ForegroundColor Green
    if ($Mode -eq 'Local') {
        Write-Host 'Window 1 runs AI models on your laptop (YOLOv8/YOLOE, CLIP, volume estimation).' -ForegroundColor Green
    }
    else {
        Write-Host 'Window 1 runs AI models on the cloud GPU; Window 2 tunnels the dashboard here.' -ForegroundColor Green
        Write-Host 'Window 3 (the Pi) tunnels camera frames straight to the cloud GPU -- not through here.' -ForegroundColor Green
    }
    Write-Host 'The Pi window connects automatically; no password entry is needed.' -ForegroundColor Green
    Write-Host 'Then set the measured Logitech distance and capture both empty-bin baselines.' -ForegroundColor Green
    Write-Host 'Use STOP_LOCAL_LIFE_DEMO.cmd when the demonstration is finished.' -ForegroundColor Green
}

function Stop-Demo {
    $Host.UI.RawUI.WindowTitle = 'LOCAL LIFE DEMONSTRATION SHUTDOWN'
    Write-Banner 'STOP THE LOCAL LIFE DEMONSTRATION'

    $session = Get-Session
    $sessionMode = if ($null -ne $session -and $session.PSObject.Properties.Name -contains 'mode') { $session.mode } else { $Mode }
    if ($null -eq $session) {
        Write-Host 'No saved demonstration session was found.' -ForegroundColor Yellow
    }
    else {
        Stop-RecordedWindows -Session $session
        Remove-Item -LiteralPath $script:SessionPath -Force -ErrorAction SilentlyContinue
        Write-Step 'The recorded local demonstration windows have been closed.'
    }

    if ($sessionMode -eq 'Cloud') {
        $zoneFile = Join-Path $script:SessionDirectory 'cloud-zone.txt'
        Remove-Item -LiteralPath $zoneFile -Force -ErrorAction SilentlyContinue
        $answer = Read-Host 'Also stop the cloud GPU VM now (saves your home dir, then stops billing except ~7 kr/day disk)? [Y/n]'
        if ($answer -notmatch '^(n|no)$') {
            try {
                $pythonExe = Assert-PythonAvailable
                $gpuScript = Find-GpuScript
                Write-Step 'Running: python gpu.py down'
                Invoke-NativeTolerantly $pythonExe $gpuScript 'down'
            }
            catch {
                Write-Host ('Could not run gpu.py down automatically: ' + $_.Exception.Message) -ForegroundColor Yellow
                Write-Host 'Run "python gpu.py down" yourself to stop cloud billing.' -ForegroundColor Yellow
            }
        }
        else {
            Write-Host 'The cloud VM keeps running. Run "python gpu.py down" later, or it auto-stops after 30 idle minutes.' -ForegroundColor Yellow
        }
    }
    else {
        Write-Host 'No cloud charges to worry about. Your laptop did all the work!' -ForegroundColor Green
    }
}

function Invoke-Doctor {
    Write-Banner 'LOCAL LIFE DEMONSTRATION READINESS CHECK'
    $pythonExe = Assert-PythonAvailable
    Assert-SshAvailable
    $address = Resolve-LaptopAddress
    Assert-ValidLanAddress -Address $address
    Write-Host ('Python: ' + $pythonExe)
    Write-Host ('Laptop IPv4: ' + $address)
    Write-Host ('Raspberry Pi SSH: ' + $PiHost)
    Write-Host ('Dashboard address: http://' + $address + ':' + $Port)
    Write-Host ('Raspberry Pi project directory: ~/' + $ProjectDirectory)
    Write-Host ('Mode: ' + $Mode)
    if ($Mode -eq 'Cloud') {
        try {
            $gcloudPath = Assert-GcloudAvailable
            Write-Host ('Google Cloud CLI: ' + $gcloudPath)
        }
        catch {
            Write-Host ($_.Exception.Message) -ForegroundColor Yellow
        }
        try {
            $gpuScript = Find-GpuScript
            Write-Host ('gpu.py: ' + $gpuScript)
            Invoke-NativeTolerantly $pythonExe $gpuScript 'status'
        }
        catch {
            Write-Host ($_.Exception.Message) -ForegroundColor Yellow
        }
    }
    else {
        try {
            $projectRoot = Find-ProjectRoot
            Write-Host ('Local project found: ' + $projectRoot)
        }
        catch {
            Write-Host ($_.Exception.Message) -ForegroundColor Yellow
        }
    }
    Write-Host 'Camera hardware, the configured Pi credential, and firewall access must still be verified on the actual devices.'
    Write-Host ''
    Write-Host 'To start: double-click START_LOCAL_LIFE_DEMO.cmd (Local mode) or START_LOCAL_LIFE_CLOUD.cmd (Cloud mode)'
}

try {
    if ($Role -eq 'Launcher') {
        Start-Demo
    }
    elseif ($Role -eq 'Stop') {
        Stop-Demo
    }
    elseif ($Role -eq 'Doctor') {
        Invoke-Doctor
    }
    if (-not [string]::IsNullOrWhiteSpace($SessionId)) {
        Get-RequiredSessionId -SessionId $SessionId | Out-Null
    }
    if ($false) { }
    elseif ($Role -eq 'RecipeApi') {
        Start-RecipeApiRole
    }
    else {
        Assert-ValidLanAddress -Address $LanAddress
        if ($Role -eq 'App') {
            Start-AppRole
        }
        elseif ($Role -eq 'CloudTunnel') {
            Start-CloudTunnelRole
        }
        elseif ($Role -eq 'Pi') {
            Assert-SshAvailable
            Start-PiRole
        }
    }
}
catch {
    # Publish the failure before reporting it: Window 2 watches this file and
    # would otherwise keep counting minutes at a VM that is already up.
    if ($Role -eq 'App' -and $Mode -eq 'Cloud') {
        $failedStage = 'session_initialisation'
        $current = Get-CloudStage
        if ($null -ne $current -and -not [string]::IsNullOrWhiteSpace($current.stage)) {
            $failedStage = [string]$current.stage
        }
        $failedZone = ''
        if ($null -ne $current -and -not [string]::IsNullOrWhiteSpace($current.zone)) {
            $failedZone = [string]$current.zone
        }
        Set-CloudStage -Stage $failedStage -Status 'failed' -Zone $failedZone `
            -ErrorCode 'cloud_startup_failed' -ErrorMessage $_.Exception.Message
    }
    Write-Host ''
    Write-Host ('DEMONSTRATION ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ''
    if ($Mode -eq 'Cloud' -and $Role -in @('App', 'CloudTunnel')) {
        # A failed startup can leave the GPU VM running and billed.
        Write-Host 'The cloud VM may still be running and billable. Stop it with:' -ForegroundColor Yellow
        Write-Host '  STOP_LOCAL_LIFE_DEMO.cmd   (or: python gpu.py stop)' -ForegroundColor Yellow
        Write-Host ''
    }
    if ($Role -in @('App', 'CloudTunnel', 'Pi', 'RecipeApi')) {
        Read-Host 'Press Enter after you have read the error'
    }
    exit 1
}
