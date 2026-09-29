# Tests for the Window 2 cloud tunnel supervisor in Start-LocalLife-Demo.ps1.
# Loads the real function definitions (the launcher itself is not executed) and
# replaces only the external boundaries: gcloud/ssh, sockets, HTTP and sleeps.
# Run: pwsh -NoProfile -File tests/ps/test_cloud_tunnel.ps1   (exit 0 = pass)
param([string]$Launcher = (Join-Path $PSScriptRoot '..\..\..\Start-LocalLife-Demo.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $Launcher), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "Launcher does not parse: $($errors[0])" }
$wanted = 'Join-ProcessArguments', 'ConvertFrom-RemoteAppProbe', 'Get-TunnelLogTail', 'Test-TunnelProcessAlive',
          'Resolve-LocalPortConflict', 'Invoke-CloudTunnelSupervisor'
foreach ($fn in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($wanted -contains $fn.Name) { . ([scriptblock]::Create($fn.Extent.Text)) }
}

$Port = 8000; $VmName = 'depth-l4'; $CloudProject = 'p'
$script:SessionDirectory = [IO.Path]::GetTempPath()
$script:Failures = 0

function Assert-True([bool]$Condition, [string]$Name) {
    if ($Condition) { [Console]::WriteLine("PASS $Name") } else { [Console]::WriteLine("FAIL $Name"); $script:Failures++ }
}

function Reset-Fakes {
    $script:Log = New-Object System.Collections.ArrayList
    $script:Zone = 'europe-west4-b'
    $script:Probes = New-Object System.Collections.Queue
    $script:LocalOk = $true
    $script:LocalMode = 'cloud'
    $script:Started = New-Object System.Collections.ArrayList
    $script:Stopped = New-Object System.Collections.ArrayList
    $script:Identity = New-Object System.Collections.ArrayList
    $script:PortListening = $false
    $script:PortOwner = $null
    $script:Stage = $null
    $script:ExitAfterStart = $false
    $script:Cycle = 0
}
function Write-Step([string]$Message) { [void]$script:Log.Add($Message) }
function Write-Host { }
function Start-Sleep { }
function Get-CloudStage { $script:Stage }
function Get-CurrentCloudZone { $script:Zone }
function Assert-PythonAvailable { 'python' }
function Assert-CloudSshIdentity { param($PythonExe, $Zone) [void]$script:Identity.Add($Zone) }
function Get-RemoteAppProbe {
    param($Zone)
    if ($script:Probes.Count) { return (ConvertFrom-RemoteAppProbe -Output $script:Probes.Dequeue()) }
    return (ConvertFrom-RemoteAppProbe -Output 'LOCALLIFE_PROBE http=200 listen=127.0.0.1:8000 pid=42')
}
function Get-LocalDashboardStatus { param($LocalPort) @{ ok = $script:LocalOk; mode = $script:LocalMode; gpu = 'Tesla T4'; depth = 'ON'; error = 'refused' } }
function Test-LocalPortListening { param($LocalPort) $script:PortListening }
function Get-LocalPortOwner { param($LocalPort) $script:PortOwner }
function Stop-ProcessTree { param($ProcessId) [void]$script:Stopped.Add($ProcessId) }
function Start-CloudTunnelProcess {
    param($Zone)
    [void]$script:Started.Add($Zone)
    $process = [pscustomobject]@{ HasExited = $script:ExitAfterStart }
    return @{ process = $process; pid = 1000 + $script:Started.Count; via = 'fake'; log = (Join-Path $script:SessionDirectory 'no-such-log.txt') }
}

# 1. Probe parsing names the exact VM-side stage.
$p = ConvertFrom-RemoteAppProbe -Output "noise`nLOCALLIFE_PROBE http=000 listen=none pid=none"
Assert-True ($p.stage -eq 'vm_app_not_running' -and -not $p.ready) 'probe: app not running'
$p = ConvertFrom-RemoteAppProbe -Output 'LOCALLIFE_PROBE http=000 listen=none pid=77'
Assert-True ($p.stage -eq 'vm_app_loading') 'probe: running but not listening'
$p = ConvertFrom-RemoteAppProbe -Output 'LOCALLIFE_PROBE http=503 listen=127.0.0.1:8000 pid=77'
Assert-True ($p.stage -eq 'vm_app_unhealthy' -and $p.message -match '503') 'probe: listening but unhealthy'
$p = ConvertFrom-RemoteAppProbe -Output '' -Result @{ timed_out = $true; stderr = '' }
Assert-True ($p.stage -eq 'probing_vm' -and $p.message -match 'timed out') 'probe: ssh timeout is reported'
Assert-True ((Join-ProcessArguments @('a', 'b c', '--command=echo x')) -eq 'a "b c" "--command=echo x"') 'argument quoting'

# 2. Delayed server readiness: no forwarding until the VM app answers HTTP 200.
Reset-Fakes
$script:Probes.Enqueue('LOCALLIFE_PROBE http=000 listen=none pid=none')
$script:Probes.Enqueue('LOCALLIFE_PROBE http=000 listen=none pid=42')
Invoke-CloudTunnelSupervisor -MaxCycles 5
$readyIndex = @($script:Log | ForEach-Object { $_ }).IndexOf((@($script:Log | Where-Object { $_ -like 'DASHBOARD READY*' }) | Select-Object -First 1))
Assert-True ($script:Started.Count -eq 1) 'delayed readiness: one tunnel started'
Assert-True (@($script:Log | Where-Object { $_ -match 'vm_app_not_running' }).Count -eq 1 -and
             @($script:Log | Where-Object { $_ -match 'vm_app_loading' }).Count -eq 1) 'delayed readiness: each waiting stage reported once'
Assert-True ($readyIndex -gt 0 -and $script:Log[$readyIndex] -match 'processing mode CLOUD; GPU Tesla T4; Logitech depth ON' -and
             $script:Log[$readyIndex] -match 'europe-west4-b') 'delayed readiness: ready line shows URL, zone, mode, GPU'

# 3. SSH auth alone never announces readiness: tunnel up but HTTP fails -> no READY, restart after 3 misses.
Reset-Fakes
$script:LocalOk = $false
Invoke-CloudTunnelSupervisor -MaxCycles 5
Assert-True (@($script:Log | Where-Object { $_ -like 'DASHBOARD READY*' }).Count -eq 0) 'no READY without HTTP through the tunnel'
Assert-True ($script:Stopped.Count -ge 1 -and $script:Started.Count -ge 2) 'unanswered tunnel is restarted'

# 4. Zone change: reconnect to the new zone with a fresh identity check.
Reset-Fakes
$script:Cycle = 0
function Get-CurrentCloudZone { $script:Cycle++; if ($script:Cycle -le 2) { 'europe-west4-b' } else { 'europe-west1-c' } }
Invoke-CloudTunnelSupervisor -MaxCycles 4
Assert-True ($script:Started -contains 'europe-west4-b' -and $script:Started -contains 'europe-west1-c' -and $script:Stopped.Count -ge 1) 'zone change: old tunnel stopped, new zone forwarded'
Assert-True ($script:Identity -contains 'europe-west1-c') 'zone change: identity re-checked for the new zone'
function Get-CurrentCloudZone { $script:Zone }

# 5. Stale local port: a leftover plink is stopped; a LOCAL server is refused.
Reset-Fakes
$script:PortListening = $true
$script:PortOwner = @{ pid = 555; name = 'plink' }
function Test-LocalPortListening { param($LocalPort) $r = $script:PortListening; $script:PortListening = $false; $r }
Invoke-CloudTunnelSupervisor -MaxCycles 2
Assert-True ($script:Stopped -contains 555 -and $script:Started.Count -eq 1) 'stale plink tunnel on the port is stopped'
function Test-LocalPortListening { param($LocalPort) $script:PortListening }
Reset-Fakes
$script:PortListening = $true
$script:PortOwner = @{ pid = 777; name = 'python' }
$script:LocalMode = 'local'
$threw = $null
try { Invoke-CloudTunnelSupervisor -MaxCycles 2 } catch { $threw = $_.Exception.Message }
Assert-True ($threw -match 'LOCAL Local Life server' -and $script:Started.Count -eq 0) 'local-mode server on the port is refused, not mislabelled'

# 6. Tunnel reconnect with backoff, then a terminal error naming the stage.
Reset-Fakes
$script:ExitAfterStart = $true
$threw = $null
try { Invoke-CloudTunnelSupervisor -MaxCycles 40 -MaxConsecutiveTunnelFailures 3 } catch { $threw = $_.Exception.Message }
Assert-True ($script:Started.Count -eq 3 -and $threw -match 'local_forward') 'repeated tunnel exits end in a staged terminal error'

# 7. Window 1 failure stops the wait immediately.
Reset-Fakes
$script:Stage = [pscustomobject]@{ status = 'failed'; stage = 'installing_dependencies'; error_message = 'pip failed' }
$threw = $null
try { Invoke-CloudTunnelSupervisor -MaxCycles 3 } catch { $threw = $_.Exception.Message }
Assert-True ($threw -match 'installing_dependencies' -and $script:Started.Count -eq 0) 'Window 1 failure is surfaced'

# 8. A backend that reports LOCAL behind the tunnel is flagged, never called cloud.
Reset-Fakes
$script:LocalMode = 'local'
Invoke-CloudTunnelSupervisor -MaxCycles 2
Assert-True (@($script:Log | Where-Object { $_ -match 'processing mode LOCAL' }).Count -eq 1) 'mode reported by the backend is shown as-is'

# 9. Pi cloud-tunnel key setup reaches the Pi without raw double quotes (PowerShell 5.1 mangles
#    them for ssh.exe) and, run by bash, creates the key once and prints the public half.
$bootstrapFn = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'ConvertTo-RemoteBootstrap' }, $true) | Select-Object -First 1
. ([scriptblock]::Create($bootstrapFn.Extent.Text))
$keyAssign = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$keySetupCommand' }, $true) | Select-Object -First 1
$keySetupCommand = & ([scriptblock]::Create($keyAssign.Right.Extent.Text))
$keyCall = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.Extent.Text -match 'keySetupCommand' -and $n.GetCommandName() -eq 'ssh' }, $true) | Select-Object -First 1
Assert-True ($null -ne $keyCall -and $keyCall.Extent.Text -match 'ConvertTo-RemoteBootstrap') 'Pi key setup is sent base64-wrapped'
$wire = ConvertTo-RemoteBootstrap -Command $keySetupCommand
Assert-True ($wire -notmatch '"' -and $keySetupCommand -match '""') 'bootstrap removes the embedded double quotes'
$bash = Get-Command bash -ErrorAction SilentlyContinue
$keygen = Get-Command ssh-keygen -ErrorAction SilentlyContinue
if ($bash -and $keygen -and -not $IsWindows) {
    $fakeHome = Join-Path ([IO.Path]::GetTempPath()) ('pi-home-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $fakeHome | Out-Null
    $env:HOME_BACKUP = $env:HOME; $env:HOME = $fakeHome
    try {
        $first = (& bash -c $wire) -join "`n"
        $second = (& bash -c $wire) -join "`n"
    }
    finally { $env:HOME = $env:HOME_BACKUP }
    Assert-True ($first -match '^ssh-ed25519 \S+ locallife-cloud-tunnel$' -and $first -eq $second) 'key setup runs in bash, creates the key once, prints it'
    Remove-Item -Recurse -Force $fakeHome
}

# 10. Top-level roles (Stop, Doctor, Launcher) end after their own work and never reach the
#     child-window LAN address check; child roles still get it.
$mainTry = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.TryStatementAst] -and
    $n.Body.Extent.Text -match "Role -eq 'Stop'" -and $n.Body.Extent.Text -match 'Assert-ValidLanAddress' }, $false) | Select-Object -First 1
function Stop-Demo { [void]$script:Calls.Add('Stop-Demo') }
function Start-Demo { [void]$script:Calls.Add('Start-Demo') }
function Invoke-Doctor { [void]$script:Calls.Add('Invoke-Doctor') }
function Start-AppRole { [void]$script:Calls.Add('Start-AppRole') }
function Assert-ValidLanAddress { param($Address) throw ('The laptop network address is invalid: ' + $Address) }
$mainBody = $mainTry.Body.Extent.Text.Trim()
$mainBody = $mainBody.Substring(1, $mainBody.Length - 2)   # drop the try block's own braces
foreach ($case in @(@('Stop', 'Stop-Demo'), @('Doctor', 'Invoke-Doctor'), @('Launcher', 'Start-Demo'))) {
    $script:Calls = New-Object System.Collections.ArrayList
    $Role = $case[0]; $SessionId = ''; $LanAddress = ''
    $threw = $null
    try { . ([scriptblock]::Create($mainBody)) } catch { $threw = $_.Exception.Message }
    Assert-True ($null -eq $threw -and $script:Calls -contains $case[1]) ("role " + $case[0] + " finishes without the LAN address check")
}
$script:Calls = New-Object System.Collections.ArrayList
$Role = 'App'; $LanAddress = ''
$threw = $null
try { . ([scriptblock]::Create($mainBody)) } catch { $threw = $_.Exception.Message }
Assert-True ($threw -match 'network address is invalid' -and -not ($script:Calls -contains 'Start-AppRole')) 'child role still validates its LAN address'

if ($script:Failures) { Write-Output "$($script:Failures) FAILED"; exit 1 }
Write-Output 'ALL PASSED'
exit 0
