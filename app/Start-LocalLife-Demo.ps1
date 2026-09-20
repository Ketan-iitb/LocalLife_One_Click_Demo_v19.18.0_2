# The Windows launchers (LocalLife.exe / LocalLife_Check.exe /
# LocalLife_Stop.exe) run "<folder>\app\Start-LocalLife-Demo.ps1 -Role <role>".
# The real launcher lives at the repository root so there is exactly one copy
# to maintain; this forwards every argument to it unchanged.
$launcher = Join-Path (Split-Path -Parent $PSScriptRoot) 'Start-LocalLife-Demo.ps1'
if (-not (Test-Path -LiteralPath $launcher)) {
    Write-Error "Could not locate the Local Life launcher at $launcher"
    exit 1
}
& $launcher @args
exit $LASTEXITCODE
