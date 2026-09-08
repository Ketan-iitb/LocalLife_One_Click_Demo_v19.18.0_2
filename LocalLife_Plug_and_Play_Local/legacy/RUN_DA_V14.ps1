param(
    [string]$Pi = "http://192.168.0.123:5005"
)

$ErrorActionPreference = "Stop"

Set-Location "C:\Users\HP\LocalLife_Final"

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: C:\Users\HP\LocalLife_Final\.venv was not found."
    Write-Host "Activate/create the Windows virtual environment first."
    exit 1
}

Write-Host "Starting LocalLife V14 Depth Anything worker..."
Write-Host "Pi endpoint: $Pi"

& ".\.venv\Scripts\python.exe" ".\da_worker_v14.py" --pi $Pi
