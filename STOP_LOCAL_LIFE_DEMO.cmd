@echo off
setlocal enabledelayedexpansion
REM Stops whichever mode (Local or Cloud) is currently running -- the
REM launcher remembers which one it started and, in Cloud mode, offers
REM to run "python gpu.py down" for you so billing stops.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalLife-Demo.ps1" -Role Stop

pause
