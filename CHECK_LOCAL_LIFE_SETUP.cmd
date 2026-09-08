@echo off
setlocal enabledelayedexpansion
REM Checks readiness for Local mode by default. To check Cloud mode
REM instead, run in PowerShell:
REM   Start-LocalLife-Demo.ps1 -Role Doctor -Mode Cloud

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalLife-Demo.ps1" -Role Doctor -Mode Local

pause
