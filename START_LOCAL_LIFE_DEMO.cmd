@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM LocalLife One-Click Demonstration Launcher
REM Local Laptop Mode (No Cloud, Free) -- this comment is cosmetic only
REM and is not kept in sync with the package version below; the launcher
REM always prints and runs the real installed version.
REM ============================================================
REM Double-click to start the entire demo automatically.
REM For the cloud GPU version instead, use START_LOCAL_LIFE_CLOUD.cmd
REM ============================================================

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalLife-Demo.ps1" -Role Launcher -Mode Local

pause
