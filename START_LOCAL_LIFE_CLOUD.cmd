@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM LocalLife One-Click Demonstration Launcher
REM Cloud GPU Mode (via gpu.py, needs Google Cloud) -- this comment is
REM cosmetic only and is not kept in sync with the package version below;
REM the launcher always prints and runs the real installed version.
REM ============================================================
REM Double-click to bring up the GPU VM (hunts across zones for
REM capacity automatically) and start the demo on it.
REM Requires: Google Cloud CLI installed and signed in, and
REM gpu.py sitting next to this file (see DEMONSTRATION_INSTRUCTIONS.md).
REM Cloud charges apply while the VM is running.
REM For the free, laptop-only version, use START_LOCAL_LIFE_DEMO.cmd
REM ============================================================

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalLife-Demo.ps1" -Role Launcher -Mode Cloud

pause
