@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalLife-Demo.ps1" -Role Launcher
if errorlevel 1 pause
