@echo off
REM One-click start: brings up the local control service and opens the welcome
REM page, where the operator chooses local, cloud or automatic mode.
cd /d "%~dp0LocalLife_Plug_and_Play_Local"
python -m locallife_cloud.launcher_service
if errorlevel 1 pause
