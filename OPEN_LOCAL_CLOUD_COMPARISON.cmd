@echo off
setlocal
REM Opens the local-vs-cloud comparison table (http://127.0.0.1:8765/compare).
REM Keep it open during a local run and during a cloud run: it records the last
REM numbers of each mode, so both columns fill in. It starts nothing and costs nothing.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0LocalLife_Plug_and_Play_Local;%PYTHONPATH%"
where /q py.exe && (
    py -3 -m locallife_cloud.launcher_service --compare
) || (
    python -m locallife_cloud.launcher_service --compare
)
if errorlevel 1 pause
