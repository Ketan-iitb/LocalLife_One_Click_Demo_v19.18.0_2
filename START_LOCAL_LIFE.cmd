@echo off
setlocal
REM One-click start: brings up the local control service and opens the welcome
REM page, where the operator chooses local, cloud or automatic mode.
REM Runs from the repository root -- Start-LocalLife-Demo.ps1 lives here, next
REM to this file, and the control service invokes it by absolute path.
cd /d "%~dp0"

if not exist "%~dp0Start-LocalLife-Demo.ps1" (
    echo.
    echo ERROR: Start-LocalLife-Demo.ps1 was not found next to this file.
    echo Expected: %~dp0Start-LocalLife-Demo.ps1
    echo.
    echo You are probably on a branch that does not contain it, or the folder
    echo was only partly extracted. Run:  git switch Working_branch_v21_cloud_local_launcher
    echo.
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0LocalLife_Plug_and_Play_Local;%PYTHONPATH%"

REM Prefer the py launcher (installed by python.org and not shadowed by the
REM Microsoft Store stub), then fall back to python on PATH.
where /q py.exe && (
    py -3 -m locallife_cloud.launcher_service
) || (
    where /q python.exe && (
        python -m locallife_cloud.launcher_service
    ) || (
        echo.
        echo ERROR: No Python 3 interpreter was found on PATH.
        echo Install Python 3 from python.org, then run this file again.
        echo.
        pause
        exit /b 1
    )
)
if errorlevel 1 pause
