@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Local Python environment not found at:
    echo   %PYTHON%
    echo.
    echo Create the repository virtual environment and install boto3 first.
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0scripts\aws\start_instance_and_vscode_remote.py" --stop-on-exit
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo AWS training launcher failed with exit code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
