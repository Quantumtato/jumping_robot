@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Local Python environment not found at:
    echo   %PYTHON%
    pause
    exit /b 1
)

echo This will stop the jumping robot AWS instance and terminate its training jobs.
choice /C YN /N /M "Stop the AWS instance now? [Y/N] "
if errorlevel 2 exit /b 0

"%PYTHON%" "%~dp0scripts\aws\stop_instance.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo AWS shutdown failed with exit code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
