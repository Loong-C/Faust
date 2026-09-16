@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0eval\ab_v1\launch.ps1"
set "FAUST_EXIT=%ERRORLEVEL%"
echo.
echo Evaluation exit code: %FAUST_EXIT%
pause
exit /b %FAUST_EXIT%
