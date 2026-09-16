@echo off
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0training\local8gb\launch.ps1" %*
set "FAUST_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %FAUST_EXIT%
pause
exit /b %FAUST_EXIT%
