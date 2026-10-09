@echo off
setlocal
title Bonsai 2 - RTX 5060 Ti
rem This terminal owns the server. Ctrl+C stops it and releases GPU memory.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\rtx5060-run.ps1" %*
set "ninfer_exit=%ERRORLEVEL%"
if not "%ninfer_exit%"=="0" (
    echo.
    echo Bonsai could not start. Read the error above and the logs\launch directory.
    pause
)
endlocal & exit /b %ninfer_exit%
