@echo off
rem Execution policy override applies only to this PowerShell process.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_windows.ps1" %*
exit /b %errorlevel%
