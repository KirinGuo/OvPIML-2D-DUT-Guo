@echo off
setlocal
cd /d "%~dp0"
python "%~dp0setup_env.py" %*
set "code=%ERRORLEVEL%"
pause
exit /b %code%
