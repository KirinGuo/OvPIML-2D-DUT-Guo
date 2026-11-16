@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run SETUP.cmd once to install the local Python environment.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0run.py" %*
set "code=%ERRORLEVEL%"
pause
exit /b %code%
