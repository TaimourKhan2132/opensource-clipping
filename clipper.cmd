@echo off
rem Runs opensource-clipping with the project's .venv (created by setup.cmd).
rem Usage: clipper --url "VIDEO_URL" [options]   -- defaults live in clipper.py
set PYTHONUTF8=1
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo .venv not found. Run setup.cmd first.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0clipper.py" %*
