@echo off
rem One-time Windows setup. Runs setup.ps1 without changing your PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
