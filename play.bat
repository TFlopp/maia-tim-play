@echo off
python "%~dp0maia_tim\server.py" %*
if errorlevel 1 pause
