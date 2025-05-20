@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_HOME=%SCRIPT_DIR%env\python-3.9.13-embed-amd64
set PATH=%PYTHON_HOME%;%PYTHON_HOME%\Scripts;%PATH%
set PYTHONPATH=%SCRIPT_DIR%env\Lib\site-packages

cd %SCRIPT_DIR%
python feature_map_app.py
pause 