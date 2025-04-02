REM filepath: c:\Users\HP\OneDrive\Desktop\drf\bloom_api\run_server.bat
@echo off
set DJANGO_SETTINGS_MODULE=bloom_api.settings
set PYTHONPATH=%PYTHONPATH%;%CD%
set PATH=%PATH%;%USERPROFILE%\env1\Scripts
daphne -p 8000 bloom_api.asgi:application