@echo off
REM This batch file activates the Python virtual environment
REM and then runs the real-time transcription script.

echo Activating Python virtual environment...
call ".\.venv\Scripts\activate.bat"

echo Setting UV cache directory...
set UV_CACHE_DIR=D:\uv-cache

echo.
echo uv cache dir is
uv cache dir

echo.
echo Starting batch process stt result (batch.py)...
python batch.py

echo.
echo Script has finished or was exited. Press any key to close this window.
pause

