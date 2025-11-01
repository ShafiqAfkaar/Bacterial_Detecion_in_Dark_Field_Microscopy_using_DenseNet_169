@echo off
echo Starting Bacterial Detection Training...
echo.

cd /d "%~dp0.."
python main.py --mode train

echo.
echo Training completed!
pause