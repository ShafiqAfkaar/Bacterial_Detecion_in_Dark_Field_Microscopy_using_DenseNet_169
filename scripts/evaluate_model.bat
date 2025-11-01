@echo off
echo Starting Bacterial Detection Evaluation...
echo.

cd /d "%~dp0.."
python main.py --mode evaluate

echo.
echo Evaluation completed!
pause
