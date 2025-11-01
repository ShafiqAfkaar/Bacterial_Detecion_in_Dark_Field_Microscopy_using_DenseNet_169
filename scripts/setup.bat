@echo off
echo Setting up Bacterial Detection Project...
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul

echo.
echo Setup completed!
echo You can now run:
echo - scripts\train_model.bat for training
echo - scripts\evaluate_model.bat for evaluation
pause