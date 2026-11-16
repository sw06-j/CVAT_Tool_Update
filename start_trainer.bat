@echo off
title CVAT Auto-Trainer Live Logs
color 0A
cd /d "%~dp0"
echo ========================================================
echo CVAT Real-time Auto Fine-Tuning Receiver
echo Port: 5000 - Model Weights: model_weights.pt
echo ========================================================
python auto_trainer.py
if errorlevel 1 (
    echo.
    echo [ERROR] Auto-trainer exited with error.
)
pause
