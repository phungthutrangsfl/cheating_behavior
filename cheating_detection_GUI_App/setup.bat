@echo off
REM Quick Setup Script for Cheating Detection GUI (Windows)
REM Run this batch file to set up the project quickly

echo.
echo 🎓 Cheating Detection GUI - Setup Script (Windows)
echo ================================================
echo.

REM Step 1: Create Virtual Environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate venv
call venv\Scripts\activate.bat

REM Step 2: Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo 🚀 To run the application:
echo    python main.py
echo.
echo 📋 For GPU support (optional - NVIDIA CUDA):
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
pause
