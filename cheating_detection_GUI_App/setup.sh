#!/bin/bash
# Quick Setup Script for Cheating Detection GUI
# Run this script to set up the project quickly

echo "🎓 Cheating Detection GUI - Setup Script"
echo "========================================"

# Step 1: Create Virtual Environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate venv
if [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    venv\Scripts\activate
fi

# Step 2: Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Download YOLO model (optional - auto downloads on first run)
echo "🤖 YOLO11L model will auto-download on first run"
echo ""

# Step 4: Ready to run
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   python main.py"
echo ""
echo "📋 For GPU support (optional):"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
