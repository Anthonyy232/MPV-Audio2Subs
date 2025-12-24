#!/bin/bash
set -e

echo "============================================"
echo " AI Subtitle Service Installation"
echo "============================================"
echo ""

# --- Prerequisite Checks ---
echo "[*] Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "[X] Error: Python 3 not found"
    echo "    Please install Python 3.9+"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "[X] Error: FFmpeg not found"
    echo "    Please install FFmpeg"
    exit 1
fi

echo "[OK] Prerequisites found"

# --- Create Virtual Environment ---
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# --- Activate and Install ---
echo ""
echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[*] Installing package in development mode..."
pip install -e . --quiet

echo "[*] Installing ASR dependencies..."
pip install -r requirements.txt --quiet || echo "[!] Warning: Some ASR dependencies failed"

# --- GPU Support ---
echo ""
echo "[*] Installing PyTorch..."

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "[*] NVIDIA GPU detected, installing CUDA support..."
    pip install torch torchvision torchaudio --quiet || echo "[!] Warning: GPU support installation failed"
else
    echo "[*] No NVIDIA GPU detected, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet || echo "[!] Warning: PyTorch installation failed"
fi

echo ""
echo "============================================"
echo "[OK] Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Configure MPV IPC in mpv.conf:"
echo "     input-ipc-server=/tmp/mpv-socket"
echo ""
echo "  2. Add keybinding in input.conf:"
echo "     l script-message toggle_ai_subtitles"
echo ""
echo "  3. Copy this folder to MPV scripts directory:"
echo "     ~/.config/mpv/scripts/"
echo "============================================"