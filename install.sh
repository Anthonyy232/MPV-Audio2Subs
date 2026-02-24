#!/bin/bash
set -e

echo "============================================"
echo " AI Subtitle Service Installation"
echo "============================================"
echo ""

# --- Prerequisite Checks ---
echo "[*] Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    echo "[*] uv not found, attempting to install..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
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
    echo "[*] Creating virtual environment with uv..."
    uv venv "$VENV_DIR"
fi

# --- Install Dependencies ---
echo ""
echo "[*] Installing dependencies with uv..."

# Detect CUDA availability (targeting cu128 for Blackwell/RTX 50 series)
if command -v nvidia-smi &> /dev/null; then
    echo "[*] NVIDIA GPU detected, installing CUDA 12.8 support..."
    uv pip install --python "$VENV_DIR/bin/python" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet || {
        echo "[!] Warning: GPU support installation failed"
        echo "[*] Falling back to standard PyTorch..."
        uv pip install --python "$VENV_DIR/bin/python" torch torchvision torchaudio --quiet
    }
else
    echo "[*] No NVIDIA GPU detected, installing CPU version..."
    uv pip install --python "$VENV_DIR/bin/python" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo "[*] Installing requirements and local package..."
uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt --quiet
uv pip install --python "$VENV_DIR/bin/python" -e . --quiet

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