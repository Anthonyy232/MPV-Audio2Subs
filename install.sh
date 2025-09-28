#!/bin/bash

# Minimal environment setup for Linux/macOS to create reproducible venv and deps
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

echo "Starting AI Subtitle Service client installation for Linux/macOS..."

check_dependency() {
    # Fail early if required system tools are missing to avoid partial installs
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not found in your PATH."
        echo "Please install $1 and ensure it is accessible."
        exit 1
    fi
}

check_dependency python3
check_dependency ffmpeg

# Create a virtualenv once so repeated runs are safe and idempotent
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in './$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate and install pinned dependencies for reproducible environments
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages."
    deactivate
    exit 1
fi

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Deactivate the virtual environment after installation to leave a clean shell
deactivate

echo "--------------------------------------------------"
echo "[V] Installation complete!"
echo "You can now use the AI Subtitle service in mpv."
echo "--------------------------------------------------"