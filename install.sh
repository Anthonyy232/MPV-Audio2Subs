#!/bin/bash

# --- Configuration ---
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

echo "Starting AI Subtitle Service client installation for Linux/macOS..."
echo "-----------------------------------------------------------------"

# --- Dependency Checks ---

check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not found in your PATH."
        echo "Please install $1 and ensure it is accessible."
        exit 1
    fi
}

check_dependency python3
check_dependency ffmpeg
check_dependency docker

# --- Virtual Environment Setup ---

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in './$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# --- Activation and Installation ---

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

# Deactivate the virtual environment after installation
deactivate

echo "--------------------------------------------------"
echo "[V] Installation complete!"
echo "Remember to start Docker Desktop/Daemon before running the service."
echo "--------------------------------------------------"