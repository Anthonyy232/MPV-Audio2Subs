@echo off
setlocal

echo ============================================
echo  AI Subtitle Service Installation
echo ============================================
echo.

REM --- Prerequisite Checks ---
echo [*] Checking prerequisites...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Error: Python not found in PATH
    echo     Please install Python 3.9+ and add it to PATH
    goto :error
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Error: FFmpeg not found in PATH
    echo     Please install FFmpeg and add it to PATH
    goto :error
)

echo [OK] Prerequisites found

REM --- Create Virtual Environment ---
set VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo.
    echo [*] Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [X] Failed to create virtual environment
        goto :error
    )
)

REM --- Activate and Install ---
echo.
echo [*] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [*] Installing package in development mode...
python -m pip install -e . --quiet
if %errorlevel% neq 0 (
    echo [X] Failed to install package
    goto :error
)

echo [*] Installing ASR dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: Some ASR dependencies failed
)

REM --- GPU Support ---
echo.
echo [*] Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: GPU support installation failed
    echo     Falling back to CPU mode
)

echo.
echo ============================================
echo [OK] Installation complete!
echo.
echo Next steps:
echo   1. Configure MPV IPC in mpv.conf:
echo      input-ipc-server=\\.\pipe\mpv-socket
echo.
echo   2. Add keybinding in input.conf:
echo      l script-message toggle_ai_subtitles
echo.
echo   3. Copy this folder to MPV scripts directory:
echo      %%APPDATA%%\mpv\scripts\
echo ============================================
goto :end

:error
echo.
echo ============================================
echo [X] Installation failed
echo     Please fix the errors above and retry
echo ============================================

:end
endlocal
pause
