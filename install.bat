@echo off
setlocal

echo ============================================
echo  AI Subtitle Service Installation
echo ============================================
echo.

REM --- Prerequisite Checks ---
echo [*] Checking prerequisites...

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] uv not found, attempting to install...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo [X] Error: Failed to install uv
        goto :error
    )
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
    echo [*] Creating virtual environment with uv...
    uv venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [X] Failed to create virtual environment
        goto :error
    )
)

REM --- Install Dependencies ---
echo.
echo [*] Installing dependencies with uv...

REM Install specialized PyTorch for RTX 50 series (Blackwell)
uv pip install --python "%VENV_DIR%\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: GPU support installation failed
    echo     Falling back to CPU mode
    uv pip install --python "%VENV_DIR%\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)

REM Install remaining dependencies and local package
uv pip install --python "%VENV_DIR%\Scripts\python.exe" -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: Some ASR dependencies failed
)

uv pip install --python "%VENV_DIR%\Scripts\python.exe" -e . --quiet
if %errorlevel% neq 0 (
    echo [X] Failed to install package in development mode
    goto :error
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
