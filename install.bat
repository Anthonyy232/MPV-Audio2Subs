@echo off
setlocal

echo Starting AI Subtitle Service client installation...

REM --- Prerequisite Checks ---
echo Checking for required software...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Error: python is not found in your PATH.
    echo Please install Python and ensure it's added to your system's PATH.
    goto :error
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Error: ffmpeg is not found in your PATH.
    echo Please install ffmpeg and ensure it's added to your system's PATH.
    goto :error
)

echo [V] All prerequisites found.

REM --- Installation ---
set VENV_DIR=venv

REM Create venv if missing so repeated runs are safe
if not exist "%VENV_DIR%" (
    echo Creating Python virtual environment in '.\%VENV_DIR%'...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [X] Failed to create the virtual environment.
        goto :error
    )
)

REM Activate and install pinned dependencies for reproducible environments
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing Python packages from requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [X] Failed to install required packages from requirements.txt.
    goto :error
)

REM Optionally install GPU-accelerated PyTorch wheel if desired
echo Installing PyTorch for CUDA...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [!] Warning: Failed to install GPU-accelerated PyTorch. The service may fall back to CPU.
)


echo.
echo --------------------------------------------------
echo [V] Installation complete!
echo You can now use the AI Subtitle service in mpv.
echo --------------------------------------------------
goto :end

:error
echo.
echo --------------------------------------------------
echo [X] Installation failed.
echo Please resolve the errors above and run the script again.
echo --------------------------------------------------

:end
endlocal
pause
