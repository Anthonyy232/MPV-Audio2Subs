@echo off
setlocal

echo Starting AI Subtitle Service client installation...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: python is not found in your PATH.
    @echo off
    setlocal

    REM Ensure required executables exist to fail early and avoid partial installs
    echo Starting AI Subtitle Service client installation...
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: python is not found in your PATH.
        goto :error
    )
    where ffmpeg >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: ffmpeg is not found in your PATH.
        goto :error
    )

    set VENV_DIR=venv

    REM Create venv if missing so repeated runs are safe
    if not exist "%VENV_DIR%" (
        echo Creating Python virtual environment in '.\%VENV_DIR%'...
        python -m venv "%VENV_DIR%"
    )

    REM Activate and install pinned dependencies for reproducible environments
    echo Activating virtual environment and installing dependencies...
    call "%VENV_DIR%\Scripts\activate.bat"

    echo Upgrading pip...
    python -m pip install --upgrade pip

    echo Installing Python packages from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install required packages.
        goto :error
    )

    REM Optionally install GPU-accelerated PyTorch wheel if desired
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

    echo.
    echo --------------------------------------------------
    echo [V] Installation complete!
    echo You can now use the AI Subtitle service in mpv.
    echo --------------------------------------------------
    goto :end

    :error
    echo.
    echo Installation failed. Please resolve the errors above and run the script again.

    :end
    endlocal
    pause