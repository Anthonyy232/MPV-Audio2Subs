@echo off
setlocal

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
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: docker is not found in your PATH.
    goto :error
)

set VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo Creating Python virtual environment in '.\%VENV_DIR%'...
    python -m venv "%VENV_DIR%"
)

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

echo.
echo --------------------------------------------------
echo [V] Installation complete!
echo Make sure Docker Desktop is running before you start the service.
echo --------------------------------------------------
goto :end

:error
echo.
echo Installation failed. Please resolve the errors above and run the script again.

:end
endlocal
pause