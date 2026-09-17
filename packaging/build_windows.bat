@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
set "BUILD_VENV=%ROOT%\.build-venv"
set "PYTHON_CMD=py -3"

where py >nul 2>&1
if errorlevel 1 (
    set "PYTHON_CMD=python"
)

if not exist "%BUILD_VENV%" (
    call %PYTHON_CMD% -m venv "%BUILD_VENV%"
    if errorlevel 1 exit /b 1
)

pushd "%ROOT%"

call "%BUILD_VENV%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r "%ROOT%\requirements.txt" pyinstaller
if errorlevel 1 exit /b 1

python -m PyInstaller --clean --noconfirm "%ROOT%\packaging\NeuroPOMDP.spec"
if errorlevel 1 exit /b 1

echo.
echo Build complete.
echo Output: %ROOT%\dist\NeuroPOMDP\NeuroPOMDP.exe

popd

endlocal
