@echo off
title EcoVision Hibrido - Instalacion
cd /d "%~dp0"

echo =========================================
echo  ECOVISION HIBRIDO YOLO + CLIP
echo =========================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    py -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Instalacion terminada.
echo Ejecuta ahora EJECUTAR.bat
pause
