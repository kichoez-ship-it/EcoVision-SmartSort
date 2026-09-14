@echo off
title EcoVision Hibrido YOLO + CLIP
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Falta el entorno virtual.
    echo Ejecuta primero INSTALAR.bat
    pause
    exit /b 1
)

if not exist "models\best.pt" (
    echo.
    echo FALTA models\best.pt
    echo Copia tu modelo entrenado dentro de la carpeta models.
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
streamlit run app.py
pause
