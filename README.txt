ECOVISION SMARTSORT - VERSIÓN HÍBRIDA YOLO + CLIP
==================================================

OBJETIVO
--------
Esta versión intenta reducir las clasificaciones erróneas del modelo YOLO.

Flujo:
1. YOLO localiza una región candidata.
2. CLIP vuelve a revisar visualmente esa región.
3. Si CLIP considera que es una persona, fondo u objeto no relacionado,
   la detección se rechaza.
4. Si YOLO y CLIP difieren, CLIP puede corregir la categoría cuando tiene
   suficiente seguridad.
5. Si la predicción es ambigua, EcoVision muestra "No identificado" en
   lugar de forzar una respuesta.

IMPORTANTE
----------
Este proyecto NO incluye el archivo best.pt porque tu modelo entrenado está
guardado en Google Drive.

Descarga:
EcoVision_Entrenamiento/modelo_v2-2/weights/best.pt

y colócalo en:

EcoVision_Hibrido_YOLO_CLIP/models/best.pt


PRIMERA EJECUCIÓN
-----------------
La primera vez, Transformers descargará:
openai/clip-vit-base-patch32

Por eso necesitarás conexión a Internet y puede tardar algunos minutos.
Después queda en la caché del equipo.

WINDOWS / VS CODE
-----------------
1. Abre esta carpeta en VS Code.
2. Abre Terminal > New Terminal.
3. Ejecuta:

py -m venv .venv
.\.venv\Scripts\Activate.ps1

Si PowerShell bloquea la activación:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Luego:

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py


MODOS
-----
- Cámara en vivo con área central de escaneo.
- Tomar fotografía.
- Subir imagen.

NOTA DE RENDIMIENTO
-------------------
CLIP es más pesado que YOLO. La cámara en vivo no verifica todos los frames,
sino uno cada varios fotogramas para evitar que el equipo se vuelva demasiado lento.

CLASES
------
0 papel
1 vidrio
2 organico
3 plastico
