from pathlib import Path
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="EcoVision SmartSort",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
YOLO_PATH = BASE_DIR / "models" / "best.pt"

YOLO_CONF = 0.30
CLIP_MIN_CONF = 0.42
CLIP_MIN_MARGIN = 0.06
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LIVE_EVERY_N_FRAMES = 6
MODEL_LOCK = threading.Lock()


# ============================================================
# PALETA / ESTILO
# ============================================================

st.markdown(
    """
<style>
:root {
    --eco-green: #1f7a4c;
    --eco-green-dark: #145c39;
    --eco-green-soft: #eaf6ef;
    --eco-mint: #dff3e8;
    --eco-cream: #f8fbf8;
    --eco-text: #17352a;
    --eco-muted: #66766f;
    --eco-border: #dbe8df;
    --eco-white: #ffffff;
    --eco-shadow: 0 10px 30px rgba(31, 122, 76, 0.08);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(98, 190, 132, 0.11), transparent 28%),
        radial-gradient(circle at top right, rgba(208, 232, 216, 0.25), transparent 24%),
        #f8fbf8;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.eco-hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 34px 38px;
    background: linear-gradient(135deg, #145c39 0%, #1f7a4c 52%, #3f9d68 100%);
    box-shadow: 0 20px 60px rgba(31, 122, 76, 0.18);
    margin-bottom: 24px;
    color: white;
}

.eco-hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -70px;
    top: -85px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}

.eco-hero::before {
    content: "";
    position: absolute;
    width: 170px;
    height: 170px;
    right: 80px;
    bottom: -105px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}

.eco-badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.20);
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .03em;
    margin-bottom: 14px;
}

.eco-title {
    font-size: clamp(2.15rem, 5vw, 3.35rem);
    font-weight: 850;
    line-height: 1.04;
    margin: 0;
    letter-spacing: -0.03em;
}

.eco-subtitle {
    max-width: 720px;
    font-size: 1.03rem;
    line-height: 1.65;
    margin-top: 12px;
    margin-bottom: 0;
    color: rgba(255,255,255,0.90);
}

.section-kicker {
    color: var(--eco-green);
    font-weight: 800;
    font-size: .82rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.section-title {
    color: var(--eco-text);
    font-size: 1.5rem;
    font-weight: 820;
    margin: 0 0 16px 0;
}

.waste-card {
    position: relative;
    min-height: 145px;
    border-radius: 22px;
    padding: 20px;
    background: white;
    border: 1px solid var(--eco-border);
    box-shadow: var(--eco-shadow);
    transition: transform .18s ease, box-shadow .18s ease;
}

.waste-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 38px rgba(31, 122, 76, 0.12);
}

.waste-icon {
    font-size: 1.9rem;
    margin-bottom: 12px;
}

.waste-name {
    color: var(--eco-text);
    font-weight: 820;
    font-size: 1.08rem;
    margin-bottom: 5px;
}

.waste-bin {
    color: var(--eco-muted);
    font-size: .9rem;
}

.color-dot {
    width: 9px;
    height: 9px;
    display: inline-block;
    border-radius: 50%;
    margin-right: 6px;
}

.eco-tip {
    background: linear-gradient(135deg, #edf8f1, #f8fcf9);
    border: 1px solid #d5eadc;
    border-radius: 18px;
    padding: 15px 18px;
    color: #315646;
    margin: 20px 0 8px 0;
    box-shadow: 0 8px 24px rgba(31,122,76,0.05);
}

.eco-tip strong {
    color: #1f7a4c;
}

.result-card {
    border-radius: 22px;
    padding: 20px;
    background: white;
    border: 1px solid var(--eco-border);
    box-shadow: var(--eco-shadow);
    margin-bottom: 12px;
}

.result-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #eaf6ef;
    color: #1f7a4c;
    font-size: .78rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.result-title {
    color: var(--eco-text);
    font-size: 1.35rem;
    font-weight: 850;
    margin-bottom: 10px;
}

.result-line {
    font-size: .95rem;
    color: #4b6258;
    margin-bottom: 6px;
}

.confidence-track {
    width: 100%;
    height: 9px;
    border-radius: 999px;
    background: #edf3ef;
    overflow: hidden;
    margin: 9px 0 4px 0;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2f8b58, #64b47f);
}

.step-card {
    min-height: 115px;
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--eco-border);
}

.step-number {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: #1f7a4c;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    margin-bottom: 10px;
}

.step-title {
    font-weight: 820;
    color: var(--eco-text);
    margin-bottom: 4px;
}

.step-desc {
    font-size: .88rem;
    color: var(--eco-muted);
    line-height: 1.45;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 6px;
    border-radius: 18px;
    background: #eef5f0;
    border: 1px solid #dce9e0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 13px;
    padding: 9px 16px;
    height: auto;
    color: #496257;
    font-weight: 720;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1f7a4c !important;
    box-shadow: 0 6px 16px rgba(31,122,76,0.10);
}

[data-testid="stFileUploader"] {
    border-radius: 18px;
}

.stButton > button {
    border-radius: 14px;
    font-weight: 750;
    border: none;
    background: #1f7a4c;
    color: white;
}

.eco-footer {
    margin-top: 34px;
    padding-top: 22px;
    border-top: 1px solid var(--eco-border);
    text-align: center;
    color: var(--eco-muted);
    font-size: .85rem;
}

@media (max-width: 700px) {
    .eco-hero {
        padding: 26px 22px;
        border-radius: 22px;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CLASES
# ============================================================

CLASES = {
    0: {"nombre": "Papel", "contenedor": "Contenedor azul", "color": (255, 0, 0), "emoji": "📘"},
    1: {"nombre": "Vidrio", "contenedor": "Contenedor verde", "color": (0, 170, 0), "emoji": "🍾"},
    2: {"nombre": "Orgánico", "contenedor": "Contenedor marrón", "color": (19, 69, 139), "emoji": "🍃"},
    3: {"nombre": "Plástico", "contenedor": "Contenedor amarillo", "color": (0, 215, 255), "emoji": "🧴"},
}


# ============================================================
# PROMPTS CLIP
# ============================================================

CLIP_PROMPTS = {
    "papel": [
        "a paper bag",
        "a sheet of paper",
        "cardboard",
        "paper waste",
        "crumpled paper",
    ],
    "vidrio": [
        "a glass bottle",
        "a glass jar",
        "a drinking glass",
        "glass waste",
        "a transparent glass container",
    ],
    "organico": [
        "fruit waste",
        "food scraps",
        "organic waste",
        "fruit peel",
        "vegetable waste",
    ],
    "plastico": [
        "a plastic bottle",
        "a plastic bag",
        "a plastic container",
        "plastic waste",
        "a plastic cup",
        "a reusable plastic bottle",
        "a plastic water bottle",
    ],
    "otro": [
        "a person",
        "a human face",
        "furniture",
        "a room",
        "a phone",
        "a computer",
        "a random background",
        "an object that is not waste",
    ],
}

CLIP_TO_ID = {
    "papel": 0,
    "vidrio": 1,
    "organico": 2,
    "plastico": 3,
}


# ============================================================
# MODELOS
# ============================================================

@st.cache_resource
def cargar_yolo():
    if not YOLO_PATH.exists():
        return None
    return YOLO(str(YOLO_PATH))


@st.cache_resource
def cargar_clip():
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model = model.to(DEVICE)
    model.eval()
    return model, processor


yolo_model = cargar_yolo()

if yolo_model is None:
    st.error(
        "No encuentro el modelo `models/best.pt`. "
        "Copia tu archivo entrenado dentro de la carpeta `models`."
    )
    st.code("models/best.pt", language="text")
    st.stop()

with st.spinner("Preparando EcoVision..."):
    clip_model, clip_processor = cargar_clip()


# ============================================================
# CLIP
# ============================================================

def normalizar_tensor(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@st.cache_resource
def crear_embeddings_texto():
    etiquetas = []
    vectores_categoria = []

    with torch.no_grad():
        for categoria, prompts in CLIP_PROMPTS.items():
            inputs = clip_processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            text_outputs = clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            pooled_output = text_outputs.pooler_output
            txt_features = clip_model.text_projection(pooled_output)
            txt_features = normalizar_tensor(txt_features)

            promedio = txt_features.mean(dim=0, keepdim=True)
            promedio = normalizar_tensor(promedio)

            etiquetas.append(categoria)
            vectores_categoria.append(promedio)

    matriz = torch.cat(vectores_categoria, dim=0)
    return etiquetas, matriz


CLIP_LABELS, CLIP_TEXT_FEATURES = crear_embeddings_texto()


def clasificar_clip(imagen_bgr):
    if imagen_bgr is None or imagen_bgr.size == 0:
        return "otro", 0.0, 0.0

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(imagen_rgb)

    inputs = clip_processor(
        images=pil_image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with MODEL_LOCK, torch.no_grad():
        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        img_features = clip_model.visual_projection(
            pooled_output
        )

        img_features = normalizar_tensor(img_features)

        logits = 45.0 * img_features @ CLIP_TEXT_FEATURES.T

        probs = torch.softmax(logits, dim=-1)[0]

    valores, indices = torch.topk(probs, k=2)

    best_idx = int(indices[0].item())
    categoria = CLIP_LABELS[best_idx]
    confianza = float(valores[0].item())
    margen = float((valores[0] - valores[1]).item())

    return categoria, confianza, margen


# ============================================================
# FUSIÓN YOLO + CLIP
# ============================================================

def decidir_categoria(yolo_id, yolo_conf, crop_bgr):
    clip_cat, clip_conf, clip_margin = clasificar_clip(crop_bgr)

    if clip_cat == "otro":
        return None, {
            "yolo_conf": yolo_conf,
            "clip_cat": clip_cat,
            "clip_conf": clip_conf,
            "clip_margin": clip_margin,
            "motivo": "Fuera de las categorías",
        }

    clip_id = CLIP_TO_ID[clip_cat]

    if clip_conf < CLIP_MIN_CONF or clip_margin < CLIP_MIN_MARGIN:
        if clip_id == yolo_id and yolo_conf >= 0.72:
            final_id = yolo_id
        else:
            return None, {
                "yolo_conf": yolo_conf,
                "clip_cat": clip_cat,
                "clip_conf": clip_conf,
                "clip_margin": clip_margin,
                "motivo": "Predicción ambigua",
            }
    else:
        final_id = clip_id

    if final_id == yolo_id:
        final_conf = min(
            0.99,
            0.50 * yolo_conf
            + 0.50 * clip_conf
            + 0.08
        )
    else:
        final_conf = clip_conf

    return final_id, {
        "final_conf": final_conf,
        "yolo_id": yolo_id,
        "yolo_conf": yolo_conf,
        "clip_cat": clip_cat,
        "clip_conf": clip_conf,
        "clip_margin": clip_margin,
        "motivo": "Aceptado",
    }


# ============================================================
# DIBUJO
# ============================================================

def dibujar_etiqueta(imagen, x1, y1, x2, y2, clase_id, confianza):
    datos = CLASES[clase_id]
    color = datos["color"]

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    texto = f'{datos["nombre"]} {confianza * 100:.1f}%'

    (tw, th), _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    top = max(0, y1 - th - 16)
    bottom = max(th + 12, y1)

    cv2.rectangle(
        imagen,
        (x1, top),
        (x1 + tw + 12, bottom),
        color,
        -1,
    )

    cv2.putText(
        imagen,
        texto,
        (x1 + 6, max(th + 2, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# DETECCIÓN HÍBRIDA
# ============================================================

def detectar_hibrido(imagen_bgr):
    resultados = yolo_model.predict(
        source=imagen_bgr,
        conf=YOLO_CONF,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    salida = imagen_bgr.copy()
    detecciones = []

    alto, ancho = imagen_bgr.shape[:2]

    for resultado in resultados:
        if resultado.boxes is None:
            continue

        for caja in resultado.boxes:
            yolo_id = int(caja.cls[0])
            yolo_conf = float(caja.conf[0])

            if yolo_id not in CLASES:
                continue

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            x1 = max(0, min(x1, ancho - 1))
            x2 = max(1, min(x2, ancho))
            y1 = max(0, min(y1, alto - 1))
            y2 = max(1, min(y2, alto))

            if x2 <= x1 or y2 <= y1:
                continue

            area_caja = (x2 - x1) * (y2 - y1)
            area_imagen = ancho * alto
            proporcion = area_caja / max(area_imagen, 1)

            if proporcion > 0.72:
                continue

            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.08)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(ancho, x2 + pad_x)
            cy2 = min(alto, y2 + pad_y)

            crop = imagen_bgr[
                cy1:cy2,
                cx1:cx2
            ]

            final_id, meta = decidir_categoria(
                yolo_id,
                yolo_conf,
                crop,
            )

            if final_id is None:
                continue

            final_conf = meta["final_conf"]

            dibujar_etiqueta(
                salida,
                x1,
                y1,
                x2,
                y2,
                final_id,
                final_conf,
            )

            datos = CLASES[final_id]

            detecciones.append({
                "clase": datos["nombre"],
                "confianza": final_conf,
                "contenedor": datos["contenedor"],
                "emoji": datos["emoji"],
                "yolo": CLASES[yolo_id]["nombre"],
                "yolo_conf": yolo_conf,
                "clip": meta["clip_cat"],
                "clip_conf": meta["clip_conf"],
            })

    detecciones.sort(
        key=lambda d: d["confianza"],
        reverse=True
    )

    return salida, detecciones


# ============================================================
# RESULTADOS
# ============================================================

def mostrar_resultados(detecciones):
    if not detecciones:
        html = (
            '<div class="result-card">'
            '<div class="result-label">🔎 Resultado</div>'
            '<div class="result-title">No identificado</div>'
            '<div class="result-line">EcoVision no encontró suficiente seguridad para asignar una categoría.</div>'
            '<div class="result-line">Intenta acercar el objeto, mejorar la iluminación o usar otro ángulo.</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        return

    st.markdown(
        '<div class="section-kicker">Análisis completado</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Resultado de clasificación</div>',
        unsafe_allow_html=True,
    )

    for d in detecciones:
        porcentaje = max(
            0,
            min(100, d["confianza"] * 100)
        )

        html = (
            '<div class="result-card">'
            '<div class="result-label">✓ Verificación híbrida</div>'
            f'<div class="result-title">{d["emoji"]} {d["clase"]}</div>'
            f'<div class="result-line">Confianza combinada: <b>{porcentaje:.1f}%</b></div>'
            '<div class="confidence-track">'
            f'<div class="confidence-fill" style="width:{porcentaje:.1f}%"></div>'
            '</div>'
            f'<div class="result-line" style="margin-top:12px;">♻️ Depositar en: <b>{d["contenedor"]}</b></div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


# ============================================================
# CÁMARA EN VIVO
# ============================================================

class EcoVisionHybridProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_annotated = None

    def recv(self, frame):
        imagen = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        display = imagen.copy()
        alto, ancho = display.shape[:2]

        rx1 = int(ancho * 0.18)
        ry1 = int(alto * 0.18)
        rx2 = int(ancho * 0.82)
        ry2 = int(alto * 0.82)

        cv2.rectangle(
            display,
            (rx1, ry1),
            (rx2, ry2),
            (215, 245, 225),
            2,
        )

        cv2.putText(
            display,
            "Coloca el residuo dentro del area",
            (rx1 + 8, max(28, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if (
            self.frame_count % LIVE_EVERY_N_FRAMES == 0
            or self.last_annotated is None
        ):
            try:
                roi = imagen[
                    ry1:ry2,
                    rx1:rx2
                ]

                analizada, _ = detectar_hibrido(
                    roi
                )

                display[
                    ry1:ry2,
                    rx1:rx2
                ] = analizada

                self.last_annotated = display

            except Exception:
                self.last_annotated = display

        return av.VideoFrame.from_ndarray(
            self.last_annotated
            if self.last_annotated is not None
            else display,
            format="bgr24",
        )


RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="eco-hero">
    <div class="eco-badge">IA + VISIÓN ARTIFICIAL</div>
    <h1 class="eco-title">EcoVision SmartSort</h1>
    <p class="eco-subtitle">
        Identifica residuos de forma inteligente y recibe una recomendación
        inmediata sobre el contenedor adecuado.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CATEGORÍAS
# ============================================================

st.markdown(
    '<div class="section-kicker">Clasificación disponible</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">¿Qué puede reconocer EcoVision?</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="waste-card"><div class="waste-icon">📘</div><div class="waste-name">Papel</div><div class="waste-bin"><span class="color-dot" style="background:#2f80ed;"></span>Contenedor azul</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="waste-card"><div class="waste-icon">🍾</div><div class="waste-name">Vidrio</div><div class="waste-bin"><span class="color-dot" style="background:#2e9d58;"></span>Contenedor verde</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        '<div class="waste-card"><div class="waste-icon">🍃</div><div class="waste-name">Orgánico</div><div class="waste-bin"><span class="color-dot" style="background:#8a5b35;"></span>Contenedor marrón</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        '<div class="waste-card"><div class="waste-icon">🧴</div><div class="waste-name">Plástico</div><div class="waste-bin"><span class="color-dot" style="background:#e9b824;"></span>Contenedor amarillo</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TIP
# ============================================================

st.markdown(
    '<div class="eco-tip"><strong>💡 Consejo:</strong> para obtener una mejor clasificación, muestra uno o pocos residuos claramente visibles, con buena iluminación y sin demasiados objetos alrededor.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODOS
# ============================================================

st.markdown(
    '<div class="section-kicker" style="margin-top:28px;">Modo de análisis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Elige cómo quieres identificar el residuo</div>',
    unsafe_allow_html=True,
)

tab_live, tab_upload = st.tabs(
    [
        "📹 Cámara en vivo",
        "🖼️ Subir imagen",
    ]
)


# ============================================================
# TAB 1 - CÁMARA EN VIVO
# ============================================================

with tab_live:
    col_a, col_b = st.columns([1.55, 1])

    with col_a:
        st.markdown("### Escáner en tiempo real")
        st.write(
            "Activa la cámara y coloca el residuo dentro del área central."
        )

        webrtc_streamer(
            key="ecovision-hybrid-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EcoVisionHybridProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

    with col_b:
        html = (
            '<div class="result-card">'
            '<div class="result-label">📹 Modo en vivo</div>'
            '<div class="result-title">Cómo usar el escáner</div>'
            '<div class="result-line">1. Pulsa <b>START</b>.</div>'
            '<div class="result-line">2. Permite el acceso a la cámara.</div>'
            '<div class="result-line">3. Coloca el residuo dentro del recuadro.</div>'
            '<div class="result-line">4. Mantén el objeto estable unos segundos.</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


# ============================================================
# TAB 2 - SUBIR IMAGEN
# ============================================================

with tab_upload:
    st.markdown("### Analiza una imagen guardada")
    st.write(
        "Sube una fotografía desde tu dispositivo."
    )

    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if archivo is not None:
        imagen_rgb = np.array(
            Image.open(archivo).convert("RGB")
        )

        imagen_bgr = cv2.cvtColor(
            imagen_rgb,
            cv2.COLOR_RGB2BGR
        )

        with st.spinner("Analizando imagen..."):
            resultado_bgr, detecciones = detectar_hibrido(
                imagen_bgr
            )

        resultado_rgb = cv2.cvtColor(
            resultado_bgr,
            cv2.COLOR_BGR2RGB
        )

        izquierda, derecha = st.columns(
            [1.45, 1]
        )

        with izquierda:
            st.image(
                resultado_rgb,
                caption="Vista analizada",
                use_container_width=True,
            )

        with derecha:
            mostrar_resultados(detecciones)


# ============================================================
# CÓMO FUNCIONA
# ============================================================

st.markdown(
    '<div class="section-kicker" style="margin-top:36px;">Tecnología</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">¿Cómo trabaja EcoVision?</div>',
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        '<div class="step-card"><div class="step-number">1</div><div class="step-title">Detecta</div><div class="step-desc">YOLO localiza el posible residuo dentro de la imagen.</div></div>',
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        '<div class="step-card"><div class="step-number">2</div><div class="step-title">Verifica</div><div class="step-desc">CLIP revisa visualmente la categoría propuesta.</div></div>',
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        '<div class="step-card"><div class="step-number">3</div><div class="step-title">Recomienda</div><div class="step-desc">EcoVision indica la categoría y el contenedor correspondiente.</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="eco-footer">
    <b>EcoVision SmartSort</b><br>
    Proyecto académico de visión artificial · YOLO + CLIP<br>
    <span style="font-size:.78rem;">
        La clasificación generada por inteligencia artificial puede presentar errores.
    </span>
</div>
""",
    unsafe_allow_html=True,
)
