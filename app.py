import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
import cv2
import io
import os
import tempfile
import time

# Fonction de prétraitement

def apply_clahe(image_np):
    """Améliore le contraste local pour les scènes sombres ou à faible contraste."""
    # Conversion PIL -> OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)

    # Passage en espace LAB pour traiter la luminosité séparément
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Application du CLAHE sur le canal L (Lunimosité)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Recomposition
    limg = cv2.merge((cl,a,b))
    final_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Retour en RGB pour Streamlit/PIL
    return cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VideoCap · Détection d'obstacles",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "VideoCap — Système de détection d'obstacles en temps réel"}
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #080c12;
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,220,200,0.07) 0%, transparent 70%),
        linear-gradient(180deg, #080c12 0%, #0b1019 100%);
}

[data-testid="stSidebar"] {
    background: #0d1520 !important;
    border-right: 1px solid rgba(0,220,200,0.12);
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #6ee7e0 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid rgba(0,220,200,0.1);
    margin-bottom: 1.5rem;
}
.sidebar-brand .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #00dcc8; box-shadow: 0 0 8px #00dcc8;
    animation: pulse 2s infinite;
}
.sidebar-brand span {
    font-size: 0.8rem; color: #00dcc8 !important;
    letter-spacing: 0.15em; text-transform: uppercase; font-weight: 700;
}

[data-testid="stSlider"] > div > div > div > div { background: #00dcc8 !important; }
[data-testid="stSlider"] > div > div > div { background: rgba(0,220,200,0.15) !important; }

[data-testid="stFileUploader"] {
    background: rgba(0,220,200,0.03);
    border: 1px dashed rgba(0,220,200,0.25) !important;
    border-radius: 8px;
}

.page-header {
    display: flex; align-items: flex-end; justify-content: space-between;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.page-header .title-block .eyebrow {
    font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase;
    color: #00dcc8; font-weight: 600; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.page-header .title-block h1 {
    font-size: 2.2rem; font-weight: 800; color: #ffffff;
    line-height: 1; margin: 0; letter-spacing: -0.02em;
}
.page-header .title-block h1 span { color: #00dcc8; }
.page-header .meta-block { display: flex; gap: 1.5rem; align-items: center; }
.meta-pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; padding: 6px 14px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; color: #8a9ab8;
}
.meta-pill .label { color: #6ee7e0; margin-right: 6px; }

.mode-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 14px; border-radius: 100px;
    font-size: 0.65rem; font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;
    border: 1px solid;
}
.mode-badge.image  { color: #00dcc8; border-color: rgba(0,220,200,0.3);  background: rgba(0,220,200,0.06); }
.mode-badge.video  { color: #a78bfa; border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.06); }
.mode-badge.webcam { color: #f87171; border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.06); }

.status-bar { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.status-chip {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px; padding: 8px 16px;
    font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #6b7e9a;
}
.status-chip.active { border-color: rgba(0,220,200,0.25);  color: #00dcc8; }
.status-chip.warn   { border-color: rgba(255,180,0,0.25);   color: #ffb400; }
.status-chip.live   { border-color: rgba(248,113,113,0.35); color: #f87171; }
.status-chip.vio    { border-color: rgba(167,139,250,0.3);  color: #a78bfa; }

.panel-label {
    font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #4a5f7a; font-family: 'JetBrains Mono', monospace; font-weight: 500;
    margin-bottom: 8px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.stats-header { display: flex; align-items: center; gap: 12px; margin: 2.5rem 0 1.5rem; }
.stats-header .line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,220,200,0.3), transparent); }
.stats-header .label {
    font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #00dcc8; font-family: 'JetBrains Mono', monospace; font-weight: 600;
}

div[data-testid="stMetric"] {
    background: #0d1520 !important;
    border: 1px solid rgba(0,220,200,0.12) !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00dcc8, #0057ff);
}
div[data-testid="stMetric"] label {
    font-size: 0.62rem !important; letter-spacing: 0.18em !important;
    text-transform: uppercase !important; color: #4a6a8a !important;
    font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 2.4rem !important; font-weight: 800 !important;
    color: #ffffff !important; line-height: 1.1 !important;
}

.stProgress > div > div > div > div { background: linear-gradient(90deg, #00dcc8, #0057ff) !important; }
.stProgress > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 4px !important; }

div[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid rgba(0,220,200,0.4) !important;
    color: #00dcc8 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase; border-radius: 6px !important; padding: 10px 20px !important;
    transition: all 0.2s ease; width: 100%;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,220,200,0.08) !important;
    box-shadow: 0 0 20px rgba(0,220,200,0.15);
}

div[data-testid="stButton"] > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 0.08em !important;
    border-radius: 6px !important; padding: 10px 20px !important;
}

.stSpinner > div { border-top-color: #00dcc8 !important; }

div[data-testid="stAlert"] {
    background: rgba(255,180,0,0.06) !important;
    border: 1px solid rgba(255,180,0,0.2) !important;
    border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; color: #ffb400 !important;
}

.video-info-card {
    background: #0d1520; border: 1px solid rgba(167,139,250,0.2);
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.video-info-card .vi-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
}
.video-info-card .vi-row:last-child { border-bottom: none; }
.video-info-card .vi-key { color: #4a6a8a; }
.video-info-card .vi-val { color: #c8d8e8; font-weight: 600; }

.webcam-note {
    background: rgba(248,113,113,0.05); border: 1px solid rgba(248,113,113,0.2);
    border-radius: 10px; padding: 1.4rem; text-align: center;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #f87171;
    line-height: 1.8;
}
.webcam-note .wn-icon { font-size: 2rem; margin-bottom: 8px; }

hr { border-color: rgba(255,255,255,0.06) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: rgba(0,220,200,0.3); border-radius: 4px; }

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #00dcc8; }
    50%       { opacity: 0.4; box-shadow: 0 0 3px #00dcc8; }
}

.empty-state { text-align: center; padding: 5rem 2rem; color: #2a3a50; }
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state p {
    font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def get_config():
    config_path = 'models/pipeline_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"model_name": "YOLO11m", "version": "1.0.0"}

@st.cache_resource
def load_model():
    return YOLO("models/best_model_v11m_v2.pt")

config     = get_config()
model      = load_model()
model_name = config.get("model_name", "YOLO11m")
version    = config.get("version", "1.0.0")


# ─────────────────────────────────────────────
# HELPER : rendu des statistiques
# ─────────────────────────────────────────────
def render_stats(results_list):
    """Agrège et affiche les stats de détection sur une liste de résultats YOLO."""
    valid = [r for r in results_list if len(r.boxes)]
    if not valid:
        st.info("⚠️  Aucun objet détecté à ce seuil. Réduire la valeur de confiance.")
        return

    all_cls  = np.concatenate([r.boxes.cls.cpu().numpy()  for r in valid])
    all_conf = np.concatenate([r.boxes.conf.cpu().numpy() for r in valid])

    st.markdown("""
    <div class="stats-header">
        <div class="label">Objets détectés</div>
        <div class="line"></div>
    </div>""", unsafe_allow_html=True)

    unique_classes, counts = np.unique(all_cls, return_counts=True)
    n_cols     = min(len(unique_classes), 5)
    stats_cols = st.columns(n_cols)

    for idx, (cls_idx, count) in enumerate(zip(unique_classes, counts)):
        name     = model.names[int(cls_idx)]
        mask     = all_cls == cls_idx
        avg_conf = float(all_conf[mask].mean()) if mask.sum() > 0 else 0.0
        with stats_cols[idx % n_cols]:
            st.metric(label=name.upper(), value=int(count), delta=f"conf moy. {avg_conf:.0%}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total détections",   int(len(all_cls)))
    with c2: st.metric("Classes distinctes", int(len(unique_classes)))
    with c3: st.metric("Confiance globale",  f"{float(all_conf.mean()):.1%}")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="dot"></div>
        <span>Système actif</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🎛️ Mode d'entrée")
    mode = st.radio(
        "mode",
        ["📷  Image", "🎬  Vidéo", "📡  Temps réel (webcam)"],
        label_visibility="collapsed"
    )
    mode_key = "image" if "Image" in mode else ("video" if "Vidéo" in mode else "webcam")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    st.markdown("<br>", unsafe_allow_html=True)

    conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.15, 0.01,
                                help="Seuil minimal pour afficher une détection")
    
    st.markdown("---")
    st.markdown("### Amélioration Image")
    use_clahe = st.checkbox("Optimisation Contraste (CLAHE)", value=False,
                            help="Utile pour les piétons dans l'ombre ou par mauvais temps.")
    
    img_size = st.selectbox("Résolution d'inférence", [640, 800, 1024], index=1,
                             help="Taille d'image pour l'inférence YOLO")

    frame_skip = 2
    save_video = True
    webcam_idx = 0
    max_frames = 30

    if mode_key == "video":
        st.markdown("<br>", unsafe_allow_html=True)
        frame_skip = st.slider("Traiter 1 frame sur N", 1, 10, 2,
                                help="1 = toutes les frames (lent), 10 = rapide mais moins précis")
        save_video = st.toggle("Sauvegarder la vidéo annotée", value=True)

    elif mode_key == "webcam":
        st.markdown("<br>", unsafe_allow_html=True)
        webcam_idx = st.number_input("Index caméra", min_value=0, max_value=5, value=0, step=1)
        max_frames = st.slider("Durée max (secondes)", 5, 120, 30)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    if mode_key == "image":
        st.markdown("### 📂 Image source")
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"],
                                          label_visibility="collapsed")
    elif mode_key == "video":
        st.markdown("### 📂 Vidéo source")
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Vidéo", type=["mp4", "avi", "mov", "mkv"],
                                          label_visibility="collapsed")
    else:
        uploaded_file = None

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#2a3a50;line-height:1.8">
        MODÈLE &nbsp;·&nbsp; <span style="color:#4a6a8a">{model_name}</span><br>
        VERSION &nbsp;·&nbsp; <span style="color:#4a6a8a">v{version}</span><br>
        IOU &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:#4a6a8a">0.70</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────
mode_badges = {
    "image":  '<span class="mode-badge image">◉ Image</span>',
    "video":  '<span class="mode-badge video">▶ Vidéo</span>',
    "webcam": '<span class="mode-badge webcam">● Live</span>',
}

st.markdown(f"""
<div class="page-header">
    <div class="title-block">
        <div class="eyebrow">// Analyse de trafic urbain &nbsp; {mode_badges[mode_key]}</div>
        <h1>Video<span>Cap</span></h1>
    </div>
    <div class="meta-block">
        <div class="meta-pill"><span class="label">modèle</span>{model_name}</div>
        <div class="meta-pill"><span class="label">conf</span>{conf_threshold:.2f}</div>
        <div class="meta-pill"><span class="label">imgsz</span>{img_size}</div>
    </div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODE : IMAGE
# ═════════════════════════════════════════════
if mode_key == "image":

    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🎯</div>
            <p>Chargez une image via la barre latérale<br>pour lancer l'analyse de détection d'obstacles.</p>
        </div>""", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)

        # --- LOGIQUE DE PRÉTRAITEMENT ---
        if use_clahe:
            with st.spinner("Optimisation du contraste..."):
                image_to_predict = apply_clahe(image)
                label_source = "// Image traitée (CLAHE)"
        else:
            image_to_predict = image
            label_source = "// Image originale"
        #-----------------------------------------------------

        st.markdown(f"""
        <div class="status-bar">
            <div class="status-chip active">◉ Modèle — {model_name}</div>
            <div class="status-chip active">◈ {image.width}×{image.height}px</div>
            <div class="status-chip warn">◎ Conf ≥ {conf_threshold:.0%}</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(f'<div class="panel-label">{label_source}</div>', unsafe_allow_html=True)
            st.image(image_to_predict, use_container_width=True)

        with st.spinner("🔍 Analyse en cours..."):
            results = model.predict(image_to_predict, conf=conf_threshold, imgsz=img_size, iou=0.7, augment=True)
            res_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown('<div class="panel-label">// Détections annotées</div>', unsafe_allow_html=True)
            st.image(res_rgb, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(res_rgb).save(buf, format="PNG")
            st.download_button("↓  Télécharger le résultat", buf.getvalue(),
                               "videocap_detection.png", "image/png")

        render_stats([results[0]])


# ═════════════════════════════════════════════
# MODE : VIDÉO
# ═════════════════════════════════════════════
elif mode_key == "video":

    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🎬</div>
            <p>Chargez une vidéo via la barre latérale<br>pour lancer l'analyse frame par frame.</p>
        </div>""", unsafe_allow_html=True)
    else:
        # Sauvegarde temporaire (OpenCV ne lit pas les BytesIO)
        suffix   = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tfile    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name
        tfile.close()

        cap      = cv2.VideoCapture(video_path)
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_fr / fps_src if fps_src else 0
        cap.release()

        st.markdown(f"""
        <div class="video-info-card">
            <div class="vi-row"><span class="vi-key">Fichier</span><span class="vi-val">{uploaded_file.name}</span></div>
            <div class="vi-row"><span class="vi-key">Résolution</span><span class="vi-val">{w}×{h}</span></div>
            <div class="vi-row"><span class="vi-key">FPS source</span><span class="vi-val">{fps_src:.1f}</span></div>
            <div class="vi-row"><span class="vi-key">Frames totales</span><span class="vi-val">{total_fr:,}</span></div>
            <div class="vi-row"><span class="vi-key">Durée estimée</span><span class="vi-val">{duration:.1f}s</span></div>
            <div class="vi-row"><span class="vi-key">Frames à analyser</span><span class="vi-val">~{total_fr // frame_skip:,} (1/{frame_skip})</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="status-bar">
            <div class="status-chip vio">▶ Vidéo chargée</div>
            <div class="status-chip active">◉ Modèle — {model_name}</div>
            <div class="status-chip warn">◎ Conf ≥ {conf_threshold:.0%}</div>
        </div>""", unsafe_allow_html=True)

        if st.button("▶  Lancer l'analyse vidéo"):

            out_path = None
            writer   = None
            if save_video:
                out_path = tempfile.mktemp(suffix="_annotated.mp4")
                writer   = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps_src / frame_skip, 1),
                    (w, h)
                )

            cap         = cv2.VideoCapture(video_path)
            frame_idx   = 0
            processed   = 0
            all_results = []

            st.markdown('<div class="panel-label">// Aperçu — dernière frame annotée</div>',
                        unsafe_allow_html=True)
            preview_slot = st.empty()
            prog_slot    = st.empty()
            status_slot  = st.empty()

            t_start = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res   = model.predict(rgb, conf=conf_threshold, imgsz=img_size, iou=0.7, verbose=False)
                    annot = res[0].plot()   # BGR
                    all_results.append(res[0])
                    processed += 1

                    if writer is not None:
                        writer.write(annot)

                    if processed % 5 == 0 or processed == 1:
                        preview_slot.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB),
                                           use_container_width=True)

                    pct     = frame_idx / max(total_fr - 1, 1)
                    elapsed = time.time() - t_start
                    fps_cur = processed / elapsed if elapsed > 0 else 0
                    prog_slot.progress(pct)
                    status_slot.markdown(
                        f'<div class="status-bar">'
                        f'<div class="status-chip vio">Frame {frame_idx+1}/{total_fr}</div>'
                        f'<div class="status-chip active">{processed} analysées</div>'
                        f'<div class="status-chip warn">{fps_cur:.1f} fps</div>'
                        f'</div>', unsafe_allow_html=True
                    )

                frame_idx += 1

            cap.release()
            if writer:
                writer.release()

            elapsed_total = time.time() - t_start
            status_slot.markdown(
                f'<div class="status-bar">'
                f'<div class="status-chip active">✓ Terminé — {processed} frames en {elapsed_total:.1f}s</div>'
                f'</div>', unsafe_allow_html=True
            )

            # Téléchargement vidéo annotée
            if save_video and out_path and os.path.exists(out_path):
                st.markdown('<div class="panel-label" style="margin-top:1.5rem">// Vidéo annotée</div>',
                            unsafe_allow_html=True)
                with open(out_path, "rb") as f:
                    st.download_button("↓  Télécharger la vidéo annotée", f.read(),
                                       "videocap_detection.mp4", "video/mp4")

            if all_results:
                render_stats(all_results)

            # Nettoyage
            try:
                os.unlink(video_path)
            except Exception:
                pass
            if out_path:
                try:
                    os.unlink(out_path)
                except Exception:
                    pass


# ═════════════════════════════════════════════
# MODE : WEBCAM — TEMPS RÉEL
# ═════════════════════════════════════════════
elif mode_key == "webcam":

    st.markdown(f"""
    <div class="status-bar">
        <div class="status-chip live">● Mode temps réel</div>
        <div class="status-chip active">◉ Modèle — {model_name}</div>
        <div class="status-chip warn">◎ Conf ≥ {conf_threshold:.0%}</div>
        <div class="status-chip live">⏱ Max {max_frames}s</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="webcam-note">
        <div class="wn-icon">📡</div>
        <strong>Mode flux webcam</strong><br>
        Ce mode capture votre webcam locale et applique la détection YOLO frame par frame.<br>
        <span style="color:#fca5a5">Cliquez sur "Démarrer" pour activer la caméra.</span><br>
        <span style="opacity:.6">Autorisez l'accès à la caméra dans votre navigateur si demandé.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_b1, col_b2, _ = st.columns([1, 1, 4])
    start_btn = col_b1.button("▶  Démarrer")
    stop_btn  = col_b2.button("■  Arrêter")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        st.markdown('<div class="panel-label">// Flux temps réel annoté</div>', unsafe_allow_html=True)
        frame_slot = st.empty()
        info_slot  = st.empty()

        cap = cv2.VideoCapture(int(webcam_idx))

        if not cap.isOpened():
            st.error(f"❌ Impossible d'ouvrir la caméra (index {webcam_idx}). "
                     f"Vérifiez les permissions ou changez l'index dans la barre latérale.")
            st.session_state.webcam_running = False
        else:
            target_frames = int(max_frames * 30)
            frame_count   = 0
            t_start       = time.time()
            all_results   = []

            while st.session_state.webcam_running and frame_count < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                t0    = time.time()
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = model.predict(rgb, conf=conf_threshold, imgsz=img_size, iou=0.7, verbose=False)
                annot = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
                all_results.append(res[0])

                inf_ms  = (time.time() - t0) * 1000
                fps     = 1000 / inf_ms if inf_ms > 0 else 0
                elapsed = time.time() - t_start
                n_det   = len(res[0].boxes)

                frame_slot.image(annot, use_container_width=True)
                info_slot.markdown(
                    f'<div class="status-bar">'
                    f'<div class="status-chip live">● LIVE &nbsp; {fps:.0f} fps</div>'
                    f'<div class="status-chip active">⏱ {elapsed:.0f}s / {max_frames}s</div>'
                    f'<div class="status-chip active">🎯 {n_det} objet(s)</div>'
                    f'<div class="status-chip warn">⚡ {inf_ms:.0f} ms/frame</div>'
                    f'</div>', unsafe_allow_html=True
                )

                frame_count += 1

            cap.release()
            st.session_state.webcam_running = False

            elapsed_total = time.time() - t_start
            st.success(f"✅ Session terminée — {frame_count} frames analysées en {elapsed_total:.1f}s")

            if all_results:
                render_stats(all_results)