import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
import cv2
import io
import os

st.set_page_config(page_title="Détection VidéoCap - Sécurité routière", layout="wide")

@st.cache_resource
def get_config(): 
    # On charge le JSON pour avoir les noms de classes et les paramètres
    config_path = 'models/pipeline_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"model_name": "YOLOv11s", "version": "1.0.0"}
    
# Chargement du modèle
@st.cache_resource
def load_model():
    return YOLO("models/best_model_v11s.pt")

config = get_config()
model = load_model()

# --- Interface ---

st.title("Analyse de Trafic - VideoCap")
st.write(f"Modèle : **{config['model_name']}** | Version : {config['version']}")

# Ajout du slider de confiance dans la sidebar
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Seuil de confiance (Confidence)", 0.0, 1.0, 0.25)

img_size = st.sidebar.selectbox("Taille d'image (Inférence)", [640, 800, 1024], index=1)

uploaded_file = st.sidebar.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Conversion de l'image pour PIL
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image originale", width="stretch")

    # Inférence
    with st.spinner('Analyse en cours...'):
        results = model.predict(image, conf=conf_threshold, imgsz=img_size, iou=0.7, augment=True)
        
        # Récupération de l'image annotée
        res_plottes_bgr = results[0].plot()
        res_plottes_rgb = cv2.cvtColor(res_plottes_bgr, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.image(res_plottes_rgb, caption=f"Détections {config['model_name']}", width="stretch")

        result_img_pil = Image.fromarray(res_plottes_rgb)
        buf = io.BytesIO()
        result_img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Télécharger le résultat",
            data=byte_im,
            file_name="detection_videocap.png",
            mime="image/png"
        )
        
    # Affichages des statistiques
    st.write("---")
    st.write("### Résultats de l'analyse")

    detected_classes = results[0].boxes.cls.cpu().numpy()
    if len(detected_classes) == 0:
        st.info("Aucun objet détecté avec ce seuil de confiance. Essaie de baisser le seuil dans la barre latérale.")

    else:
        unique_classes, counts = np.unique(detected_classes, return_counts=True)
        # Création des colonnes pour les badges de stats
        stats_cols = st.columns(len(unique_classes) if len(unique_classes) < 5 else 4)
        for idx, (cls_idx, count) in enumerate(zip(unique_classes, counts)):
            name = model.names[int(cls_idx)]
            with stats_cols[idx % 4]:
                st.metric(label=name.upper(), value=int(count))

print("Succès")