import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# --- CONFIGURATION ---
# Remplacez ceci par l'ID trouvé dans votre lien Google Drive
FILE_ID = 'https://drive.google.com/file/d/1S1dYVBf4nrXlTJSAesc33OxVlrUcgb2B/view?usp=sharing'
MODEL_FILE = 'vgg16_finetuned_janvier.keras'

@st.cache_resource
def load_my_model():
# Vérifier si le modèle est déjà là, sinon le télécharger
if not os.path.exists(MODEL_FILE):
url = f'https://drive.google.com/uc?id={FILE_ID}'
st.write("Téléchargement du modèle depuis Google Drive... (cela peut prendre une minute)")
gdown.download(url, MODEL_FILE, quiet=False)

# Charger le modèle
model = load_model(MODEL_FILE)
return model

st.title("Détection Dermatologie AI")

try:
with st.spinner('Chargement du modèle IA en cours...'):
model = load_my_model()
st.success("Modèle chargé !")
except Exception as e:
st.error(f"Erreur : Impossible de charger le modèle. Vérifiez l'ID Google Drive. Détails : {e}")

# Interface utilisateur
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
img = Image.open(uploaded_file)
st.image(img, caption='Image chargée', use_column_width=True)

if st.button('Analyser'):
# Redimensionnement (assurez-vous que 224x224 est la bonne taille pour votre modèle)
img = img.resize((224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
# img_array = img_array / 255.0 # Décommentez si nécessaire

prediction = model.predict(img_array)
st.write(f"Résultat brut : {prediction}")
