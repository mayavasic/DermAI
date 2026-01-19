
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Configuration du chemin du modèle (Chemin absolu vers votre bureau)
MODEL_PATH = r"C:\Users\mayav\OneDrive\Bureau\DermAI_Final\model\vgg16_finetuned_janvier.keras"

# Chargement du modèle au démarrage
try:
    model = load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier reçu'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'})

    # Sauvegarde temporaire de l'image pour l'analyse
    temp_filename = "predict_temp.jpg"
    file.save(temp_filename)
    
    try:
        # Prétraitement de l'image (224x224 comme lors de l'entraînement)
        img = image.load_img(temp_filename, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction par le modèle
        preds = model.predict(img_array)
        score = float(preds[0][0]) # Probabilité que ce soit malin
        
        # Détermination du label et de la confiance
        if score > 0.5:
            label = "Maligne (Cancer suspecté)"
            confidence = score * 100
        else:
            label = "Bénigne (Sain)"
            confidence = (1 - score) * 100
            
        return jsonify({
            'prediction': label,
            'probability': f"{confidence:.2f}%",
            'raw_score': score # Utilisé pour l'aiguille de la jauge
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Suppression de l'image temporaire
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
