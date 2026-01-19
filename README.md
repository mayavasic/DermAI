http://127.0.0.1:5000

---

# DermAI - Classification de Lésions Cutanées via CNN (VGG16)

DermAI est une application web d'aide au dépistage dermatologique utilisant l'apprentissage profond (Deep Learning) pour classifier des images de lésions cutanées en deux catégories : **Bénignes** ou **Malignes**.

## 🚀 Résumé de l'Entraînement IA

L'objectif de cette partie était de concevoir un modèle robuste capable de généraliser sur des clichés variés (luminosité, types de peau) malgré les contraintes de données médicales.

### 📊 Dataset

* **Sources :** Combinaison des bases ISIC 2020 et ISIC_DICM_17k.
* **Volume :** 5 000 images réelles.
* 3 000 images bénignes.
* 2 000 images malignes.


* **Stratégie d'équilibrage :** Équilibrage par échantillonnage réel (Oversampling manuel) sans recours à la pondération mathématique (`class_weights`) pour préserver une sensibilité naturelle.

### 🧠 Architecture & Modèle

Nous avons opté pour le **Transfer Learning** avec l'architecture **VGG16**.

* **Pourquoi VGG16 ?** Pour sa stabilité et sa capacité à capturer les textures dermatologiques sans sur-apprentissage (Overfitting), contrairement aux modèles plus profonds comme ResNet.
* **Prétraitement :** Redimensionnement en 224x224 et normalisation des pixels [0, 1].
* **Data Augmentation :** Rotations, zooms et flips horizontaux pour renforcer la robustesse face aux photos prises par smartphone.

### 📈 Résultats obtenus

Le modèle a été entraîné sous environnement GPU (Google Colab) :



* **Précision (Accuracy) de validation :** **87.1%**
* **Stabilité :** Écart de seulement 2% entre l'entraînement et la validation.
* **Sensibilité :** Optimisée pour réduire les faux négatifs (sécurité médicale).

## 🛠️ Installation et Utilisation (Local)

> **Note importante :** En raison des limitations de taille de GitHub, le fichier du modèle entraîné (`vgg16_finetuned_janvier.keras`) n'est pas inclus dans ce dépôt. Pour lancer l'analyse le propriétaire doit activer le Serveur Flask en local.

Accéder à l'interface via `http://127.0.0.1:5000`.

## 💻 Technologies utilisées

* **Langage :** Python
* **IA :** TensorFlow / Keras
* **Backend :** Flask
* **Frontend :** HTML5 / CSS3 / JavaScript
* **IDE :** Visual Studio Code

---

**Avertissement :** *Ce projet est un prototype à but éducatif. Il ne remplace en aucun cas l'avis d'un professionnel de santé.*

---
