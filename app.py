import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import xgboost as xgb
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configurazione percorsi modelli
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
XGBOOST_MODEL_PATH = 'xgboost_model.json'

# Variabili globali per i modelli e metadati
kmeans_model = None
xgboost_model = None
model_features = []
pegi_mapping_rev = {0: 3, 1: 7, 2: 12, 3: 16, 4: 18}

def init_models():
    global kmeans_model, xgboost_model, model_features
    
    # 1. Caricamento K-Means
    if os.path.exists(KMEANS_MODEL_PATH):
        try:
            kmeans_model = joblib.load(KMEANS_MODEL_PATH)
            print("[INFO] K-Means model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load K-Means model: {e}")
    else:
        print("[WARNING] kmeans_model.pkl not found!")

    # 2. Caricamento XGBoost
    if os.path.exists(XGBOOST_MODEL_PATH):
        try:
            xgboost_model = xgb.XGBClassifier()
            xgboost_model.load_model(XGBOOST_MODEL_PATH)
            # Leggiamo i nomi delle feature attese dal modello
            model_features = xgboost_model.get_booster().feature_names
            print(f"[INFO] XGBoost model loaded successfully. Features: {model_features}")
        except Exception as e:
            print(f"[ERROR] Failed to load XGBoost model: {e}")
    else:
        print("[WARNING] xgboost_model.json not found!")

# Chiamata di inizializzazione all'avvio
init_models()

def lab_to_rgb(lab_color):
    """Converte un colore LAB (OpenCV scale: L, A, B in [0, 255]) in RGB"""
    # OpenCV cvtColor si aspetta un array numpy a 3 dimensioni
    lab_pixel = np.array([[lab_color]], dtype=np.uint8)
    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
    return rgb_pixel[0][0].tolist()

def extract_color_features(image_bytes):
    """
    Riceve i byte dell'immagine, esegue la pipeline del Modulo 1:
    - Conversione in LAB
    - Ridimensionamento a 300x200
    - Predizione dei 5 cluster cromatici dominanti
    - Ordinamento per frequenza
    - Restituisce i centroidi flattati (15 elementi) e la palette RGB con percentuali
    """
    if kmeans_model is None:
        raise ValueError("Il modello K-Means non è stato caricato correttamente sul server.")

    # Carica l'immagine
    image_raw = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_bgr = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
    
    # Conversione LAB e resize
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    image_resized = cv2.resize(image_lab, (300, 200))
    pixels = image_resized.reshape(-1, 3)
    
    # Predizione dei cluster
    pixel_assignments = kmeans_model.predict(pixels)
    
    # Calcolo frequenze
    frequencies = np.bincount(pixel_assignments, minlength=5)
    
    # Ordinamento per dominanza (decrescente)
    sorted_indices = np.argsort(frequencies)[::-1]
    
    # Centroidi globali del modello
    global_centroids = kmeans_model.cluster_centers_
    sorted_centroids = global_centroids[sorted_indices]
    
    # Feature flattate (15 feature cromatiche: L1, A1, B1, ..., L5, A5, B5)
    flat_features = sorted_centroids.flatten()
    
    # Generazione dei dati RGB per la visualizzazione nel frontend
    total_pixels = pixels.shape[0]
    palette = []
    for i in range(5):
        lab_val = sorted_centroids[i]
        rgb = lab_to_rgb(lab_val)
        percentage = (frequencies[sorted_indices[i]] / total_pixels) * 100
        palette.append({
            "r": int(rgb[0]),
            "g": int(rgb[1]),
            "b": int(rgb[2]),
            "percentage": round(percentage, 1)
        })
        
    return flat_features, palette

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if xgboost_model is None or kmeans_model is None:
        return jsonify({
            "success": False, 
            "error": "I modelli di Machine Learning (K-Means o XGBoost) non sono caricati sul server."
        }), 500

    try:
        # 1. Recupero metadati
        price = float(request.form.get('price', 0.0))
        selected_genres = request.form.getlist('genres') # lista di generi selezionati (es. ['Action', 'Indie'])
        
        # 2. Recupero file immagine
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "Nessuna immagine fornita."}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # 3. Estrazione feature cromatiche LAB
        flat_color_features, color_palette = extract_color_features(image_bytes)
        
        # 4. Costruzione del dizionario delle feature per XGBoost
        # Prepariamo un record vuoto con tutte le feature attese dal modello in ordine
        feature_dict = {}
        
        # Inseriamo il prezzo
        feature_dict['price'] = price
        
        # Inseriamo le 15 feature cromatiche
        for idx in range(5):
            feature_dict[f'L{idx+1}'] = flat_color_features[idx*3]
            feature_dict[f'A{idx+1}'] = flat_color_features[idx*3 + 1]
            feature_dict[f'B{idx+1}'] = flat_color_features[idx*3 + 2]
            
        # Inseriamo i generi binari (Is_<Genre>)
        # Selezioniamo le feature di tipo genere attese dal modello
        genre_features_in_model = [f for f in model_features if f.startswith('Is_')]
        
        for gf in genre_features_in_model:
            # Estraiamo il nome del genere rimuovendo il prefisso 'Is_' e rimettendo gli spazi al posto di '_'
            genre_name = gf[3:].replace('_', ' ')
            # Gestiamo le sostituzioni specifiche effettuate nel MultiLabelBinarizer (es. '&' in 'Animation & Modeling')
            # I generi inviati dal frontend corrisponderanno ai nomi puliti
            is_present = 0
            for sg in selected_genres:
                # Controlliamo la corrispondenza tollerando caratteri speciali
                if sg.lower().replace(' ', '_').replace('&', '_').replace('-', '_').replace('/', '_') == gf[3:].lower():
                    is_present = 1
                    break
            feature_dict[gf] = is_present

        # Creiamo il DataFrame rispettando esattamente l'ordine delle feature del modello
        input_df = pd.DataFrame([feature_dict])
        input_df = input_df[model_features] # Riordina le colonne secondo le specifiche del modello
        
        # 5. Predizione con XGBoost
        # Predizione della classe e delle probabilità
        pred_class_idx = int(xgboost_model.predict(input_df)[0])
        pred_probs = xgboost_model.predict_proba(input_df)[0].tolist()
        
        # Associa le probabilità alle etichette PEGI originali
        probabilities = {}
        for idx, prob in enumerate(pred_probs):
            pegi_label = f"PEGI {pegi_mapping_rev[idx]}"
            probabilities[pegi_label] = round(prob * 100, 1) # percentuale
            
        pegi_prediction = pegi_mapping_rev[pred_class_idx]
        
        return jsonify({
            "success": True,
            "prediction": f"PEGI {pegi_prediction}",
            "pegi_val": pegi_prediction,
            "probabilities": probabilities,
            "palette": color_palette
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Avvio del server Flask sulla porta 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
