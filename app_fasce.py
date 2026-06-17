import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configurazione percorsi modelli
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
KMEANS_PATH = os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'tag_vocabulary.pkl')

model = None
km_model = None
model_features = []

def init_models():
    global model, km_model, model_features
    model_features = list(joblib.load(FEATURES_PATH))
    print(f"[INFO] Feature list loaded: {len(model_features)} features.")
    km_model = joblib.load(KMEANS_PATH)
    print(f"[INFO] KMeans model loaded successfully.")
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded successfully.")

# Inizializzazione modelli all'avvio
init_models()

@app.route('/')
def index():
    return render_template('index_fasce.html', tags=model_features)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Il modello di Machine Learning non è caricato sul server."
        }), 500

    try:
        # Estrazione tag dal form
        selected_tags = request.form.getlist('tags')
        data = {feat: 1.0 if feat in selected_tags else 0.0 for feat in model_features}

        # Preparazione feature per il modello
        X_tags = pd.DataFrame([data], columns=model_features)
        km_dists = km_model.transform(X_tags)
        X_final = np.hstack([X_tags.values, km_dists])
        
        # 1. Predizione continua dal modello e arrotondamento
        pred_raw = model.predict(X_final)[0]
        pred_rounded = round(pred_raw)

        # 2. Mappatura in macro-categorie
        if pred_rounded <= 7:
            categoria_macro = "Bambino"
        elif pred_rounded >= 18:
            categoria_macro = "Adulto"
        else:
            categoria_macro = "Teenager"

        # 3. Calcolo probabilità aggregate per le 3 macro-categorie (usando il massimo per ciascuna fascia per non falsare le probabilità)
        pred_probs = model.predict_proba(X_final)[0]
        classes = model.classes_
        
        probs_by_class = {int(classes[idx]): prob for idx, prob in enumerate(pred_probs)}
        
        prob_bambino = max(probs_by_class.get(3, 0.0), probs_by_class.get(7, 0.0))
        prob_teenager = max(probs_by_class.get(12, 0.0), probs_by_class.get(16, 0.0))
        prob_adulto = probs_by_class.get(18, 0.0)
        
        total_prob = prob_bambino + prob_teenager + prob_adulto
        if total_prob > 0:
            probs_macro = {
                "Bambino": prob_bambino / total_prob,
                "Teenager": prob_teenager / total_prob,
                "Adulto": prob_adulto / total_prob
            }
        else:
            probs_macro = {"Bambino": 0.0, "Teenager": 0.0, "Adulto": 0.0}
        
        # Trasformiamo in percentuali arrotondate per l'interfaccia grafica
        probabilities = {k: round(v * 100, 1) for k, v in probs_macro.items()}

        return jsonify({
            "success": True,
            "prediction": categoria_macro,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)