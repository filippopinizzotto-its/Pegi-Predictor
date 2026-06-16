import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configurazione percorsi modelli
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
RF_MODEL_PATH = 'random_forest_model.pkl'
TAG_VOCAB_PATH = 'tag_vocabulary.pkl'

# Variabili globali per i modelli e metadati
kmeans_model = None
rf_model = None
tag_vocabulary = []
pegi_mapping = {0: 3, 1: 7, 2: 12, 3: 16, 4: 18}

def init_models():
    global kmeans_model, rf_model, tag_vocabulary

    tag_vocabulary = joblib.load(TAG_VOCAB_PATH)
    print(f"[INFO] Tag vocabulary loaded: {len(tag_vocabulary)} tags.")

    if os.path.exists(KMEANS_MODEL_PATH):
        try:
            kmeans_model = joblib.load(KMEANS_MODEL_PATH)
            print(f"[INFO] K-Means model loaded successfully ({kmeans_model.n_clusters} clusters).")
        except Exception as e:
            print(f"[ERROR] Failed to load K-Means model: {e}")
    else:
        print("[WARNING] kmeans_model.pkl not found!")

    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
            print(f"[INFO] Random Forest model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load Random Forest model: {e}")
    else:
        print("[WARNING] random_forest_model.pkl not found!")

init_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None or kmeans_model is None:
        return jsonify({
            "success": False, 
            "error": "I modelli di Machine Learning non sono caricati sul server."
        }), 500

    try:
        selected_tags = request.form.getlist('tags')

        tag_vector = np.zeros(len(tag_vocabulary))
        for tag in selected_tags:
            if tag in tag_vocabulary:
                idx = tag_vocabulary.index(tag)
                tag_vector[idx] = 1

        cluster_dist = kmeans_model.transform([tag_vector])
        X = np.hstack([tag_vector.reshape(1, -1), cluster_dist])

        pred_class = int(rf_model.predict(X)[0])
        pred_probs = rf_model.predict_proba(X)[0].tolist()

        probabilities = {}
        for idx, prob in enumerate(pred_probs):
            pegi_label = f"PEGI {int(rf_model.classes_[idx])}"
            probabilities[pegi_label] = round(prob * 100, 1)

        return jsonify({
            "success": True,
            "prediction": f"PEGI {pred_class}",
            "pegi_val": pred_class,
            "probabilities": probabilities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Avvio del server Flask sulla porta 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
