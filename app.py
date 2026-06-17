import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

import numpy as np

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

init_models()

@app.route('/')
def index():
    return render_template('index.html', tags=model_features)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Il modello di Machine Learning non è caricato sul server."
        }), 500

    try:
        selected_tags = request.form.getlist('tags')
        data = {}
        for feat in model_features:
            data[feat] = 1.0 if feat in selected_tags else 0.0

        X_tags = pd.DataFrame([data], columns=model_features)
        km_dists = km_model.transform(X_tags)
        X_final = np.hstack([X_tags.values, km_dists])
        
        pred_class = int(model.predict(X_final)[0])
        pred_probs = model.predict_proba(X_final)[0].tolist()

        pegi_val = pred_class
        probabilities = {}
        for idx, prob in enumerate(pred_probs):
            pegi_label = f"PEGI {int(model.classes_[idx])}"
            probabilities[pegi_label] = round(prob * 100, 1)

        return jsonify({
            "success": True,
            "prediction": f"PEGI {pegi_val}",
            "pegi_val": pegi_val,
            "probabilities": probabilities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
