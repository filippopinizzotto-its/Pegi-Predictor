"""
app.py – Predittore PEGI basato su Steam Tags
Modello: StackingClassifier (XGBoost + LightGBM + CatBoost + ExtraTrees → LogReg)

Struttura cartella attesa:
  app.py
  stacking_pegi_model.pkl
  model_features.pkl          ← lista dei 200 nomi feature selezionati (sel_names)
  templates/
      index.html
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Mappature PEGI ────────────────────────────────────────────────────────────
# Il modello predice classi 0-4; qui le rimappiamo ai valori PEGI reali
PEGI_INV_MAP = {0: 3, 1: 7, 2: 12, 3: 16, 4: 18}

PEGI_DESCRIPTIONS = {
    3:  "Adatto a tutti – Nessun contenuto inappropriato",
    7:  "Adatto dai 7 anni – Violenza cartoonesca molto leggera",
    12: "Adatto dai 12 anni – Violenza e temi moderatamente intensi",
    16: "Adatto dai 16 anni – Violenza realistica, tematiche forti",
    18: "Solo adulti – Violenza esplicita, contenuti per adulti",
}

PEGI_COLORS = {3: "#2ecc71", 7: "#3498db", 12: "#f39c12", 16: "#e67e22", 18: "#e74c3c"}

# ── Caricamento modello e feature list ───────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "modello_ottimizzato\stacking_pegi_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "modello_ottimizzato\model_features.pkl")

model         = None
model_features = []   # lista ordinata dei 200 nomi feature (es. "Tag_Action", …)


def load_artifacts():
    """
    Carica il modello e la lista delle feature all'avvio dell'app.
    model_features.pkl contiene una lista Python di stringhe come:
      ['Tag_2D', 'Tag_Action', 'Tag_Violent', ...]
    NON è un modello KMeans. Il pipeline di predizione è diretto:
      X_new → stacking_clf.predict_proba(X_new)
    """
    global model, model_features
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Modello non trovato: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"[ERROR] Feature list non trovata: {FEATURES_PATH}")

    model_features = list(joblib.load(FEATURES_PATH))
    model          = joblib.load(MODEL_PATH)

    print(f"[INFO] Feature caricate: {len(model_features)}")
    print(f"[INFO] Modello caricato: {type(model).__name__}")


def tags_to_feature_vector(selected_tags: list) -> pd.DataFrame:
    """
    Converte una lista di tag Steam in un vettore di feature binario.

    La convenzione usata in fase di training è:
        feature_name = "Tag_" + tag.replace(' ', '_').replace('-', '_').replace('/', '_')

    Esempi:
        "Action"          → "Tag_Action"
        "Beat 'em up"     → "Tag_Beat_'em_up"
        "Shoot 'Em Up"    → "Tag_Shoot_'Em_Up"

    Il vettore risultante ha esattamente le colonne in `model_features`,
    inizializzate a 0 e impostate a 1 dove il tag è presente.
    """
    # Inizializza tutto a 0
    feature_dict = {name: 0 for name in model_features}

    for tag in selected_tags:
        tag = str(tag).strip()
        if not tag:
            continue
        # Replica la stessa trasformazione usata durante il training
        feature_name = "Tag_" + tag.replace(' ', '_').replace('-', '_').replace('/', '_')
        if feature_name in feature_dict:
            feature_dict[feature_name] = 1

    return pd.DataFrame([feature_dict], columns=model_features)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Pagina principale con la UI per la selezione dei tag."""
    # Passa la lista dei tag leggibili al template
    tags_readable = []
    for feat in model_features:
        raw   = feat.replace('Tag_', '', 1)          # es. "Action_RPG"
        label = raw.replace('_', ' ')                 # es. "Action RPG"
        tags_readable.append({
            "value":   raw,    # valore inviato via form
            "label":   label,  # etichetta mostrata all'utente
            "feature": feat,   # nome feature completo (Tag_...)
        })

    return render_template('index.html', tags=tags_readable)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint di predizione.
    Accetta sia form-data (dalla UI) che JSON.

    Form-data: campo 'tags' multiplo (lista di stringhe, es. "Action", "Violent")
    JSON body: {"tags": ["Action", "Violent", ...]}

    Risposta JSON:
    {
        "success": true,
        "predicted_pegi": 18,
        "predicted_class": 4,
        "description": "Solo adulti – ...",
        "color": "#e74c3c",
        "probabilities": {"PEGI 3": 1.2, "PEGI 7": 3.4, ...},
        "selected_tags": ["Action", "Violent"],
        "active_features": ["Tag_Action", "Tag_Violent"]
    }
    """
    if model is None:
        return jsonify({"success": False, "error": "Modello non caricato."}), 500

    # ── Leggi i tag dalla richiesta ─────────────────────────────────────────
    content_type = request.content_type or ""
    if "application/json" in content_type:
        body = request.get_json(silent=True) or {}
        selected_tags = body.get("tags", [])
    else:
        # Form-data standard (checkbox HTML)
        selected_tags = request.form.getlist("tags")

    if not selected_tags:
        return jsonify({
            "success": False,
            "error": "Nessun tag selezionato. Seleziona almeno un tag per ottenere una predizione."
        }), 400

    try:
        # ── Costruisci vettore feature ──────────────────────────────────────
        X_new = tags_to_feature_vector(selected_tags)

        # ── Feature attive (debug / trasparenza) ───────────────────────────
        active_features = [
            feat for feat in model_features if X_new[feat].iloc[0] == 1
        ]

        # ── Predizione ──────────────────────────────────────────────────────
        # Il StackingClassifier restituisce probabilità per le 5 classi [0-4]
        proba       = model.predict_proba(X_new)[0]   # array di 5 valori
        pred_class  = int(np.argmax(proba))
        pred_pegi   = PEGI_INV_MAP[pred_class]

        # ── Formatta le probabilità con etichette PEGI reali ────────────────
        probabilities = {}
        for class_idx, prob in enumerate(proba):
            pegi_val = PEGI_INV_MAP[class_idx]
            probabilities[f"PEGI {pegi_val}"] = round(float(prob) * 100, 1)

        return jsonify({
            "success":        True,
            "predicted_pegi": pred_pegi,
            "predicted_class": pred_class,
            "description":    PEGI_DESCRIPTIONS[pred_pegi],
            "color":          PEGI_COLORS[pred_pegi],
            "probabilities":  probabilities,
            "selected_tags":  selected_tags,
            "active_features": active_features,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route('/health')
def health():
    """Endpoint di controllo stato – utile per deployment."""
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "n_features":    len(model_features),
        "model_type":    type(model).__name__ if model else None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
load_artifacts()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)