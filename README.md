# PEGI Hybrid ML - Classificazione Automatica di Videogiochi

## Descrizione

PEGI Hybrid ML è un'applicazione web basata su **Machine Learning** per la classificazione automatica di videogiochi secondo il sistema PEGI (Pan European Game Information). Il sistema utilizza **tag descrittivi** come input e predice la fascia d'età appropriata utilizzando un ensemble di algoritmi di machine learning.

L'applicazione offre due modalità di classificazione:
- **Classificazione Standard**: Predice direttamente le categorie PEGI (3, 7, 12, 16, 18)
- **Classificazione per Fasce**: Raggruppa in tre macro-categorie (Bambino, Teenager, Adulto)

## Caratteristiche Principali

- **Interfaccia Web Intuitiva**: Applicazione Flask con interfaccia HTML/CSS moderna
- **Modello Ibrido**: Combinazione di Random Forest + KMeans per features avanzate
- **Selezione Tag Dinamica**: Interfaccia interattiva per selezionare tag del gioco
- **Probabilità di Classificazione**: Visualizza le probabilità per ogni classe predetta
- **EDA Completa**: Dataset puliti e analizzati per lo sviluppo del modello
- **Notebook Jupyter**: Pipeline completa di training e validazione

## Struttura del Progetto

```
ML Progetto/
├── app.py                          # App Flask - Classificazione PEGI singola
├── app_fasce.py                    # App Flask - Classificazione per fasce
├── pegi_hybrid_ml.ipynb            # Notebook Jupyter - Training e validazione
├── for_EDA.csv                     # Dataset originale per EDA
├── for_EDA_pulito.csv              # Dataset pulito e preprocessato
│
├── models/                         # Modelli pre-allenati
│   ├── random_forest_model.pkl     # Modello Random Forest principale
│   ├── kmeans_model.pkl            # Modello KMeans per features aggiuntive
│   ├── tag_vocabulary.pkl          # Vocabolario dei tag utilizzati
│   ├── pegi_classes.pkl            # Classi PEGI disponibili
│   └── min_freq.pkl                # Frequenza minima dei tag
│
├── static/                         # Asset statici
│   ├── style.css                   # Stylesheet principale
│   └── pegi/                       # Risorse PEGI
│
└── templates/                      # Template HTML
    ├── index.html                  # Template - Classificazione PEGI singola
    └── index_fasce.html            # Template - Classificazione per fasce
```

## Guida all'Utilizzo

### Prerequisiti

- Python 3.7+
- Flask
- pandas
- scikit-learn
- joblib
- numpy

### Installazione

1. **Clonare o scaricare il progetto**

2. **Installare le dipendenze**
```bash
pip install flask pandas scikit-learn joblib numpy
```

3. **Verificare la presenza dei modelli**
   - Assicurarsi che i file `.pkl` siano presenti nella cartella `models/`

### Avvio dell'Applicazione

#### Modalità Standard (Classificazione PEGI)
```bash
python app.py
```
Accedere a `http://localhost:5000`

#### Modalità Fasce (Classificazione per macro-categorie)
```bash
python app_fasce.py
```
Accedere a `http://localhost:5000`

## Come Funziona il Modello

### Pipeline di Predizione

1. **Input**: L'utente seleziona i tag descrittivi del videogioco (es: "Violenza", "Azione", "Multiplayer")

2. **Feature Engineering**:
   - Conversione dei tag selezionati in vettore binario (1 se presente, 0 altrimenti)
   - Calcolo delle distanze dal modello KMeans per features aggiuntive
   - Concatenazione delle feature per il modello finale

3. **Predizione**:
   - Il Random Forest classifica il gioco nella categoria PEGI appropriata
   - Calcolo delle probabilità per ciascuna classe

4. **Output**: 
   - Categoria PEGI predetta (Modalità standard) o fascia macro (Modalità fasce)
   - Percentuali di probabilità per cada classe

### Architettura del Modello

```
Input (Tag selezionati)
        ↓
   [Feature Engineering]
        ↓
   Tag Features + KMeans Features
        ↓
   [Random Forest Classifier]
        ↓
   Predizione PEGI + Probabilità
```

## Dataset

### for_EDA.csv
- Dataset originale con tutti i dati grezzi
- Utilizzato per l'analisi esplorativa

### for_EDA_pulito.csv
- Dataset pulito e preprocessato
- Rimossi outlier e valori mancanti
- Standardizzate le feature
- Pronto per il training

## Notebook Jupyter

Il file `pegi_hybrid_ml.ipynb` contiene:
- Caricamento e esplorazione dei dati
- Preprocessing e feature engineering
- Training del modello KMeans
- Training del Random Forest
- Validazione incrociata e metriche di valutazione
- Salvataggio dei modelli

## API Endpoints

### GET `/`
Restituisce l'interfaccia web con il modulo di selezione tag.

**Parametri Query:**
- `tags`: Lista di tag del vocabolario modello

### POST `/predict`
Esegue la predizione della categoria PEGI.

**Body (form-data):**
- `tags`: Array di tag selezionati

**Risposta (JSON):**
```json
{
  "success": true,
  "prediction": "PEGI 12",
  "probabilities": {
    "PEGI 3": 5.2,
    "PEGI 7": 15.3,
    "PEGI 12": 68.5,
    "PEGI 16": 8.9,
    "PEGI 18": 2.1
  }
}
```

## Metriche di Performance

Il modello viene valutato su:
- **Accuracy**: Percentuale di predizioni corrette
- **Precision & Recall**: Per ogni categoria PEGI
- **F1-Score**: Media armonica di precisione e recall
- **Cross-Validation**: Validazione incrociata k-fold

## Customizzazione

### Aggiungere Nuovi Tag
1. Modificare il dataset di training
2. Ri-allenare il modello tramite il notebook
3. Salvare il nuovo vocabolario in `models/tag_vocabulary.pkl`

### Modificare le Soglie di Classificazione
In `app_fasce.py`, modificare le condizioni nel metodo `predict()`:
```python
if pred_rounded <= 7:
    categoria_macro = "Bambino"
elif pred_rounded >= 18:
    categoria_macro = "Adulto"
else:
    categoria_macro = "Teenager"
```

## Troubleshooting

### Errore: "Il modello non è caricato"
- Verificare che i file `.pkl` siano presenti in `models/`
- Controllare i permessi di lettura della cartella

### Errore: "Feature non trovata"
- Assicurarsi di selezionare tag dal vocabolario disponibile
- Il tag selezionato potrebbe non essere nel vocabolario del modello

### Porta 5000 già in uso
```bash
python app.py --port 5001
```

## Licenza

Progetto realizzato per scopi didattici.

## Autore

Progetto ML - Classificazione PEGI

---

**Nota**: Il sistema è basato su dati storici di videogiochi. La classificazione PEGI è una guida e non deve essere considerata sostitutiva della valutazione officiale PEGI.
