"""
===============================================================
    Machine Learning Emotion Recognition (2020)
    
    Autori:
        - Lecchi Matilde (759875)
        - Pellegrini Gaia (759909)
        - Caredda Anna Eleonora (762576)

    Anno Accademico: 2025/2026
    Corso: Interfacce Uomo-Macchina

    Descrizione:
        Questo file fa parte del progetto basato sul paper
        "Machine Learning Emotion Recognition (2020)", dedicato
        all'analisi delle emozioni tramite segnali EEG attraverso
        tecniche di preprocessing, estrazione di feature e modelli
        di machine learning.
===============================================================
"""

"""
src/utils.py
Funzioni di utilità generiche + funzioni specifiche DEAP.

Include:
- gestione seed
- creazione directory
- salvataggio JSON
- binarizzazione label DEAP
- salvataggio/caricamento modelli ML (pickle)
"""

import os
import json
import pickle
import numpy as np
import random

# ============================
# 1. RIPRODUCIBILITÀ
# ============================

def set_seed(seed=42):
    """
    Imposta il seed globale per numpy e random.
    Garantisce riproducibilità dei risultati.
    """
    np.random.seed(seed)
    random.seed(seed)


# ============================
# 2. DIRECTORY & FILE MANAGEMENT
# ============================

def ensure_dir(path):
    """
    Crea la directory se non esiste.
    Utile per 'results/', 'models/', ecc.
    """
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_json(data, filename):
    """
    Salva un dizionario in formato JSON.
    """
    ensure_dir(os.path.dirname(filename))
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ============================
# 3. LABEL DEAP (binarizzazione)
# ============================

# Colonne di y come restituite da load_deap_dataset:
# [valence, arousal, dominance, liking]
LABEL_COLUMNS = {
    "valence": 0,
    "arousal": 1,
    "dominance": 2,
    "liking": 3
}

def binarize_labels(y, dimension="valence", threshold=5.0):
    """
    Converte i rating continui (1-9) in classi binarie (alto/basso),
    come nel paper originale DEAP.
    dimension: 'valence' | 'arousal' | 'dominance' | 'liking'
    Ritorna: array di 0/1 (1 = sopra soglia)
    """
    col = LABEL_COLUMNS[dimension]
    return (y[:, col] > threshold).astype(int)


# ============================
# 4. MODEL MANAGEMENT (pickle)
# ============================

def save_model_pickle(model, path):
    """
    Salva un modello ML in formato pickle.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modello salvato in: {path}")


def load_model_pickle(path):
    """
    Carica un modello ML salvato in pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
