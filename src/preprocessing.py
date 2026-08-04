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
src/preprocessing.py
Caricamento e preprocessing dei file DEAP.

Funzionalità:
- Caricamento dei file .dat (pickle Python 2)
- Selezione canali EEG
- Filtraggio opzionale (notch + bandpass) tramite src/filtering.py
- Segmentazione in finestre
- Normalizzazione opzionale
"""

import os
import pickle
import numpy as np

# Import ASSOLUTO (compatibile con main.py)
from src.filtering import filter_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PRE_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

FS = 128
N_EEG_CHANNELS = 32

os.makedirs(PRE_DIR, exist_ok=True)


# ============================================================
# 1. CARICAMENTO DI UN SINGOLO SOGGETTO
# ============================================================

def load_deap_subject(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["data"], d["labels"]


# ============================================================
# 2. CARICAMENTO INTERO DATASET
# ============================================================

def load_deap_dataset(raw_dir=RAW_DIR, eeg_only=True):
    X_list, y_list, subj_list = [], [], []

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Cartella del dataset non trovata: {raw_dir}")

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".dat"))
    if not files:
        raise FileNotFoundError(f"Nessun file .dat trovato in {raw_dir}")

    for filename in files:
        subject_id = filename.replace(".dat", "")
        path = os.path.join(raw_dir, filename)
        data, labels = load_deap_subject(path)

        if eeg_only:
            data = data[:, :N_EEG_CHANNELS, :]

        X_list.append(data)
        y_list.append(labels)
        subj_list.extend([subject_id] * data.shape[0])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.array(subj_list)

    return X, y, subject_ids


# ============================================================
# 3. SEGMENTAZIONE
# ============================================================

def segment_signal(data, segment_length=15, fs=FS, overlap=0.0):
    win = int(segment_length * fs)
    step = int(win * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap troppo alto: step <= 0")

    n_trial, n_ch, n_samples = data.shape
    segments = []
    trial_index = []

    for t in range(n_trial):
        start = 0
        while start + win <= n_samples:
            segments.append(data[t, :, start:start + win])
            trial_index.append(t)
            start += step

    return np.array(segments), np.array(trial_index)


# ============================================================
# 4. NORMALIZZAZIONE
# ============================================================

def normalize_signal(data):
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std[std == 0] = 1e-8
    return (data - mean) / std


# ============================================================
# 5. PREPROCESSING COMPLETO (filtri + normalizzazione)
# ============================================================

def full_preprocess(data, apply_filter=True, apply_norm=True):
    if apply_filter:
        data = filter_data(data)

    if apply_norm:
        data = normalize_signal(data)

    return data


# ============================================================
# 6. SALVATAGGIO PREPROCESSATO
# ============================================================

def preprocess_and_save_subject(path, out_dir=PRE_DIR, eeg_only=True):
    data, labels = load_deap_subject(path)
    if eeg_only:
        data = data[:, :N_EEG_CHANNELS, :]

    subject_id = os.path.basename(path).replace(".dat", "")
    out_path = os.path.join(out_dir, f"{subject_id}_preprocessed.npz")
    np.savez(out_path, data=data, labels=labels)
    print(f"Salvato: {out_path}")
    return out_path


if __name__ == "__main__":
    print("=== Avvio preprocessing DEAP ===")

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".dat"))
    for filename in files:
        preprocess_and_save_subject(os.path.join(RAW_DIR, filename))

    print("=== Preprocessing completato ===")
