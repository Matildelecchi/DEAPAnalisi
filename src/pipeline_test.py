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
src/pipeline_test.py
Test end-to-end della pipeline DEAP:
- Caricamento
- Filtraggio
- Segmentazione
- Feature extraction
- Modelli (solo LogReg per velocità)
"""

import numpy as np

from preprocessing import load_deap_dataset, segment_signal
from filtering import filter_data
from features import extract_features
from models import evaluate_models


print("\n==============================")
print("   TEST PIPELINE COMPLETA")
print("==============================\n")


# ============================================================
# 1. CARICAMENTO DATI
# ============================================================

print(">>> 1. Caricamento dataset")
X, y, subject_ids = load_deap_dataset(eeg_only=True)

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("Shape subject_ids:", subject_ids.shape)

assert X.ndim == 3, "X deve essere (trial, canali, campioni)"
assert y.ndim == 2, "y deve essere (trial, 4)"
assert len(subject_ids) == X.shape[0], "subject_ids deve avere un id per trial"

print("OK: caricamento dati\n")


# ============================================================
# 2. FILTRAGGIO
# ============================================================

print(">>> 2. Filtraggio EEG (notch + bandpass)")
X_filtered = filter_data(X)

print("Shape X_filtered:", X_filtered.shape)
print("NaN:", np.isnan(X_filtered).sum())
print("Inf:", np.isinf(X_filtered).sum())

assert np.isnan(X_filtered).sum() == 0, "Ci sono NaN dopo filtraggio"
assert np.isinf(X_filtered).sum() == 0, "Ci sono Inf dopo filtraggio"

print("OK: filtraggio\n")


# ============================================================
# 3. SEGMENTAZIONE
# ============================================================

print(">>> 3. Segmentazione")
X_segments, trial_idx = segment_signal(X_filtered, segment_length=15, fs=128)

print("Shape X_segments:", X_segments.shape)
print("Shape trial_idx:", trial_idx.shape)

assert X_segments.ndim == 3, "Segmenti devono essere (segmenti, canali, campioni)"
assert len(trial_idx) == X_segments.shape[0], "trial_idx deve avere un indice per segmento"

print("OK: segmentazione\n")


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

print(">>> 4. Feature extraction")
X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
    X_segments, y, trial_idx, subject_ids
)

print("Shape X_features:", X_features.shape)
print("NaN nelle feature:", np.isnan(X_features).sum())
print("Inf nelle feature:", np.isinf(X_features).sum())

assert np.isnan(X_features).sum() == 0, "Ci sono NaN nelle feature"
assert np.isinf(X_features).sum() == 0, "Ci sono Inf nelle feature"

print("OK: feature extraction\n")


# ============================================================
# 5. MODELLI (solo LogReg per velocità)
# ============================================================

print(">>> 5. Test modelli (solo LogReg per velocità)")
results_valence = evaluate_models(
    X_features, y_valence_seg, subj_seg, n_splits=5, task_name="valence"
)

print("Risultati valence:", results_valence)

print("\n>>> Pipeline COMPLETA FUNZIONANTE ✔")
