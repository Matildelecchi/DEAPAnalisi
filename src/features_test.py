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

import numpy as np
from src.preprocessing import load_deap_dataset, segment_signal
from src.features import extract_features, extract_segment_features

print("=== TEST FEATURE EXTRACTION ===")

# 1. Carica dataset
X, y, subject_ids = load_deap_dataset(eeg_only=True)
print("Dataset caricato:", X.shape, y.shape, subject_ids.shape)

# 2. Segmentazione
X_segments, trial_idx = segment_signal(X, segment_length=15, fs=128, overlap=0.0)
print("Segmenti generati:", X_segments.shape)
print("Trial_idx primi 20:", trial_idx[:20])

# 3. Test diagnostico: estrazione singolo segmento
print("\n=== Test diagnostico: estrazione feature segmento per segmento ===")

for i in range(0, min(300, len(X_segments))):
    if i % 50 == 0:
        print(f"Processing segment {i}/{len(X_segments)}")
    _ = extract_segment_features(X_segments[i])

print("\nOK: estrazione singolo segmento funziona.")

# 4. Test completo extract_features
print("\n=== Test extract_features completo ===")

X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
    X_segments, y, trial_idx, subject_ids
)

print("Feature shape:", X_features.shape)
print("Valence seg shape:", y_valence_seg.shape)
print("Arousal seg shape:", y_arousal_seg.shape)
print("Subject seg shape:", subj_seg.shape)

# 5. Test NaN / Inf
print("\nNaN nelle feature:", np.isnan(X_features).sum())
print("Inf nelle feature:", np.isinf(X_features).sum())

# 6. Test coerenza label
i = 0
print("\nSegmento 0:")
print("  trial_idx:", trial_idx[i])
print("  valence originale:", y[trial_idx[i], 0])
print("  valence segmentata:", y_valence_seg[i])
print("  arousal originale:", y[trial_idx[i], 1])
print("  arousal segmentata:", y_arousal_seg[i])

# 7. Test soggetti
print("\nSoggetti unici nei segmenti:", np.unique(subj_seg))
