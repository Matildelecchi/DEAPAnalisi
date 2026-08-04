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
Feature extraction per DEAP (valence e arousal).

Implementa:
- PSD (Welch)
- Bande EEG: theta, alpha, beta, gamma
- Differential Entropy (DE)
- Statistiche temporali: mean, std, skewness, kurtosis

Si interfaccia con src/preprocessing.py:
    X, y, subject_ids     = load_deap_dataset(...)
    X_segments, trial_idx = segment_signal(X, ...)
    X_features, y_valence, y_arousal, subj_seg = extract_features(
        X_segments, y, trial_idx, subject_ids
    )
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

FS = 128

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

EPS = 1e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 1. PSD con Welch
# ============================================================

def compute_psd(signal, fs=FS):
    nperseg = min(fs * 2, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


# ============================================================
# 2. Potenza nelle bande EEG
# ============================================================

def band_power(freqs, psd, band):
    low, high = BANDS[band]
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not idx.any():
        return EPS
    return _trapz(psd[idx], freqs[idx]) + EPS


def log_band_power(freqs, psd, band):
    return np.log(band_power(freqs, psd, band))


# ============================================================
# 3. Differential Entropy (DE)
# ============================================================

def differential_entropy(freqs, psd, band):
    bp = band_power(freqs, psd, band)
    return 0.5 * np.log(2 * np.pi * np.e * bp + EPS)


# ============================================================
# 4. Feature per singolo canale
# ============================================================

def extract_channel_features(signal):
    freqs, psd = compute_psd(signal)

    features = []

    for band in BANDS:
        features.append(log_band_power(freqs, psd, band))
        features.append(differential_entropy(freqs, psd, band))

    # Statistiche temporali robuste
    mean = np.mean(signal)
    std = np.std(signal)

    sk = skew(signal, nan_policy="omit")
    ku = kurtosis(signal, nan_policy="omit")

    # Sostituzione NaN con 0
    if np.isnan(sk): sk = 0.0
    if np.isnan(ku): ku = 0.0

    features.extend([mean, std, sk, ku])

    return np.array(features)


# ============================================================
# 5. Feature per segmento
# ============================================================

def extract_segment_features(segment):
    return np.concatenate([extract_channel_features(ch) for ch in segment])


# ============================================================
# 6. Feature extraction completa
# ============================================================

def extract_features(X_segments, y, trial_idx, subject_ids):
    assert len(subject_ids) == y.shape[0], "subject_ids deve essere per trial"
    assert X_segments.shape[0] == len(trial_idx), "trial_idx deve avere un indice per segmento"

    y_valence = (y[:, 0] >= 5).astype(int)
    y_arousal = (y[:, 1] >= 5).astype(int)

    y_valence_seg = y_valence[trial_idx]
    y_arousal_seg = y_arousal[trial_idx]
    subject_ids_seg = subject_ids[trial_idx]

    X_features = np.array([extract_segment_features(seg) for seg in X_segments])

    return X_features, y_valence_seg, y_arousal_seg, subject_ids_seg

