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
src/filtering.py

Filtri per segnali EEG DEAP:
- Notch 50 Hz
- Bandpass 4–45 Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

FS = 128


# ============================================================
# 1. Notch 50 Hz
# ============================================================

def notch_filter(signal, freq=50, fs=FS, Q=30):
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, signal)


# ============================================================
# 2. Bandpass 4–45 Hz
# ============================================================

def bandpass_filter(signal, low=4, high=45, fs=FS, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


# ============================================================
# 3. Filtraggio completo per un singolo canale
# ============================================================

def filter_channel(signal):
    x = notch_filter(signal)
    x = bandpass_filter(x)
    return x


# ============================================================
# 4. Filtraggio completo per trial × canali
# ============================================================

def filter_data(data):
    """
    data: (n_trial, n_ch, n_samples)
    """
    filtered = np.zeros_like(data)
    for t in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[t, ch] = filter_channel(data[t, ch])
    return filtered
