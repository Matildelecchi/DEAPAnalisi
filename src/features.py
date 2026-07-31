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

NB su subject_ids: extract_features accetta subject_ids A LIVELLO DI TRIAL
(l'output diretto di load_deap_dataset, un id per ciascuno dei 40*n_soggetti
trial) e fa lei stessa il mapping a livello di segmento con trial_idx.
Non passare qui un array già indicizzato a mano (subject_ids[trial_idx]):
verrebbe rimappato una seconda volta con risultati sbagliati. L'assert in
extract_features blocca subito l'errore più comune (passare un array della
lunghezza sbagliata) invece di lasciarlo propagare silenziosamente.

Output:
    X_features : (n_segmenti_totali, n_feature_totali)
    y_valence_seg  : (n_segmenti_totali,)
    y_arousal_seg  : (n_segmenti_totali,)
    subject_ids_seg: (n_segmenti_totali,)
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

FS = 128  # sampling rate DEAP

# Bande EEG (Hz)
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

EPS = 1e-12  # evita log(0) -> -inf / nan che si propagano silenziosamente nei modelli

# NumPy 2.0+ ha rinominato trapz in trapezoid; questo fallback funziona con entrambe le versioni
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 1. PSD con Welch
# ============================================================

def compute_psd(signal, fs=FS):
    """
    Calcola la PSD con Welch per un singolo canale.
    Ritorna:
        freqs : array frequenze
        psd   : array PSD
    """
    # nperseg non può superare la lunghezza del segnale (capita con segmenti
    # corti, es. finestre da poche centinaia di campioni): altrimenti welch
    # solleva un errore o restituisce una PSD inaffidabile
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
    bp = _trapz(psd[idx], freqs[idx])
    return bp + EPS  # stabilità numerica, evita 0 esatto prima dei log a valle


def log_band_power(freqs, psd, band):
    """
    Potenza di banda in scala logaritmica: senza il log, la potenza spettrale
    grezza può variare di diversi ordini di grandezza tra canali/soggetti e
    finisce per dominare le altre feature quando i dati non sono standardizzati
    prima di SVM/KNN/LogReg (sensibili alla scala, a differenza degli alberi).
    """
    return np.log(band_power(freqs, psd, band))


# ============================================================
# 3. Differential Entropy (DE)
# ============================================================

def differential_entropy(freqs, psd, band):
    """
    DE = 0.5 * log(2 * pi * e * sigma^2)
    dove sigma^2 è la potenza nella banda.
    """
    bp = band_power(freqs, psd, band)
    return 0.5 * np.log(2 * np.pi * np.e * bp)


# ============================================================
# 4. Feature per singolo segmento e singolo canale
# ============================================================

def extract_channel_features(signal):
    """
    Estrae tutte le feature per un singolo canale EEG.
    Ritorna un vettore 1D.
    """
    freqs, psd = compute_psd(signal)

    features = []

    # Bande EEG: potenza (in log) + DE
    for band in BANDS:
        lbp = log_band_power(freqs, psd, band)
        de = differential_entropy(freqs, psd, band)
        features.extend([lbp, de])

    # Statistiche temporali
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(skew(signal))
    features.append(kurtosis(signal))

    return np.array(features)


# ============================================================
# 5. Feature per un segmento (tutti i canali)
# ============================================================

def extract_segment_features(segment):
    """
    segment: (n_canali, n_campioni)
    Ritorna: vettore concatenato delle feature di tutti i canali.
    """
    all_features = [extract_channel_features(ch) for ch in segment]
    return np.concatenate(all_features)


# ============================================================
# 6. Feature extraction completa
# ============================================================

def extract_features(X_segments, y, trial_idx, subject_ids):
    """
    X_segments  : (n_segmenti, n_canali, n_campioni)
    y           : (n_trial_totali, 4) -> valence, arousal, dominance, liking
                  (output di preprocessing.load_deap_dataset)
    trial_idx   : (n_segmenti,) indice di trial GLOBALE per ogni segmento
                  (output di preprocessing.segment_signal)
    subject_ids : (n_trial_totali,) id soggetto PER TRIAL
                  (output di preprocessing.load_deap_dataset — NON indicizzarlo
                  già con trial_idx prima di chiamare questa funzione)

    Ritorna:
        X_features
        y_valence_seg
        y_arousal_seg
        subject_ids_seg
    """
    assert len(subject_ids) == y.shape[0], (
        "subject_ids deve avere un elemento per TRIAL (stessa lunghezza di y), "
        "non per segmento: passa l'output grezzo di load_deap_dataset, non "
        "subject_ids[trial_idx]."
    )
    assert X_segments.shape[0] == len(trial_idx), (
        "X_segments e trial_idx devono avere lo stesso numero di segmenti: "
        "controlla di aver passato l'output di segment_signal senza modifiche."
    )

    print("=== Estrazione feature DEAP ===")

    # Binarizzazione valence/arousal come nel paper
    y_valence = (y[:, 0] >= 5).astype(int)
    y_arousal = (y[:, 1] >= 5).astype(int)

    # Propagazione delle label/subject_id ai segmenti tramite l'indice di
    # trial GLOBALE calcolato da segment_signal: è l'unica fonte affidabile
    # di quale trial appartiene a ciascun segmento, evita disallineamenti
    # silenziosi tra segmenti e label.
    y_valence_seg = y_valence[trial_idx]
    y_arousal_seg = y_arousal[trial_idx]
    subject_ids_seg = subject_ids[trial_idx]

    # Estrazione feature
    X_features = np.array([extract_segment_features(seg) for seg in X_segments])

    print("Feature extraction completata.")
    print(f"Shape feature: {X_features.shape}")

    return X_features, y_valence_seg, y_arousal_seg, subject_ids_seg
