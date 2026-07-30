import numpy as np
import os
from scipy.signal import butter, filtfilt, iirnotch

RAW_DIR = "data/raw/"
PRE_DIR = "data/preprocessed/"

os.makedirs(PRE_DIR, exist_ok=True)


# ============================================================
# 1. FILTRI
# ============================================================

def bandpass_filter(signal, low=0.5, high=45, fs=128):
    """Applica un filtro bandpass Butterworth."""
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, freq=50, fs=128):
    """Applica filtro notch per rimuovere rumore elettrico."""
    nyq = fs / 2
    b, a = iirnotch(freq / nyq, 30)
    return filtfilt(b, a, signal)


# ============================================================
# 2. RICOSTRUZIONE STRUTTURA DEI FILE KAGGLE
# ============================================================

def reshape_kaggle_dat(data):
    """
    I file Kaggle non sono nel formato DEAP originale.
    Questa funzione prova a ricostruire:
    - 40 trial
    - 40 canali
    - lunghezza dinamica per trial
    """

    total_len = len(data)

    # 40 trial × 40 canali = 1600 segmenti
    n_segments = 40 * 40

    segment_len = total_len // n_segments

    reshaped = data[:segment_len * n_segments].reshape(40, 40, segment_len)

    return reshaped


# ============================================================
# 3. NORMALIZZAZIONE
# ============================================================

def normalize(signal):
    """Normalizza ogni canale nel range [-1, 1]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    return 2 * (signal - min_val) / (max_val - min_val) - 1


# ============================================================
# 4. PREPROCESSING COMPLETO
# ============================================================

def preprocess_subject(path):
    print(f"\n=== Preprocessing: {path} ===")

    raw = np.fromfile(path, dtype=np.float32)

    # Ricostruzione trial × canali × campioni
    data = reshape_kaggle_dat(raw)

    preprocessed = []

    for trial in range(data.shape[0]):
        trial_data = []

        for ch in range(data.shape[1]):
            sig = data[trial, ch]

            # Downsampling a 128 Hz (DEAP ufficiale)
            sig = sig[::4]

            # Filtri
            sig = bandpass_filter(sig)
            sig = notch_filter(sig)

            # Normalizzazione
            sig = normalize(sig)

            trial_data.append(sig)

        preprocessed.append(trial_data)

    preprocessed = np.array(preprocessed)

    # Salvataggio
    subject_id = os.path.basename(path).replace(".dat", "")
    out_path = os.path.join(PRE_DIR, f"{subject_id}_preprocessed.npy")

    np.save(out_path, preprocessed)

    print(f"Salvato: {out_path}")


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("=== Avvio preprocessing DEAP (versione Kaggle) ===")

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".dat"):
            preprocess_subject(os.path.join(RAW_DIR, filename))

    print("\n=== Preprocessing completato ===")
