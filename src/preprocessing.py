"""
src/preprocessing.py
Caricamento e preprocessing dei file DEAP.

I file in data/raw/*.dat NON sono array binari grezzi: sono pickle Python 2
contenenti un dizionario {'data': (40, 40, 8064) float64, 'labels': (40, 4) float64}.
Questo È già il formato ufficiale "data_preprocessed_python" di DEAP:
- 40 trial
- 40 canali (i primi 32 = EEG, gli ultimi 8 = segnali periferici: EOG, EMG, GSR, respirazione,
  temperatura, pletismografia)
- 8064 campioni per trial, a 128 Hz (già filtrato 4-45 Hz e con EOG rimosso dagli autori)
- labels: [valence, arousal, dominance, liking], scala continua 1-9

NB: essendo già filtrato e ricampionato dagli autori del dataset, il preprocessing "vero"
qui si riduce a: caricamento, eventuale selezione canali, segmentazione, normalizzazione.
Il filtraggio aggiuntivo (bandpass/notch) va comunque applicato per coerenza con le
richieste del progetto (punto 3 della pipeline) e può servire a pulire ulteriormente
il segnale, ma NON serve più a rimuovere il rumore di rete grezzo: quello è già stato fatto.
"""

import os
import pickle
import numpy as np

RAW_DIR = "data/raw/"
PRE_DIR = "data/preprocessed/"

FS = 128  # sampling rate dei file preprocessati DEAP
N_EEG_CHANNELS = 32  # canali 0-31 = EEG, 32-39 = periferici

os.makedirs(PRE_DIR, exist_ok=True)


# ============================================================
# 1. CARICAMENTO DI UN SINGOLO SOGGETTO
# ============================================================

def load_deap_subject(path):
    """
    Carica un singolo file .dat DEAP (pickle Python 2).

    Ritorna:
        data   : np.ndarray (40 trial, 40 canali, 8064 campioni)
        labels : np.ndarray (40 trial, 4) -> [valence, arousal, dominance, liking]
    """
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["data"], d["labels"]


# ============================================================
# 2. CARICAMENTO DELL'INTERO DATASET (tutti i soggetti)
# ============================================================

def load_deap_dataset(raw_dir=RAW_DIR, eeg_only=True):
    """
    Carica tutti i soggetti presenti in raw_dir e li concatena.

    IMPORTANTE: viene restituito anche un array subject_ids, perché per dati
    fisiologici uno split train/test casuale sui trial mescola trial dello
    stesso soggetto tra train e test, sovrastimando le performance
    (data leakage a livello di soggetto). Va usato subject_ids per fare uno
    split o una cross-validation raggruppata per soggetto (es. GroupKFold /
    LeaveOneGroupOut), come raccomandato dalle linee guida del progetto.

    Ritorna:
        X : np.ndarray (n_subjects * 40, n_canali, 8064)
        y : np.ndarray (n_subjects * 40, 4)
        subject_ids : np.ndarray (n_subjects * 40,) -> id soggetto per ogni trial
    """
    X_list, y_list, subj_list = [], [], []

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
# 3. SEGMENTAZIONE IN FINESTRE
# ============================================================

def segment_signal(data, segment_length=15, fs=FS, overlap=0.0):
    """
    Divide ogni trial in finestre temporali di segment_length secondi.

    data: (n_trial, n_canali, n_campioni)
    Ritorna: (n_trial * n_segmenti_per_trial, n_canali, segment_length * fs)
    più un array trial_index che indica a quale trial originale appartiene
    ogni segmento (utile per propagare correttamente le label dopo la
    segmentazione).
    """
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

def normalize_signal(data, method="zscore", axis=-1):
    """
    Normalizza il segnale canale per canale (non su tutto l'array insieme:
    farlo su tutto l'array mescola le scale di canali EEG e periferici,
    che hanno unità di misura molto diverse).

    method: 'zscore' (consigliato per feature statistiche/FFT) oppure
            'minmax' (range [-1, 1]).
    """
    if method == "zscore":
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        std[std == 0] = 1e-8
        return (data - mean) / std

    elif method == "minmax":
        min_val = data.min(axis=axis, keepdims=True)
        max_val = data.max(axis=axis, keepdims=True)
        rng = max_val - min_val
        rng[rng == 0] = 1e-8
        return 2 * (data - min_val) / rng - 1

    else:
        raise ValueError(f"Metodo di normalizzazione non riconosciuto: {method}")


# ============================================================
# 5. SALVATAGGIO PREPROCESSATO PER SOGGETTO (facoltativo, per cache su disco)
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


# ============================================================
# 6. MAIN (rigenera la cache preprocessata per tutti i soggetti)
# ============================================================

if __name__ == "__main__":
    print("=== Avvio preprocessing DEAP (formato pickle ufficiale) ===")

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".dat"))
    for filename in files:
        preprocess_and_save_subject(os.path.join(RAW_DIR, filename))

    print("\n=== Preprocessing completato ===")