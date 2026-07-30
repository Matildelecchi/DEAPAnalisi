import numpy as np
import os

def load_dat_file(path):
    """Carica un file .dat del DEAP (raw o preprocessato)."""
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data
    except Exception as e:
        print(f"Errore nel caricamento di {path}: {e}")
        return None


def check_sampling_rate(data_length):
    """
    Determina il sampling rate in base alla lunghezza del segnale.
    RAW DEAP: 40 trial × 40 canali × 8064 campioni → 8064 campioni per trial
    PREPROCESSED DEAP: 128 Hz → 60s → 7680 campioni
    """
    if data_length % 8064 == 0:
        return "RAW (512 Hz)"
    elif data_length % 7680 == 0:
        return "PREPROCESSED (128 Hz)"
    else:
        return "UNKNOWN"


def check_normalization(data):
    """Controlla se il segnale è normalizzato (range tipico [-1, 1])."""
    min_val = np.min(data)
    max_val = np.max(data)

    if -1.5 < min_val < -0.5 and 0.5 < max_val < 1.5:
        return "Probabile NORMALIZZATO"
    else:
        return "Probabile NON normalizzato"


def check_filtering(data):
    """
    Controlla se il segnale è filtrato.
    RAW DEAP contiene molto rumore ad alta frequenza → valori > 1000 possibili.
    PREPROCESSED DEAP è filtrato → valori molto più piccoli.
    """
    if np.max(np.abs(data)) > 1000:
        return "Probabile NON filtrato (RAW)"
    else:
        return "Probabile filtrato (PREPROCESSED)"


def analyze_file(path):
    print(f"\n=== Analisi file: {path} ===")

    data = load_dat_file(path)
    if data is None:
        return

    print(f"Dimensione totale: {len(data)} campioni")

    # Sampling rate
    sr = check_sampling_rate(len(data))
    print(f"- Sampling rate: {sr}")

    # Normalizzazione
    norm = check_normalization(data)
    print(f"- Normalizzazione: {norm}")

    # Filtraggio
    filt = check_filtering(data)
    print(f"- Filtraggio: {filt}")

    # Conclusione
    if sr.startswith("RAW"):
        print(">>> RISULTATO: Il file è quasi certamente RAW.")
    elif sr.startswith("PREPROCESSED"):
        print(">>> RISULTATO: Il file è quasi certamente PREPROCESSED.")
    else:
        print(">>> RISULTATO: Impossibile determinare con certezza.")


if __name__ == "__main__":
    RAW_DIR = "data/raw/"

    print("=== Verifica preprocessamento DEAP ===")

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".dat"):
            analyze_file(os.path.join(RAW_DIR, filename))
