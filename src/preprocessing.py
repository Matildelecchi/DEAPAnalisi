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

Questo modulo contiene le funzioni utilizzate per:
- caricare i dati del dataset DEAP;
- selezionare solamente i canali EEG;
- segmentare i segnali in finestre temporali;
- normalizzare i segnali;
- applicare il preprocessing completo;
- salvare i dati preprocessati.
"""

# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# os viene utilizzato per gestire percorsi, file e directory.
import os

# pickle permette di leggere i file .dat del dataset DEAP,
# che sono stati salvati utilizzando il formato pickle.
import pickle

# NumPy viene utilizzato per gestire gli array contenenti
# i segnali EEG e per effettuare operazioni numeriche.
import numpy as np


# Importa la funzione che applica il filtraggio ai segnali EEG.
#
# La funzione filter_data() viene definita nel modulo
# src/filtering.py e applica i filtri previsti dalla pipeline.
from src.filtering import filter_data


# ============================================================
# DIRECTORY E PARAMETRI GENERALI
# ============================================================

# Determina il percorso della directory principale del progetto.
#
# __file__ rappresenta il percorso del file preprocessing.py.
# abspath() restituisce il percorso assoluto.
# dirname() viene utilizzato per risalire alle directory superiori.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Directory contenente i file RAW del dataset DEAP.
RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)


# Directory destinata ai dati preprocessati.
PRE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "preprocessed"
)


# Frequenza di campionamento utilizzata nella pipeline.
#
# I dati DEAP preprocessati utilizzati dal progetto
# hanno una frequenza di campionamento di 128 Hz.
FS = 128


# Numero di canali EEG presenti nel dataset DEAP.
#
# Il dataset contiene 32 canali EEG, mentre gli altri canali
# presenti nei dati originali appartengono ad altre tipologie
# di segnali fisiologici.
N_EEG_CHANNELS = 32


# Crea la directory dei dati preprocessati se non esiste.
#
# exist_ok=True evita che venga generato un errore nel caso
# in cui la directory sia già presente.
os.makedirs(
    PRE_DIR,
    exist_ok=True
)


# ============================================================
# 1. CARICAMENTO DI UN SINGOLO SOGGETTO
# ============================================================

def load_deap_subject(path):
    """
    Carica i dati relativi a un singolo soggetto del dataset DEAP.

    Parametri:
        path:
            percorso del file .dat del soggetto.

    Restituisce:
        data:
            segnali EEG e altri segnali fisiologici presenti
            nel file.

        labels:
            etichette associate ai 40 trial del soggetto.
    """

    # Apre il file in modalità binaria.
    #
    # I file DEAP sono memorizzati in formato pickle,
    # quindi devono essere aperti con modalità "rb".
    with open(path, "rb") as f:

        # Carica il contenuto del file pickle.
        #
        # encoding="latin1" permette di leggere correttamente
        # file pickle creati originariamente con Python 2.
        d = pickle.load(
            f,
            encoding="latin1"
        )

    # Il dizionario DEAP contiene la chiave "data",
    # che contiene i segnali registrati.
    #
    # Restituisce inoltre la chiave "labels", contenente
    # le valutazioni associate ai trial.
    return d["data"], d["labels"]


# ============================================================
# 2. CARICAMENTO DELL'INTERO DATASET
# ============================================================

def load_deap_dataset(raw_dir=RAW_DIR, eeg_only=True):
    """
    Carica tutti i soggetti presenti nella directory DEAP.

    Parametri:
        raw_dir:
            directory contenente i file .dat.

        eeg_only:
            se True vengono mantenuti solamente i 32 canali EEG.

    Restituisce:
        X:
            array contenente i dati di tutti i soggetti.

        y:
            array contenente le label dei trial.

        subject_ids:
            identificativo del soggetto associato a ogni trial.
    """

    # Liste temporanee utilizzate per raccogliere
    # progressivamente i dati dei vari soggetti.
    X_list, y_list, subj_list = [], [], []


    # Controlla che la directory specificata esista.
    #
    # Se la cartella non viene trovata, viene sollevato
    # un errore FileNotFoundError.
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Cartella del dataset non trovata: {raw_dir}"
        )


    # Recupera tutti i file con estensione .dat presenti
    # nella directory del dataset.
    #
    # sorted() garantisce che i file vengano elaborati
    # in ordine alfabetico.
    files = sorted(
        f for f in os.listdir(raw_dir)
        if f.endswith(".dat")
    )


    # Controlla che sia stato trovato almeno un file .dat.
    if not files:
        raise FileNotFoundError(
            f"Nessun file .dat trovato in {raw_dir}"
        )


    # Ciclo attraverso tutti i file dei soggetti.
    for filename in files:

        # Ricava l'identificativo del soggetto dal nome del file.
        #
        # Ad esempio:
        # "s01.dat" -> "s01"
        subject_id = filename.replace(
            ".dat",
            ""
        )


        # Costruisce il percorso completo del file.
        path = os.path.join(
            raw_dir,
            filename
        )


        # Carica dati e label del soggetto corrente.
        data, labels = load_deap_subject(path)


        # Se eeg_only è True, mantiene solamente
        # i primi 32 canali, corrispondenti ai canali EEG.
        if eeg_only:
            data = data[:, :N_EEG_CHANNELS, :]


        # Aggiunge i dati del soggetto alla lista generale.
        X_list.append(data)


        # Aggiunge le label del soggetto alla lista generale.
        y_list.append(labels)


        # Crea un identificativo del soggetto per ogni trial.
        #
        # data.shape[0] rappresenta il numero di trial
        # appartenenti al soggetto corrente.
        #
        # In questo modo ogni trial viene associato
        # al soggetto corretto.
        subj_list.extend(
            [subject_id] * data.shape[0]
        )


    # Unisce i dati di tutti i soggetti lungo la dimensione
    # dei trial.
    X = np.concatenate(
        X_list,
        axis=0
    )


    # Unisce le label di tutti i soggetti.
    y = np.concatenate(
        y_list,
        axis=0
    )


    # Converte la lista degli identificativi dei soggetti
    # in un array NumPy.
    subject_ids = np.array(
        subj_list
    )


    # Restituisce i tre elementi principali utilizzati
    # dalla pipeline:
    # - segnali;
    # - label;
    # - identificativi dei soggetti.
    return X, y, subject_ids


# ============================================================
# 3. SEGMENTAZIONE
# ============================================================

def segment_signal(
    data,
    segment_length=15,
    fs=FS,
    overlap=0.0
):
    """
    Divide ogni trial EEG in finestre temporali più piccole.

    Parametri:
        data:
            array con forma
            (numero_trial, numero_canali, numero_campioni).

        segment_length:
            durata di ogni segmento in secondi.

        fs:
            frequenza di campionamento del segnale.

        overlap:
            percentuale di sovrapposizione tra segmenti.
            0.0 significa nessuna sovrapposizione.

    Restituisce:
        segments:
            array contenente tutti i segmenti generati.

        trial_index:
            indice del trial originale a cui appartiene
            ogni segmento.
    """

    # Converte la durata del segmento da secondi a campioni.
    #
    # Ad esempio:
    # 15 secondi × 128 Hz = 1920 campioni.
    win = int(
        segment_length * fs
    )


    # Calcola lo spostamento temporale tra l'inizio
    # di un segmento e quello successivo.
    #
    # Se overlap = 0:
    # step = win
    #
    # Se overlap = 0.5:
    # step = win * 0.5
    step = int(
        win * (1 - overlap)
    )


    # Controlla che lo spostamento sia valido.
    #
    # Se lo step fosse 0 o negativo, il ciclo di segmentazione
    # non potrebbe procedere correttamente.
    if step <= 0:
        raise ValueError(
            "overlap troppo alto: step <= 0"
        )


    # Recupera le dimensioni del dataset.
    #
    # n_trial:
    # numero di trial.
    #
    # n_ch:
    # numero di canali.
    #
    # n_samples:
    # numero di campioni per trial.
    n_trial, n_ch, n_samples = data.shape


    # Lista che conterrà tutti i segmenti generati.
    segments = []


    # Lista che permetterà di mantenere il collegamento
    # tra ogni segmento e il trial originale.
    trial_index = []


    # Scorre tutti i trial del dataset.
    for t in range(n_trial):

        # Posizione iniziale della prima finestra.
        start = 0


        # Continua a creare segmenti finché rimane
        # una finestra completa all'interno del trial.
        while start + win <= n_samples:

            # Estrae una finestra temporale del trial corrente.
            #
            # Vengono mantenuti:
            # - tutti i canali;
            # - i campioni compresi tra start e start + win.
            segments.append(
                data[
                    t,
                    :,
                    start:start + win
                ]
            )


            # Memorizza l'indice del trial originale.
            #
            # Questo sarà necessario successivamente
            # per associare le label corrette ai segmenti.
            trial_index.append(t)


            # Sposta la finestra in avanti dello step calcolato.
            start += step


    # Converte la lista dei segmenti in un array NumPy.
    return (
        np.array(segments),
        np.array(trial_index)
    )


# ============================================================
# 4. NORMALIZZAZIONE
# ============================================================

def normalize_signal(data):
    """
    Normalizza il segnale lungo la dimensione temporale.

    La normalizzazione viene effettuata separatamente
    per ogni segmento/canale.

    La trasformazione utilizzata è:

        (x - media) / deviazione_standard
    """

    # Calcola la media lungo l'ultima dimensione,
    # cioè lungo il tempo.
    #
    # keepdims=True mantiene la dimensione necessaria
    # per poter effettuare successivamente la sottrazione
    # tramite broadcasting.
    mean = data.mean(
        axis=-1,
        keepdims=True
    )


    # Calcola la deviazione standard lungo la dimensione temporale.
    std = data.std(
        axis=-1,
        keepdims=True
    )


    # Evita una divisione per zero nei casi in cui
    # un segnale abbia deviazione standard uguale a zero.
    #
    # In questi casi viene utilizzato un valore molto piccolo.
    std[std == 0] = 1e-8


    # Applica la normalizzazione z-score:
    #
    # valore normalizzato =
    # (valore originale - media) / deviazione standard
    return (
        data - mean
    ) / std


# ============================================================
# 5. PREPROCESSING COMPLETO
# ============================================================

def full_preprocess(
    data,
    apply_filter=True,
    apply_norm=True
):
    """
    Esegue il preprocessing completo del segnale.

    Il preprocessing può comprendere:
    1. filtraggio;
    2. normalizzazione.

    I due passaggi possono essere attivati o disattivati
    tramite i parametri della funzione.
    """

    # Se apply_filter è True, applica il filtraggio
    # notch + bandpass definito in filtering.py.
    if apply_filter:
        data = filter_data(data)


    # Se apply_norm è True, applica la normalizzazione
    # tramite z-score.
    if apply_norm:
        data = normalize_signal(data)


    # Restituisce il segnale preprocessato.
    return data


# ============================================================
# 6. SALVATAGGIO DEI DATI PREPROCESSATI
# ============================================================

def preprocess_and_save_subject(
    path,
    out_dir=PRE_DIR,
    eeg_only=True
):
    """
    Preprocessa e salva i dati di un singolo soggetto.

    Parametri:
        path:
            percorso del file .dat originale.

        out_dir:
            directory nella quale salvare il file preprocessato.

        eeg_only:
            se True vengono mantenuti solamente i 32 canali EEG.

    Restituisce:
        Il percorso del file preprocessato salvato.
    """

    # Carica dati e label del soggetto.
    data, labels = load_deap_subject(path)


    # Se richiesto, mantiene solamente i primi 32 canali EEG.
    if eeg_only:
        data = data[:, :N_EEG_CHANNELS, :]


    # Ricava l'identificativo del soggetto dal nome del file.
    #
    # Esempio:
    # "s01.dat" -> "s01"
    subject_id = os.path.basename(path).replace(
        ".dat",
        ""
    )


    # Costruisce il nome del file di output.
    #
    # Esempio:
    # s01.dat -> s01_preprocessed.npz
    out_path = os.path.join(
        out_dir,
        f"{subject_id}_preprocessed.npz"
    )


    # Salva dati e label in formato NumPy .npz.
    np.savez(
        out_path,
        data=data,
        labels=labels
    )


    # Comunica nel terminale il percorso del file salvato.
    print(
        f"Salvato: {out_path}"
    )


    # Restituisce il percorso del file creato.
    return out_path


# ============================================================
# MAIN
# ============================================================

# Questa condizione fa sì che il codice sottostante
# venga eseguito solamente quando preprocessing.py
# viene avviato direttamente.
#
# Se il modulo viene importato da un altro file,
# questa parte non viene eseguita.
if __name__ == "__main__":

    # Messaggio iniziale visualizzato nel terminale.
    print(
        "=== Avvio preprocessing DEAP ==="
    )


    # Recupera tutti i file .dat presenti nella directory RAW.
    #
    # sorted() permette di elaborarli in ordine.
    files = sorted(
        f
        for f in os.listdir(RAW_DIR)
        if f.endswith(".dat")
    )


    # Scorre tutti i file del dataset.
    for filename in files:

        # Costruisce il percorso completo del file.
        path = os.path.join(
            RAW_DIR,
            filename
        )


        # Carica e salva i dati del soggetto corrente.
        preprocess_and_save_subject(
            path
        )


    # Messaggio finale che indica il completamento
    # dell'elaborazione.
    print(
        "=== Preprocessing completato ==="
    )