"""
===============================================================
    Machine Learning Emotion Recognition (2020)

    Autori:
        - Lecchi Matilde 759875
        - Pellegrini Gaia 759909
        - Caredda Anna Eleonora 762576

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
Feature extraction per il dataset DEAP.

Questo modulo si occupa di trasformare i segnali EEG in un insieme
di feature numeriche che possono essere utilizzate successivamente
dai modelli di Machine Learning per classificare le emozioni.

Le feature estratte comprendono:

- Power Spectral Density (PSD) tramite il metodo di Welch;
- potenza nelle bande EEG theta, alpha, beta e gamma;
- logaritmo della potenza nelle bande;
- Differential Entropy (DE);
- statistiche temporali:
    - media;
    - deviazione standard;
    - skewness;
    - kurtosis.

Il modulo lavora in combinazione con src/preprocessing.py:

    X, y, subject_ids = load_deap_dataset(...)

    X_segments, trial_idx = segment_signal(X, ...)

    X_features, y_valence, y_arousal, subj_seg = extract_features(
        X_segments, y, trial_idx, subject_ids
    )
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

import numpy as np

#Calcolare la Power Spectral Density del segnale EEG
from scipy.signal import welch

# calcolare due statistiche descrittive della distribuzione dei valori del segnale
from scipy.stats import skew, kurtosis


# ============================================================
# PARAMETRI GENERALI
# ============================================================

# Frequenza di campionamento del segnale EEG
FS = 128

# Definizione delle principali bande di frequenza EEG.
# Ogni banda è rappresentata da: nome: (frequenza_minima, frequenza_massima)
#
# Le bande considerate sono:
# - theta: 4-8 Hz
# - alpha: 8-13 Hz
# - beta: 13-30 Hz
# - gamma: 30-45 Hz
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# Piccolo valore positivo utilizzato per evitare problemi numerici quando 
# si calcolano logaritmi o valori molto vicini allo zero
EPS = 1e-12

#Il codice cerca quindi prima trapezoid e, se non disponibile, utilizza trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 1. POWER SPECTRAL DENSITY (PSD) CON WELCH
# ============================================================

def compute_psd(signal, fs=FS):
    """
    Calcola la Power Spectral Density (PSD) di un segnale.
    La PSD descrive come la potenza del segnale è distribuita nelle diverse frequenze.
    Viene utilizzato il metodo di Welch, che divide il segnale
    in più finestre e calcola una stima mediata dello spettro.

    Parametri:
        signal: segnale EEG di un singolo canale.
        fs: frequenza di campionamento del segnale. Per default è 128 Hz.

    Restituisce:
        freqs: frequenze considerate.
        psd: potenza associata alle diverse frequenze.
    """

    # Imposta la lunghezza della finestra utilizzata da Welch
    # fs * 2 corrisponde a 2 secondi di segnale min() evita di utilizzare una finestra più lunga del segnale disponibile
    nperseg = min(
        fs * 2,
        len(signal)
    )

    # Calcola la Power Spectral Density utilizzando il metodo di Welch
    # freqs contiene le frequenze
    # psd contiene la potenza associata a ciascuna frequenza
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg
    )
    
    # Restituisce frequenze e PSD
    return freqs, psd


# ============================================================
# 2. POTENZA NELLE BANDE EEG
# ============================================================

def band_power(freqs, psd, band):
    """
    Calcola la potenza del segnale all'interno di una specifica banda di frequenza EEG.
    La potenza viene calcolata integrando la PSD nell'intervallo di frequenze corrispondente alla banda.

    Parametri:
        freqs: array delle frequenze.
        psd: Power Spectral Density.
        band: nome della banda EEG da analizzare.

    Restituisce:
        La potenza della banda considerata.
    """

    # Recupera i limiti inferiore e superiore della banda
    low, high = BANDS[band]

    # Crea una maschera booleana che seleziona solamente le frequenze comprese nell'intervallo della banda
    idx = np.logical_and(
        freqs >= low,
        freqs <= high
    )

    # Controlla se esiste almeno una frequenza appartenente alla banda
    # Se non viene trovata nessuna frequenza, restituisce EPS per evitare problemi nei calcoli successivi
    if not idx.any():
        return EPS

    # Calcola la potenza della banda integrando la PSD rispetto alla frequenza
    # _trapz rappresenta np.trapezoid oppure np.trapz, a seconda della versione di NumPy disponibile
    # EPS viene aggiunto per evitare che il risultato sia esattamente zero
    return _trapz(
        psd[idx],
        freqs[idx]
    ) + EPS


# ============================================================
# LOGARITMO DELLA POTENZA DI BANDA
# ============================================================

def log_band_power(freqs, psd, band):
    """
    Calcola il logaritmo naturale della potenza contenuta in una determinata banda EEG.

    Il logaritmo viene utilizzato per comprimere la scala dei valori della potenza e 
    rendere le feature più gestibili dai modelli di Machine Learning.
    """

    # Calcola prima la potenza della banda e il suo logaritmo naturale
    return np.log(
        band_power(freqs, psd, band)
    )


# ============================================================
# 3. DIFFERENTIAL ENTROPY (DE)
# ============================================================

def differential_entropy(freqs, psd, band):
    """
    Calcola la Differential Entropy (DE) associata a una determinata banda EEG.

    Nel codice viene utilizzata la relazione:

        DE = 0.5 * log(2 * pi * e * band_power)

    dove band_power rappresenta la potenza della banda.

    La Differential Entropy viene utilizzata come feature
    per descrivere la distribuzione del segnale EEG.
    """

    bp = band_power(
        freqs,
        psd,
        band
    )
    return 0.5 * np.log(
        2 * np.pi * np.e * bp + EPS
    )


# ============================================================
# 4. ESTRAZIONE DELLE FEATURE DI UN SINGOLO CANALE
# ============================================================

def extract_channel_features(signal):
    """
    Estrae tutte le feature da un singolo canale EEG.

    Per ogni canale vengono calcolate:

    Per ciascuna banda EEG:
        - log della potenza;
        - Differential Entropy.

    Inoltre vengono calcolate quattro statistiche temporali:
        - media;
        - deviazione standard;
        - skewness;
        - kurtosis.

    Restituisce un array contenente tutte le feature del canale.
    """

    # Calcola lo spettro del segnale tramite Welch
    freqs, psd = compute_psd(signal)

    # Lista vuota nella quale verranno raccolte tutte le feature del canale
    features = []


    # ========================================================
    # FEATURE FREQUENZIALI
    # ========================================================

    # Scorre tutte le bande EEG definite precedentemente: theta, alpha, beta e gamma
    for band in BANDS:

        # Calcola il logaritmo della potenza della banda e aggiunge il risultato alla lista delle feature
        features.append(
            log_band_power(
                freqs,
                psd,
                band
            )
        )

        # Calcola la Differential Entropy della stessa banda e la aggiunge alle feature
        features.append(
            differential_entropy(
                freqs,
                psd,
                band
            )
        )


    # ========================================================
    # STATISTICHE TEMPORALI
    # ========================================================

    # Calcola la media dei campioni del segnale.ì
    mean = np.mean(signal)

    # Calcola la deviazione standard del segnale
    # Indica quanto i valori si discostano dalla loro media
    std = np.std(signal)

    # Calcola la misura dell'asimmetria della distribuzione dei valori del segnale
    # nan_policy="omit" indica di ignorare eventuali valori NaN
    sk = skew(
        signal,
        nan_policy="omit"
    )

    # Calcola la forma della distribuzione dei valori
    ku = kurtosis(
        signal,
        nan_policy="omit"
    )


    # ========================================================
    # GESTIONE DI EVENTUALI NaN
    # ========================================================

    # Se la skewness restituisce NaN, viene sostituita con il valore 0
    # Questo evita di introdurre valori NaN nella matrice finale delle feature
    if np.isnan(sk):
        sk = 0.0

    # Se la kurtosis restituisce NaN, viene anch'essa sostituita con 0
    if np.isnan(ku):
        ku = 0.0

    # Aggiunge alle feature le quattro statistiche temporali:
    # - media
    # - deviazione standard
    # - skewness
    # - kurtosis
    features.extend([
        mean,
        std,
        sk,
        ku
    ])

    # Converte la lista delle feature in un array NumPy
    return np.array(features)


# ============================================================
# 5. ESTRAZIONE DELLE FEATURE DI UN SEGMENTO
# ============================================================

def extract_segment_features(segment):
    """
    Estrae le feature da tutti i canali contenuti
    in un singolo segmento EEG.

    Il segmento contiene più canali EEG.
    Per ogni canale viene chiamata extract_channel_features().

    Le feature dei diversi canali vengono poi concatenate
    in un unico vettore.
    """

    # Scorre tutti i canali del segmento.
    # Per ogni canale:
    #   1. estrae le relative feature
    #   2. restituisce un vettore di feature
    #
    # np.concatenate() unisce tutti questi vettori in un unico vettore rappresentativo dell'intero segmento
    return np.concatenate([
        extract_channel_features(ch)
        for ch in segment
    ])


# ============================================================
# 6. ESTRAZIONE COMPLETA DELLE FEATURE
# ============================================================

def extract_features(
    X_segments,
    y,
    trial_idx,
    subject_ids
):
    """
    Esegue l'estrazione completa delle feature per tutti
    i segmenti EEG.

    Oltre alle feature, la funzione assegna ad ogni segmento:
    - la classe di valence;
    - la classe di arousal;
    - l'identificativo del soggetto.

    Le label originali del DEAP vengono trasformate
    in classificazione binaria utilizzando la soglia 5:

        valore >= 5 -> classe 1
        valore < 5  -> classe 0
    """


    # ========================================================
    # CONTROLLO DELLA COERENZA DEI DATI
    # ========================================================

    # Verifica che ci sia un subject ID per ogni trial
    # y.shape[0] rappresenta il numero di trial
    assert len(subject_ids) == y.shape[0], \
        "subject_ids deve essere per trial"

    # Verifica che esista un trial_idx per ogni segmento
    # Ogni segmento deve infatti essere associato al trial originale dal quale proviene
    assert X_segments.shape[0] == len(trial_idx), \
        "trial_idx deve avere un indice per segmento"


    # ========================================================
    # CONVERSIONE DELLE LABEL IN CLASSI BINARIE
    # ========================================================

    # Trasforma i valori originali di valence in due classi:
    # - valence >= 5 -> 1
    # - valence < 5  -> 0
    # y[:, 0] indica la colonna contenente la valence
    y_valence = (
        y[:, 0] >= 5
    ).astype(int)

    # Trasforma allo stesso modo i valori di arousal:
    # - arousal >= 5 -> 1
    # - arousal < 5  -> 0
    # y[:, 1] indica la colonna contenente l'arousal
    y_arousal = (
        y[:, 1] >= 5
    ).astype(int)


    # ========================================================
    # PROPAGAZIONE DELLE LABEL AI SEGMENTI
    # ========================================================

    # Ogni segmento eredita la label di valence del trial originale a cui appartiene
    # trial_idx contiene, per ogni segmento, l'indice del trial originale
    y_valence_seg = y_valence[
        trial_idx
    ]

    # Ogni segmento eredita anche la label di arousal del trial originale
    y_arousal_seg = y_arousal[
        trial_idx
    ]

    # Allo stesso modo viene associato a ogni segmento l'identificativo del soggetto
    subject_ids_seg = subject_ids[
        trial_idx
    ]


    # ========================================================
    # ESTRAZIONE DELLE FEATURE
    # ========================================================

    # Estrae le feature da ogni segmento EEG
    # Per ogni segmento viene chiamata extract_segment_features()
    #
    # Il risultato finale è una matrice:
    #   righe    -> segmenti EEG
    #   colonne  -> feature estratte
    X_features = np.array([
        extract_segment_features(seg)
        for seg in X_segments
    ])

    # Restituisce:
    # X_features: matrice delle feature
    # y_valence_seg: classe di valence di ogni segmento
    # y_arousal_seg: classe di arousal di ogni segmento
    # subject_ids_seg: soggetto associato a ogni segmento
    return (
        X_features,
        y_valence_seg,
        y_arousal_seg,
        subject_ids_seg
    )