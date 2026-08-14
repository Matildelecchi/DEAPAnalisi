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

Questo modulo contiene le funzioni utilizzate per filtrare
i segnali EEG del dataset DEAP.

Vengono applicati due tipi di filtro:

- Notch a 50 Hz:
    utilizzato per attenuare il rumore elettrico associato
    alla frequenza di rete.

- Bandpass 4–45 Hz:
    utilizzato per mantenere le principali componenti
    di interesse del segnale EEG e attenuare le frequenze
    al di fuori dell'intervallo considerato.

Il filtraggio viene applicato prima al singolo canale
e successivamente a tutti i trial e a tutti i canali.
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# NumPy viene utilizzato per creare e gestire gli array
# contenenti i segnali EEG.
import numpy as np


# Importa le funzioni necessarie per progettare i filtri:
#
# - butter:
#   crea un filtro Butterworth;
#
# - filtfilt:
#   applica il filtro sia in avanti che all'indietro,
#   evitando lo sfasamento del segnale;
#
# - iirnotch:
#   crea un filtro notch per attenuare una frequenza specifica.
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch
)


# Frequenza di campionamento del segnale EEG.
#
# Il segnale viene campionato a 128 Hz, quindi vengono
# acquisiti 128 campioni ogni secondo.
FS = 128


# ============================================================
# 1. FILTRO NOTCH A 50 Hz
# ============================================================

def notch_filter(signal, freq=50, fs=FS, Q=30):
    """
    Applica un filtro notch al segnale.

    Il filtro notch viene utilizzato per attenuare una
    frequenza specifica. In questo caso la frequenza
    considerata è 50 Hz, corrispondente alla frequenza
    della rete elettrica in Italia e in gran parte
    dell'Europa.

    Parametri:
        signal:
            segnale EEG da filtrare.

        freq:
            frequenza da attenuare.
            Per default è 50 Hz.

        fs:
            frequenza di campionamento.
            Per default è 128 Hz.

        Q:
            fattore di qualità del filtro.
            Un valore maggiore rende il filtro più selettivo
            attorno alla frequenza da eliminare.

    Restituisce:
        Il segnale dopo l'applicazione del filtro notch.
    """

    # Crea i coefficienti del filtro notch.
    #
    # freq / (fs / 2) normalizza la frequenza rispetto
    # alla frequenza di Nyquist.
    #
    # La frequenza di Nyquist è fs / 2, quindi nel nostro caso:
    #
    # 128 / 2 = 64 Hz.
    b, a = iirnotch(
        freq / (fs / 2),
        Q
    )


    # Applica il filtro al segnale.
    #
    # filtfilt applica il filtro in entrambe le direzioni
    # (forward e backward).
    #
    # Questo permette di ridurre lo sfasamento temporale
    # introdotto normalmente dai filtri.
    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# 2. FILTRO BANDPASS 4–45 Hz
# ============================================================

def bandpass_filter(
    signal,
    low=4,
    high=45,
    fs=FS,
    order=4
):
    """
    Applica un filtro passa-banda al segnale EEG.

    Il filtro mantiene le frequenze comprese tra 4 e 45 Hz
    e attenua quelle al di fuori di questo intervallo.

    L'intervallo comprende le principali bande EEG utilizzate
    successivamente nell'estrazione delle feature:

        theta -> 4-8 Hz
        alpha -> 8-13 Hz
        beta  -> 13-30 Hz
        gamma -> 30-45 Hz

    Parametri:
        signal:
            segnale EEG da filtrare.

        low:
            frequenza minima del filtro.
            Default: 4 Hz.

        high:
            frequenza massima del filtro.
            Default: 45 Hz.

        fs:
            frequenza di campionamento.
            Default: 128 Hz.

        order:
            ordine del filtro Butterworth.
            Default: 4.

    Restituisce:
        Il segnale filtrato.
    """

    # Crea i coefficienti del filtro Butterworth
    # passa-banda.
    #
    # Le frequenze vengono normalizzate rispetto
    # alla frequenza di Nyquist:
    #
    #     fs / 2 = 64 Hz
    #
    # quindi:
    #
    #     4 / 64
    #     45 / 64
    #
    # definiscono i limiti normalizzati del filtro.
    b, a = butter(
        order,
        [
            low / (fs / 2),
            high / (fs / 2)
        ],
        btype="band"
    )


    # Applica il filtro al segnale.
    #
    # Anche in questo caso viene utilizzato filtfilt()
    # per evitare lo sfasamento temporale del segnale.
    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# 3. FILTRAGGIO COMPLETO DI UN SINGOLO CANALE
# ============================================================

def filter_channel(signal):
    """
    Applica in sequenza tutti i filtri previsti
    a un singolo canale EEG.

    La sequenza è:

        segnale originale
                ↓
        filtro notch 50 Hz
                ↓
        filtro bandpass 4–45 Hz
                ↓
        segnale filtrato
    """

    # Primo passaggio:
    # rimuove/attenua la componente a 50 Hz associata
    # principalmente all'interferenza della rete elettrica.
    x = notch_filter(
        signal
    )


    # Secondo passaggio:
    # mantiene le frequenze comprese tra 4 e 45 Hz.
    x = bandpass_filter(
        x
    )


    # Restituisce il segnale dopo entrambi i filtraggi.
    return x


# ============================================================
# 4. FILTRAGGIO COMPLETO DI TUTTI I TRIAL E CANALI
# ============================================================

def filter_data(data):
    """
    Applica il filtraggio completo a tutti i trial
    e a tutti i canali EEG.

    Parametri:
        data:
            array tridimensionale con struttura:

            (n_trial, n_ch, n_samples)

            dove:

            n_trial   = numero di trial
            n_ch      = numero di canali EEG
            n_samples = numero di campioni per canale

    Restituisce:
        filtered:
            array della stessa dimensione di data,
            contenente i segnali filtrati.
    """

    # Crea un array vuoto con la stessa forma e lo stesso
    # tipo di dato di data.
    #
    # In questo array verranno salvati progressivamente
    # i segnali filtrati.
    filtered = np.zeros_like(
        data
    )


    # Scorre tutti i trial del dataset.
    #
    # data.shape[0] corrisponde al numero di trial.
    for t in range(
        data.shape[0]
    ):

        # Per ogni trial, scorre tutti i canali EEG.
        #
        # data.shape[1] corrisponde al numero di canali.
        for ch in range(
            data.shape[1]
        ):

            # Seleziona il segnale del canale ch
            # appartenente al trial t.
            #
            # data[t, ch] contiene tutti i campioni
            # di quel particolare canale.
            #
            # Il segnale viene passato a filter_channel(),
            # che applica prima il notch e poi il bandpass.
            filtered[t, ch] = filter_channel(
                data[t, ch]
            )


    # Restituisce l'intero dataset filtrato.
    return filtered