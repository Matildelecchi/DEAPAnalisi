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
src/utils.py

Questo modulo contiene funzioni di utilità generiche utilizzate
all'interno del progetto DEAP.

Le principali funzionalità sono:
- impostazione del seed per la riproducibilità;
- creazione delle directory necessarie;
- salvataggio di dati in formato JSON;
- conversione delle label DEAP da valori continui a classi binarie;
- salvataggio e caricamento dei modelli di machine learning.
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# os viene utilizzato per gestire directory e percorsi
# dei file all'interno del progetto.
import os

# json permette di salvare i risultati e le metriche
# in formato JSON.
import json

# pickle viene utilizzato per salvare e caricare
# i modelli di machine learning.
import pickle

# NumPy viene utilizzato per impostare il seed
# delle operazioni casuali basate su NumPy.
import numpy as np

# random è il modulo standard di Python utilizzato
# per la generazione di numeri casuali.
import random


# ============================================================
# 1. RIPRODUCIBILITÀ
# ============================================================

def set_seed(seed=42):
    """
    Imposta il seed globale per le principali librerie
    utilizzate per la generazione di numeri casuali.

    Parametri:
        seed:
            valore utilizzato come seme per la generazione
            casuale. Il valore predefinito è 42.

    Questa funzione permette di ottenere risultati
    maggiormente riproducibili tra diverse esecuzioni
    del programma.
    """

    # Imposta il seed per il generatore casuale di NumPy.
    #
    # In questo modo le operazioni casuali effettuate
    # tramite NumPy possono produrre gli stessi risultati
    # nelle diverse esecuzioni.
    np.random.seed(seed)


    # Imposta lo stesso seed per il modulo random
    # della libreria standard di Python.
    random.seed(seed)


# ============================================================
# 2. DIRECTORY E GESTIONE DEI FILE
# ============================================================

def ensure_dir(path):
    """
    Crea una directory se non esiste già.

    Parametri:
        path:
            percorso della directory da creare.

    Questa funzione viene utilizzata, ad esempio, prima
    di salvare risultati, grafici o modelli in una directory
    che potrebbe non essere ancora presente.
    """

    # Controlla che il percorso sia valido e che la directory
    # non esista già.
    #
    # Il controllo "if path" evita di tentare di creare
    # una directory quando il percorso è una stringa vuota.
    if path and not os.path.exists(path):

        # Crea la directory specificata.
        #
        # os.makedirs() permette anche di creare
        # eventuali directory intermedie mancanti.
        os.makedirs(path)


def save_json(data, filename):
    """
    Salva un oggetto Python in formato JSON.

    Parametri:
        data:
            dati da salvare, generalmente un dizionario
            contenente risultati o metriche.

        filename:
            percorso completo del file JSON da creare.
    """

    # Recupera la directory contenente il file.
    #
    # Ad esempio:
    # "results/custom/results.json"
    #
    # diventa:
    # "results/custom"
    #
    # ensure_dir() verifica che questa directory esista
    # e, se necessario, la crea.
    ensure_dir(
        os.path.dirname(filename)
    )


    # Apre il file in modalità scrittura.
    #
    # "w" indica che il file viene creato oppure
    # sovrascritto se esiste già.
    with open(filename, "w") as f:

        # Converte i dati Python in formato JSON
        # e li scrive all'interno del file.
        #
        # indent=4 rende il file più leggibile
        # inserendo una formattazione con indentazione.
        json.dump(
            data,
            f,
            indent=4
        )


# ============================================================
# 3. LABEL DEAP - BINARIZZAZIONE
# ============================================================

# Indica la posizione delle diverse dimensioni emotive
# all'interno dell'array delle label restituito
# dal dataset DEAP.
#
# Le quattro colonne sono:
# 0 -> valence
# 1 -> arousal
# 2 -> dominance
# 3 -> liking
LABEL_COLUMNS = {
    "valence": 0,
    "arousal": 1,
    "dominance": 2,
    "liking": 3
}


def binarize_labels(
    y,
    dimension="valence",
    threshold=5.0
):
    """
    Converte i valori continui delle valutazioni DEAP
    in due classi binarie.

    Parametri:
        y:
            matrice contenente le label originali del dataset.

        dimension:
            dimensione emotiva da utilizzare.
            Può essere:
            - valence
            - arousal
            - dominance
            - liking

        threshold:
            soglia utilizzata per separare le due classi.

    Restituisce:
        Un array contenente valori binari:
        - 0 -> valore non superiore alla soglia;
        - 1 -> valore superiore alla soglia.
    """

    # Recupera l'indice della colonna corrispondente
    # alla dimensione emotiva scelta.
    #
    # Ad esempio:
    # dimension = "valence"
    # -> col = 0
    col = LABEL_COLUMNS[dimension]


    # Seleziona la colonna corrispondente alla dimensione
    # emotiva e confronta ogni valore con la soglia.
    #
    # Il risultato del confronto è un array booleano:
    # True  -> valore maggiore della soglia
    # False -> valore minore o uguale alla soglia
    #
    # astype(int) converte:
    # True  -> 1
    # False -> 0
    return (
        y[:, col] >= threshold
    ).astype(int)


# ============================================================
# 4. GESTIONE DEI MODELLI
# ============================================================

def save_model_pickle(model, path):
    """
    Salva un modello di machine learning utilizzando
    il formato pickle.

    Parametri:
        model:
            modello di machine learning già addestrato.

        path:
            percorso nel quale salvare il modello.
    """

    # Controlla che la directory destinazione esista.
    #
    # Se non esiste, viene creata automaticamente.
    ensure_dir(
        os.path.dirname(path)
    )


    # Apre il file in modalità binaria di scrittura.
    #
    # "wb" significa:
    # write + binary.
    with open(path, "wb") as f:

        # Serializza il modello e lo salva nel file.
        #
        # In questo modo il modello addestrato può essere
        # successivamente ricaricato senza doverlo
        # riaddestrare.
        pickle.dump(
            model,
            f
        )


    # Comunica nel terminale il percorso nel quale
    # è stato salvato il modello.
    print(
        f"Modello salvato in: {path}"
    )


def load_model_pickle(path):
    """
    Carica un modello di machine learning precedentemente
    salvato tramite pickle.

    Parametri:
        path:
            percorso del file contenente il modello.

    Restituisce:
        Il modello precedentemente salvato.
    """

    # Apre il file in modalità binaria di lettura.
    #
    # "rb" significa:
    # read + binary.
    with open(path, "rb") as f:

        # Ricostruisce il modello a partire dal contenuto
        # serializzato del file.
        return pickle.load(f)