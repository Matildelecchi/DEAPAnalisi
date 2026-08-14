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


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# NumPy viene utilizzato per eseguire operazioni numeriche
# e, in particolare, per controllare la presenza di valori
# NaN e Inf nelle feature estratte.
import numpy as np


# Importa le funzioni necessarie per:
# - caricare il dataset DEAP;
# - dividere i segnali EEG in segmenti.
from src.preprocessing import (
    load_deap_dataset,
    segment_signal
)


# Importa le funzioni necessarie per:
# - estrarre le feature da un singolo segmento;
# - estrarre le feature dall'intero insieme di segmenti.
from src.features import (
    extract_features,
    extract_segment_features
)


# ============================================================
# INIZIO DEL TEST
# ============================================================

# Messaggio iniziale visualizzato nel terminale per indicare
# che sta iniziando il test dell'estrazione delle feature.
print("=== TEST FEATURE EXTRACTION ===")


# ============================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================

# Carica il dataset DEAP utilizzando solamente i segnali EEG.
#
# X:
#   contiene i segnali EEG.
#
# y:
#   contiene le etichette associate ai trial, tra cui
#   valence e arousal.
#
# subject_ids:
#   contiene l'identificativo del soggetto associato
#   a ciascun trial.
X, y, subject_ids = load_deap_dataset(
    eeg_only=True
)


# Stampa le dimensioni dei tre array caricati.
#
# X.shape indica la struttura dei dati EEG.
# y.shape indica il numero e la struttura delle etichette.
# subject_ids.shape indica il numero di identificativi dei soggetti.
print(
    "Dataset caricato:",
    X.shape,
    y.shape,
    subject_ids.shape
)


# ============================================================
# 2. SEGMENTAZIONE DEL SEGNALE
# ============================================================

# Divide i segnali EEG in segmenti di 15 secondi.
#
# fs=128 indica la frequenza di campionamento utilizzata
# per il segnale.
#
# overlap=0.0 significa che i segmenti non si sovrappongono.
#
# X_segments:
#   contiene i segmenti EEG generati.
#
# trial_idx:
#   permette di sapere a quale trial originale appartiene
#   ciascun segmento.
X_segments, trial_idx = segment_signal(
    X,
    segment_length=15,
    fs=128,
    overlap=0.0
)


# Stampa la dimensione dell'array contenente i segmenti.
print(
    "Segmenti generati:",
    X_segments.shape
)


# Stampa i primi 20 identificativi dei trial.
#
# Questo permette di verificare manualmente a quali trial
# appartengono i primi segmenti generati.
print(
    "Trial_idx primi 20:",
    trial_idx[:20]
)


# ============================================================
# 3. TEST DIAGNOSTICO:
#    ESTRAZIONE DELLE FEATURE DI UN SINGOLO SEGMENTO
# ============================================================

# Messaggio che indica l'inizio del test diagnostico.
print(
    "\n=== Test diagnostico: "
    "estrazione feature segmento per segmento ==="
)


# Analizza i primi 300 segmenti, oppure tutti i segmenti
# se il dataset contiene meno di 300 segmenti.
#
# range(0, min(300, len(X_segments))) permette quindi
# di limitare il test per evitare di elaborare inutilmente
# l'intero dataset.
for i in range(
    0,
    min(300, len(X_segments))
):

    # Ogni 50 segmenti viene stampato un messaggio
    # per monitorare l'avanzamento dell'elaborazione.
    if i % 50 == 0:
        print(
            f"Processing segment {i}/{len(X_segments)}"
        )


    # Estrae le feature dal segmento corrente.
    #
    # Il risultato viene assegnato a "_" perché in questa fase
    # interessa solamente verificare che la funzione funzioni
    # correttamente, non utilizzare direttamente le feature.
    _ = extract_segment_features(
        X_segments[i]
    )


# Messaggio che conferma il completamento del test
# di estrazione delle feature per singolo segmento.
print(
    "\nOK: estrazione singolo segmento funziona."
)


# ============================================================
# 4. TEST COMPLETO DELLA FUNZIONE extract_features
# ============================================================

# Indica l'inizio del test completo dell'estrazione delle feature.
print(
    "\n=== Test extract_features completo ==="
)


# Esegue l'estrazione delle feature su tutti i segmenti.
#
# X_features:
#   matrice contenente le feature estratte.
#
# y_valence_seg:
#   etichette di valence associate ai segmenti.
#
# y_arousal_seg:
#   etichette di arousal associate ai segmenti.
#
# subj_seg:
#   identificativo del soggetto associato a ciascun segmento.
X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
    X_segments,
    y,
    trial_idx,
    subject_ids
)


# Stampa la dimensione della matrice delle feature.
#
# Questa informazione permette di controllare:
# - quanti segmenti sono stati elaborati;
# - quante feature sono state estratte per ogni segmento.
print(
    "Feature shape:",
    X_features.shape
)


# Stampa la dimensione delle etichette di valence
# associate ai segmenti.
print(
    "Valence seg shape:",
    y_valence_seg.shape
)


# Stampa la dimensione delle etichette di arousal
# associate ai segmenti.
print(
    "Arousal seg shape:",
    y_arousal_seg.shape
)


# Stampa la dimensione degli identificativi dei soggetti
# associati ai segmenti.
print(
    "Subject seg shape:",
    subj_seg.shape
)


# ============================================================
# 5. CONTROLLO DI NaN E Inf
# ============================================================

# Verifica quanti valori NaN sono presenti nella matrice
# delle feature.
#
# np.isnan() crea una matrice booleana:
# True dove trova un NaN e False negli altri casi.
#
# .sum() conta quindi quanti NaN sono presenti complessivamente.
print(
    "\nNaN nelle feature:",
    np.isnan(X_features).sum()
)


# Verifica quanti valori Inf (infinito) sono presenti
# nelle feature.
#
# Anche in questo caso .sum() conta il numero totale
# di valori che soddisfano la condizione.
print(
    "Inf nelle feature:",
    np.isinf(X_features).sum()
)


# ============================================================
# 6. CONTROLLO DELLA COERENZA DELLE LABEL
# ============================================================

# Seleziona il primo segmento per effettuare un controllo
# manuale della corrispondenza tra:
# - trial originale;
# - etichetta originale;
# - etichetta assegnata al segmento.
i = 0


# Indica che stanno per essere visualizzate le informazioni
# relative al segmento numero 0.
print("\nSegmento 0:")


# Stampa l'identificativo del trial originale al quale
# appartiene il segmento 0.
print(
    "  trial_idx:",
    trial_idx[i]
)


# Recupera dalla matrice y la valence originale del trial
# associato al segmento.
#
# trial_idx[i] indica quale trial originale corrisponde
# al segmento i.
#
# La colonna 0 di y contiene la valence.
print(
    "  valence originale:",
    y[trial_idx[i], 0]
)


# Stampa la valence assegnata al segmento dalla funzione
# extract_features().
#
# Serve per verificare che l'etichetta del segmento
# corrisponda a quella del trial originale.
print(
    "  valence segmentata:",
    y_valence_seg[i]
)


# Recupera dalla matrice y l'arousal originale del trial
# associato al segmento.
#
# La colonna 1 di y contiene l'arousal.
print(
    "  arousal originale:",
    y[trial_idx[i], 1]
)


# Stampa l'arousal assegnato al segmento.
#
# Anche questo valore viene confrontato con quello
# originale per verificare la corretta propagazione
# delle etichette durante la segmentazione.
print(
    "  arousal segmentata:",
    y_arousal_seg[i]
)


# ============================================================
# 7. CONTROLLO DEI SOGGETTI
# ============================================================

# np.unique() restituisce tutti gli identificativi distinti
# dei soggetti presenti nei segmenti.
#
# Questo controllo permette di verificare che gli ID dei
# soggetti siano stati correttamente propagati dal dataset
# originale ai segmenti creati durante il preprocessing.
print(
    "\nSoggetti unici nei segmenti:",
    np.unique(subj_seg)
)