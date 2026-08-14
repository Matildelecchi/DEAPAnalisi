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
src/pipeline_test.py

Questo file esegue un test end-to-end della pipeline DEAP,
verificando il corretto funzionamento delle principali fasi
dell'elaborazione dei dati:

- caricamento del dataset;
- filtraggio dei segnali EEG;
- segmentazione dei trial;
- estrazione delle feature;
- classificazione tramite modello di machine learning.

Il test utilizza solamente la Logistic Regression nella fase
di classificazione, così da ridurre il tempo necessario
all'esecuzione del test completo.
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# NumPy viene utilizzato per eseguire controlli numerici sui dati,
# in particolare per verificare la presenza di valori NaN e Inf
# dopo il filtraggio e dopo l'estrazione delle feature.
import numpy as np


# Importa dal modulo preprocessing le funzioni necessarie
# per caricare il dataset DEAP e segmentare i segnali EEG.
from preprocessing import load_deap_dataset, segment_signal


# Importa la funzione responsabile del filtraggio dei segnali EEG.
# Il filtraggio comprende il notch filter e il bandpass filter.
from filtering import filter_data


# Importa la funzione utilizzata per estrarre le feature
# dai segmenti EEG e associare a ciascun segmento
# le relative etichette e il soggetto di appartenenza.
from features import extract_features


# Importa la funzione utilizzata per addestrare e valutare
# i modelli di classificazione sulla matrice delle feature.
from models import evaluate_models


# ============================================================
# INIZIO DEL TEST
# ============================================================

# Stampa un'intestazione per rendere immediatamente riconoscibile
# nel terminale l'inizio del test completo della pipeline.
print("\n==============================")
print("   TEST PIPELINE COMPLETA")
print("==============================\n")


# ============================================================
# 1. CARICAMENTO DATI
# ============================================================

# Stampa un messaggio che indica l'inizio della fase
# di caricamento del dataset DEAP.
print(">>> 1. Caricamento dataset")


# Carica il dataset utilizzando solamente i 32 canali EEG.
#
# X:
#   contiene i segnali EEG organizzati per trial, canale
#   e campione temporale.
#
# y:
#   contiene le etichette associate a ciascun trial,
#   tra cui valence, arousal, dominance e liking.
#
# subject_ids:
#   contiene l'identificativo del soggetto associato
#   a ogni trial.
X, y, subject_ids = load_deap_dataset(eeg_only=True)


# Stampa le dimensioni degli array caricati.
# Questo permette di controllare rapidamente la struttura
# dei dati prima di procedere con le fasi successive.
print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("Shape subject_ids:", subject_ids.shape)


# Verifica che X abbia tre dimensioni:
# trial × canali × campioni.
#
# Se la condizione non viene rispettata, Python interrompe
# l'esecuzione e mostra il messaggio specificato.
assert X.ndim == 3, "X deve essere (trial, canali, campioni)"


# Verifica che y sia una matrice bidimensionale.
#
# Nel dataset DEAP le quattro colonne rappresentano:
# valence, arousal, dominance e liking.
assert y.ndim == 2, "y deve essere (trial, 4)"


# Verifica che sia presente un identificativo del soggetto
# per ogni trial contenuto nel dataset.
assert len(subject_ids) == X.shape[0], "subject_ids deve avere un id per trial"


# Messaggio che conferma il superamento dei controlli
# relativi al caricamento del dataset.
print("OK: caricamento dati\n")


# ============================================================
# 2. FILTRAGGIO
# ============================================================

# Indica l'inizio della fase di filtraggio dei segnali EEG.
#
# Il filtraggio previsto dalla pipeline comprende:
# - notch filter per attenuare il rumore a 50 Hz;
# - bandpass filter per mantenere la banda di frequenze
#   di interesse dell'EEG.
print(">>> 2. Filtraggio EEG (notch + bandpass)")


# Applica il filtraggio a tutti i trial e a tutti i canali EEG.
#
# Il risultato mantiene la stessa struttura dei dati originali:
# trial × canali × campioni.
X_filtered = filter_data(X)


# Stampa la dimensione dei dati dopo il filtraggio.
print("Shape X_filtered:", X_filtered.shape)


# Conta e stampa il numero di valori NaN presenti
# nei segnali filtrati.
#
# I valori NaN rappresentano dati non numerici
# e potrebbero causare problemi nelle fasi successive.
print("NaN:", np.isnan(X_filtered).sum())


# Conta e stampa il numero di valori Inf presenti
# nei segnali filtrati.
#
# Anche i valori infiniti possono compromettere
# l'estrazione delle feature e l'addestramento dei modelli.
print("Inf:", np.isinf(X_filtered).sum())


# Verifica che dopo il filtraggio non siano presenti NaN.
#
# Se viene trovato anche un solo NaN, il test viene interrotto.
assert np.isnan(X_filtered).sum() == 0, "Ci sono NaN dopo filtraggio"


# Verifica che dopo il filtraggio non siano presenti valori
# infiniti.
assert np.isinf(X_filtered).sum() == 0, "Ci sono Inf dopo filtraggio"


# Messaggio che conferma il corretto completamento
# della fase di filtraggio.
print("OK: filtraggio\n")


# ============================================================
# 3. SEGMENTAZIONE
# ============================================================

# Indica l'inizio della fase di segmentazione.
print(">>> 3. Segmentazione")


# Divide ogni trial EEG in finestre temporali di 15 secondi.
#
# fs=128:
#   indica la frequenza di campionamento del segnale.
#
# segment_length=15:
#   indica la durata, in secondi, di ogni segmento.
#
# La funzione restituisce:
#
# X_segments:
#   contiene tutti i segmenti ottenuti.
#
# trial_idx:
#   indica, per ogni segmento, il trial originale
#   dal quale è stato ricavato.
X_segments, trial_idx = segment_signal(
    X_filtered,
    segment_length=15,
    fs=128
)


# Stampa la dimensione dell'array contenente i segmenti.
# Questo permette di controllare il numero di segmenti prodotti
# e la loro struttura.
print("Shape X_segments:", X_segments.shape)


# Stampa la dimensione dell'array trial_idx.
# Deve contenere un elemento per ogni segmento generato.
print("Shape trial_idx:", trial_idx.shape)


# Verifica che i segmenti siano rappresentati da un array
# tridimensionale:
#
# segmenti × canali × campioni.
assert X_segments.ndim == 3, "Segmenti devono essere (segmenti, canali, campioni)"


# Verifica che ogni segmento abbia un corrispondente
# identificativo del trial originale.
assert len(trial_idx) == X_segments.shape[0], "trial_idx deve avere un indice per segmento"


# Messaggio che conferma il superamento dei controlli
# relativi alla segmentazione.
print("OK: segmentazione\n")


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

# Indica l'inizio della fase di estrazione delle feature.
print(">>> 4. Feature extraction")


# Estrae le feature da tutti i segmenti EEG.
#
# X_features:
#   matrice contenente le feature numeriche estratte
#   da ogni segmento.
#
# y_valence_seg:
#   etichetta binaria di valence associata a ogni segmento.
#
# y_arousal_seg:
#   etichetta binaria di arousal associata a ogni segmento.
#
# subj_seg:
#   identificativo del soggetto associato a ogni segmento.
#
# trial_idx viene utilizzato per collegare ogni segmento
# al trial originale e quindi recuperare la relativa label.
X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
    X_segments,
    y,
    trial_idx,
    subject_ids
)


# Stampa la dimensione della matrice delle feature.
#
# La prima dimensione corrisponde al numero di segmenti,
# mentre la seconda corrisponde al numero di feature estratte
# per ciascun segmento.
print("Shape X_features:", X_features.shape)


# Conta il numero di valori NaN presenti nelle feature.
print("NaN nelle feature:", np.isnan(X_features).sum())


# Conta il numero di valori infiniti presenti nelle feature.
print("Inf nelle feature:", np.isinf(X_features).sum())


# Verifica che non siano presenti valori NaN nella matrice
# delle feature.
#
# L'assenza di NaN è importante perché i modelli di machine
# learning generalmente non possono essere addestrati
# direttamente su dati contenenti valori mancanti.
assert np.isnan(X_features).sum() == 0, "Ci sono NaN nelle feature"


# Verifica che non siano presenti valori infiniti
# nella matrice delle feature.
assert np.isinf(X_features).sum() == 0, "Ci sono Inf nelle feature"


# Messaggio che conferma il corretto completamento
# della fase di estrazione delle feature.
print("OK: feature extraction\n")


# ============================================================
# 5. MODELLI
#    SOLO LOGISTIC REGRESSION PER VELOCITÀ
# ============================================================

# Indica l'inizio della fase di test del modello.
#
# In questo test viene utilizzata solamente la Logistic Regression
# per ridurre il tempo di esecuzione rispetto all'utilizzo
# di tutti i classificatori previsti dalla pipeline completa.
print(">>> 5. Test modelli (solo LogReg per velocità)")


# Esegue la valutazione dei modelli utilizzando:
#
# X_features:
#   feature estratte dai segmenti EEG.
#
# y_valence_seg:
#   classificazione binaria della valence.
#
# subj_seg:
#   identificativo del soggetto associato a ogni segmento,
#   utilizzato per effettuare una valutazione subject-independent.
#
# n_splits=5:
#   indica l'utilizzo di 5 suddivisioni nella cross-validation.
#
# task_name="valence":
#   specifica che il compito di classificazione riguarda
#   la dimensione emotiva della valence.
results_valence = evaluate_models(
    X_features,
    y_valence_seg,
    subj_seg,
    n_splits=5,
    task_name="valence"
)


# Stampa a terminale i risultati ottenuti dalla valutazione
# del modello sulla classificazione della valence.
print("Risultati valence:", results_valence)


# ============================================================
# CONCLUSIONE DEL TEST
# ============================================================

# Se tutte le fasi precedenti sono state completate senza
# generare errori o fallimenti degli assert, viene visualizzato
# un messaggio che indica il completamento della pipeline.
print("\n>>> Pipeline COMPLETA FUNZIONANTE ✔")