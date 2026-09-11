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
main.py — Pipeline DEAPAnalisi

Funzionalità:
- Esecuzione della pipeline baseline, basata sull'approccio
  descritto da Koelstra et al. (2012).
- Esecuzione della pipeline custom, con segmentazione e
  normalizzazione personalizzate.
- Addestramento e confronto di diversi classificatori.
- Valutazione tramite metriche di classificazione.
- Generazione delle matrici di confusione e delle curve ROC.
- Salvataggio delle metriche e dei modelli addestrati.
- Gestione del caso in cui la ROC AUC risulti NaN.
"""
# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# Permette di gestire i percorsi dei file e delle directory
# in maniera indipendente dal sistema operativo.
import os

# Libreria utilizzata per operazioni numeriche e gestione degli array.
import numpy as np

#per stampare a schermo il procedimento
import sys
sys.stdout.reconfigure(line_buffering=True)

# Permette di filtrare alcuni warning prodotti dalle librerie
# durante l'esecuzione della pipeline.
import warnings


# ============================================================
# IMPORT DEI MODULI DEL PROGETTO
# ============================================================

from src.filtering import filter_data

# Funzioni relative al preprocessing del dataset:
# - load_deap_dataset: carica il dataset DEAP
# - segment_signal: divide i segnali EEG in segmenti
# - normalize_signal: normalizza i segmenti
from src.preprocessing import load_deap_dataset, segment_signal, normalize_signal

# Funzione che estrae le feature dai segnali EEG.
from src.features import extract_features

# Funzioni per creare e addestrare i diversi classificatori.
from src.models import (
    train_gaussian_nb,
    train_svm,
    train_knn,
    train_logreg,
    train_decision_tree,
)

# Funzioni utilizzate per:
# - eseguire la cross-validation
# - salvare le metriche
# - generare la confusion matrix
# - generare la curva ROC
from src.evaluation import (
    cross_validate_subject_independent,
    cross_validate_leave_one_trial_out,
    save_metrics_json,
    plot_confusion_matrix,
    plot_roc_curve,
)

# Funzioni di utilità per:
# - salvare un modello in formato pickle
# - creare una directory se non esiste
from src.utils import save_model_pickle, ensure_dir


# ============================================================
# FILTRO WARNING
# ============================================================
# Ignora i UserWarning prodotti dalle librerie utilizzate.
# Questo permette di avere un output da terminale più pulito.
warnings.filterwarnings("ignore", category=UserWarning)

# Ignora anche i RuntimeWarning, ad esempio quelli che possono
# comparire durante alcune operazioni numeriche.
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# DIRECTORY
# ============================================================

# Directory contenente i dati grezzi del dataset DEAP.
RAW_DIR = "data/raw/"

# Directory in cui vengono salvati i risultati della pipeline
# baseline.
BASELINE_DIR = "results/baseline/"

# Directory in cui vengono salvati i risultati della pipeline
# custom.
CUSTOM_DIR = "results/custom/"

# Directory contenente i modelli di machine learning salvati.
MODELS_DIR = "results/models/"

# Directory contenente i grafici generati durante la valutazione.
FIGURES_DIR = "results/figures/"

# Assicura che tutte le directory necessarie esistano.
# Se una directory non esiste, ensure_dir() la crea.
for d in [BASELINE_DIR, CUSTOM_DIR, MODELS_DIR, FIGURES_DIR]:
    ensure_dir(d)


# ============================================================
# BASELINE
# ============================================================

def run_baseline(dimension="valence"):
    """
    Esegue la pipeline baseline per una delle due dimensioni
    emotive considerate dal dataset DEAP:
    - valence
    - arousal

    La baseline utilizza:
    - segmenti di 60 secondi
    - nessuna sovrapposizione tra segmenti
    - classificatore Gaussian Naive Bayes
    - Leave-One-Trial-Out cross-validation per ogni soggetto.
    """

    # Stampa a terminale quale configurazione della baseline
    # sta per essere eseguita.
    print(f"\n=== BASELINE ({dimension}) ===")

    # --------------------------------------------------------
    # CARICAMENTO DEL DATASET
    # --------------------------------------------------------

    # Carica i segnali EEG dal dataset DEAP.
    #
    # X            -> segnali EEG
    # y_raw        -> etichette originali relative alle emozioni
    # subject_ids  -> identificativo del soggetto associato
    #                 a ciascun trial
    #
    # eeg_only=True indica che vengono utilizzati solamente
    # i canali EEG e non altri segnali fisiologici del dataset.
    X, y_raw, subject_ids = load_deap_dataset(RAW_DIR, eeg_only=True)
    
    X = filter_data(X)

    # --------------------------------------------------------
    # SEGMENTAZIONE
    # --------------------------------------------------------

    # Divide ogni segnale EEG in segmenti della durata di
    # 60 secondi.
    #
    # overlap=0.0 significa che i segmenti non si sovrappongono.
    #
    # segments  -> segmenti EEG ottenuti
    # trial_idx -> indice del trial originale a cui appartiene
    #              ogni segmento
    segments, trial_idx = segment_signal(X, segment_length=60, overlap=0.0)

    # --------------------------------------------------------
    # ESTRAZIONE DELLE FEATURE
    # --------------------------------------------------------

    # Estrae le feature dai segmenti EEG.
    #
    # Vengono inoltre associate ad ogni segmento:
    # - etichetta di valence
    # - etichetta di arousal
    # - identificativo del soggetto
    #
    # X_features      -> matrice finale delle feature
    # y_valence_seg   -> etichette valence per segmento
    # y_arousal_seg   -> etichette arousal per segmento
    # subj_seg        -> soggetto associato a ogni segmento
    X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
        segments, y_raw, trial_idx, subject_ids
    )

    # --------------------------------------------------------
    # SCELTA DELLA DIMENSIONE EMOTIVA
    # --------------------------------------------------------

    # Il dataset DEAP permette di studiare due dimensioni:
    # - valence: quanto l'emozione è positiva o negativa
    # - arousal: quanto l'emozione è attivante o rilassante
    #
    # In base al parametro passato alla funzione viene scelta
    # una delle due serie di etichette.
    y_seg = y_valence_seg if dimension == "valence" else y_arousal_seg

    # Dizionario nel quale verranno memorizzate le metriche
    # ottenute separatamente per ogni soggetto.
    results_per_subject = {}

    # ========================================================
    # VALUTAZIONE SOGGETTO PER SOGGETTO
    # ========================================================

    # np.unique() restituisce tutti gli identificativi dei
    # soggetti presenti nel dataset.
    #
    # La baseline esegue una valutazione separata per ciascun
    # soggetto.
    for subj in np.unique(subj_seg):
        
        # Crea una maschera booleana che seleziona solamente
        # i segmenti appartenenti al soggetto corrente.
        mask = subj_seg == subj
        # Estrae feature ed etichette relative al soggetto.
        X_subj, y_subj = X_features[mask], y_seg[mask]

        # ----------------------------------------------------
        # CONTROLLO SULLE CLASSI
        # ----------------------------------------------------

        # Per poter eseguire una classificazione binaria è
        # necessario che siano presenti entrambe le classi.
        #
        # Se il soggetto contiene una sola classe, viene
        # ignorato perché non sarebbe possibile effettuare
        # una corretta valutazione.
        if len(np.unique(y_subj)) < 2:
            continue

         # Crea un identificativo locale per ogni segmento.
        #
        # Questi ID vengono utilizzati dalla funzione di
        # Leave-One-Trial-Out cross-validation.
        local_trial_ids = np.arange(X_subj.shape[0])

        # ----------------------------------------------------
        # CROSS-VALIDATION
        # ----------------------------------------------------

        # Esegue la Leave-One-Trial-Out cross-validation
        # utilizzando Gaussian Naive Bayes.
        #
        # La funzione restituisce:
        # - metriche dei singoli fold
        # - media delle metriche
        _, avg_metrics = cross_validate_leave_one_trial_out(
            train_gaussian_nb, X_subj, y_subj, local_trial_ids
        )

        # ----------------------------------------------------
        # CORREZIONE ROC AUC
        # ----------------------------------------------------

        # In alcuni casi la ROC AUC può risultare NaN, ad esempio
        # quando in un fold è presente una sola classe.
        #
        # Una ROC AUC pari a 0.5 rappresenta una classificazione
        # equivalente al caso casuale.        
        if np.isnan(avg_metrics["roc_auc"]):
            avg_metrics["roc_auc"] = 0.5

        # Salva le metriche ottenute per il soggetto corrente.
        results_per_subject[subj] = avg_metrics


        # Stampa alcune delle principali metriche.
        print(
            f"  Soggetto {subj}: "
            f"acc={avg_metrics['accuracy']:.3f}  "
            f"f1={avg_metrics['f1']:.3f}"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        # Addestra un modello Gaussian Naive Bayes utilizzando
        # tutti i dati del soggetto.
        #
        # ATTENZIONE:
        # questo modello viene utilizzato qui per generare il
        # grafico e non per calcolare le metriche della
        # cross-validation.
        model = train_gaussian_nb(X_subj, y_subj)

        # Predice le classi sugli stessi dati utilizzati
        # per l'addestramento.
        y_pred = model.predict(X_subj)


        # Costruisce il percorso in cui salvare la confusion matrix.
        cm_path = os.path.join(
            FIGURES_DIR,
            f"baseline_cm_{dimension}_{subj}.png"
        )


        # Genera e salva la matrice di confusione.
        plot_confusion_matrix(
            y_subj,
            y_pred,
            title=f"Baseline CM - {dimension} - {subj}",
            save_path=cm_path
        )


    # ========================================================
    # METRICHE COMPLESSIVE
    # ========================================================

    # Calcola la media delle metriche ottenute sui diversi
    # soggetti.
    #
    # next(iter(...)) permette di recuperare le chiavi presenti
    # nel dizionario delle metriche, ad esempio:
    # accuracy, precision, recall, f1, roc_auc...
    overall = {
        key: float(
            np.mean([
                m[key]
                for m in results_per_subject.values()
            ])
        )
        for key in next(iter(results_per_subject.values()))
    }


    # Salva su file JSON:
    # - le metriche di ogni soggetto
    # - le metriche complessive
    save_metrics_json(
        {
            "per_subject": results_per_subject,
            "overall": overall
        },
        os.path.join(
            BASELINE_DIR,
            f"baseline_{dimension}_metrics.json"
        ),
    )


    # Restituisce le metriche complessive.
    return overall


# ============================================================
# CUSTOM
# ============================================================

def run_custom(dimension="valence"):
    """
    Esegue la pipeline custom.

    Rispetto alla baseline:
    - utilizza segmenti di 15 secondi;
    - normalizza i segnali;
    - confronta più classificatori;
    - utilizza una cross-validation subject-independent.
    """

    print(f"\n=== CUSTOM ({dimension}) ===")


    # --------------------------------------------------------
    # CARICAMENTO DATASET
    # --------------------------------------------------------

    # Carica nuovamente i segnali EEG e le relative etichette.
    X, y_raw, subject_ids = load_deap_dataset(
        RAW_DIR,
        eeg_only=True
    )
    
    X = filter_data(X)


    # --------------------------------------------------------
    # SEGMENTAZIONE
    # --------------------------------------------------------

    # Divide i segnali in segmenti più brevi rispetto alla baseline:
    # 15 secondi invece di 60.
    #
    # Anche in questo caso non viene utilizzata sovrapposizione.
    segments, trial_idx = segment_signal(
        X,
        segment_length=15,
        overlap=0.0
    )


    # --------------------------------------------------------
    # NORMALIZZAZIONE
    # --------------------------------------------------------

    # Normalizza i segmenti EEG prima dell'estrazione delle feature.
    #
    # La normalizzazione permette di ridurre differenze di scala
    # tra segnali/soggetti e rendere le feature più confrontabili.
    segments_norm = normalize_signal(segments)


    # --------------------------------------------------------
    # ESTRAZIONE DELLE FEATURE
    # --------------------------------------------------------

    # Estrae le feature dai segnali normalizzati.
    X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
        segments_norm,
        y_raw,
        trial_idx,
        subject_ids
    )


    # Seleziona la dimensione emotiva da classificare.
    y_seg = (
        y_valence_seg
        if dimension == "valence"
        else y_arousal_seg
    )


    # ========================================================
    # CLASSIFICATORI
    # ========================================================

    # Dizionario che associa il nome del classificatore
    # alla relativa funzione di addestramento.
    #
    # In questo modo è possibile iterare automaticamente
    # sui diversi modelli senza duplicare il codice.
    classifiers = {
        "svm": train_svm,
        "knn": train_knn,
        "logreg": train_logreg,
        "decision_tree": train_decision_tree,
    }


    # Dizionario che conterrà le metriche finali di ogni
    # classificatore.
    all_results = {}


    # ========================================================
    # CONFRONTO DEI CLASSIFICATORI
    # ========================================================

    # Esegue la pipeline per ogni classificatore.
    for name, train_fn in classifiers.items():

        print(f"\n--- {name.upper()} ---")


        # ----------------------------------------------------
        # SUBJECT-INDEPENDENT CROSS-VALIDATION
        # ----------------------------------------------------

        # Esegue una cross-validation subject-independent.
        #
        # n_splits=5 significa che il dataset viene suddiviso
        # in 5 fold.
        #
        # L'obiettivo è evitare che dati appartenenti allo stesso
        # soggetto siano contemporaneamente presenti nel training
        # e nel test.
        fold_metrics, avg = cross_validate_subject_independent(
            train_fn,
            X_features,
            y_seg,
            subj_seg,
            n_splits=5
        )


        # Corregge eventuali valori NaN della ROC AUC.
        if np.isnan(avg["roc_auc"]):
            avg["roc_auc"] = 0.5


        # Salva le metriche medie associate al classificatore.
        all_results[name] = avg


        # Mostra accuracy e F1-score nel terminale.
        print(
            f"  acc={avg['accuracy']:.3f}  "
            f"f1={avg['f1']:.3f}"
        )


        # ====================================================
        # GRAFICI
        # ====================================================

        # Addestra il classificatore sull'intero dataset.
        #
        # Anche qui questo modello viene utilizzato per creare
        # i grafici finali e NON per calcolare le metriche della
        # cross-validation.
        model = train_fn(X_features, y_seg)


        # Predice le classi sull'intero dataset.
        y_pred = model.predict(X_features)


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        # Costruisce il percorso del file PNG.
        cm_path = os.path.join(
            FIGURES_DIR,
            f"custom_cm_{dimension}_{name}.png"
        )


        # Genera e salva la matrice di confusione.
        plot_confusion_matrix(
            y_seg,
            y_pred,
            title=f"Custom CM - {dimension} - {name}",
            save_path=cm_path
        )


        # ----------------------------------------------------
        # ROC CURVE
        # ----------------------------------------------------

        # Controlla se il modello dispone del metodo predict_proba.
        #
        # Questo metodo permette di ottenere la probabilità
        # stimata per ciascuna classe.
        if hasattr(model, "predict_proba"):

            # Prende la probabilità della classe positiva.
            # [:, 1] seleziona la seconda colonna, cioè la
            # probabilità della classe 1.
            y_score = model.predict_proba(X_features)[:, 1]


            # Percorso in cui salvare il grafico ROC.
            roc_path = os.path.join(
                FIGURES_DIR,
                f"custom_roc_{dimension}_{name}.png"
            )


            # Genera e salva la curva ROC.
            plot_roc_curve(
                y_seg,
                y_score,
                title=f"Custom ROC - {dimension} - {name}",
                save_path=roc_path
            )


    # ========================================================
    # SALVATAGGIO DELLE METRICHE
    # ========================================================

    # Salva in formato JSON le metriche di tutti i classificatori.
    save_metrics_json(
        all_results,
        os.path.join(
            CUSTOM_DIR,
            f"custom_{dimension}_summary.json"
        ),
    )


    # ========================================================
    # SELEZIONE DEL MODELLO MIGLIORE
    # ========================================================

    # Seleziona il classificatore che ha ottenuto il valore
    # di F1-score più alto.
    #
    # max(..., key=...) confronta i valori di F1 contenuti
    # nel dizionario all_results.
    best_name = max(
        all_results,
        key=lambda k: all_results[k]["f1"]
    )


    # Riaddestra il miglior classificatore sull'intero dataset.
    best_model = classifiers[best_name](
        X_features,
        y_seg
    )


    # Salva il modello addestrato in formato .pkl.
    #
    # Il file potrà essere successivamente caricato senza
    # dover riaddestrare il modello.
    save_model_pickle(
        best_model,
        os.path.join(
            MODELS_DIR,
            f"custom_{dimension}_{best_name}.pkl"
        ),
    )


    # Restituisce le metriche di tutti i classificatori.
    return all_results


# ============================================================
# MAIN
# ============================================================

# Questa condizione verifica che il file venga eseguito
# direttamente e non importato come modulo da un altro file.
if __name__ == "__main__":

    # Esegue entrambe le dimensioni emotive considerate:
    # - valence
    # - arousal
    for dim in ["valence", "arousal"]:

        # Esegue la pipeline baseline.
        run_baseline(dimension=dim)

        # Esegue la pipeline custom.
        run_custom(dimension=dim)


    # Messaggio finale visualizzato quando tutte le elaborazioni
    # sono state completate.
    print("\n=== Pipeline completata ===")