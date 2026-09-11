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
src/evaluation.py

Questo modulo contiene tutte le funzioni utilizzate per valutare
le prestazioni dei modelli di Machine Learning applicati al
dataset DEAP.

Le principali funzionalità sono:

- calcolo delle metriche di classificazione;
- cross-validation subject-independent tramite GroupKFold;
- Leave-One-Trial-Out validation;
- generazione della confusion matrix;
- generazione della curva ROC;
- salvataggio dei risultati in formato CSV e JSON.
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# Gestire i percorsi dei file e delle directory in cui vengono salvati grafici e risultati
import os

# Salvare le metriche in formato JSON
import json

# Operazioni numeriche e per calcolare le medie delle metriche ottenute nei vari fold
import numpy as np

# Organizzare i risultati in DataFrame e salvarli in formato CSV
import pandas as pd


# ============================================================
# CONFIGURAZIONE DI MATPLOTLIB
# ============================================================

# Importa matplotlib.
import matplotlib

# Imposta il backend "Agg" quindi può creare e salvare immagini senza aprire una finestra grafica
matplotlib.use("Agg")

# Importa pyplot, utilizzato per creare e salvare i grafici.
import matplotlib.pyplot as plt


# ============================================================
# IMPORT DELLE METRICHE DI SCIKIT-LEARN
# ============================================================

from sklearn.metrics import (
    # Percentuale di predizioni corrette.
    accuracy_score,

    # F1-score, utile per combinare precision e recall
    f1_score,

    # Misura quante delle predizioni positive sono corrette
    precision_score,

    # Misura quante delle istanze positive reali vengono correttamente riconosciute
    recall_score,

    # Calcola la matrice di confusione
    confusion_matrix,

    # Calcola l'area sotto la curva ROC
    roc_auc_score,

    # Calcola i punti necessari per costruire la curva ROC
    roc_curve,

    # Calcola l'area sotto una curva
    auc,
)


# I dati appartenenti allo stesso soggetto non devono comparire contemporaneamente nel training e nel test
from sklearn.model_selection import GroupKFold


# Per creare una directory se questa non esiste
from src.utils import ensure_dir


# ============================================================
# 1. VALUTAZIONE DI UN MODELLO SU TEST SET SEPARATO
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Valuta un modello già addestrato su un test set separato.

    Il modello non viene addestrato all'interno di questa funzione:
    viene semplicemente utilizzato per effettuare le predizioni
    sul test set e calcolare le metriche.

    Parametri:
        model: modello di Machine Learning già addestrato.
        X_test: feature relative ai dati di test.
        y_test: etichette reali dei dati di test.

    Restituisce:
        Un dizionario contenente le principali metriche
        di classificazione.
    """

    y_pred = model.predict(X_test)


    # ========================================================
    # CALCOLO DELLE METRICHE
    # ========================================================

    # Crea un dizionario contenente le metriche principali.
    metrics = {

        # Accuracy:
        # Rapporto tra il numero di predizioni corrette e il numero totale di predizioni
        "accuracy": accuracy_score(y_test, y_pred),

        # F1-score:
        # Combina precision e recall in un'unica metrica
        # average="binary" indica che si tratta di una classificazione binaria
        # zero_division=0 evita errori nel caso in cui precision o recall non siano definibili
        "f1": f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        ),

        # Precision:
        # Indica la percentuale di predizioni positive che sono effettivamente positive
        "precision": precision_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        ),

        # Recall:
        # Indica la percentuale delle vere istanze positive che il modello riesce a riconoscere
        "recall": recall_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        ),

        # Matrice di confusione:
        # .tolist() converte l'array NumPy in una lista Python (per salvarlo JSON)
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred
        ).tolist(),
    }


    # ========================================================
    # ROC AUC
    # ========================================================

    # Controlla se il modello dispone del metodo predict_proba():
    # Questo metodo permette di ottenere le probabilità associate alle classi
    if hasattr(model, "predict_proba"):

        # Prova a calcolare la ROC AUC.
        try:

            # Ottiene la probabilità stimata per la classe positiva
            # predict_proba() restituisce generalmente una matrice con una colonna per ogni classe
            # [:, 1] seleziona la seconda colonna, cioè la classe 1
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calcola la ROC AUC utilizzando le etichette reali e le probabilità previste dal modello
            metrics["roc_auc"] = roc_auc_score(
                y_test,
                y_proba
            )

        # Se il calcolo della ROC AUC genera un'eccezione, il programma continua senza aggiungere questa metrica
        except Exception:
            pass


    # Restituisce il dizionario contenente le metriche
    return metrics


# ============================================================
# 2. CROSS-VALIDATION SUBJECT-INDEPENDENT
# ============================================================

def cross_validate_subject_independent(
    train_fn,
    X,
    y,
    groups,
    n_splits=5,
    **train_kwargs
):
    """
    Esegue una cross-validation subject-independent utilizzando GroupKFold.

    L'obiettivo è fare in modo che i soggetti presenti nel
    test set di un determinato fold non siano mai presenti
    nel training set dello stesso fold.

    Questo è particolarmente importante nel riconoscimento
    delle emozioni da EEG, perché segnali provenienti dallo
    stesso soggetto possono avere caratteristiche molto simili.

    Parametri:
        train_fn: funzione utilizzata per creare e addestrare il modello.
        X: matrice delle feature.
        y: etichette delle classi.
        groups: identificativo del soggetto associato a ogni campione.
        n_splits: numero di fold della cross-validation.
        train_kwargs: eventuali parametri aggiuntivi passati alla funzione di training.

    Restituisce:
        - fold_metrics: metriche ottenute in ogni fold;
        - avg: media delle metriche sui vari fold.
    """


    # Crea l'oggetto GroupKFold.
    # Con n_splits=5 il dataset viene diviso in 5 fold, mantenendo separati i gruppi, cioè i soggetti
    gkf = GroupKFold(n_splits=n_splits)


    # Lista che conterrà le metriche ottenute in ciascun fold.
    fold_metrics = []


    # ========================================================
    # CICLO SUI FOLD
    # ========================================================

    # gkf.split() restituisce gli indici dei dati da utilizzare per training e test in ciascun fold
    # groups=groups specifica che la divisione deve essere effettuata rispettando i gruppi dei soggetti
    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):

        # Seleziona le feature destinate al training
        X_train = X[train_idx]

        # Seleziona le feature destinate al test
        X_test = X[test_idx]

        # Seleziona le etichette del training set
        y_train = y[train_idx]

        # Seleziona le etichette del test set
        y_test = y[test_idx]


        # ----------------------------------------------------
        # ADDESTRAMENTO
        # ----------------------------------------------------

        # Addestra il classificatore utilizzando solamente i dati del training set
        model = train_fn(
            X_train,
            y_train,
            **train_kwargs
        )


        # ----------------------------------------------------
        # VALUTAZIONE
        # ----------------------------------------------------

        # Valuta il modello sui dati di test che non sono stati utilizzati durante l'addestramento
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Memorizza il numero del fold all'interno del dizionario.
        metrics["fold"] = fold

        # Aggiunge le metriche del fold alla lista complessiva.
        fold_metrics.append(metrics)


    # ========================================================
    # MEDIA DELLE METRICHE
    # ========================================================

    # Calcola la media delle metriche ottenute nei vari fold
    # La confusion matrix non viene mediata perché è una matrice e non una singola metrica numerica
    # Anche "fold" viene escluso perché rappresenta solamente l'identificativo del fold
    avg = {
        key: float(
            np.mean([
                m[key]
                for m in fold_metrics
            ])
        )
        for key in fold_metrics[0]
        if key not in ("confusion_matrix", "fold")
    }


    # Restituisce metriche dei singoli fold e media
    return fold_metrics, avg


# ============================================================
# 3. LEAVE-ONE-TRIAL-OUT
# ============================================================

def cross_validate_leave_one_trial_out(
    train_fn,
    X,
    y,
    trial_ids,
    **train_kwargs
):
    """
    Esegue la validazione Leave-One-Trial-Out.

    Questa metodologia viene utilizzata soggetto per soggetto:
    ad ogni iterazione un trial viene escluso dal training
    e utilizzato come test set.

    Tutti gli altri trial dello stesso soggetto vengono
    utilizzati per addestrare il modello.

    Parametri:
        train_fn: funzione utilizzata per addestrare il classificatore.
        X: feature dei dati.
        y: etichette.
        trial_ids: identificativo del trial associato a ogni campione.
        train_kwargs: eventuali parametri aggiuntivi per il training.

    Restituisce:
        - metriche di ogni trial;
        - media delle metriche sui trial.
    """


    # Recupera tutti gli identificativi distinti dei trial
    unique_trials = np.unique(trial_ids)


    # Lista in cui verranno memorizzate le metriche
    fold_metrics = []


    # ========================================================
    # CICLO SUI TRIAL
    # ========================================================

    # Ogni iterazione seleziona un trial da utilizzare come test set
    for held_out in unique_trials:

        # Crea una maschera che identifica i campioni appartenenti al trial lasciato fuori
        test_mask = trial_ids == held_out
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue


        # ----------------------------------------------------
        # ADDESTRAMENTO
        # ----------------------------------------------------

        # Addestra il modello utilizzando tutti i trial tranne quello attualmente lasciato fuori
        model = train_fn(
            X[train_mask],
            y[train_mask],
            **train_kwargs
        )


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        # Valuta il modello solamente sul trial lasciato fuori
        metrics = evaluate_model(
            model,
            X[test_mask],
            y[test_mask]
        )

        # Salva le metriche del trial corrente
        fold_metrics.append(metrics)


    # ========================================================
    # MEDIA DELLE METRICHE
    # ========================================================

    # Calcola la media delle metriche ottenute nei vari trial
    avg = {
        key: float(
            np.mean([
                m[key]
                for m in fold_metrics
            ])
        )
        for key in fold_metrics[0]
        if key != "confusion_matrix"
    }


    # Restituisce le metriche dei singoli trial e la loro media
    return fold_metrics, avg


# ============================================================
# 4. GRAFICI
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    save_path
):
    """
    Crea e salva una confusion matrix.

    Parametri:
        y_true: etichette reali.
        y_pred: etichette predette dal modello.
        title: titolo del grafico.
        save_path: percorso in cui salvare l'immagine.
    """

    # Crea la directory contenitrice del file di output se non esiste
    # os.path.dirname() estrae solamente la cartella dal percorso completo
    ensure_dir(os.path.dirname(save_path))


    # Calcola la confusion matrix confrontando:
    # - classi reali
    # - classi predette
    cm = confusion_matrix(
        y_true,
        y_pred
    )


    # Crea una nuova figura con dimensione 5x4 pollici
    plt.figure(figsize=(5, 4))


    # Visualizza la matrice come immagine
    # cmap="Blues" scala di colori blu
    plt.imshow(
        cm,
        cmap="Blues"
    )

    # Imposta il titolo del grafico
    plt.title(title)

    # Aggiunge una barra laterale che indica l'intensità dei valori rappresentati
    plt.colorbar()

    # Etichetta dell'asse orizzontale: classe predetta dal modello
    plt.xlabel("Predicted")

    # Etichetta dell'asse verticale: classe reale
    plt.ylabel("True")

    # Salva il grafico nel percorso specificato
    plt.savefig(save_path)

    # Chiude la figura per liberare memoria 
    plt.close()


# ============================================================
# CURVA ROC
# ============================================================

def plot_roc_curve(
    y_true,
    y_score,
    title,
    save_path
):
    """
    Crea e salva la curva ROC.

    La curva ROC mostra il rapporto tra:
    - True Positive Rate;
    - False Positive Rate.

    L'area sotto la curva viene utilizzata come misura
    riassuntiva delle prestazioni del classificatore.
    """


    # Assicura che la directory di destinazione esista
    ensure_dir(os.path.dirname(save_path))


    # Calcola:
    # - FPR: False Positive Rate
    # - TPR: True Positive Rate
    # y_score contiene le probabilità/stime del modello per la classe positiva
    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )


    # Calcola l'area sotto la curva ROC
    roc_auc = auc(
        fpr,
        tpr
    )


    # Crea una nuova figura
    plt.figure(figsize=(5, 4))


    # Disegna la curva ROC
    # Nel label viene mostrato anche il valore dell'AUC arrotondato a tre cifre decimali
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )


    # Disegna la diagonale corrispondente al comportamento di un classificatore casuale
    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.title(title)

    # Mostra la legenda contenente il valore dell'AUC
    plt.legend()

    plt.savefig(save_path)
    plt.close()


# ============================================================
# 5. SALVATAGGIO RISULTATI
# ============================================================

def save_results_csv(results, filename):
    """
    Salva i risultati in formato CSV.

    I risultati vengono prima trasformati in un DataFrame Pandas e successivamente salvati su file.
    """

    # Crea la directory contenitrice del file, se necessario
    ensure_dir(os.path.dirname(filename))


    # Converte i risultati in un DataFrame
    # .T effettua la trasposizione del DataFrame
    df = pd.DataFrame(results).T

    # Salva il DataFrame in formato CSV
    df.to_csv(filename)
    

# ============================================================
# SALVATAGGIO DELLE METRICHE IN JSON
# ============================================================

def save_metrics_json(metrics, path):
    """
    Salva le metriche in formato JSON.

    Il formato JSON permette di conservare in maniera strutturata
    i risultati della valutazione e di poterli ricaricare
    successivamente.
    """

    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:

        # Converte il dizionario delle metriche in JSON
        # indent=2 rende il file più leggibile
        # default=str permette di convertire in stringa eventuali oggetti che il modulo 
        # JSON non riesce a serializzare direttamente
        json.dump(
            metrics,
            f,
            indent=2,
            default=str
        )

    # Comunica a terminale dove sono state salvate le metriche.
    print(f"Metriche salvate in: {path}")