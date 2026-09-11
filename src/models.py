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
Questo modulo contiene i modelli di Machine Learning utilizzati per classificare le emozioni nel dataset DEAP.
Il problema di classificazione riguarda due dimensioni emotive:
- valence
- arousal

Le label vengono trasformate precedentemente in due classi:
    0 -> valore inferiore a 5
    1 -> valore maggiore o uguale a 5

I modelli implementati sono:
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

Ogni funzione train_* riceve:
    X -> matrice delle feature
    y -> label corrispondenti
e restituisce un modello già addestrato.
"""


# ============================================================
# IMPORT DEI MODELLI
# ============================================================

# GaussianNB implementa il classificatore Gaussian Naive Bayes
# È un classificatore probabilistico che assume una distribuzione gaussiana delle feature
from sklearn.naive_bayes import GaussianNB

# SVC implementa la Support Vector Machine
# La SVM cerca un iperpiano che permetta di separare nel modo migliore le classi del problema
from sklearn.svm import SVC

# KNeighborsClassifier implementa il metodo K-Nearest Neighbors
# La classificazione viene effettuata considerando i campioni più vicini al nuovo punto
from sklearn.neighbors import KNeighborsClassifier

# LogisticRegression implementa la regressione logistica
# Nonostante il nome "regressione", viene utilizzata anche per problemi di classificazione binaria
from sklearn.linear_model import LogisticRegression

# DecisionTreeClassifier implementa un albero decisionale.
# Il modello prende decisioni attraverso una sequenza di condizioni sulle feature
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. TRAINER PER GAUSSIAN NAIVE BAYES
# ============================================================

def train_gaussian_nb(X, y):
    """
    Addestra un classificatore Gaussian Naive Bayes.

    Parametri:
        X: matrice delle feature utilizzate per il training.
        y: label associate ai campioni.

    Restituisce:
        Il modello Gaussian Naive Bayes già addestrato.
    """

    # Crea un nuovo classificatore Gaussian Naive Bayes.
    model = GaussianNB()

    # Addestra il modello utilizzando:
    # - X -> feature di input
    # - y -> classi corrette associate ai campioni
    model.fit(
        X,
        y
    )

    # Restituisce il modello addestrato.
    return model


# ============================================================
# 2. TRAINER PER SUPPORT VECTOR MACHINE (SVM)
# ============================================================

def train_svm(X, y):
    """
    Addestra un classificatore Support Vector Machine.

    La SVM utilizza un kernel RBF per poter rappresentare anche relazioni non lineari tra le feature.

    Restituisce:
        Il modello SVM già addestrato.
    """

    # Crea il classificatore SVM.

    # kernel="rbf": utilizza il Radial Basis Function kernel.
    # Questo permette al modello di gestire separazioni non lineari tra le classi

    # C=1.0: controlla il compromesso tra:
    # - errore di classificazione
    # - complessità del modello
    # Un valore maggiore di C tende a penalizzare maggiormente gli errori sul training set

    # gamma="scale": permette a scikit-learn di determinare automaticamente
    # un valore appropriato per il parametro gamma in base alle feature

    # probability=True: abilita la stima delle probabilità delle classi
    # Questo è importante nel progetto perché permette successivamente
    # di utilizzare predict_proba() per il calcolo della ROC curve e della ROC AUC
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True
    )

    # Addestra il modello sui dati disponibili
    model.fit(
        X,
        y
    )

    # Restituisce il modello SVM addestrato
    return model


# ============================================================
# 3. TRAINER PER K-NEAREST NEIGHBORS (KNN)
# ============================================================

def train_knn(X, y):
    """
    Addestra un classificatore K-Nearest Neighbors.

    Il modello classifica un campione in base alle classi dei suoi vicini più prossimi.

    In questo caso vengono considerati 5 vicini.
    """

    # Crea il classificatore KNN
    # n_neighbors=5 significa che per classificare un nuovo campione vengono 
    # considerati i 5 campioni più vicini nello spazio delle feature
    model = KNeighborsClassifier(
        n_neighbors=5
    )

    # Addestra il modello utilizzando i dati disponibili
    model.fit(
        X,
        y
    )

    # Restituisce il modello addestrato
    return model


# ============================================================
# 4. TRAINER PER REGRESSIONE LOGISTICA
# ============================================================

def train_logreg(X, y):
    """
    Addestra un classificatore basato sulla regressione logistica.

    La regressione logistica è adatta al problema perché le label sono binarie: 0 oppure 1.
    """

    # Crea il classificatore di regressione logistica.
    # max_iter=1000 aumenta il numero massimo di iterazioni consentite dall'algoritmo di ottimizzazione
    # Questo aiuta ad evitare che il modello raggiunga il limite di iterazioni prima della convergenza,
    # soprattutto quando il numero di feature è elevato
    model = LogisticRegression(
        max_iter=1000
    )

    # Addestra il modello sui dati di training
    model.fit(
        X,
        y
    )

    # Restituisce il modello addestrato
    return model


# ============================================================
# 5. TRAINER PER DECISION TREE
# ============================================================

def train_decision_tree(X, y):
    """
    Addestra un classificatore Decision Tree.

    Un albero decisionale classifica i dati attraverso una sequenza di decisioni basate sui valori delle feature.
    """

    # Crea il classificatore Decision Tree
    # max_depth=None significa che non viene imposto un limite massimo alla profondità dell'albero
    # L'albero può quindi continuare a creare suddivisioni fino a quando vengono 
    # soddisfatti i criteri di crescita previsti dal modello
    model = DecisionTreeClassifier(
        max_depth=None
    )

    # Addestra l'albero decisionale utilizzando le feature X e le label y
    model.fit(
        X,
        y
    )

    # Restituisce il modello addestrato
    return model