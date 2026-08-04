"""
src/models.py
Modelli di classificazione per DEAP (valence e arousal).

Compatibile con main.py:
- train_gaussian_nb
- train_svm
- train_knn
- train_logreg
- train_decision_tree
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. TRAINER PER I MODELLI
# ============================================================

def train_gaussian_nb(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model


def train_svm(X, y):
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X, y)
    return model


def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model


def train_logreg(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def train_decision_tree(X, y):
    model = DecisionTreeClassifier(max_depth=None)
    model.fit(X, y)
    return model
