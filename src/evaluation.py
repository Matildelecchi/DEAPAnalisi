"""
src/evaluation.py
Valutazione dei modelli DEAP + grafici + salvataggio risultati.

Include:
- metriche complete (accuracy, f1, precision, recall, confusion matrix, ROC AUC)
- cross-validation subject-independent (GroupKFold)
- leave-one-trial-out (Koelstra)
- funzioni per grafici (confusion matrix, ROC)
- salvataggio CSV e JSON
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interattivo: evita errori se lo script gira senza display (es. da terminale/CI)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GroupKFold

from src.utils import ensure_dir


# ============================================================
# 1. VALUTAZIONE DI UN MODELLO SU TEST SET SEPARATO
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Valuta un modello già addestrato su un set di test SEPARATO dal training.
    Ritorna un dizionario di metriche (non solo accuracy).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC AUC se disponibile
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    return metrics


# ============================================================
# 2. CROSS-VALIDATION SUBJECT-INDEPENDENT (GroupKFold)
# ============================================================

def cross_validate_subject_independent(train_fn, X, y, groups, n_splits=5, **train_kwargs):
    """
    Cross-validation subject-independent: i soggetti nel fold di test non
    compaiono mai nel fold di training (GroupKFold sui subject_ids).

    train_fn: funzione tipo src.models.train_svm(X, y) -> model già addestrato
    Ritorna: lista di dict di metriche, uno per fold, e la loro media.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_fn(X_train, y_train, **train_kwargs)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    avg = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
        if key not in ("confusion_matrix", "fold")
    }
    return fold_metrics, avg


# ============================================================
# 3. LEAVE-ONE-TRIAL-OUT (Koelstra)
# ============================================================

def cross_validate_leave_one_trial_out(train_fn, X, y, trial_ids, **train_kwargs):
    """
    Validazione "alla Koelstra": leave-one-trial-out, pensata per essere
    usata SOGGETTO PER SOGGETTO (chiamare questa funzione una volta per
    ciascun soggetto, passando solo i suoi trial).
    """
    unique_trials = np.unique(trial_ids)
    fold_metrics = []

    for held_out in unique_trials:
        test_mask = trial_ids == held_out
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        model = train_fn(X[train_mask], y[train_mask], **train_kwargs)
        metrics = evaluate_model(model, X[test_mask], y[test_mask])
        fold_metrics.append(metrics)

    avg = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
        if key != "confusion_matrix"
    }
    return fold_metrics, avg


# ============================================================
# 4. GRAFICI (Confusion Matrix + ROC)
# ============================================================

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    # ensure_dir riceve la cartella CONTENITRICE di save_path (non un nome
    # fisso): così funziona qualunque percorso/sottocartella gli venga
    # passato, es. result/figures/baseline/valence_svm_cm.png
    ensure_dir(os.path.dirname(save_path))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_score, title, save_path):
    ensure_dir(os.path.dirname(save_path))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# 5. SALVATAGGIO RISULTATI (CSV + JSON)
# ============================================================

def save_results_csv(results, filename):
    ensure_dir(os.path.dirname(filename))
    df = pd.DataFrame(results).T
    df.to_csv(filename)


def save_metrics_json(metrics, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metriche salvate in: {path}")
