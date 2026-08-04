"""
main.py — Pipeline DEAPAnalisi
Autrice: Matilde

Funzionalità:
- Baseline (Koelstra 2012)
- Custom (subject-independent)
- Sovrascrittura file (metriche, modelli, grafici)
- ROC AUC corretto (NaN → 0.5)
"""

import os
import numpy as np
import warnings

from src.preprocessing import load_deap_dataset, segment_signal, normalize_signal
from src.features import extract_features
from src.models import (
    train_gaussian_nb,
    train_svm,
    train_knn,
    train_logreg,
    train_decision_tree,
)
from src.evaluation import (
    cross_validate_subject_independent,
    cross_validate_leave_one_trial_out,
    save_metrics_json,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.utils import save_model_pickle, ensure_dir


# ============================================================
# FILTRO WARNING
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# DIRECTORY
# ============================================================

RAW_DIR = "data/raw/"
BASELINE_DIR = "results/baseline/"
CUSTOM_DIR = "results/custom/"
MODELS_DIR = "results/models/"
FIGURES_DIR = "results/figures/"

for d in [BASELINE_DIR, CUSTOM_DIR, MODELS_DIR, FIGURES_DIR]:
    ensure_dir(d)


# ============================================================
# BASELINE
# ============================================================

def run_baseline(dimension="valence"):
    print(f"\n=== BASELINE ({dimension}) ===")

    X, y_raw, subject_ids = load_deap_dataset(RAW_DIR, eeg_only=True)

    segments, trial_idx = segment_signal(X, segment_length=60, overlap=0.0)

    X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
        segments, y_raw, trial_idx, subject_ids
    )

    y_seg = y_valence_seg if dimension == "valence" else y_arousal_seg

    results_per_subject = {}

    for subj in np.unique(subj_seg):
        mask = subj_seg == subj
        X_subj, y_subj = X_features[mask], y_seg[mask]

        if len(np.unique(y_subj)) < 2:
            continue

        local_trial_ids = np.arange(X_subj.shape[0])

        _, avg_metrics = cross_validate_leave_one_trial_out(
            train_gaussian_nb, X_subj, y_subj, local_trial_ids
        )

        # CORREZIONE ROC AUC (NaN → 0.5)
        if np.isnan(avg_metrics["roc_auc"]):
            avg_metrics["roc_auc"] = 0.5

        results_per_subject[subj] = avg_metrics
        print(f"  Soggetto {subj}: acc={avg_metrics['accuracy']:.3f}  f1={avg_metrics['f1']:.3f}")

        # ============================
        # GRAFICO CONFUSION MATRIX
        # ============================
        model = train_gaussian_nb(X_subj, y_subj)
        y_pred = model.predict(X_subj)

        cm_path = os.path.join(FIGURES_DIR, f"baseline_cm_{dimension}_{subj}.png")
        plot_confusion_matrix(
            y_subj,
            y_pred,
            title=f"Baseline CM - {dimension} - {subj}",
            save_path=cm_path
        )

    overall = {
        key: float(np.mean([m[key] for m in results_per_subject.values()]))
        for key in next(iter(results_per_subject.values()))
    }

    save_metrics_json(
        {"per_subject": results_per_subject, "overall": overall},
        os.path.join(BASELINE_DIR, f"baseline_{dimension}_metrics.json"),
    )

    return overall


# ============================================================
# CUSTOM
# ============================================================

def run_custom(dimension="valence"):
    print(f"\n=== CUSTOM ({dimension}) ===")

    X, y_raw, subject_ids = load_deap_dataset(RAW_DIR, eeg_only=True)

    segments, trial_idx = segment_signal(X, segment_length=15, overlap=0.0)
    segments_norm = normalize_signal(segments)

    X_features, y_valence_seg, y_arousal_seg, subj_seg = extract_features(
        segments_norm, y_raw, trial_idx, subject_ids
    )

    y_seg = y_valence_seg if dimension == "valence" else y_arousal_seg

    classifiers = {
        "svm": train_svm,
        "knn": train_knn,
        "logreg": train_logreg,
        "decision_tree": train_decision_tree,
    }

    all_results = {}

    for name, train_fn in classifiers.items():
        print(f"\n--- {name.upper()} ---")

        fold_metrics, avg = cross_validate_subject_independent(
            train_fn, X_features, y_seg, subj_seg, n_splits=5
        )

        # CORREZIONE ROC AUC (NaN → 0.5)
        if np.isnan(avg["roc_auc"]):
            avg["roc_auc"] = 0.5

        all_results[name] = avg
        print(f"  acc={avg['accuracy']:.3f}  f1={avg['f1']:.3f}")

        # ============================
        # GRAFICI (CM + ROC)
        # ============================

        model = train_fn(X_features, y_seg)
        y_pred = model.predict(X_features)

        cm_path = os.path.join(FIGURES_DIR, f"custom_cm_{dimension}_{name}.png")
        plot_confusion_matrix(
            y_seg,
            y_pred,
            title=f"Custom CM - {dimension} - {name}",
            save_path=cm_path
        )

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_features)[:, 1]
            roc_path = os.path.join(FIGURES_DIR, f"custom_roc_{dimension}_{name}.png")
            plot_roc_curve(
                y_seg,
                y_score,
                title=f"Custom ROC - {dimension} - {name}",
                save_path=roc_path
            )

    save_metrics_json(
        all_results,
        os.path.join(CUSTOM_DIR, f"custom_{dimension}_summary.json"),
    )

    # ============================
    # SALVATAGGIO MODELLO MIGLIORE
    # ============================

    best_name = max(all_results, key=lambda k: all_results[k]["f1"])
    best_model = classifiers[best_name](X_features, y_seg)

    save_model_pickle(
        best_model,
        os.path.join(MODELS_DIR, f"custom_{dimension}_{best_name}.pkl"),
    )

    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for dim in ["valence", "arousal"]:
        run_baseline(dimension=dim)
        run_custom(dimension=dim)

    print("\n=== Pipeline completata ===")
