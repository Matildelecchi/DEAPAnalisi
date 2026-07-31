"""
src/models.py
Modelli di classificazione per DEAP (valence e arousal).

Pipeline tipica:

    from src.preprocessing import load_deap_dataset, segment_signal
    from src.filtering import filter_data
    from src.features import extract_features
    from src.models import evaluate_models

    X, y, subject_ids = load_deap_dataset(eeg_only=True)
    X = filter_data(X)   # opzionale ma consigliato
    X_segments, trial_idx = segment_signal(X, segment_length=15, fs=128)
    X_features, y_valence, y_arousal, subj_seg = extract_features(
        X_segments, y, trial_idx, subject_ids
    )

    results_valence = evaluate_models(X_features, y_valence, subj_seg, task_name="valence")
    results_arousal = evaluate_models(X_features, y_arousal, subj_seg, task_name="arousal")
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 1. MODELLI
# ============================================================

def get_classifiers():
    """
    Restituisce un dizionario di modelli da confrontare.
    Tutti sono pensati per classificazione binaria (0/1).
    """
    return {
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "LogReg": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# ============================================================
# 2. VALUTAZIONE MODELLI CON GROUPKFold
# ============================================================

def evaluate_models(X, y, groups, n_splits=5, task_name="valence"):
    """
    Valuta diversi modelli con GroupKFold per soggetto.

    X      : (n_segmenti, n_feature)
    y      : (n_segmenti,) label binarie (0/1)
    groups : (n_segmenti,) id soggetto per ogni segmento

    Ritorna:
        results : dict -> {
            modello: {
                acc_mean, acc_std,
                f1_mean, f1_std
            }
        }
    """

    print(f"=== Valutazione modelli per task: {task_name} ===")
    print("Shape X:", X.shape)
    print("Distribuzione label:", np.bincount(y))

    classifiers = get_classifiers()
    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for name, clf in classifiers.items():
        print(f"\n>> Modello: {name}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        acc_scores = []
        f1_scores = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            acc_scores.append(acc)
            f1_scores.append(f1)

            print(f"  Fold {fold+1}: acc={acc:.3f}, f1={f1:.3f}")

        results[name] = {
            "acc_mean": np.mean(acc_scores),
            "acc_std": np.std(acc_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
        }

        print(
            f">> {name}: "
            f"ACC={results[name]['acc_mean']:.3f}±{results[name]['acc_std']:.3f}, "
            f"F1={results[name]['f1_mean']:.3f}±{results[name]['f1_std']:.3f}"
        )

    print("\n=== Valutazione completata per task:", task_name, "===\n")
    return results


# ============================================================
# 3. MAIN (opzionale)
# ============================================================

if __name__ == "__main__":
    from src.preprocessing import load_deap_dataset, segment_signal
    from src.filtering import filter_data
    from src.features import extract_features

    print("=== Pipeline completa: preprocessing + filtering + features + modelli ===")

    X, y, subject_ids = load_deap_dataset(eeg_only=True)
    X = filter_data(X)
    X_segments, trial_idx = segment_signal(X, segment_length=15, fs=128)
    X_features, y_valence, y_arousal, subj_seg = extract_features(
        X_segments, y, trial_idx, subject_ids
    )

    results_valence = evaluate_models(X_features, y_valence, subj_seg, task_name="valence")
    results_arousal = evaluate_models(X_features, y_arousal, subj_seg, task_name="arousal")

    print("Risultati valence:", results_valence)
    print("Risultati arousal:", results_arousal)
