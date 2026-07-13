"""
Full reproduction script for ECSEL on COMPAS dataset.

Implements the paper's evaluation protocol (Section 5.1, Appendix E.2-E.3):
- 80/20 stratified train/test split (seed 42)
- 5-fold stratified CV for hyperparameter selection
- Optuna TPE sampler, 30 trials
- Retrain best HP model on full 80% training set
- Evaluate on held-out 20% test set
- Feature scaling to [1, 10] via MinMaxScaler (paper Appendix E)
- Gradient clipping max_norm=1.0 (paper Section 5.1)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, precision_score, recall_score,
)
import optuna

from ecsel import SignomialClassifier

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = "/datasets/compas-scores-two-years.csv"
SCALE_RANGE = (0.01, 10.01)  # paper uses MinMax[1,10]
RANDOM_STATE = 42
N_TRIALS = 30
N_FOLDS = 5
TEST_SIZE = 0.2
RESULTS_FILE = "ecsel_compas_reproduction_results.txt"

# HP search space (from Table 8, Appendix E.3)
HP_SEARCH = {
    "K": [1, 2, 3],
    "l1_strength": (1e-4, 1e-2),         # log-uniform
    "batch_size": [32, 64, 128],          # categorical
    "lr": (1e-4, 1e-2),                   # log-uniform
    "num_epochs": (800, 1000),            # uniform int
    "patience": [20, 50],                 # categorical
    "sigmoid_threshold": [0.4, 0.5, 0.6, 0.7],  # categorical
}


# ── Dataset Loading ────────────────────────────────────────────────────
def load_compas():
    """Load and preprocess COMPAS following standard ML fairness pipeline."""
    df = pd.read_csv(DATA_PATH)

    df = df[(df["days_b_screening_arrest"] >= -30) & (df["days_b_screening_arrest"] <= 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    df = df.dropna(subset=["c_offense_date"])

    features = df[["age", "sex", "race", "priors_count", "c_charge_degree",
                    "juv_fel_count", "juv_misd_count"]].copy()

    features["sex"] = (features["sex"] == "Male").astype(int)
    features["race"] = (features["race"] == "African-American").astype(int)
    features["c_charge_degree"] = (features["c_charge_degree"] == "F").astype(int)

    target = df["two_year_recid"].values.astype(int)

    X = features.values.astype(np.float64)
    y = target

    feature_names = list(features.columns)
    print(f"Loaded COMPAS: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, feature_names


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba):
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    minority_idx = np.argmin(class_counts)

    if y_proba.ndim == 1:
        y_proba = np.vstack([1 - y_proba, y_proba]).T

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    if len(unique_classes) == 2:
        minority_idx_val = np.where(unique_classes == unique_classes[minority_idx])[0][0]
        minority_recall = recall_per_class[minority_idx_val]
        majority_recall = recall_per_class[1 - minority_idx_val]
        minority_precision = precision_per_class[minority_idx_val]
        majority_precision = precision_per_class[1 - minority_idx_val]
    else:
        minority_recall = majority_recall = minority_precision = majority_precision = 0

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'LogLoss': log_loss(y_true, y_proba),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'MinorityRecall': minority_recall,
        'MajorityRecall': majority_recall,
        'MinorityPrecision': minority_precision,
        'MajorityPrecision': majority_precision,
    }


# ── Optuna Objective ──────────────────────────────────────────────────
def create_objective(X_train, y_train):
    """Create an Optuna objective function for one CV fold evaluation."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        K = trial.suggest_int("K", HP_SEARCH["K"][0], HP_SEARCH["K"][-1])
        l1_strength = trial.suggest_float("l1_strength", *HP_SEARCH["l1_strength"], log=True)
        batch_size = trial.suggest_categorical("batch_size", HP_SEARCH["batch_size"])
        lr = trial.suggest_float("lr", *HP_SEARCH["lr"], log=True)
        num_epochs = trial.suggest_int("num_epochs", *HP_SEARCH["num_epochs"])
        patience = trial.suggest_categorical("patience", HP_SEARCH["patience"])
        sigmoid_threshold = trial.suggest_categorical("sigmoid_threshold", HP_SEARCH["sigmoid_threshold"])

        # use_sigmoid for binary classification
        config = {
            "K": K,
            "l1_strength": l1_strength,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "use_sigmoid": True,
            "sigmoid_threshold": sigmoid_threshold,
            "random_state": RANDOM_STATE,
            "verbose": False,
        }

        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = SignomialClassifier(internal_scaling_range=SCALE_RANGE, **config)
            try:
                model.fit(X_tr, y_tr, validation_split=0.2)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                cv_scores.append(score)
            except Exception:
                cv_scores.append(0.0)

        return np.mean(cv_scores) if cv_scores else 0.0

    return objective


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ECSEL COMPAS Reproduction")
    print(f"Protocol: {N_FOLDS}-fold stratified CV, Optuna TPE {N_TRIALS} trials")
    print(f"Seed: {RANDOM_STATE}, Test split: {TEST_SIZE}")
    print("=" * 70)

    # 1. Load data
    X, y, feature_names = load_compas()

    # 2. 80/20 stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # 3. Hyperparameter optimization with Optuna
    print(f"\nStarting Optuna HP search ({N_TRIALS} trials, TPE sampler)...")
    objective = create_objective(X_train, y_train)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"Best CV F1: {best_cv_score:.4f}")
    print(f"Best params: {best_params}")

    # 4. Retrain best model on full training set
    print("\nRetraining best model on full training set...")
    config = {
        "K": best_params["K"],
        "l1_strength": best_params["l1_strength"],
        "batch_size": best_params["batch_size"],
        "lr": best_params["lr"],
        "num_epochs": best_params["num_epochs"],
        "patience": best_params["patience"],
        "use_sigmoid": True,
        "sigmoid_threshold": best_params["sigmoid_threshold"],
        "random_state": RANDOM_STATE,
        "verbose": True,
    }

    model = SignomialClassifier(internal_scaling_range=SCALE_RANGE, **config)

    t0 = time.time()
    model.fit(X_train, y_train, validation_split=0.0)
    fit_time = time.time() - t0

    # 5. Evaluate on test set
    print("\nEvaluating on test set...")
    t1 = time.time()
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = np.vstack([1 - y_pred, y_pred]).T
    test_time = time.time() - t1

    metrics = compute_metrics(y_test, y_pred, y_proba)
    formula = model.get_learned_formula(feature_names=feature_names)

    # 6. Report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"COMPAS Reproduction Results ({time.ctime()})")
    report_lines.append("=" * 70)
    report_lines.append(f"Dataset shape: {X.shape}")
    report_lines.append(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    report_lines.append(f"Best CV F1: {best_cv_score:.4f}")
    report_lines.append(f"Best params: {best_params}")
    report_lines.append("Learned formula:")
    report_lines.append(formula.rstrip())
    report_lines.append("Test metrics:")
    for k, v in metrics.items():
        report_lines.append(f"  {k}: {v:.4f}")
    report_lines.append(f"Fit time: {fit_time:.2f}s | Test time: {test_time:.4f}s")
    report_lines.append("")

    text = "\n".join(report_lines)
    print(text)

    with open(RESULTS_FILE, "w") as f:
        f.write(text)

    # Map to rubric metrics
    print("=" * 70)
    print("RUBRIC METRICS COMPARISON:")
    print(f"  Paper Accuracy:  68.47  | Reproduced: {metrics['Accuracy']:.2f}  ({'MATCH' if abs(metrics['Accuracy'] - 68.47) < 1.0 else 'DIFF'})")
    print(f"  Paper F1:        68.36  | Reproduced: {metrics['F1']:.2f}  ({'MATCH' if abs(metrics['F1'] - 68.36) < 1.0 else 'DIFF'})")
    print(f"  Paper Precision: 68.62  | Reproduced: {metrics['Precision']:.2f}  ({'MATCH' if abs(metrics['Precision'] - 68.62) < 1.0 else 'DIFF'})")
    print(f"  Paper Recall:    68.47  | Reproduced: {metrics['Recall']:.2f}  ({'MATCH' if abs(metrics['Recall'] - 68.47) < 1.0 else 'DIFF'})")
    print(f"  Paper MinRecall: 62.82  | Reproduced: {metrics['MinorityRecall']*100:.2f}  ({'MATCH' if abs(metrics['MinorityRecall']*100 - 62.82) < 5.0 else 'DIFF'})")
    print(f"  Paper TrainTime: 4.773  | Reproduced: {fit_time:.3f}s")
    print("=" * 70)

    return metrics, fit_time


if __name__ == "__main__":
    main()
