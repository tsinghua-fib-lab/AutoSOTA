"""
Deterministic evaluation script for ECSEL on COMPAS.
Uses the best hyperparameters found by Optuna to train and evaluate in one shot.
This produces the exact same results as the full pipeline for the baseline metric.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, precision_score, recall_score,
)
from ecsel import SignomialClassifier

DATA_PATH = "/datasets/compas-scores-two-years.csv"
SCALE_RANGE = (0.01, 10.01)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Best HP from Optuna TPE 30 trials (reproduction run)
BEST_CONFIG = {
    "K": 5,
    "l1_strength": 0.001623616273384131,
    "batch_size": 64,
    "lr": 0.0017516096124643734,
    "num_epochs": 1200,
    "patience": 50,
    "use_sigmoid": True,
    "sigmoid_threshold": 0.5,
    "n_restarts": 3,
    "random_state": 42,
    "verbose": False,
}


def load_compas():
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

    X = features.values.astype(np.float64)
    y = df["two_year_recid"].values.astype(int)
    feature_names = list(features.columns)
    return X, y, feature_names


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
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'MinorityRecall': minority_recall,
    }


def main():
    X, y, feature_names = load_compas()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = SignomialClassifier(internal_scaling_range=SCALE_RANGE, **BEST_CONFIG)
    t0 = time.time()
    model.fit(X_train, y_train, validation_split=0.0)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = np.vstack([1 - y_pred, y_pred]).T

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("=" * 60)
    print("ECSEL COMPAS Evaluation")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Training time (s): {fit_time:.3f}")
    formula = model.get_learned_formula(feature_names=feature_names)
    print("Learned formula:")
    print(formula)


if __name__ == "__main__":
    main()
