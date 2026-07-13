"""
Minimal reproduction script for the ECSEL classification results.

For each requested dataset, this script loads the data, makes a stratified
80/20 train/test split (seed 42), trains a SignomialClassifier using the
documented configuration in ``paper_configs.PAPER_CONFIGS``, and reports test
metrics together with the learned closed-form signomial formula.

This is a lightweight alternative to the full hyperparameter-search benchmark:
it does not run Optuna or cross-validation. The hyperparameters come from
``paper_configs.py``; the metrics and formula are recomputed by actually
training and evaluating the model each run (nothing is hardcoded). Results are
printed to the console and appended to a results file.

Feature scaling into the positive orthant required by the signomial structure
is handled inside the model via ``internal_scaling_range``, fit on the training
split only.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, precision_score, recall_score,
)

from ecsel import SignomialClassifier
from paper_configs import PAPER_CONFIGS
import Datasets


# ================= CONFIG =================
DATA_PATH = "./datasets/classification/"
DATASETS_TO_RUN = ["iris", "hearts", "seeds", "ilpd", "transfusion", "loan"]
SCALE_RANGE = (0.01, 10.01)
RESULTS_FILE = "ecsel_reproduce_results.txt"
# Internal validation split passed to SignomialClassifier.fit. The default of
# 0.0 trains on all data and early-stops on the training loss (reproducing the
# published experiments). Set a positive value (e.g. 0.2) to reserve an internal
# validation split for early stopping, matching the description in Appendix E.2.
VALIDATION_SPLIT = 0.0


def compute_metrics(y_true, y_pred, y_proba):
    """Compute test metrics for one set of predictions.

    Includes overall accuracy, log loss, weighted F1/precision/recall, and
    per-class minority/majority recall and precision (the latter only defined
    for binary problems; set to 0 otherwise).
    """
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    minority_idx = np.argmin(class_counts)

    if y_proba.ndim == 1:
        y_proba = np.vstack([1 - y_proba, y_proba]).T

    log_loss_val = log_loss(y_true, y_proba)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

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
        'LogLoss': log_loss_val,
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Precision': weighted_precision,
        'Recall': weighted_recall,
        'MinorityRecall': minority_recall,
        'MajorityRecall': majority_recall,
        'MinorityPrecision': minority_precision,
        'MajorityPrecision': majority_precision,
    }


def load_dataset(name, data_path):
    """Load a dataset by name from Datasets.py and return a stratified split."""
    if not hasattr(Datasets, name):
        raise ValueError(f"Dataset {name} not found in Datasets.py")
    df = getattr(Datasets, name)(data_path)
    X = df.drop(columns=["output"]).values
    y = df["output"].values
    feature_names = [c for c in df.columns if c != "output"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names


def report(dataset_name, config, metrics, formula, fit_time, test_time):
    """Print results to the console and append them to the results file."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Dataset: {dataset_name}   ({time.ctime()})")
    lines.append("=" * 70)
    lines.append("Config:")
    lines.append(f"  {config}")
    lines.append("Learned formula:")
    lines.append(formula.rstrip())
    lines.append("Test metrics:")
    for k, v in metrics.items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append(f"Fit time: {fit_time:.2f}s | Test time: {test_time:.4f}s")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "a") as f:
        f.write(text + "\n")


def run():
    for dataset_name in DATASETS_TO_RUN:
        if dataset_name not in PAPER_CONFIGS:
            print(f"[skip] No documented config for '{dataset_name}' in paper_configs.py")
            continue

        X_train, X_test, y_train, y_test, feature_names = load_dataset(dataset_name, DATA_PATH)

        config = dict(PAPER_CONFIGS[dataset_name])
        model = SignomialClassifier(internal_scaling_range=SCALE_RANGE, **config)

        t0 = time.time()
        model.fit(X_train, y_train, validation_split=VALIDATION_SPLIT)
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = np.vstack([1 - y_pred, y_pred]).T
        test_time = time.time() - t1

        metrics = compute_metrics(y_test, y_pred, y_proba)
        formula = model.get_learned_formula(feature_names=feature_names)

        report(dataset_name, config, metrics, formula, fit_time, test_time)


if __name__ == "__main__":
    run()