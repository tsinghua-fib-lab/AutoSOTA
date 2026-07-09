"""
IDEA-12: Statistical features concatenation + multi-classifier
================================================================
Adds 6 per-series statistical features (mean, std, min, max, skew, kurtosis)
from raw pre-normalized time series to 512-dim embedding -> 518-dim.
Uses multi-classifier with CV selection (best from IDEA-03).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer

UCR_DIR = "/datasets/ucr"
DEVICE = "cuda"
SEQ_LEN = 512
BATCH_SIZE = 256
RANDOM_STATE = 0
OUTPUT_TOKEN = "combined"
RETURN_LAYER = 2  # layer 3

SKIP_DATASETS = {
    "AllGestureWiimoteX", "AllGestureWiimoteZ",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ",
}

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def compute_stat_features(X):
    """Compute 6 per-series statistical features from raw time series."""
    features = []
    for x in X:
        feats = [
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            sp_stats.skew(x),
            sp_stats.kurtosis(x),
        ]
        features.append(feats)
    return np.array(features, dtype=np.float32)


def load_ucr_dataset(dataset_path):
    train_file = None
    test_file = None
    for f in os.listdir(dataset_path):
        if f.endswith("_TRAIN.ts"):
            train_file = os.path.join(dataset_path, f)
        elif f.endswith("_TEST.ts"):
            test_file = os.path.join(dataset_path, f)
    if train_file is None or test_file is None:
        raise FileNotFoundError(f"Missing TRAIN/TEST .ts files in {dataset_path}")

    def _parse_ts(filepath):
        data, labels = [], []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("@"):
                    continue
                parts = line.rsplit(":", 1)
                values = np.array([float(v) for v in parts[0].split(",")], dtype=np.float32)
                label = float(parts[1])
                data.append(values)
                labels.append(label)
        return np.array(data), np.array(labels, dtype=np.int64)

    X_train, y_train = _parse_ts(train_file)
    X_test, y_test = _parse_ts(test_file)
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    y_train = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_test = np.array([label_map[l] for l in y_test], dtype=np.int64)
    return X_train, y_train, X_test, y_test


def resize_to_seq_len(X, target_len=512):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    X_scaled = F.interpolate(X_tensor, size=target_len, mode="linear", align_corners=False)
    return X_scaled.numpy()


def evaluate_dataset(dataset_name, model, classifier_counts):
    dataset_path = os.path.join(UCR_DIR, dataset_name)
    try:
        X_train_raw, y_train, X_test_raw, y_test = load_ucr_dataset(dataset_path)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: load error - {e}")
        return None, None

    # Compute statistical features BEFORE interpolation (on raw data)
    try:
        stat_train = compute_stat_features(X_train_raw)
        stat_test = compute_stat_features(X_test_raw)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: stat features error - {e}")
        return None, None

    try:
        X_train = resize_to_seq_len(X_train_raw, SEQ_LEN)
        X_test = resize_to_seq_len(X_test_raw, SEQ_LEN)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: resize error - {e}")
        return None, None

    try:
        Z_train = model.transform(X_train, batch_size=BATCH_SIZE)
        Z_test = model.transform(X_test, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: feature extraction error - {e}")
        return None, None

    # Concatenate statistical features to embeddings
    Z_train = np.concatenate([Z_train, stat_train], axis=1)
    Z_test = np.concatenate([Z_test, stat_test], axis=1)

    # Standardize
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_test_scaled = scaler.transform(Z_test)

    n_samples = len(y_train)
    cv_folds = min(3, n_samples) if n_samples >= 6 else min(2, n_samples)

    candidates = []
    if n_samples >= 10:
        candidates.append(("RF", RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)))
    for C_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((f"LR(C={C_val})", LogisticRegression(
            C=C_val, max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)))
    for alpha_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((f"Ridge(a={alpha_val})", RidgeClassifier(
            alpha=alpha_val, random_state=RANDOM_STATE)))

    best_cv = -1
    best_clf_name = "none"
    for name, clf in candidates:
        try:
            if cv_folds >= 2 and n_samples >= cv_folds * 2:
                cv_scores = cross_val_score(clf, Z_train_scaled, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                cv_acc = np.mean(cv_scores)
            else:
                clf.fit(Z_train_scaled, y_train)
                cv_acc = clf.score(Z_train_scaled, y_train)
        except Exception:
            continue
        if cv_acc > best_cv:
            best_cv = cv_acc
            best_clf_name = name

    try:
        best_found = False
        for name, clf in candidates:
            if name == best_clf_name:
                clf.fit(Z_train_scaled, y_train)
                y_pred = clf.predict(Z_test_scaled)
                test_acc = np.mean(y_test == y_pred)
                best_found = True
                break
        if not best_found:
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
            clf.fit(Z_train_scaled, y_train)
            y_pred = clf.predict(Z_test_scaled)
            test_acc = np.mean(y_test == y_pred)
            best_clf_name = "RF(fallback)"

        classifier_counts[best_clf_name] = classifier_counts.get(best_clf_name, 0) + 1
        return test_acc, best_clf_name
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: classifier error - {e}")
        return None, None


def main():
    print("=" * 70)
    print("IDEA-12: Mantis UCR — Statistical Features + Multi-classifier")
    print("=" * 70)

    available = sorted(os.listdir(UCR_DIR))
    available = [d for d in available if os.path.isdir(os.path.join(UCR_DIR, d))]
    evaluated = [d for d in available if d not in SKIP_DATASETS]
    skipped = [d for d in available if d in SKIP_DATASETS]
    print(f"\nFound {len(available)} UCR dataset directories")
    print(f"Evaluating {len(evaluated)}, skipping {len(skipped)}")

    print("\nLoading Mantis model...")
    network = MantisV1(device=DEVICE, output_token=OUTPUT_TOKEN, return_transf_layer=RETURN_LAYER)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device=DEVICE, network=network)

    print(f"\nEvaluating {len(evaluated)} datasets...")
    print("-" * 70)
    results = {}
    clf_choices = {}
    start_time = time.time()
    for i, ds in enumerate(evaluated):
        acc, clf_name = evaluate_dataset(ds, model, clf_choices)
        if acc is not None:
            results[ds] = acc
            print(f"  [{i+1:3d}/{len(evaluated)}] {ds:40s}  acc={acc:.4f}  [{clf_name}]")
        else:
            print(f"  [{i+1:3d}/{len(evaluated)}] {ds:40s}  FAILED")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    n_success = len(results)
    if n_success > 0:
        accs = list(results.values())
        avg_acc = np.mean(accs)
        print(f"  Datasets evaluated: {n_success}/{len(evaluated)}")
        print(f"  Average accuracy:   {avg_acc:.4f}")
        print(f"  Min accuracy:       {np.min(accs):.4f}")
        print(f"  Max accuracy:       {np.max(accs):.4f}")
        print(f"  Median accuracy:    {np.median(accs):.4f}")
        print(f"  Baseline (RF200):   0.8190")
        print(f"  Best so far (IDEA-03): 0.8419")
        print(f"  Delta vs best:      {avg_acc - 0.8419:+.4f}")
    else:
        print("  NO datasets evaluated successfully!")
        avg_acc = None

    print(f"  Elapsed time:       {elapsed:.1f}s")
    print(f"\n  Classifier selection counts:")
    for name, count in sorted(clf_choices.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count} datasets")

    return avg_acc


if __name__ == "__main__":
    avg = main()
    if avg is not None:
        print(f"\nFINAL: UCR Average Accuracy = {avg:.4f}")
