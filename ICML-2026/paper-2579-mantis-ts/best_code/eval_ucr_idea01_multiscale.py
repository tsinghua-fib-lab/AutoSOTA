"""
IDEA-01: Multi-scale self-ensembling
======================================
Extract frozen embeddings at multiple interpolation scales (128, 256, 512, 1024).
Two variants tested: concatenation (2048-dim) and averaging (512-dim).
Uses multi-classifier with CV selection (best from IDEA-03).

Expected gain: +0.005 to +0.010 UCR_Accuracy
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer

UCR_DIR = "/datasets/ucr"
DEVICE = "cuda"
SCALES = [128, 256, 512, 1024]
BATCH_SIZE = 256
RANDOM_STATE = 0
OUTPUT_TOKEN = "combined"
RETURN_LAYER = 2  # layer 3
FUSION_MODE = "concat"  # "concat" or "average"

SKIP_DATASETS = {
    "AllGestureWiimoteX", "AllGestureWiimoteZ",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ",
}

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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


def resize_to_seq_len(X, target_len):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    X_scaled = F.interpolate(X_tensor, size=target_len, mode="linear", align_corners=False)
    return X_scaled.numpy()


def extract_multiscale_embeddings(model, X, scales, batch_size, fusion="concat"):
    """Extract embeddings at multiple scales and fuse them."""
    all_embeddings = []
    for scale in scales:
        X_scaled = resize_to_seq_len(X, scale)
        Z = model.transform(X_scaled, batch_size=batch_size)
        all_embeddings.append(Z)

    if fusion == "concat":
        return np.concatenate(all_embeddings, axis=1)
    elif fusion == "average":
        return np.mean(np.stack(all_embeddings, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")


def evaluate_dataset(dataset_name, model, classifier_counts, fusion_mode):
    dataset_path = os.path.join(UCR_DIR, dataset_name)
    try:
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_path)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: load error - {e}")
        return None, None

    try:
        Z_train = extract_multiscale_embeddings(model, X_train, SCALES, BATCH_SIZE, fusion_mode)
        Z_test = extract_multiscale_embeddings(model, X_test, SCALES, BATCH_SIZE, fusion_mode)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: feature extraction error - {e}")
        return None, None

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
    print(f"IDEA-01: Mantis UCR — Multi-scale Self-ensembling [{FUSION_MODE.upper()}]")
    print(f"  Scales: {SCALES}, Embed dim: {'2048' if FUSION_MODE == 'concat' else '512'}")
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
        acc, clf_name = evaluate_dataset(ds, model, clf_choices, FUSION_MODE)
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
        print(f"  Best so far (IDEA-12): 0.8437")
        print(f"  Delta vs best:      {avg_acc - 0.8437:+.4f}")
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
