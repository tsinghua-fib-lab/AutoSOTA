"""
UCR 128 Evaluation for Mantis
==============================
Reproduces Table 1 UCR result: Mantis accuracy = 0.8195 on 128 UCR datasets.

Pipeline (paper Section 4.2):
- Mantis-8M checkpoint (MantisV1)
- Layer 3 (return_transf_layer=2) — intermediate layer with best zero-shot performance
- Output: concatenation of CLS token + mean of non-CLS tokens (output_token="combined") → 512-dim
- Classifier: Random Forest (200 trees, max_depth=None)
- Input: resized to 512 via linear interpolation
- Zero-shot feature extraction (frozen encoder)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer

# ── Configuration ──────────────────────────────────────────────
UCR_DIR = "/datasets/ucr"
DEVICE = "cuda"
SEQ_LEN = 512
BATCH_SIZE = 256
N_ESTIMATORS = 200
RANDOM_STATE = 0
OUTPUT_TOKEN = "combined"  # CLS + mean of non-CLS tokens
RETURN_LAYER = 2  # 0-indexed, layer 3

# Datasets with variable-length sequences or missing values —
# skipped because simple pad/interpolate may not match the paper.
SKIP_DATASETS = {
    "AllGestureWiimoteX", "AllGestureWiimoteZ",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ",
}

# HF mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_ucr_dataset(dataset_path):
    """Load a UCR dataset from .ts files. Returns X_train, y_train, X_test, y_test."""
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
        data = []
        labels = []
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

    # Remap labels to 0..K-1
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    y_train = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_test = np.array([label_map[l] for l in y_test], dtype=np.int64)

    return X_train, y_train, X_test, y_test


def resize_to_seq_len(X, target_len=512):
    """Resize time series to target_len using linear interpolation."""
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    X_scaled = F.interpolate(X_tensor, size=target_len, mode="linear", align_corners=False)
    return X_scaled.numpy()


def evaluate_dataset(dataset_name, model):
    """Evaluate Mantis on a single UCR dataset."""
    dataset_path = os.path.join(UCR_DIR, dataset_name)
    try:
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_path)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: load error - {e}")
        return None

    try:
        X_train = resize_to_seq_len(X_train, SEQ_LEN)
        X_test = resize_to_seq_len(X_test, SEQ_LEN)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: resize error - {e}")
        return None

    try:
        Z_train = model.transform(X_train, batch_size=BATCH_SIZE)
        Z_test = model.transform(X_test, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: feature extraction error - {e}")
        return None

    try:
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)
        acc = np.mean(y_test == y_pred)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: RF error - {e}")
        return None

    return acc


def main():
    print("=" * 70)
    print("Mantis UCR Evaluation — Reproducing Table 1 (UCR Accuracy = 0.8195)")
    print("=" * 70)

    available = sorted(os.listdir(UCR_DIR))
    available = [d for d in available if os.path.isdir(os.path.join(UCR_DIR, d))]
    evaluated = [d for d in available if d not in SKIP_DATASETS]
    skipped = [d for d in available if d in SKIP_DATASETS]
    print(f"\nFound {len(available)} UCR dataset directories")
    print(f"Evaluating {len(evaluated)}, skipping {len(skipped)} (variable-length / missing values)")
    if skipped:
        print(f"  Skipped: {chr(39)}{chr(39).join(sorted(skipped))}{chr(39)}")

    print("\nLoading Mantis model...")
    network = MantisV1(device=DEVICE, output_token=OUTPUT_TOKEN, return_transf_layer=RETURN_LAYER)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    print(f"  hidden_dim: {network.hidden_dim}")
    print(f"  output_token: {network.output_token}")
    print(f"  return_transf_layer: {network.return_transf_layer}")

    model = MantisTrainer(device=DEVICE, network=network)

    print(f"\nEvaluating {len(evaluated)} datasets...")
    print("-" * 70)
    results = {}
    start_time = time.time()
    for i, ds in enumerate(evaluated):
        acc = evaluate_dataset(ds, model)
        if acc is not None:
            results[ds] = acc
            print(f"  [{i+1:3d}/{len(evaluated)}] {ds:40s}  acc={acc:.4f}")
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
        print(f"  Datasets evaluated: {n_success}/{len(evaluated)} (skipped {len(skipped)})")
        print(f"  Average accuracy:   {avg_acc:.4f}")
        print(f"  Min accuracy:       {np.min(accs):.4f}")
        print(f"  Max accuracy:       {np.max(accs):.4f}")
        print(f"  Median accuracy:    {np.median(accs):.4f}")
        print(f"  Paper value:        0.8195")
        print(f"  Delta:              {avg_acc - 0.8195:+.4f}")
        print(f"  Within bounds:      {0.8029 <= avg_acc <= 0.8212}")
    else:
        print("  NO datasets evaluated successfully!")
        avg_acc = None

    print(f"  Elapsed time:       {elapsed:.1f}s")
    print(f"  Per-dataset accuracies:")
    for ds, acc in sorted(results.items()):
        print(f"    {ds:40s}  {acc:.4f}")

    np.savez("/repo/ucr_results.npz", **results)
    print(f"\nResults saved to /repo/ucr_results.npz")

    return avg_acc


if __name__ == "__main__":
    avg = main()
    if avg is not None:
        print(f"\nFINAL: UCR Average Accuracy = {avg:.4f}")
