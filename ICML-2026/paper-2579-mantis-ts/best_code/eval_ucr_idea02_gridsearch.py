"""
IDEA-02: RF GridSearchCV hyperparameter tuning
================================================
Same feature extraction as baseline, but replaces fixed RandomForest(n=200)
with GridSearchCV(RandomForestClassifier(), param_grid={...}, cv=3).

Expected gain: +0.002 to +0.005 UCR_Accuracy
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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

RF_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}


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


def evaluate_dataset(dataset_name, model):
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
        n_samples = len(y_train)
        cv_folds = min(3, n_samples) if n_samples >= 6 else n_samples
        if cv_folds < 2:
            # Too few samples for CV, use fixed RF
            clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
        else:
            clf = GridSearchCV(
                RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
                param_grid=RF_PARAM_GRID,
                cv=min(3, cv_folds),
                scoring='accuracy',
                n_jobs=-1,
            )
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)
        acc = np.mean(y_test == y_pred)
    except Exception as e:
        print(f"  [SKIP] {dataset_name}: RF error - {e}")
        return None

    return acc


def main():
    print("=" * 70)
    print("IDEA-02: Mantis UCR — RF GridSearchCV Hyperparameter Tuning")
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
        print(f"  Datasets evaluated: {n_success}/{len(evaluated)}")
        print(f"  Average accuracy:   {avg_acc:.4f}")
        print(f"  Min accuracy:       {np.min(accs):.4f}")
        print(f"  Max accuracy:       {np.max(accs):.4f}")
        print(f"  Median accuracy:    {np.median(accs):.4f}")
        print(f"  Baseline (RF200):   0.8190")
        print(f"  Delta vs baseline:  {avg_acc - 0.8190:+.4f}")
    else:
        print("  NO datasets evaluated successfully!")
        avg_acc = None

    print(f"  Elapsed time:       {elapsed:.1f}s")
    return avg_acc


if __name__ == "__main__":
    avg = main()
    if avg is not None:
        print(f"\nFINAL: UCR Average Accuracy = {avg:.4f}")
