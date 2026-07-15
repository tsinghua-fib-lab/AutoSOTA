"""Proper CV: feature selection on calibration set only, then evaluate on test."""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path("/repo")
sys.path.insert(0, str(BASE_DIR / "src"))

from sae_tools.data_loader import get_adapter
from sae_tools.model import load_sae_predictions_pt, filter_data_by_label
from sae_tools.statistical import (
    build_sentence_feature_matrix_from_sparse,
    evaluate_features,
)
from sklearn.metrics import precision_recall_curve, f1_score

PT_FILE = Path("/repo/results/Guard_Qwen3-8B_20260206_1005/Guard_Qwen3-8B_20260714_1533/predictions/Aegis2.0.pt")
DATASET_PATH = os.path.join(os.getenv("DATASET_ROOT", "/datasets"), "Aegis-AI-Content-Safety-Dataset-2.0")

def load_data():
    sparse_data = load_sae_predictions_pt(PT_FILE)
    num_samples = len(sparse_data["seq_lens"])
    adapter = get_adapter("Aegis2.0")
    dataset = adapter.load(DATASET_PATH, num_samples)
    metadata_list = [item for item in dataset]
    sparse_data, data_list, _ = filter_data_by_label(sparse_data, metadata_list, label_field="prompt_label", verbose=False)
    X = build_sentence_feature_matrix_from_sparse(sparse_data)
    y = np.array([1 if item.get("prompt_label") == "Unsafe" else 0 for item in data_list])
    return X, y

X, y = load_data()
n = len(y)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=22)
parser.add_argument("--seeds", type=str, default="42,123,456")
parser.add_argument("--cal-ratio", type=float, default=0.8)
args = parser.parse_args()

seeds = [int(s.strip()) for s in args.seeds.split(",")]
f1_vals = []

for seed in seeds:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    cal_size = int(n * args.cal_ratio)
    cal_idx = indices[:cal_size]
    test_idx = indices[cal_size:]

    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Feature selection on CALIBRATION SET ONLY (proper methodology)
    metrics_result = evaluate_features(X_cal, y_cal, top_k=args.top_k, batch_size=1000)
    selected = metrics_result.top_diff_ids[:args.top_k]
    weights = np.array([float(metrics_result.feature_diff[idx]) for idx in selected])

    num_features = X_cal.shape[1]
    W = sp.csr_matrix((weights, (np.array(selected), np.zeros_like(selected))), shape=(num_features, 1))

    cal_scores = X_cal.dot(W).toarray().flatten()
    prec, rec, thresh = precision_recall_curve(y_cal, cal_scores)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.0

    X_test_csr = X_test.tocsr() if not sp.isspmatrix_csr(X_test) else X_test
    test_scores = X_test_csr.dot(W).toarray().flatten()
    y_pred = (test_scores > best_thresh).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_vals.append(f1)
    print("Seed {}: F1={:.4f}".format(seed, f1), flush=True)

mean_f1 = np.mean(f1_vals)
std_f1 = np.std(f1_vals)
print("\nProper CV (K={}, cal_ratio={}): F1 = {:.4f} +/- {:.4f}".format(args.top_k, args.cal_ratio, mean_f1, std_f1))
print("F1 Score: {:.4f}".format(mean_f1))
