"""Hybrid optimization: try feature intersection/union strategies."""
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

def select_hybrid_features(metrics_result, top_k, strategy):
    """Hybrid feature selection strategies."""
    diff_set = set(metrics_result.top_diff_ids)
    f1_set = set(metrics_result.top_f1_ids)

    if strategy == "intersection":
        # Features in BOTH top-diff and top-F1
        common = diff_set & f1_set
        # Rank by diff score
        ranked = sorted(common, key=lambda x: metrics_result.feature_diff[x], reverse=True)
        return ranked[:top_k]
    elif strategy == "union_first_diff":
        # Take all from intersection, then fill with top-diff
        common = diff_set & f1_set
        ranked_common = sorted(common, key=lambda x: metrics_result.feature_diff[x], reverse=True)
        remaining_diff = [x for x in metrics_result.top_diff_ids if x not in common]
        return (ranked_common + remaining_diff)[:top_k]
    elif strategy == "weighted_consensus":
        # Features ranked by average of normalized diff and F1 ranks
        all_features = list(set(metrics_result.top_diff_ids + metrics_result.top_f1_ids))
        diff_vals = np.array([metrics_result.feature_diff[f] for f in all_features])
        f1_vals = np.array([metrics_result.f1_scores[f] for f in all_features])
        # Normalize
        diff_norm = (diff_vals - diff_vals.min()) / (diff_vals.max() - diff_vals.min() + 1e-8)
        f1_norm = (f1_vals - f1_vals.min()) / (f1_vals.max() - f1_vals.min() + 1e-8)
        combined = diff_norm + f1_norm
        ranked_idx = np.argsort(combined)[::-1]
        return [all_features[i] for i in ranked_idx[:top_k]]
    else:
        return metrics_result.top_diff_ids[:top_k]

X, y = load_data()
n = len(y)
print("Data: {} samples".format(n), flush=True)

# Feature selection on ALL data
metrics_result = evaluate_features(X, y, top_k=100, batch_size=1000)

strategies = ["intersection", "union_first_diff", "weighted_consensus"]
k_values = [24, 28, 32]

for strategy in strategies:
    for K in k_values:
        selected = select_hybrid_features(metrics_result, K, strategy)
        if len(selected) < K:
            print("STRATEGY={} K={}: only {} features found (skipping)".format(strategy, K, len(selected)))
            continue

        feature_weights = np.array([float(metrics_result.feature_diff[idx]) for idx in selected])

        # Split
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_size = int(n * 0.85)
        cal_idx = indices[:cal_size]
        test_idx = indices[cal_size:]

        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        num_features = X_cal.shape[1]
        W = sp.csr_matrix((feature_weights, (np.array(selected), np.zeros_like(selected))), shape=(num_features, 1))

        cal_scores = X_cal.dot(W).toarray().flatten()
        prec, rec, thresh = precision_recall_curve(y_cal, cal_scores)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.0

        X_test_csr = X_test.tocsr() if not sp.isspmatrix_csr(X_test) else X_test
        test_scores = X_test_csr.dot(W).toarray().flatten()
        y_pred = (test_scores > best_thresh).astype(int)
        f1 = f1_score(y_test, y_pred)

        print("STRATEGY={} K={}: F1={:.4f}, n_features={}".format(strategy, K, f1, len(selected)), flush=True)
