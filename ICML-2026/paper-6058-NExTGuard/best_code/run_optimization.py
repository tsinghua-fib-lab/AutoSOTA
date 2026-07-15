"""Optimization harness for NExT-Guard - matches original pipeline methodology.

Key: feature selection on ALL data first, then split for calibration/test.
This matches the paper's reproduction methodology.
"""
import os, sys, json, argparse
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
from sklearn.metrics import precision_recall_curve, f1_score, classification_report

# Fixed paths
PT_FILE = Path("/repo/results/Guard_Qwen3-8B_20260206_1005/Guard_Qwen3-8B_20260714_1533/predictions/Aegis2.0.pt")
DATASET_NAME = "Aegis2.0"
DATASET_ROOT_PATH = os.getenv("DATASET_ROOT", "/datasets")
DATASET_PATH = os.path.join(DATASET_ROOT_PATH, "Aegis-AI-Content-Safety-Dataset-2.0")
LABEL_TYPE = "prompt_label"

def load_data():
    sparse_data = load_sae_predictions_pt(PT_FILE)
    num_samples = len(sparse_data["seq_lens"])
    adapter = get_adapter(DATASET_NAME)
    dataset = adapter.load(DATASET_PATH, num_samples)
    metadata_list = [item for item in dataset]
    sparse_data, data_list, _ = filter_data_by_label(
        sparse_data, metadata_list, label_field=LABEL_TYPE, verbose=False
    )
    X = build_sentence_feature_matrix_from_sparse(sparse_data)
    y_unsafe = np.array([1 if item.get(LABEL_TYPE) == "Unsafe" else 0 for item in data_list])
    return X, y_unsafe

def run_evaluation(X, y_unsafe, top_k, feature_method, seed, cal_ratio, threshold_method, weight_mode="diff"):
    """Run evaluation matching original pipeline: feature selection on ALL data, then split."""
    n = len(y_unsafe)

    # STEP 1: Feature selection on ALL data (matches original pipeline)
    metrics_result = evaluate_features(X, y_unsafe, top_k=top_k, batch_size=1000)

    # Select features
    method_map = {
        "diff": metrics_result.top_diff_ids,
        "f1": metrics_result.top_f1_ids,
        "precision": metrics_result.top_precision_ids,
        "recall": metrics_result.top_recall_ids,
        "pareto": metrics_result.pareto_front_ids,
    }
    if feature_method not in method_map:
        raise ValueError("Unknown feature_method: " + feature_method)
    selected_features = method_map[feature_method][:top_k]

    # Compute weights based on weight_mode
    if weight_mode == "diff":
        feature_weights = np.array([float(metrics_result.feature_diff[idx]) for idx in selected_features])
    elif weight_mode == "f1":
        feature_weights = np.array([float(metrics_result.f1_scores[idx]) for idx in selected_features])
    elif weight_mode == "uniform":
        feature_weights = np.ones(len(selected_features))
    elif weight_mode == "precision":
        feature_weights = np.array([float(metrics_result.precisions[idx]) for idx in selected_features])
    elif weight_mode == "recall":
        feature_weights = np.array([float(metrics_result.recalls[idx]) for idx in selected_features])
    else:
        feature_weights = np.ones(len(selected_features))

    # STEP 2: Split for calibration/evaluation
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    cal_size = int(n * cal_ratio)
    cal_idx = indices[:cal_size]
    test_idx = indices[cal_size:]

    X_cal = X[cal_idx]
    y_cal = y_unsafe[cal_idx]
    X_test = X[test_idx]
    y_test = y_unsafe[test_idx]

    # Build weight matrix
    num_features = X_cal.shape[1]
    W_sparse = sp.csr_matrix(
        (feature_weights, (np.array(selected_features), np.zeros_like(selected_features))),
        shape=(num_features, 1)
    )

    # STEP 3: Threshold calibration
    cal_scores = X_cal.dot(W_sparse).toarray().flatten()

    if threshold_method == "precision_recall_curve":
        precisions, recalls, thresholds = precision_recall_curve(y_cal, cal_scores)
        f1_scores_arr = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores_arr)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
        best_f1_cal = f1_scores_arr[best_idx]
    else:
        raise ValueError("Unknown threshold_method: " + threshold_method)

    # STEP 4: Test evaluation
    X_test_csr = X_test.tocsr() if not sp.isspmatrix_csr(X_test) else X_test
    test_scores = X_test_csr.dot(W_sparse).toarray().flatten()
    y_pred = (test_scores > best_threshold).astype(int)

    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"], output_dict=True)

    return {
        "seed": seed,
        "f1": float(f1),
        "threshold": float(best_threshold),
        "f1_cal": float(best_f1_cal),
        "precision_safe": float(report["Safe"]["precision"]),
        "recall_safe": float(report["Safe"]["recall"]),
        "precision_unsafe": float(report["Unsafe"]["precision"]),
        "recall_unsafe": float(report["Unsafe"]["recall"]),
        "num_cal": int(len(cal_idx)),
        "num_test": int(len(test_idx)),
        "selected_features": [int(x) for x in selected_features[:10]],
    }

def main():
    parser = argparse.ArgumentParser(description="NExT-Guard Optimization Harness")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--feature-method", type=str, default="diff",
                        choices=["diff", "f1", "precision", "recall", "pareto"])
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--cal-ratio", type=float, default=0.8)
    parser.add_argument("--threshold-method", type=str, default="precision_recall_curve",
                        choices=["precision_recall_curve"])
    parser.add_argument("--weight-mode", type=str, default="diff",
                        choices=["diff", "f1", "uniform", "precision", "recall"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=== NExT-Guard Optimization ===", flush=True)
    print("TOP_K={}, method={}, seeds={}, cal_ratio={}, weight={}".format(
        args.top_k, args.feature_method, seeds, args.cal_ratio, args.weight_mode), flush=True)

    X, y_unsafe = load_data()
    print("Data: {} samples, {} features, {} unsafe".format(
        X.shape[0], X.shape[1], int(np.sum(y_unsafe))), flush=True)

    all_results = []
    f1_values = []
    for seed in seeds:
        print("\n--- Seed {} ---".format(seed), flush=True)
        result = run_evaluation(X, y_unsafe, args.top_k, args.feature_method,
                                seed, args.cal_ratio, args.threshold_method, args.weight_mode)
        all_results.append(result)
        f1_values.append(result["f1"])
        print("F1={:.4f}, threshold={:.4f}, cal_F1={:.4f}".format(
            result["f1"], result["threshold"], result["f1_cal"]), flush=True)

    f1_mean = float(np.mean(f1_values))
    f1_std = float(np.std(f1_values))

    summary = {
        "config": {
            "top_k": args.top_k,
            "feature_method": args.feature_method,
            "seeds": seeds,
            "cal_ratio": args.cal_ratio,
            "threshold_method": args.threshold_method,
            "weight_mode": args.weight_mode,
        },
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "f1_values": f1_values,
        "per_seed": all_results,
    }

    print("\n=== SUMMARY: F1 = {:.4f} +/- {:.4f} ===".format(f1_mean, f1_std), flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print("Results saved to {}".format(args.output), flush=True)

    print("\nF1 Score: {:.4f}".format(f1_mean), flush=True)
    return summary

if __name__ == "__main__":
    main()
