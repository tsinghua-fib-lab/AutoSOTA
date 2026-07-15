"""Run NExT-Guard pipeline end to end: statistical selection + combination + evaluation"""
import os, sys, yaml, json, gc
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path("/repo")
MODEL_ROOT = os.getenv("MODEL_ROOT")
SAE_ROOT = os.getenv("SAE_ROOT")
DATASET_ROOT = os.getenv("DATASET_ROOT")

# Find latest results directory
results_base = BASE_DIR / "results/Guard_Qwen3-8B_20260206_1005"
subdirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
OUTPUT_DIR = subdirs[-1]  # latest
PT_DIR = OUTPUT_DIR / "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using results directory: {OUTPUT_DIR}", flush=True)
print(f"Predictions directory: {PT_DIR}", flush=True)

# ============================================================
# STEP 1: Statistical Selection
# ============================================================
print("\n" + "="*60, flush=True)
print("STEP 1: Feature Selection on Aegis2.0", flush=True)
print("="*60, flush=True)

from sae_tools.data_loader import get_adapter
from sae_tools.model import load_sae_predictions_pt, filter_data_by_label
from sae_tools.statistical import (
    build_sentence_feature_matrix_from_sparse,
    evaluate_features,
    print_metrics_overview,
    print_top_features,
)

DATASET_NAME = "Aegis2.0"
DATASETS_CONFIG = BASE_DIR / "configs/datasets/datasets_aegis2_only.yaml"
with open(DATASETS_CONFIG, "r") as f:
    dataset_config = yaml.safe_load(f)

DATASET_PATH = None
for d in dataset_config["datasets"]:
    if d["name"] == DATASET_NAME:
        DATASET_PATH = os.path.join(DATASET_ROOT, d["folder"])
        DATA_TYPE = d["type"]
        LABEL_TYPE = f"{DATA_TYPE}_label"
        break

PT_FILE = PT_DIR / f"{DATASET_NAME}.pt"
print(f"Loading activations: {PT_FILE}", flush=True)

sparse_data = load_sae_predictions_pt(PT_FILE)
num_samples = len(sparse_data["seq_lens"])
print(f"Loaded {num_samples} samples", flush=True)

adapter = get_adapter(DATASET_NAME)
dataset = adapter.load(DATASET_PATH, num_samples)
metadata_list = [item for item in dataset]

sparse_data, data_list, valid_indices = filter_data_by_label(
    sparse_data, metadata_list, label_field=LABEL_TYPE, verbose=True
)

X = build_sentence_feature_matrix_from_sparse(sparse_data)
y_unsafe = np.array([1 if item.get(LABEL_TYPE) == "Unsafe" else 0 for item in data_list])
print(f"Feature matrix: {X.shape}, sparsity: {(1 - X.nnz/(X.shape[0]*X.shape[1]))*100:.2f}%", flush=True)

TOP_K = 32
metrics_result = evaluate_features(X, y_unsafe, top_k=TOP_K, batch_size=1000)
print_metrics_overview(metrics_result)
print_top_features(metrics_result, "high_f1", "Top 10 F1 Features")

selected_features = metrics_result.top_diff_ids[:TOP_K]
safety_features_config = {
    str(int(idx)): float(metrics_result.feature_diff[idx])
    for idx in selected_features
    if metrics_result.feature_diff[idx] > 0
}
output_json = OUTPUT_DIR / "safety_features.json"
with open(output_json, "w") as f:
    json.dump(safety_features_config, f, indent=4)
print(f"Saved {len(safety_features_config)} safety features to {output_json}", flush=True)

# Save top feature indices and scores
safety_indices = np.array([int(k) for k in safety_features_config.keys()])
safety_weights = np.array([float(v) for v in safety_features_config.values()])

# ============================================================
# STEP 3: Threshold Calibration
# ============================================================
print("\n" + "="*60, flush=True)
print("STEP 3: Threshold Calibration", flush=True)
print("="*60, flush=True)

from sklearn.metrics import precision_recall_curve, f1_score

# Use a split of Aegis2.0 for calibration
# Split data into 80% calibration / 20% test
np.random.seed(42)
n = len(y_unsafe)
indices = np.random.permutation(n)
cal_size = int(n * 0.8)
cal_idx = indices[:cal_size]
test_idx = indices[cal_size:]

print(f"Calibration samples: {len(cal_idx)}, Test samples: {len(test_idx)}", flush=True)

X_cal = X[cal_idx]
y_cal = y_unsafe[cal_idx]
X_test_csr = X[test_idx]
y_test = y_unsafe[test_idx]

# Compute risk scores
num_features = X_cal.shape[1]
W_sparse = sp.csr_matrix(
    (safety_weights, (safety_indices, np.zeros_like(safety_indices))),
    shape=(num_features, 1)
)

print("Computing calibration scores...", flush=True)
cal_scores = X_cal.dot(W_sparse).toarray().flatten()

precisions, recalls, thresholds = precision_recall_curve(y_cal, cal_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
best_f1 = f1_scores[best_idx]

print(f"Best F1 (calibration): {best_f1:.4f}", flush=True)
print(f"Optimal Threshold: {best_threshold:.6f}", flush=True)

# Save intervention config
config_export = {
    "threshold": float(best_threshold),
    "features": safety_features_config,
    "calibration_dataset": DATASET_NAME,
    "best_f1_cal": float(best_f1)
}
config_path = OUTPUT_DIR / "intervention_config.json"
with open(config_path, "w") as f:
    json.dump(config_export, f, indent=4)
print(f"Saved intervention config to {config_path}", flush=True)

# ============================================================
# STEP 4: Evaluation
# ============================================================
print("\n" + "="*60, flush=True)
print("STEP 4: Evaluation on held-out test set", flush=True)
print("="*60, flush=True)

from sklearn.metrics import classification_report

THRESHOLD = best_threshold

# Use sparse matrix for efficiency
print("Computing test scores...", flush=True)
test_scores = X_test_csr.dot(W_sparse).toarray().flatten()

y_pred = (test_scores > THRESHOLD).astype(int)

print("\nClassification Report:", flush=True)
print(classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"]), flush=True)

f1 = f1_score(y_test, y_pred)
print(f"\nF1 Score: {f1:.4f}", flush=True)

# Save results
results = {
    "f1_score": float(f1),
    "threshold": float(THRESHOLD),
    "num_features": len(safety_indices),
    "num_calibration_samples": len(cal_idx),
    "num_test_samples": len(test_idx)
}
with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\nPipeline complete! F1={f1:.4f}", flush=True)
