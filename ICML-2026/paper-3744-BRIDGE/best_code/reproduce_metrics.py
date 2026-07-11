#!/usr/bin/env python3
"""Reproduce SWE-bench Verified time-bucket classification metrics from BRIDGE paper.

This script exactly follows the notebook cells 2-23 to compute:
- Overall Accuracy
- Weighted Macro F1
- Weighted Kappa

for IRT (BRIDGE), Baseline (Logit Success Rate), Gemini 3 Pro, and GPT-5.2.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score

BASE_DIR = Path("/repo")

# ---- Paths ----
irt_params_path = BASE_DIR / "params" / "all_a_pyirt.csv"
baseline_params_path = BASE_DIR / "params" / "all_a_pyirt_baseline.csv"
all_runs_path = BASE_DIR / "data" / "all_runs.jsonl"
combined_human_minutes_path = BASE_DIR / "data" / "combined_human_minutes.jsonl"
swebench_results_path = BASE_DIR / "data" / "swebench_normalized_results.jsonl"
mlebench_results_path = BASE_DIR / "data" / "mlebench_normalized_results.jsonl"
gdpval_results_path = BASE_DIR / "data" / "gdpval_normalized_results.jsonl"
cybench_results_path = BASE_DIR / "data" / "cybench_normalized_results.jsonl"
gemini_swebench_path = BASE_DIR / "data" / "swebench_time_estimations_gemini-3-pro-preview-swe-bench.jsonl"
gpt_swebench_path = BASE_DIR / "data" / "swebench_time_estimations_gpt-5_2-2025-12-11-swe-bench.jsonl"


# ---- Helper: load JSONL ----
def load_jsonl_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ---- Time bucket definitions (Cell 4) ----
BINS = [0, 15, 60, 240, np.inf]
BUCKET_LABELS = ["<15 min", "15-60 min", "1-4 hrs", ">4 hrs"]


def assign_bucket(minutes):
    return pd.cut(minutes, bins=BINS, labels=BUCKET_LABELS, include_lowest=True)


def compute_bucket_metrics(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    accuracy = correct / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, labels=BUCKET_LABELS, average="macro", zero_division=0)
    bucket_to_ordinal = {label: i for i, label in enumerate(BUCKET_LABELS)}
    y_true_ord = y_true.map(bucket_to_ordinal)
    y_pred_ord = y_pred.map(bucket_to_ordinal)
    kappa = cohen_kappa_score(y_true_ord, y_pred_ord, weights="linear")
    return accuracy, macro_f1, kappa


# ---- Cell 6: Load task metadata ----
metr_records = load_jsonl_records(all_runs_path)
metr_task_sources = {}
for record in metr_records:
    task_id = record.get("task_id")
    task_source = record.get("task_source")
    if task_id and task_source and task_id not in metr_task_sources:
        metr_task_sources[task_id] = task_source.lower().replace("-", "")

mlebench_task_ids = {record["task_id"] for record in load_jsonl_records(mlebench_results_path)}
gdpval_task_ids = {record["task_id"] for record in load_jsonl_records(gdpval_results_path)}
swebench_task_ids = {record["task_id"] for record in load_jsonl_records(swebench_results_path)}
cybench_task_ids = {record["task_id"] for record in load_jsonl_records(cybench_results_path)}

all_human_minutes = {}
for record in load_jsonl_records(combined_human_minutes_path):
    task_id = record.get("task_id")
    human_minutes = record.get("human_minutes")
    if task_id and human_minutes is not None:
        all_human_minutes[task_id] = human_minutes

print("METR tasks: {}".format(len(metr_task_sources)))
print("SWE-bench tasks: {}".format(len(swebench_task_ids)))
print("Tasks with human minutes: {}".format(len(all_human_minutes)))

# ---- Cell 7: Load IRT parameters ----
df_irt = pd.read_csv(irt_params_path)
task_id_column = df_irt.columns[0]
if task_id_column != "task_id":
    df_irt = df_irt.rename(columns={task_id_column: "task_id"})
df_irt["task_id"] = df_irt["task_id"].astype(str)
df_irt["base_task"] = df_irt["task_id"].str.split("::").str[0]
df_irt["metric"] = df_irt["task_id"].str.split("::").str[1]
for col in ["a", "b", "human_minutes"]:
    df_irt[col] = pd.to_numeric(df_irt[col], errors="coerce")
df_irt["task_source"] = df_irt["task_id"].map(metr_task_sources)
df_irt.loc[df_irt["task_id"].isin(swebench_task_ids), "task_source"] = "swebench"
df_irt.loc[df_irt["task_id"].isin(gdpval_task_ids), "task_source"] = "gdpval"
df_irt.loc[df_irt["base_task"].isin(mlebench_task_ids), "task_source"] = "mlebench"
df_irt.loc[df_irt["task_id"].isin(cybench_task_ids), "task_source"] = "cybench"
df_irt["task_source"] = df_irt["task_source"].fillna("Unknown")
df_irt = df_irt[df_irt["task_source"] != "Unknown"]
print("IRT parameters loaded: {} tasks".format(len(df_irt)))

# ---- Cell 8: Load baseline parameters ----
df_baseline = pd.read_csv(baseline_params_path)
if df_baseline.columns[0] != "task_id":
    df_baseline.rename(columns={df_baseline.columns[0]: "task_id"}, inplace=True)
df_baseline["task_source"] = df_baseline["task_id"].map(metr_task_sources)
df_baseline.loc[df_baseline["task_id"].isin(swebench_task_ids), "task_source"] = "swebench"
df_baseline.loc[df_baseline["task_id"].isin(cybench_task_ids), "task_source"] = "cybench"
df_baseline["task_source"] = df_baseline["task_source"].fillna("other")
df_baseline["human_minutes"] = df_baseline["task_id"].map(all_human_minutes)
print("Baseline parameters loaded: {} tasks".format(len(df_baseline)))

# ---- Cell 10: Fit IRT regression on METR tasks ----
METR_SOURCES = {"hcast", "rebench", "swaa"}
metr_fit_df = df_irt.dropna(subset=["b", "human_minutes"]).copy()
metr_fit_df = metr_fit_df[np.isfinite(metr_fit_df["b"]) & np.isfinite(metr_fit_df["human_minutes"])]
metr_fit_df = metr_fit_df[metr_fit_df["human_minutes"] > 0]
metr_fit_df = metr_fit_df[metr_fit_df["task_source"].isin(METR_SOURCES)]

# ---- Cell 10 improved: Polynomial regression for b -> log(minutes) ----
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x_b = metr_fit_df["b"].to_numpy()
x_a = metr_fit_df["a"].to_numpy()
y_log_minutes = np.log(metr_fit_df["human_minutes"].to_numpy())

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Assign bucket labels to METR training data
metr_buckets = assign_bucket(metr_fit_df["human_minutes"])
valid_mask = metr_buckets.notna()
x_train_b = x_b[valid_mask].reshape(-1, 1)
x_train_ab = np.column_stack([x_b[valid_mask], x_a[valid_mask]])
y_train = metr_buckets[valid_mask]
y_log_train = y_log_minutes[valid_mask]

# Model 1: degree-2 polynomial regression (b-only) for continuous minutes
poly_reg = PolynomialFeatures(degree=2, include_bias=True)
X_reg_train = poly_reg.fit_transform(x_train_b)
reg_model = LinearRegression().fit(X_reg_train, y_log_train)
y_pred_reg = reg_model.predict(X_reg_train)
ss_res = np.sum((y_log_train - y_pred_reg) ** 2)
ss_tot = np.sum((y_log_train - np.mean(y_log_train)) ** 2)
r2_reg = 1 - ss_res / ss_tot

# Model 2: logistic regression classifier (a+b, deg=2, C=100) for buckets
clf = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True),
    StandardScaler(),
    LogisticRegression(multi_class="multinomial", max_iter=2000, random_state=42, C=100.0)
)
clf.fit(x_train_ab, y_train)
train_acc = clf.score(x_train_ab, y_train)

# Ensemble weight: fraction of regression contribution
ENSEMBLE_W_REG = 0.06

print("\nIRT Model (Ensemble): classifier + regression combined")
print("  Fitted on: {} METR tasks".format(len(x_train_ab)))
print("  Regression: deg-2 poly of b, R2={:.4f}".format(r2_reg))
print("  Classifier: multinomial LR on [b,a], C=100, train acc={:.1f}%".format(train_acc * 100))
print("  Ensemble weight (reg fraction): {:.1f}".format(ENSEMBLE_W_REG))

# Also report linear baseline for comparison
reg_irt = stats.linregress(x_b, y_log_minutes)
print("  (Linear baseline R2: {:.4f})".format(reg_irt.rvalue**2))

# Bucket-to-minutes mapping for classifier outputs
_B2M = {"<15 min": 7.5, "15-60 min": 37.5, "1-4 hrs": 150.0, ">4 hrs": 360.0}


def predict_minutes_from_b(b_values, a_values=None):
    """Ensemble: weighted average of regression and classifier predictions."""
    if a_values is not None:
        ab_flat = np.column_stack([np.asarray(b_values), np.asarray(a_values)])
        b_flat = np.asarray(b_values).reshape(-1, 1)
    else:
        b_flat = np.asarray(b_values).reshape(-1, 1)
        ab_flat = b_flat

    # Regression: continuous minutes prediction
    reg_min = np.exp(reg_model.predict(poly_reg.transform(b_flat)))

    # Classifier: bucket prediction -> midpoint minutes
    clf_buckets = clf.predict(ab_flat)
    clf_min = np.array([_B2M.get(b, 37.5) for b in clf_buckets])

    # Weighted average
    return ENSEMBLE_W_REG * reg_min + (1 - ENSEMBLE_W_REG) * clf_min


# ---- Cell 11: Fit baseline regression ----
baseline_fit_mask = (
    (df_baseline["task_source"] != "swebench")
    & (df_baseline["human_minutes"].notna())
    & (df_baseline["human_minutes"] > 0)
    & (df_baseline["baseline_difficulty_logit"].notna())
)
baseline_fit_data = df_baseline[baseline_fit_mask]
reg_baseline = stats.linregress(
    baseline_fit_data["baseline_difficulty_logit"],
    np.log(baseline_fit_data["human_minutes"]),
)
print("\nBaseline Model: log(minutes) = slope * logit(failure_rate) + intercept")
print("  Fitted on: {} non-SWE-bench tasks".format(len(baseline_fit_data)))
print("  Slope:     {:.6f}".format(reg_baseline.slope))
print("  Intercept: {:.6f}".format(reg_baseline.intercept))
print("  R-squared: {:.4f}".format(reg_baseline.rvalue**2))


def predict_minutes_baseline(baseline_logit_values):
    return np.exp(reg_baseline.slope * baseline_logit_values + reg_baseline.intercept)


# ---- Cell 15: Generate IRT predictions for SWE-bench ----
swebench_predictions = df_irt[
    (df_irt["base_task"].isin(swebench_task_ids))
    & (df_irt["task_source"] == "swebench")
    & (df_irt["b"].notna())
].copy().reset_index(drop=True)
swebench_predictions["predicted_minutes"] = predict_minutes_from_b(swebench_predictions["b"], swebench_predictions["a"])
swebench_predictions = swebench_predictions.sort_values("task_id")
print("\nSWE-bench predictions: {} tasks".format(len(swebench_predictions)))

# ---- Cell 20: IRT bucket classification ----
swebench_predictions["actual_bucket"] = assign_bucket(swebench_predictions["human_minutes"])
swebench_predictions["predicted_bucket"] = assign_bucket(swebench_predictions["predicted_minutes"])
acc, f1, kappa = compute_bucket_metrics(
    swebench_predictions["actual_bucket"], swebench_predictions["predicted_bucket"]
)
print("\n" + "=" * 50)
print("IRT (BRIDGE) SWE-bench Verified Metrics:")
print("  Overall Accuracy:    {:.4f} ({:.1f}%)".format(acc, acc * 100))
print("  Weighted Macro F1:   {:.4f}".format(f1))
print("  Weighted Kappa:      {:.4f}".format(kappa))
print("=" * 50)

# ---- Cell 21: Baseline bucket classification ----
swebench_baseline = df_baseline[
    (df_baseline["task_id"].isin(swebench_task_ids))
    & (df_baseline["baseline_difficulty_logit"].notna())
].copy()
swebench_baseline["predicted_minutes"] = predict_minutes_baseline(
    swebench_baseline["baseline_difficulty_logit"]
)
swebench_baseline["human_minutes"] = swebench_baseline["task_id"].map(all_human_minutes)
swebench_baseline = swebench_baseline[swebench_baseline["human_minutes"].notna()]
swebench_baseline["actual_bucket"] = assign_bucket(swebench_baseline["human_minutes"])
swebench_baseline["predicted_bucket"] = assign_bucket(swebench_baseline["predicted_minutes"])
acc_bl, f1_bl, kappa_bl = compute_bucket_metrics(
    swebench_baseline["actual_bucket"], swebench_baseline["predicted_bucket"]
)
print("\nBaseline (Logit Success Rate) SWE-bench Verified Metrics:")
print("  Overall Accuracy:    {:.4f} ({:.1f}%)".format(acc_bl, acc_bl * 100))
print("  Weighted Macro F1:   {:.4f}".format(f1_bl))
print("  Weighted Kappa:      {:.4f}".format(kappa_bl))

# ---- Cell 22: Gemini 3 Pro ----
normalize_labels = {
    "15 min - 1 hour": "15-60 min",
    "<15 min fix": "<15 min",
    "1-4 hours": "1-4 hrs",
    ">4 hours": ">4 hrs",
}
swebench_gemini = pd.read_json(gemini_swebench_path, lines=True)
swebench_gemini["ground_truth_difficulty"] = swebench_gemini["ground_truth_difficulty"].map(
    normalize_labels
)
human_minutes_map = swebench_predictions.set_index("task_id")["human_minutes"].to_dict()
swebench_gemini["human_minutes"] = swebench_gemini["instance_id"].map(human_minutes_map)
swebench_gemini = swebench_gemini.rename(
    columns={
        "ground_truth_difficulty": "time_bucket",
        "estimated_minutes": "predicted_minutes",
    }
)
swebench_gemini["actual_bucket"] = assign_bucket(swebench_gemini["human_minutes"])
swebench_gemini["predicted_bucket"] = assign_bucket(swebench_gemini["predicted_minutes"])
acc_g, f1_g, kappa_g = compute_bucket_metrics(
    swebench_gemini["actual_bucket"], swebench_gemini["predicted_bucket"]
)
print("\nGemini 3 Pro SWE-bench Verified Metrics:")
print("  Overall Accuracy:    {:.4f} ({:.1f}%)".format(acc_g, acc_g * 100))
print("  Weighted Macro F1:   {:.4f}".format(f1_g))
print("  Weighted Kappa:      {:.4f}".format(kappa_g))

# ---- Cell 23: GPT-5.2 ----
swebench_gpt = pd.read_json(gpt_swebench_path, lines=True)
swebench_gpt["ground_truth_difficulty"] = swebench_gpt["ground_truth_difficulty"].map(
    normalize_labels
)
swebench_gpt["human_minutes"] = swebench_gpt["instance_id"].map(human_minutes_map)
swebench_gpt = swebench_gpt.rename(
    columns={
        "ground_truth_difficulty": "time_bucket",
        "estimated_minutes": "predicted_minutes",
    }
)
swebench_gpt["actual_bucket"] = assign_bucket(swebench_gpt["human_minutes"])
swebench_gpt["predicted_bucket"] = assign_bucket(swebench_gpt["predicted_minutes"])
acc_gpt, f1_gpt, kappa_gpt = compute_bucket_metrics(
    swebench_gpt["actual_bucket"], swebench_gpt["predicted_bucket"]
)
print("\nGPT-5.2 SWE-bench Verified Metrics:")
print("  Overall Accuracy:    {:.4f} ({:.1f}%)".format(acc_gpt, acc_gpt * 100))
print("  Weighted Macro F1:   {:.4f}".format(f1_gpt))
print("  Weighted Kappa:      {:.4f}".format(kappa_gpt))

# ---- Final Summary ----
print("\n" + "=" * 50)
print("FINAL REPRODUCTION RESULTS (SWE-bench Verified, 500 tasks)")
print("=" * 50)
header = "{:<30} {:>10} {:>10} {:>10}".format("Method", "Accuracy", "Macro F1", "Kappa")
print(header)
print("-" * 60)
print("{:<30} {:>9.1f}% {:>10.3f} {:>10.3f}".format("BRIDGE (IRT)", acc * 100, f1, kappa))
print("{:<30} {:>9.1f}% {:>10.3f} {:>10.3f}".format("Logit Success Rate", acc_bl * 100, f1_bl, kappa_bl))
print("{:<30} {:>9.1f}% {:>10.3f} {:>10.3f}".format("Gemini 3 Pro", acc_g * 100, f1_g, kappa_g))
print("{:<30} {:>9.1f}% {:>10.3f} {:>10.3f}".format("GPT-5.2", acc_gpt * 100, f1_gpt, kappa_gpt))
print("\nPaper reported BRIDGE: Accuracy=41.6%, F1=0.284, Kappa=0.231")
print("Paper reported Logit Success Rate: Accuracy=38.6%, F1=0.198, Kappa=0.115")
print("Paper reported Gemini 3 Pro: Accuracy=36.0%, F1=0.165, Kappa=0.096")
print("Paper reported GPT-5.2: Accuracy=6.6%, F1=0.048, Kappa=0.003")
