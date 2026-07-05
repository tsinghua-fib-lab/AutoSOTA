from pathlib import Path
import os
import ast
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# --------------------------
# 1. Setup working directory
# --------------------------
# SCRIPT_DIR = Path(__file__).resolve().parent
# os.chdir(SCRIPT_DIR)

# --------------------------
# 2. Metric functions
# --------------------------
def tp(lhat, ltrue):
    return len([x for x in lhat if x in ltrue])

def fp(lhat, ltrue):
    return len([x for x in lhat if x not in ltrue])

def fn(lhat, ltrue):
    return len([x for x in ltrue if x not in lhat])

def precision(lhat, ltrue):
    t = tp(lhat, ltrue); f = fp(lhat, ltrue)
    if t == 0:
        return 0.0
    return t / (t + f)

def recall(lhat, ltrue):
    t = tp(lhat, ltrue); f = fn(lhat, ltrue)
    if t == 0:
        return 0.0
    return t / (t + f)

def f1_score(lhat, ltrue):
    # treat missing predicted set as NaN
    if lhat is None:
        return np.nan
    p = precision(lhat, ltrue)
    r = recall(lhat, ltrue)
    if p == 0 and r == 0:
        return 0.0
    if (p + r) == 0:
        return np.nan
    return 2 * p * r / (p + r)

# --------------------------
# 3. Parse python-style list column
# --------------------------
def parse_py_list(s):
    # s expected like "['A','B']" or "[]" or NA-ish
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s == "" or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
        if val is None:
            return None
        if isinstance(val, (list, tuple, set)):
            return [str(x) for x in val]
        # fallback: split on comma
        return [x.strip() for x in s.strip("[]'\"").split(",") if x.strip() != ""]
    except Exception:
        # fallback splitting
        return [x.strip() for x in s.strip("[]'\"").split(",") if x.strip() != ""]

# --------------------------
# 4. Load data helper
# --------------------------
def load_results(path_small, path_large):
    a = pd.read_csv(path_small)
    b = pd.read_csv(path_large)
    return pd.concat([a, b], ignore_index=True)

# File paths (same as R)
res_identifiable_gauss = load_results(
    "output_experiments_gaussian/final_results_identifiable_small.csv",
    "output_experiments_gaussian/final_results_identifiable_large.csv"
)
res_non_identifiable_gauss = load_results(
    "output_experiments_gaussian/final_results_nonidentifiable_small.csv",
    "output_experiments_gaussian/final_results_nonidentifiable_large.csv"
)
res_identifiable_bin = load_results(
    "output_experiments_binary/final_results_identifiable_small.csv",
    "output_experiments_binary/final_results_identifiable_large.csv"
)
res_non_identifiable_bin = load_results(
    "output_experiments_binary/final_results_nonidentifiable_small.csv",
    "output_experiments_binary/final_results_nonidentifiable_large.csv"
)

# --------------------------
# 5. Compute F1 and parse lists
# --------------------------
def compute_f1(df):
    df = df.copy()
    df["adjustment_set_vec"] = df["adjustment_set"].apply(parse_py_list)
    df["true_parents_vec"] = df["true_parents"].apply(parse_py_list)
    df["f1_score"] = df.apply(
        lambda row: f1_score(row["adjustment_set_vec"], row["true_parents_vec"]), axis=1
    )
    return df

res_identifiable_gauss = compute_f1(res_identifiable_gauss)
res_identifiable_bin = compute_f1(res_identifiable_bin)

# --------------------------
# 6. Plotting parameters
# --------------------------
# colors normalized 0-1
sorbonne_colors = {
    "LocPC-CDE": (234/255, 67/255, 40/255),
    "LDECC":     (3/255, 40/255, 89/255),
    "PC":        (100/255, 190/255, 230/255),
    "CMB":       (50/255, 150/255, 50/255),
    "MBbyMB":    (180/255, 120/255, 200/255),
}

sorbonne_markers = {
    "LocPC-CDE": "^",
    "PC": "s",
    "LDECC": "o",
    "CMB": "D",
    "MBbyMB": "X",
}

method_levels = ["PC", "CMB", "MBbyMB", "LDECC", "LocPC-CDE"]

# --------------------------
# 7. Summarisation helpers
# --------------------------
def recode_method(m):
    if pd.isna(m):
        return m
    m0 = str(m).lower()
    mapping = {
        "locpc": "LocPC-CDE",
        "pc": "PC",
        "ldecc": "LDECC",
        "cmb": "CMB",
        "mbbymb": "MBbyMB",
        "mbbymb": "MBbyMB",  # redundancy
        "mbbymb": "MBbyMB"
    }
    return mapping.get(m0, m)

def summarise_metric(df, value_col, ident_label, log_y=False):
    d = df.copy()
    d["method"] = d["method"].apply(recode_method)
    d["value"] = d[value_col]
    # ensure dag_size is treated as categorical but keep as-is for x labels
    grouped = d.groupby(["dag_size", "method"], dropna=False)
    summary = grouped["value"].agg(
        n="count",
        mean_val=lambda x: np.nanmean(x),
        std=lambda x: np.nanstd(x, ddof=1)
    ).reset_index()
    summary["se"] = summary.apply(lambda r: (r["std"] / math.sqrt(r["n"])) if r["n"]>0 else np.nan, axis=1)
    summary["lower"] = summary["mean_val"] - 1.96 * summary["se"]
    summary["lower"] = summary["lower"].clip(lower=0)
    summary["upper"] = summary["mean_val"] + 1.96 * summary["se"]
    summary["method"] = pd.Categorical(summary["method"], categories=method_levels, ordered=True)
    summary["type"] = ident_label
    return summary

def summarise_prop(df, label):
    return summarise_metric(df, "identifiability", label)

def summarise_ci(df, label):
    return summarise_metric(df, "nb_CI_tests", label, log_y=True)

def summarise_f1(df, label):
    return summarise_metric(df, "f1_score", label)

def summarise_prop_nonid(df, label):
    d = df.copy()
    # ensure identifiability boolean / numeric to boolean
    d["non_identifiable"] = ~d["identifiability"].astype(bool)
    return summarise_metric(d, "non_identifiable", label)

# --------------------------
# 8. Summarise results
# --------------------------
# Gaussian
prop_id_gauss = summarise_prop(res_identifiable_gauss, "Identifiable")
prop_nonid_gauss = summarise_prop_nonid(res_non_identifiable_gauss, "Non-identifiable")
ci_id_gauss = summarise_ci(res_identifiable_gauss, "Identifiable")
ci_nonid_gauss = summarise_ci(res_non_identifiable_gauss, "Non-identifiable")
f1_id_gauss = summarise_f1(res_identifiable_gauss, "Identifiable")

# Binary
prop_id_bin = summarise_prop(res_identifiable_bin, "Identifiable")
prop_nonid_bin = summarise_prop_nonid(res_non_identifiable_bin, "Non-identifiable")
ci_id_bin = summarise_ci(res_identifiable_bin, "Identifiable")
ci_nonid_bin = summarise_ci(res_non_identifiable_bin, "Non-identifiable")
f1_id_bin = summarise_f1(res_identifiable_bin, "Identifiable")

# --------------------------
# 9. Plotting helper
# --------------------------
sns.set(style="whitegrid")
def plot_metric(ax, df, y_col, y_label, log_y=False, title=""):
    # df expected summary table with columns: dag_size, method, mean_val, lower, upper
    methods = method_levels
    all_dag_sizes = sorted(df["dag_size"].unique(), key=lambda x: float(x) if pd.api.types.is_numeric_dtype(type(x)) else str(x))
    x = np.arange(len(all_dag_sizes))
    ax.set_title(title)
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        # align values to all_dag_sizes
        sub_idx = [all_dag_sizes.index(v) for v in sub["dag_size"]]
        mean_vals = np.array(sub["mean_val"], dtype=float)
        lower = np.array(sub["lower"], dtype=float)
        upper = np.array(sub["upper"], dtype=float)
        ax.plot(sub_idx, mean_vals, label=method, color=sorbonne_colors.get(method), marker=sorbonne_markers.get(method), linewidth=0.9)
        ax.fill_between(sub_idx, lower, upper, color=sorbonne_colors.get(method), alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(all_dag_sizes, rotation=45, ha="right")
    ax.set_xlabel("DAG size")
    ax.set_ylabel(y_label)
    if y_label == "TPR (%)":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0,1)
    elif y_label == "F1 Score":
        ax.set_ylim(0,1)
    if log_y:
        ax.set_yscale("log")

# --------------------------
# 10. plots
# --------------------------

# Identifiable figure: 2 rows x 3 cols (Gaussian row, Binary row)
print("PLotting identifiable results")
fig, axs = plt.subplots(2, 3, figsize=(6,6), constrained_layout=True)
plot_metric(axs[0,0], ci_id_gauss, "mean_val", "# CI tests", log_y=True, title="Gaussian SCM")
plot_metric(axs[0,1], prop_id_gauss, "mean_val", "TPR (%)", title="")
plot_metric(axs[0,2], f1_id_gauss, "mean_val", "F1 Score", title="")
plot_metric(axs[1,0], ci_id_bin, "mean_val", "# CI tests", log_y=True, title="Binary SCM")
plot_metric(axs[1,1], prop_id_bin, "mean_val", "TPR (%)", title="")
plot_metric(axs[1,2], f1_id_bin, "mean_val", "F1 Score", title="")

# common legend at bottom
handles, labels = axs[0,0].get_legend_handles_labels()
if not handles:
    # build custom legend handles
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=sorbonne_colors[m], marker=sorbonne_markers[m], lw=1) for m in method_levels]
    labels = method_levels
fig.legend(handles, labels, loc="lower center", ncol=len(method_levels), frameon=False)
# adjust space to fit legend
fig.subplots_adjust(bottom=0.18)

plt.show()

# Non-identifiable figure: 2 rows x 2 cols
print("PLotting non-identifiable results")
fig2, axs2 = plt.subplots(2, 2, figsize=(6,6), constrained_layout=True)
plot_metric(axs2[0,0], ci_nonid_gauss, "mean_val", "# CI tests", log_y=True, title="Gaussian SCM")
plot_metric(axs2[0,1], prop_nonid_gauss, "mean_val", "TPR (%)", title="")
plot_metric(axs2[1,0], ci_nonid_bin, "mean_val", "# CI tests", log_y=True, title="Binary SCM")
plot_metric(axs2[1,1], prop_nonid_bin, "mean_val", "TPR (%)", title="")

handles2, labels2 = axs2[0,0].get_legend_handles_labels()
if not handles2:
    from matplotlib.lines import Line2D
    handles2 = [Line2D([0],[0], color=sorbonne_colors[m], marker=sorbonne_markers[m], lw=1) for m in method_levels]
    labels2 = method_levels
fig2.legend(handles2, labels2, loc="lower center", ncol=len(method_levels), frameon=False)
fig2.subplots_adjust(bottom=0.18)
plt.show()

