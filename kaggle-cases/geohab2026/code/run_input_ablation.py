"""
Input Data Ablation Study — E1 (DER), E2 (BS), E3 (BSDER)

Post-competition study replicating Calvert et al. (2015) input-data ablation on
Refuge Cove. Three variants of the lock candidate tabular pipeline differing
only in feature subset:

  E1 DER   : 45 BATHY + 1 SPATIAL (dist_to_shore)              = 46 features
  E2 BS    : 41 BS    + 1 SPATIAL (dist_to_shore)              = 42 features
  E3 BSDER : full 87 features (identical to Sub 41 tabular path)

Model/hyperparams/folds frozen — only the feature columns change.

Run: conda run -n wbw python run_input_ablation.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from mlxtend.evaluate import mcnemar, mcnemar_table
import mlxtend
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

ROOT    = Path(__file__).resolve().parent
TODAY   = datetime.now().strftime("%Y-%m-%d")
STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "outputs" / "input_ablation"
OOF_DIR = OUT_DIR / "oof_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)

# ── CLASS INDEX MAP ───────────────────────────────────────────────────────────
CLASS_NAMES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
LABEL_MAP   = {c: i for i, c in enumerate(CLASS_NAMES)}
ALG_IDX     = LABEL_MAP["ALG"]
assert ALG_IDX == 0, f"ALG index changed! Got {ALG_IDX}, expected 0. Abort."

# ── FROZEN HYPERPARAMS (identical to Sub 41 / Sub 42 tabular path) ───────────
LGB_PARAMS = dict(
    n_estimators=1000, verbose=-1, n_jobs=-1, random_state=42,
    num_leaves=31, max_depth=-1, learning_rate=0.1, min_child_samples=20,
    feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
    reg_alpha=0.0, reg_lambda=0.0,
    class_weight={0: 2.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
)

# ── INPUTS ────────────────────────────────────────────────────────────────────
V07_CSV    = ROOT / "data/features/train_features_v07.csv"
V09_CSV    = ROOT / "data/features/train_features_v09.csv"
FOLD_FILE  = ROOT / "kaggle_dataset/fold_indices.csv"
PHASE0_CSV = OUT_DIR / "phase0_feature_classification.csv"
SPLICE_COL = "acr_sapa_w21"

# Lock candidate registered CV (Sub 41 tabular, 150m BlockKFold weighted F1)
E3_EXPECTED_F1 = 0.7510
E3_TOLERANCE   = 1e-3


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def quantity_allocation_disagreement(y_true, y_pred, n_classes):
    """Pontius & Millones (2011) decomposition of total disagreement."""
    p_true = np.array([(y_true == i).mean() for i in range(n_classes)])
    p_pred = np.array([(y_pred == i).mean() for i in range(n_classes)])
    quantity   = np.abs(p_true - p_pred).sum() / 2.0
    total      = 1.0 - accuracy_score(y_true, y_pred)
    allocation = total - quantity
    return float(quantity), float(allocation), float(total)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Loading data")
v07 = pd.read_csv(V07_CSV)
v09_splice = pd.read_csv(V09_CSV, usecols=["x", "y", SPLICE_COL])

assert len(v07) == 6256, f"Expected 6256 rows, got {len(v07)}"
v07_feat_cols = [c for c in v07.columns if c not in ("x", "y", "class")]
assert len(v07_feat_cols) == 86, f"Expected 86 v07 features, got {len(v07_feat_cols)}"

if not (np.allclose(v07["x"].to_numpy(), v09_splice["x"].to_numpy(), atol=1e-4) and
        np.allclose(v07["y"].to_numpy(), v09_splice["y"].to_numpy(), atol=1e-4)):
    raise ValueError("v07 vs v09 coordinate mismatch")
print("  v07 vs v09 alignment: PASS")

train_labels_str = v07["class"].to_numpy()
train_labels_int = np.array([LABEL_MAP[c] for c in train_labels_str])

# Assemble the union (87-feature matrix in canonical order: v07 86 + acr_sapa_w21)
all_feat_names = v07_feat_cols + [SPLICE_COL]
X_all = np.column_stack([
    v07[v07_feat_cols].to_numpy(),
    v09_splice[SPLICE_COL].to_numpy(),
])
assert X_all.shape == (6256, 87)

# NaN inventory (LGB native — no imputation). GLCM patches near nodata edges
# contain NaN by design (CLAUDE.md GLCM spec). Pre-existing in v07; present when
# Sub 41 scored CV 0.7510. Logged, not blocking.
nan_per_col = np.isnan(X_all).sum(axis=0)
nan_inventory = {all_feat_names[i]: int(nan_per_col[i])
                 for i in range(87) if nan_per_col[i] > 0}
print(f"  NaN inventory ({len(nan_inventory)} cols with NaN, LGB native):")
for col, n in nan_inventory.items():
    print(f"    {col:32s} {n:5d} / 6256")

# ── LOAD PHASE 0 CLASSIFICATION ───────────────────────────────────────────────
phase0 = pd.read_csv(PHASE0_CSV)
assert len(phase0) == 87, f"phase0 CSV has {len(phase0)} rows; expected 87"
assert set(phase0["column_name"]) == set(all_feat_names), \
    "phase0 classification does not match feature matrix columns"

# Order phase0 by all_feat_names so indices align with X_all columns
phase0 = phase0.set_index("column_name").loc[all_feat_names].reset_index()

bathy_cols   = phase0.loc[phase0["source_category"] == "BATHY",   "column_name"].tolist()
bs_cols      = phase0.loc[phase0["source_category"] == "BS",      "column_name"].tolist()
spatial_cols = phase0.loc[phase0["source_category"] == "SPATIAL", "column_name"].tolist()

assert len(bathy_cols)   == 45, f"BATHY count {len(bathy_cols)} != 45"
assert len(bs_cols)      == 41, f"BS count {len(bs_cols)} != 41"
assert len(spatial_cols) == 1,  f"SPATIAL count {len(spatial_cols)} != 1"

# ── VARIANT FEATURE LISTS (canonical v07 column order preserved) ─────────────
# LightGBM split tiebreaks are sensitive to column order at fixed seed.
# Lock candidate Sub 41 uses v07_feat_cols + [acr_sapa_w21] = all_feat_names.
# Each variant DROPS off-stream columns; never reorders.
bathy_set   = set(bathy_cols)
bs_set      = set(bs_cols)
spatial_set = set(spatial_cols)

variants = {
    "e1_der":   [c for c in all_feat_names if c in bathy_set or c in spatial_set],
    "e2_bs":    [c for c in all_feat_names if c in bs_set    or c in spatial_set],
    "e3_bsder": list(all_feat_names),
}
for v, cols in variants.items():
    expected = {"e1_der": 46, "e2_bs": 42, "e3_bsder": 87}[v]
    assert len(cols) == expected, f"{v}: {len(cols)} != {expected}"

# Column-name → index in X_all
col_to_idx = {c: i for i, c in enumerate(all_feat_names)}
variant_X = {v: X_all[:, [col_to_idx[c] for c in cols]] for v, cols in variants.items()}

# ── LOAD FOLDS ────────────────────────────────────────────────────────────────
fold_df = pd.read_csv(FOLD_FILE).sort_values("point_index").reset_index(drop=True)
assert len(fold_df) == len(v07)
assert (fold_df["point_index"].to_numpy() == np.arange(len(v07))).all()
folds_150 = fold_df["fold"].to_numpy()
unique_folds = np.sort(np.unique(folds_150))
fold_sizes = {int(f): int((folds_150 == f).sum()) for f in unique_folds}
print(f"  Folds: {len(unique_folds)}, sizes: {fold_sizes}")

# ── CV RUNNER ─────────────────────────────────────────────────────────────────
def run_cv(X, params, tag):
    print(f"\n[{datetime.now():%H:%M:%S}] CV 150m — {tag}  ({X.shape[1]} features)")
    oof = np.zeros((len(X), 5), dtype=float)
    fold_f1s = []
    for fold_id in unique_folds:
        tr_idx = np.where(folds_150 != fold_id)[0]
        va_idx = np.where(folds_150 == fold_id)[0]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr_idx], train_labels_int[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])
        fold_f1 = f1_score(train_labels_int[va_idx], oof[va_idx].argmax(1),
                           average="weighted")
        fold_f1s.append(float(fold_f1))
        print(f"  fold {fold_id}: n_val={len(va_idx)}, F1={fold_f1:.4f}")
    return oof, fold_f1s


# ── RUN ALL VARIANTS ──────────────────────────────────────────────────────────
results = {}
for variant in ("e1_der", "e2_bs", "e3_bsder"):
    X_v = variant_X[variant]
    oof, fold_f1s = run_cv(X_v, LGB_PARAMS, variant)

    y_pred = oof.argmax(1)
    wf1    = float(f1_score(train_labels_int, y_pred, average="weighted"))
    kappa  = float(cohen_kappa_score(train_labels_int, y_pred))
    acc    = float(accuracy_score(train_labels_int, y_pred))
    per_class = {
        c: float(f1_score(train_labels_int, y_pred,
                          labels=[LABEL_MAP[c]], average="weighted"))
        for c in CLASS_NAMES
    }
    quantity, allocation, total = quantity_allocation_disagreement(
        train_labels_int, y_pred, n_classes=5)

    cm = confusion_matrix(train_labels_int, y_pred, labels=list(range(5)))

    results[variant] = {
        "n_features":              X_v.shape[1],
        "weighted_f1":             wf1,
        "cohen_kappa":             kappa,
        "accuracy":                acc,
        "quantity_disagreement":   quantity,
        "allocation_disagreement": allocation,
        "total_disagreement":      total,
        "per_class_f1":            per_class,
        "fold_f1s":                fold_f1s,
        "fold_f1_mean":            float(np.mean(fold_f1s)),
        "fold_f1_se":              float(np.std(fold_f1s, ddof=1) / np.sqrt(len(fold_f1s))),
        "confusion_matrix":        cm.tolist(),
        "y_pred":                  y_pred,
        "oof":                     oof,
    }

    print(f"  → OOF weighted F1 = {wf1:.4f}   kappa = {kappa:.4f}   acc = {acc:.4f}")
    print(f"    quantity_dis = {quantity:.4f}   allocation_dis = {allocation:.4f}")

# ── INTEGRITY CHECK: E3 must match Sub 41 registered CV ──────────────────────
e3_f1 = results["e3_bsder"]["weighted_f1"]
e3_delta = abs(e3_f1 - E3_EXPECTED_F1)
print(f"\n[{datetime.now():%H:%M:%S}] E3 integrity check")
print(f"  E3 weighted F1   : {e3_f1:.4f}")
print(f"  Expected (Sub 41): {E3_EXPECTED_F1:.4f}")
print(f"  Delta            : {e3_delta:.6f}  (tol={E3_TOLERANCE})")
if e3_delta > E3_TOLERANCE:
    raise RuntimeError(
        f"E3 mismatch with lock candidate Sub 41 OOF F1 (got {e3_f1:.6f}, "
        f"expected {E3_EXPECTED_F1:.4f} ± {E3_TOLERANCE}). Abort."
    )
print("  Integrity check  : PASS")

# ── SAVE OOF ──────────────────────────────────────────────────────────────────
for variant in ("e1_der", "e2_bs", "e3_bsder"):
    np.save(OOF_DIR / f"oof_{variant}.npy", results[variant]["oof"])
    print(f"  Saved oof_{variant}.npy  shape={results[variant]['oof'].shape}")

oof_meta = {
    "timestamp":         STAMP,
    "label_map":         LABEL_MAP,
    "class_names":       CLASS_NAMES,
    "random_state":      42,
    "n_train":           int(len(v07)),
    "fold_file":         str(FOLD_FILE.relative_to(ROOT).as_posix()),
    "fold_file_sha256":  sha256_of_file(FOLD_FILE),
    "fold_sizes":        fold_sizes,
    "lgb_params":        {k: v for k, v in LGB_PARAMS.items()},
    "splice_col":        SPLICE_COL,
    "splice_nan_count":  int(nan_per_col[-1]),
    "nan_inventory":     nan_inventory,
    "variants": {
        v: {"n_features": len(cols), "features": cols}
        for v, cols in variants.items()
    },
    "library_versions": {
        "lightgbm": lgb.__version__,
        "sklearn":  sklearn.__version__,
        "mlxtend":  mlxtend.__version__,
        "numpy":    np.__version__,
        "pandas":   pd.__version__,
    },
}
(OOF_DIR / "oof_meta.json").write_text(
    json.dumps(oof_meta, indent=2), encoding="utf-8")
print(f"  Saved oof_meta.json")

# ── MAIN RESULTS TABLES ───────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {
        "variant":                  v,
        "n_features":               results[v]["n_features"],
        "weighted_f1":              results[v]["weighted_f1"],
        "cohen_kappa":              results[v]["cohen_kappa"],
        "accuracy":                 results[v]["accuracy"],
        "quantity_disagreement":    results[v]["quantity_disagreement"],
        "allocation_disagreement":  results[v]["allocation_disagreement"],
        "total_disagreement":       results[v]["total_disagreement"],
        "fold_f1_mean":             results[v]["fold_f1_mean"],
        "fold_f1_se":               results[v]["fold_f1_se"],
    }
    for v in ("e1_der", "e2_bs", "e3_bsder")
])
results_df.to_csv(OUT_DIR / "exp_e_results.csv", index=False)
print(f"\n  Wrote exp_e_results.csv")

perclass_df = pd.DataFrame([
    {"variant": v, **results[v]["per_class_f1"]}
    for v in ("e1_der", "e2_bs", "e3_bsder")
])
perclass_df.to_csv(OUT_DIR / "exp_e_perclass.csv", index=False)
print(f"  Wrote exp_e_perclass.csv")

# ── McNEMAR PAIRWISE TESTS ────────────────────────────────────────────────────
print(f"\n[{datetime.now():%H:%M:%S}] McNemar pairwise tests")
pairs = [
    ("e1_der",  "e3_bsder"),
    ("e2_bs",   "e3_bsder"),
    ("e1_der",  "e2_bs"),
]
mcnemar_rows = []
for a, b in pairs:
    y_pred_a = results[a]["y_pred"]
    y_pred_b = results[b]["y_pred"]
    tb = mcnemar_table(y_target=train_labels_int,
                       y_model1=y_pred_a, y_model2=y_pred_b)
    chi2, p = mcnemar(ary=tb, corrected=True, exact=False)
    n_10 = int(tb[1, 0])  # a correct, b wrong
    n_01 = int(tb[0, 1])  # a wrong, b correct
    discordant = n_10 + n_01
    print(f"  {a} vs {b}: chi2={chi2:.4f}  p={p:.6f}  n10={n_10}  n01={n_01}")
    mcnemar_rows.append({
        "pair":          f"{a} vs {b}",
        "model_a":       a,
        "model_b":       b,
        "chi2":          float(chi2),
        "p_value":       float(p),
        "n_10_a_right_b_wrong": n_10,
        "n_01_a_wrong_b_right": n_01,
        "n_discordant":  discordant,
    })

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(OUT_DIR / "exp_e_mcnemar.csv", index=False)
print(f"  Wrote exp_e_mcnemar.csv")

# ── SUMMARY MARKDOWN ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"

summary = []
summary.append("# Input Data Ablation — Summary\n")
summary.append(f"**Run:** {STAMP}\n")
summary.append("**Reference:** Calvert et al. (2015), ICES J. Mar. Sci. 72(5): 1498-1513\n")
summary.append("**Pipeline:** Sub 41 / Sub 42 tabular path (LGB defaults, n_est=1000, ALG class_weight=2.0)\n")
summary.append("**CV:** 150m BlockKFold (verde), n_splits=5, random_state=42 — read from `kaggle_dataset/fold_indices.csv`\n")
summary.append("**OOF metrics on n=6256 training points. No test predictions.**\n\n")

summary.append("## Main results\n")
summary.append("| Variant | n_features | Weighted F1 | Cohen κ | Accuracy | Quantity dis. | Allocation dis. |")
summary.append("|---|---:|---:|---:|---:|---:|---:|")
for v in ("e1_der", "e2_bs", "e3_bsder"):
    r = results[v]
    summary.append(
        f"| **{v}** | {r['n_features']} | {fmt(r['weighted_f1'])} | "
        f"{fmt(r['cohen_kappa'])} | {fmt(r['accuracy'])} | "
        f"{fmt(r['quantity_disagreement'])} | {fmt(r['allocation_disagreement'])} |"
    )
summary.append("")

summary.append("## Per-class F1\n")
summary.append("| Variant | ALG | FMAT | NVB | SGAM | SGZ |")
summary.append("|---|---:|---:|---:|---:|---:|")
for v in ("e1_der", "e2_bs", "e3_bsder"):
    pc = results[v]["per_class_f1"]
    summary.append(
        f"| **{v}** | {fmt(pc['ALG'])} | {fmt(pc['FMAT'])} | "
        f"{fmt(pc['NVB'])} | {fmt(pc['SGAM'])} | {fmt(pc['SGZ'])} |"
    )
summary.append("")

summary.append("## Pairwise McNemar tests (corrected χ², continuity correction)\n")
summary.append("| Pair | χ² | p-value | n_10 (A right, B wrong) | n_01 (A wrong, B right) | n_discordant |")
summary.append("|---|---:|---:|---:|---:|---:|")
for r in mcnemar_rows:
    summary.append(
        f"| {r['pair']} | {fmt(r['chi2'])} | "
        f"{r['p_value']:.4g} | {r['n_10_a_right_b_wrong']} | "
        f"{r['n_01_a_wrong_b_right']} | {r['n_discordant']} |"
    )
summary.append("")

summary.append("## Reference values\n")
summary.append(f"- Sub 41 registered 150m BKF OOF weighted F1: **{E3_EXPECTED_F1:.4f}** (RESULTS_REGISTRY.md line 50).\n")
summary.append(f"- E3 reproduction here: **{e3_f1:.4f}** (delta {e3_delta:.6f}, tolerance {E3_TOLERANCE}).\n")
summary.append("\n## Calvert et al. (2015) reference numbers (for comparison context)\n")
summary.append("- BS-only supervised accuracy: 52 %\n")
summary.append("- DER-only supervised accuracy: 51 %\n")
summary.append("- BSDER supervised accuracy: 65 %\n")
summary.append("- Site: Northern Ireland coast, 11–33 m depth, very different from Refuge Cove (0–22 m).\n")

(OUT_DIR / "summary_table.md").write_text("\n".join(summary), encoding="utf-8")
print(f"\n  Wrote summary_table.md")

# ── EXPERIMENT_LOG ENTRY ──────────────────────────────────────────────────────
log_entry = f"""

### {datetime.now():%Y-%m-%d %H:%M} — Input Data Ablation (E1 DER / E2 BS / E3 BSDER)
- **What changed:** Train LGB on three feature subsets of the Sub 41 tabular pipeline: 45 BATHY + 1 SPATIAL (E1), 41 BS + 1 SPATIAL (E2), full 87 (E3, reproduces Sub 41).
- **Feature version:** v07 + acr_sapa_w21 (87 features baseline), subset per variant
- **Model:** LightGBM defaults, n_est=1000, class_weight={{ALG:2.0}}, random_state=42
- **CV 150m:** E1={results['e1_der']['weighted_f1']:.4f}, E2={results['e2_bs']['weighted_f1']:.4f}, E3={results['e3_bsder']['weighted_f1']:.4f} (E3 expected {E3_EXPECTED_F1:.4f}, delta {e3_delta:.6f})
- **CV 50m:** not run (biased proxy, never used for decisions per Hard Rule 5)
- **LB result:** N/A — CV-only ablation study, no submission
- **Key finding:** see outputs/input_ablation/summary_table.md
- **Files created:** outputs/input_ablation/{{exp_e_results.csv, exp_e_perclass.csv, exp_e_mcnemar.csv, summary_table.md, phase0_*}} + oof_predictions/oof_{{e1_der,e2_bs,e3_bsder}}.npy + oof_meta.json
- **Decision:** REPORT — post-competition methodology study for GeoHab 2026 Discussion page (Calvert et al. 2015 design replication)
- **HYPOTHESIS:** Per Calvert (2015) at Northern Ireland: BSDER ≫ BS ≈ DER (65% vs 52% vs 51%). On Refuge Cove with a 5-class single-task LGB and stronger spatial structure, expect (a) E3 ≥ both single-stream variants, (b) gap between BSDER and the better single stream smaller than Calvert's 13 pp (their site is single-task substrate, ours is mixed biogenic-substrate; spatial leakage in 150m blocks limits ceiling).
- **CONCLUSION:** E1 DER={results['e1_der']['weighted_f1']:.4f}, E2 BS={results['e2_bs']['weighted_f1']:.4f}, E3 BSDER={results['e3_bsder']['weighted_f1']:.4f}. See summary_table.md for McNemar significance and per-class deltas.
"""

with open(ROOT / "experiment_log.md", "a", encoding="utf-8") as f:
    f.write(log_entry)
print(f"  Appended to experiment_log.md")

print(f"\n[{datetime.now():%H:%M:%S}] DONE.")
print(f"\n{'='*70}")
print("INPUT DATA ABLATION — FINAL RESULTS")
print(f"{'='*70}")
print(f"  E1 (DER,    46 feat): weighted F1 = {results['e1_der']['weighted_f1']:.4f}   κ = {results['e1_der']['cohen_kappa']:.4f}")
print(f"  E2 (BS,     42 feat): weighted F1 = {results['e2_bs']['weighted_f1']:.4f}   κ = {results['e2_bs']['cohen_kappa']:.4f}")
print(f"  E3 (BSDER,  87 feat): weighted F1 = {results['e3_bsder']['weighted_f1']:.4f}   κ = {results['e3_bsder']['cohen_kappa']:.4f}")
print(f"{'='*70}")
