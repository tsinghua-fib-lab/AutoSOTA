#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# ---------------- config ----------------

CATEGORIES = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

SCOPES = ["roi"]  # , "full"]

# use test split for GLOBAL_MODELS
GLOBAL_SPLIT = "test"

# nt / hyena / phyloGPN / caduceus-theirs models
NT_MODELS = [
    "hyenaDNA-random-init",
    "hyenaDNA",
    "phyloGPN",
    "nt-ms",
    "nt-human",
    "phyloGPN-random-init",
    "nt-ms-random-init",
    "nt-human-random-init",
    "caduceus-theirs",
    "caduceus-theirs-random-init",
    "evo2"
]

# old global root (still used for gamba/caduceus)
NT_ROOT = "/home/mica/NucleotideTransformer/global_representations"

# NEW: upstream-pair region embeddings for NT-style models
NT_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/upstream_pairs"
PAIR_GROUP_NAME = "all"          # corresponds to <group_name> in reps_<model>_<group>_<category>_<scope>.npz
PAIR_LABEL_FILTER = "feature"    # use only feature regions; set to None to use all

# gamba / caduceus global models (non-upstream)
GLOBAL_MODELS = [
    "gamba_cons_only_ALLPOSstep_44000",
    "gamba_dual_ALLPOSstep_44000",
    "gamba_seq_only_ALLPOSstep_44000",
    "gamba_cons_only_step_random_init",
    "gamba_dual_step_random_init",
    "gamba_seq_only_step_random_init",
    "caduceus_cons_only_ALLPOSstep_44000",
    "caduceus_dual_ALLPOSstep_44000",
    "caduceus_seq_only_ALLPOSstep_44000",
    "caduceus_cons_only_step_random_init",
    "caduceus_dual_step_random_init",
    "caduceus_seq_only_step_random_init",
]

GLOBAL_ROOT = "/home/mica/gamba/data_processing/data/240-mammalian/global_representations"

# baseline models (kmer6, phylop) – assumed to have global reps too
BASELINE_MODELS = ["kmer6", "phylop"]
BASELINE_ROOT = "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/baseline"


# ---------------- core metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """
    LOO 1-NN balanced accuracy and SE (multiclass).
    X: [N, D], y: [N]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] < 3:
        raise ValueError(f"too few samples: {X.shape[0]}")

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # idx[:, 0] is self, idx[:, 1] is nearest neighbor excluding self
    y_pred = y[idx[:, 1]]

    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)
    n_per_class = cm.sum(axis=1).astype(float)
    correct_per_class = np.diag(cm).astype(float)

    # avoid div by zero for empty classes
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(
            correct_per_class,
            np.where(n_per_class == 0, 1.0, n_per_class),
        )

    K = len(classes)
    ba = float(np.nanmean(recalls))  # 0–1

    # variance from per-class binomial approximation
    var = np.nansum(
        recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)
    ) / (K ** 2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0  # in percent


# ---------------- helpers: file naming ----------------

def infer_model_short(model_folder: str) -> str:
    """
    for gamba/caduceus folder names, derive the short prefix used in reps_* filenames.
    examples:
      gamba_seq_only_ALLPOSstep_44000   -> "gamba"
      caduceus_dual_ALLPOSstep_44000    -> "caduceus"
    """
    if model_folder.startswith("gamba_"):
        return "gamba"
    if model_folder.startswith("caduceus_"):
        return "caduceus"
    return model_folder.split("_")[0]


# ---------------- loaders ----------------
# (1) NT-style models from upstream pair files (NEW)

def load_nt_pairs_multiclass(
    model: str,
    scope: str,
    group_name: str = PAIR_GROUP_NAME,
    pair_label_filter: str | None = PAIR_LABEL_FILTER,
):
    """
    load per-region embeddings from upstream pair reps for a given NT-style model/scope.

    expected files (from upstream_pairs script):
      NT_PAIRS_ROOT/<model>/
        reps_<model>_<group_name>_<category>_<scope>.npz


    npz must contain:
      - embeddings: [N_cat, D]
      - labels: [N_cat] with values like "feature" / "upstream"

    we:
      - optionally filter to labels == pair_label_filter (e.g. "feature")
      - assign each remaining row the category label (for multiclass BA)
      - concatenate across categories
    """
    model_dir = os.path.join(NT_PAIRS_ROOT, model)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing NT pairs dir: {model_dir}")
        return None, None

    X_list = []
    y_list = []

    for cat in CATEGORIES:
        base = os.path.join(
            model_dir,
            f"reps_{model}_{group_name}_{cat}_{scope}",
        )
        npz_path = base + ".npz"

        if not os.path.exists(npz_path):
            # optional: try older naming with "_upstream_" if needed
            alt_base = os.path.join(
                model_dir,
                f"reps_{model}_{group_name}_{cat}_upstream_{scope}",
            )
            alt_npz = alt_base + ".npz"
            if os.path.exists(alt_npz):
                npz_path = alt_npz
            else:
                print(f"[warn] missing NT pairs npz for {model} {cat} {scope}: {npz_path}")
                continue

        z = np.load(npz_path, allow_pickle=True)
        X_cat = np.asarray(z["embeddings"])

        labels_pair = None
        if "labels" in z:
            labels_pair = np.asarray(z["labels"]).astype(str)

        # optional filter: use only feature (ROI) or upstream segments
        if labels_pair is not None and pair_label_filter is not None:
            mask = labels_pair == pair_label_filter
            if not np.any(mask):
                print(
                    f"[warn] NT {model} {cat} {scope}: "
                    f"no samples with pair_label == '{pair_label_filter}'"
                )
                continue
            X_cat = X_cat[mask]

        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        # category label for multiclass
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        print(f"[warn] NT {model} {scope}: no categories loaded from pairs")
        return None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if X_all.shape[0] < 3:
        print(f"[warn] NT {model} {scope}: too few samples after combining categories")
        return None, None

    return X_all, y_all


# (2) global gamba/caduceus (unchanged)

def load_global_gamba_caduceus(model_folder: str, scope: str, split: str = "all"):
    """
    load global gamba/caduceus embeddings + labels for given model folder, split, and scope.

    expected files:
      GLOBAL_ROOT/<model_folder>/reps_<short>_<split>_<scope>.npz
      GLOBAL_ROOT/<model_folder>/reps_<short>_<split>_<scope>_meta.parquet

    where <short> is "gamba" or "caduceus" (inferred), and split is e.g. "test".
    labels taken from meta["category"].
    """
    model_dir = os.path.join(GLOBAL_ROOT, model_folder)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing global model dir: {model_dir}")
        return None, None

    short = infer_model_short(model_folder)
    base = os.path.join(model_dir, f"reps_{short}_{split}_{scope}")
    npz_path = base + ".npz"
    meta_path = base + "_meta.parquet"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        print(f"[warn] missing global reps for {model_folder} {split} {scope}: {npz_path} / {meta_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    meta = pd.read_parquet(meta_path)

    if "category" not in meta.columns:
        print(f"[warn] global meta has no 'category': {meta_path}")
        return None, None

    mask = meta["category"].isin(CATEGORIES).values
    X = X[mask]
    y = meta.loc[mask, "category"].astype(str).values

    if X.shape[0] < 3:
        print(f"[warn] global {model_folder} {split} {scope}: too few samples after filtering")
        return None, None

    return X, y


def load_baseline_global(model: str, scope: str):
    """
    load global baseline (kmer6 / phylop) embeddings + labels for a given model/scope.

    we try two layouts:
      1) BASELINE_ROOT/<model>/reps_<model>_all_<scope>.npz
      2) BASELINE_ROOT/<model>/all/reps_<model>_all_<scope>.npz

    labels taken from meta["category"].
    """
    # layout 1
    base1_dir = os.path.join(BASELINE_ROOT, model)
    base1 = os.path.join(base1_dir, f"reps_{model}_all_{scope}")
    npz1 = base1 + ".npz"
    meta1 = base1 + "_meta.parquet"

    # layout 2 (with 'all' subfolder)
    base2_dir = os.path.join(BASELINE_ROOT, model, "all")
    base2 = os.path.join(base2_dir, f"reps_{model}_all_{scope}")
    npz2 = base2 + ".npz"
    meta2 = base2 + "_meta.parquet"

    if os.path.exists(npz1) and os.path.exists(meta1):
        npz_path, meta_path = npz1, meta1
    elif os.path.exists(npz2) and os.path.exists(meta2):
        npz_path, meta_path = npz2, meta2
    else:
        print(f"[warn] missing baseline global reps for {model} {scope}: "
              f"{npz1} / {meta1} and {npz2} / {meta2}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    meta = pd.read_parquet(meta_path)

    if "category" not in meta.columns:
        print(f"[warn] baseline meta has no 'category': {meta_path}")
        return None, None

    mask = meta["category"].isin(CATEGORIES).values
    X = X[mask]
    y = meta.loc[mask, "category"].astype(str).values

    if X.shape[0] < 3:
        print(f"[warn] baseline {model} {scope}: too few samples after filtering")
        return None, None

    return X, y


# ---------------- collection ----------------

def collect_nt_multiclass_rows_from_pairs():
    """
    for each NT-style model/scope:
      - load per-region embeddings from upstream pair files
      - use category as the multiclass label
      - compute LOO 1-NN balanced accuracy + SE
    """
    rows = []
    for model in NT_MODELS:
        for scope in SCOPES:
            X, y = load_nt_pairs_multiclass(
                model,
                scope,
                group_name=PAIR_GROUP_NAME,
                pair_label_filter=PAIR_LABEL_FILTER,
            )
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] NT-pairs {model} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model,
                    Family="NT_pairs",
                    Group=PAIR_GROUP_NAME,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )
    return rows


def collect_global_multiclass_rows(default_split: str = GLOBAL_SPLIT):
    rows = []
    for model_folder in GLOBAL_MODELS:
        # random-init models use the "all" split
        if "random_init" in model_folder:
            split = "all"
        else:
            split = default_split

        for scope in SCOPES:
            X, y = load_global_gamba_caduceus(model_folder, scope, split=split)
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] global {model_folder} {split} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model_folder,
                    Family="gamba/caduceus",
                    Group=split,   # "test" for non-random, "all" for random-init
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )
    return rows


def collect_baseline_multiclass_rows():
    rows = []
    for model in BASELINE_MODELS:
        for scope in SCOPES:
            X, y = load_baseline_global(model, scope)
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] baseline {model} {scope}: {e}")
                continue

            rows.append(
                dict(
                    Model=model,
                    Family="baseline",
                    Group="all",
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )
    return rows


# ---------------- aggregation: per-model global BA ----------------

def aggregate_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    each row is already a multiclass BA for one (model, group, scope),
    so aggregation just averages across runs if there are multiple.
    """
    summaries = []
    for (model, group, scope), sub in df.groupby(["Model", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()

        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))

        # combine SEs assuming independence
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0

        std_across_runs = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(
            dict(
                Model=model,
                Group=group,
                Scope=scope,
                N_Runs=K,
                GlobalBalancedAccuracyPct=ba_global,
                GlobalBalancedAccuracyStdPct=std_across_runs,
                GlobalBalancedAccuracySEPct=se_global,
            )
        )
    return pd.DataFrame(summaries)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="multiclass global balanced accuracy using upstream pair region embeddings (NT) + global reps (gamba/caduceus/baseline)"
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_multiclass",
        help="output directory for per-model multiclass BA tables",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows_nt_pairs = collect_nt_multiclass_rows_from_pairs()
    rows_global = collect_global_multiclass_rows()
    rows_bl = collect_baseline_multiclass_rows()

    rows = rows_nt_pairs + rows_global + rows_bl
    if not rows:
        raise SystemExit("no rows collected; check paths and config / filenames.")

    df = pd.DataFrame(rows)
    per_model_path = os.path.join(args.outdir, "multiclass_balacc_per_model.tsv")
    df.to_csv(per_model_path, sep="\t", index=False)
    print(f"[info] wrote per-model multiclass table: {per_model_path}")

    df_global = aggregate_global(df)
    global_path = os.path.join(args.outdir, "multiclass_balacc_global.tsv")
    df_global.to_csv(global_path, sep="\t", index=False)
    print(f"[info] wrote aggregated table: {global_path}")


if __name__ == "__main__":
    main()
