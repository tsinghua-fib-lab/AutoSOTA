#!/usr/bin/env python3
import os
import math
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# ---------------- config ----------------

CATEGORIES = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters",
]

SCOPES = ["roi"]  # add "full" if needed

# NT / hyena / phyloGPN / caduceus-theirs / evo2 models (other-models)
NT_MODELS = [
    "hyenaDNA-random-init", "hyenaDNA", "phyloGPN", "nt-ms", "nt-human",
    "phyloGPN-random-init", "nt-ms-random-init", "nt-human-random-init",
    "caduceus-theirs", "caduceus-theirs-random-init", "evo2",
]

# gamba/caduceus models (gamba_onepass)
GAMBA_MODELS = [
    "gamba_cons_only_step44000",
    "gamba_dual_step44000",
    "gamba_seq_only_step44000",
    "gamba_cons_only_step0",
    "gamba_dual_step0",
    "gamba_seq_only_step0",
    "caduceus_cons_only_step44000",
    "caduceus_dual_step44000",
    "caduceus_seq_only_step44000",
    "caduceus_cons_only_step0",
    "caduceus_dual_step0",
    "caduceus_seq_only_step0",
]

BASELINE_MODELS = ["kmer6", "phylop"]

# ---------- roots ----------
OTHER_MODELS_ROOT = "/home/mica/gamba/other-models/final_representations/all_tasks"
GAMBA_ONEPASS_ROOT = "/home/mica/gamba/other-models/final_representations/gamba_onepass"


# ---------------- core metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """LOO 1-NN balanced accuracy and SE (binary or multiclass)."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] < 3:
        raise ValueError(f"too few samples: {X.shape[0]}")

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    y_pred = y[idx[:, 1]]

    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)
    n_per_class = cm.sum(axis=1).astype(float)
    correct_per_class = np.diag(cm).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(
            correct_per_class,
            np.where(n_per_class == 0, 1.0, n_per_class),
        )

    K = len(classes)
    ba = float(np.nanmean(recalls))
    var = np.nansum(
        recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)
    ) / (K ** 2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0


# ---------------- helpers ----------------

def load_nt_binary(
    model: str,
    group: str,
    task: str,
    category: str,
    scope: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load NT-style binary task (feature vs random/upstream).
    Path: OTHER_MODELS_ROOT/{model}/{group}/reps_{model}_{group}_{task}_{cat}_{scope}.npz
    Returns: (X_pos, X_neg) where pos=feature, neg=task
    """
    npz_path = os.path.join(
        OTHER_MODELS_ROOT,
        model,
        group,
        f"reps_{model}_{group}_{task}_{category}_{scope}.npz",
    )
    if not os.path.exists(npz_path):
        print(f"[warn] missing NT binary: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    labels = np.asarray(z["labels"]).astype(str)

    mask_pos = labels == "feature"
    mask_neg = labels == task

    if not np.any(mask_pos) or not np.any(mask_neg):
        print(f"[warn] {npz_path}: pos={mask_pos.sum()}, neg={mask_neg.sum()}")
        return None, None

    return X[mask_pos], X[mask_neg]


def load_gamba_binary(
    model_id: str,
    group: str,
    task: str,
    category: str,
    scope: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load gamba/caduceus binary task.
    Path: GAMBA_ONEPASS_ROOT/{model_id}/tasks/binary/{task}/{cat}/reps_{model_id}_{group}_{cat}_binary-{task}_{scope}.npz
    Returns: (X_pos, X_neg) where pos=feature, neg=task
    """
    npz_path = os.path.join(
        GAMBA_ONEPASS_ROOT,
        model_id,
        "tasks",
        "binary",
        task,
        category,
        f"reps_{model_id}_{group}_{category}_binary-{task}_{scope}.npz",
    )
    if not os.path.exists(npz_path):
        print(f"[warn] missing gamba binary: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    labels = np.asarray(z["labels"]).astype(str)

    mask_pos = labels == "feature"
    mask_neg = labels == task

    if not np.any(mask_pos) or not np.any(mask_neg):
        print(f"[warn] {npz_path}: pos={mask_pos.sum()}, neg={mask_neg.sum()}")
        return None, None

    return X[mask_pos], X[mask_neg]


def load_nt_multiclass(model: str, group: str, scope: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load NT multiclass (all categories combined).
    Path: OTHER_MODELS_ROOT/{model}/{group}/reps_{model}_{group}_multiclass_{scope}.npz
    """
    npz_path = os.path.join(
        OTHER_MODELS_ROOT,
        model,
        group,
        f"reps_{model}_{group}_multiclass_{scope}.npz",
    )
    if not os.path.exists(npz_path):
        print(f"[warn] missing NT multiclass: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"]).astype(str)

    if X.shape[0] < 3:
        print(f"[warn] {npz_path}: too few samples ({X.shape[0]})")
        return None, None

    return X, y


def load_gamba_multiclass(model_id: str, group: str, scope: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load gamba/caduceus multiclass (all categories combined).
    Path: GAMBA_ONEPASS_ROOT/{model_id}/tasks/multiclass/reps_{model_id}_{group}_multiclass_{scope}.npz
    """
    npz_path = os.path.join(
        GAMBA_ONEPASS_ROOT,
        model_id,
        "tasks",
        "multiclass",
        f"reps_{model_id}_{group}_multiclass_{scope}.npz",
    )
    if not os.path.exists(npz_path):
        print(f"[warn] missing gamba multiclass: {npz_path}")
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"]).astype(str)

    if X.shape[0] < 3:
        print(f"[warn] {npz_path}: too few samples ({X.shape[0]})")
        return None, None

    return X, y


# ---------------- task collectors ----------------

def collect_binary_rows(task: str, group: str = "all"):
    """Collect rows for binary task (random or upstream)."""
    rows = []

    # NT models
    for model in NT_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                X_pos, X_neg = load_nt_binary(model, group, task, cat, scope)
                if X_pos is None:
                    continue

                # balance classes
                n = min(X_pos.shape[0], X_neg.shape[0])
                if n < 5:
                    print(f"[warn] {task} NT {model} {scope} {cat}: n={n} too small")
                    continue

                rng = np.random.default_rng(1337)
                idx_pos = rng.choice(X_pos.shape[0], size=n, replace=False)
                idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                X = np.concatenate([X_pos[idx_pos], X_neg[idx_neg]], axis=0)
                y = np.array(["feature"] * n + [task] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] {task} NT {model} {scope} {cat}: {e}")
                    continue

                rows.append(dict(
                    Model=model,
                    Family=f"NT_{task}",
                    Group=group,
                    Category=cat,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                    N_pos=int(n),
                    N_neg=int(n),
                ))

    # gamba/caduceus models
    for model_id in GAMBA_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                X_pos, X_neg = load_gamba_binary(model_id, "all", task, cat, scope)
                if X_pos is None:
                    continue

                n = min(X_pos.shape[0], X_neg.shape[0])
                if n < 5:
                    print(f"[warn] {task} gamba {model_id} {scope} {cat}: n={n} too small")
                    continue

                rng = np.random.default_rng(1337)
                idx_pos = rng.choice(X_pos.shape[0], size=n, replace=False)
                idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                X = np.concatenate([X_pos[idx_pos], X_neg[idx_neg]], axis=0)
                y = np.array(["feature"] * n + [task] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] {task} gamba {model_id} {scope} {cat}: {e}")
                    continue

                rows.append(dict(
                    Model=model_id,
                    Family=f"gamba_{task}",
                    Group="all",
                    Category=cat,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                    N_pos=int(n),
                    N_neg=int(n),
                ))

    # baselines
    for baseline in BASELINE_MODELS:
        for scope in SCOPES:
            for cat in CATEGORIES:
                X_pos, X_neg = load_gamba_binary(baseline, "all", task, cat, scope)
                if X_pos is None:
                    continue

                n = min(X_pos.shape[0], X_neg.shape[0])
                if n < 5:
                    print(f"[warn] {task} baseline {baseline} {scope} {cat}: n={n} too small")
                    continue

                rng = np.random.default_rng(1337)
                idx_pos = rng.choice(X_pos.shape[0], size=n, replace=False)
                idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                X = np.concatenate([X_pos[idx_pos], X_neg[idx_neg]], axis=0)
                y = np.array(["feature"] * n + [task] * n, dtype=object)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] {task} baseline {baseline} {scope} {cat}: {e}")
                    continue

                rows.append(dict(
                    Model=baseline,
                    Family=f"baseline_{task}",
                    Group="all",
                    Category=cat,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                    N_pos=int(n),
                    N_neg=int(n),
                ))

    return rows


def collect_multiclass_rows(group: str = "all"):
    """Collect multiclass rows (category vs category)."""
    rows = []

    # NT models
    for model in NT_MODELS:
        for scope in SCOPES:
            X, y = load_nt_multiclass(model, group, scope)
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] multiclass NT {model} {scope}: {e}")
                continue

            rows.append(dict(
                Model=model,
                Family="NT_multiclass",
                Group=group,
                Scope=scope,
                BA_pct=ba,
                BA_SE_pct=se,
            ))

    # gamba/caduceus models
    for model_id in GAMBA_MODELS:
        for scope in SCOPES:
            X, y = load_gamba_multiclass(model_id, "all", scope)
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] multiclass gamba {model_id} {scope}: {e}")
                continue

            rows.append(dict(
                Model=model_id,
                Family="gamba_multiclass",
                Group="all",
                Scope=scope,
                BA_pct=ba,
                BA_SE_pct=se,
            ))

    # baselines
    for baseline in BASELINE_MODELS:
        for scope in SCOPES:
            X, y = load_gamba_multiclass(baseline, "all", scope)
            if X is None:
                continue

            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception as e:
                print(f"[skip] multiclass baseline {baseline} {scope}: {e}")
                continue

            rows.append(dict(
                Model=baseline,
                Family="baseline_multiclass",
                Group="all",
                Scope=scope,
                BA_pct=ba,
                BA_SE_pct=se,
            ))

    return rows


# ---------------- aggregation ----------------

def aggregate_per_category(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (model, family, group, scope), sub in df.groupby(["Model", "Family", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()
        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0
        std_across_cats = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(dict(
            Model=model,
            Family=family,
            Group=group,
            Scope=scope,
            N_Categories=K,
            GlobalBalancedAccuracyPct=ba_global,
            GlobalBalancedAccuracyStdPct=std_across_cats,
            GlobalBalancedAccuracySEPct=se_global,
        ))
    return pd.DataFrame(summaries)


def aggregate_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (model, family, group, scope), sub in df.groupby(["Model", "Family", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()
        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0
        std_across_runs = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(dict(
            Model=model,
            Family=family,
            Group=group,
            Scope=scope,
            N_Runs=K,
            GlobalBalancedAccuracyPct=ba_global,
            GlobalBalancedAccuracyStdPct=std_across_runs,
            GlobalBalancedAccuracySEPct=se_global,
        ))
    return pd.DataFrame(summaries)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="compare embeddings via LOO 1-NN BA: multiclass / random / upstream"
    )
    ap.add_argument(
        "--task",
        choices=["multiclass", "random", "upstream"],
        required=True,
        help="comparison type",
    )
    ap.add_argument(
        "-o", "--outdir",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined",
        help="output directory for TSVs",
    )
    ap.add_argument("--group_name", type=str, default="all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.task in ["random", "upstream"]:
        rows = collect_binary_rows(args.task, group=args.group_name)
        if not rows:
            raise SystemExit(f"no rows collected for {args.task}")
        df = pd.DataFrame(rows)
        percat_path = os.path.join(args.outdir, f"balacc_{args.task}_per_category.tsv")
        df.to_csv(percat_path, sep="\t", index=False)
        print(f"[info] wrote per-category {args.task.upper()}: {percat_path}")

        df_global = aggregate_per_category(df)
        global_path = os.path.join(args.outdir, f"balacc_{args.task}_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote global {args.task.upper()}: {global_path}")

    else:  # multiclass
        rows = collect_multiclass_rows(group=args.group_name)
        if not rows:
            raise SystemExit("no rows collected for multiclass")
        df = pd.DataFrame(rows)
        per_model_path = os.path.join(args.outdir, "balacc_multiclass_per_model.tsv")
        df.to_csv(per_model_path, sep="\t", index=False)
        print(f"[info] wrote per-model MULTICLASS: {per_model_path}")

        df_global = aggregate_multiclass(df)
        global_path = os.path.join(args.outdir, "balacc_multiclass_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote aggregated MULTICLASS: {global_path}")


if __name__ == "__main__":
    main()