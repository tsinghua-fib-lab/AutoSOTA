#!/usr/bin/env python3
"""
gather embeddings from:
- nt pair npz (feature vs upstream / random)
- baselines (kmer6 / phylop) (feature vs upstream / random + multiclass)
- onepass outputs (gamba/caduceus/bigamba) [older layout]
- all_tasks outputs (MODEL list; your new layout) for:
    - upstream (feature vs upstream)
    - random (feature vs random)
    - random-noannot (feature vs random-noannot)
    - multiclass (categories; scope=roi)
    - multiclass100bproi (categories; scope=roi100bp)

writes per-task TSVs of per-category results (binary) or per-model (multiclass),
plus aggregated "global" TSVs.
"""

import os
import math
import argparse
from typing import Optional, Tuple, List, Dict

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

BINARY_SCOPES = ["roi"]  # add "full" if you have them
MULTICLASS_SCOPES = ["roi"]  # baseline + nt
ONEPASS_MULTICLASS_SCOPES = ["roi", "roi100bp"]  # roi100bp == multiclass100bproi

# nt / hyena / phyloGPN / caduceus-theirs models (pair reps)
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
    "evo2",
    "evo2-random-init",
]

# ---------- roots for each task ----------

# nt pair roots (pair-based npz with labels)
RANDOM_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/random_pairs"
UPSTREAM_PAIRS_ROOT = "/home/mica/NucleotideTransformer/final_representations/upstream_pairs"

# baselines for random/upstream (already combined feature+control per category)
RANDOM_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/random_pairs/baseline"
)
UPSTREAM_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/upstream_pairs/baseline"
)
RANDOM_NOANNOT_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/random_noannot_pairs/baseline"
)

# global baseline root for multiclass (per-category dirs)
GLOBAL_BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/final_representations/upstream_pairs/baseline"
)
BASELINE_MODELS = ["kmer6", "phylop"]

# -------- NEW: all_tasks root (your new one-pass outputs) --------
ALL_TASKS_ROOT = "/home/mica/gamba/other-models/final_representations/all_tasks"
ALL_TASKS_GROUP = "all"  # per your example; if you later write training/test, this can be overridden

# -------- older onepass root (gamba/caduceus/bigamba legacy layout) --------
ONEPASS_ROOT = "/home/mica/gamba/other-models/final_representations/gamba_onepass"

GLOBAL_MODELS = [
    "gamba_cons_only_ALLPOSstep_44000",
    "gamba_dual_ALLPOSstep_44000",
    "gamba_seq_only_ALLPOSstep_44000",
    "gamba_cons_only_step0",
    "gamba_dual_step0",
    "gamba_seq_only_step0",
    "caduceus_cons_only_ALLPOSstep_44000",
    "caduceus_dual_ALLPOSstep_44000",
    "caduceus_seq_only_ALLPOSstep_44000",
    "caduceus_cons_only_step0",
    "caduceus_dual_step0",
    "caduceus_seq_only_step0",
]

GLOBAL_SPLIT = "all"  # legacy onepass split (training/test) for non-random-init models

# multiclass (nt pairs) config
NT_PAIRS_ROOT = UPSTREAM_PAIRS_ROOT
PAIR_GROUP_NAME = "all"
PAIR_LABEL_FILTER = "feature"


# ---------------- core metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """
    LOO 1-NN balanced accuracy and SE (binary or multiclass).
    cosine distance.
    X: [N, D], y: [N]
    """
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
    ba = float(np.nanmean(recalls))  # 0–1

    var = np.nansum(
        recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)
    ) / (K ** 2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0  # percent


def compute_ba_and_se_from_npz(npz_path: str):
    """
    npz with 'embeddings' [N,D] and 'labels' [N]
    """
    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"])
    return compute_ba_and_se(X, y)


def _load_npz_embeddings_labels(npz_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not npz_path or not os.path.exists(npz_path):
        return None, None
    z = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in z or "labels" not in z:
        return None, None
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"]).astype(str)
    if X.shape[0] < 3:
        return None, None
    return X, y


# ---------------- NEW: all_tasks helpers ----------------

def _alltasks_binary_npz_path(
    model: str,
    group: str,
    category: str,
    task: str,   # "upstream" | "random" | "random-noannot"
    scope: str,  # "roi" | "full"
) -> str:
    # /all_tasks/MODEL/all/reps_MODEL_all_TASK_CATEGORY_roi.npz
    return os.path.join(
        ALL_TASKS_ROOT,
        model,
        group,
        f"reps_{model}_{group}_{task}_{category}_{scope}.npz",
    )


def _alltasks_multiclass_npz_path(
    model: str,
    group: str,
    scope: str,  # "roi" | "roi100bp"
) -> str:
    # /all_tasks/MODEL/all/reps_MODEL_all_multiclass_roi.npz
    return os.path.join(
        ALL_TASKS_ROOT,
        model,
        group,
        f"reps_{model}_{group}_multiclass_{scope}.npz",
    )


def _alltasks_group_for_model(model: str) -> str:
    # currently everything is in "all" per your description.
    # if you later add training/test, you can extend this easily.
    return ALL_TASKS_GROUP


# ---------------- legacy onepass helpers (unchanged) ----------------

def to_onepass_model_id(model_folder: str) -> str:
    """
    map older naming to onepass model_id:
      gamba_dual_ALLPOSstep_44000 -> gamba_dual_step44000
      gamba_dual_step_random_init -> gamba_dual_step0
    """
    s = model_folder
    s = s.replace("_ALLPOSstep_", "_step")
    s = s.replace("_step_random_init", "_step0")
    s = s.replace("-random-init", "_step0")
    s = s.replace("_random-init", "_step0")
    return s


def _onepass_group_for(model_folder: str) -> str:
    return "all" if "random_init" in model_folder else GLOBAL_SPLIT


def _onepass_binary_npz_path(
    model_folder: str,
    group: str,
    category: str,
    task: str,
    scope: str,
) -> str:
    mid = to_onepass_model_id(model_folder)
    tag = f"{group}_{category}_binary-{task}_{scope}"
    return os.path.join(
        ONEPASS_ROOT,
        mid,
        "tasks",
        "binary",
        task,
        category,
        f"reps_{mid}_{tag}.npz",
    )


def _onepass_multiclass_npz_path(
    model_folder: str,
    group: str,
    scope: str,
) -> str:
    mid = to_onepass_model_id(model_folder)
    tag = f"{group}_multiclass_{scope}"
    return os.path.join(
        ONEPASS_ROOT,
        mid,
        "tasks",
        "multiclass",
        f"reps_{mid}_{tag}.npz",
    )

# ---------------- NEW baseline-in-onepass helpers ----------------

def _onepass_baseline_binary_npz_path(
    baseline_model: str,   # "kmer6" | "phylop"
    group: str,            # "test" (or "training"/"all" if you have)
    category: str,
    task: str,             # "upstream" | "random" | "random-noannot"
    scope: str,            # "roi" | "full"
) -> str:
    # example:
    # gamba_onepass/kmer6/tasks/binary/upstream/coding_regions/reps_kmer6_test_coding_regions_binary-upstream_roi.npz
    return os.path.join(
        ONEPASS_ROOT,
        baseline_model,
        "tasks",
        "binary",
        task,
        category,
        f"reps_{baseline_model}_{group}_{category}_binary-{task}_{scope}.npz",
    )


def _onepass_baseline_multiclass_npz_path(
    baseline_model: str,
    group: str,            # "test"
    scope: str,            # "roi" | "roi100bp" (if you ever write it)
) -> str:
    # gamba_onepass/kmer6/tasks/multiclass/reps_kmer6_test_multiclass_roi.npz
    return os.path.join(
        ONEPASS_ROOT,
        baseline_model,
        "tasks",
        "multiclass",
        f"reps_{baseline_model}_{group}_multiclass_{scope}.npz",
    )



# ---------------- nt pair loader ----------------

def _load_pair_binary(
    root: str,
    model: str,
    category: str,
    scope: str,
    group_name: str,
    pos_label: str,
    neg_label: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    expects:
      root/<model>/reps_<model>_<group_name>_<category>_<scope>.npz
    """
    model_dir = os.path.join(root, model)
    if not os.path.isdir(model_dir):
        return None, None

    npz_path = os.path.join(model_dir, f"reps_{model}_{group_name}_{category}_{scope}.npz")
    if not os.path.exists(npz_path):
        return None, None

    z = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in z or "labels" not in z:
        return None, None

    X = np.asarray(z["embeddings"])
    labels = np.asarray(z["labels"]).astype(str)

    mask_pos = labels == pos_label
    mask_neg = labels == neg_label
    if not np.any(mask_pos) or not np.any(mask_neg):
        return None, None

    X_pos = X[mask_pos]
    X_neg = X[mask_neg]
    if X_pos.shape[0] < 3 or X_neg.shape[0] < 3:
        return None, None

    return X_pos, X_neg


# ---------------- MULTICLASS loaders ----------------

def load_nt_pairs_multiclass(
    model: str,
    scope: str,
    group_name: str = PAIR_GROUP_NAME,
    pair_label_filter: Optional[str] = PAIR_LABEL_FILTER,
):
    """
    NT multiclass from upstream pair reps: filter labels == 'feature' and label by category.
    """
    model_dir = os.path.join(NT_PAIRS_ROOT, model)
    if not os.path.isdir(model_dir):
        return None, None

    X_list, y_list = [], []

    for cat in CATEGORIES:
        npz_path = os.path.join(model_dir, f"reps_{model}_{group_name}_{cat}_{scope}.npz")
        if not os.path.exists(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)
        if "embeddings" not in z:
            continue

        X_cat = np.asarray(z["embeddings"])
        labels_pair = np.asarray(z["labels"]).astype(str) if "labels" in z else None

        if labels_pair is not None and pair_label_filter is not None:
            mask = labels_pair == pair_label_filter
            if not np.any(mask):
                continue
            X_cat = X_cat[mask]

        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        return None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    if X_all.shape[0] < 3:
        return None, None

    return X_all, y_all


def load_global_baseline(model: str, scope: str):
    """
    baseline multiclass from upstream_pairs/baseline:
      GLOBAL_BASELINE_ROOT/<model>/all/<category>/reps_<model>_all_<category>_<scope>.npz
    """
    X_list, y_list = [], []

    for cat in CATEGORIES:
        npz_path = os.path.join(
            GLOBAL_BASELINE_ROOT,
            model,
            "all",
            cat,
            f"reps_{model}_all_{cat}_{scope}.npz",
        )
        if not os.path.exists(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)
        if "embeddings" not in z:
            continue
        X_cat = np.asarray(z["embeddings"])
        if X_cat.shape[0] == 0:
            continue

        X_list.append(X_cat)
        y_list.append(np.full(X_cat.shape[0], cat, dtype=object))

    if not X_list:
        return None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if X.shape[0] < 3:
        return None, None

    return X, y


# ---------------- row collectors ----------------

def collect_binary_rows(task: str, group_name: str = "all") -> List[Dict]:
    """
    task in {"random","upstream","random-noannot"}
    """
    rows = []

    # ---------- NT-style models (pair-based) ----------
    nt_root = (
        RANDOM_PAIRS_ROOT if task == "random"
        else UPSTREAM_PAIRS_ROOT if task == "upstream"
        else None
    )
    nt_neg_label = (
        "random" if task == "random"
        else "upstream" if task == "upstream"
        else "random-noannot"
    )

    if nt_root is not None:
        for model in NT_MODELS:
            for scope in BINARY_SCOPES:
                for cat in CATEGORIES:
                    X_feat, X_neg = _load_pair_binary(
                        nt_root,
                        model,
                        cat,
                        scope,
                        group_name,
                        pos_label="feature",
                        neg_label=nt_neg_label,
                    )
                    if X_feat is None:
                        continue

                    n = min(X_feat.shape[0], X_neg.shape[0])
                    if n < 5:
                        continue

                    rng = np.random.default_rng(1337)
                    idx_feat = rng.choice(X_feat.shape[0], size=n, replace=False)
                    idx_neg = rng.choice(X_neg.shape[0], size=n, replace=False)

                    X = np.concatenate([X_feat[idx_feat], X_neg[idx_neg]], axis=0)
                    y = np.array(["feature"] * n + [nt_neg_label] * n, dtype=object)

                    try:
                        ba, se = compute_ba_and_se(X, y)
                    except Exception:
                        continue

                    rows.append(
                        dict(
                            Model=model,
                            Family=f"NT_{task}",
                            Group=group_name,
                            Category=cat,
                            Scope=scope,
                            BA_pct=ba,
                            BA_SE_pct=se,
                            N_pos=int(n),
                            N_neg=int(n),
                        )
                    )

    # ---------- NEW all_tasks models ----------
    # these npzs already contain both labels ("feature" and task label)
    for model in NT_MODELS:
        group = _alltasks_group_for_model(model)
        for scope in BINARY_SCOPES:
            for cat in CATEGORIES:
                npz_path = _alltasks_binary_npz_path(
                    model=model,
                    group=group,
                    category=cat,
                    task=task,
                    scope=scope,
                )
                X, y = _load_npz_embeddings_labels(npz_path)
                if X is None:
                    continue

                neg_lbl = task
                if not (np.any(y == "feature") and np.any(y == neg_lbl)):
                    continue

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception:
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family=f"all_tasks_{task}",
                        Group=group,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(np.sum(y == "feature")),
                        N_neg=int(np.sum(y == neg_lbl)),
                    )
                )

    # ---------- legacy onepass gamba/caduceus/bigamba ----------
    for model_folder in GLOBAL_MODELS:
        group = _onepass_group_for(model_folder)  # "test" or "all"
        for scope in BINARY_SCOPES:
            for cat in CATEGORIES:
                npz_path = _onepass_binary_npz_path(
                    model_folder=model_folder,
                    group=group,
                    category=cat,
                    task=task,
                    scope=scope,
                )
                X, y = _load_npz_embeddings_labels(npz_path)
                if X is None:
                    continue

                neg_lbl = task
                if not (np.any(y == "feature") and np.any(y == neg_lbl)):
                    continue

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception:
                    continue

                rows.append(
                    dict(
                        Model=to_onepass_model_id(model_folder),
                        Family=f"onepass_{task}",
                        Group=group,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(np.sum(y == "feature")),
                        N_neg=int(np.sum(y == neg_lbl)),
                    )
                )
    # ---------- baselines (new: stored under gamba_onepass/<baseline>/tasks/...) ----------
    for baseline in BASELINE_MODELS:
        group = GLOBAL_SPLIT  # typically "test"
        for scope in BINARY_SCOPES:
            for cat in CATEGORIES:
                npz_path = _onepass_baseline_binary_npz_path(
                    baseline_model=baseline,
                    group=group,
                    category=cat,
                    task=task,
                    scope=scope,
                )
                X, y = _load_npz_embeddings_labels(npz_path)
                if X is None:
                    continue

                # expected labels: "feature" and task name
                if not (np.any(y == "feature") and np.any(y == task)):
                    continue

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception:
                    continue

                rows.append(
                    dict(
                        Model=baseline,
                        Family=f"baseline_onepass_{task}",
                        Group=group,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_pos=int(np.sum(y == "feature")),
                        N_neg=int(np.sum(y == task)),
                    )
                )
    return rows


def collect_multiclass_rows(scope: str) -> List[Dict]:
    """
    scope in {"roi","roi100bp"} where:
      - nt/baseline only support "roi"
      - all_tasks supports "roi" and "roi100bp"
      - legacy onepass supports "roi" and "roi100bp"
    """
    rows = []

    # NT-style multiclass only for roi
    if scope == "roi":
        for model in NT_MODELS:
            X, y = load_nt_pairs_multiclass(model, scope)
            if X is None:
                continue
            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception:
                continue
            rows.append(
                dict(
                    Model=model,
                    Family="NT_pairs_multiclass",
                    Group=PAIR_GROUP_NAME,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )

    # NEW all_tasks multiclass (roi or roi100bp)
    for model in NT_MODELS:
        group = _alltasks_group_for_model(model)
        npz_path = _alltasks_multiclass_npz_path(model=model, group=group, scope=scope)
        X, y = _load_npz_embeddings_labels(npz_path)
        if X is None:
            continue
        try:
            ba, se = compute_ba_and_se(X, y)
        except Exception:
            continue
        rows.append(
            dict(
                Model=model,
                Family="all_tasks_multiclass",
                Group=group,
                Scope=scope,
                BA_pct=ba,
                BA_SE_pct=se,
            )
        )

    # legacy onepass multiclass (roi or roi100bp)
    for model_folder in GLOBAL_MODELS:
        group = _onepass_group_for(model_folder)
        npz_path = _onepass_multiclass_npz_path(model_folder=model_folder, group=group, scope=scope)
        X, y = _load_npz_embeddings_labels(npz_path)
        if X is None:
            continue
        try:
            ba, se = compute_ba_and_se(X, y)
        except Exception:
            continue
        rows.append(
            dict(
                Model=to_onepass_model_id(model_folder),
                Family="onepass_multiclass",
                Group=group,
                Scope=scope,
                BA_pct=ba,
                BA_SE_pct=se,
            )
        )

    # baselines multiclass (new: stored under gamba_onepass/<baseline>/tasks/...)
    if scope == "roi":
        for baseline in BASELINE_MODELS:
            group = GLOBAL_SPLIT  # typically "test"
            npz_path = _onepass_baseline_multiclass_npz_path(
                baseline_model=baseline,
                group=group,
                scope=scope,
            )
            X, y = _load_npz_embeddings_labels(npz_path)
            if X is None:
                continue
            try:
                ba, se = compute_ba_and_se(X, y)
            except Exception:
                continue
            rows.append(
                dict(
                    Model=baseline,
                    Family="baseline_onepass_multiclass",
                    Group=group,
                    Scope=scope,
                    BA_pct=ba,
                    BA_SE_pct=se,
                )
            )


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

        summaries.append(
            dict(
                Model=model,
                Family=family,
                Group=group,
                Scope=scope,
                N_Categories=K,
                GlobalBalancedAccuracyPct=ba_global,
                GlobalBalancedAccuracyStdPct=std_across_cats,
                GlobalBalancedAccuracySEPct=se_global,
            )
        )
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

        summaries.append(
            dict(
                Model=model,
                Family=family,
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
    global ALL_TASKS_GROUP
    ap = argparse.ArgumentParser(
        description=(
            "compute LOO 1-NN balanced accuracy from saved reps npz.\n"
            "tasks:\n"
            "  upstream           : feature vs upstream (per-category)\n"
            "  random             : feature vs random (per-category)\n"
            "  random_noannot     : feature vs random-noannot (per-category)\n"
            "  multiclass         : category vs category (roi)\n"
            "  multiclass100bproi : category vs category (roi100bp)\n"
        )
    )
    ap.add_argument(
        "--task",
        choices=["multiclass", "multiclass100bproi", "random", "random_noannot", "upstream"],
        required=True,
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_combined",
        help="output directory for TSVs",
    )
    ap.add_argument("--group_name", type=str, default="all", help="nt pair group name (usually all)")
    ap.add_argument(
        "--all_tasks_group",
        type=str,
        default=ALL_TASKS_GROUP,
        help="group folder inside all_tasks (default: all)",
    )
    args = ap.parse_args()

    # allow overriding the all_tasks group at runtime
    ALL_TASKS_GROUP = args.all_tasks_group

    os.makedirs(args.outdir, exist_ok=True)

    if args.task in ("random", "upstream", "random_noannot"):
        task = "random" if args.task == "random" else "upstream" if args.task == "upstream" else "random-noannot"
        rows = collect_binary_rows(task=task, group_name=args.group_name)
        if not rows:
            raise SystemExit(f"no rows collected for {args.task}; check paths + outputs")
        df = pd.DataFrame(rows)

        percat_path = os.path.join(args.outdir, f"balacc_{args.task}_per_category.tsv")
        df.to_csv(percat_path, sep="\t", index=False)
        print(f"[info] wrote per-category {args.task} table: {percat_path}")

        df_global = aggregate_per_category(df)
        global_path = os.path.join(args.outdir, f"balacc_{args.task}_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote global {args.task} table: {global_path}")
        return

    if args.task == "multiclass":
        scope = "roi"
        rows = collect_multiclass_rows(scope=scope)
        if not rows:
            raise SystemExit("no rows collected for multiclass; check paths")
        df = pd.DataFrame(rows)

        per_model_path = os.path.join(args.outdir, "balacc_multiclass_per_model.tsv")
        df.to_csv(per_model_path, sep="\t", index=False)
        print(f"[info] wrote per-model MULTICLASS table: {per_model_path}")

        df_global = aggregate_multiclass(df)
        global_path = os.path.join(args.outdir, "balacc_multiclass_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote aggregated MULTICLASS table: {global_path}")
        return

    if args.task == "multiclass100bproi":
        scope = "roi100bp"
        rows = collect_multiclass_rows(scope=scope)
        if not rows:
            raise SystemExit("no rows collected for multiclass100bproi; check roi100bp outputs")
        df = pd.DataFrame(rows)

        per_model_path = os.path.join(args.outdir, "balacc_multiclass100bproi_per_model.tsv")
        df.to_csv(per_model_path, sep="\t", index=False)
        print(f"[info] wrote per-model MULTICLASS100BPROI table: {per_model_path}")

        df_global = aggregate_multiclass(df)
        global_path = os.path.join(args.outdir, "balacc_multiclass100bproi_global.tsv")
        df_global.to_csv(global_path, sep="\t", index=False)
        print(f"[info] wrote aggregated MULTICLASS100BPROI table: {global_path}")
        return


if __name__ == "__main__":
    main()
