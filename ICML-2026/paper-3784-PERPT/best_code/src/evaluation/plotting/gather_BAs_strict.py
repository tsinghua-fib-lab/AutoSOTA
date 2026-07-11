#!/usr/bin/env python3
"""
strict roi matching across *all* tasks:
- upstream, random, random-noannot, multiclass (and optionally roi100bp multiclass)

requirement you stated:
- the ROI (positive) pair_id sets are identical across *every* task.
  - binary tasks: "feature" rows define ROI set; negatives differ by task label
  - multiclass: ROI set is the same; label is the category; "negatives" are other categories

this script:
1) builds canonical per-category ROI pair_id sets from a reference model/task
2) checks, for every model/family, that ROI pair_id sets match the canonical set for:
   - upstream, random, random-noannot (binary)
   - multiclass (and multiclass100bproi if selected)
3) computes BA using ONLY the canonical ROI pair_ids:
   - binary: restrict to canonical ROI pair_ids for the category, then require both labels exist
   - multiclass: for each category, restrict to canonical ROI pair_ids, then concatenate and compute BA
4) writes:
   - canonical stats
   - per-model ROI consistency report (this is the key “same pair_ids everywhere” proof)
   - BA tables + global aggregation
   - coverage tables

notes:
- expects every reps_*.npz has adjacent reps_*_meta.parquet with same row order/length
- meta must include a pair_id-like column
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

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

BINARY_TASKS = ["upstream", "random", "random-noannot"]
MULTICLASS_TASKS = ["multiclass"]  # plus "multiclass100bproi" (scope=roi100bp) if requested
BINARY_SCOPES = ["roi"]  # extend if needed
MULTICLASS_SCOPES = ["roi", "roi100bp"]  # roi100bp corresponds to multiclass100bproi

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

# roots
ALL_TASKS_ROOT = "/home/mica/gamba/other-models/final_representations/all_tasks"
ONEPASS_ROOT = "/home/mica/gamba/other-models/final_representations/gamba_onepass"

# typical group naming
ALL_TASKS_GROUP_DEFAULT = "all"
ONEPASS_GROUP_DEFAULT = "all"

BASELINE_MODELS = ["kmer6", "phylop"]

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

# meta schema candidates
PAIR_COL_CANDIDATES = ["pair_id", "pairid", "pairID", "pair_id_feature", "pair_id_control"]
# role col is not required for strict matching (we trust npz labels), but keep it in case you want it later.
ROLE_COL_CANDIDATES = ["role", "region_type", "regiontype", "region_kind", "regionkind", "label", "labels", "class_label"]


# ---------------- metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """
    LOO 1-NN balanced accuracy and SE (binary or multiclass).
    cosine distance.
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
        recalls = np.divide(correct_per_class, np.where(n_per_class == 0, 1.0, n_per_class))

    K = len(classes)
    ba = float(np.nanmean(recalls))  # 0–1

    var = np.nansum(recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)) / (K**2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0


# ---------------- helpers: paths ----------------

def to_onepass_model_id(model_folder: str) -> str:
    s = model_folder
    s = s.replace("_ALLPOSstep_", "_step")
    s = s.replace("_step_random_init", "_step0")
    s = s.replace("-random-init", "_step0")
    s = s.replace("_random-init", "_step0")
    return s


def _alltasks_binary_npz_path(model: str, group: str, category: str, task: str, scope: str) -> str:
    return os.path.join(ALL_TASKS_ROOT, model, group, f"reps_{model}_{group}_{task}_{category}_{scope}.npz")


def _alltasks_multiclass_npz_path(model: str, group: str, scope: str) -> str:
    return os.path.join(ALL_TASKS_ROOT, model, group, f"reps_{model}_{group}_multiclass_{scope}.npz")


def _onepass_binary_npz_path(model_folder: str, group: str, category: str, task: str, scope: str) -> str:
    mid = to_onepass_model_id(model_folder)
    tag = f"{group}_{category}_binary-{task}_{scope}"
    return os.path.join(ONEPASS_ROOT, mid, "tasks", "binary", task, category, f"reps_{mid}_{tag}.npz")


def _onepass_multiclass_npz_path(model_folder: str, group: str, scope: str) -> str:
    mid = to_onepass_model_id(model_folder)
    tag = f"{group}_multiclass_{scope}"
    return os.path.join(ONEPASS_ROOT, mid, "tasks", "multiclass", f"reps_{mid}_{tag}.npz")


def _onepass_baseline_binary_npz_path(baseline_model: str, group: str, category: str, task: str, scope: str) -> str:
    return os.path.join(
        ONEPASS_ROOT,
        baseline_model,
        "tasks",
        "binary",
        task,
        category,
        f"reps_{baseline_model}_{group}_{category}_binary-{task}_{scope}.npz",
    )


def _onepass_baseline_multiclass_npz_path(baseline_model: str, group: str, scope: str) -> str:
    return os.path.join(ONEPASS_ROOT, baseline_model, "tasks", "multiclass", f"reps_{baseline_model}_{group}_multiclass_{scope}.npz")


# ---------------- helpers: meta alignment ----------------

def _pick_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def normalize_role(x: str) -> str:
    s = str(x)
    s = s.replace("random_noannot", "random-noannot")
    s = s.replace("random-no-annot", "random-noannot")
    s = s.replace("random_no_annot", "random-noannot")
    return s


def _meta_path_from_npz(npz_path: str) -> str:
    p = Path(npz_path)
    return str(p.with_name(p.name.replace(".npz", "_meta.parquet")))


def load_npz_Xy_pairids(npz_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    loads X, y from npz and pair_ids from adjacent meta parquet.
    returns (X, y, pair_ids, meta_path) or (None,...)
    """
    if not npz_path or not os.path.exists(npz_path):
        return None, None, None, None

    meta_path = _meta_path_from_npz(npz_path)
    if not os.path.exists(meta_path):
        return None, None, None, None

    try:
        z = np.load(npz_path, allow_pickle=True)
        if "embeddings" not in z or "labels" not in z:
            return None, None, None, None
        X = np.asarray(z["embeddings"])
        y = np.asarray(z["labels"]).astype(str)
        y = np.array([normalize_role(v) for v in y], dtype=object)
    except Exception:
        return None, None, None, None

    try:
        m = pd.read_parquet(meta_path)
    except Exception:
        return None, None, None, None

    if len(m) != len(y):
        return None, None, None, None

    pair_col = _pick_col(m.columns, PAIR_COL_CANDIDATES)
    if pair_col is None:
        return None, None, None, None

    pair_ids = m[pair_col].astype(str).to_numpy()
    return X, y, pair_ids, meta_path


# ---------------- canonical ROI sets ----------------

def build_canonical_roi_sets_from_binary(
    ref_family: str,
    ref_model: str,
    ref_group: str,
    scope: str,
    ref_task: str,
) -> Dict[str, Set[str]]:
    """
    canonical per-category ROI pair_id sets:
      - binary: ROI == rows where y == "feature"
    """
    out: Dict[str, Set[str]] = {}
    for cat in CATEGORIES:
        if ref_family == "all_tasks":
            npz = _alltasks_binary_npz_path(ref_model, ref_group, cat, task=ref_task, scope=scope)
        elif ref_family == "onepass":
            npz = _onepass_binary_npz_path(ref_model, ref_group, cat, task=ref_task, scope=scope)
        else:
            raise ValueError(f"unknown ref_family={ref_family}")

        X, y, pids, _ = load_npz_Xy_pairids(npz)
        if X is None:
            out[cat] = set()
            continue
        out[cat] = set(pids[np.asarray(y) == "feature"].tolist())
    return out


def build_canonical_roi_sets_from_multiclass(
    ref_family: str,
    ref_model: str,
    ref_group: str,
    scope: str,
) -> Dict[str, Set[str]]:
    """
    canonical per-category ROI pair_id sets:
      - multiclass: ROI for category c == rows where y == c
    """
    if ref_family == "all_tasks":
        npz = _alltasks_multiclass_npz_path(ref_model, ref_group, scope=scope)
    elif ref_family == "onepass":
        npz = _onepass_multiclass_npz_path(ref_model, ref_group, scope=scope)
    else:
        raise ValueError(f"unknown ref_family={ref_family}")

    X, y, pids, _ = load_npz_Xy_pairids(npz)
    out: Dict[str, Set[str]] = {c: set() for c in CATEGORIES}
    if X is None:
        return out
    y = np.asarray(y).astype(str)
    for c in CATEGORIES:
        out[c] = set(pids[y == c].tolist())
    return out


# ---------------- ROI consistency checks (the important part) ----------------

def compare_set(a: Set[str], b: Set[str]) -> Tuple[int, int]:
    return len(a - b), len(b - a)  # missing_from_b, extra_vs_b


def roi_set_for_model_task_category_binary(npz_path: str) -> Tuple[Optional[Set[str]], Optional[str]]:
    X, y, pids, meta_path = load_npz_Xy_pairids(npz_path)
    if X is None:
        return None, None
    s = set(pids[np.asarray(y) == "feature"].tolist())
    return s, meta_path


def roi_set_for_model_category_multiclass(npz_path: str, category: str) -> Tuple[Optional[Set[str]], Optional[str]]:
    X, y, pids, meta_path = load_npz_Xy_pairids(npz_path)
    if X is None:
        return None, None
    y = np.asarray(y).astype(str)
    s = set(pids[y == category].tolist())
    return s, meta_path


def build_roi_consistency_report(
    canonical_by_cat: Dict[str, Set[str]],
    include_multiclass: bool,
    include_multiclass100bproi: bool,
    strict_require_all_canonical: bool,
) -> pd.DataFrame:
    """
    checks (family, model) against canonical ROI sets:
      - binary tasks: per (task, category, scope=roi) compare feature-pair set to canonical
      - multiclass: per (category, scope=roi) compare y==category pair set to canonical
    """
    rows = []

    def add_row(family, model, task, category, scope, s, meta_path):
        canon = canonical_by_cat.get(category, set())
        if canon is None:
            canon = set()
        if s is None:
            rows.append(dict(
                family=family, model=model, task=task, category=category, scope=scope,
                status="missing_npz_or_meta",
                n_canonical=len(canon), n_model=0,
                n_missing=len(canon), n_extra=0,
                meta_path=meta_path or "",
            ))
            return
        n_missing, n_extra = compare_set(canon, s)
        status = "ok"
        if n_missing or n_extra:
            status = "mismatch"
        if strict_require_all_canonical and n_missing:
            status = "missing_canonical"
        rows.append(dict(
            family=family, model=model, task=task, category=category, scope=scope,
            status=status,
            n_canonical=len(canon), n_model=len(s),
            n_missing=n_missing, n_extra=n_extra,
            meta_path=meta_path or "",
        ))

    scope = "roi"
    # ---- all_tasks binary ----
    for model in NT_MODELS:
        group = ALL_TASKS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                npz = _alltasks_binary_npz_path(model, group, cat, task=task, scope=scope)
                s, meta_path = roi_set_for_model_task_category_binary(npz)
                add_row("all_tasks", model, task, cat, scope, s, meta_path)

    # ---- onepass binary ----
    for model_folder in GLOBAL_MODELS:
        mid = to_onepass_model_id(model_folder)
        group = ONEPASS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                npz = _onepass_binary_npz_path(model_folder, group, cat, task=task, scope=scope)
                s, meta_path = roi_set_for_model_task_category_binary(npz)
                add_row("onepass", mid, task, cat, scope, s, meta_path)

    # ---- onepass baselines binary (they should match canonical too) ----
    for baseline in BASELINE_MODELS:
        group = ONEPASS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                npz = _onepass_baseline_binary_npz_path(baseline, group, cat, task=task, scope=scope)
                s, meta_path = roi_set_for_model_task_category_binary(npz)
                add_row("onepass_baseline", baseline, task, cat, scope, s, meta_path)

    # ---- multiclass (roi) ----
    if include_multiclass:
        mscope = "roi"
        # all_tasks multiclass
        for model in NT_MODELS:
            group = ALL_TASKS_GROUP_DEFAULT
            npz = _alltasks_multiclass_npz_path(model, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("all_tasks", model, "multiclass", cat, mscope, s, meta_path)
        # onepass multiclass
        for model_folder in GLOBAL_MODELS:
            mid = to_onepass_model_id(model_folder)
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_multiclass_npz_path(model_folder, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("onepass", mid, "multiclass", cat, mscope, s, meta_path)
        # baselines multiclass
        for baseline in BASELINE_MODELS:
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_baseline_multiclass_npz_path(baseline, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("onepass_baseline", baseline, "multiclass", cat, mscope, s, meta_path)

    # ---- multiclass100bproi (roi100bp) ----
    if include_multiclass100bproi:
        mscope = "roi100bp"
        # all_tasks multiclass
        for model in NT_MODELS:
            group = ALL_TASKS_GROUP_DEFAULT
            npz = _alltasks_multiclass_npz_path(model, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("all_tasks", model, "multiclass100bproi", cat, mscope, s, meta_path)
        # onepass multiclass
        for model_folder in GLOBAL_MODELS:
            mid = to_onepass_model_id(model_folder)
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_multiclass_npz_path(model_folder, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("onepass", mid, "multiclass100bproi", cat, mscope, s, meta_path)

        # baselines probably don't have roi100bp multiclass; still attempt
        for baseline in BASELINE_MODELS:
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_baseline_multiclass_npz_path(baseline, group, scope=mscope)
            for cat in CATEGORIES:
                s, meta_path = roi_set_for_model_category_multiclass(npz, category=cat)
                add_row("onepass_baseline", baseline, "multiclass100bproi", cat, mscope, s, meta_path)

    return pd.DataFrame(rows)


# ---------------- strict evaluation ----------------

def filter_binary_to_canonical_pairs(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
    canonical_feature_pairs: Set[str],
    neg_label: str,
    rng: np.random.Generator,
    cap_per_class: Optional[int],
    strict_require_all_canonical: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    strict binary evaluation:
    - require ROI feature pair_ids == canonical (if strict_require_all_canonical)
    - evaluation uses intersection that has both feature+neg rows per pair_id
    """
    y = np.asarray(y).astype(str)
    pair_ids = np.asarray(pair_ids).astype(str)

    # compute feature set before filtering (for strict check)
    feature_pairs_full = set(pair_ids[y == "feature"].tolist())
    missing_vs_canon = len(canonical_feature_pairs - feature_pairs_full)
    extra_vs_canon = len(feature_pairs_full - canonical_feature_pairs)

    stats = dict(
        missing_canonical_feature_pairs=missing_vs_canon,
        extra_feature_pairs_vs_canonical=extra_vs_canon,
        n_canonical=len(canonical_feature_pairs),
        n_feature_full=len(feature_pairs_full),
    )

    if strict_require_all_canonical and missing_vs_canon > 0:
        return None, None, {**stats, "reason": "missing_canonical_feature_pairs"}

    # restrict all rows to canonical pair_ids
    keep = np.array([pid in canonical_feature_pairs for pid in pair_ids], dtype=bool)
    X, y, pair_ids = X[keep], y[keep], pair_ids[keep]

    pos_mask = (y == "feature")
    neg_mask = (y == neg_label)

    stats.update(dict(
        n_rows_after_canonical_filter=int(X.shape[0]),
        n_feature_rows_after_filter=int(np.sum(pos_mask)),
        n_neg_rows_after_filter=int(np.sum(neg_mask)),
        n_unique_pairs_after_filter=int(len(set(pair_ids.tolist()))),
    ))

    if not np.any(pos_mask) or not np.any(neg_mask):
        return None, None, {**stats, "reason": "missing_label_after_filter"}

    # pid -> indices
    pid_to_pos = {}
    pid_to_neg = {}
    for i in np.where(pos_mask)[0]:
        pid_to_pos.setdefault(pair_ids[i], []).append(i)
    for i in np.where(neg_mask)[0]:
        pid_to_neg.setdefault(pair_ids[i], []).append(i)

    common = sorted(set(pid_to_pos.keys()) & set(pid_to_neg.keys()))
    stats["n_common_pairs_feature_and_neg"] = int(len(common))
    if len(common) < 5:
        return None, None, {**stats, "reason": "too_few_common_pairs"}

    # pick one per pid per side (handle duplicates)
    picked_pos = [rng.choice(pid_to_pos[pid]) for pid in common]
    picked_neg = [rng.choice(pid_to_neg[pid]) for pid in common]

    Xpos = X[picked_pos]
    Xneg = X[picked_neg]

    n = min(len(Xpos), len(Xneg))
    if n < 5:
        return None, None, {**stats, "reason": "too_few_pairs_after_pick"}

    # balance (they're already same length, but keep logic)
    take = n
    if cap_per_class is not None:
        take = min(take, cap_per_class)

    sel = rng.choice(n, size=take, replace=False)
    Xb = np.concatenate([Xpos[sel], Xneg[sel]], axis=0)
    yb = np.array(["feature"] * take + [neg_label] * take, dtype=object)

    stats["n_pairs_used"] = int(take)
    return Xb, yb, stats


def filter_multiclass_to_canonical(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
    canonical_by_cat: Dict[str, Set[str]],
    cap_per_class: Optional[int],
    rng: np.random.Generator,
    strict_require_all_canonical: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    strict multiclass evaluation:
    - for each category c:
        keep rows where (y==c) AND (pair_id in canonical_by_cat[c])
        optionally require exact match to canonical set (no missing) if strict_require_all_canonical
        downsample to cap_per_class if set
    - concatenate and compute BA
    """
    y = np.asarray(y).astype(str)
    pair_ids = np.asarray(pair_ids).astype(str)

    stats = {}
    X_list, y_list = [], []
    per_cat = []

    for c in CATEGORIES:
        canon = canonical_by_cat.get(c, set())
        if not canon:
            per_cat.append(dict(category=c, status="no_canonical", n_kept=0, n_canonical=0, n_missing=0, n_extra=0))
            continue

        mask_c = (y == c)
        pids_c = pair_ids[mask_c]
        # set present for this label
        present_c = set(pids_c.tolist())

        n_missing = len(canon - present_c)
        n_extra = len(present_c - canon)

        if strict_require_all_canonical and n_missing > 0:
            per_cat.append(dict(category=c, status="missing_canonical", n_kept=0, n_canonical=len(canon), n_missing=n_missing, n_extra=n_extra))
            return None, None, {
                "reason": "missing_canonical_in_multiclass",
                "per_category": per_cat,
            }

        # keep only canonical pids
        keep_pids = canon & present_c
        if len(keep_pids) < 3:
            per_cat.append(dict(category=c, status="too_few_after_intersection", n_kept=len(keep_pids), n_canonical=len(canon), n_missing=n_missing, n_extra=n_extra))
            continue

        idx_c = np.where(mask_c)[0]
        # map pid -> one index (dedupe)
        pid_to_idx = {}
        for i in idx_c:
            pid = pair_ids[i]
            if pid in keep_pids:
                pid_to_idx.setdefault(pid, []).append(i)

        chosen = [rng.choice(pid_to_idx[pid]) for pid in sorted(pid_to_idx.keys())]
        if cap_per_class is not None and len(chosen) > cap_per_class:
            chosen = rng.choice(np.array(chosen), size=cap_per_class, replace=False).tolist()

        X_list.append(X[chosen])
        y_list.append(np.full(len(chosen), c, dtype=object))
        per_cat.append(dict(category=c, status="ok", n_kept=len(chosen), n_canonical=len(canon), n_missing=n_missing, n_extra=n_extra))

    if not X_list:
        return None, None, {"reason": "no_categories_kept", "per_category": per_cat}

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    if X_all.shape[0] < 3:
        return None, None, {"reason": "too_few_total", "per_category": per_cat}

    stats["per_category"] = per_cat
    stats["n_total"] = int(X_all.shape[0])
    stats["n_categories_used"] = int(len(set(y_all.tolist())))
    return X_all, y_all, stats


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
        var_global = float(np.sum(var_i) / (K**2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0
        std_across = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(dict(
            Model=model, Family=family, Group=group, Scope=scope,
            N_Categories=K,
            GlobalBalancedAccuracyPct=ba_global,
            GlobalBalancedAccuracyStdPct=std_across,
            GlobalBalancedAccuracySEPct=se_global,
        ))
    return pd.DataFrame(summaries)


# ---------------- evaluation collectors ----------------

def eval_binary_npz(
    family: str,
    model_name: str,
    group: str,
    scope: str,
    task: str,
    category: str,
    npz_path: str,
    canonical_pairs: Set[str],
    rng: np.random.Generator,
    cap_per_class: Optional[int],
    strict_require_all_canonical: bool,
) -> Tuple[Optional[Dict], Dict]:
    X, y, pids, meta_path = load_npz_Xy_pairids(npz_path)
    cov = dict(
        Family=family, Model=model_name, Group=group, Scope=scope, Task=task, Category=category,
        npz_path=npz_path, meta_path=meta_path or "",
        CanonicalPairs=len(canonical_pairs),
    )
    if X is None:
        cov["EvalStatus"] = "missing_npz_or_meta"
        return None, cov

    Xb, yb, stats = filter_binary_to_canonical_pairs(
        X, y, pids,
        canonical_feature_pairs=canonical_pairs,
        neg_label=task,
        rng=rng,
        cap_per_class=cap_per_class,
        strict_require_all_canonical=strict_require_all_canonical,
    )
    cov.update({f"filter_{k}": v for k, v in stats.items()})

    if Xb is None:
        cov["EvalStatus"] = f"skip_{stats.get('reason','unknown')}"
        return None, cov

    ba, se = compute_ba_and_se(Xb, yb)
    row = dict(
        Model=model_name, Family=family, Group=group, Category=category, Scope=scope,
        BA_pct=ba, BA_SE_pct=se,
        N_pos=int(np.sum(yb == "feature")),
        N_neg=int(np.sum(yb == task)),
    )
    cov["EvalStatus"] = "ok"
    return row, cov


def eval_multiclass_npz(
    family: str,
    model_name: str,
    group: str,
    scope: str,
    npz_path: str,
    canonical_by_cat: Dict[str, Set[str]],
    rng: np.random.Generator,
    cap_per_class: Optional[int],
    strict_require_all_canonical: bool,
) -> Tuple[Optional[Dict], Dict, Optional[pd.DataFrame]]:
    X, y, pids, meta_path = load_npz_Xy_pairids(npz_path)
    cov = dict(
        Family=family, Model=model_name, Group=group, Scope=scope, Task="multiclass" if scope == "roi" else "multiclass100bproi",
        npz_path=npz_path, meta_path=meta_path or "",
    )
    if X is None:
        cov["EvalStatus"] = "missing_npz_or_meta"
        return None, cov, None

    Xf, yf, stats = filter_multiclass_to_canonical(
        X, y, pids,
        canonical_by_cat=canonical_by_cat,
        cap_per_class=cap_per_class,
        rng=rng,
        strict_require_all_canonical=strict_require_all_canonical,
    )
    cov["filter_reason"] = stats.get("reason", "")
    cov["filter_n_total"] = stats.get("n_total", 0)
    cov["filter_n_categories_used"] = stats.get("n_categories_used", 0)

    per_cat_df = None
    if "per_category" in stats:
        per_cat_df = pd.DataFrame(stats["per_category"])
        per_cat_df.insert(0, "family", family)
        per_cat_df.insert(1, "model", model_name)
        per_cat_df.insert(2, "group", group)
        per_cat_df.insert(3, "scope", scope)

    if Xf is None:
        cov["EvalStatus"] = f"skip_{stats.get('reason','unknown')}"
        return None, cov, per_cat_df

    ba, se = compute_ba_and_se(Xf, yf)
    row = dict(
        Model=model_name, Family=f"{family}_multiclass", Group=group, Category="__ALL__", Scope=scope,
        BA_pct=ba, BA_SE_pct=se,
        N_pos=int(Xf.shape[0]),  # not really pos/neg in multiclass
        N_neg=int(0),
    )
    cov["EvalStatus"] = "ok"
    return row, cov, per_cat_df


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="strict ROI matching across upstream/random/random-noannot/multiclass")

    # canonical source
    ap.add_argument("--ref_family", choices=["all_tasks", "onepass"], required=True)
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--ref_group", default="all")

    ap.add_argument(
        "--canonical_source",
        choices=["binary", "multiclass"],
        default="binary",
        help="how to define canonical ROI sets (binary uses y=='feature'; multiclass uses y==category).",
    )
    ap.add_argument(
        "--canonical_binary_task",
        choices=BINARY_TASKS,
        default="upstream",
        help="if canonical_source=binary, which binary task to use for defining canonical ROI sets",
    )
    ap.add_argument(
        "--canonical_scope",
        choices=["roi", "full", "roi100bp"],
        default="roi",
        help="scope used to define canonical sets (usually roi).",
    )

    # checks + eval
    ap.add_argument("--include_multiclass", action="store_true", help="check/eval multiclass (roi)")
    ap.add_argument("--include_multiclass100bproi", action="store_true", help="check/eval multiclass (roi100bp)")

    ap.add_argument(
        "--strict_require_all_canonical",
        action="store_true",
        help="if set: any missing canonical ROI pair_ids causes skip (enforces 'same ROIs everywhere')",
    )

    ap.add_argument("--cap_per_class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--outdir",
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_balacc_strict_roi_alltasks",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) canonical ROI sets
    if args.canonical_source == "binary":
        if args.canonical_scope not in ("roi", "full"):
            raise SystemExit("canonical_scope must be roi|full when canonical_source=binary")
        canonical_by_cat = build_canonical_roi_sets_from_binary(
            ref_family=args.ref_family,
            ref_model=args.ref_model,
            ref_group=args.ref_group,
            scope=args.canonical_scope,
            ref_task=args.canonical_binary_task,
        )
    else:
        if args.canonical_scope not in ("roi", "roi100bp"):
            raise SystemExit("canonical_scope must be roi|roi100bp when canonical_source=multiclass")
        canonical_by_cat = build_canonical_roi_sets_from_multiclass(
            ref_family=args.ref_family,
            ref_model=args.ref_model,
            ref_group=args.ref_group,
            scope=args.canonical_scope,
        )

    canon_stats = pd.DataFrame([dict(category=c, canonical_n_pairs=len(canonical_by_cat.get(c, set()))) for c in CATEGORIES])
    canon_stats_path = os.path.join(args.outdir, "canonical_roi_pair_sets_stats.tsv")
    canon_stats.to_csv(canon_stats_path, sep="\t", index=False)

    # 2) ROI consistency report across all tasks (this is the proof you want)
    roi_rep = build_roi_consistency_report(
        canonical_by_cat=canonical_by_cat,
        include_multiclass=args.include_multiclass,
        include_multiclass100bproi=args.include_multiclass100bproi,
        strict_require_all_canonical=args.strict_require_all_canonical,
    )
    roi_rep_path = os.path.join(args.outdir, "roi_pairid_consistency_across_tasks.tsv")
    roi_rep.to_csv(roi_rep_path, sep="\t", index=False)

    # 3) strict BA for binary tasks (always computed)
    binary_rows = []
    binary_covs = []
    scope = "roi"

    # all_tasks
    for model in NT_MODELS:
        group = ALL_TASKS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                canon = canonical_by_cat.get(cat, set())
                if not canon:
                    continue
                npz = _alltasks_binary_npz_path(model, group, cat, task=task, scope=scope)
                row, cov = eval_binary_npz(
                    family="all_tasks",
                    model_name=model,
                    group=group,
                    scope=scope,
                    task=task,
                    category=cat,
                    npz_path=npz,
                    canonical_pairs=canon,
                    rng=rng,
                    cap_per_class=args.cap_per_class,
                    strict_require_all_canonical=args.strict_require_all_canonical,
                )
                binary_covs.append(cov)
                if row is not None:
                    row["Task"] = task
                    binary_rows.append(row)

    # onepass
    for model_folder in GLOBAL_MODELS:
        mid = to_onepass_model_id(model_folder)
        group = ONEPASS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                canon = canonical_by_cat.get(cat, set())
                if not canon:
                    continue
                npz = _onepass_binary_npz_path(model_folder, group, cat, task=task, scope=scope)
                row, cov = eval_binary_npz(
                    family="onepass",
                    model_name=mid,
                    group=group,
                    scope=scope,
                    task=task,
                    category=cat,
                    npz_path=npz,
                    canonical_pairs=canon,
                    rng=rng,
                    cap_per_class=args.cap_per_class,
                    strict_require_all_canonical=args.strict_require_all_canonical,
                )
                binary_covs.append(cov)
                if row is not None:
                    row["Task"] = task
                    binary_rows.append(row)

    # onepass baselines
    for baseline in BASELINE_MODELS:
        group = ONEPASS_GROUP_DEFAULT
        for task in BINARY_TASKS:
            for cat in CATEGORIES:
                canon = canonical_by_cat.get(cat, set())
                if not canon:
                    continue
                npz = _onepass_baseline_binary_npz_path(baseline, group, cat, task=task, scope=scope)
                row, cov = eval_binary_npz(
                    family="onepass_baseline",
                    model_name=baseline,
                    group=group,
                    scope=scope,
                    task=task,
                    category=cat,
                    npz_path=npz,
                    canonical_pairs=canon,
                    rng=rng,
                    cap_per_class=args.cap_per_class,
                    strict_require_all_canonical=args.strict_require_all_canonical,
                )
                binary_covs.append(cov)
                if row is not None:
                    row["Task"] = task
                    binary_rows.append(row)

    binary_df = pd.DataFrame(binary_rows) if binary_rows else pd.DataFrame()
    binary_cov_df = pd.DataFrame(binary_covs) if binary_covs else pd.DataFrame()

    binary_cov_path = os.path.join(args.outdir, "strict_binary_coverage.tsv")
    binary_cov_df.to_csv(binary_cov_path, sep="\t", index=False)

    if not binary_df.empty:
        binary_percat_path = os.path.join(args.outdir, "balacc_strict_binary_per_category.tsv")
        binary_df.to_csv(binary_percat_path, sep="\t", index=False)

        # aggregate per task separately
        globals_out = []
        for task, sub in binary_df.groupby("Task"):
            g = aggregate_per_category(sub.drop(columns=["Task"]))
            g.insert(0, "Task", task)
            globals_out.append(g)
        binary_global = pd.concat(globals_out, axis=0) if globals_out else pd.DataFrame()
        binary_global_path = os.path.join(args.outdir, "balacc_strict_binary_global.tsv")
        binary_global.to_csv(binary_global_path, sep="\t", index=False)
    else:
        binary_percat_path = ""
        binary_global_path = ""

    # 4) strict multiclass BA (optional)
    multiclass_rows = []
    multiclass_covs = []
    multiclass_percat_details = []

    def run_multiclass(scope_mc: str):
        # all_tasks
        for model in NT_MODELS:
            group = ALL_TASKS_GROUP_DEFAULT
            npz = _alltasks_multiclass_npz_path(model, group, scope=scope_mc)
            row, cov, per_cat_df = eval_multiclass_npz(
                family="all_tasks",
                model_name=model,
                group=group,
                scope=scope_mc,
                npz_path=npz,
                canonical_by_cat=canonical_by_cat,
                rng=rng,
                cap_per_class=args.cap_per_class,
                strict_require_all_canonical=args.strict_require_all_canonical,
            )
            multiclass_covs.append(cov)
            if per_cat_df is not None:
                multiclass_percat_details.append(per_cat_df)
            if row is not None:
                multiclass_rows.append(row)

        # onepass
        for model_folder in GLOBAL_MODELS:
            mid = to_onepass_model_id(model_folder)
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_multiclass_npz_path(model_folder, group, scope=scope_mc)
            row, cov, per_cat_df = eval_multiclass_npz(
                family="onepass",
                model_name=mid,
                group=group,
                scope=scope_mc,
                npz_path=npz,
                canonical_by_cat=canonical_by_cat,
                rng=rng,
                cap_per_class=args.cap_per_class,
                strict_require_all_canonical=args.strict_require_all_canonical,
            )
            multiclass_covs.append(cov)
            if per_cat_df is not None:
                multiclass_percat_details.append(per_cat_df)
            if row is not None:
                multiclass_rows.append(row)

        # baselines
        for baseline in BASELINE_MODELS:
            group = ONEPASS_GROUP_DEFAULT
            npz = _onepass_baseline_multiclass_npz_path(baseline, group, scope=scope_mc)
            row, cov, per_cat_df = eval_multiclass_npz(
                family="onepass_baseline",
                model_name=baseline,
                group=group,
                scope=scope_mc,
                npz_path=npz,
                canonical_by_cat=canonical_by_cat,
                rng=rng,
                cap_per_class=args.cap_per_class,
                strict_require_all_canonical=args.strict_require_all_canonical,
            )
            multiclass_covs.append(cov)
            if per_cat_df is not None:
                multiclass_percat_details.append(per_cat_df)
            if row is not None:
                multiclass_rows.append(row)

    if args.include_multiclass:
        run_multiclass("roi")
    if args.include_multiclass100bproi:
        run_multiclass("roi100bp")

    if multiclass_covs:
        pd.DataFrame(multiclass_covs).to_csv(os.path.join(args.outdir, "strict_multiclass_coverage.tsv"), sep="\t", index=False)
    if multiclass_percat_details:
        pd.concat(multiclass_percat_details, axis=0).to_csv(
            os.path.join(args.outdir, "strict_multiclass_per_category_details.tsv"),
            sep="\t",
            index=False,
        )
    if multiclass_rows:
        mc_df = pd.DataFrame(multiclass_rows)
        mc_df.to_csv(os.path.join(args.outdir, "balacc_strict_multiclass_per_model.tsv"), sep="\t", index=False)

        # aggregate "globally" (here it's already per-model single number per scope)
        # but keep a stable file anyway
        mc_global = mc_df.rename(columns={"Category": "CategoryDummy"})
        mc_global.to_csv(os.path.join(args.outdir, "balacc_strict_multiclass_global.tsv"), sep="\t", index=False)

    # summary prints
    print(f"[wrote] {canon_stats_path}")
    print(f"[wrote] {roi_rep_path}")
    print(f"[wrote] {binary_cov_path}")
    if binary_percat_path:
        print(f"[wrote] {binary_percat_path}")
        print(f"[wrote] {binary_global_path}")
    if args.include_multiclass or args.include_multiclass100bproi:
        print(f"[wrote] {os.path.join(args.outdir, 'strict_multiclass_coverage.tsv')}")
        print(f"[wrote] {os.path.join(args.outdir, 'strict_multiclass_per_category_details.tsv')}")
        print(f"[wrote] {os.path.join(args.outdir, 'balacc_strict_multiclass_per_model.tsv')}")
        print(f"[wrote] {os.path.join(args.outdir, 'balacc_strict_multiclass_global.tsv')}")


if __name__ == "__main__":
    main()


# python src/evaluation/plotting/gather_BAs_strict.py \
#   --ref_family onepass \
#   --ref_model gamba_dual_step_random_init \
#   --ref_group all \
#   --canonical_source binary \
#   --canonical_binary_task upstream \
#   --canonical_scope roi \
#   --include_multiclass \
#   --strict_require_all_canonical \
#   --cap_per_class 1000
