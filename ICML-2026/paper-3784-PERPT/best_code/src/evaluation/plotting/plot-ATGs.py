#!/usr/bin/env python3
"""
consolidate_atg4_leaderboard.py

- loads multiple ATG 5-way rep outputs (merges labels 2+3 into single "noncoding" class)
- computes loo 1-NN (cosine) hard metrics WITH ERROR BARS (no soft scoring for 4-way)
- plots (BOTH png + svg):
  - raw leaderboard (hard BA)
  - overlay leaderboard (hard BA): trained (darker) + random-init (lighter)
  - per-model knn heatmaps (svg; also png if you want)

- outputs comprehensive TSV with all metrics in percentage format

4-way labels:
1. START CODON (label 1)
2. NON-CODING ATG (merge labels 2+3)
3. IN-FRAME METHIONINE (label 4 -> 3)
4. OUT-OF-FRAME ATG (label 5 -> 4)

--use_6mer_roi:
  searches for _6mer-suffixed rep files instead of standard ones,
  and appends _6mer to all output filenames/dirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, matthews_corrcoef

from pyfaidx import Fasta


# ---------------- config: label naming + order (4-way) ----------------

LABEL_ORDER_4WAY = [1, 2, 3, 4]

LABEL_NAME_4WAY = {
    1: "START CODON",
    2: "NON-CODING ATG",
    3: "IN-FRAME METHIONINE",
    4: "OUT-OF-FRAME ATG",
}


def remap_5way_to_4way(labels_5way: np.ndarray) -> np.ndarray:
    """
    Remap 5-way labels to 4-way:
    - 1 (start) -> 1
    - 2 (noncoding near) -> 2
    - 3 (noncoding far) -> 2
    - 4 (inframe) -> 3
    - 5 (outframe) -> 4
    """
    labels_4way = labels_5way.copy()
    labels_4way[labels_5way == 3] = 2
    labels_4way[labels_5way == 4] = 3
    labels_4way[labels_5way == 5] = 4
    return labels_4way


# ---------------- model meta ----------------

MODEL_META = {
    "gamba_seq_only_step44000": dict(
        label="ArGamba NTP-only", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_seq_only_step0": dict(
        label="ArGamba NTP-only Random-Init", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_cons_only_step44000": dict(
        label="ArGamba CEP-only", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=False),
    "gamba_cons_only_step0": dict(
        label="ArGamba CEP-only Random-Init", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=True),

    "gamba_dual_step44000": dict(
        label="ArGamba NTP+CEP", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=False),
    "gamba_dual_step0": dict(
        label="ArGamba NTP+CEP Random-Init", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=True),

    # bi-gamba
    "caduceus_seq_only_step44000": dict(
        label="Bi-Gamba MLM-only", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=False),
    "caduceus_seq_only_step0": dict(
        label="Bi-Gamba MLM-only Random-Init", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=True),

    "caduceus_cons_only_step44000": dict(
        label="Bi-Gamba MEM-only", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=False),
    "caduceus_cons_only_step0": dict(
        label="Bi-Gamba MEM-only Random-Init", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=True),

    "caduceus_dual_step44000": dict(
        label="Bi-Gamba MLM+MEM", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=False),
    "caduceus_dual_step0": dict(
        label="Bi-Gamba MLM+MEM Random-Init", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=True),

    # NT / PhyloGPN / others
    "nt-ms": dict(
        label="NT multi-species", family="Other", kind="seq_only",
        params=498_345_436, context=1000, random_init=False),
    "nt-ms-random-init": dict(
        label="NT multi-species Random-Init", family="Other", kind="seq_only",
        params=498_345_436, context=1000, random_init=True),

    "nt-human": dict(
        label="NT human-ref", family="Other", kind="seq_only",
        params=480_438_241, context=1000, random_init=False),
    "nt-human-random-init": dict(
        label="NT human-ref Random-Init", family="Other", kind="seq_only",
        params=480_438_241, context=1000, random_init=True),

    "evo2": dict(
        label="Evo2", family="Other", kind="seq_only",
        params=7_000_000_000, context=2048, random_init=False),

    "caduceus-theirs": dict(
        label="Caduceus", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=False),
    "caduceus-theirs-random-init": dict(
        label="Caduceus Random-Init", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=True),

    "phyloGPN": dict(
        label="PhyloGPN", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=False),
    "phyloGPN-random-init": dict(
        label="PhyloGPN Random-Init", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=True),

    # baselines
    "kmer6": dict(
        label="K-mer (k=6)", family="Other", kind="baseline_kmer",
        params=0, context=2048, random_init=False),
    "phylop": dict(
        label="PhyloP (6D)", family="Other", kind="baseline_phylop",
        params=0, context=2048, random_init=False),
}


# ---------------- colors ----------------

BLUE   = "#4287f5"
PURPLE = "#6F2DA8"
ORANGE = "#FF8C32"
DARK   = "#6A6A6A"


def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )


def lighten_hex(hex_color: str, amount: float = 0.60) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return _rgb01_to_hex((r, g, b))


def base_color_for(kind: str) -> str:
    if str(kind).startswith("baseline"):
        return DARK
    if kind == "seq_plus_phy":
        return BLUE
    if kind == "phy_only":
        return PURPLE
    if kind == "seq_only":
        return ORANGE
    return "#B0B0B0"


# ---------------- model key canonicalization ----------------

def _canonicalize_model_key(model_tag: str) -> str:
    s = str(model_tag)

    if s in MODEL_META:
        return s

    random_suffixes = [
        "_steprandom_init",
        "_step_random_init",
        "_random_init",
        "_random-init",
        "-random-init",
        "_randominit",
        "random_init",
    ]
    for suf in random_suffixes:
        if s.endswith(suf):
            base = s[: -len(suf)]
            cand = base + "_step0"
            if cand in MODEL_META:
                return cand

    if s.endswith("step0"):
        base = s[: -len("step0")].rstrip("_-")
        cand = base + "_step0"
        if cand in MODEL_META:
            return cand

    if s.endswith("_step44000") and s in MODEL_META:
        return s

    return s


def _base_label_from_meta_label(label: str) -> str:
    return label[:-len(" Random-Init")] if label.endswith(" Random-Init") else label


def _is_random_init_from_meta_or_name(model_key: str) -> bool:
    if model_key in MODEL_META:
        return bool(MODEL_META[model_key]["random_init"])
    return ("random-init" in model_key) or ("random_init" in model_key) or model_key.endswith("_step0")


# ---------------- loading helpers ----------------

@dataclass
class RepFile:
    model_tag: str
    npz_path: Path
    meta_path: Optional[Path]


# Anchors tried in order when inferring model tag from filename.
# The _6mer variants must come first so they are matched before the non-suffixed ones.
_FILENAME_ANCHORS = [
    "_ATG5way_all_labels_roi_6mer",
    "_ATG5way_all_labels_full_6mer",
    "_ATG_5way_all_labels_6mer",
    "_ATG5way_all_labels_6mer",
    "_ATG5way_all_labels_roi",
    "_ATG5way_all_labels_full",
    "_ATG_5way_all_labels",
    "_ATG5way_all_labels",
]


def _infer_model_tag_from_filename(npz_path: Path) -> str:
    name = npz_path.name
    if not name.startswith("reps_") or not name.endswith(".npz"):
        return npz_path.stem

    stem = name[:-4]  # drop .npz

    for anchor in _FILENAME_ANCHORS:
        if anchor in stem:
            return stem[len("reps_") : stem.index(anchor)]

    m = re.match(r"^reps_(.+?)_ATG", stem)
    if m:
        return m.group(1)

    return stem[len("reps_"):]


def _glob_patterns_for(use_6mer_roi: bool) -> list[str]:
    """Return the glob patterns used to discover rep files."""
    if use_6mer_roi:
        return [
            "**/reps_*_ATG5way_all_labels_roi_6mer.npz",
            "**/reps_*_ATG_5way_all_labels_6mer.npz",
        ]
    else:
        return [
            "**/reps_*_ATG5way_all_labels_roi.npz",
            "**/reps_*_ATG_5way_all_labels.npz",
        ]


def discover_rep_files(roots: list[Path], use_6mer_roi: bool = False) -> list[RepFile]:
    patterns = _glob_patterns_for(use_6mer_roi)
    out: list[RepFile] = []
    seen = set()

    for root in roots:
        for pat in patterns:
            for npz in root.glob(pat):
                npz = npz.resolve()
                if npz in seen:
                    continue
                seen.add(npz)

                model_tag = _infer_model_tag_from_filename(npz)
                if "hyenadna" in model_tag.lower():
                    continue

                meta = npz.with_name(npz.stem + "_meta.parquet")
                meta_path = meta if meta.exists() else None
                out.append(RepFile(model_tag=model_tag, npz_path=npz, meta_path=meta_path))

    out.sort(key=lambda r: (r.model_tag, str(r.npz_path)))
    return out


def load_embeddings_and_labels(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)

    if "embeddings" in d:
        X = d["embeddings"]
    elif "X" in d:
        X = d["X"]
    else:
        raise KeyError(f"{npz_path}: cannot find embeddings key")

    if "labels" in d:
        y = d["labels"]
    elif "y" in d:
        y = d["y"]
    else:
        raise KeyError(f"{npz_path}: cannot find labels key")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2:
        raise ValueError(f"{npz_path}: embeddings must be [N,H], got shape={X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"{npz_path}: labels must be [N]")

    return X, y


# ---------------- metrics WITH ERROR BARS ----------------

def loo_1nn_predictions(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    return y, y[idx[:, 1]]


def hard_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_order: list[int]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()

    valid = ~np.isnan(per_class_recall)
    ba = float(np.mean(per_class_recall[valid])) if np.any(valid) else float("nan")
    sem = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid))) if np.sum(valid) > 1 else 0.0
    ci95 = float(1.96 * sem)

    return {
        "micro_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": ba,
        "balanced_accuracy_sem": sem,
        "balanced_accuracy_ci95": ci95,
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": {int(l): float(r) for l, r in zip(label_order, per_class_recall)},
        "support": {int(l): int(s) for l, s in zip(label_order, cm.sum(axis=1))},
        "cm": cm,
    }


# ---------------- plotting helpers ----------------

def _save_fig(fig: plt.Figure, outbase: Path, dpi: int = 300):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi)
    fig.savefig(outbase.with_suffix(".svg"))
    plt.close(fig)


def plot_knn_heatmap(outbase: Path, y_true: np.ndarray, y_pred: np.ndarray, label_order: list[int], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    xt = [LABEL_NAME_4WAY[l] for l in label_order]
    yt = [LABEL_NAME_4WAY[l] for l in label_order]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    sns.heatmap(
        acc_matrix,
        ax=ax,
        xticklabels=xt,
        yticklabels=yt,
        vmin=0,
        vmax=1.0,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, outbase)


def plot_leaderboard(df: pd.DataFrame, outbase: Path, metric: str, sem_col: str, title: str, xlabel: str, top_k: int | None = None):
    dfp = df.sort_values(metric, ascending=False).copy()
    if top_k is not None:
        dfp = dfp.head(top_k)

    fig_h = max(5.0, 0.35 * len(dfp))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ylab = dfp["model_tag"].iloc[::-1]
    vals = dfp[metric].iloc[::-1].to_numpy(dtype=float)
    sems = dfp[sem_col].iloc[::-1].to_numpy(dtype=float)

    ax.barh(ylab, vals, xerr=sems, error_kw=dict(ecolor="black", lw=1, capsize=3, capthick=1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)

    _save_fig(fig, outbase)


def base_model_key_fallback(model_tag: str) -> str:
    s = str(model_tag)
    for suf in ["-random-init", "_random-init", "_random_init", "_steprandom_init", "_step_random_init"]:
        if s.endswith(suf):
            return s[: -len(suf)]
    if s.endswith("_step0"):
        return s[: -len("_step0")]
    if s.endswith("step0"):
        return s[: -len("step0")].rstrip("_-")
    return s


def kind_fallback(model_tag: str) -> str:
    s = str(model_tag).lower()
    if "kmer" in s:
        return "baseline_kmer"
    if "phylop" in s:
        return "baseline_phylop"
    if "dual" in s or "seq_plus_phy" in s:
        return "seq_plus_phy"
    if "cons_only" in s or "phy_only" in s:
        return "phy_only"
    return "seq_only"


def plot_atg4_leaderboard_overlay(
    df_rows: pd.DataFrame,
    outbase: Path,
    metric: str,
    sem_col: str,
    title: str,
    xlabel: str,
    top_k: int | None = None,
):
    rows_by_label: dict[str, dict] = {}

    for _, r in df_rows.iterrows():
        raw_tag = str(r["model_tag"])
        model_key = _canonicalize_model_key(raw_tag)

        if model_key in MODEL_META:
            meta = MODEL_META[model_key]
            base_label = _base_label_from_meta_label(meta["label"])
            kind = meta["kind"]
            random_init = meta["random_init"]
        else:
            base_label = base_model_key_fallback(raw_tag)
            kind = kind_fallback(raw_tag)
            random_init = _is_random_init_from_meta_or_name(raw_tag)

        entry = rows_by_label.get(base_label, dict(
            label=base_label,
            kind=kind,
            trained=np.nan,
            trained_sem=0.0,
            rand=np.nan,
            rand_sem=0.0,
        ))

        if random_init:
            entry["rand"] = float(r[metric])
            entry["rand_sem"] = float(r.get(sem_col, 0.0))
        else:
            entry["trained"] = float(r[metric])
            entry["trained_sem"] = float(r.get(sem_col, 0.0))

        rows_by_label[base_label] = entry

    tbl = pd.DataFrame(list(rows_by_label.values()))
    if tbl.empty:
        raise ValueError("no rows to plot")

    sort_key = tbl["trained"].copy()
    sort_key = sort_key.fillna(tbl["rand"])
    tbl = tbl.assign(_sort=sort_key).sort_values("_sort", ascending=True).drop(columns=["_sort"]).reset_index(drop=True)

    if top_k is not None and len(tbl) > top_k:
        tbl = tbl.iloc[-top_k:].reset_index(drop=True)

    n = len(tbl)
    fig_h = max(4.0, 0.35 * n)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))
    y = np.arange(n)

    base_colors = np.array([base_color_for(k) for k in tbl["kind"]], dtype=object)
    rand_colors = np.array([lighten_hex(c, amount=0.60) for c in base_colors], dtype=object)

    trained_vals = tbl["trained"].to_numpy(dtype=float)
    trained_sem = tbl["trained_sem"].to_numpy(dtype=float)
    trained_mask = ~np.isnan(trained_vals)

    ax.barh(
        y[trained_mask],
        trained_vals[trained_mask],
        xerr=trained_sem[trained_mask],
        color=base_colors[trained_mask],
        height=0.78,
        edgecolor="none",
        linewidth=0,
        zorder=2,
        error_kw=dict(ecolor="black", lw=1, capsize=3, capthick=1),
    )

    rand_vals = tbl["rand"].to_numpy(dtype=float)
    rand_sem = tbl["rand_sem"].to_numpy(dtype=float)
    rand_mask = ~np.isnan(rand_vals)

    ax.barh(
        y[rand_mask],
        rand_vals[rand_mask],
        xerr=rand_sem[rand_mask],
        color=rand_colors[rand_mask],
        height=0.46,
        edgecolor="none",
        linewidth=0,
        zorder=3,
        error_kw=dict(ecolor="black", lw=0.8, capsize=2.5, capthick=0.8),
    )

    flat = np.concatenate([trained_vals[~np.isnan(trained_vals)], rand_vals[~np.isnan(rand_vals)]])
    if flat.size:
        xmin = max(0.0, float(flat.min()) - 0.02)
        xmax = float(flat.max()) + 0.02
        ax.set_xlim(xmin, xmax)

    ax.set_yticks(y)
    ax.set_yticklabels(tbl["label"].tolist(), fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    _save_fig(fig, outbase)
    tbl.to_csv(outbase.with_suffix(".csv"), index=False)


# ---------------- tsv output ----------------

def create_comprehensive_tsv(df_rows: pd.DataFrame, per_model: dict, outpath: Path):
    tsv_rows = []

    for _, r in df_rows.iterrows():
        model_tag = r["model_tag"]
        detail = per_model.get(model_tag, {})
        hard = detail.get("hard", {})

        model_key = _canonicalize_model_key(model_tag)
        meta = MODEL_META.get(model_key, {})

        row = {
            "model_tag": model_tag,
            "model_label": meta.get("label", model_tag),
            "family": meta.get("family", "Unknown"),
            "kind": meta.get("kind", "Unknown"),
            "params": meta.get("params", 0),
            "context": meta.get("context", 0),
            "random_init": meta.get("random_init", False),
            "n_samples": r["n"],

            "balanced_accuracy_%": 100.0 * hard.get("balanced_accuracy", np.nan),
            "balanced_accuracy_sem_%": 100.0 * hard.get("balanced_accuracy_sem", 0.0),
            "balanced_accuracy_ci95_%": 100.0 * hard.get("balanced_accuracy_ci95", 0.0),
            "micro_accuracy_%": 100.0 * hard.get("micro_accuracy", np.nan),
            "macro_f1_%": 100.0 * hard.get("macro_f1", np.nan),
            "weighted_f1_%": 100.0 * hard.get("weighted_f1", np.nan),
            "cohens_kappa": hard.get("cohens_kappa", np.nan),
            "mcc": hard.get("mcc", np.nan),
        }

        per_class = hard.get("per_class_recall", {})
        for label_id in LABEL_ORDER_4WAY:
            label_name = LABEL_NAME_4WAY[label_id].replace(" ", "_").replace("-", "_")
            row[f"{label_name}_recall_%"] = 100.0 * per_class.get(label_id, np.nan)

        tsv_rows.append(row)

    df_tsv = pd.DataFrame(tsv_rows).sort_values("balanced_accuracy_%", ascending=False)
    df_tsv.to_csv(outpath, sep="\t", index=False, float_format="%.4f")
    print(f"\n[output] comprehensive tsv written to: {outpath}")
    return df_tsv


# ---------------- roi validation ----------------

def revcomp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]


def validate_roi_is_atg(
    meta_path: Optional[Path],
    fasta: Optional[Fasta],
    model_tag: str,
    strict: bool = False,
    max_print: int = 5,
):
    if meta_path is None or not meta_path.exists():
        return

    meta = pd.read_parquet(meta_path)

    required = {"chrom", "start", "end", "feature_start_in_window", "feature_end_in_window"}
    if not required.issubset(meta.columns):
        return
    if "sequence" not in meta.columns:
        return

    strand_col = None
    for c in ["strand", "ref_strand"]:
        if c in meta.columns:
            strand_col = c
            break

    bad = []
    for _, r in meta.iterrows():
        chrom = r["chrom"]
        fs = int(r["feature_start_in_window"])
        fe = int(r["feature_end_in_window"])
        seq_window = str(r["sequence"]).upper()

        # for 6mer ROI fe-fs == 6; for standard fe-fs == 3
        # ATG check: first 3 chars of ROI should be ATG
        roi_from_window = seq_window[fs:fs + 3]
        ok_window = (roi_from_window == "ATG")

        ok_fasta = None
        if fasta is not None:
            s0, e0 = int(r["start"]), int(r["end"])
            a, b = (s0, e0) if s0 <= e0 else (e0, s0)
            if b - a == 3:
                roi_plus = str(fasta[chrom][a:b]).upper()
                strand = "+"
                if strand_col is not None:
                    st = str(r.get(strand_col, "+")).strip()
                    if st in ["+", "-"]:
                        strand = st
                roi_checked = revcomp(roi_plus) if strand == "-" else roi_plus
                ok_fasta = (roi_checked == "ATG")

        ok = ok_window and (ok_fasta if ok_fasta is not None else True)
        if not ok:
            bad.append(f"{chrom}:{r['start']}-{r['end']} window[fs:fs+3]={roi_from_window}")
            if len(bad) >= max_print:
                break

    if bad:
        msg = f"[atg-check] {model_tag}: non-ATG ROI(s) found:\n" + "\n".join(bad)
        if strict:
            raise ValueError(msg)
        print(msg)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help="one or more root dirs to search (will glob recursively)",
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--write_per_model_json", action="store_true")
    ap.add_argument(
        "--use_6mer_roi",
        action="store_true",
        default=False,
        help=(
            "Search for _6mer-suffixed rep files (reps_*_ATG5way_all_labels_roi_6mer.npz etc.) "
            "instead of standard ones. Appends _6mer to all output filenames."
        ),
    )
    ap.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )

    args = ap.parse_args()

    suffix = "_6mer" if args.use_6mer_roi else ""

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rep_files = discover_rep_files(roots, use_6mer_roi=args.use_6mer_roi)
    if not rep_files:
        patterns = _glob_patterns_for(args.use_6mer_roi)
        raise SystemExit(
            f"no rep files found under roots={roots}\n"
            f"searched patterns: {patterns}\n"
            f"hint: did you forget --use_6mer_roi, or run the embedding jobs yet?"
        )

    print(f"[discover] found {len(rep_files)} rep files (use_6mer_roi={args.use_6mer_roi})")
    for rf in rep_files:
        print(f"  {rf.model_tag}: {rf.npz_path}")

    rows = []
    per_model = {}

    fasta = Fasta(args.genome_fasta, as_raw=True, sequence_always_upper=True)

    for rf in rep_files:
        X, y_5way = load_embeddings_and_labels(rf.npz_path)

        y = remap_5way_to_4way(y_5way)

        mask = np.isin(y, LABEL_ORDER_4WAY)
        X = X[mask]
        y = y[mask]
        if len(y) == 0:
            continue

        y_true, y_pred = loo_1nn_predictions(X, y)
        hard = hard_metrics(y_true, y_pred, LABEL_ORDER_4WAY)

        per_model[rf.model_tag] = {
            "model_tag": rf.model_tag,
            "npz_path": str(rf.npz_path),
            "meta_path": str(rf.meta_path) if rf.meta_path else None,
            "n": int(len(y)),
            "hard": {k: v for k, v in hard.items() if k != "cm"},
        }

        rows.append({
            "model_tag": rf.model_tag,
            "n": int(len(y)),
            "balanced_accuracy": hard["balanced_accuracy"],
            "balanced_accuracy_sem": hard["balanced_accuracy_sem"],
            "balanced_accuracy_ci95": hard["balanced_accuracy_ci95"],
            "micro_accuracy": hard["micro_accuracy"],
            "npz_path": str(rf.npz_path),
        })

        title = (
            f"{rf.model_tag}{suffix} | "
            f"micro={hard['micro_accuracy']:.2%} ba={hard['balanced_accuracy']:.2%}"
        )

        heat_base = outdir / "heatmaps" / f"knn_heatmap_{rf.model_tag}{suffix}"
        plot_knn_heatmap(
            outbase=heat_base,
            y_true=y_true,
            y_pred=y_pred,
            label_order=LABEL_ORDER_4WAY,
            title=title,
        )

        try:
            validate_roi_is_atg(
                rf.meta_path,
                fasta=fasta,
                model_tag=rf.model_tag,
                strict=False,
                max_print=5,
            )
        except Exception as e:
            print(f"[atg-check] {rf.model_tag}: error: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no models produced rows")

    df = df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)

    df.to_csv(outdir / f"leaderboard_atg4{suffix}.csv", index=False)

    plot_leaderboard(
        df=df,
        outbase=outdir / f"leaderboard_atg4_ba{suffix}",
        metric="balanced_accuracy",
        sem_col="balanced_accuracy_sem",
        title=f"ATG 4-way leaderboard (balanced accuracy){' [6-mer ROI]' if suffix else ''}",
        xlabel="balanced accuracy (loo 1-nn)",
        top_k=args.top_k,
    )

    plot_atg4_leaderboard_overlay(
        df_rows=df,
        outbase=outdir / f"leaderboard_atg4_ba_overlay{suffix}",
        metric="balanced_accuracy",
        sem_col="balanced_accuracy_sem",
        title=f"ATG 4-way leaderboard (balanced accuracy){' [6-mer ROI]' if suffix else ''}",
        xlabel="balanced accuracy (loo 1-nn)",
        top_k=args.top_k,
    )

    create_comprehensive_tsv(
        df_rows=df,
        per_model=per_model,
        outpath=outdir / f"atg4_all_metrics{suffix}.tsv"
    )

    with open(outdir / f"leaderboard_atg4_details{suffix}.json", "w") as f:
        json.dump(per_model, f, indent=2)

    if args.write_per_model_json:
        per_model_dir = outdir / f"per_model_json{suffix}"
        per_model_dir.mkdir(exist_ok=True)
        for k, v in per_model.items():
            with open(per_model_dir / f"{k}.json", "w") as f:
                json.dump(v, f, indent=2)

    print(f"\n[done] outputs written to: {outdir}")
    print(f"  leaderboard_atg4{suffix}.csv")
    print(f"  leaderboard_atg4_ba{suffix}.png/svg")
    print(f"  leaderboard_atg4_ba_overlay{suffix}.png/svg")
    print(f"  atg4_all_metrics{suffix}.tsv")
    print(f"  leaderboard_atg4_details{suffix}.json")
    print(f"  heatmaps/knn_heatmap_*{suffix}.png/svg")


if __name__ == "__main__":
    main()

# example (standard ROI):
# python /home/mica/gamba/src/evaluation/plotting/plot-ATGs.py \
#   --roots /home/mica/gamba/other-models/ATG_reps_5way \
#           /home/mica/gamba/data_processing/data/240-mammalian/ATG_reps_5way \
#   --outdir /home/mica/gamba/ATG4_leaderboard_out

# example (6-mer ROI):
# python /home/mica/gamba/src/evaluation/plotting/plot-ATGs.py \
#   --roots /home/mica/gamba/other-models/ATG_reps_5way \
#           /home/mica/gamba/data_processing/data/240-mammalian/ATG_reps_5way \
#   --outdir /home/mica/gamba/ATG4_leaderboard_out \
#   --use_6mer_roi