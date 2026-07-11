#!/usr/bin/env python3
"""
consolidate_atg4_leaderboard_from_full.py

Extracts ATG embeddings from full window embeddings and creates:
1. ATG-only accuracy (3bp, all models)
2. ATG+flanks accuracy (6-9bp context for non-NT models vs NT's native 6-mer tokenization)

This makes fairer comparisons since NT's tokenization includes flanking bases.
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
    """Remap 5-way labels to 4-way"""
    labels_4way = labels_5way.copy()
    labels_4way[labels_5way == 3] = 2  # merge far with near
    labels_4way[labels_5way == 4] = 3  # inframe becomes 3
    labels_4way[labels_5way == 5] = 4  # outframe becomes 4
    return labels_4way


# ---------------- model meta ----------------

MODEL_META = {
    "gamba_seq_only_step44000": dict(
        label="ArGamba NTP-only", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=False, is_nt=False),
    "gamba_seq_only_step0": dict(
        label="ArGamba NTP-only Random-Init", family="Gamba", kind="seq_only",
        params=66_492_392, context=2048, random_init=True, is_nt=False),

    "gamba_cons_only_step44000": dict(
        label="ArGamba CEP-only", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=False, is_nt=False),
    "gamba_cons_only_step0": dict(
        label="ArGamba CEP-only Random-Init", family="Gamba", kind="phy_only",
        params=66_492_392, context=2048, random_init=True, is_nt=False),

    "gamba_dual_step44000": dict(
        label="ArGamba NTP+CEP", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=False, is_nt=False),
    "gamba_dual_step0": dict(
        label="ArGamba NTP+CEP Random-Init", family="Gamba", kind="seq_plus_phy",
        params=66_493_418, context=2048, random_init=True, is_nt=False),

    "caduceus_seq_only_step44000": dict(
        label="Bi-Gamba MLM-only", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=False, is_nt=False),
    "caduceus_seq_only_step0": dict(
        label="Bi-Gamba MLM-only Random-Init", family="Bi-Gamba", kind="seq_only",
        params=3_864_832, context=2048, random_init=True, is_nt=False),

    "caduceus_cons_only_step44000": dict(
        label="Bi-Gamba MEM-only", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=False, is_nt=False),
    "caduceus_cons_only_step0": dict(
        label="Bi-Gamba MEM-only Random-Init", family="Bi-Gamba", kind="phy_only",
        params=3_864_832, context=2048, random_init=True, is_nt=False),

    "caduceus_dual_step44000": dict(
        label="Bi-Gamba MLM+MEM", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=False, is_nt=False),
    "caduceus_dual_step0": dict(
        label="Bi-Gamba MLM+MEM Random-Init", family="Bi-Gamba", kind="seq_plus_phy",
        params=3_869_442, context=2048, random_init=True, is_nt=False),

    "nt-ms": dict(
        label="NT multi-species", family="Other", kind="seq_only",
        params=498_345_436, context=6000, random_init=False, is_nt=True),
    "nt-ms-random-init": dict(
        label="NT multi-species Random-Init", family="Other", kind="seq_only",
        params=498_345_436, context=6000, random_init=True, is_nt=True),

    "nt-human": dict(
        label="NT human-ref", family="Other", kind="seq_only",
        params=480_438_241, context=6000, random_init=False, is_nt=True),
    "nt-human-random-init": dict(
        label="NT human-ref Random-Init", family="Other", kind="seq_only",
        params=480_438_241, context=6000, random_init=True, is_nt=True),

    "evo2": dict(
        label="Evo2", family="Other", kind="seq_only",
        params=7_000_000_000, context=2048, random_init=False, is_nt=False),

    "caduceus-theirs": dict(
        label="Caduceus", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=False, is_nt=False),
    "caduceus-theirs-random-init": dict(
        label="Caduceus Random-Init", family="Other", kind="seq_only",
        params=7_725_312, context=131_000, random_init=True, is_nt=False),

    "phyloGPN": dict(
        label="PhyloGPN", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=False, is_nt=False),
    "phyloGPN-random-init": dict(
        label="PhyloGPN Random-Init", family="Other", kind="seq_only",
        params=83_185_924, context=481, random_init=True, is_nt=False),

    "kmer6": dict(
        label="K-mer (k=6)", family="Other", kind="baseline_kmer",
        params=0, context=2048, random_init=False, is_nt=False),
    "kmer6_flanked": dict(
        label="K-mer (k=6, flanked)", family="Other", kind="baseline_kmer",
        params=0, context=2048, random_init=False, is_nt=False),
    "phylop": dict(
        label="PhyloP (6D)", family="Other", kind="baseline_phylop",
        params=0, context=2048, random_init=False, is_nt=False),
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
        "_steprandom_init", "_step_random_init", "_random_init",
        "_random-init", "-random-init", "_randominit", "random_init",
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


def _is_nt_model(model_key: str) -> bool:
    model_key = _canonicalize_model_key(model_key)
    if model_key in MODEL_META:
        return bool(MODEL_META[model_key].get("is_nt", False))
    return "nt-" in model_key.lower()


# ---------------- loading helpers ----------------

@dataclass
class RepFile:
    model_tag: str
    full_npz: Path
    meta_path: Optional[Path]


def _infer_model_tag_from_filename(npz_path: Path) -> str:
    name = npz_path.name
    if not name.startswith("reps_") or not name.endswith(".npz"):
        return npz_path.stem

    stem = name[:-4]
    for anchor in [
        "_ATG5way_all_labels_full",
        "_ATG_5way_all_labels_full",
        "_ATG5way_all_labels",
    ]:
        if anchor in stem:
            return stem[len("reps_") : stem.index(anchor)]

    m = re.match(r"^reps_(.+?)_ATG", stem)
    if m:
        return m.group(1)

    return stem[len("reps_") :]


def discover_rep_files(roots: list[Path]) -> list[RepFile]:
    """Discover full embedding files"""
    patterns = [
        "**/reps_*_ATG5way_all_labels_full.npz",
        "**/reps_*_ATG_5way_all_labels_full.npz",
    ]
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
                out.append(RepFile(model_tag=model_tag, full_npz=npz, meta_path=meta_path))

    out.sort(key=lambda r: (r.model_tag, str(r.full_npz)))
    return out


def load_full_embeddings_and_metadata(
    npz_path: Path,
    meta_path: Optional[Path]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load full embeddings, labels, and metadata"""
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

    # Load metadata
    if meta_path is None or not meta_path.exists():
        raise ValueError(f"metadata required but not found: {meta_path}")

    meta = pd.read_parquet(meta_path)
    if len(meta) != len(X):
        raise ValueError(f"metadata length mismatch: {len(meta)} vs {len(X)}")

    return X, y, meta


def extract_atg_embedding_from_full(
    full_emb: np.ndarray,
    window_len: int,
    fs: int,
    fe: int,
    flank_bp: int = 0,
) -> np.ndarray:
    """
    Extract ATG (or ATG+flanks) embedding from full window embedding.
    
    full_emb: [T, H] token-level embeddings
    window_len: length of sequence window in bp
    fs, fe: feature start/end in window (bp coordinates)
    flank_bp: how many bp of flanking context to include on each side
    
    Returns: [H] pooled embedding
    """
    T, H = full_emb.shape
    
    # Expand region with flanks
    fs_flanked = max(0, fs - flank_bp)
    fe_flanked = min(window_len, fe + flank_bp)
    
    # Map bp coordinates to token coordinates
    scale = T / float(window_len)
    tfs = max(0, min(int(np.floor(fs_flanked * scale)), T - 1))
    tfe = max(tfs + 1, min(int(np.ceil(fe_flanked * scale)), T))
    
    # Pool tokens in this range
    return full_emb[tfs:tfe].mean(axis=0).astype(np.float32)


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
            cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True),
        )

    xt = [LABEL_NAME_4WAY[l] for l in label_order]
    yt = [LABEL_NAME_4WAY[l] for l in label_order]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    sns.heatmap(
        acc_matrix, ax=ax, xticklabels=xt, yticklabels=yt,
        vmin=0, vmax=1.0, cmap="Blues", annot=True, fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, outbase)


def plot_leaderboard(
    df: pd.DataFrame,
    outbase: Path,
    metric: str,
    sem_col: str,
    title: str,
    xlabel: str,
    top_k: int | None = None
):
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


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="ATG 4-way evaluation from full embeddings")
    ap.add_argument(
        "--roots", type=str, nargs="+", required=True,
        help="directories containing *_full.npz files"
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--flank_bp", type=int, default=3,
        help="bp of flanking context for non-NT models (default: 3)")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--write_per_model_json", action="store_true")

    args = ap.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rep_files = discover_rep_files(roots)
    if not rep_files:
        raise SystemExit(f"no full rep files found under roots={roots}")

    print(f"[info] found {len(rep_files)} models with full embeddings")
    print(f"[info] extracting ATG-only (3bp) and ATG+flanks ({args.flank_bp}bp each side) embeddings")

    rows_atg_only = []
    rows_atg_flanked = []
    per_model = {}

    for rf in rep_files:
        print(f"\n[processing] {rf.model_tag}")
        
        try:
            X_full, y_5way, meta = load_full_embeddings_and_metadata(rf.full_npz, rf.meta_path)
        except Exception as e:
            print(f"[error] {rf.model_tag}: {e}")
            continue

        # Remap to 4-way
        y = remap_5way_to_4way(y_5way)
        mask = np.isin(y, LABEL_ORDER_4WAY)
        X_full = X_full[mask]
        y = y[mask]
        meta = meta.iloc[mask].reset_index(drop=True)

        if len(y) == 0:
            print(f"[warning] {rf.model_tag}: no valid labels")
            continue

        # Extract ATG-only embeddings (3bp)
        X_atg_only = []
        for i in range(len(X_full)):
            row = meta.iloc[i]
            window_len = int(row.get("window_len", 2048))
            fs = int(row["feature_start_in_window"])
            fe = int(row["feature_end_in_window"])
            
            atg_emb = extract_atg_embedding_from_full(
                X_full[i].reshape(-1, X_full.shape[1]),  # [T, H]
                window_len=window_len,
                fs=fs,
                fe=fe,
                flank_bp=0  # ATG only
            )
            X_atg_only.append(atg_emb)
        
        X_atg_only = np.stack(X_atg_only)
        
        # Compute metrics for ATG-only
        y_true, y_pred = loo_1nn_predictions(X_atg_only, y)
        metrics_atg = hard_metrics(y_true, y_pred, LABEL_ORDER_4WAY)
        
        rows_atg_only.append({
            "model_tag": rf.model_tag,
            "n": int(len(y)),
            "balanced_accuracy": metrics_atg["balanced_accuracy"],
            "balanced_accuracy_sem": metrics_atg["balanced_accuracy_sem"],
            "balanced_accuracy_ci95": metrics_atg["balanced_accuracy_ci95"],
            "micro_accuracy": metrics_atg["micro_accuracy"],
        })
        
        per_model[rf.model_tag] = {
            "model_tag": rf.model_tag,
            "full_npz_path": str(rf.full_npz),
            "meta_path": str(rf.meta_path) if rf.meta_path else None,
            "n": int(len(y)),
            "atg_only": {k: v for k, v in metrics_atg.items() if k != "cm"},
        }
        
        # Heatmap for ATG-only
        heat_base = outdir / "heatmaps_atg_only" / f"knn_heatmap_{rf.model_tag}"
        plot_knn_heatmap(
            outbase=heat_base,
            y_true=y_true,
            y_pred=y_pred,
            label_order=LABEL_ORDER_4WAY,
            title=f"{rf.model_tag} (ATG-only, 3bp) | ba={metrics_atg['balanced_accuracy']:.2%}",
        )
        
        # Extract ATG+flanks embeddings
        is_nt = _is_nt_model(rf.model_tag)
        
        if is_nt:
            # NT already has flanks in its tokenization, use ATG-only for fair comparison
            X_atg_flanked = X_atg_only
            flank_note = "native 6-mer"
        else:
            # Non-NT models: extract ATG + flanking context
            X_atg_flanked = []
            for i in range(len(X_full)):
                row = meta.iloc[i]
                window_len = int(row.get("window_len", 2048))
                fs = int(row["feature_start_in_window"])
                fe = int(row["feature_end_in_window"])
                
                atg_flanked = extract_atg_embedding_from_full(
                    X_full[i].reshape(-1, X_full.shape[1]),
                    window_len=window_len,
                    fs=fs,
                    fe=fe,
                    flank_bp=args.flank_bp
                )
                X_atg_flanked.append(atg_flanked)
            
            X_atg_flanked = np.stack(X_atg_flanked)
            flank_note = f"ATG+{args.flank_bp}bp"
        
        # Compute metrics for ATG+flanks
        y_true_f, y_pred_f = loo_1nn_predictions(X_atg_flanked, y)
        metrics_flanked = hard_metrics(y_true_f, y_pred_f, LABEL_ORDER_4WAY)
        
        rows_atg_flanked.append({
            "model_tag": rf.model_tag,
            "context_type": flank_note,
            "n": int(len(y)),
            "balanced_accuracy": metrics_flanked["balanced_accuracy"],
            "balanced_accuracy_sem": metrics_flanked["balanced_accuracy_sem"],
            "balanced_accuracy_ci95": metrics_flanked["balanced_accuracy_ci95"],
            "micro_accuracy": metrics_flanked["micro_accuracy"],
        })
        
        per_model[rf.model_tag]["atg_flanked"] = {
            k: v for k, v in metrics_flanked.items() if k != "cm"
        }
        per_model[rf.model_tag]["flank_note"] = flank_note
        
        # Heatmap for ATG+flanks
        heat_base_f = outdir / "heatmaps_atg_flanked" / f"knn_heatmap_{rf.model_tag}"
        plot_knn_heatmap(
            outbase=heat_base_f,
            y_true=y_true_f,
            y_pred=y_pred_f,
            label_order=LABEL_ORDER_4WAY,
            title=f"{rf.model_tag} ({flank_note}) | ba={metrics_flanked['balanced_accuracy']:.2%}",
        )
        
        print(f"  ATG-only: {metrics_atg['balanced_accuracy']:.3f}")
        print(f"  ATG+flanks ({flank_note}): {metrics_flanked['balanced_accuracy']:.3f}")

    # Save results
    df_atg_only = pd.DataFrame(rows_atg_only).sort_values("balanced_accuracy", ascending=False)
    df_atg_only.to_csv(outdir / "leaderboard_atg_only.csv", index=False)
    
    df_atg_flanked = pd.DataFrame(rows_atg_flanked).sort_values("balanced_accuracy", ascending=False)
    df_atg_flanked.to_csv(outdir / "leaderboard_atg_flanked.csv", index=False)
    
    # Plot leaderboards
    plot_leaderboard(
        df=df_atg_only,
        outbase=outdir / "leaderboard_atg_only",
        metric="balanced_accuracy",
        sem_col="balanced_accuracy_sem",
        title="ATG 4-way (ATG-only, 3bp)",
        xlabel="balanced accuracy (loo 1-nn)",
        top_k=args.top_k,
    )
    
    plot_leaderboard(
        df=df_atg_flanked,
        outbase=outdir / "leaderboard_atg_flanked",
        metric="balanced_accuracy",
        sem_col="balanced_accuracy_sem",
        title=f"ATG 4-way (ATG+context: NT=native 6-mer, others=ATG±{args.flank_bp}bp)",
        xlabel="balanced accuracy (loo 1-nn)",
        top_k=args.top_k,
    )
    
    # Save detailed JSON
    with open(outdir / "leaderboard_from_full_details.json", "w") as f:
        json.dump(per_model, f, indent=2)
    
    if args.write_per_model_json:
        per_model_dir = outdir / "per_model_json"
        per_model_dir.mkdir(exist_ok=True)
        for k, v in per_model.items():
            with open(per_model_dir / f"{k}.json", "w") as f:
                json.dump(v, f, indent=2)
    
    print(f"\n[done] Results written to {outdir}")
    print(f"  - ATG-only leaderboard: leaderboard_atg_only.csv")
    print(f"  - ATG+flanks leaderboard: leaderboard_atg_flanked.csv")


if __name__ == "__main__":
    main()

# Example usage:
# python /home/mica/gamba/src/evaluation/plotting/plot-ATGs-from-full.py\
#   --roots /home/mica/gamba/other-models/ATG_reps_5way /home/mica/gamba/data_processing/data/240-mammalian/ATG_reps_5way \
#   --outdir /home/mica/gamba/ATG4_leaderboard_out \
#   --flank_bp 3 \
#   --top_k 50