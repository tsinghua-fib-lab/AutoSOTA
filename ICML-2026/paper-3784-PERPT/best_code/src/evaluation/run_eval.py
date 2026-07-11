#!/usr/bin/env python3
"""
one-pass embedding cache + 4 tasks for gamba / bigamba + baselines (kmer6, phylop)

inputs (under regions_root):
  regions/CATEGORY/chr*.bed
  regions/CATEGORY_upstream/chr*.bed
  regions/CATEGORY_random/chr*.bed
  regions/CATEGORY_random-noannot/chr*.bed

assumptions:
- all BEDs have 7 columns: chrom, start, end, name, score, strand, pair_id
- pair_id is shared across ROI + upstream + random + random-noannot (per category)

goals:
- minimize forward passes: embed each extracted window exactly once per split
- avoid memory blowup: never keep token-level reps; pool immediately; save cache per split
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm

# plotting / metrics
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

# your project imports
import sys
sys.path.append("/home/mica/gamba/")
sys.path.append("/home/mica/scratch/gamba/")
from src.evaluation.utils.helpers import extract_context  # interval-safe version
from src.evaluation.utils.specific_helpers import load_model, predict_scores_batched  # gamba-specific
# At the top of run_eval.py, after imports:
logging.getLogger('src.evaluation.utils.helpers').setLevel(logging.ERROR)  # Suppress WARNING

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters",
]

REGION_TYPES = ["roi", "upstream", "random", "random-noannot"]
SCOPES = ["roi", "full", "roi100bp"]  # roi100bp only used for multiclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress phyloP warnings specifically:
logging.getLogger('helpers').setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# bed reading + pairing
# -----------------------------------------------------------------------------

def read_pair_bed(path: Path, category: str) -> List[dict]:
    """
    chrom  start  end  name  score  strand  pair_id
    """
    regions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3]
                score = float(parts[4])
                strand = parts[5]
                pair_id = parts[6]
            except ValueError:
                continue
            if end <= start:
                continue
            regions.append(
                dict(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name,
                    score=score,
                    strand=strand,
                    pair_id=str(pair_id),
                    category=category,
                )
            )
    return regions

def _sample_roi100bp_seeded(fs, fe, seed_int):
    if fe - fs < 100:
        return None
    rng = np.random.default_rng(int(seed_int) & 0xFFFFFFFF)
    start = int(rng.integers(fs, fe - 100 + 1))
    return start, start + 100


def load_category_region_maps(
    regions_root: Path,
    category: str,
    chroms: List[str],
) -> Dict[str, Dict[str, dict]]:
    """
    returns: region_type -> {pair_id: region_dict}
    expects:
      regions_root/category/chr*.bed
      regions_root/category_upstream/chr*.bed
      regions_root/category_random/chr*.bed
      regions_root/category_random-noannot/chr*.bed
    """
    chroms_set = set(chroms)
    maps: Dict[str, Dict[str, dict]] = {rt: {} for rt in REGION_TYPES}

    folder_for = {
        "roi": category,
        "upstream": f"{category}_upstream",
        "random": f"{category}_random",
        "random-noannot": f"{category}_random-noannot",
    }

    for rt in REGION_TYPES:
        d = regions_root / folder_for[rt]
        if not d.exists():
            logging.warning(f"[missing] {d}")
            continue
        for bf in sorted(d.glob("chr*.bed")):
            # fast filter by chromosome name in filename (bf.stem == "chr1")
            if bf.stem not in chroms_set:
                continue
            for r in read_pair_bed(bf, category):
                if r["chrom"] not in chroms_set:
                    continue
                maps[rt][r["pair_id"]] = r

    return maps


def common_pair_ids(maps: Dict[str, Dict[str, dict]]) -> List[str]:
    """
    require intersection across all 4 region types (roi/upstream/random/random-noannot).
    if any region type is missing/empty for this category+split, skip.
    """
    present = [set(maps[rt].keys()) for rt in REGION_TYPES]
    if any(len(s) == 0 for s in present):
        return []
    return sorted(set.intersection(*present))

# -----------------------------------------------------------------------------
# baseline feature extraction (kmer6, phylop)
# -----------------------------------------------------------------------------

def _roi_span(fs: int, fe: int) -> Optional[Tuple[int, int]]:
    fs = int(fs)
    fe = int(fe)
    if fe <= fs:
        return None
    return fs, fe


def _build_kmer_index(k: int = 6, alphabet: str = "ACGT") -> Dict[str, int]:
    from itertools import product
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def _seq_to_kmer_vec(seq: str, k: int, kmer_index: Dict[str, int]) -> np.ndarray:
    n = len(kmer_index)
    vec = np.zeros(n, dtype=np.float32)
    seq = (seq or "").upper()
    L = len(seq)
    if L < k:
        return vec
    for i in range(L - k + 1):
        kmer = seq[i:i + k]
        j = kmer_index.get(kmer)
        if j is not None:
            vec[j] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


def _summarize_scores(scores: np.ndarray) -> Optional[np.ndarray]:
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0 or np.isnan(s).all():
        return None
    m = np.nanmean(s)
    st = np.nanstd(s)
    denom = float(np.sum(~np.isnan(s))) if np.sum(~np.isnan(s)) else 0.0
    fpos = float(np.sum(s > 0)) / denom if denom else 0.0
    fneg = float(np.sum(s < 0)) / denom if denom else 0.0
    pos = s[(s > 0) & ~np.isnan(s)]
    neg = s[(s < 0) & ~np.isnan(s)]
    mpos = float(np.nanmean(pos)) if pos.size else 0.0
    mneg = float(np.nanmean(neg)) if neg.size else 0.0
    return np.array([m, st, fpos, fneg, mpos, mneg], dtype=np.float32)


# -----------------------------------------------------------------------------
# pooled embedding extraction (no token-level storage)
# -----------------------------------------------------------------------------

def _pool_span(rep: np.ndarray, window_len: int, fs: int, fe: int) -> np.ndarray:
    """
    rep: [T, H] token/bp-level reps
    fs,fe: bp offsets in the window (0..window_len)
    robust to T != window_len using proportional mapping
    """
    rep = np.asarray(rep)
    if rep.ndim != 2 or rep.shape[0] < 1:
        raise ValueError("rep must be [T,H]")

    T = rep.shape[0]
    window_len = int(max(1, window_len))
    fs = int(max(0, min(fs, window_len)))
    fe = int(max(fs + 1, min(fe, window_len)))

    scale = T / float(window_len)
    tfs = max(0, min(int(np.floor(fs * scale)), T - 1))
    tfe = max(tfs + 1, min(int(np.ceil(fe * scale)), T))

    return rep[tfs:tfe].mean(axis=0).astype(np.float32)


def _pool_full(rep: np.ndarray) -> np.ndarray:
    rep = np.asarray(rep)
    if rep.ndim != 2 or rep.shape[0] < 1:
        raise ValueError("rep must be [T,H]")
    return rep.mean(axis=0).astype(np.float32)


def _sample_roi100bp(fs: int, fe: int, rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    if fe - fs < 100:
        return None
    start = int(rng.integers(fs, fe - 100 + 1))
    return start, start + 100


@dataclass
class CacheOutputs:
    # pooled embeddings
    full: np.ndarray              # [N, H]
    roi: np.ndarray               # [N, H]
    roi100: np.ndarray            # [N, H] (NaNs when not applicable)
    # metadata
    labels_task: np.ndarray       # [N] string labels ("feature"/"upstream"/"random"/"random-noannot")
    category: np.ndarray          # [N] string
    region_type: np.ndarray       # [N] string (roi/upstream/random/random-noannot)
    pair_id: np.ndarray           # [N] string
    chrom: np.ndarray             # [N] string
    start: np.ndarray             # [N] int
    end: np.ndarray               # [N] int
    fs: np.ndarray                # [N] int
    fe: np.ndarray                # [N] int
    window_len: np.ndarray        # [N] int
    roi100_fs: np.ndarray         # [N] int (or -1)
    roi100_fe: np.ndarray         # [N] int (or -1)


def embed_contexts_pooled_onepass(
    *,
    model,
    tokenizer,
    contexts: List[dict],
    model_type: str,
    training_task: str,
    batch_size: int,
    device: torch.device,
    rng_seed: int,
    baseline: str,     # "none" | "kmer6" | "phylop"
    kmer_k: int = 6,
) -> CacheOutputs:
    """
    - baseline=none: uses predict_scores_batched -> token reps -> pooled mean
    - baseline=kmer6: compute k-mer frequency unit-norm vecs from sequence (ROI/full/ROI100)
    - baseline=phylop: compute 6D summary vecs from scores (ROI/full/ROI100)

    returns pooled arrays + metadata aligned.
    """
    rng = np.random.default_rng(int(rng_seed))

    full_buf: List[np.ndarray] = []
    roi_buf: List[np.ndarray] = []
    roi100_buf: List[np.ndarray] = []

    labels_task: List[str] = []
    cat_buf: List[str] = []
    rt_buf: List[str] = []
    pid_buf: List[str] = []
    chrom_buf: List[str] = []
    start_buf: List[int] = []
    end_buf: List[int] = []
    fs_buf: List[int] = []
    fe_buf: List[int] = []
    wlen_buf: List[int] = []
    r100_fs_buf: List[int] = []
    r100_fe_buf: List[int] = []

    if baseline not in ("none", "kmer6", "phylop"):
        raise ValueError(f"unsupported baseline: {baseline}")

    # baseline prep
    kmer_index = _build_kmer_index(k=kmer_k) if baseline == "kmer6" else None
    # infer embedding dim for baseline so we can create NaN vectors consistently
    baseline_dim = None
    if baseline == "kmer6":
        baseline_dim = 4 ** kmer_k
    elif baseline == "phylop":
        baseline_dim = 6

    def _nan_vec(dim: int) -> np.ndarray:
        return np.full((dim,), np.nan, dtype=np.float32)

    for i in tqdm(range(0, len(contexts), batch_size), desc="embed+pool", leave=False):
        batch = contexts[i : i + batch_size]

        if baseline == "none":
            reps, infos = predict_scores_batched(
                model,
                tokenizer,
                batch,
                batch_size=len(batch),
                device=device,
                model_type=model_type,
                training_task=training_task,
            )

            for rep, info, ctx in zip(reps, infos, batch):
                rep = np.asarray(rep)
                if rep.ndim != 2 or rep.shape[0] < 1:
                    continue

                window_len = int(ctx.get("window_len", len(ctx.get("sequence", "")) or rep.shape[0]))
                fs = int(ctx.get("feature_start_in_window", 0))
                fe = int(ctx.get("feature_end_in_window", window_len))

                try:
                    full_vec = _pool_full(rep)
                    roi_vec = _pool_span(rep, window_len=window_len, fs=fs, fe=fe)
                except Exception:
                    continue

                rt = str(ctx.get("region_type", "unknown"))
                roi100_fs = -1
                roi100_fe = -1
                roi100_vec = np.full_like(roi_vec, np.nan, dtype=np.float32)
                if rt == "roi":
                    seed = (hash(f"{ctx.get('category')}|{ctx.get('pair_id')}") & 0x7FFFFFFF) + rng_seed
                    seg = _sample_roi100bp_seeded(fs, fe, seed)
                    if seg is not None:
                        s100, e100 = seg
                        roi100_fs, roi100_fe = s100, e100
                        try:
                            roi100_vec = _pool_span(rep, window_len=window_len, fs=s100, fe=e100)
                        except Exception:
                            roi100_vec = np.full_like(roi_vec, np.nan, dtype=np.float32)

                full_buf.append(full_vec)
                roi_buf.append(roi_vec)
                roi100_buf.append(roi100_vec)

                labels_task.append(str(ctx.get("class_label", "unknown")))
                cat_buf.append(str(ctx.get("category", "unknown")))
                rt_buf.append(rt)
                pid_buf.append(str(ctx.get("pair_id", "NA")))
                chrom_buf.append(str(ctx.get("chrom", "NA")))
                start_buf.append(int(ctx.get("start", -1)))
                end_buf.append(int(ctx.get("end", -1)))
                fs_buf.append(fs)
                fe_buf.append(fe)
                wlen_buf.append(int(window_len))
                r100_fs_buf.append(int(roi100_fs))
                r100_fe_buf.append(int(roi100_fe))

            del reps, infos
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # ------------------------------------------------------------------
        # baselines: compute directly from ctx["sequence"] / ctx["scores"]
        # ------------------------------------------------------------------
        for ctx in batch:
            rt = str(ctx.get("region_type", "unknown"))
            seq = ctx.get("sequence", "")
            window_len = int(ctx.get("window_len", len(seq) or 1))
            fs = int(ctx.get("feature_start_in_window", 0))
            fe = int(ctx.get("feature_end_in_window", window_len))
            span = _roi_span(fs, fe)
            if span is None:
                continue

            roi100_fs = -1
            roi100_fe = -1

            if baseline == "kmer6":
                assert kmer_index is not None
                # full
                full_vec = _seq_to_kmer_vec(seq, k=kmer_k, kmer_index=kmer_index)
                # roi
                sroi = seq[fs:fe]
                roi_vec = _seq_to_kmer_vec(sroi, k=kmer_k, kmer_index=kmer_index)
                # roi100
                roi100_vec = _nan_vec(baseline_dim)
                if rt == "roi":
                    seed = (hash(f"{ctx.get('category')}|{ctx.get('pair_id')}") & 0x7FFFFFFF) + rng_seed
                    seg = _sample_roi100bp_seeded(fs, fe, seed)
                    if seg is not None:
                        s100, e100 = seg
                        roi100_fs, roi100_fe = s100, e100
                        s100seq = seq[s100:e100]
                        roi100_vec = _seq_to_kmer_vec(s100seq, k=kmer_k, kmer_index=kmer_index)

            elif baseline == "phylop":
                scores = ctx.get("scores", None)
                if scores is None:
                    continue
                scores = np.asarray(scores, dtype=np.float32)

                full_feat = _summarize_scores(scores)
                if full_feat is None:
                    continue
                roi_feat = _summarize_scores(scores[fs:fe])
                if roi_feat is None:
                    continue

                full_vec = full_feat
                roi_vec = roi_feat

                roi100_vec = _nan_vec(baseline_dim)
                if rt == "roi":
                    seed = (hash(f"{ctx.get('category')}|{ctx.get('pair_id')}") & 0x7FFFFFFF) + rng_seed
                    seg = _sample_roi100bp_seeded(fs, fe, seed)
                    if seg is not None:
                        s100, e100 = seg
                        roi100_fs, roi100_fe = s100, e100
                        feat100 = _summarize_scores(scores[s100:e100])
                        if feat100 is not None:
                            roi100_vec = feat100

            else:
                raise ValueError("unreachable")

            full_buf.append(full_vec.astype(np.float32))
            roi_buf.append(roi_vec.astype(np.float32))
            roi100_buf.append(roi100_vec.astype(np.float32))

            labels_task.append(str(ctx.get("class_label", "unknown")))
            cat_buf.append(str(ctx.get("category", "unknown")))
            rt_buf.append(rt)
            pid_buf.append(str(ctx.get("pair_id", "NA")))
            chrom_buf.append(str(ctx.get("chrom", "NA")))
            start_buf.append(int(ctx.get("start", -1)))
            end_buf.append(int(ctx.get("end", -1)))
            fs_buf.append(int(fs))
            fe_buf.append(int(fe))
            wlen_buf.append(int(window_len))
            r100_fs_buf.append(int(roi100_fs))
            r100_fe_buf.append(int(roi100_fe))

    # finalize
    if not full_buf:
        # keep shapes valid-ish
        dim = baseline_dim if baseline != "none" else 1
        full = np.empty((0, dim), np.float32)
        roi = np.empty((0, dim), np.float32)
        roi100 = np.empty((0, dim), np.float32)
    else:
        full = np.stack(full_buf).astype(np.float32)
        roi = np.stack(roi_buf).astype(np.float32)
        roi100 = np.stack(roi100_buf).astype(np.float32)

    return CacheOutputs(
        full=full,
        roi=roi,
        roi100=roi100,
        labels_task=np.asarray(labels_task, dtype=object),
        category=np.asarray(cat_buf, dtype=object),
        region_type=np.asarray(rt_buf, dtype=object),
        pair_id=np.asarray(pid_buf, dtype=object),
        chrom=np.asarray(chrom_buf, dtype=object),
        start=np.asarray(start_buf, dtype=np.int64),
        end=np.asarray(end_buf, dtype=np.int64),
        fs=np.asarray(fs_buf, dtype=np.int64),
        fe=np.asarray(fe_buf, dtype=np.int64),
        window_len=np.asarray(wlen_buf, dtype=np.int64),
        roi100_fs=np.asarray(r100_fs_buf, dtype=np.int64),
        roi100_fe=np.asarray(r100_fe_buf, dtype=np.int64),
    )

# -----------------------------------------------------------------------------
# cache i/o
# -----------------------------------------------------------------------------

def save_cache(outdir: Path, model_id: str, group_name: str, cache: CacheOutputs) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"cache_{model_id}_{group_name}.npz"
    meta_path = outdir / f"cache_{model_id}_{group_name}_meta.parquet"

    np.savez_compressed(
        npz_path,
        full=cache.full,
        roi=cache.roi,
        roi100=cache.roi100,
        labels_task=cache.labels_task,
        category=cache.category,
        region_type=cache.region_type,
        pair_id=cache.pair_id,
        chrom=cache.chrom,
        start=cache.start,
        end=cache.end,
        fs=cache.fs,
        fe=cache.fe,
        window_len=cache.window_len,
        roi100_fs=cache.roi100_fs,
        roi100_fe=cache.roi100_fe,
    )

    mdf = pd.DataFrame(
        dict(
            labels_task=cache.labels_task,
            category=cache.category,
            region_type=cache.region_type,
            pair_id=cache.pair_id,
            chrom=cache.chrom,
            start=cache.start,
            end=cache.end,
            feature_start_in_window=cache.fs,
            feature_end_in_window=cache.fe,
            window_len=cache.window_len,
            roi100_fs=cache.roi100_fs,
            roi100_fe=cache.roi100_fe,
        )
    )
    mdf.to_parquet(meta_path, index=False)
    return npz_path, meta_path

# -----------------------------------------------------------------------------
# evaluation utilities
# -----------------------------------------------------------------------------

def plot_umap(embeddings: np.ndarray, labels: List[str], output_path: Path, title: str):
    if embeddings.shape[0] == 0:
        return
    um = umap.UMAP()
    emb2d = um.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb2d[:, 0], y=emb2d[:, 1], hue=labels, s=20, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def loo_1nn_predictions(embeddings: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[idx[:, 1]]
    return y_true, y_pred


def eval_metrics(y_true, y_pred, label_order=None):
    if label_order is None:
        label_order = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()

    valid = ~np.isnan(per_class_recall)
    ba = float(np.mean(per_class_recall[valid]))
    sem = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid))) if np.sum(valid) > 1 else 0.0
    ci95 = float(1.96 * sem)

    metrics = {
        "micro_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": ba,
        "balanced_accuracy_sem": sem,
        "balanced_accuracy_ci95": ci95,
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order


def plot_knn_heatmap(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        return None, None, None

    labels = np.asarray(labels)
    present = sorted(set(labels.tolist()))
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True)
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1.0,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Per-class recall"},
    )
    plt.title(
        f"{title}\n"
        f"micro={metrics['micro_accuracy']:.2%} | "
        f"balanced={metrics['balanced_accuracy']:.2%} | "
        f"macro-F1={metrics['macro_f1']:.2%}"
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return metrics, label_order, acc_matrix


def _save_per_class_json(json_path: Path, label_order, acc_matrix):
    data = {
        "label_order": list(map(str, label_order)),
        "per_class_recall": {str(lbl): float(acc_matrix[i, i]) for i, lbl in enumerate(label_order)},
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _append_summary(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_reps(outdir: Path, model_id: str, tag: str, X: np.ndarray, labels: np.ndarray, metas: pd.DataFrame):
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"reps_{model_id}_{tag}.npz", embeddings=X.astype(np.float32), labels=labels)
    metas.to_parquet(outdir / f"reps_{model_id}_{tag}_meta.parquet", index=False)

# -----------------------------------------------------------------------------
# task derivation from cache
# -----------------------------------------------------------------------------

def derive_and_save_tasks(
    cache: CacheOutputs,
    group_name: str,
    model_id: str,
    outdir: Path,
    do_full: bool,
):
    """
    creates reps + plots for:
      1) binary_upstream: roi vs upstream
      2) binary_random: roi vs random
      3) binary_random-noannot: roi vs random-noannot
      4) multiclass: roi categories (roi scope) + roi100bp categories (roi100bp scope)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "summary_knn.csv"

    dfmeta = pd.DataFrame(
        dict(
            group=group_name,
            category=cache.category,
            region_type=cache.region_type,
            pair_id=cache.pair_id,
            chrom=cache.chrom,
            start=cache.start,
            end=cache.end,
            feature_start_in_window=cache.fs,
            feature_end_in_window=cache.fe,
            window_len=cache.window_len,
            roi100_fs=cache.roi100_fs,
            roi100_fe=cache.roi100_fe,
        )
    )

    def rows(rt: str, category: Optional[str] = None):
        m = (cache.region_type == rt)
        if category is not None:
            m = m & (cache.category == category)
        return np.where(m)[0]

    binary_tasks = [
        ("upstream", "roi", "upstream"),
        ("random", "roi", "random"),
        ("random-noannot", "roi", "random-noannot"),
    ]

    for cat in CATEGORY_ORDER:
        for task_name, pos_rt, neg_rt in binary_tasks:
            pos_idx = rows(pos_rt, cat)
            neg_idx = rows(neg_rt, cat)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_pids = set(cache.pair_id[pos_idx].tolist())
            neg_pids = set(cache.pair_id[neg_idx].tolist())
            common = sorted(pos_pids & neg_pids)
            if len(common) == 0:
                continue

            pos_map = {pid: i for pid, i in zip(cache.pair_id[pos_idx].tolist(), pos_idx.tolist())}
            neg_map = {pid: i for pid, i in zip(cache.pair_id[neg_idx].tolist(), neg_idx.tolist())}
            pos_keep = [pos_map[pid] for pid in common]
            neg_keep = [neg_map[pid] for pid in common]

            for scope in (["roi"] + (["full"] if do_full else [])):
                Xpos = cache.roi[pos_keep] if scope == "roi" else cache.full[pos_keep]
                Xneg = cache.roi[neg_keep] if scope == "roi" else cache.full[neg_keep]
                X = np.vstack([Xpos, Xneg])
                y = np.asarray(["feature"] * len(Xpos) + [task_name] * len(Xneg), dtype=object)

                tag = f"{group_name}_{cat}_binary-{task_name}_{scope}"
                sub = outdir / "binary" / task_name / cat
                sub.mkdir(parents=True, exist_ok=True)

                plot_umap(X, y.tolist(), sub / f"umap_{model_id}_{tag}.png",
                          f"{cat}: feature vs {task_name} ({group_name}, {scope})")
                metrics, order, mat = plot_knn_heatmap(
                    X, y, sub / f"knn_{model_id}_{tag}.png",
                    f"{cat}: feature vs {task_name} ({group_name}, {scope})"
                )
                if metrics is not None:
                    _save_per_class_json(sub / f"per_class_{model_id}_{tag}.json", order, mat)
                    _append_summary(
                        summary_csv,
                        dict(
                            Model=model_id,
                            Group=group_name,
                            Task=f"binary-{task_name}",
                            Category=cat,
                            Scope=scope,
                            N_pairs=len(common),
                            BalancedAccuracyPct=100.0 * metrics["balanced_accuracy"],
                            BalancedAccuracySEM_Pct=100.0 * metrics["balanced_accuracy_sem"],
                            MicroAccuracyPct=100.0 * metrics["micro_accuracy"],
                            MacroF1Pct=100.0 * metrics["macro_f1"],
                        ),
                    )

                metas = pd.concat(
                    [
                        dfmeta.iloc[pos_keep].assign(label="feature", scope=scope, task=f"binary-{task_name}"),
                        dfmeta.iloc[neg_keep].assign(label=task_name, scope=scope, task=f"binary-{task_name}"),
                    ],
                    axis=0,
                    ignore_index=True,
                )
                save_reps(sub, model_id, tag, X, y, metas)

    # multiclass
    roi_idx = np.where(cache.region_type == "roi")[0]
    if len(roi_idx) > 0:
        y_mc = cache.category[roi_idx].astype(object)

        sub = outdir / "multiclass"
        sub.mkdir(parents=True, exist_ok=True)

        # roi mean
        X_roi = cache.roi[roi_idx]
        tag = f"{group_name}_multiclass_roi"
        plot_umap(X_roi, y_mc.tolist(), sub / f"umap_{model_id}_{tag}.png",
                  f"multiclass (roi mean) ({group_name})")
        metrics, order, mat = plot_knn_heatmap(
            X_roi, y_mc, sub / f"knn_{model_id}_{tag}.png",
            f"multiclass (roi mean) ({group_name})"
        )
        if metrics is not None:
            _save_per_class_json(sub / f"per_class_{model_id}_{tag}.json", order, mat)
            _append_summary(
                summary_csv,
                dict(
                    Model=model_id,
                    Group=group_name,
                    Task="multiclass",
                    Category="ALL",
                    Scope="roi",
                    N=len(roi_idx),
                    BalancedAccuracyPct=100.0 * metrics["balanced_accuracy"],
                    BalancedAccuracySEM_Pct=100.0 * metrics["balanced_accuracy_sem"],
                    MicroAccuracyPct=100.0 * metrics["micro_accuracy"],
                    MacroF1Pct=100.0 * metrics["macro_f1"],
                ),
            )
        metas = dfmeta.iloc[roi_idx].assign(label=y_mc, scope="roi", task="multiclass")
        save_reps(sub, model_id, tag, X_roi, y_mc, metas)

        # roi100bp (drop NaNs)
        X_100 = cache.roi100[roi_idx]
        ok = ~np.isnan(X_100).any(axis=1)
        if np.sum(ok) > 0:
            X_100_ok = X_100[ok]
            y_100_ok = y_mc[ok]
            roi_idx_ok = roi_idx[ok]

            tag = f"{group_name}_multiclass_roi100bp"
            plot_umap(X_100_ok, y_100_ok.tolist(), sub / f"umap_{model_id}_{tag}.png",
                      f"multiclass (roi100bp) ({group_name})")
            metrics, order, mat = plot_knn_heatmap(
                X_100_ok, y_100_ok, sub / f"knn_{model_id}_{tag}.png",
                f"multiclass (roi100bp) ({group_name})"
            )
            if metrics is not None:
                _save_per_class_json(sub / f"per_class_{model_id}_{tag}.json", order, mat)
                _append_summary(
                    summary_csv,
                    dict(
                        Model=model_id,
                        Group=group_name,
                        Task="multiclass",
                        Category="ALL",
                        Scope="roi100bp",
                        N=int(np.sum(ok)),
                        BalancedAccuracyPct=100.0 * metrics["balanced_accuracy"],
                        BalancedAccuracySEM_Pct=100.0 * metrics["balanced_accuracy_sem"],
                        MicroAccuracyPct=100.0 * metrics["micro_accuracy"],
                        MacroF1Pct=100.0 * metrics["macro_f1"],
                    ),
                )
            metas = dfmeta.iloc[roi_idx_ok].assign(label=y_100_ok, scope="roi100bp", task="multiclass")
            save_reps(sub, model_id, tag, X_100_ok, y_100_ok, metas)

# -----------------------------------------------------------------------------
# building contexts per split (one extraction per region window)
# -----------------------------------------------------------------------------
def build_contexts_for_group(
    genome: Fasta,
    bigwig_file: str,
    regions_root: Path,
    group_chroms: List[str],
    num_regions: Optional[int],
    ctx_model_type: str,
) -> List[dict]:
    all_contexts: List[dict] = []
    
    # ADD THESE COUNTERS:
    total_attempted = 0
    failed_extract = 0
    failed_no_sequence = 0
    
    for cat in CATEGORY_ORDER:
        maps = load_category_region_maps(regions_root, cat, group_chroms)
        pids = common_pair_ids(maps)
        if not pids:
            logging.warning(f"[{cat}] no common pair_ids across all 4 region types in this split")
            continue
        if num_regions is not None:
            pids = pids[: int(num_regions)]

        import warnings

        for pid in pids:
            for rt in REGION_TYPES:
                total_attempted += 1
                r = maps[rt][pid]
                
                # Suppress warnings just for this call:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Also suppress logging warnings:
                    old_level = logging.getLogger().level
                    logging.getLogger().setLevel(logging.ERROR)
                    
                    ctx = extract_context(bigwig_file, r, genome, model_type=ctx_model_type)
                    
                    # Restore logging level:
                    logging.getLogger().setLevel(old_level)
                
                if not ctx:
                    failed_extract += 1
                    continue
                if "sequence" not in ctx:
                    failed_no_sequence += 1
                    continue

                ctx["category"] = cat
                ctx["region_type"] = rt
                ctx["pair_id"] = pid
                ctx["chrom"] = r["chrom"]
                ctx["start"] = int(r["start"])
                ctx["end"] = int(r["end"])
                ctx["strand"] = r.get("strand", "+")
                ctx["window_len"] = len(ctx["sequence"])
                ctx["class_label"] = "feature" if rt == "roi" else rt
                all_contexts.append(ctx)
    
    # LOG SUMMARY:
    logging.info(f"[build_contexts] total_attempted: {total_attempted}")
    logging.info(f"[build_contexts] failed_extract (no ctx): {failed_extract}")
    logging.info(f"[build_contexts] failed_no_sequence: {failed_no_sequence}")
    logging.info(f"[build_contexts] succeeded: {len(all_contexts)}")
    logging.info(f"[build_contexts] success_rate: {100*len(all_contexts)/max(1,total_attempted):.1f}%")

    return all_contexts

# def build_contexts_for_group(
#     genome: Fasta,
#     bigwig_file: str,
#     regions_root: Path,
#     group_chroms: List[str],
#     num_regions: Optional[int],
#     ctx_model_type: str,
# ) -> List[dict]:
#     """
#     max coverage:
#       - include ALL roi rows
#       - include ALL upstream/random/random-noannot rows
#       - no requirement that a pair_id exists across all 4 types
#     later, task derivation will intersect pair_ids per task.
#     """
#     all_contexts: List[dict] = []
#     chroms_set = set(group_chroms)

#     for cat in CATEGORY_ORDER:
#         maps = load_category_region_maps(regions_root, cat, group_chroms)

#         # optional cap per (category, region_type), deterministic
#         for rt in REGION_TYPES:
#             pids = list(maps[rt].keys())
#             if not pids:
#                 continue

#             # make deterministic
#             pids = sorted(pids)

#             if num_regions is not None and len(pids) > int(num_regions):
#                 pids = pids[: int(num_regions)]

#             for pid in pids:
#                 r = maps[rt][pid]
#                 if r["chrom"] not in chroms_set:
#                     continue

#                 ctx = extract_context(bigwig_file, r, genome, model_type=ctx_model_type)
#                 if not ctx or "sequence" not in ctx or not ctx["sequence"]:
#                     continue

#                 ctx["category"] = cat
#                 ctx["region_type"] = rt
#                 ctx["pair_id"] = str(pid)
#                 ctx["chrom"] = r["chrom"]
#                 ctx["start"] = int(r["start"])
#                 ctx["end"] = int(r["end"])
#                 ctx["strand"] = r.get("strand", "+")
#                 ctx["window_len"] = len(ctx["sequence"])

#                 # labels for downstream tasks
#                 ctx["class_label"] = "feature" if rt == "roi" else rt

#                 all_contexts.append(ctx)

#     return all_contexts


# -----------------------------------------------------------------------------
# main pipeline
# -----------------------------------------------------------------------------

def run_onepass(
    *,
    genome_fasta: str,
    bigwig_file: str,
    regions_root: str,
    output_dir: str,
    checkpoint_dir: str,
    config_fpath: str,
    last_step: int,
    batch_size: int,
    num_regions: Optional[int],
    chromosomes: List[str],
    training_chromosomes: Optional[List[str]],
    test_chromosomes: Optional[List[str]],
    model_type: str,        # gamba/bigamba
    training_task: str,     # dual/cons_only/seq_only/...
    cache_seed: int,
    do_full: bool,
    resume: bool,
    baseline: str,          # none/kmer6/phylop
    kmer_k: int,
):
    regions_root = Path(regions_root)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    genome = Fasta(genome_fasta)

    if training_chromosomes and test_chromosomes:
        groups = {"training": training_chromosomes, "test": test_chromosomes}
    else:
        groups = {"all": chromosomes}

    # decide extract_context model_type
    # - for baselines: use "baseline" so extract_context includes "scores" etc.
    # - for gamba/bigamba: keep stable string that your helper checks
    if baseline in ("kmer6", "phylop"):
        ctx_model_type = "baseline"
    else:
        ctx_model_type = model_type  # keep stable for your asym rules

    # load model only if baseline == none
    model = tokenizer = None
    if baseline == "none":
        model, tokenizer = load_model(
            checkpoint_dir=checkpoint_dir,
            config_fpath=config_fpath,
            last_step=last_step,
            device=device,
            training_task=training_task,
            model_type=model_type,
        )

    # model id
    if baseline == "none":
        model_id = f"{model_type}_{training_task}_step{last_step}"
    else:
        model_id = baseline

    for group_name, group_chroms in groups.items():
        logging.info(f"[group={group_name}] building contexts across {len(group_chroms)} chroms")

        cache_dir = outdir / model_id / "cache"
        cache_npz = cache_dir / f"cache_{model_id}_{group_name}.npz"
        cache_meta = cache_dir / f"cache_{model_id}_{group_name}_meta.parquet"

        if resume and cache_npz.exists() and cache_meta.exists():
            logging.info(f"[resume] cache exists for {group_name}: {cache_npz.name}")
            z = np.load(cache_npz, allow_pickle=True)
            cache = CacheOutputs(
                full=z["full"],
                roi=z["roi"],
                roi100=z["roi100"],
                labels_task=z["labels_task"],
                category=z["category"],
                region_type=z["region_type"],
                pair_id=z["pair_id"],
                chrom=z["chrom"],
                start=z["start"],
                end=z["end"],
                fs=z["fs"],
                fe=z["fe"],
                window_len=z["window_len"],
                roi100_fs=z["roi100_fs"],
                roi100_fe=z["roi100_fe"],
            )
        else:
            contexts = build_contexts_for_group(
                genome=genome,
                bigwig_file=bigwig_file,
                regions_root=regions_root,
                group_chroms=group_chroms,
                num_regions=num_regions,
                ctx_model_type=ctx_model_type,
            )
            logging.info(f"[group={group_name}] total contexts to embed: {len(contexts)}")
            if len(contexts) == 0:
                logging.warning(f"[group={group_name}] no contexts; skipping")
                continue

            # SAVE THE SUCCESSFUL KEYS:
            success_keys = set()
            for ctx in contexts:
                key = f"{ctx['category']}|{ctx['pair_id']}|{ctx['region_type']}"
                success_keys.add(key)

            success_keys_file = outdir / model_id / "cache" / f"success_keys_{group_name}.txt"
            success_keys_file.parent.mkdir(parents=True, exist_ok=True)
            with open(success_keys_file, 'w') as f:
                for key in sorted(success_keys):
                    f.write(f"{key}\n")
            logging.info(f"[saved] {len(success_keys)} successful keys to {success_keys_file}")

            cache = embed_contexts_pooled_onepass(
                model=model,
                tokenizer=tokenizer,
                contexts=contexts,
                model_type=model_type,
                training_task=training_task,
                batch_size=batch_size,
                device=device,
                rng_seed=cache_seed,
                baseline=baseline,
                kmer_k=kmer_k,
            )
            save_cache(cache_dir, model_id, group_name, cache)

            del contexts
            if device.type == "cuda":
                torch.cuda.empty_cache()

        task_out = outdir / model_id / "tasks"
        derive_and_save_tasks(
            cache=cache,
            group_name=group_name,
            model_id=model_id,
            outdir=task_out,
            do_full=do_full,
        )

        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# cli
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="one-pass cache embedding + upstream/random/random-noannot/multiclass tasks for gamba/bigamba + baselines"
    )
    p.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )
    p.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    p.add_argument(
        "--regions_root",
        type=str,
        default="/home/mica/gamba/data_processing/data/regions_common",
        help="root containing CATEGORY/{chr}.bed, CATEGORY_upstream/{chr}.bed, CATEGORY_random/{chr}.bed, CATEGORY_random-noannot/{chr}.bed",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/other-models/final_representations/gamba_onepass",
    )

    # model (only used when baseline=none)
    p.add_argument("--checkpoint_dir", type=str, default="/home/mica/gamba/")
    p.add_argument("--config_fpath", type=str, default="/home/mica/gamba/configs/jamba-small-240mammalian.json")
    p.add_argument("--last_step", type=int, default=44000)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument(
        "--baseline",
        type=str,
        choices=["none", "kmer6", "phylop"],
        default="none",
        help="use baseline features instead of gamba/caduceus model",
    )
    p.add_argument(
        "--model_type",
        type=str,
        choices=["gamba", "caduceus"],
        default=None,
        help="required when --baseline=none (passed to load_model/predict_scores_batched)",
    )
    p.add_argument(
        "--training_task",
        type=str,
        default=None,
        help="required when --baseline=none (dual/cons_only/seq_only etc)",
    )

    p.add_argument("--kmer_k", type=int, default=6, help="k for kmer baseline (only used for baseline=kmer6)")

    # data selection
    p.add_argument("--num_regions", type=int, default=None)
    p.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11",
            "chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21", "chr22", "chrX"
        ],
    )
    p.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        #default = None,
        default=[
            "chr1","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11",
            "chr12","chr13","chr14","chr15","chr17","chr18","chr19","chr20","chr21","chrX"
        ],
    )
    p.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        #default= None,
        default=["chr2", "chr22", "chr16", "chr3"],
    )

    # behavior
    p.add_argument("--cache_seed", type=int, default=1337, help="controls roi100bp sampling")
    p.add_argument("--do_full", action="store_true", help="also derive/save full-window binary reps")
    p.add_argument("--resume", action="store_true", help="reuse cache_{model_id}_{split}.npz if present")

    return p.parse_args()


def main():
    args = parse_args()

    if args.baseline == "none":
        if args.model_type is None or args.training_task is None:
            raise SystemExit("when --baseline=none, provide --model_type and --training_task")
        model_type = args.model_type
        training_task = args.training_task
    else:
        # baselines don’t need these
        model_type = "baseline"
        training_task = "baseline"
        # baselines: keep the same split behavior (training/test) unless you want to disable it.
        # if you want old behavior (no test split), run with:
        #   --training_chromosomes <all chroms> --test_chromosomes (omit)
        # keeping as-is.

    logging.info(f"baseline={args.baseline}")
    logging.info(f"model_type={model_type} training_task={training_task} step={args.last_step}")
    logging.info(f"regions_root={args.regions_root}")
    logging.info(f"output_dir={args.output_dir}")
    logging.info(f"batch_size={args.batch_size} num_regions={args.num_regions} do_full={args.do_full} resume={args.resume}")

    run_onepass(
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        regions_root=args.regions_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        config_fpath=args.config_fpath,
        last_step=args.last_step,
        batch_size=args.batch_size,
        num_regions=args.num_regions,
        chromosomes=args.chromosomes,
        training_chromosomes=args.training_chromosomes,
        test_chromosomes=args.test_chromosomes,
        model_type=model_type,
        training_task=training_task,
        cache_seed=args.cache_seed,
        do_full=args.do_full,
        resume=args.resume,
        baseline=args.baseline,
        kmer_k=args.kmer_k,
    )


if __name__ == "__main__":
    main()
