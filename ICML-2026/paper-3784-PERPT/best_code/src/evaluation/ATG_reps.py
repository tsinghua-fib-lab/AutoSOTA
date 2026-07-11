#!/usr/bin/env python3
"""
ATG 5-way (gamba/caduceus) with chromosome-stratified sampling.
NOW WITH BASELINES: kmer6 and phylop6D

- loads ONE TSV: the simplified 5-way format
- samples N examples total, approximately evenly across chr1..chr22 (default N=1000)
- builds 5 contexts per example (so total contexts = 5*N; default 5000)
- embeds with gamba/caduceus OR baseline features (kmer6/phylop)
- saves reps_{model_tag}_ATG_5way_all_labels.{npz,parquet}
- runs:
  - 5-way 1-NN confusion heatmap on labels 1..5
  - binary 1-NN tasks: 1 vs each of 2..5

use_6mer_roi:
  - snaps context window to 6-mer boundary (strand-aware, via extract_context)
  - baseline=="none" (gamba/caduceus): pools 6 consecutive tokens starting at fs
    (char-level, no CLS token for either model)
  - baseline=="phylop": pools 6 phyloP scores (ATG + 3 flanking) instead of 3
  - baseline=="kmer6" / "kmer6_flanked": unchanged — k-mer identity is
    sequence-determined; snapping only gives consistent boundary alignment
  - appends '_6mer' suffix to all saved filenames and plots
"""

import argparse
import os
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pyBigWig
from pyfaidx import Fasta

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

import sys
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")

from src.evaluation.utils.helpers import extract_context
from src.evaluation.utils.specific_helpers import load_model, predict_scores_batched


# ---------------- logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------- baseline feature extraction ----------------

def _build_kmer_index(k: int = 6, alphabet: str = "ACGT"):
    """Build mapping from k-mer to index."""
    from itertools import product
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def _seq_to_kmer_vec(seq: str, k: int, kmer_index: dict) -> np.ndarray:
    """Convert sequence to normalized k-mer frequency vector."""
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


def _summarize_scores(scores: np.ndarray) -> np.ndarray:
    """Summarize phyloP scores into 6D feature vector."""
    s = np.asarray(scores, dtype=np.float32)

    if s.size == 0 or np.isnan(s).all():
        return np.full(6, np.nan, dtype=np.float32)

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


# ---------------- KNN + metrics helpers ----------------

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]
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
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order


def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN"):
    if len(embeddings) == 0:
        logging.warning("[plot_knn_heatmap] no embeddings to plot")
        return None, None, None

    labels = np.asarray(labels)
    present = sorted(set(labels))
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)

    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1.0,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(
        f"{title}\n"
        f"micro={metrics['micro_accuracy']:.2%} | balanced={metrics['balanced_accuracy']:.2%} | macro-F1={metrics['macro_f1']:.2%}"
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | "
        f"micro={metrics['micro_accuracy']:.3f}, bal={metrics['balanced_accuracy']:.3f}, macroF1={metrics['macro_f1']:.3f}"
    )

    return metrics, label_order, acc_matrix


def plot_binary_knn(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        logging.warning(f"[KNN] no embeddings for {title}")
        return None, None, None

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    present = sorted(set(labels))
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | micro={metrics['micro_accuracy']:.3f}, "
        f"balanced={metrics['balanced_accuracy']:.3f}, "
        f"macroF1={metrics['macro_f1']:.3f}, "
        f"weightedF1={metrics['weighted_f1']:.3f}, "
        f"kappa={metrics['cohens_kappa']:.3f}, "
        f"mcc={metrics['mcc']:.3f}"
    )
    return metrics, label_order, acc_matrix


# ---------------- saving reps ----------------

def save_reps(base_dir, model_tag, name, X, labels, metas, extra=None):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    prefix = f"reps_{model_tag}_{name}"
    np.savez_compressed(base_dir / f"{prefix}.npz", embeddings=X, labels=labels)

    mdf = pd.DataFrame(metas)
    if "label" in mdf.columns:
        mdf["label"] = labels
    else:
        mdf.insert(0, "label", labels)

    if extra:
        for k, v in extra.items():
            mdf[k] = v

    mdf.to_parquet(base_dir / f"{prefix}_meta.parquet", index=False)


def save_sampled_examples_tsv(output_dir: Path, df_sampled: pd.DataFrame, seed: int, n_examples: int):
    out = output_dir / "sampled_examples_atg5.tsv"
    meta = {
        "n_examples_requested": int(n_examples),
        "n_examples_saved": int(len(df_sampled)),
        "seed": int(seed),
    }
    df_sampled.to_csv(out, sep="\t", index=False)
    with open(output_dir / "sampled_examples_atg5.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"wrote sampled examples TSV: {out}")


# ---------------- ATG 5-way context loading ----------------

LABEL_COLS_5WAY = {
    1: "label1_start_pos",
    2: "label2_noncoding_near_pos",
    3: "label3_noncoding_far_pos",
    4: "label4_same_inframe_met_pos",
    5: "label5_same_outframe_atg_pos",
}

DELTA_COLS_5WAY = {
    1: None,
    2: "label2_delta_bp",
    3: "label3_delta_bp",
    4: "label4_delta_bp",
    5: "label5_delta_bp",
}


def _even_sample_by_chrom(df: pd.DataFrame, chromosomes: list[str], n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = df[df["chrom"].isin(chromosomes)].copy()
    if len(df) == 0:
        return df

    base = n_total // len(chromosomes)
    rem = n_total % len(chromosomes)

    targets = {c: base for c in chromosomes}
    for c in chromosomes[:rem]:
        targets[c] += 1

    selected = []
    remaining = df.copy()
    remaining_targets = targets.copy()

    while True:
        progress = False
        carry = 0

        for c in chromosomes:
            sub = remaining[remaining["chrom"] == c]
            want = remaining_targets.get(c, 0)
            if want <= 0:
                continue

            have = len(sub)
            take = min(want, have)

            if take > 0:
                idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
                selected.append(remaining.loc[idx])
                remaining = remaining.drop(index=idx)
                progress = True

            if have < want:
                carry += (want - have)

            remaining_targets[c] = 0

        if not progress:
            break

        if len(pd.concat(selected, axis=0)) >= n_total:
            break

        if carry <= 0:
            break

        avail = []
        for c in chromosomes:
            have_left = (remaining["chrom"] == c).sum()
            if have_left > 0:
                avail.append((c, have_left))

        if not avail:
            break

        i = 0
        while carry > 0 and avail:
            c, _have_left = avail[i % len(avail)]
            remaining_targets[c] = remaining_targets.get(c, 0) + 1
            carry -= 1
            i += 1

    out = pd.concat(selected, axis=0) if selected else df.iloc[0:0].copy()
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=seed)
    return out


def load_atg_5way_contexts_from_tsv(
    atg_tsv_path: str,
    bigwig_file: str,
    genome: Fasta,
    model_type: str,
    n_examples: int = 1000,
    seed: int = 42,
    chromosomes: list[str] | None = None,
    sampled_examples_tsv: str | None = None,
    output_dir_for_sampling: Path | None = None,
    snap_to_6mer: bool = False,
):
    _ = pyBigWig.open(bigwig_file).close()

    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in range(1, 23)]

    if sampled_examples_tsv is not None:
        sampled = pd.read_csv(sampled_examples_tsv, sep="\t")
        logging.info(f"loaded sampled examples from: {sampled_examples_tsv} (n={len(sampled)})")
    else:
        df = pd.read_csv(atg_tsv_path, sep="\t")
        required = list(LABEL_COLS_5WAY.values()) + ["transcript_id", "gene_id", "strand", "chrom"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"missing required column in TSV: {c}")

        for col in LABEL_COLS_5WAY.values():
            df = df[df[col].astype(str) != "."]
        df = df.dropna(subset=list(LABEL_COLS_5WAY.values()) + ["chrom", "transcript_id", "strand"])

        sampled = _even_sample_by_chrom(df, chromosomes=chromosomes, n_total=n_examples, seed=seed)
        logging.info(f"sampling: requested n_examples={n_examples}, got n={len(sampled)}")

        if output_dir_for_sampling is not None:
            output_dir_for_sampling = Path(output_dir_for_sampling)
            output_dir_for_sampling.mkdir(parents=True, exist_ok=True)
            save_sampled_examples_tsv(output_dir_for_sampling, sampled, seed=seed, n_examples=n_examples)

    if len(sampled) == 0:
        return []

    counts = sampled["chrom"].value_counts().reindex(chromosomes, fill_value=0)
    logging.info("per-chrom example counts:\n" + "\n".join([f"  {c}: {int(counts[c])}" for c in chromosomes]))

    contexts = []
    for _, row in sampled.iterrows():
        try:
            pos_dict = {lid: int(row[col]) for lid, col in LABEL_COLS_5WAY.items()}
        except Exception:
            continue

        anchor = pos_dict[1]
        example_id = f"{row['chrom']}|{row['transcript_id']}|{row['strand']}|{anchor}"

        ok = True
        example_contexts = []
        for lid, pos in pos_dict.items():
            region = {
                "chrom": row["chrom"],
                "start": pos,
                "end": pos + 3,
                "feature_id": f"{row['transcript_id']}_L{lid}",
                "strand": row["strand"],
            }
            ctx = extract_context(
                bigwig_file, region, genome, model_type,
                snap_to_6mer=snap_to_6mer,
            )
            if not ctx or "sequence" not in ctx:
                ok = False
                break

            ctx["example_id"] = example_id
            ctx["label_id"] = lid
            ctx["delta_bp"] = 0 if DELTA_COLS_5WAY[lid] is None else int(row[DELTA_COLS_5WAY[lid]])
            ctx["transcript_id"] = row["transcript_id"]
            ctx["gene_id"] = row["gene_id"]
            ctx["strand"] = row["strand"]

            example_contexts.append(ctx)

        if ok:
            contexts.extend(example_contexts)

    logging.info(f"total contexts loaded: {len(contexts)} (expected ~ {5*len(sampled)})")
    return contexts


def maybe_load_cached_reps(output_dir: Path, model_tag: str, name: str):
    prefix = f"reps_{model_tag}_{name}"
    npz = output_dir / f"{prefix}.npz"
    meta = output_dir / f"{prefix}_meta.parquet"
    if npz.exists() and meta.exists():
        logging.info(f"[cache] found existing reps, skipping embedding: {npz.name}")
        d = np.load(npz, allow_pickle=True)
        X = d["embeddings"].astype(np.float32)
        y = d["labels"].astype(int)
        mdf = pd.read_parquet(meta)
        metas = mdf.to_dict(orient="records")
        return X, y, metas
    return None


# ---------------- embedding ----------------

def compute_atg_roi_embeddings(
    model,
    tokenizer,
    contexts,
    batch_size,
    device,
    model_type,
    training_task,
    baseline,
    kmer_k,
    use_6mer_roi: bool = False,
):
    """Compute embeddings using model or baselines.

    use_6mer_roi behaviour per baseline:
      - "none"          : pool 6 char-level tokens (gamba/caduceus have no CLS)
      - "phylop"        : summarize 6 phyloP scores instead of 3
      - "kmer6"         : unchanged — k-mer identity is sequence-determined
      - "kmer6_flanked" : unchanged — already uses a 6bp window by design
    """
    logging.info(
        f"computing atg roi embeddings for {len(contexts)} contexts, "
        f"model_type={model_type}, task={training_task}, baseline={baseline}, "
        f"use_6mer_roi={use_6mer_roi}"
    )

    roi_embeds = []
    full_embeds = []
    label_ids = []
    metas = []

    if baseline == "none":
        seq_reps, region_info = predict_scores_batched(
            model,
            tokenizer,
            contexts,
            batch_size=batch_size,
            device=device,
            model_type=model_type,
            training_task=training_task,
        )

        assert len(seq_reps) == len(region_info) == len(contexts)

        for ctx, info in zip(contexts, region_info):
            info["example_id"] = ctx["example_id"]
            info["label_id"] = ctx["label_id"]
            info["delta_bp"] = ctx.get("delta_bp", 0)
            info["chrom"] = ctx.get("chrom", info.get("chrom", None))
            info["start"] = ctx.get("start", info.get("start", -1))
            info["end"] = ctx.get("end", info.get("end", -1))
            info["transcript_id"] = ctx.get("transcript_id", "")
            info["gene_id"] = ctx.get("gene_id", "")
            info["strand"] = ctx.get("strand", "")

        for rep, info in zip(seq_reps, region_info):
            rep = np.asarray(rep, dtype=np.float32)
            if rep.ndim != 2:
                continue

            T = rep.shape[0]
            fs = int(info.get("feature_start_in_window", 0))
            fe = int(info.get("feature_end_in_window", T))

            full_vec = rep.mean(axis=0)

            if use_6mer_roi:
                # gamba and caduceus are both char-level with no CLS token:
                # pool 6 consecutive tokens starting at fs (after snap, fs % 6 == 0)
                tfs = max(0, min(fs, T - 1))
                tfe = max(tfs + 1, min(fs + 6, T))
            else:
                if fe <= fs or fs < 0 or fe > T:
                    continue
                tfs, tfe = fs, fe

            rep_slice = rep[tfs:tfe]
            if rep_slice.shape[0] == 0:
                continue

            pooled = rep_slice.mean(axis=0)

            full_embeds.append(full_vec.astype(np.float32))
            roi_embeds.append(pooled.astype(np.float32))
            label_ids.append(int(info["label_id"]))
            metas.append(
                {
                    "example_id": info["example_id"],
                    "label_id": int(info["label_id"]),
                    "chrom": info.get("chrom"),
                    "start": int(info.get("start", -1)),
                    "end": int(info.get("end", -1)),
                    "delta_bp": int(info.get("delta_bp", 0)),
                    "feature_start_in_window": tfs,
                    "feature_end_in_window": tfe,
                    "transcript_id": info.get("transcript_id", ""),
                    "gene_id": info.get("gene_id", ""),
                    "strand": info.get("strand", ""),
                }
            )

    elif baseline == "kmer6_flanked":
        # Already extracts a 6bp window (2bp before ATG + ATG + 1bp after).
        # snap_to_6mer ensures consistent boundary alignment but the logic here is unchanged.
        # use_6mer_roi has no further effect.
        kmer_index = _build_kmer_index(k=kmer_k)

        for ctx in contexts:
            seq = ctx.get("sequence", "")
            if not seq:
                continue

            fs = int(ctx.get("feature_start_in_window", 0))
            fe = int(ctx.get("feature_end_in_window", len(seq)))

            full_vec = _seq_to_kmer_vec(seq, k=kmer_k, kmer_index=kmer_index)

            window_start = max(0, fs - 2)
            window_end = min(len(seq), fe + 1)
            roi_seq = seq[window_start:window_end]

            if len(roi_seq) != 6:
                logging.warning(f"flanked kmer: got {len(roi_seq)}bp instead of 6bp, skipping")
                continue

            kmer_vec = _seq_to_kmer_vec(roi_seq, k=kmer_k, kmer_index=kmer_index)

            full_embeds.append(full_vec)
            roi_embeds.append(kmer_vec)
            label_ids.append(int(ctx["label_id"]))
            metas.append({
                "example_id": ctx["example_id"],
                "label_id": int(ctx["label_id"]),
                "chrom": ctx.get("chrom"),
                "start": int(ctx.get("start", -1)),
                "end": int(ctx.get("end", -1)),
                "delta_bp": int(ctx.get("delta_bp", 0)),
                "feature_start_in_window": window_start,
                "feature_end_in_window": window_end,
                "flanked_sequence": roi_seq,
                "transcript_id": ctx.get("transcript_id", ""),
                "gene_id": ctx.get("gene_id", ""),
                "strand": ctx.get("strand", ""),
            })

    elif baseline == "kmer6":
        # ROI is ATG-only (3bp). use_6mer_roi has no effect here —
        # k-mer identity is sequence-determined; snapping only gives consistent
        # boundary alignment for the surrounding context window.
        kmer_index = _build_kmer_index(k=3)

        for ctx in contexts:
            seq = ctx.get("sequence", "")
            if not seq:
                continue

            fs = int(ctx.get("feature_start_in_window", 0))
            fe = int(ctx.get("feature_end_in_window", len(seq)))

            full_vec = _seq_to_kmer_vec(seq, k=3, kmer_index=kmer_index)

            roi_seq = seq[fs:fe]
            kmer_vec = _seq_to_kmer_vec(roi_seq, k=3, kmer_index=kmer_index)

            full_embeds.append(full_vec)
            roi_embeds.append(kmer_vec)
            label_ids.append(int(ctx["label_id"]))
            metas.append(
                {
                    "example_id": ctx["example_id"],
                    "label_id": int(ctx["label_id"]),
                    "chrom": ctx.get("chrom"),
                    "start": int(ctx.get("start", -1)),
                    "end": int(ctx.get("end", -1)),
                    "delta_bp": int(ctx.get("delta_bp", 0)),
                    "feature_start_in_window": fs,
                    "feature_end_in_window": fe,
                    "transcript_id": ctx.get("transcript_id", ""),
                    "gene_id": ctx.get("gene_id", ""),
                    "strand": ctx.get("strand", ""),
                }
            )

    elif baseline == "phylop":
        for ctx in contexts:
            scores = ctx.get("scores")
            if scores is None:
                continue

            scores = np.asarray(scores, dtype=np.float32)
            fs = int(ctx.get("feature_start_in_window", 0))
            fe = int(ctx.get("feature_end_in_window", len(scores)))

            full_vec = _summarize_scores(scores)
            if full_vec is None or np.isnan(full_vec).all():
                continue

            if use_6mer_roi:
                # pool 6 phyloP scores (ATG + 3 flanking) to match the 6-nt ROI
                # of the neural models and the kmer6_flanked baseline
                roi_end = min(fs + 6, len(scores))
                roi_scores = scores[fs:roi_end]
            else:
                roi_scores = scores[fs:fe]  # original 3bp ATG window

            phylop_vec = _summarize_scores(roi_scores)
            if np.isnan(phylop_vec).all():
                continue

            full_embeds.append(full_vec)
            roi_embeds.append(phylop_vec)
            label_ids.append(int(ctx["label_id"]))
            metas.append(
                {
                    "example_id": ctx["example_id"],
                    "label_id": int(ctx["label_id"]),
                    "chrom": ctx.get("chrom"),
                    "start": int(ctx.get("start", -1)),
                    "end": int(ctx.get("end", -1)),
                    "delta_bp": int(ctx.get("delta_bp", 0)),
                    "feature_start_in_window": fs,
                    "feature_end_in_window": fs + 6 if use_6mer_roi else fe,
                    "transcript_id": ctx.get("transcript_id", ""),
                    "gene_id": ctx.get("gene_id", ""),
                    "strand": ctx.get("strand", ""),
                }
            )

    else:
        raise ValueError(f"unsupported baseline: {baseline}")

    if len(roi_embeds) == 0:
        logging.error("no roi embeddings produced")
        return np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32), np.array([]), []

    roi_embeds = np.stack(roi_embeds)
    full_embeds = np.stack(full_embeds)
    label_ids = np.asarray(label_ids, dtype=int)
    logging.info(f"roi_embeds shape={roi_embeds.shape}, full_embeds shape={full_embeds.shape}, n={len(label_ids)}")
    return roi_embeds, full_embeds, label_ids, metas


# ---------------- main analysis ----------------

def analyze_atg_5way_knn(
    atg_tsv_path,
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    model_type,
    training_task,
    last_step,
    batch_size,
    n_examples,
    seed,
    chromosomes,
    sampled_examples_tsv,
    baseline,
    kmer_k,
    use_6mer_roi: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_6mer" if use_6mer_roi else ""

    if baseline == "none":
        step_tag = "random_init" if last_step == 0 else str(last_step)
        model_tag = f"{model_type}_{training_task}_step{step_tag}"
    else:
        model_tag = baseline

    cached = maybe_load_cached_reps(output_dir, model_tag, f"ATG_5way_all_labels{suffix}")
    if cached is not None:
        roi_embeds, label_ids, metas = cached
        logging.info(f"[cache] loaded roi_embeds shape={roi_embeds.shape}")

        plot_knn_heatmap(
            roi_embeds,
            label_ids,
            output_path=output_dir / f"knn_heatmap_{model_tag}_ATG5way_all_labels{suffix}.png",
            title=f"ATG 5-way 1-NN ({model_tag})",
        )

        for target_label in range(2, 6):
            indices = np.where((label_ids == 1) | (label_ids == target_label))[0]
            if len(indices) == 0:
                logging.warning(f"[KNN] no examples for 1-vs-{target_label}, skipping plot")
                continue
            plot_binary_knn(
                roi_embeds[indices],
                label_ids[indices],
                output_path=output_dir / f"knn_heatmap_{model_tag}_ATG1_vs_{target_label}{suffix}.png",
                title=f"ATG 1-vs-{target_label} 1-NN ({model_tag})",
            )
        return

    # --- full run ---
    if baseline in ("kmer6", "phylop"):
        ctx_model_type = "baseline"
    else:
        ctx_model_type = model_type

    if baseline == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"using device: {device}")
        model, tokenizer = load_model(
            checkpoint_dir,
            config_fpath,
            last_step=last_step,
            device=device,
            training_task=training_task,
            model_type=model_type,
        )
    else:
        model = None
        tokenizer = None
        device = None

    genome = Fasta(genome_fasta)

    contexts = load_atg_5way_contexts_from_tsv(
        atg_tsv_path=atg_tsv_path,
        bigwig_file=bigwig_file,
        genome=genome,
        model_type=ctx_model_type,
        n_examples=n_examples,
        seed=seed,
        chromosomes=chromosomes,
        sampled_examples_tsv=sampled_examples_tsv,
        output_dir_for_sampling=output_dir,
        snap_to_6mer=use_6mer_roi,
    )
    if not contexts:
        logging.error("no contexts loaded, aborting")
        return

    roi_embeds, full_embeds, label_ids, metas = compute_atg_roi_embeddings(
        model,
        tokenizer,
        contexts,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
        training_task=training_task,
        baseline=baseline,
        kmer_k=kmer_k,
        use_6mer_roi=use_6mer_roi,
    )
    if roi_embeds.shape[0] == 0:
        logging.error("empty embeddings, aborting")
        return

    # Enforce strict 5-per-example
    index_by_example = defaultdict(dict)
    for i, meta in enumerate(metas):
        index_by_example[meta["example_id"]][int(meta["label_id"])] = i

    valid_examples = [
        ex_id for ex_id, lids in index_by_example.items()
        if all(l in lids for l in (1, 2, 3, 4, 5))
    ]
    logging.info(f"valid examples with all 5 labels after embedding: {len(valid_examples)}")

    keep_indices = []
    for ex in valid_examples:
        for lid in (1, 2, 3, 4, 5):
            keep_indices.append(index_by_example[ex][lid])
    keep_indices = np.asarray(keep_indices, dtype=int)

    roi_embeds = roi_embeds[keep_indices]
    full_embeds = full_embeds[keep_indices]
    label_ids = label_ids[keep_indices]
    metas = [metas[i] for i in keep_indices.tolist()]

    extra_roi = {
        "model_type": model_type if baseline == "none" else baseline,
        "training_task": training_task if baseline == "none" else "N/A",
        "last_step": last_step if baseline == "none" else 0,
        "scope": "roi_all",
        "n_examples_requested": int(n_examples),
        "seed": int(seed),
        "baseline": baseline,
        "use_6mer_roi": use_6mer_roi,
    }
    extra_full = {
        "model_type": model_type if baseline == "none" else baseline,
        "training_task": training_task if baseline == "none" else "N/A",
        "last_step": last_step if baseline == "none" else 0,
        "scope": "full_all",
        "n_examples_requested": int(n_examples),
        "seed": int(seed),
        "baseline": baseline,
        "use_6mer_roi": use_6mer_roi,
    }
    save_reps(output_dir, model_tag, f"ATG_5way_all_labels{suffix}", roi_embeds, label_ids, metas, extra=extra_roi)
    save_reps(output_dir, model_tag, f"ATG_5way_all_labels_full{suffix}", full_embeds, label_ids, metas, extra=extra_full)

    # Plots
    metrics5, _, _ = plot_knn_heatmap(
        roi_embeds,
        label_ids,
        output_path=output_dir / f"knn_heatmap_{model_tag}_ATG5way_all_labels{suffix}.png",
        title=f"ATG 5-way 1-NN ({model_tag})",
    )

    task_metrics = {}
    if metrics5:
        task_metrics["task5way_balanced_accuracy"] = float(metrics5["balanced_accuracy"])
        task_metrics["task5way_micro_accuracy"] = float(metrics5["micro_accuracy"])

    for target_label in range(2, 6):
        indices = np.where((label_ids == 1) | (label_ids == target_label))[0]
        if len(indices) == 0:
            continue
        mk, _, _ = plot_binary_knn(
            roi_embeds[indices],
            label_ids[indices],
            output_path=output_dir / f"knn_heatmap_{model_tag}_ATG1_vs_{target_label}{suffix}.png",
            title=f"ATG 1-vs-{target_label} 1-NN ({model_tag})",
        )
        if mk:
            task_metrics[f"1_vs_{target_label}_balanced_accuracy"] = float(mk["balanced_accuracy"])

    with open(output_dir / f"balanced_accuracy_{model_tag}_ATG5way{suffix}.json", "w") as f:
        json.dump(task_metrics, f, indent=2)

    logging.info(f"task balanced accuracies: {task_metrics}")


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="ATG 5-way codon representation tasks for gamba / caduceus / baselines (1-NN)"
    )
    parser.add_argument(
        "--atg_tsv_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/ATGs/all_chr_atg_5way.tsv",
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="/home/mica/gamba/")
    parser.add_argument(
        "--config_fpath",
        type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/ATG_reps_5way",
    )

    parser.add_argument("--model_type", type=str, choices=["gamba", "caduceus"], default=None)
    parser.add_argument("--training_task", type=str, choices=["dual", "cons_only", "seq_only"], default=None)
    parser.add_argument("--last_step", type=int, default=44000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--baseline",
        type=str,
        choices=["none", "kmer6", "kmer6_flanked", "phylop"],
        default="none",
    )
    parser.add_argument("--kmer_k", type=int, default=6)

    parser.add_argument(
        "--use_6mer_roi",
        action="store_true",
        default=False,
        help=(
            "Snap context window to 6-mer boundary and pool a 6-nt ROI. "
            "For baseline==none (gamba/caduceus): pools 6 char-level tokens. "
            "For baseline==phylop: summarizes 6 phyloP scores instead of 3. "
            "For kmer6/kmer6_flanked: no change to ROI (k-mer identity is sequence-determined). "
            "Appends '_6mer' to all saved filenames."
        ),
    )

    parser.add_argument("--n_examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[f"chr{i}" for i in range(1, 23)],
    )
    parser.add_argument("--sampled_examples_tsv", type=str, default=None)

    args = parser.parse_args()

    if args.baseline == "none":
        if args.model_type is None or args.training_task is None:
            raise SystemExit("when --baseline=none, provide --model_type and --training_task")

    checkpoint_dir = (
        os.path.join(args.checkpoint_dir, "clean_dcps/CCP/")
        if args.model_type == "gamba"
        else args.checkpoint_dir
    )

    if args.baseline == "none":
        last_tag = "random_init" if args.last_step == 0 else args.last_step
        outdir = os.path.join(args.output_dir, f"ATG5_{args.model_type}_{args.training_task}_step_{last_tag}")
    else:
        outdir = os.path.join(args.output_dir, f"ATG5_{args.baseline}")

    os.makedirs(outdir, exist_ok=True)

    analyze_atg_5way_knn(
        atg_tsv_path=args.atg_tsv_path,
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        checkpoint_dir=checkpoint_dir,
        config_fpath=args.config_fpath,
        output_dir=outdir,
        model_type=args.model_type if args.baseline == "none" else "baseline",
        training_task=args.training_task if args.baseline == "none" else "N/A",
        last_step=args.last_step,
        batch_size=args.batch_size,
        n_examples=args.n_examples,
        seed=args.seed,
        chromosomes=args.chromosomes,
        sampled_examples_tsv=args.sampled_examples_tsv,
        baseline=args.baseline,
        kmer_k=args.kmer_k,
        use_6mer_roi=args.use_6mer_roi,
    )


if __name__ == "__main__":
    main()