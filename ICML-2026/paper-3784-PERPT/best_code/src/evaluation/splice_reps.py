#!/usr/bin/env python3
"""
Splice site 8-way (gamba/caduceus) with chromosome-stratified sampling.

- loads ONE TSV: /home/mica/gamba/data_processing/data/splice_sites/all_chr1_22_splice_8way_complete.tsv
- samples N examples total, approximately evenly across chr1..chr22 (default N=1000)
- builds 8 contexts per example (so total contexts = 8*N; default 8000)
- embeds with gamba/caduceus via predict_scores_batched
- saves reps_{model_tag}_SPLICE_8way_all_labels.{npz,parquet}
- runs:
  - 8-way 1-NN confusion heatmap on labels 1..8
  - binary 1-NN tasks: 1 vs each of 2..8
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


# ---------------- KNN + metrics helpers ----------------

# --- drop into splice_reps.py (or a small utils section) ---

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# label groups (splice 8-way)
DONOR_SET = {1, 7}
ACCEPTOR_SET = {2, 8}

# “within-group” binary subproblems you likely care about
WITHIN_GROUP_TASKS = [
    ("donor: L1 vs L7", (1, 7)),
    ("acceptor: L2 vs L8", (2, 8)),
    ("GT: exon vs intron (L3 vs L4)", (3, 4)),
    ("AG: exon vs intron (L5 vs L6)", (5, 6)),
]

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]
    return y_true, y_pred

def _balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    # cm rows = true, cols = pred
    row_sums = cm.sum(axis=1, keepdims=True)
    recalls = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()
    valid = ~np.isnan(recalls)
    return float(np.mean(recalls[valid])) if np.any(valid) else float("nan")

def eval_hard_metrics(y_true, y_pred, label_order=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if label_order is None:
        label_order = sorted(set(y_true.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    ba = _balanced_accuracy_from_cm(cm)
    micro = float((y_true == y_pred).mean())
    per_class_recall = {}
    for i, lab in enumerate(label_order):
        denom = cm[i, :].sum()
        per_class_recall[int(lab)] = float(cm[i, i] / denom) if denom else float("nan")
    support = {int(lab): int(cm[i, :].sum()) for i, lab in enumerate(label_order)}
    return {
        "micro_accuracy": micro,
        "balanced_accuracy": ba,
        "per_class_recall": per_class_recall,
        "support": support,
        "cm": cm,
        "label_order": label_order,
    }

def _collapse_splice_donor_acceptor(labels_8way: np.ndarray) -> np.ndarray:
    """
    collapse to 2-class:
      donor (L1,L7) -> 0
      acceptor (L2,L8) -> 1
    all other labels (L3-6) are excluded by caller.
    """
    out = np.empty_like(labels_8way, dtype=int)
    out[:] = -1
    out[np.isin(labels_8way, list(DONOR_SET))] = 0
    out[np.isin(labels_8way, list(ACCEPTOR_SET))] = 1
    return out

def donor_vs_acceptor_ba(embeddings, labels_8way):
    """
    Uses only {1,2,7,8}, collapses to donor vs acceptor, then LOO 1-NN BA.
    """
    labels_8way = np.asarray(labels_8way, dtype=int)
    keep = np.isin(labels_8way, [1, 2, 7, 8])
    X = np.asarray(embeddings)[keep]
    y8 = labels_8way[keep]
    if len(y8) == 0:
        return {"micro_accuracy": float("nan"), "balanced_accuracy": float("nan"), "n": 0}

    y2 = _collapse_splice_donor_acceptor(y8)
    # sanity: no -1 left
    mask = y2 >= 0
    X, y2 = X[mask], y2[mask]

    y_true, y_pred = loo_1nn_predictions(X, y2)
    m = eval_hard_metrics(y_true, y_pred, label_order=[0, 1])
    return {"micro_accuracy": m["micro_accuracy"], "balanced_accuracy": m["balanced_accuracy"], "n": int(len(y2))}

def within_group_mean_ba(embeddings, labels_8way, tasks=WITHIN_GROUP_TASKS):
    """
    For each 2-class subtask (a,b), restrict to those labels and compute BA via LOO 1-NN.
    Returns mean BA across tasks (unweighted), plus per-task details.
    """
    labels_8way = np.asarray(labels_8way, dtype=int)
    X_all = np.asarray(embeddings)

    per_task = {}
    bas = []

    for name, (a, b) in tasks:
        keep = np.isin(labels_8way, [a, b])
        X = X_all[keep]
        y = labels_8way[keep]
        if len(y) == 0 or len(set(y.tolist())) < 2:
            per_task[name] = {"balanced_accuracy": float("nan"), "micro_accuracy": float("nan"), "n": int(len(y))}
            continue

        y_true, y_pred = loo_1nn_predictions(X, y)
        m = eval_hard_metrics(y_true, y_pred, label_order=[a, b])
        per_task[name] = {
            "balanced_accuracy": m["balanced_accuracy"],
            "micro_accuracy": m["micro_accuracy"],
            "n": int(len(y)),
        }
        if not np.isnan(m["balanced_accuracy"]):
            bas.append(m["balanced_accuracy"])

    mean_ba = float(np.mean(bas)) if len(bas) else float("nan")
    return {"mean_balanced_accuracy": mean_ba, "per_task": per_task}

def _binary_ba_from_embeddings(X, y, a, b):
    mask = (y == a) | (y == b)
    X2 = X[mask]
    y2 = y[mask]
    if len(y2) == 0 or (np.sum(y2 == a) == 0) or (np.sum(y2 == b) == 0):
        return None  # missing a class

    yt, yp = loo_1nn_predictions(X2, y2)
    cm = confusion_matrix(yt, yp, labels=[a, b])
    rec_a = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() else np.nan
    rec_b = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() else np.nan
    if np.isnan(rec_a) or np.isnan(rec_b):
        return None
    return float(0.5 * (rec_a + rec_b))


def splice_three_metrics(X, y):
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    # 8-way hard BA (on whatever labels are present in X,y)
    yt, yp = loo_1nn_predictions(X, y)
    hard = eval_hard_metrics(yt, yp, label_order=sorted(set(y.tolist())))
    hard_ba = float(hard["balanced_accuracy"])

    # donor vs acceptor BA: only defined on {1,7} vs {2,8}
    da_mask = np.isin(y, [1, 2, 7, 8])
    donor_acceptor_ba = None
    if np.any(da_mask) and len(set(y[da_mask].tolist())) == 2:
        # map donors->0, acceptors->1
        y_da = np.where(np.isin(y[da_mask], [1, 7]), 0, 1)
        yt_da, yp_da = loo_1nn_predictions(X[da_mask], y_da)
        cm = confusion_matrix(yt_da, yp_da, labels=[0, 1])
        rec0 = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() else np.nan
        rec1 = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() else np.nan
        donor_acceptor_ba = float(0.5 * (rec0 + rec1)) if not (np.isnan(rec0) or np.isnan(rec1)) else None

    # within-group BAs (only compute if both labels exist)
    pairs = {
        "donor: L1 vs L7": (1, 7),
        "acceptor: L2 vs L8": (2, 8),
        "GT: exon vs intron (L3 vs L4)": (3, 4),
        "AG: exon vs intron (L5 vs L6)": (5, 6),
    }

    details = {}
    bas = []
    for name, (a, b) in pairs.items():
        ba = _binary_ba_from_embeddings(X, y, a, b)
        n = int(np.sum((y == a) | (y == b)))
        details[name] = {"balanced_accuracy": (float("nan") if ba is None else float(ba)), "n": n}
        if ba is not None:
            bas.append(ba)

    within_mean = float(np.mean(bas)) if bas else float("nan")

    return {
        "8way_hard_ba": hard_ba,
        "donor_vs_acceptor_ba": float("nan") if donor_acceptor_ba is None else float(donor_acceptor_ba),
        "within_group_mean_ba": within_mean,
        "within_group_details": details,
    }

def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN", do_splice_three=False):
    labels = np.asarray(labels).astype(int)  # important
    present = sorted(set(labels.tolist()))

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    hard = eval_hard_metrics(y_true, y_pred, label_order=present)

    three = None
    if do_splice_three and set(present) == set(range(1, 9)):
        three = splice_three_metrics(np.asarray(embeddings), labels)
        for k, v in three["within_group_details"].items():
            logging.info(f"[within] {k}: BA={v['balanced_accuracy']:.3f} (n={v['n']})")

    # build heatmap from hard cm
    cm = hard["cm"]
    label_order = hard["label_order"]
    row_sums = cm.sum(axis=1, keepdims=True)
    acc_matrix = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    plt.figure(figsize=(6.8, 5.8))
    sns.heatmap(
        acc_matrix,
        xticklabels=[f"L{l}" for l in label_order],
        yticklabels=[f"L{l}" for l in label_order],
        vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )

    if three is not None:
        plt.title(
            f"{title}\n"
            f"8-way hard BA={three['8way_hard_ba']:.2%} | "
            f"donor/acceptor BA={three['donor_vs_acceptor_ba']:.2%} | "
            f"within mean BA={three['within_group_mean_ba']:.2%}"
        )
    else:
        plt.title(f"{title}\nBA={hard['balanced_accuracy']:.2%}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    out = dict(hard)
    if three is not None:
        out.update(three)
    return out, label_order, acc_matrix

# ---------------- saving reps ----------------

def save_reps(base_dir, model_tag, name, X, labels, metas, extra=None):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)

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
    """
    Save the exact sampled example rows so other models can reuse the identical set.
    """
    out = output_dir / "sampled_examples_splice8.tsv"
    meta = {
        "n_examples_requested": int(n_examples),
        "n_examples_saved": int(len(df_sampled)),
        "seed": int(seed),
    }
    df_sampled.to_csv(out, sep="\t", index=False)
    with open(output_dir / "sampled_examples_splice8.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"wrote sampled examples TSV: {out}")


# ---------------- Splice 8-way context loading ----------------

LABEL_COLS_8WAY = {
    1: "label1_donor_pos",
    2: "label2_same_acceptor_pos",
    3: "label3_same_gt_exon_pos",
    4: "label4_same_gt_intron_pos",
    5: "label5_same_ag_exon_pos",
    6: "label6_same_ag_intron_pos",
    7: "label7_diff_donor_pos",
    8: "label8_diff_acceptor_pos",
}

DELTA_COLS_8WAY = {
    1: None,
    2: "label2_delta_bp",
    3: "label3_delta_bp",
    4: "label4_delta_bp",
    5: "label5_delta_bp",
    6: "label6_delta_bp",
    7: "label7_delta_bp",
    8: "label8_delta_bp",
}


def _even_sample_by_chrom(df: pd.DataFrame, chromosomes: list[str], n_total: int, seed: int) -> pd.DataFrame:
    """
    sample ~evenly across chromosomes. if a chromosome has fewer rows than requested,
    we take all of them and re-distribute the remaining quota across others.

    returns sampled df (<= n_total).
    """
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


def load_splice_8way_contexts_from_tsv(
    splice_tsv_path: str,
    bigwig_file: str,
    genome: Fasta,
    model_type: str,
    n_examples: int = 1000,
    seed: int = 42,
    chromosomes: list[str] | None = None,
    sampled_examples_tsv: str | None = None,
    output_dir_for_sampling: Path | None = None,
):
    """
    If sampled_examples_tsv is provided, uses it directly (no re-sampling).
    Otherwise samples from splice_tsv_path and (optionally) writes sampled_examples_splice8.tsv
    into output_dir_for_sampling for reuse by other models.
    """
    _ = pyBigWig.open(bigwig_file).close()

    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in range(1, 23)]

    if sampled_examples_tsv is not None:
        sampled = pd.read_csv(sampled_examples_tsv, sep="\t")
        logging.info(f"loaded sampled examples from: {sampled_examples_tsv} (n={len(sampled)})")
    else:
        df = pd.read_csv(splice_tsv_path, sep="\t")
        required = list(LABEL_COLS_8WAY.values()) + ["ref_transcript_id", "ref_gene_id", "ref_strand", "chrom"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"missing required column in TSV: {c}")

        for col in LABEL_COLS_8WAY.values():
            df = df[df[col].astype(str) != "."]
        df = df.dropna(subset=list(LABEL_COLS_8WAY.values()) + ["chrom", "ref_transcript_id", "ref_strand"])

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
            pos_dict = {lid: int(row[col]) for lid, col in LABEL_COLS_8WAY.items()}
        except Exception:
            continue

        anchor = pos_dict[1]
        example_id = f"{row['chrom']}|{row['ref_transcript_id']}|{row['ref_strand']}|{anchor}"

        ok = True
        example_contexts = []
        for lid, pos in pos_dict.items():
            # donor/acceptor are 2bp, GT/AG motifs are 2bp
            region = {
                "chrom": row["chrom"],
                "start": pos,
                "end": pos + 2,
                "feature_id": f"{row['ref_transcript_id']}_L{lid}",
            }
            ctx = extract_context(bigwig_file, region, genome, model_type)
            if not ctx or "sequence" not in ctx:
                ok = False
                break

            ctx["example_id"] = example_id
            ctx["label_id"] = lid
            ctx["delta_bp"] = 0 if DELTA_COLS_8WAY[lid] is None else int(row[DELTA_COLS_8WAY[lid]])

            ctx["ref_transcript_id"] = row["ref_transcript_id"]
            ctx["ref_gene_id"] = row["ref_gene_id"]
            ctx["ref_strand"] = row["ref_strand"]
            ctx["diff_transcript_id"] = row.get("diff_transcript_id", ".")
            ctx["diff_gene_id"] = row.get("diff_gene_id", ".")
            ctx["diff_strand"] = row.get("diff_strand", ".")

            example_contexts.append(ctx)

        if ok:
            contexts.extend(example_contexts)

    logging.info(f"total contexts loaded: {len(contexts)} (expected ~ {8*len(sampled)})")
    return contexts


# ---------------- cached reps loader ----------------

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

def compute_splice_roi_embeddings(
    model,
    tokenizer,
    contexts,
    batch_size,
    device,
    model_type,
    training_task,
):
    logging.info(
        f"computing splice roi embeddings for {len(contexts)} contexts, model_type={model_type}, task={training_task}"
    )

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

        info["ref_transcript_id"] = ctx.get("ref_transcript_id", "")
        info["ref_gene_id"] = ctx.get("ref_gene_id", "")
        info["ref_strand"] = ctx.get("ref_strand", "")
        info["diff_transcript_id"] = ctx.get("diff_transcript_id", "")
        info["diff_gene_id"] = ctx.get("diff_gene_id", "")
        info["diff_strand"] = ctx.get("diff_strand", "")

    roi_embeds = []
    label_ids = []
    metas = []

    for rep, info in zip(seq_reps, region_info):
        rep = np.asarray(rep, dtype=np.float32)
        if rep.ndim != 2:
            continue

        fs = int(info.get("feature_start_in_window", 0))
        fe = int(info.get("feature_end_in_window", rep.shape[0]))
        if fe <= fs or fs < 0 or fe > rep.shape[0]:
            continue

        rep_slice = rep[fs:fe]
        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)

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
                "feature_start_in_window": fs,
                "feature_end_in_window": fe,
                "ref_transcript_id": info.get("ref_transcript_id", ""),
                "ref_gene_id": info.get("ref_gene_id", ""),
                "ref_strand": info.get("ref_strand", ""),
                "diff_transcript_id": info.get("diff_transcript_id", ""),
                "diff_gene_id": info.get("diff_gene_id", ""),
                "diff_strand": info.get("diff_strand", ""),
            }
        )

    if len(roi_embeds) == 0:
        logging.error("no roi embeddings produced")
        return np.empty((0, 1), dtype=np.float32), np.array([]), []

    roi_embeds = np.stack(roi_embeds)
    label_ids = np.asarray(label_ids, dtype=int)
    logging.info(f"roi_embeds shape={roi_embeds.shape}, n={len(label_ids)}")
    return roi_embeds, label_ids, metas


# ---------------- main analysis ----------------

def analyze_splice_8way_knn(
    splice_tsv_path,
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
    sampled_examples_tsv=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_tag = "random_init" if last_step == 0 else str(last_step)
    model_tag = f"{model_type}_{training_task}_step{step_tag}"

    cached = maybe_load_cached_reps(output_dir, model_tag, "SPLICE_8way_all_labels")
    if cached is not None:
        roi_embeds, label_ids, metas = cached
        label_ids = np.asarray(label_ids).astype(int)
        logging.info(f"[cache] loaded roi_embeds shape={roi_embeds.shape}")

        # 8-way heatmap
        plot_knn_heatmap(
            roi_embeds,
            label_ids,
            output_path=output_dir / f"knn_heatmap_{model_tag}_SPLICE8way_all_labels.png",
            title=f"Splice 8-way 1-NN ({model_tag})", do_splice_three=True
        )

        # binary 1-vs-rest plots
        for target_label in range(2, 9):
            indices = np.where((label_ids == 1) | (label_ids == target_label))[0]
            if len(indices) == 0:
                logging.warning(f"[KNN] no examples for 1-vs-{target_label}, skipping plot")
                continue

            sub_embeds = roi_embeds[indices]
            sub_labels = label_ids[indices]
            plot_knn_heatmap(
                sub_embeds,
                sub_labels,
                output_path=output_dir / f"knn_heatmap_{model_tag}_SPLICE1_vs_{target_label}.png",
                title=f"Splice 1-vs-{target_label} 1-NN ({model_tag})", do_splice_three=False
            )

    else:
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

        genome = Fasta(genome_fasta)

        contexts = load_splice_8way_contexts_from_tsv(
            splice_tsv_path=splice_tsv_path,
            bigwig_file=bigwig_file,
            genome=genome,
            model_type=model_type,
            n_examples=n_examples,
            seed=seed,
            chromosomes=chromosomes,
            sampled_examples_tsv=sampled_examples_tsv,
            output_dir_for_sampling=output_dir,
        )
        if not contexts:
            logging.error("no contexts loaded, aborting")
            return

        roi_embeds, label_ids, metas = compute_splice_roi_embeddings(
            model,
            tokenizer,
            contexts,
            batch_size=batch_size,
            device=device,
            model_type=model_type,
            training_task=training_task,
        )
        if roi_embeds.shape[0] == 0:
            logging.error("empty embeddings, aborting")
            return

        # ensure strict 8-per-example after embedding
        index_by_example = defaultdict(dict)
        for i, meta in enumerate(metas):
            index_by_example[meta["example_id"]][int(meta["label_id"])] = i

        valid_examples = [
            ex_id for ex_id, lids in index_by_example.items()
            if all(l in lids for l in (1, 2, 3, 4, 5, 6, 7, 8))
        ]
        logging.info(f"valid examples with all 8 labels after embedding: {len(valid_examples)}")

        keep_indices = []
        for ex in valid_examples:
            for lid in (1, 2, 3, 4, 5, 6, 7, 8):
                keep_indices.append(index_by_example[ex][lid])
        keep_indices = np.asarray(keep_indices, dtype=int)

        roi_embeds = roi_embeds[keep_indices]
        label_ids = label_ids[keep_indices]
        metas = [metas[i] for i in keep_indices.tolist()]

        extra_all = {
            "model_type": model_type,
            "training_task": training_task,
            "last_step": last_step,
            "scope": "roi_all",
            "n_examples_requested": int(n_examples),
            "seed": int(seed),
        }
        save_reps(output_dir, model_tag, "SPLICE_8way_all_labels", roi_embeds, label_ids, metas, extra=extra_all)

        # make plots after saving
        plot_knn_heatmap(
            roi_embeds,
            label_ids,
            output_path=output_dir / f"knn_heatmap_{model_tag}_SPLICE8way_all_labels.png",
            title=f"Splice 8-way 1-NN ({model_tag})",
        )

        for target_label in range(2, 9):
            indices = np.where((label_ids == 1) | (label_ids == target_label))[0]
            if len(indices) == 0:
                logging.warning(f"[KNN] no examples for 1-vs-{target_label}, skipping plot")
                continue

            sub_embeds = roi_embeds[indices]
            sub_labels = label_ids[indices]
            plot_knn_heatmap(
                sub_embeds,
                sub_labels,
                output_path=output_dir / f"knn_heatmap_{model_tag}_SPLICE1_vs_{target_label}.png",
                title=f"Splice 1-vs-{target_label} 1-NN ({model_tag})",
            )


# ---------------- cli ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Splice site 8-way representation tasks for gamba / caduceus (1-NN; chrom-stratified sampling)"
    )
    parser.add_argument(
        "--splice_tsv_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/splice_sites/all_chr1_22_splice_8way_complete.tsv",
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
        default="/home/mica/gamba/data_processing/data/240-mammalian/splice_reps_8way",
    )
    parser.add_argument("--model_type", type=str, choices=["gamba", "caduceus"], required=True)
    parser.add_argument("--training_task", type=str, choices=["dual", "cons_only", "seq_only"], required=True)
    parser.add_argument("--last_step", type=int, default=44000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--n_examples",
        type=int,
        default=1000,
        help="total examples across chr1..chr22 (8 contexts each). default => 8000 contexts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[f"chr{i}" for i in range(1, 23)],
    )
    parser.add_argument(
        "--sampled_examples_tsv",
        type=str,
        default=None,
        help="if provided, uses this TSV of sampled examples instead of re-sampling",
    )

    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.checkpoint_dir, "clean_dcps/CCP/") if args.model_type == "gamba" else args.checkpoint_dir
    last_tag = "random_init" if args.last_step == 0 else args.last_step
    outdir = os.path.join(args.output_dir, f"SPLICE8_{args.model_type}_{args.training_task}_step_{last_tag}")
    os.makedirs(outdir, exist_ok=True)

    analyze_splice_8way_knn(
        splice_tsv_path=args.splice_tsv_path,
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        checkpoint_dir=checkpoint_dir,
        config_fpath=args.config_fpath,
        output_dir=outdir,
        model_type=args.model_type,
        training_task=args.training_task,
        last_step=args.last_step,
        batch_size=args.batch_size,
        n_examples=args.n_examples,
        seed=args.seed,
        chromosomes=args.chromosomes,
        sampled_examples_tsv=args.sampled_examples_tsv,
    )


if __name__ == "__main__":
    main()


# Example usage:
# python /home/mica/gamba/src/evaluation/splice_reps.py --model_type gamba --training_task dual --last_step 44000