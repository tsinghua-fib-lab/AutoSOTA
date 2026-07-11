#!/usr/bin/env python3
"""
Predict phyloP conservation scores for genomic regions using gamba/caduceus models.

Supports sliding window inference for long sequences (>2048bp).

Sliding-window strategy (effective-region-only predictions):
- The model sees `window_size` bp (default 2048)
- We use `context_size` bp of upstream context per window (default 1000)
- We only *keep* predictions for the last `predict_size = window_size - context_size` bp of each window
- We slide forward by `predict_size` bp each step to cover [start, end)

For a region [start, end):
  window i predicts [start + i*predict_size, min(start+(i+1)*predict_size, end))
"""

import argparse
import os
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

import sys
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")

from src.evaluation.utils.helpers import extract_context
from src.evaluation.utils.specific_helpers import load_model


# ---------------- logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------- helpers (missing in original) ----------------

def _masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean over `dim` per row, ignoring masked-out entries (mask is 0/1 float)."""
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (x * mask).sum(dim=dim) / denom


def apply_effective_region_mask(
    labels: torch.Tensor,
    feature_spans,
    is_mlm: bool,
    last_k: int = 1000,
) -> torch.Tensor:
    """
    labels: (B,2,T)
      labels[:,0,:] = token labels (CE) with -100 meaning ignore
      labels[:,1,:] = conservation targets with -100 meaning ignore
    feature_spans: list[(fs, fe)] in token space (already shifted for START if needed)
    last_k: keep only the last_k tokens of the feature span
    """
    labels = labels.clone()
    B, _, T = labels.shape
    for b, (fs, fe) in enumerate(feature_spans):
        fs = int(max(0, min(T, fs)))
        fe = int(max(0, min(T, fe)))
        if fe <= fs:
            labels[b, 0, :] = -100
            labels[b, 1, :] = -100
            continue

        if last_k is not None and (fe - fs) > last_k:
            fs_eff = fe - last_k
            fe_eff = fe
        else:
            fs_eff, fe_eff = fs, fe

        keep = torch.zeros((T,), dtype=torch.bool, device=labels.device)
        keep[fs_eff:fe_eff] = True

        labels[b, 0, ~keep] = -100
        labels[b, 1, ~keep] = -100

    return labels


# ---------------- core batched inference ----------------

def predict_scores_batched(
    model,
    tokenizer,
    tokenized: bool,
    regions,
    batch_size: int = 8,
    device=None,
    model_type: str = "gamba",
    training_task: str = "dual",
    effective_only: bool = True,
    last_k: int = 1000,
):
    """
    Run predictions on regions with masking applied over the feature region.

    Returns:
      all_predictions:     per-example CONS loss proxy (MSE over effective region) or NaN
      all_true_scores:     list of 1D true phyloP arrays per region (full window targets)
      region_info:         list of dicts with metadata incl. feature spans
      all_seq_predictions: per-example CE over effective region
      all_true_seqs:       list of token arrays per region
      all_pred_scores:     list of 1D per-base predicted cons scores per region
                           - if effective_only: only over the effective region (post-mask span)
                           - else: full window length (after dropping START)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torch.nn import functional as F
    from gamba.collators import gLMCollator, gLMMLMCollator

    all_predictions = []
    all_true_scores = []
    all_seq_predictions = []
    all_true_seqs = []
    region_info = []
    all_pred_scores = []

    logging.info(f"Running predictions on {len(regions)} regions with batch size {batch_size}...")

    if model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

    for i in tqdm(range(0, len(regions), batch_size), desc="Batch predictions"):
        batch_regions = regions[i:i + batch_size]
        batch_inputs = []
        batch_region_info = []

        for region in batch_regions:
            if tokenized:
                sequence_tokens = region["sequence"]
            else:
                sequence_tokens = tokenizer.tokenizeMSA(region["sequence"])

            scores = region["scores"]
            fs = int(region.get("feature_start_in_window", 0))
            fe = int(region.get("feature_end_in_window", len(scores)))

            batch_inputs.append((sequence_tokens, scores))
            meta = {
                "chrom": region.get("chrom", "unknown"),
                "start": int(region.get("start", -1)),
                "end": int(region.get("end", -1)),
                "feature_id": region.get("feature_id", "unknown"),
                "mean_score": float(region.get("mean_score", 0.0)),
                "feature_start_in_window": fs,
                "feature_end_in_window": fe,
            }
            batch_region_info.append(meta)
            region_info.append(meta)

            all_true_scores.append(scores)
            all_true_seqs.append(sequence_tokens)

        if not batch_inputs:
            continue

        # ---------------- Gamba ----------------
        if model_type == "gamba":
            inputs, labels = collator(batch_inputs)  # inputs: (B,2,T), labels: (B,2,T)
            inputs, labels = inputs.to(device), labels.to(device)

            # shift spans for [START] token (collator prepends START)
            feature_spans = [(m["feature_start_in_window"] + 1, m["feature_end_in_window"] + 1)
                             for m in batch_region_info]

            labels_masked = apply_effective_region_mask(labels, feature_spans, is_mlm=False, last_k=last_k)

            with torch.no_grad():
                out = model(inputs, labels_masked)  # dict-like

            seq_logits = out["seq_logits"].float() if "seq_logits" in out else None       # (B,T,V)
            cons_pred = out["scaling_logits"].float() if "scaling_logits" in out else None  # (B,T,2?) or (B,T,*)

            # CE per-example (AR shift)
            if seq_logits is not None:
                ce_labels = labels_masked[:, 0, :].long()
                logit_shift = seq_logits[:, :-1, :]
                label_shift = ce_labels[:, 1:]
                mask_shift = label_shift.ne(-100).float()

                ce_tok = F.cross_entropy(
                    logit_shift.reshape(-1, logit_shift.size(-1)),
                    label_shift.reshape(-1),
                    reduction="none",
                ).view(label_shift.size())

                ce_per_ex = _masked_mean_per_row(ce_tok, mask_shift, dim=1)
                all_seq_predictions.extend(ce_per_ex.detach().cpu().tolist())
            else:
                all_seq_predictions.extend([float("nan")] * inputs.size(0))

            # CONS MSE per-example + per-base predictions
            if cons_pred is not None:
                # use channel 0 as mean if present; if shape is (B,T), keep it
                if cons_pred.ndim == 3:
                    cons_mean = cons_pred[..., 0].float()  # (B,T)
                else:
                    cons_mean = cons_pred.float()          # (B,T)

                cons_tgt = labels_masked[:, 1, :].float()     # (B,T)
                cons_mask = cons_tgt.ne(-100).float()

                mse_tok = (cons_mean - cons_tgt).pow(2)
                mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)
                all_predictions.extend(mse_per_ex.detach().cpu().tolist())

                # per-base predictions:
                # drop START to align to original window coordinate system
                cons_np = cons_mean.detach().cpu().numpy()[:, 1:]      # (B, T-1)
                cons_keep = cons_mask.detach().cpu().numpy()[:, 1:]    # (B, T-1) 0/1

                for b in range(cons_np.shape[0]):
                    if effective_only:
                        idx = np.where(cons_keep[b] > 0.0)[0]
                        all_pred_scores.append(cons_np[b, idx].astype(np.float32))
                    else:
                        all_pred_scores.append(cons_np[b].astype(np.float32))
            else:
                all_predictions.extend([float("nan")] * inputs.size(0))
                for _ in range(inputs.size(0)):
                    all_pred_scores.append(np.array([], dtype=np.float32) if effective_only
                                           else np.full(inputs.size(2) - 1, np.nan, dtype=np.float32))

        # ---------------- Caduceus ----------------
        elif model_type == "caduceus":
            raw_spans = [(r["feature_start_in_window"], r["feature_end_in_window"])
                         for r in batch_region_info]

            try:
                batch = collator(batch_inputs, region=raw_spans)
            except TypeError:
                batch = collator(batch_inputs, region=raw_spans)

            sequence_input = batch[0][:, 0, :].long().to(device)
            labels_pack = batch[1].to(device)  # (B,2,T)

            feature_spans_shifted = [(fs + 1, fe + 1) for (fs, fe) in raw_spans]
            labels_masked = apply_effective_region_mask(labels_pack, feature_spans_shifted, is_mlm=True, last_k=last_k)

            with torch.no_grad():
                outputs = model(input_ids=sequence_input, return_dict=True)

            logits = outputs["logits"].float() if "logits" in outputs else None  # (B,T,V)
            ce_labels = labels_masked[:, 0, :].long()
            cons_tgt = labels_masked[:, 1, :].float()

            # CE per-example (MLM)
            if logits is not None:
                ce_tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ce_labels.reshape(-1),
                    reduction="none",
                ).view(ce_labels.size())

                ce_mask = ce_labels.ne(-100).float()
                ce_per_ex = _masked_mean_per_row(ce_tok, ce_mask, dim=1)
                all_seq_predictions.extend(ce_per_ex.detach().cpu().tolist())
            else:
                all_seq_predictions.extend([float("nan")] * sequence_input.size(0))

            # CONS predictions if exposed
            if "scaling_logits" in outputs:
                cons_logits = outputs["scaling_logits"]
                cons_mean = cons_logits[..., 0].float() if cons_logits.ndim == 3 else cons_logits.float()  # (B,T)

                cons_mask = cons_tgt.ne(-100).float()
                mse_tok = (cons_mean - cons_tgt).pow(2)
                mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)
                all_predictions.extend(mse_per_ex.detach().cpu().tolist())

                cons_np = cons_mean.detach().cpu().numpy()[:, 1:]   # drop START
                cons_keep = cons_mask.detach().cpu().numpy()[:, 1:]
                for b in range(cons_np.shape[0]):
                    if effective_only:
                        idx = np.where(cons_keep[b] > 0.0)[0]
                        all_pred_scores.append(cons_np[b, idx].astype(np.float32))
                    else:
                        all_pred_scores.append(cons_np[b].astype(np.float32))
            else:
                all_predictions.extend([float("nan")] * sequence_input.size(0))
                for _ in range(sequence_input.size(0)):
                    all_pred_scores.append(np.array([], dtype=np.float32) if effective_only
                                           else np.full(sequence_input.size(1) - 1, np.nan, dtype=np.float32))
        else:
            raise ValueError(f"unknown model_type={model_type}")

    return all_predictions, all_true_scores, region_info, all_seq_predictions, all_true_seqs, all_pred_scores


# ---------------- single-window (effective-only) ----------------

def predict_single_window_effective(
    model,
    tokenizer,
    genome,
    bigwig_file,
    chrom,
    window_start,
    window_end,
    model_type,
    training_task,
    device,
    context_size,  # 1000
    last_k,  # 1024
):
    """
    Predict for ONE window [window_start, window_end).
    Context: [window_start, window_start + context_size)
    Predict: [window_start + context_size, window_end)
    """
    
    # Manually extract sequence and scores (bypass extract_context windowing)
    try:
        seq = str(genome[chrom][window_start:window_end].seq)
        
        with pyBigWig.open(bigwig_file) as bw:
            scores = np.array(bw.values(chrom, window_start, window_end), dtype=np.float32)
        
        if len(seq) != len(scores):
            raise ValueError(f"Length mismatch: seq={len(seq)}, scores={len(scores)}")
            
    except Exception as e:
        raise ValueError(f"Failed to extract {chrom}:{window_start}-{window_end}: {e}")
    
    region = {
        "chrom": chrom,
        "start": window_start,
        "end": window_end,
        "sequence": seq,  # Already extracted
        "scores": scores,  # Already extracted
        "feature_start_in_window": context_size,  # Prediction starts here
        "feature_end_in_window": window_end - window_start,  # Prediction ends here
        "category": "sliding_window",
    }
    
    # Now predict (sequence already extracted, no need for tokenization yet)
    (_, _, _, _, _, all_pred_scores) = predict_scores_batched(
        model=model,
        tokenizer=tokenizer,
        tokenized=False,  # Will tokenize the sequence
        regions=[region],
        batch_size=1,
        device=device,
        model_type=model_type,
        training_task=training_task,
        effective_only=True,
        last_k=last_k,  # This should be 1024 if you want 1024 predictions
    )
    
    if len(all_pred_scores) != 1:
        raise ValueError(f"prediction failed for {chrom}:{window_start}-{window_end} (got {len(all_pred_scores)})")

    means_eff = np.asarray(all_pred_scores[0], dtype=np.float32)

    eff_start = window_start + context_size
    eff_end = window_end
    target_len = eff_end - eff_start

    # if last_k < predict_size, means_eff is shorter by design; pad with nan so caller can slice safely
    if len(means_eff) < target_len:
        means_eff = np.pad(means_eff, (target_len - len(means_eff), 0), constant_values=np.nan)
    elif len(means_eff) > target_len:
        means_eff = means_eff[-target_len:]

    return eff_start, eff_end, means_eff


# ---------------- sliding window over a region ----------------

def predict_region_sliding_window(
    model,
    tokenizer,
    genome,
    bigwig_file,
    chrom,
    start,
    end,
    model_type,
    training_task,
    device,
    window_size=2048,
    context_size=1000,
    last_k=1000,
):
    """
    Predict phyloP scores for [start, end) using sliding windows.
    Only keep effective predictions per window over [window_start+context_size, window_end).
    """
    region_length = end - start
    predict_size = window_size - context_size
    if predict_size <= 0:
        raise ValueError(f"window_size ({window_size}) must be > context_size ({context_size})")

    chrom_len = len(genome[chrom])

    # how many strides of predict_size needed to cover [start,end)
    num_windows = int(np.ceil(region_length / predict_size))

    logging.info(
        f"region {region_length}bp requires {num_windows} windows "
        f"(window={window_size}bp, context={context_size}bp, predict={predict_size}bp per window)"
    )

    all_means = []
    all_positions = []

    for i in range(num_windows):
        # desired predicted chunk for this stride
        chunk_start = start + i * predict_size
        chunk_end = min(chunk_start + predict_size, end)

        # model window aligned so its effective region begins at chunk_start
        window_start = chunk_start - context_size
        window_end = window_start + window_size

        # clamp to chromosome bounds by shifting (prefer full window_size if possible)
        if window_start < 0:
            window_start = 0
            window_end = min(window_size, chrom_len)
        if window_end > chrom_len:
            window_end = chrom_len
            window_start = max(0, window_end - window_size)

        eff_start, eff_end, eff_means = predict_single_window_effective(
            model=model,
            tokenizer=tokenizer,
            genome=genome,
            bigwig_file=bigwig_file,
            chrom=chrom,
            window_start=window_start,
            window_end=window_end,
            model_type=model_type,
            training_task=training_task,
            device=device,
            context_size=context_size,
            last_k=last_k,
        )

        # eff region is [eff_start, eff_end) == [window_start+context_size, window_end)
        # take the overlap with [chunk_start, chunk_end)
        take_start = max(chunk_start, eff_start)
        take_end = min(chunk_end, eff_end)
        take_len = max(0, take_end - take_start)

        if take_len == 0:
            logging.warning(
                f"window {i+1}/{num_windows}: no overlap? "
                f"chunk [{chunk_start},{chunk_end}) eff [{eff_start},{eff_end})"
            )
            continue

        offset = take_start - eff_start
        chunk = eff_means[offset:offset + take_len]
        if len(chunk) < take_len:
            chunk = np.pad(chunk, (0, take_len - len(chunk)), constant_values=np.nan)

        logging.info(
            f"window {i+1}/{num_windows}: model sees [{window_start},{window_end}) "
            f"eff [{eff_start},{eff_end}) → keep [{take_start},{take_end})"
        )

        all_means.append(chunk)
        all_positions.extend(range(take_start, take_end))

    predicted_means = np.concatenate(all_means) if all_means else np.array([], dtype=np.float32)
    positions = np.array(all_positions, dtype=np.int64)

    if len(predicted_means) != (end - start):
        # align to exactly [start,end): fill gaps with nan if any
        full = np.full((end - start,), np.nan, dtype=np.float32)
        if len(positions) > 0:
            idx = positions - start
            valid = (idx >= 0) & (idx < len(full))
            full[idx[valid]] = predicted_means[valid]
        predicted_means = full
        positions = np.arange(start, end, dtype=np.int64)

    predicted_vars = np.ones_like(predicted_means, dtype=np.float32)  # placeholder
    logging.info(f"combined prediction: {len(predicted_means)} positions")
    return predicted_means, predicted_vars, positions


# ---------------- plotting ----------------

def plot_predictions(
    positions,
    predicted_means,
    predicted_vars,
    true_scores=None,
    output_path=None,
    title="Predicted phyloP Scores",
):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(positions, predicted_means, label="Predicted Mean", linewidth=1.5)

    if predicted_vars is not None:
        ax.fill_between(
            positions,
            predicted_means - np.sqrt(predicted_vars),
            predicted_means + np.sqrt(predicted_vars),
            alpha=0.3,
            label="±1 Std Dev",
        )

    if true_scores is not None:
        ax.scatter(
            positions,
            true_scores,
            s=2,
            alpha=0.6,
            label="True phyloP",
        )

        valid_mask = ~np.isnan(true_scores) & ~np.isnan(predicted_means)
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(true_scores[valid_mask], predicted_means[valid_mask])[0, 1]
            ax.text(
                0.02, 0.98,
                f"Pearson r = {corr:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("phyloP Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        logging.info(f"saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def save_predictions_bigwig(
    chrom,
    positions,
    scores,
    output_path,
    chrom_sizes_path,
):
    output_path = str(output_path)
    chrom_sizes_path = str(chrom_sizes_path)

    chrom_sizes = {}
    with open(chrom_sizes_path, "r") as f:
        for line in f:
            chrom_name, size = line.strip().split()
            chrom_sizes[chrom_name] = int(size)

    bw = pyBigWig.open(output_path, "w")
    bw.addHeader([(chrom, chrom_sizes[chrom])])

    starts = positions.astype(int)
    ends = starts + 1
    bw.addEntries(
        [chrom] * len(scores),
        starts.tolist(),
        ends=ends.tolist(),
        values=[float(x) if np.isfinite(x) else float("nan") for x in scores],
    )

    bw.close()
    logging.info(f"saved bigWig to {output_path}")


def save_predictions_bedgraph(
    chrom,
    positions,
    scores,
    output_path,
):
    output_path = str(output_path)
    with open(output_path, "w") as f:
        # Add track definition line
        f.write("track type=bedGraph name=\"Predicted phyloP\" "
                "description=\"Model predicted conservation scores\" "
                "visibility=full autoScale=on color=0,0,255 "
                "altColor=255,0,0 priority=20\n")
        
        for pos, score in zip(positions, scores):
            if not np.isfinite(score):
                continue
            f.write(f"{chrom}\t{pos}\t{pos+1}\t{score:.6f}\n")
    logging.info(f"saved bedGraph to {output_path}")

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict phyloP conservation scores using gamba/caduceus"
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to genome fasta",
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to phyloP bigwig (for true scores, optional)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/mica/gamba/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json",
        help="Model config JSON",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gamba", "caduceus"],
        default="gamba",
    )
    parser.add_argument(
        "--training_task",
        type=str,
        choices=["dual", "cons_only", "seq_only"],
        default="dual",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
        help="Checkpoint step",
    )
    parser.add_argument(
        "--chrom",
        type=str,
        required=True,
        help="Chromosome (e.g., chr16)",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start position (0-based)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End position (exclusive)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/phylop_predictions",
        help="Output directory",
    )
    parser.add_argument(
        "--chrom_sizes",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes",
        help="Chromosome sizes file (for bigWig output)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2048,
        help="Prediction window size (bp)",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=1000,
        help="Upstream context size (bp) for sliding windows",
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=1000,
        help="Within each window, keep only the last_k bases of the effective region (masking).",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plotting",
    )
    parser.add_argument(
        "--no_bigwig",
        action="store_true",
        help="Skip bigWig output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    if args.model_type == "gamba":
        checkpoint_dir = os.path.join(args.checkpoint_dir, "clean_dcps/CCP/")
    else:
        checkpoint_dir = args.checkpoint_dir

    logging.info(f"loading model from {checkpoint_dir}, step {args.last_step}")
    model, tokenizer = load_model(
        checkpoint_dir,
        args.config_fpath,
        last_step=args.last_step,
        device=device,
        training_task=args.training_task,
        model_type=args.model_type,
    )

    genome = Fasta(args.genome_fasta)

    logging.info(f"predicting {args.chrom}:{args.start}-{args.end}")
    predicted_means, predicted_vars, positions = predict_region_sliding_window(
        model=model,
        tokenizer=tokenizer,
        genome=genome,
        bigwig_file=args.bigwig_file,
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        model_type=args.model_type,
        training_task=args.training_task,
        device=device,
        window_size=args.window_size,
        context_size=args.context_size,
        last_k=args.last_k,
    )

    # true scores for comparison (optional)
    true_scores = None
    try:
        bw = pyBigWig.open(args.bigwig_file)
        true_scores = np.array(bw.values(args.chrom, args.start, args.end), dtype=np.float32)
        bw.close()

        # align lengths
        L = min(len(true_scores), len(predicted_means))
        true_scores = true_scores[:L]
        predicted_means = predicted_means[:L]
        predicted_vars = predicted_vars[:L]
        positions = positions[:L]

        valid_mask = ~np.isnan(true_scores) & ~np.isnan(predicted_means)
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(true_scores[valid_mask], predicted_means[valid_mask])[0, 1]
            logging.info(f"Pearson correlation: {corr:.4f}")
    except Exception as e:
        logging.warning(f"could not load true scores: {e}")
        true_scores = None

    region_str = f"{args.chrom}_{args.start}_{args.end}"

    if not args.no_plot:
        plot_path = output_dir / f"{region_str}_predictions.png"
        plot_predictions(
            positions,
            predicted_means,
            predicted_vars,
            true_scores=true_scores,
            output_path=plot_path,
            title=f"phyloP Predictions: {args.chrom}:{args.start:,}-{args.end:,}",
        )

    if not args.no_bigwig:
        bw_path = output_dir / f"{region_str}_predictions.bw"
        save_predictions_bigwig(
            args.chrom,
            positions,
            predicted_means,
            bw_path,
            args.chrom_sizes,
        )

    bg_path = output_dir / f"{region_str}_predictions.bedGraph"
    save_predictions_bedgraph(args.chrom, positions, predicted_means, bg_path)

    summary = {
        "chrom": args.chrom,
        "start": int(args.start),
        "end": int(args.end),
        "length": int(args.end - args.start),
        "model_type": args.model_type,
        "training_task": args.training_task,
        "last_step": int(args.last_step),
        "window_size": int(args.window_size),
        "context_size": int(args.context_size),
        "last_k": int(args.last_k),
        "num_positions": int(len(predicted_means)),
        "mean_predicted": float(np.nanmean(predicted_means)),
        "std_predicted": float(np.nanstd(predicted_means)),
    }

    if true_scores is not None:
        valid_mask = ~np.isnan(true_scores) & ~np.isnan(predicted_means)
        if np.sum(valid_mask) > 1:
            summary["pearson_r"] = float(
                np.corrcoef(true_scores[valid_mask], predicted_means[valid_mask])[0, 1]
            )

    summary_path = output_dir / f"{region_str}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"saved summary to {summary_path}")

    logging.info("done!")


if __name__ == "__main__":
    main()

# Example usage:
#
# short region (single coverage via sliding logic):
# python src/evaluation/bigwig_predict.py \
#   --chrom chr16 --start 23683829 --end 23685877 \
#   --model_type gamba --training_task dual --last_step 44000
#
# long region (10kb, sliding windows):
# python src/evaluation/bigwig_predict.py \
#   --chrom chr16 --start 23680000 --end 23690000 \
#   --model_type gamba --training_task cons_only --last_step 44000 \
#   --window_size 2048 --context_size 1000 --last_k 1000


# python src/evaluation/bigwig_predict.py \
#   --chrom chr16 --start 4734537 --end 4749396
#   --model_type gamba --training_task cons_only --last_step 44000 \
#   --window_size 2048 --context_size 1000 --last_k 1000

# python src/evaluation/bigwig_predict.py \
#   --chrom chr6 --start 31160371 --end 31174649
#   --model_type gamba --training_task cons_only --last_step 44000 \
#   --window_size 2048 --context_size 1000 --last_k 1000

