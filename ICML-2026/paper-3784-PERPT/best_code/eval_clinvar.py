#!/usr/bin/env python3
"""
Self-contained ClinVar evaluation for Bi-Gamba MLM+MEM (4M) model.
Reproduces AUROC (Log-likelihood) and AUROC (Predicted conservation) from Table A9.

Usage:
  python eval_clinvar.py \
    --model_path /models/bigamba-dual-step44000 \
    --genome_fasta /datasets/hg38.ml.fa \
    --bigwig /datasets/241-mammalian-2020v2.bigWig \
    --output_dir /repo/eval_output
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/my_caduceus')
sys.path.insert(0, '/models/bigamba-dual-step44000')

from evodiff.utils import Tokenizer
from gamba.constants import DNA_ALPHABET_PLUS
from transformers import AutoModel

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
VALID_BASES = set(COMP.keys())


def revcomp(seq: str) -> str:
    tbl = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(tbl)[::-1]


def normalize_chrom(v) -> str:
    s = str(v).strip()
    return s if s.startswith("chr") else "chr" + s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/models/bigamba-dual-step44000")
    parser.add_argument("--genome_fasta", type=str, default="/datasets/hg38.ml.fa")
    parser.add_argument("--bigwig", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/repo/eval_output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--strand_mode", type=str, default="mean",
                        choices=["mean", "max", "min", "fwd_only", "rev_only"],
                        help="Strand combination strategy for LLR (default: mean)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for log_softmax in LLR (default: 1.0)")
    parser.add_argument("--llr_radius", type=int, default=0,
                        help="Radius (in bp) around variant for multi-position LLR aggregation (default: 0 = single position)")
    parser.add_argument("--llr_aggregation", type=str, default="max",
                        choices=["max", "mean"],
                        help="Aggregation method for multi-position LLR (default: max)")
    parser.add_argument("--cons_radius", type=int, default=0,
                        help="Radius for multi-position conservation aggregation (default: 0 = single position)")
    parser.add_argument("--cons_aggregation", type=str, default="mean",
                        choices=["max", "mean"],
                        help="Aggregation method for multi-position conservation (default: mean)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determinism: fixed random seed for reproducibility (CODE-04)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Note: cudnn.deterministic not set — conflicts with mamba-ssm Triton kernels
    print("Determinism: seed=42")

    # Load tokenizer
    tokenizer = Tokenizer(DNA_ALPHABET_PLUS)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    ).eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Load genome
    print(f"Loading genome from {args.genome_fasta}...")
    genome = Fasta(args.genome_fasta)

    # Load bigwig (optional)
    bw = None
    if args.bigwig and os.path.exists(args.bigwig):
        print(f"Loading PhyloP from {args.bigwig}...")
        bw = pyBigWig.open(args.bigwig)

    # Load ClinVar dataset
    print("Loading ClinVar dataset...")
    ds = load_dataset("songlab/clinvar_vs_benign", split="test")
    df = ds.to_pandas()
    print(f"Dataset size: {len(df)}")

    # Filter to missense variants
    if "consequence" in df.columns:
        before = len(df)
        df = df[df["consequence"].str.contains("missense", case=False, na=False)].copy()
        print(f"After missense filter: {len(df)} (from {before})")

    # Normalize labels
    y_series = df["label"].map(lambda v: 1 if str(v).strip().lower().startswith("path") else 0)

    # Drop rows with bad positions
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype(int)

    if args.max_samples > 0:
        df = df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples")

    print(f"Evaluating {len(df)} variants with batch_size={args.batch_size}...")

    # Pre-process: collect valid variants
    valid_rows = []
    valid_chromosomes = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

    for idx in tqdm(range(len(df)), desc="Pre-processing variants"):
        row = df.iloc[idx]
        chrom = normalize_chrom(row["chrom"])
        if chrom not in valid_chromosomes:
            continue
        try:
            pos0 = int(row["pos"]) - 1
        except (ValueError, TypeError):
            continue
        ref = str(row["ref"]).upper()
        alt = str(row["alt"]).upper()
        if len(ref) != 1 or len(alt) != 1 or ref not in VALID_BASES or alt not in VALID_BASES:
            continue

        # Extract sequence window
        target_pos = args.window_size // 2
        start0 = pos0 - target_pos
        end0 = start0 + args.window_size
        try:
            seq = genome[chrom][start0:end0].seq.upper()
        except Exception:
            continue
        if len(seq) != args.window_size:
            continue
        if seq[target_pos] != ref:
            continue
        if any(b not in "ACGT" for b in seq):
            continue

        valid_rows.append({
            "idx": idx,
            "chrom": chrom,
            "pos0": pos0,
            "start0": start0,
            "target_pos": target_pos,
            "ref": ref,
            "alt": alt,
            "label": int(y_series.iloc[idx]),
            "seq": seq,
            "seq_rev": revcomp(seq),
        })

    # Sort by chromosome and position for deterministic ordering (CODE-04)
    valid_rows.sort(key=lambda x: (x["chrom"], x["start0"]))
    print(f"Valid variants: {len(valid_rows)} (from {len(df)} total)")

    # Batch evaluation
    all_ref_ll = []
    all_alt_ll = []
    all_cons_pred = []
    all_phylop = []
    all_labels = []

    n_batches = (len(valid_rows) + args.batch_size - 1) // args.batch_size

    for batch_start in tqdm(range(0, len(valid_rows), args.batch_size), desc="Evaluating batches", total=n_batches):
        batch_end = min(batch_start + args.batch_size, len(valid_rows))
        batch = valid_rows[batch_start:batch_end]

        # Prepare batched inputs
        max_len = args.window_size
        B = len(batch)
        fwd_tokens = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
        rev_tokens = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)

        for j, item in enumerate(batch):
            fwd_tokens[j, :] = torch.tensor(tokenizer.tokenizeMSA(item["seq"]), dtype=torch.long)
            rev_tokens[j, :] = torch.tensor(tokenizer.tokenizeMSA(item["seq_rev"]), dtype=torch.long)

        # Forward pass - forward strand
        with torch.no_grad():
            out_fwd = model(input_ids=fwd_tokens.to(device))
        logits_fwd = out_fwd["logits"].float()
        cons_fwd = out_fwd["scaling_logits"].float()

        # Forward pass - reverse strand
        with torch.no_grad():
            out_rev = model(input_ids=rev_tokens.to(device))
        logits_rev = out_rev["logits"].float()
        cons_rev = out_rev["scaling_logits"].float()

        # Extract scores for each variant
        for j, item in enumerate(batch):
            tpos = item["target_pos"]
            tpos_rev = args.window_size - 1 - tpos
            ref = item["ref"]
            alt = item["alt"]

            ref_tok_fwd = tokenizer.tokenizeMSA(ref)[0]
            alt_tok_fwd = tokenizer.tokenizeMSA(alt)[0]
            ref_rc = COMP[ref]
            alt_rc = COMP[alt]
            ref_tok_rev = tokenizer.tokenizeMSA(ref_rc)[0]
            alt_tok_rev = tokenizer.tokenizeMSA(alt_rc)[0]

            # LLR: temperature-scaled log probabilities with configurable strand combination (ALGO-07)
            # Multi-position aggregation around variant (Iter-3)
            T = args.temperature
            radius = args.llr_radius
            agg = args.llr_aggregation

            # Collect LLR values across positions [tpos-radius, tpos+radius]
            llr_values = []
            for offset in range(-radius, radius + 1):
                p_fwd = tpos + offset
                p_rev = args.window_size - 1 - p_fwd
                if p_fwd < 0 or p_fwd >= args.window_size:
                    continue

                logp_ref_f = (F.log_softmax(logits_fwd[j, p_fwd] / T, dim=-1)[ref_tok_fwd]).item()
                logp_alt_f = (F.log_softmax(logits_fwd[j, p_fwd] / T, dim=-1)[alt_tok_fwd]).item()
                logp_ref_r = (F.log_softmax(logits_rev[j, p_rev] / T, dim=-1)[ref_tok_rev]).item()
                logp_alt_r = (F.log_softmax(logits_rev[j, p_rev] / T, dim=-1)[alt_tok_rev]).item()

                # Strand combination for LLR at this position
                sm = args.strand_mode
                if sm == "mean":
                    ref_ll_p = 0.5 * (logp_ref_f + logp_ref_r)
                    alt_ll_p = 0.5 * (logp_alt_f + logp_alt_r)
                elif sm == "max":
                    ref_ll_p = max(logp_ref_f, logp_ref_r)
                    alt_ll_p = max(logp_alt_f, logp_alt_r)
                elif sm == "min":
                    ref_ll_p = min(logp_ref_f, logp_ref_r)
                    alt_ll_p = min(logp_alt_f, logp_alt_r)
                elif sm == "fwd_only":
                    ref_ll_p = logp_ref_f
                    alt_ll_p = logp_alt_f
                elif sm == "rev_only":
                    ref_ll_p = logp_ref_r
                    alt_ll_p = logp_alt_r
                else:
                    ref_ll_p = 0.5 * (logp_ref_f + logp_ref_r)
                    alt_ll_p = 0.5 * (logp_alt_f + logp_alt_r)

                llr_values.append(alt_ll_p - ref_ll_p)

            # Aggregate LLR across positions
            if agg == "max":
                llr_pos = max(llr_values) if llr_values else 0.0
            else:  # mean
                llr_pos = sum(llr_values) / len(llr_values) if llr_values else 0.0
            ref_ll = 0.0  # placeholder
            alt_ll = llr_pos  # store LLR directly

            all_ref_ll.append(ref_ll)
            all_alt_ll.append(alt_ll)

            # Conservation score: multi-position aggregation (Iter-6)
            c_radius = args.cons_radius
            c_agg = args.cons_aggregation
            cons_vals = []
            for coffset in range(-c_radius, c_radius + 1):
                cp_fwd = tpos + coffset
                cp_rev = args.window_size - 1 - cp_fwd
                if cp_fwd < 0 or cp_fwd >= args.window_size:
                    continue
                cf_p = cons_fwd[j, cp_fwd, 0].item()
                cr_p = cons_rev[j, cp_rev, 0].item()
                cons_vals.append(0.5 * (cf_p + cr_p))

            if c_agg == "max":
                all_cons_pred.append(max(cons_vals) if cons_vals else 0.0)
            else:  # mean
                all_cons_pred.append(sum(cons_vals) / len(cons_vals) if cons_vals else 0.0)

            all_labels.append(item["label"])

            # PhyloP baseline (if available)
            if bw is not None:
                scores = np.zeros(args.window_size, dtype=np.float32)
                intervals = bw.intervals(item["chrom"], item["start0"], item["start0"] + args.window_size)
                if intervals is not None:
                    for s, e, val in intervals:
                        a = max(0, s - item["start0"])
                        b = min(args.window_size, e - item["start0"])
                        if a < b:
                            scores[a:b] = val
                all_phylop.append(float(np.round(scores[tpos], 2)))

    # Compute metrics
    ref_ll = np.array(all_ref_ll, dtype=np.float32)
    alt_ll = np.array(all_alt_ll, dtype=np.float32)
    cons_pred = np.array(all_cons_pred, dtype=np.float32)
    y_arr = np.array(all_labels, dtype=np.int64)

    n_pos = (y_arr == 1).sum()
    n_neg = (y_arr == 0).sum()
    print(f"\nResults:")
    print(f"  N pathogenic: {n_pos}")
    print(f"  N benign: {n_neg}")
    print(f"  Total: {len(y_arr)}")

    # LLR scoring
    # Note: with multi-position aggregation, alt_ll already contains the LLR
    # For single-position (radius=0), ref_ll/alt_ll are log-probs, llr = alt_ll - ref_ll
    if args.llr_radius > 0:
        llr = np.array(all_alt_ll, dtype=np.float32)  # alt_ll already stores LLR
    else:
        llr = np.array(all_alt_ll, dtype=np.float32) - np.array(all_ref_ll, dtype=np.float32)
    auroc_llr = roc_auc_score(y_arr, llr)
    print(f"  AUROC (Log-likelihood ratio): {auroc_llr:.4f}")

    # Conservation scoring
    valid_cons = np.isfinite(cons_pred)
    if valid_cons.sum() > 0 and len(np.unique(y_arr[valid_cons])) >= 2:
        auroc_cons = roc_auc_score(y_arr[valid_cons], cons_pred[valid_cons])
        print(f"  AUROC (Predicted conservation): {auroc_cons:.4f}")
    else:
        auroc_cons = float("nan")
        print(f"  AUROC (Predicted conservation): N/A")

    # PhyloP baseline
    auroc_phylop = float("nan")
    if len(all_phylop) > 0:
        phylop_arr = np.array(all_phylop, dtype=np.float32)
        valid_phylop = np.isfinite(phylop_arr)
        if valid_phylop.sum() > 0 and len(np.unique(y_arr[valid_phylop])) >= 2:
            auroc_phylop = roc_auc_score(y_arr[valid_phylop], phylop_arr[valid_phylop])
            print(f"  AUROC (PhyloP baseline): {auroc_phylop:.4f}")

    # Save results
    results = {
        "model": "Bi-Gamba MLM+MEM",
        "model_scale": "4M",
        "benchmark": "ClinVar",
        "benchmark_type": "missense pathogenic vs benign",
        "n_pathogenic": int(n_pos),
        "n_benign": int(n_neg),
        "scoring_method_llr": "log_likelihood_ratio",
        "strand_averaging": "forward+reverse",
        "evaluation_metric": "AUROC",
        "context_window": args.window_size,
        "auroc_log_likelihood": float(auroc_llr),
        "auroc_predicted_conservation": float(auroc_cons),
        "auroc_phylop_baseline": float(auroc_phylop),
        "n_processed": len(y_arr),
    }

    output_path = os.path.join(args.output_dir, "clinvar_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save raw scores
    np.savez(
        os.path.join(args.output_dir, "clinvar_scores.npz"),
        ref_ll=ref_ll,
        alt_ll=alt_ll,
        cons_pred=cons_pred,
        y=y_arr,
    )
    print(f"Raw scores saved to {args.output_dir}/clinvar_scores.npz")


if __name__ == "__main__":
    main()
