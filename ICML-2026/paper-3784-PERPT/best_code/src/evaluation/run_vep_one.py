#!/usr/bin/env python3
"""
run_vep_one.py

single-job VEP evaluation for one (model_type, training_task, task_name).

design goals:
- no temp parquet
- no hacks
- no duplicated logic (shared preprocessing + shared scoring)
- --save_npz (default true) to persist per-variant scores for re-bootstrapping SEs
- report a phyloP baseline AUROC/AUPRC at the variant position
- debug why rows are dropped (especially OMIM): counters + small examples per drop reason
  written to: <output_dir>/debug_drops.json (when --debug)

tasks / hf datasets:
A: songlab/clinvar_vs_benign (AUROC; label: "Pathogenic"/"Benign")
    evo2: songlab/clinvar_vs_benign/predictions/Evo2_7B.parquet
B: songlab/cosmic (AUPRC; label bool/int)
    evo2: songlab/cosmic/predictions/Evo2_7B.parquet
C: songlab/ukb_finemapped_coding (AUPRC; label bool/int)
    evo2: songlab/ukb_finemapped_coding/predictions/Evo2_7B.parquet
E: songlab/omim_traitgym (AUPRC; label bool/int)
    evo2: songlab/omim_traitgym/predictions/Evo2_7B.parquet
F: songlab/gnomad_balanced (AUROC; label bool/int)
    evo2: not applicable
G: songlab/ukb_finemapped_nc_traitgym (AUPRC; label bool/int)
    evo2: songlab/ukb_finemapped_nc_traitgym/predictions/Evo2_7B.parquet
H (derived): promoter subset from songlab/ukb_finemapped_nc_traitgym (AUPRC; label bool/int)
    evo2: not applicable
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- project paths ----------------
import sys

sys.path.append("../gamba")
sys.path.append("/home/mica/scratch/gamba/")
sys.path.append("/home/mica/scratch/gamba/src/")

from src.evaluation.utils.specific_helpers import load_model  # your existing loader
from gamba.collators import gLMCollator, gLMMLMCollator


# ---------------- task specs ----------------

@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    metric: str  # "AUROC" or "AUPRC"
    derived: bool = False


TASKS: Dict[str, TaskSpec] = {
    "A_clinvar_pathogenic_vs_benign_missense": TaskSpec(
        name="A_clinvar_pathogenic_vs_benign_missense",
        dataset="songlab/clinvar_vs_benign",
        metric="AUROC",
    ),
    "B_cosmic_frequent_in_cancer_vs_common_missense": TaskSpec(
        name="B_cosmic_frequent_in_cancer_vs_common_missense",
        dataset="songlab/cosmic",
        metric="AUPRC",
    ),
    "C_gwas_finemapped_missense_causal_vs_matched": TaskSpec(
        name="C_gwas_finemapped_missense_causal_vs_matched",
        dataset="songlab/ukb_finemapped_coding",
        metric="AUPRC",
    ),
    "E_omim_noncoding_pathogenic_vs_common": TaskSpec(
        name="E_omim_noncoding_pathogenic_vs_common",
        dataset="songlab/omim_traitgym",
        metric="AUPRC",
    ),
    "F_gnomad_rare_pathogenic_vs_common_missense": TaskSpec(
        name="F_gnomad_rare_pathogenic_vs_common_missense",
        dataset="songlab/gnomad_balanced",
        metric="AUROC",
    ),
    "G_gwas_finemapped_noncoding_causal_vs_matched": TaskSpec(
        name="G_gwas_finemapped_noncoding_causal_vs_matched",
        dataset="songlab/ukb_finemapped_nc_traitgym",
        metric="AUPRC",
    ),
    "H_promoter_variants_derived_from_ukb_nc": TaskSpec(
        name="H_promoter_variants_derived_from_ukb_nc",
        dataset="songlab/ukb_finemapped_nc_traitgym",
        metric="AUPRC",
        derived=True,
    ),
}

# evo2 parquet baselines (task->hf parquet path). H is derived, F has none.
EVO2_PARQUET: Dict[str, str] = {
    "A_clinvar_pathogenic_vs_benign_missense": "hf://datasets/songlab/clinvar_vs_benign/predictions/Evo2_7B.parquet",
    "B_cosmic_frequent_in_cancer_vs_common_missense": "hf://datasets/songlab/cosmic/predictions/Evo2_7B.parquet",
    "C_gwas_finemapped_missense_causal_vs_matched": "hf://datasets/songlab/ukb_finemapped_coding/predictions/Evo2_7B.parquet",
    "E_omim_noncoding_pathogenic_vs_common": "hf://datasets/songlab/omim_traitgym/predictions/Evo2_7B.parquet",
    "G_gwas_finemapped_noncoding_causal_vs_matched": "hf://datasets/songlab/ukb_finemapped_nc_traitgym/predictions/Evo2_7B.parquet",
}


# ---------------- shared utilities ----------------

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
_VALID_BASES = set(_COMP.keys())


def load_evo2_scores_row_aligned(evo2_parquet_path: str, score_col: str = "score") -> np.ndarray:
    """
    Evo2 parquet sometimes only contains a score column and is assumed to be row-aligned
    to the HF dataset split order.

    Returns: scores_all shape (N,)
    """
    evo = pd.read_parquet(evo2_parquet_path)
    if score_col not in evo.columns:
        raise ValueError(f"evo2 parquet missing '{score_col}'. have={list(evo.columns)}")
    s = pd.to_numeric(evo[score_col], errors="coerce").to_numpy(dtype=np.float32)
    return s


def maybe_flip_scores_for_metric(y: np.ndarray, s: np.ndarray, metric: str) -> tuple[np.ndarray, bool]:
    """
    Some baselines have inverted direction. Try both and keep the better one.
    Returns (scores_used, flipped)
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    # must be comparable on the same finite mask
    m = np.isfinite(s)
    y2, s2 = y[m], s[m]
    if len(y2) == 0 or len(np.unique(y2)) < 2:
        return s, False

    fn = metric_fn(metric)
    try:
        a = float(fn(y2, s2))
        b = float(fn(y2, -s2))
    except Exception:
        return s, False

    if np.isfinite(b) and (not np.isfinite(a) or b > a):
        return -s, True
    return s, False



def metric_fn(metric: str):
    if metric == "AUROC":
        return roc_auc_score
    if metric == "AUPRC":
        return average_precision_score
    raise ValueError(f"unknown metric: {metric}")


def bootstrap_se(
    y: np.ndarray,
    scores: np.ndarray,
    metric: str,
    n_boot: int = 200,
    seed: int = 1,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return float("nan")

    fn = metric_fn(metric)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            vals.append(float(fn(yb, sb)))
        except Exception:
            continue

    if len(vals) < 5:
        return float("nan")
    return float(np.std(vals, ddof=1))


def safe_metric(y, s, metric, *, n_boot: int, seed: int):
    """
    Returns (score, se, n_used) with NaN-safe filtering + 2-class guard.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    m = np.isfinite(s)
    y = y[m]
    s = s[m]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), int(len(y))

    fn = metric_fn(metric)
    score = float(fn(y, s))
    se = float(bootstrap_se(y, s, metric, n_boot=n_boot, seed=seed))
    return score, se, int(len(y))


def compute_llr(ref_ll: np.ndarray, alt_ll: np.ndarray) -> np.ndarray:
    return np.asarray(alt_ll, dtype=float) - np.asarray(ref_ll, dtype=float)


def revcomp_str(seq: str) -> str:
    tbl = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(tbl)[::-1]


def normalize_chrom(v) -> str:
    s = str(v).strip()
    if s.startswith("chr"):
        return s
    return "chr" + s


def get_sequence_window(
    genome: Fasta, chromosome: str, position0: int, window_size: int = 2048
) -> Tuple[str, int, int]:
    """
    mutation centered in window.
    position0: 0-based genomic coordinate.
    returns (sequence, start0, target_pos_in_window)
    """
    target_pos = window_size // 2
    start0 = position0 - target_pos
    end0 = start0 + window_size
    seq = genome[chromosome][start0:end0].seq.upper()
    if len(seq) != window_size:
        raise ValueError("could not extract full window")
    return seq, start0, target_pos


def extract_phylop_window(
    bw: pyBigWig.pyBigWig, chrom: str, start0: int, window: int = 2048
) -> np.ndarray:
    scores = np.zeros(window, dtype=np.float32)
    intervals = bw.intervals(chrom, start0, start0 + window)
    if intervals is not None:
        for s, e, val in intervals:
            a = max(0, s - start0)
            b = min(window, e - start0)
            if a < b:
                scores[a:b] = val
    return np.round(scores, 2)


def load_task_df(spec: TaskSpec) -> pd.DataFrame:
    """
    robust HF loader (avoids pandas hf://parquet + pyarrow weirdness on clusters)
    """
    from datasets import load_dataset

    ds = load_dataset(spec.dataset, split="test")
    df = ds.to_pandas()

    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str)

    if "pos" in df.columns:
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")

    if "ref" in df.columns:
        df["ref"] = df["ref"].astype(str)
    if "alt" in df.columns:
        df["alt"] = df["alt"].astype(str)

    return df


def normalize_labels(df: pd.DataFrame) -> np.ndarray:
    if "label" not in df.columns:
        raise ValueError("missing label column")

    s = df["label"]
    if s.dtype == object:
        y = s.map(lambda v: 1 if str(v).strip().lower().startswith("path") else 0)
        return y.to_numpy(dtype=int)

    return s.astype(int).to_numpy(dtype=int)


# ---------------- promoter derivation (H) ----------------

def derive_promoter_from_ukb_nc(
    df: pd.DataFrame,
    y: np.ndarray,
    tss_bp: int = 1500,
    k_neg: int = 10,
    use_match_group: bool = True,
    seed: int = 1,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)

    promoter_classes = {"PLS", "PLS_flank", "DNase-H3K4me3", "DNase-H3K4me3_flank"}

    if "consequence" not in df.columns:
        raise ValueError("missing consequence column for promoter derivation")
    if "tss_dist" not in df.columns:
        raise ValueError("missing tss_dist column for promoter derivation")

    m = df["consequence"].isin(promoter_classes) & (df["tss_dist"].abs() <= tss_bp)
    df = df.loc[m].reset_index(drop=True)
    y = y[m.to_numpy()]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return df.iloc[0:0].copy(), y[0:0]

    keep = set()

    if use_match_group and ("match_group" in df.columns):
        neg_by_group = {}
        for i in neg_idx:
            g = df.at[i, "match_group"]
            neg_by_group.setdefault(g, []).append(i)

        for i in pos_idx:
            keep.add(int(i))
            g = df.at[i, "match_group"]
            pool = neg_by_group.get(g, [])
            if len(pool) == 0:
                continue
            take = min(k_neg, len(pool))
            sampled = rng.choice(pool, size=take, replace=False)
            for j in sampled:
                keep.add(int(j))
    else:
        keep.update(pos_idx.tolist())
        take = min(len(neg_idx), len(pos_idx) * k_neg)
        sampled = rng.choice(neg_idx, size=take, replace=False)
        keep.update(sampled.tolist())

    keep = np.array(sorted(keep), dtype=int)
    df2 = df.iloc[keep].reset_index(drop=True)
    y2 = y[keep]
    return df2, y2


# ---------------- shared batch builder (WITH DEBUG + kept indices) ----------------

def _build_batch_inputs(
    genome: Fasta,
    bw: pyBigWig.pyBigWig,
    df: pd.DataFrame,
    y: np.ndarray,
    tokenizer,
    window_size: int,
    *,
    debug: bool = False,
    debug_max: int = 5,
):
    """
    Returns:
      fwd_inputs, rev_inputs, fwd_info, rev_info,
      y_out, phylop_at_var, kept_local_idx,
      meta, debug_examples
    """
    valid_chromosomes = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM", "chrMT"}

    try:
        fasta_chroms = set(genome.keys())
    except Exception:
        fasta_chroms = set()
    try:
        bw_chroms = set(bw.chroms().keys())
    except Exception:
        bw_chroms = set()

    fwd_inputs, rev_inputs = [], []
    fwd_info, rev_info = [], []
    y_out = []
    phylop_at_var = []
    kept_local: List[int] = []

    total = 0
    reasons = {
        "bad_chrom": 0,
        "missing_fasta_chrom": 0,
        "missing_bw_chrom": 0,
        "bad_pos": 0,
        "bad_allele": 0,
        "short_window": 0,
        "non_acgt_window": 0,
        "ref_mismatch": 0,
    }

    debug_examples = {k: [] for k in reasons.keys()} if debug else None

    def _maybe_add(reason: str, payload: dict):
        if not debug:
            return
        lst = debug_examples[reason]
        if len(lst) < debug_max:
            lst.append(payload)

    for i, row in df.iterrows():
        total += 1

        chrom_raw = row.get("chrom", "")
        chrom = normalize_chrom(chrom_raw)

        if chrom not in valid_chromosomes:
            reasons["bad_chrom"] += 1
            _maybe_add("bad_chrom", {"chrom_raw": chrom_raw, "chrom_norm": chrom})
            continue

        if fasta_chroms and chrom not in fasta_chroms:
            reasons["missing_fasta_chrom"] += 1
            _maybe_add("missing_fasta_chrom", {"chrom_norm": chrom, "chrom_raw": chrom_raw})
            continue

        if bw_chroms and chrom not in bw_chroms:
            reasons["missing_bw_chrom"] += 1
            _maybe_add("missing_bw_chrom", {"chrom_norm": chrom, "chrom_raw": chrom_raw})
            continue

        try:
            pos0 = int(row["pos"]) - 1
        except Exception:
            reasons["bad_pos"] += 1
            _maybe_add("bad_pos", {"chrom_norm": chrom, "pos": row.get("pos", None)})
            continue

        ref = str(row.get("ref", "")).upper()
        alt = str(row.get("alt", "")).upper()
        if len(ref) != 1 or len(alt) != 1 or (ref not in _VALID_BASES) or (alt not in _VALID_BASES):
            reasons["bad_allele"] += 1
            _maybe_add("bad_allele", {"chrom_norm": chrom, "pos0": pos0, "ref": ref, "alt": alt})
            continue

        try:
            seq, start0, tpos = get_sequence_window(genome, chrom, pos0, window_size=window_size)
        except Exception as e:
            reasons["short_window"] += 1
            _maybe_add(
                "short_window",
                {"chrom_norm": chrom, "pos0": pos0, "ref": ref, "alt": alt, "err": str(e)},
            )
            continue

        if any(b not in "ACGT" for b in seq):
            reasons["non_acgt_window"] += 1
            bad_chars = sorted({b for b in seq if b not in "ACGT"})
            _maybe_add(
                "non_acgt_window",
                {
                    "chrom_norm": chrom,
                    "pos0": pos0,
                    "ref": ref,
                    "alt": alt,
                    "bad_chars": bad_chars[:10],
                    "seq_center_21": seq[tpos - 10 : tpos + 11] if 10 <= tpos < len(seq) - 10 else None,
                },
            )
            continue

        ref_base = seq[tpos]
        if ref_base != ref:
            reasons["ref_mismatch"] += 1
            _maybe_add(
                "ref_mismatch",
                {
                    "chrom_norm": chrom,
                    "pos0": pos0,
                    "ref": ref,
                    "alt": alt,
                    "fasta_base": ref_base,
                    "start0": start0,
                    "tpos": tpos,
                    "seq_center_21": seq[tpos - 10 : tpos + 11],
                },
            )
            continue

        scores_fwd = extract_phylop_window(bw, chrom, start0, window=window_size)
        seq_rev = revcomp_str(seq)
        scores_rev = scores_fwd[::-1].copy()

        tok_fwd = tokenizer.tokenizeMSA(seq)
        tok_rev = tokenizer.tokenizeMSA(seq_rev)

        ref_rc = _COMP[ref]
        alt_rc = _COMP[alt]

        ref_tok_fwd = tokenizer.tokenizeMSA(ref)[0]
        alt_tok_fwd = tokenizer.tokenizeMSA(alt)[0]
        ref_tok_rev = tokenizer.tokenizeMSA(ref_rc)[0]
        alt_tok_rev = tokenizer.tokenizeMSA(alt_rc)[0]

        fwd_inputs.append((tok_fwd, scores_fwd))
        rev_inputs.append((tok_rev, scores_rev))
        fwd_info.append((ref_tok_fwd, alt_tok_fwd, tpos))
        rev_info.append((ref_tok_rev, alt_tok_rev, window_size - 1 - tpos))

        y_out.append(int(y[i]))
        phylop_at_var.append(float(scores_fwd[tpos]))
        kept_local.append(int(i))

    meta = dict(
        n_total=int(total),
        n_used=int(len(y_out)),
        **{k: int(v) for k, v in reasons.items()},
    )

    return (
        fwd_inputs,
        rev_inputs,
        fwd_info,
        rev_info,
        np.asarray(y_out, dtype=np.int64),
        np.asarray(phylop_at_var, dtype=np.float32),
        np.asarray(kept_local, dtype=np.int64),
        meta,
        debug_examples,
    )


def _merge_debug_examples(dst: dict, src: Optional[dict], debug_max: int):
    if src is None:
        return
    for k, v in src.items():
        if k not in dst:
            dst[k] = []
        space = max(0, debug_max - len(dst[k]))
        if space <= 0:
            continue
        dst[k].extend(v[:space])


# ---------------- model-specific inference ----------------

def process_variants_gamba(
    genome: Fasta,
    bw: pyBigWig.pyBigWig,
    model,
    collator,
    tokenizer,
    device,
    df: pd.DataFrame,
    y: np.ndarray,
    batch_size: int = 64,
    window_size: int = 2048,
    *,
    debug: bool = False,
    debug_max: int = 5,
):
    """
    Returns:
      ref_ll, alt_ll, cons_pred, y_out, phylop_scores, kept_global_idx, meta, debug_examples
    """
    ref_ll_out, alt_ll_out = [], []
    cons_out = []
    y_out_all = []
    phylop_all = []
    kept_global: List[int] = []

    meta_total = {
        "n_total": int(len(df)),
        "n_used": 0,
        "bad_chrom": 0,
        "missing_fasta_chrom": 0,
        "missing_bw_chrom": 0,
        "bad_pos": 0,
        "bad_allele": 0,
        "short_window": 0,
        "non_acgt_window": 0,
        "ref_mismatch": 0,
    }
    debug_examples_total = {k: [] for k in meta_total if k not in ("n_total", "n_used")} if debug else None

    for start in tqdm(range(0, len(df), batch_size)):
        end = min(start + batch_size, len(df))
        df_b = df.iloc[start:end].reset_index(drop=True)
        y_b = y[start:end]

        (
            fwd_inputs,
            rev_inputs,
            fwd_info,
            rev_info,
            y_out,
            phylop_b,
            kept_local,
            meta_b,
            dbg_b,
        ) = _build_batch_inputs(
            genome, bw, df_b, y_b, tokenizer, window_size, debug=debug, debug_max=debug_max
        )

        for k in meta_total:
            if k in meta_b and k != "n_total":
                meta_total[k] += int(meta_b[k])

        if debug:
            _merge_debug_examples(debug_examples_total, dbg_b, debug_max)

        if len(y_out) == 0:
            continue

        phylop_all.extend(phylop_b.tolist())
        kept_global.extend((start + kept_local).tolist())

        def run_model(batch_inputs):
            collated = collator(batch_inputs)
            x = collated[0].to(device)
            s = collated[1].to(device)
            with torch.no_grad():
                out = model(x, s)
            seq_logits = out.get("seq_logits", None)
            cons_pred = out.get("scaling_logits", None)
            if seq_logits is not None:
                seq_logits = seq_logits.float()
            if cons_pred is not None:
                cons_pred = cons_pred.float()
            return seq_logits, cons_pred

        logits_fwd, cons_fwd = run_model(fwd_inputs)
        logits_rev, cons_rev = run_model(rev_inputs)

        have_seq = (logits_fwd is not None) and (logits_rev is not None)
        have_cons = (cons_fwd is not None) and (cons_rev is not None)

        for j in range(len(y_out)):
            y_out_all.append(int(y_out[j]))

            if have_seq:
                ref_tf, alt_tf, pos_f = fwd_info[j]
                ref_tr, alt_tr, pos_r = rev_info[j]
                pf = pos_f + 1
                pr = pos_r + 1

                lf = logits_fwd[j, pf]
                lr = logits_rev[j, pr]
                logp_ref_f = F.log_softmax(lf, dim=-1)[ref_tf].item()
                logp_alt_f = F.log_softmax(lf, dim=-1)[alt_tf].item()
                logp_ref_r = F.log_softmax(lr, dim=-1)[ref_tr].item()
                logp_alt_r = F.log_softmax(lr, dim=-1)[alt_tr].item()
                ref_ll_out.append(0.5 * (logp_ref_f + logp_ref_r))
                alt_ll_out.append(0.5 * (logp_alt_f + logp_alt_r))
            else:
                ref_ll_out.append(np.nan)
                alt_ll_out.append(np.nan)

            if have_cons:
                _, _, pos_f = fwd_info[j]
                _, _, pos_r = rev_info[j]
                pf = pos_f + 1
                pr = pos_r + 1
                cf = cons_fwd[j, pf, 0].item()
                cr = cons_rev[j, pr, 0].item()
                cons_out.append(0.5 * (cf + cr))

    ref_ll = np.asarray(ref_ll_out, dtype=np.float32)
    alt_ll = np.asarray(alt_ll_out, dtype=np.float32)
    cons_pred = np.asarray(cons_out, dtype=np.float32) if len(cons_out) else None
    y_out = np.asarray(y_out_all, dtype=np.int64)
    phylop_scores = np.asarray(phylop_all, dtype=np.float32)

    meta_total["n_used"] = int(len(y_out))
    return (
        ref_ll,
        alt_ll,
        cons_pred,
        y_out,
        phylop_scores,
        np.asarray(kept_global, dtype=np.int64),
        meta_total,
        debug_examples_total,
    )


def process_variants_caduceus(
    genome: Fasta,
    bw: pyBigWig.pyBigWig,
    model,
    collator,
    tokenizer,
    device,
    df: pd.DataFrame,
    y: np.ndarray,
    batch_size: int = 64,
    window_size: int = 2048,
    *,
    debug: bool = False,
    debug_max: int = 5,
):
    """
    Returns:
      ref_ll, alt_ll, cons_pred, y_out, phylop_scores, kept_global_idx, meta, debug_examples
    """
    ref_ll_out, alt_ll_out = [], []
    cons_out = []
    y_out_all = []
    phylop_all = []
    kept_global: List[int] = []

    meta_total = {
        "n_total": int(len(df)),
        "n_used": 0,
        "bad_chrom": 0,
        "missing_fasta_chrom": 0,
        "missing_bw_chrom": 0,
        "bad_pos": 0,
        "bad_allele": 0,
        "short_window": 0,
        "non_acgt_window": 0,
        "ref_mismatch": 0,
    }
    debug_examples_total = {k: [] for k in meta_total if k not in ("n_total", "n_used")} if debug else None

    for start in tqdm(range(0, len(df), batch_size)):
        end = min(start + batch_size, len(df))
        df_b = df.iloc[start:end].reset_index(drop=True)
        y_b = y[start:end]

        (
            fwd_inputs,
            rev_inputs,
            fwd_info,
            rev_info,
            y_out,
            phylop_b,
            kept_local,
            meta_b,
            dbg_b,
        ) = _build_batch_inputs(
            genome, bw, df_b, y_b, tokenizer, window_size, debug=debug, debug_max=debug_max
        )

        for k in meta_total:
            if k in meta_b and k != "n_total":
                meta_total[k] += int(meta_b[k])

        if debug:
            _merge_debug_examples(debug_examples_total, dbg_b, debug_max)

        if len(y_out) == 0:
            continue

        phylop_all.extend(phylop_b.tolist())
        kept_global.extend((start + kept_local).tolist())

        fwd_pos = [t[2] for t in fwd_info]
        rev_pos = [t[2] for t in rev_info]

        def run_model(batch_inputs, positions):
            collated = collator(batch_inputs, region=[(p, p + 1) for p in positions])
            x = collated[0][:, 0, :].long().to(device)
            labels = collated[1][:, 0, :].long().to(device)

            with torch.no_grad():
                out = model(input_ids=x, labels=labels)

            logits = out.get("logits", None)
            cons_pred = out.get("scaling_logits", None)
            if logits is not None:
                logits = logits.float()
            if cons_pred is not None:
                cons_pred = cons_pred.float()
            return logits, cons_pred

        logits_fwd, cons_fwd = run_model(fwd_inputs, fwd_pos)
        logits_rev, cons_rev = run_model(rev_inputs, rev_pos)

        have_seq = (logits_fwd is not None) and (logits_rev is not None)
        have_cons = (cons_fwd is not None) and (cons_rev is not None)

        for j in range(len(y_out)):
            y_out_all.append(int(y_out[j]))

            if have_seq:
                ref_tf, alt_tf, pos_f = fwd_info[j]
                ref_tr, alt_tr, pos_r = rev_info[j]
                pf = pos_f + 1
                pr = pos_r + 1

                lf = logits_fwd[j, pf]
                lr = logits_rev[j, pr]
                logp_ref_f = F.log_softmax(lf, dim=-1)[ref_tf].item()
                logp_alt_f = F.log_softmax(lf, dim=-1)[alt_tf].item()
                logp_ref_r = F.log_softmax(lr, dim=-1)[ref_tr].item()
                logp_alt_r = F.log_softmax(lr, dim=-1)[alt_tr].item()
                ref_ll_out.append(0.5 * (logp_ref_f + logp_ref_r))
                alt_ll_out.append(0.5 * (logp_alt_f + logp_alt_r))
            else:
                ref_ll_out.append(np.nan)
                alt_ll_out.append(np.nan)

            if have_cons:
                _, _, pos_f = fwd_info[j]
                _, _, pos_r = rev_info[j]
                pf = pos_f + 1
                pr = pos_r + 1
                cf = cons_fwd[j, pf, 0].item()
                cr = cons_rev[j, pr, 0].item()
                cons_out.append(0.5 * (cf + cr))

    ref_ll = np.asarray(ref_ll_out, dtype=np.float32)
    alt_ll = np.asarray(alt_ll_out, dtype=np.float32)
    cons_pred = np.asarray(cons_out, dtype=np.float32) if len(cons_out) else None
    y_out = np.asarray(y_out_all, dtype=np.int64)
    phylop_scores = np.asarray(phylop_all, dtype=np.float32)

    meta_total["n_used"] = int(len(y_out))
    return (
        ref_ll,
        alt_ll,
        cons_pred,
        y_out,
        phylop_scores,
        np.asarray(kept_global, dtype=np.int64),
        meta_total,
        debug_examples_total,
    )


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, choices=sorted(TASKS.keys()), required=True)
    p.add_argument("--model_type", type=str, choices=["gamba", "caduceus"], required=True)
    p.add_argument("--training_task", type=str, choices=["dual", "seq_only", "cons_only"], required=True)

    p.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    p.add_argument(
        "--big_wig",
        type=str,
        default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )

    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--config_fpath", type=str, required=True)
    p.add_argument("--last_step", type=int, default=-1)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--promoter_tss_bp", type=int, default=1500)
    p.add_argument("--promoter_k_neg", type=int, default=10)
    p.add_argument("--promoter_use_match_group", action="store_true", default=True)

    p.add_argument("--no_save_npz", action="store_true", help="disable saving per-variant npz")

    p.add_argument("--debug", action="store_true", help="write debug drop reasons + examples to debug_drops.json")
    p.add_argument("--debug_max", type=int, default=5, help="max examples per drop reason to keep")

    p.add_argument(
        "--evo2_parquet",
        type=str,
        default=None,
        help="optional hf://datasets/... parquet containing evo2 7b per-variant scores for this task",
    )
    p.add_argument(
        "--evo2_score_col",
        type=str,
        default="score",
        help="column name in evo2 parquet to use as the prediction score",
    )

    return p.parse_args()


def main():
    args = parse_args()
    spec = TASKS[args.task_name]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_task_df(spec)

    if "pos" in df.columns:
        before = len(df)
        df = df.dropna(subset=["pos"]).copy()
        df["pos"] = df["pos"].astype(int)
        if len(df) != before:
            print(f"[load_task_df] dropped {before - len(df)} rows with non-numeric pos")

    pmax = int(df["pos"].max())
    if pmax < 1000:
        raise ValueError(f"pos looks corrupted (max={pmax}). check HF loading / pyarrow.")

    y = normalize_labels(df)

    if spec.derived:
        df, y = derive_promoter_from_ukb_nc(
            df=df,
            y=y,
            tss_bp=args.promoter_tss_bp,
            k_neg=args.promoter_k_neg,
            use_match_group=args.promoter_use_match_group,
            seed=args.seed,
        )

    genome = Fasta(args.genome_fasta)
    bw = pyBigWig.open(args.big_wig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(
        args.checkpoint_dir,
        args.config_fpath,
        last_step=args.last_step,
        device=device,
        training_task=args.training_task,
        model_type=args.model_type,
    )

    if args.model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
        ref_ll, alt_ll, cons_pred, y_out, phylop_scores, kept_idx, meta, dbg = process_variants_gamba(
            genome=genome,
            bw=bw,
            model=model,
            collator=collator,
            tokenizer=tokenizer,
            device=device,
            df=df,
            y=y,
            batch_size=args.batch_size,
            debug=args.debug,
            debug_max=args.debug_max,
        )
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)
        ref_ll, alt_ll, cons_pred, y_out, phylop_scores, kept_idx, meta, dbg = process_variants_caduceus(
            genome=genome,
            bw=bw,
            model=model,
            collator=collator,
            tokenizer=tokenizer,
            device=device,
            df=df,
            y=y,
            batch_size=args.batch_size,
            debug=args.debug,
            debug_max=args.debug_max,
        )

    n = int(len(y_out))

    phylop_score, phylop_se, phylop_n = safe_metric(
        y_out, phylop_scores, spec.metric, n_boot=args.n_boot, seed=args.seed
    )

    llr_metric = spec.metric if args.training_task in ("dual", "seq_only") else "N/A"
    cons_metric = spec.metric if args.training_task in ("dual", "cons_only") else "N/A"

    llr_score = "N/A"
    llr_se = "N/A"
    cons_score = "N/A"
    cons_se = "N/A"

    if args.training_task in ("dual", "seq_only"):
        llr = compute_llr(ref_ll, alt_ll)
        llr_score, llr_se, _ = safe_metric(y_out, llr, spec.metric, n_boot=args.n_boot, seed=args.seed)

    if args.training_task in ("dual", "cons_only"):
        if cons_pred is None or len(cons_pred) == 0:
            cons_score = float("nan")
            cons_se = float("nan")
        else:
            cons_score, cons_se, _ = safe_metric(y_out, cons_pred, spec.metric, n_boot=args.n_boot, seed=args.seed)


    # ---------------- evo2 baseline (row-aligned + kept_idx aligned) ----------------
    evo2_score = "N/A"
    evo2_se = "N/A"
    evo2_n = 0
    evo2_flipped = False  # optional diagnostic

    evo2_path = args.evo2_parquet or EVO2_PARQUET.get(spec.name, None)
    if evo2_path is not None and (not spec.derived):
        try:
            evo2_all = load_evo2_scores_row_aligned(evo2_path, score_col=args.evo2_score_col)

            if len(evo2_all) != len(df):
                raise ValueError(
                    f"evo2 row-aligned length mismatch: len(evo2)={len(evo2_all)} vs len(task_df)={len(df)}. "
                    "cannot align by row order."
                )

            # align to exactly the same subset/order used by your model metrics
            kept_idx = np.asarray(kept_idx, dtype=np.int64)
            if kept_idx.ndim != 1:
                kept_idx = kept_idx.reshape(-1)

            # guard: indices should be valid in [0, len(df)-1]
            if len(kept_idx) and (kept_idx.min() < 0 or kept_idx.max() >= len(df)):
                raise ValueError(
                    f"kept_idx out of bounds for df: min={int(kept_idx.min())}, max={int(kept_idx.max())}, len(df)={len(df)}"
                )

            evo2_kept = evo2_all[kept_idx]

            # direction check (flip if better)
            evo2_kept_used, evo2_flipped = maybe_flip_scores_for_metric(y_out, evo2_kept, spec.metric)

            evo2_score_val, evo2_se_val, evo2_n_val = safe_metric(
                y_out, evo2_kept_used, spec.metric, n_boot=args.n_boot, seed=args.seed
            )
            evo2_score, evo2_se, evo2_n = evo2_score_val, evo2_se_val, evo2_n_val

        except Exception as e:
            print(f"[evo2] failed to load/score evo2 baseline: {e}")



    if not args.no_save_npz:
        np.savez_compressed(
            outdir / "scores.npz",
            y=y_out.astype(np.int8),
            phylop=phylop_scores.astype(np.float32),
            ref_ll=ref_ll.astype(np.float32),
            alt_ll=alt_ll.astype(np.float32),
            llr=compute_llr(ref_ll, alt_ll).astype(np.float32),
            cons_pred=(cons_pred.astype(np.float32) if cons_pred is not None else np.array([], dtype=np.float32)),
            task_name=spec.name,
            dataset=spec.dataset,
            metric=spec.metric,
            model_type=args.model_type,
            training_task=args.training_task,
            seed=args.seed,
            n_boot=args.n_boot,
            meta=json.dumps(meta),
        )


    row = {
        "Model": args.model_type,
        "TrainingTask": args.training_task,
        "Task": spec.name,
        "LLR_Metric": llr_metric,
        "LLR_Score": llr_score,
        "LLR_SE": llr_se,
        "Cons_Metric": cons_metric,
        "Cons_Score": cons_score,
        "Cons_SE": cons_se,
        "PhyloP_Metric": spec.metric,
        "PhyloP_Score": phylop_score,
        "PhyloP_SE": phylop_se,
        "PhyloP_N": phylop_n,
        "Evo2_Score": evo2_score,
        "Evo2_SE": evo2_se,
        "Evo2_N": evo2_n,
        "N": n,
    }
    pd.DataFrame([row]).to_csv(outdir / "result.tsv", sep="\t", index=False)

    with open(outdir / "result.txt", "w") as f:
        f.write(json.dumps(row, indent=2))
        f.write("\n")
        f.write(json.dumps(meta, indent=2))
        f.write("\n")

    if args.debug:
        debug_payload = {
            "task": spec.name,
            "dataset": spec.dataset,
            "metric": spec.metric,
            "meta": meta,
            "examples": dbg,
        }
        with open(outdir / "debug_drops.json", "w") as f:
            json.dump(debug_payload, f, indent=2)
        print("[debug] wrote:", outdir / "debug_drops.json")
        print("[debug] drop counters:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
