#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
from torch.nn import MSELoss, CrossEntropyLoss
from sequence_models.constants import MSA_PAD, START, STOP
from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator, gLMMLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)
from src.evaluation.utils.helpers import load_bed_file, extract_context
from src.evaluation.utils.specific_helpers import load_model #predict_scores_batched
from scipy import stats

def _spread_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    m = np.mean(arr)
    sd = np.std(arr, ddof=1) if arr.size > 1 else 0.0
    return {
        "n_categories": int(arr.size),
        "mean_of_means": float(m),
        "std_of_means": float(sd),
        "var_of_means": float(sd**2),
        "cv_of_means": float(sd / m) if m != 0 else np.nan,
        "range": float(np.ptp(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "mad_norm": float(stats.median_abs_deviation(arr, scale='normal'))
    }
def _finite(series):
    x = pd.to_numeric(series, errors="coerce")
    return x[np.isfinite(x)]

def _build_bins(x: pd.Series, n_bins: int = 40, fallback=(1.0, 1.4)):
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = fallback
    return np.linspace(lo, hi, n_bins + 1)

def _pmf(vals: np.ndarray, bins: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # uniform bins → counts normalized to 1 is sufficient
    h, _ = np.histogram(vals, bins=bins)
    p = h.astype(np.float64) + eps
    p /= p.sum()
    return p

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)

def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p / max(p.sum(), eps)
    return float(-np.sum(p * np.log(p + eps)))

def _generalized_jsd(pmfs: list[np.ndarray], weights: np.ndarray | None = None, eps: float = 1e-12) -> float:
    k = len(pmfs)
    if weights is None:
        weights = np.ones(k, dtype=np.float64) / k
    m = np.zeros_like(pmfs[0], dtype=np.float64)
    for w, p in zip(weights, pmfs):
        m += w * (p / max(p.sum(), eps))
    return float(_entropy(m, eps) - np.sum([w * _entropy(p, eps) for w, p in zip(weights, pmfs)]))

def category_stats(df, split="Training", value_col="loss"):
    if split == "Training":
        d = df[(df["data_split"]=="Training") & df[value_col].notna()].copy()
    else:
        d = df[df[value_col].notna()].copy()
    d = d[d[value_col].notna()]
    cats = [c for c in CATEGORY_ORDER if c in set(d["category"])]
    out = []
    for c in cats:
        v = pd.to_numeric(d.loc[d["category"]==c, value_col], errors="coerce").dropna().values
        if v.size==0: continue
        mean = float(np.mean(v)); std = float(np.std(v, ddof=1)); n = int(v.size)
        se = std/np.sqrt(n)
        q1,q2,q3 = np.percentile(v,[25,50,75])
        p5,p95   = np.percentile(v,[5,95])
        out.append({
            "category": c, "n": n, "mean": mean, "std": std, "cv": std/mean if mean!=0 else np.nan,
            "se": se, "ci95_lo": mean-1.96*se, "ci95_hi": mean+1.96*se,
            "median": float(q2), "iqr": float(q3-q1), "p5": float(p5), "p95": float(p95)
        })
    return pd.DataFrame(out)

def plot_mean_ci_iqr(stats_df, out_dir, title="CE loss: mean, 95% CI, IQR", fname="ce_mean_ci_iqr.png"):
    os.makedirs(out_dir, exist_ok=True)
    dfp = stats_df.sort_values("mean")
    y = np.arange(len(dfp))
    plt.figure(figsize=(10, max(4, 0.5*len(dfp))))
    # IQR as thick bars
    for i, r in enumerate(dfp.itertuples()):
        plt.plot([r.median - r.iqr/2, r.median + r.iqr/2], [y[i], y[i]], linewidth=6, alpha=0.5)
    # 95% CI as thin whiskers
    for i, r in enumerate(dfp.itertuples()):
        plt.plot([r.ci95_lo, r.ci95_hi], [y[i], y[i]], linewidth=2)
    # Mean as point
    plt.scatter(dfp["mean"], y, s=30, zorder=3)
    plt.yticks(y, dfp["category"])
    plt.xlabel("CE loss"); plt.ylabel("Category")
    plt.title(title)
    plt.tight_layout()
    f = os.path.join(out_dir, fname)
    plt.savefig(f, dpi=300); plt.close()
    logging.info(f"Saved {f}")

def plot_ce_violin(
    df: pd.DataFrame,
    out_dir: str,
    split="Training",
    value_col="loss",
    clip_pct=(1,99),
    ylim=(0.5, 2.0)   # <-- consistent across all plots
):
    os.makedirs(out_dir, exist_ok=True)
    if split == "Training":
        d = df[(df["data_split"]==split) & df[value_col].notna()].copy()
    else:
        d = df[df[value_col].notna()].copy()
    if d.empty: return
    # lo = float(np.nanpercentile(df[value_col], clip_pct[0]))
    # hi = float(np.nanpercentile(df[value_col], clip_pct[1]))
    # d[value_col] = d[value_col].clip(lo, hi)

    cats = [c for c in CATEGORY_ORDER if c in set(d["category"])]
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=d, x="category", y=value_col,
        order=cats, scale="width", inner="quartile", cut=0
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.xlabel("Feature Category"); plt.ylabel("CE loss")
    plt.title(f"CE loss distributions by category — {split}") # (clipped {clip_pct[0]}–{clip_pct[1]}%)")
    if ylim:
        plt.ylim(*ylim)   # consistent y-axis
    plt.tight_layout()
    f = os.path.join(out_dir, f"violin_CE_{split.lower()}_fixed_ylim.png")
    plt.savefig(f, dpi=300); plt.close()
    logging.info(f"Saved {f}")


def compute_category_divergence(
    df: pd.DataFrame,
    out_dir: str,
    value_col: str = "loss",
    n_bins: int = 40,
    split = "Training"
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    if split == "Training":
        d = df[(df["data_split"] == "Training") & df[value_col].notna()].copy()
    else:
        d = df[df[value_col].notna()].copy()
    cats = [c for c in CATEGORY_ORDER if c in set(d["category"])]
    if not cats:
        logging.warning(f"No categories for {split} split.")
        return pd.DataFrame()

    bins = _build_bins(d[value_col], n_bins=n_bins)
    # build PMFs per category
    pmf = {}
    for cat in cats:
        vals = _finite(d.loc[d["category"] == cat, value_col].values)
        if len(vals) == 0: 
            continue
        pmf[cat] = _pmf(vals, bins)

    cats = [c for c in cats if c in pmf]
    if not cats:
        logging.warning("No non-empty categories to compare.")
        return pd.DataFrame()

    # pairwise KL and JS
    n = len(cats)
    KL_ij = np.zeros((n, n)); KL_ji = np.zeros((n, n)); JS = np.zeros((n, n))
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            pi, pj = pmf[ci], pmf[cj]
            KL_ij[i, j] = _kl(pi, pj)
            KL_ji[i, j] = _kl(pj, pi)
            JS[i, j] = _js(pi, pj)

    # save matrices
    kl_df = pd.DataFrame(KL_ij, index=cats, columns=cats)
    js_df = pd.DataFrame(JS, index=cats, columns=cats)
    kl_df.to_csv(os.path.join(out_dir, f"KL_{split}_categories.csv"))
    js_df.to_csv(os.path.join(out_dir, f"JS_{split}_categories.csv"))

    # heatmap for JS
    plt.figure(figsize=(0.6*len(cats)+3, 0.6*len(cats)+3))
    ax = sns.heatmap(js_df, annot=False, cmap="viridis", square=True, cbar_kws={"label": "JS divergence"})
    ax.set_title(f"Pairwise JS divergence of CE-loss distributions — {split}")
    plt.tight_layout()
    f = os.path.join(out_dir, f"JS_{split}_categories_heatmap.png")
    plt.savefig(f, dpi=300); plt.close()
    logging.info(f"Saved {f}")

    # total spread summary: generalized JSD + pairwise stats
    pmfs_ordered = [pmf[c] for c in cats]
    gjsd = _generalized_jsd(pmfs_ordered)
    summary = {
        "categories": cats,
        "generalized_JSD": float(gjsd),
        "pairwise_JS_mean": float(np.mean(JS[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        "pairwise_JS_max": float(np.max(JS[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        "pairwise_JS_min": float(np.min(JS[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
    }
    with open(os.path.join(out_dir, f"ce_loss_distribution_spread_{split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # long-form table
    rows = []
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            rows.append({
                "cat_i": ci, "cat_j": cj,
                "KL_i||j": float(KL_ij[i, j]),
                "KL_j||i": float(KL_ji[i, j]),
                "JS": float(JS[i, j])
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"divergence_{split}_categories_long.csv"), index=False)
    return out


def write_category_spread_report(df: pd.DataFrame, out_dir: str, value_col: str = "loss", fname="ce_category_spread.txt"):
    # If split exists, keep only Held Out rows. If none, fall back to all.
    if "data_split" in df.columns:
        df_sel = df[df["data_split"].eq("Held Out")]
        if df_sel.empty:
            logging.warning("No 'Held Out' rows found; using all rows.")
            df_sel = df.copy()
        split_label = "Held Out"
    else:
        df_sel = df.copy()
        split_label = "All"

    # per-category means (now single-split)
    means = (df_sel.groupby(["category"])[value_col].mean()
             .reset_index().rename(columns={value_col: "mean_loss"}))

    # spread across category means
    arr = means["mean_loss"].to_numpy()
    report = _spread_stats(arr)

    # include per-category means for traceability
    cat_table = means.set_index("category")["mean_loss"].round(6).to_dict()

    out = {
        "category_means": {k: {split_label: v} for k, v in cat_table.items()},
        "spread": {split_label: report}
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        f.write(json.dumps(out, indent=2))
    print(f"[INFO] Wrote spread report to {out_path}")

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters", #"phyloP_negative", "phyloP_neutral", "phyloP_positive",
]


def mask_gamba_ce_to_last_k(labels_2ch: torch.Tensor, last_k: int = 1000) -> torch.Tensor:
    """
    labels_2ch: (B, 2, T). Channel 0 = seq CE labels; Channel 1 = conservation labels.
    If T > last_k, keep only last_k tokens for CE; else keep the whole window.
    """
    labels = labels_2ch.clone()
    B, two, T = labels.shape
    assert two == 2
    k = min(last_k, T)
    start = T - k  # = 0 when T <= last_k → whole window kept
    labels[:, 0, :start] = -100  # ignore CE before start
    return labels



def apply_effective_region_mask(
    labels: torch.Tensor,                      # (B, 2, T): [:,0,:]=seq labels, [:,1,:]=cons labels
    feature_spans: list[tuple[int, int]],      # per-sample (fs, fe) in *token* indices (already shifted for [START] if needed)
    is_mlm: bool,                              # True for Caduceus (MLM), False for Gamba (AR)
    last_k: int = 1000,
) -> torch.Tensor:
    """
    Constrains both sequence CE and conservation losses to the *same* effective region:
      - If ROI length >= last_k: last `last_k` tokens *within the ROI*
      - Else: the entire ROI
    For MLM: CE is further restricted to masked tokens ∩ effective region (labels== -100 outside).
    For AR:  CE is restricted exactly to the effective region (labels== -100 outside).
    Conservation labels are always restricted to the effective region.

    NOTE: This function expects spans already adjusted for any special tokens the collator added.
    """
    labels = labels.clone()
    B, two, T = labels.shape
    assert two == 2, "labels must have 2 channels (seq, cons)"

    for b, (fs, fe) in enumerate(feature_spans):
        # clamp ROI to [0, T]
        fs = max(0, min(fs, T))
        fe = max(0, min(fe, T))

        # compute effective region inside ROI: tail-k of ROI or whole ROI
        roi_len = max(0, fe - fs)
        if roi_len == 0:
            # no region → ignore everything
            labels[b, 0, :] = -100
            labels[b, 1, :] = -100
            continue

        k = min(last_k, roi_len)
        if not is_mlm:
            eff_fs = fe - k   # last k inside ROI
            eff_fe = fe
        else:
            eff_fs = fs
            eff_fe = fe

        # ---- SEQUENCE (channel 0) ----
        if is_mlm:
            # keep masked tokens only if they fall inside [eff_fs:eff_fe)
            keep = torch.zeros(T, dtype=torch.bool, device=labels.device)
            keep[eff_fs:eff_fe] = True
            masked = labels[b, 0, :] != -100         # collator set masked tokens to labels!= -100
            kill = masked & (~keep)
            labels[b, 0, kill] = -100                # ignore masked tokens outside the effective region
        else:
            # AR / Gamba: compute CE only on the effective region
            labels[b, 0, :eff_fs] = -100
            labels[b, 0, eff_fe:] = -100

        # ---- CONSERVATION (channel 1) ----
        labels[b, 1, :eff_fs] = -100
        labels[b, 1, eff_fe:] = -100

        # (Optional safety) ignore [START]/[STOP] if present at 0 / T-1
        labels[b, 0, 0]  = -100
        labels[b, 1, 0]  = -100
        labels[b, 0, -1] = -100
        labels[b, 1, -1] = -100

    return labels

def _masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1)
    return num / den


def predict_scores_batched(model, tokenizer, regions, batch_size=8, device=None, model_type="gamba", training_task="dual"):
    """Run predictions on sampled regions with masking applied only over the feature region."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torch.nn import functional as F
    from gamba.collators import gLMCollator

    all_predictions = []
    all_true_scores = []
    all_seq_predictions = []
    all_true_seqs = []
    region_info = []

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
            sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            fs = region.get('feature_start_in_window', 0)
            fe = region.get('feature_end_in_window', len(scores))

            batch_inputs.append((sequence_tokens,  scores))
            # Record metadata
            batch_region_info.append({
                'chrom': region['chrom'],
                'start': region['start'],
                'end': region['end'],
                'feature_id': region.get('feature_id', 'unknown'),
                'mean_score': region.get('mean_score', 0.0),
                'feature_start_in_window': fs,
                'feature_end_in_window': fe
            })
            region_info.append(batch_region_info[-1])
            all_true_scores.append(scores)
            all_true_seqs.append(sequence_tokens)

        # Skip empty batches
        if not batch_inputs:
            continue
        # === Gamba Forward (per-example losses) ===
        elif model_type == "gamba":
            inputs, labels = collator(batch_inputs)  # (B,2,T)
            inputs, labels = inputs.to(device), labels.to(device)

            # shift ROI spans by +1 for [START]
            feature_spans = [(int(m["feature_start_in_window"]) + 1,
                            int(m["feature_end_in_window"])   + 1)
                            for m in batch_region_info]

            # restrict CE and CONS to the effective region
            labels = apply_effective_region_mask(labels, feature_spans,
                                                is_mlm=False, last_k=1000)

            with torch.no_grad():
                out = model(inputs, labels)  # still returns logits we need

            # logits
            seq_logits = out["seq_logits"]            # (B,T,V)
            cons_pred  = out.get("scaling_logits", None)  # (B,T,2) if present

            # ----- CE per-example (AR shift) -----
            ce_labels = labels[:, 0, :].long()        # (B,T), -100 outside ROI
            # shift for AR: logits[:, :-1] -> labels[:, 1:]
            logit_shift = seq_logits[:, :-1, :]       # (B,T-1,V)
            label_shift = ce_labels[:, 1:]            # (B,T-1)
            mask_shift  = label_shift.ne(-100).float()

            ce_tok = F.cross_entropy(
                logit_shift.reshape(-1, logit_shift.size(-1)),
                label_shift.reshape(-1),
                reduction="none"
            ).view(label_shift.size())                 # (B,T-1)

            ce_per_ex = _masked_mean_per_row(ce_tok, mask_shift, dim=1)  # (B,)

            # ----- MSE per-example (if conservation head exists) -----
            if cons_pred is not None:
                cons_mean = cons_pred[..., 0].float()          # (B,T)
                cons_tgt  = labels[:, 1, :].float()            # (B,T)
                cons_mask = cons_tgt.ne(-100).float()          # (B,T)
                mse_tok   = (cons_mean - cons_tgt).pow(2)      # (B,T)
                mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)  # (B,)
            else:
                mse_per_ex = torch.full_like(ce_per_ex, float("nan"))

            # collect
            all_seq_predictions.extend(ce_per_ex.detach().cpu().tolist())
            all_predictions.extend(mse_per_ex.detach().cpu().tolist())


        # === Caduceus Forward (per-example losses with repeats) ===
        elif model_type == "caduceus":
            raw_spans = [(r["feature_start_in_window"], r["feature_end_in_window"])
                        for r in batch_region_info]

            R = 7  # 15% * 7 ≈ covers ROI once on average
            B = len(batch_inputs)
            ce_accum  = torch.zeros(B, dtype=torch.float32, device=device)
            mse_accum = torch.zeros(B, dtype=torch.float32, device=device)
            used = 0

            for _ in range(R):
                try:
                    batch = collator(batch_inputs, region=raw_spans)
                except TypeError:
                    batch = collator(batch_inputs, region=raw_spans)

                sequence_input = batch[0][:, 0, :].long().to(device)
                labels_pack    = batch[1].to(device)  # (B,2,T)

                # shift spans by +1 for [START]
                feature_spans_shifted = [(fs + 1, fe + 1) for (fs, fe) in raw_spans]
                # CE = masked tokens ∩ ROI; CONS = ROI
                labels_pack = apply_effective_region_mask(
                    labels_pack, feature_spans_shifted, is_mlm=True, last_k=1000
                )

                with torch.no_grad():
                    # run to get logits; labels not needed here
                    outputs = model(input_ids=sequence_input, return_dict=True)
                
                if "logits" in outputs:
                    logits = outputs["logits"].float()          # (B,T,V)
                    ce_labels = labels_pack[:, 0, :].long()     # (B,T), -100 outside ROI
                    cons_tgt  = labels_pack[:, 1, :].float()    # (B,T), -100 outside ROI
                    # ----- CE per-example (MLM, no shift) -----
                    ce_tok = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        ce_labels.reshape(-1),
                        reduction="none"
                    ).view(ce_labels.size())                    # (B,T)

                    ce_mask = ce_labels.ne(-100).float()        # (B,T)
                    ce_per_ex = _masked_mean_per_row(ce_tok, ce_mask, dim=1)  # (B,)
                    ce_accum += ce_per_ex
                else:
                    # accumulate NaNs to keep vector length
                    cons_tgt = labels_pack[:, 1, :].float()    # (B,T), -100 outside ROI
                    ce_accum += torch.full_like(mse_accum, float("nan"))

                # ----- MSE per-example if conservation head exposed -----
                if "scaling_logits" in outputs:
                    cons_mean = outputs["scaling_logits"][..., 0].float()  # (B,T)
                    cons_mask = cons_tgt.ne(-100).float()
                    mse_tok   = (cons_mean - cons_tgt).pow(2)
                    mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)  # (B,)
                    mse_accum += mse_per_ex
                else:
                    # accumulate NaNs to keep vector length
                    mse_accum += torch.full_like(ce_per_ex, float("nan"))

                used += 1

            # finalize
            ce_mean  = (ce_accum / max(used, 1)).detach().cpu().tolist()
            mse_mean = (mse_accum / max(used, 1)).detach().cpu().tolist()
            all_seq_predictions.extend(ce_mean)
            all_predictions.extend(mse_mean)



    return all_predictions, all_true_scores, region_info, all_seq_predictions, all_true_seqs


def reindex_categories(df):
    return df.reindex([cat for cat in CATEGORY_ORDER if cat in df.index])

from Bio.Seq import Seq

def extract_sequence_from_genome(genome: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    """
    Extract a sequence from the genome, reverse complementing it if on the minus strand.

    Args:
        genome: pyfaidx.Fasta object with loaded genome.
        chrom: Chromosome name (must match keys in genome, e.g., 'chr1').
        start: 0-based start coordinate (inclusive).
        end: 0-based end coordinate (exclusive).
        strand: '+' or '-'.

    Returns:
        DNA sequence as a string.
    """
    try:
        if chrom not in genome:
            raise ValueError(f"Chromosome {chrom} not found in genome FASTA.")

        seq = genome[chrom][start:end].seq.upper()

        if strand == '-':
            seq = str(Seq(seq).reverse_complement())

        return seq
    except Exception as e:
        print(f"Error extracting sequence from {chrom}:{start}-{end} ({strand}): {e}")
        return "N" * (end - start)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_latest_dcp_checkpoint_path(ckpt_dir, last_step=-1):
    """Find the latest checkpoint path."""
    ckpt_path = None
    if last_step == -1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path

#need to get the CE loss & conservation  ONLY in the region of interest
def plot_feature_bars(data, value_col, ylabel, title, out_file, output_dir, ylim=None):
    logging.info(f"Creating bar plot: {title}")

    # Filter to the two splits we care about
    data = data[data['data_split'].isin(['Training', 'Held Out'])].copy()

    # Aggregate across ALL chromosomes within each split & category
    agg = (data
           .groupby(['category', 'data_split'])[value_col]
           .agg(['mean', 'std', 'count'])
           .reset_index())

    # Pivot and enforce consistent category order
    pivot = agg.pivot(index='category', columns='data_split', values=['mean', 'std'])
    pivot = reindex_categories(pivot)

    categories = pivot.index.tolist()
    if not categories:
        logging.warning("No categories to plot after aggregation.")
        return

    x = np.arange(len(categories))
    width = 0.42

    plt.figure(figsize=(12, 8))

    # Bars for the two splits, same x with offsets
    for split, offset in [('Held Out', -width/2), ('Training', width/2)]:
        mean_key = ('mean', split)
        if mean_key not in pivot.columns:
            continue
        means = pivot[mean_key].values
        stds = pivot.get(('std', split), None)
        stds = stds.values if stds is not None else None

        # consistent styling, no per-chrom labels
        plt.bar(x + offset, means, width, label=split,
                yerr=stds, capsize=5, linewidth=1.2, alpha=0.9)

        # Annotate values
        for j, v in enumerate(means):
            if np.isfinite(v):
                plt.text(x[j] + offset, v + (0.02 if ylim is None else (ylim[1]-ylim[0])*0.02),
                         f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel('Feature Category', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Bar plot saved to {out_path}")

def plot_feature_heatmap_by_split(
    data, value_col, ylabel, title, out_file, output_dir,
    vmin, vmax, cmap
):
    """
    One column per split ('Held Out', 'Training'), rows = categories.
    Values are the mean across *all* chromosomes within each split.
    """
    logging.info(f"Creating split-averaged heatmap: {title}")

    # Keep only the two splits of interest
    df = data[data['data_split'].isin(['Training', 'Held Out'])].copy()

    # Average over all chromosomes (and regions) per category x split
    agg = (df.groupby(['category', 'data_split'])[value_col]
             .mean()
             .reset_index())

    # Pivot to (categories x splits)
    pivot = agg.pivot(index='category', columns='data_split', values=value_col)

    # Ensure both columns exist and order them
    for col in ['Held Out', 'Training']:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[['Held Out', 'Training']]

    # Enforce consistent row order and drop empty rows
    pivot = pivot.reindex([c for c in CATEGORY_ORDER if c in pivot.index]).dropna(how='all')

    if pivot.empty:
        logging.warning("No data to plot after aggregation.")
        return

    plt.figure(figsize=(6, max(6, 0.5*len(pivot))))
    ax = sns.heatmap(
        pivot,
        annot=True, fmt='.3f',
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0.5,
        cbar_kws={'label': ylabel},
        annot_kws={"size": 10, "weight": "bold"}
    )
    ax.set_xlabel("")  # columns already labeled
    ax.set_ylabel("Feature Category")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Split-averaged heatmap saved to {out_path}")


def debug_cons_mse_batch(tag, outputs, cons_labels_roi):
    """
    Print stats for conservation targets & predictions used in MSE.
    cons_labels_roi: (B, T) with -100 outside ROI
    """
    with torch.no_grad():
        mean = outputs["scaling_logits"][..., 0].float()  # (B, T)
        mask = cons_labels_roi != -100

        # counts
        used = mask.sum(dim=1)
        print(f"[{tag}] tokens used per sample (min/median/max):",
              int(used.min()), int(used.median()), int(used.max()))

        # stats for targets/preds on the used positions
        tgt = cons_labels_roi.masked_select(mask)
        pred = mean.masked_select(mask)

        def _q(t):  # quantiles
            q = torch.quantile(t, torch.tensor([0., 0.25, 0.5, 0.75, 1.], device=t.device))
            return [float(x) for x in q]

        print(f"[{tag}] target phyloP q=[min, q1, med, q3, max]:", _q(tgt))
        print(f"[{tag}] pred   phyloP q=[min, q1, med, q3, max]:", _q(pred))

        mse = torch.mean((pred - tgt) ** 2).item()
        rmse = (torch.mean((pred - tgt) ** 2).sqrt()).item()
        print(f"[{tag}] MSE={mse:.3f}  RMSE={rmse:.3f}")


from pathlib import Path
import torch
import glob
import logging
from pyfaidx import Fasta
import pyBigWig
from tqdm import tqdm

# After you get:
# mse_losses, true_scores, region_info, ce_losses, all_true_sequences = predict_scores_batched(...)

def build_results_df(
    mse_losses, ce_losses, true_scores, region_info,
    category: str, group_name: str,
    training_chromosomes: list[str] | None,
    test_chromosomes: list[str] | None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    # --- Alignment checks across lists ---
    n = len(region_info)
    lens = {
        "mse_losses": len(mse_losses),
        "ce_losses": len(ce_losses),
        "true_scores": len(true_scores),
        "region_info": len(region_info),
    }
    if len(set(lens.values())) != 1:
        logger.warning(f"[ALIGNMENT] Length mismatch: {lens}. "
                       f"Proceeding with min length to avoid index errors.")
        n = min(lens.values())
        mse_losses = mse_losses[:n]
        ce_losses = ce_losses[:n]
        true_scores = true_scores[:n]
        region_info = region_info[:n]

    # Train/test split helper
    train_set = set(training_chromosomes or [])
    test_set  = set(test_chromosomes or [])

    def split_of(chrom: str) -> str:
        if chrom in train_set: return "Training"
        if chrom in test_set:  return "Held Out"
        return "Unknown"

    # --- Per-region rows + ROI validity checks ---
    rows = []
    n_bad_span = 0
    n_out_of_bounds = 0
    n_nan_losses = 0

    for i in range(n):
        info = region_info[i]
        chrom = info["chrom"]
        fs = int(info.get("feature_start_in_window", 0))
        fe = int(info.get("feature_end_in_window", 0))
        T  = int(info.get("window_len", fe))  # fall back to fe if not present

        # ROI span check
        if fe <= fs:
            n_bad_span += 1
            logger.debug(f"[ALIGNMENT] Empty/invalid ROI span at idx {i}: fs={fs}, fe={fe}")

        # Bounds check
        if not (0 <= fs <= T and 0 <= fe <= T):
            n_out_of_bounds += 1
            logger.debug(f"[ALIGNMENT] ROI out of bounds at idx {i}: fs={fs}, fe={fe}, T={T}")

        # Loss sanity
        ce_i  = float(ce_losses[i]) if ce_losses is not None else float("nan")
        mse_i = float(mse_losses[i]) if mse_losses is not None else float("nan")
        if not np.isfinite(ce_i) or not np.isfinite(mse_i):
            n_nan_losses += 1

        rows.append({
            "chrom":          chrom,
            "start":          info["start"],
            "end":            info["end"],
            "feature_id":     info.get("feature_id", "unknown"),
            "loss":           ce_i,     # CE (should already be ROI-only if you masked labels pre-forward)
            "mse":            mse_i,    # MSE (ROI-only if conservation labels masked pre-forward)
            "feature_start":  fs,
            "feature_end":    fe,
            "feature_length": max(0, fe - fs),
            "window_len":     T,
            "data_split":     split_of(chrom),
        })

    if n_bad_span or n_out_of_bounds or n_nan_losses:
        logger.info(
            f"[ALIGNMENT SUMMARY] bad_span={n_bad_span}, out_of_bounds={n_out_of_bounds}, "
            f"nan_losses={n_nan_losses} out of {n} rows"
        )

    df = pd.DataFrame(rows)
    df["category"] = category
    df["group"] = group_name
    return df

def analyze_agreement(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    num_regions=100,
    region_length=2048,
    chromosomes=None,
    last_step=None,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task='dual',
    model_type='gamba'
):
    """
    Analyze agreement between predicted and true phyloP scores using pre-defined BED regions.
    Outputs CE loss and phyloP correlation per region, with heatmaps per category/chromosome.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    bw = pyBigWig.open(bigwig_file)

    model, tokenizer = load_model(
        checkpoint_dir, config_fpath,
        last_step=last_step, device=device,
        training_task=training_task, model_type=model_type
    )
    genome = Fasta(genome_fasta)

    categories = [
        #"phyloP_negative", "phyloP_neutral", "phyloP_positive", 
        "UCNE", "repeats", "exons", "introns", "noncoding_regions", "coding_regions", "upstream_TSS", "UTR5", "UTR3", "promoters", "vista_enhancer"]

    chromosome_groups = {
        "training": training_chromosomes or [],
        "test": test_chromosomes or [],
    }

    all_results = []

    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"Analyzing {group_name} chromosomes: {group_chroms}")
        for category in categories:
            bed_files = glob.glob(f"/home/mica/gamba/data_processing/data/regions/{category}/*.bed")
            group_regions = []
            for bed_file in bed_files:
                loaded = load_bed_file(bed_file, category, genome, bw)
                group_regions.extend([r for r in loaded if r["chrom"] in group_chroms])

            if not group_regions:
                logging.warning(f"No regions found for {category} in {group_name}")
                continue

            group_regions = group_regions[:num_regions]
            valid_regions = []
            for i, region in enumerate(group_regions):
                context = extract_context(bigwig_file, region, genome, model_type)
                if not context or "sequence" not in context:
                    print(f"[WARN] Region {i} has invalid or truncated sequence")
                    continue
                valid_regions.append(context)

            if not valid_regions:
                logging.warning(f"[SKIP] All regions in group {group_name} were invalid.")
                continue


            mse_losses, true_scores, region_info, ce_losses, all_true_sequences = predict_scores_batched(
                model, tokenizer, valid_regions, batch_size=batch_size,
                device=device, model_type=model_type, training_task=training_task
            )
            
            results_df = build_results_df(
                mse_losses, ce_losses, true_scores, region_info,
                category=category, group_name=group_name,
                training_chromosomes=training_chromosomes,
                test_chromosomes=test_chromosomes,
                logger=logging.getLogger(__name__),
            )

            results_df_path = output_dir / f"{category}_{group_name}_mse_ce_results.csv"
            results_df.to_csv(results_df_path, index=False)
            all_results.append(results_df)


    # Merge all region results into one DataFrame
    full_df = pd.concat(all_results, ignore_index=True)


    # Save merged results
    #full_df.to_csv(output_dir / "all_region_results.csv", index=False)
    full_df.to_csv(output_dir / "all_region_results.csv", index=False)

    if training_task == "dual" or training_task == "cons_only":
        write_category_spread_report(full_df, out_dir=output_dir, value_col="mse",  fname="mse_category_spread.txt")
    if training_task == "seq_only" or training_task == "dual":
        write_category_spread_report(full_df, out_dir=output_dir, value_col="loss", fname="ce_category_spread.txt")
            # Training-only category distributions and divergences
        stats_tr = category_stats(full_df, split="All", value_col="loss")
        stats_tr.to_csv(os.path.join(output_dir, "ce_category_stats_all.csv"), index=False)
        plot_mean_ci_iqr(stats_tr, output_dir)
        parq = os.path.join(output_dir, "all_results.parquet")
        full_df.to_parquet(parq, index=False, compression="snappy")  # needs pyarrow


        plot_ce_violin(full_df, output_dir, split="All", value_col="loss", clip_pct=(1,99))

        _ = compute_category_divergence(full_df, output_dir, value_col="loss", n_bins=40, split="Training")



    # Create summary plots
    data=full_df
    from matplotlib.colors import LinearSegmentedColormap
    # plot_feature_bars(data, 'correlation',
    #               ylabel='Mean Correlation (Predicted vs True PhyloP)',
    #               title='Corre lation: Held Out vs Training Chromosomes',
    #               out_file='feature_comparison_bar_plot.png',
    #               output_dir=output_dir,
    #               ylim=(-0.1, 0.5))

    if training_task == "dual" or training_task == "cons_only":
        plot_feature_bars(
            data, 'mse',
            ylabel='Mean MSE (Predicted Mean vs True phyloP)',
            title='MSE: Held Out vs Training Chromosomes',
            out_file='feature_comparison_mse_bar_plot.png',
            output_dir=output_dir,
            # Optionally set ylim; uncomment if you want a fixed view:
            ylim=(0.0, 3.5)
        )

        plot_feature_heatmap_by_split(
            data, value_col='mse',
            ylabel='Mean MSE',
            title='Mean MSE (Held-Out vs Training, averaged)',
            out_file='feature_split_mse_heatmap.png',
            output_dir=output_dir,
            # If you know your expected range, set vmin/vmax; otherwise let it auto-scale:
            vmin=None, vmax=None,
            cmap=LinearSegmentedColormap.from_list('white_to_blue', [(1,1,1),(0,0.4,0.8)], N=100)
        )
    if training_task == "seq_only" or training_task == "dual": 
        plot_feature_bars(data, 'loss',
                        ylabel='CE Loss',
                        title='Cross-Entropy Loss: Held Out vs Training Chromosomes',
                        out_file='feature_comparison_ce_loss_plot.png',
                        output_dir=output_dir,
                        ylim=(1.0, 1.40))

        plot_feature_heatmap_by_split(
            data, value_col='loss',
            ylabel='Mean Cross-Entropy Loss',
            title='CE Loss (Held-Out vs Training, averaged)',
            out_file='feature_split_ce_loss_heatmap.png',
            output_dir=output_dir,
            vmin=1.0, vmax=1.38,
            cmap=LinearSegmentedColormap.from_list('white_to_red', [(1,1,1),(0.8,0.1,0.1)], N=100)
        )
    
    bw.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze agreement between predicted and true phyloP scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/home/mica/gamba/',
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default='/home/mica/gamba/configs/jamba-small-240mammalian.json',
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="Number of regions to sample per category",
    )
    parser.add_argument(
        "--region_length",
        type=int,
        default=2048,
        help="Length of each sampled region",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr19", "chr22"],
        help="List of chromosomes to analyze",
    )
    parser.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        default=["chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10","chr11", "chr12", "chr13", "chr14", "chr15", "chr17", "chr18", "chr19", "chr20", "chr21", "chrX"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22", "chr16", "chr3"],
        help="List of chromosomes held out for testing",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default= 44000, #0,
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model predictions",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gamba", "caduceus"], required=True,
        help="Which model type to use (gamba or caduceus)"
    )
    parser.add_argument(
        "--training_task", type=str, choices=["dual", "cons_only", "seq_only", "random_init"], required=True,
        help="Which task the model was trained on"
    )

    args = parser.parse_args()
    
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Starting analysis script with chromosomes: {args.chromosomes}")
    logging.info(f"Training chromosomes: {args.training_chromosomes}")
    logging.info(f"Test chromosomes: {args.test_chromosomes}")
    logging.info(f"Using model type: {args.model_type} on task: {args.training_task}")

    if args.model_type == 'gamba':
        checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
        #checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/focal_loss/"
        # if args.training_task == "seq_only":
        #     checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/"
        #     args.last_step = 56000
    else:
        checkpoint_dir = args.checkpoint_dir + f"/clean_caduceus_dcps/allPOSMLM"
        #args.last_step = 56000 #0
    
    if args.last_step ==0:
        last_step = "random_init"
    else:
        last_step = args.last_step
    #change outputdir to + dcp checkpoint 
    output_dir = args.output_dir + f"/{args.model_type}_{args.training_task}_ALLPOSstep_{last_step}/"
    try:
        analyze_agreement(
            args.genome_fasta,
            args.bigwig_file,
            args.checkpoint_dir,
            args.config_fpath,
            output_dir,
            num_regions=args.num_regions,
            region_length=args.region_length,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size,
            training_task= args.training_task,
            model_type=args.model_type
        )
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()