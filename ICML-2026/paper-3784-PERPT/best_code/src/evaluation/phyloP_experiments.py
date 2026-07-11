#!/usr/bin/env python3
import argparse, os, sys, glob, json, logging, pathlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig
from pyfaidx import Fasta
from scipy import stats
import torch
from tqdm import tqdm

# ---------------- project paths ----------------
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
sys.path.append("/home/mica/gamba/src/")

from src.evaluation.utils.helpers import load_bed_file, extract_context
from src.evaluation.utils.specific_helpers import load_model

REPEATS_ROOT = "/home/mica/scratch/gamba/data_processing/data/regions/repeats"
DATA_DIR_DEFAULT = "/home/mica/gamba/data_processing/data/240-mammalian"


def chrom_sort_key(x: str) -> Tuple[int, str]:
    # assumes entries like "1","2","X","Y","MT"
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


def _load_dataset_splits(data_dir: str) -> Dict[str, List[str]]:
    """load train/valid/test chrom lists from splits.json."""
    splits_path = os.path.join(data_dir, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    # splits.json uses bare chrom names like "1", "2", "X"
    return {
        "train": sorted(splits.get("train", []), key=chrom_sort_key),
        "valid": sorted(splits.get("valid", []), key=chrom_sort_key),
        "test":  sorted(splits.get("test", []),  key=chrom_sort_key),
    }


def _load_cleaned_chrom_sizes(data_dir: str) -> Dict[str, int]:
    """
    load cleaned chrom sizes as a dict mapping bare chrom names ("1","2","X") -> length.
    expects lines like: chr1 1234567
    """
    path = os.path.join(data_dir, "cleaned_chrom_sizes.txt")
    sizes: Dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            name, size = line.split()[:2]
            bare = name.replace("chr", "")
            sizes[bare] = int(size)
    return sizes


def _sample_random_regions_from_dataset(
    data_dir: str,
    split: str,                 # "train" or "test"
    n: int,
    window_len: int,
    seed: int
) -> List[Dict]:
    """
    sample n windows of length `window_len` from the dataset split's chromosomes,
    using your preprocessed npy arrays.

    For each sampled window:
      seq_file = np.load(os.path.join(data_dir, split, f"{chrom}_sequence_small.npy"), mmap_mode="r")
      cons_file = np.load(os.path.join(data_dir, split, f"{chrom}_conservation_small.npy"), mmap_mode="r")

      sequence    = seq_file[start : start + window_len]
      cons_scores = cons_file[start : start + window_len]

    Returns a list of dicts, each of which contains:
      {
        "chrom": "chr1",                 # genome-style name with 'chr' prefix
        "start": start,                  # 0-based coordinate within chromosome
        "end": start + window_len,
        "category": f"{split}_random",
        "feature_start_in_window": 0,
        "feature_end_in_window": window_len,
        "sequence": np.ndarray,          # slice from *_sequence_small.npy
        "scores":   np.ndarray,          # slice from *_conservation_small.npy
      }

    Windows that cannot be extracted (e.g., due to length mismatches or repeat exclusion)
    are skipped; the function tries up to 10*n samples before giving up.
    """
    rng = np.random.default_rng(seed)

    # load split chroms and chrom sizes
    splits = _load_dataset_splits(data_dir)
    chrom_sizes = _load_cleaned_chrom_sizes(data_dir)

    if split not in splits:
        raise ValueError(f"split '{split}' not found in splits.json")

    dataset_chroms = splits[split]  # bare names like "1","2","X"
    if not dataset_chroms:
        raise SystemExit(f"[dataset_sample] no chromosomes found for split '{split}'")

    # memoize npy memmaps per chromosome to avoid reopening
    seq_memmaps: Dict[str, np.memmap] = {}
    cons_memmaps: Dict[str, np.memmap] = {}

    def _get_seq_memmap(c_bare: str):
        if c_bare not in seq_memmaps:
            path = os.path.join(data_dir, split, f"{c_bare}_sequence_small.npy")
            seq_memmaps[c_bare] = np.load(path, mmap_mode="r")
        return seq_memmaps[c_bare]

    def _get_cons_memmap(c_bare: str):
        if c_bare not in cons_memmaps:
            path = os.path.join(data_dir, split, f"{c_bare}_conservation_small.npy")
            cons_memmaps[c_bare] = np.load(path, mmap_mode="r")
        return cons_memmaps[c_bare]

    regions: List[Dict] = []
    tries = 0
    max_tries = n * 10

    while len(regions) < n and tries < max_tries:
        tries += 1

        # choose chromosome in dataset coords
        c_bare = rng.choice(dataset_chroms)
        if c_bare not in chrom_sizes:
            continue
        L = chrom_sizes[c_bare]
        if L <= window_len:
            continue

        start = int(rng.integers(0, L - window_len, endpoint=False))
        end = start + window_len

        # extract sequence + scores from npy
        try:
            seq_file = _get_seq_memmap(c_bare)
            cons_file = _get_cons_memmap(c_bare)
        except FileNotFoundError:
            # if any file is missing, skip this chrom/sample
            continue

        # basic length safety
        if start + window_len > len(seq_file) or start + window_len > len(cons_file):
            continue

        sequence = seq_file[start:start + window_len]
        cons_scores = cons_file[start:start + window_len]

        if sequence.shape[0] != window_len or cons_scores.shape[0] != window_len:
            continue

        region = {
            "chrom": c_bare,
            "start": start,
            "end": end,
            "category": f"{split}_random",
            "feature_start_in_window": 0,
            "feature_end_in_window": window_len,
            "sequence": np.array(sequence, copy=False),
            "scores":   np.array(cons_scores, copy=False),
        }
        regions.append(region)

    if len(regions) < n:
        logging.warning(
            f"[dataset_sample] split={split}: only obtained {len(regions)}/{n} windows "
            f"after {tries} attempts (repeats or data layout may be limiting)."
        )
    else:
        logging.info(f"[dataset_sample] split={split}: sampled {len(regions)} windows of {window_len} bp")

    return regions


def _load_repeats_per_chrom(chroms: List[str], repeats_root: str = REPEATS_ROOT) -> Dict[str, List[Tuple[int, int]]]:
    """
    Load per-chromosome repeat intervals from BED files:
      /home/mica/scratch/gamba/data_processing/data/regions/repeats/chr1.bed, etc.
    Assumes first three columns are chrom, start, end (standard BED); we only use start/end.
    """
    repeats: Dict[str, List[Tuple[int, int]]] = {}
    for c in chroms:
        bed_path = os.path.join(repeats_root, f"{c}.bed")
        if not os.path.exists(bed_path):
            logging.warning(f"[repeats] no repeats BED for {c} at {bed_path}")
            continue
        # standard BED: chrom, start, end, ...
        df = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            usecols=[1, 2],       # start, end
            names=["start", "end"]
        )
        if df.empty:
            continue
        intervals = df[["start", "end"]].astype(int).to_numpy()
        repeats[c] = [(int(s), int(e)) for s, e in intervals]
        logging.info(f"[repeats] loaded {len(repeats[c])} intervals for {c}")
    return repeats


def _window_overlaps_repeats(
    chrom: str,
    start: int,
    end: int,
    repeats: Dict[str, List[Tuple[int, int]]]
) -> bool:
    """
    Returns True if [start, end) overlaps any repeat interval on this chromosome.
    """
    intervals = repeats.get(chrom)
    if not intervals:
        return False
    for rs, re in intervals:
        # overlap if not completely to the left or right
        if not (end <= rs or start >= re):
            return True
    return False


# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- config ----------------
DEFAULT_CATEGORIES = [
    "introns",
    "UCNE",
    "vista_enhancer",
    "coding_regions",   # falls back to exons if empty
]

PLOT_PALETTE = "tab10"

# ---------------- helpers ----------------
def _roi_span(info: Dict) -> Optional[Tuple[int, int]]:
    fs = int(info["feature_start_in_window"])
    fe = int(info["feature_end_in_window"])
    if fe <= fs:
        return None
    return fs, fe

def _find_beds_for_category(category: str, root="/home/mica/gamba/data_processing/data/regions") -> List[str]:
    p = os.path.join(root, category, "*.bed")
    files = glob.glob(p)
    # fallback for coding
    if category == "coding_regions" and len(files) == 0:
        files = glob.glob(os.path.join(root, "exons", "*.bed"))
    return files

def _load_regions(category: str, genome: Fasta, bw: pyBigWig.pyBigWig) -> List[Dict]:
    beds = _find_beds_for_category(category)
    out = []
    for bf in beds:
        out.extend(load_bed_file(bf, category, genome, bw))
    return out

def _sample_regions(regions: List[Dict], k: int, seed: int) -> List[Dict]:
    if len(regions) <= k:
        return regions
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(regions), size=k, replace=False)
    return [regions[i] for i in idx]

def _extract_window_with_fixed_context(
    genome: Fasta,
    bigwig_file: str,
    region: Dict,
    context_size: int = 1000,
    predict_size: int = 1024,
) -> Optional[Dict]:
    """
    Extract a window with FIXED context size for consistent evaluation.
    
    Window structure:
    - Total window: context_size + max(feature_len, predict_size)
    - Positions [0, context_size): upstream context (don't predict)
    - Positions [context_size, context_size + feature_len): actual feature (PREDICT HERE)
    
    Key: We only predict/correlate on the ACTUAL FEATURE, not padding beyond it.
    """
    chrom = region["chrom"]
    feature_start = int(region["start"])
    feature_end = int(region["end"])
    
    try:
        chrom_length = len(genome[chrom])
    except KeyError:
        logging.debug(f"Chromosome {chrom} not found in genome")
        return None
    
    feature_len = feature_end - feature_start
    if feature_len <= 0:
        return None
    
    # For features longer than predict_size, truncate to last predict_size bp
    if feature_len > predict_size:
        feature_start = feature_end - predict_size
        feature_len = predict_size
    
    # Build window: context_size bp upstream + feature
    # This gives us variable window sizes, but we'll pad to a consistent size
    min_window = context_size + feature_len
    total_window = context_size + predict_size  # For padding
    
    window_start = max(0, feature_start - context_size)
    window_end = feature_end
    
    # Adjust for chromosome boundaries
    if window_start == 0:
        # Can't get full upstream context, extend downstream if possible
        window_end = min(chrom_length, min_window)
    if window_end == chrom_length:
        # At chromosome end, shift window left
        window_start = max(0, window_end - min_window)
    
    window_len = window_end - window_start
    if window_len < feature_len:
        return None  # Can't even fit the feature
    
    # Extract sequence and scores
    try:
        seq = str(genome[chrom][window_start:window_end].seq).upper()
        
        with pyBigWig.open(bigwig_file) as bw:
            scores_raw = bw.values(chrom, window_start, window_end)
            # Replace None/NaN with 0.0 to prevent NaN loss
            scores = np.array([s if s is not None and np.isfinite(s) else 0.0 
                             for s in scores_raw], dtype=np.float32)
        
        if len(seq) != len(scores):
            return None
        
        # Calculate where feature is in the extracted window
        fs_in_window = feature_start - window_start
        fe_in_window = feature_end - window_start
        
        # Pad window to total_window for consistent batching
        if window_len < total_window:
            pad_len = total_window - window_len
            # Pad at the beginning (adds more upstream context)
            seq = "N" * pad_len + seq
            scores = np.concatenate([np.zeros(pad_len, dtype=np.float32), scores])
            # Adjust feature position
            fs_in_window += pad_len
            fe_in_window += pad_len
        
        # CRITICAL: prediction/evaluation region should be EXACTLY the feature
        # Not the full [context_size, context_size+predict_size) window
        # This ensures we only correlate on the actual feature, not random flanking sequence
        pred_start = fs_in_window
        pred_end = fe_in_window
        
        # Sanity checks
        if not (0 <= pred_start < pred_end <= len(seq)):
            logging.warning(
                f"Invalid feature span for {chrom}:{feature_start}-{feature_end}: "
                f"pred=[{pred_start}, {pred_end}), window_len={len(seq)}"
            )
            return None
        
        # Verify feature is in the prediction zone (after context)
        # For autoregressive models, feature must be after context
        if pred_start < context_size:
            # Feature starts in context region - need to shift window
            shift_needed = context_size - pred_start
            if window_start >= shift_needed:
                # Can shift left
                window_start -= shift_needed
                window_end -= shift_needed
                # Re-extract
                seq = str(genome[chrom][window_start:window_end].seq).upper()
                with pyBigWig.open(bigwig_file) as bw:
                    scores_raw = bw.values(chrom, window_start, window_end)
                    scores = np.array([s if s is not None and np.isfinite(s) else 0.0 
                                     for s in scores_raw], dtype=np.float32)
                
                # Recalculate feature position
                fs_in_window = feature_start - window_start
                fe_in_window = feature_end - window_start
                
                # Repad if needed
                if len(seq) < total_window:
                    pad_len = total_window - len(seq)
                    seq = "N" * pad_len + seq
                    scores = np.concatenate([np.zeros(pad_len, dtype=np.float32), scores])
                    fs_in_window += pad_len
                    fe_in_window += pad_len
                
                pred_start = fs_in_window
                pred_end = fe_in_window
            else:
                # Can't shift enough, skip this region
                logging.debug(
                    f"Skipping {chrom}:{feature_start}-{feature_end}: "
                    f"insufficient upstream context"
                )
                return None
        
        return {
            "chrom": chrom,
            "start": window_start,
            "end": window_end,
            "sequence": seq,
            "scores": scores,
            "feature_start_in_window": pred_start,  # ACTUAL feature start
            "feature_end_in_window": pred_end,       # ACTUAL feature end
            "original_start": feature_start,
            "original_end": feature_end,
            "category": region.get("category", region.get("label", "unknown")),
            "feature_id": region.get("feature_id", f"{chrom}:{feature_start}-{feature_end}"),
            "original_feature_len": feature_len,
        }
        
    except Exception as e:
        logging.debug(f"Failed to extract {chrom}:{window_start}-{window_end}: {e}")
        return None

def _replace_nan_scores_with_zero(scores: np.ndarray) -> np.ndarray:
    """Replace NaN phyloP scores with 0.0 to prevent NaN loss."""
    scores = scores.copy()
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores[nan_mask] = 0.0
    return scores

def _collect_contexts_using_extract_context(
    bigwig_file: str,
    genome: Fasta,
    regions: List[Dict],
    model_type: str,  # "gamba" or "caduceus"
) -> List[Dict]:
    """
    Use extract_context directly - it already knows how to window properly
    for gamba (asymmetric) vs caduceus (symmetric).
    """
    valid = []
    
    for r in regions:
        # Let extract_context do ALL the windowing logic
        ctx = extract_context(bigwig_file, r, genome, model_type=model_type)
        
        if not ctx or "sequence" not in ctx:
            continue
            
        if "scores" not in ctx or ctx["scores"] is None:
            continue
        
        # extract_context already set feature_start_in_window and feature_end_in_window
        # These mark the ACTUAL FEATURE within the window
        # Just pass through the context as-is
        ctx["category"] = r.get("category", r.get("label", "unknown"))
        ctx["feature_id"] = r.get("feature_id", f"{r['chrom']}:{r['start']}-{r['end']}")
        
        valid.append(ctx)
    
    return valid

def apply_effective_region_mask(
    labels: torch.Tensor,                      # (B, 2, T): [:,0,:]=seq labels, [:,1,:]=cons labels
    feature_spans: list[tuple[int, int]],      # per-sample (fs, fe) in *token* indices (already shifted for [START] if needed)
    is_mlm: bool,                              # True for Caduceus (MLM), False for Gamba (AR)
    last_k: int = 1024,
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


def predict_scores_batched(model, tokenizer, tokenized, regions, batch_size=8, device=None,
                           model_type="gamba", training_task="dual", last_k=1024):
    """Run predictions on sampled regions with masking applied only over the last last_k bp of feature region.
    Returns:
      all_predictions: per-example CONS loss proxy (MSE over ROI) or NaN
      all_true_scores: list of 1D true phyloP arrays per region (full window)
      region_info:     list of dicts with metadata incl. ROI spans
      all_seq_predictions: per-example CE over ROI
      all_true_seqs:   list of token arrays per region
      all_pred_scores: list of 1D per-base predicted cons scores per region (full window length)
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

    logging.info(f"Running predictions on {len(regions)} regions with batch size {batch_size}, last_k={last_k}...")

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
                sequence_tokens = region['sequence']
            else:
                sequence_tokens = tokenizer.tokenizeMSA(region['sequence'])
            scores = region['scores']
            fs = region.get('feature_start_in_window', 0)
            fe = region.get('feature_end_in_window', len(scores))

            batch_inputs.append((sequence_tokens, scores))
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

        if not batch_inputs:
            continue

        # ---------------- Gamba ----------------
        if model_type == "gamba":
            inputs, labels = collator(batch_inputs)  # (B,2,T)
            inputs, labels = inputs.to(device), labels.to(device)

            # shift ROI for [START]
            feature_spans = [(int(m["feature_start_in_window"]) + 1,
                              int(m["feature_end_in_window"])   + 1)
                             for m in batch_region_info]

            labels = apply_effective_region_mask(labels, feature_spans,
                                                 is_mlm=False, last_k=last_k)

            with torch.no_grad():
                out = model(inputs, labels)  # expects dict-like

            seq_logits = out["seq_logits"].float() if "seq_logits" in out else None                # (B,T,V)
            cons_pred  = out["scaling_logits"].float() if "scaling_logits" in out else None    # (B,T,2) if present

            # CE per-example (AR shift)
            if seq_logits is not None:
                ce_labels = labels[:, 0, :].long()
                logit_shift = seq_logits[:, :-1, :]
                label_shift = ce_labels[:, 1:]
                mask_shift  = label_shift.ne(-100).float()
                ce_tok = F.cross_entropy(
                    logit_shift.reshape(-1, logit_shift.size(-1)),
                    label_shift.reshape(-1),
                    reduction="none"
                ).view(label_shift.size())
                ce_per_ex = _masked_mean_per_row(ce_tok, mask_shift, dim=1)
                all_seq_predictions.extend(ce_per_ex.detach().cpu().tolist())
            else:
                all_seq_predictions.extend([float("nan")] * inputs.size(0))

            # CONS MSE per-example + per-base predictions
            if cons_pred is not None:
                cons_mean = cons_pred[..., 0].float()  # (B,T)
                cons_tgt  = labels[:, 1, :].float()    # (B,T)
                cons_mask = cons_tgt.ne(-100).float()
                mse_tok   = (cons_mean - cons_tgt).pow(2)
                mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)
                all_predictions.extend(mse_per_ex.detach().cpu().tolist())

                # collect per-base predictions BEFORE masking alignment fix below
                # remove the +1 START shift to align to original window length
                cons_np = cons_mean.detach().cpu().numpy()   # (B,T)
                # drop first token to undo +1 if your tokenizer prepended START
                cons_np = cons_np[:, 1:]                     # (B, T-1)
                for k in range(cons_np.shape[0]):
                    all_pred_scores.append(cons_np[k].astype(np.float32))
            else:
                all_predictions.extend([float("nan")] * inputs.size(0))
                # still append NaN arrays with expected length to preserve alignment
                T = inputs.size(2) - 1  # drop START
                for _ in range(inputs.size(0)):
                    all_pred_scores.append(np.full(T, np.nan, dtype=np.float32))


        # ---------------- Caduceus ----------------
        elif model_type == "caduceus":
            raw_spans = [(r["feature_start_in_window"], r["feature_end_in_window"])
                         for r in batch_region_info]

            # build batch; allow collator(region=…) signature
            try:
                batch = collator(batch_inputs, region=raw_spans)
            except TypeError:
                batch = collator(batch_inputs, region=raw_spans)

            sequence_input = batch[0][:, 0, :].long().to(device)
            labels_pack    = batch[1].to(device)  # (B,2,T)

            # shift for [START]
            feature_spans_shifted = [(fs + 1, fe + 1) for (fs, fe) in raw_spans]
            labels_pack = apply_effective_region_mask(
                labels_pack, feature_spans_shifted, is_mlm=True, last_k=last_k
            )

            with torch.no_grad():
                outputs = model(input_ids=sequence_input, return_dict=True)

            logits = outputs["logits"].float() if "logits" in outputs else None  # (B,T,V)
            ce_labels = labels_pack[:, 0, :].long()
            cons_tgt  = labels_pack[:, 1, :].float()

            # CE per-example (MLM)
            if logits is not None:
                ce_tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ce_labels.reshape(-1),
                    reduction="none"
                ).view(ce_labels.size())
                ce_mask = ce_labels.ne(-100).float()
                ce_per_ex = _masked_mean_per_row(ce_tok, ce_mask, dim=1)
                all_seq_predictions.extend(ce_per_ex.detach().cpu().tolist())
            else:
                all_seq_predictions.extend([float("nan")] * sequence_input.size(0))

            # CONS per-base predictions if exposed
            if "scaling_logits" in outputs:
                cons_mean = outputs["scaling_logits"][..., 0].float()  # (B,T)
                cons_mask = cons_tgt.ne(-100).float()
                mse_tok   = (cons_mean - cons_tgt).pow(2)
                mse_per_ex = _masked_mean_per_row(mse_tok, cons_mask, dim=1)
                all_predictions.extend(mse_per_ex.detach().cpu().tolist())

                cons_np = cons_mean.detach().cpu().numpy()
                cons_np = cons_np[:, 1:]  # drop START to align to original window
                for k in range(cons_np.shape[0]):
                    all_pred_scores.append(cons_np[k].astype(np.float32))
            else:
                all_predictions.extend([float("nan")] * sequence_input.size(0))
                T = sequence_input.size(1) - 1  # drop START
                for _ in range(sequence_input.size(0)):
                    all_pred_scores.append(np.full(T, np.nan, dtype=np.float32))

    return all_predictions, all_true_scores, region_info, all_seq_predictions, all_true_seqs, all_pred_scores


def _get_predictions_for_contexts(
    model,
    tokenizer,
    tokenized: bool,
    contexts: List[Dict],
    batch_size: int,
    device,
    model_type: str,
    model_label: str,
    training_task: str,
    last_k: int = 1024,
) -> List[np.ndarray]:
    """Returns per-base predictions on ORIGINAL phyloP-240 scale."""
    logging.info(f"Running predict_scores_batched with last_k={last_k} to obtain predicted per-base phyloP…")

    
    out = predict_scores_batched(
        model, tokenizer, tokenized, contexts,
        batch_size=batch_size, device=device,
        model_type=model_type, training_task=training_task,
        last_k=last_k
    )

    if isinstance(out, (list, tuple)) and len(out) == 6:
        _, _, region_info, _, _, all_pred_scores = out
    elif isinstance(out, (list, tuple)) and len(out) == 5:
        _, _, region_info, _, _ = out
        all_pred_scores = [ri.get("pred_scores", None) for ri in region_info]
    else:
        raise RuntimeError("Unexpected return signature from predict_scores_batched")

    if all_pred_scores is None or not any(s is not None for s in all_pred_scores):
        raise RuntimeError("No per-base predictions available. Ensure predict_scores_batched returns all_pred_scores.")

    preds: List[np.ndarray] = []
    for ctx, ps in zip(contexts, all_pred_scores):
        if ps is None:
            # keep alignment but avoid None in list; fill with NaNs of truth length
            L = len(ctx["scores"])
            preds.append(np.full(L, np.nan, dtype=np.float32))
            continue

        arr = np.asarray(ps, dtype=np.float32).squeeze()
        L = len(ctx["scores"])
        if arr.size != L:
            arr = arr[:L] if arr.size > L else np.concatenate([arr, np.full(L - arr.size, np.nan, dtype=np.float32)])
        preds.append(arr)

    return preds


def _pearson_ignore_nan(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(a[mask], b[mask])
    return float(r)

def _spearman_ignore_nan(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r)

def _fit_slope_intercept(x, y) -> Tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    return float(slope), float(intercept)


def _run_group(
    *,
    name: str,
    group_name: str,
    contexts: List[Dict],
    model,
    tokenizer,
    tokenized: bool,
    batch_size: int,
    device,
    model_type: str,
    model_label: str,
    model_id: str,
    training_task: str,
    outdir: Path,
    last_k: int = 1024,
):
    if not contexts:
        logging.warning(f"[{name}] group={group_name}: no contexts, skipping.")
        return

    # get per-base predictions
    pred_scores = _get_predictions_for_contexts(
        model,
        tokenizer,
        tokenized,
        contexts,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
        model_label=model_label,
        training_task=training_task,
        last_k=last_k,
    )

    # assemble per-region metrics
    rows_region = []
    rows_poscorr = []
    kept = 0

    for ctx, yhat in zip(contexts, pred_scores):
        if yhat is None:
            continue
        span = _roi_span(ctx)
        if span is None:
            continue
        fs, fe = span
        y_true_all = np.asarray(ctx["scores"], dtype=np.float32)
        if y_true_all.size == 0:
            continue

        y_true = y_true_all[fs:fe]
        y_pred = np.asarray(yhat, dtype=np.float32)[fs:fe]

        m_true = float(np.nanmean(y_true)) if np.isfinite(y_true).any() else np.nan
        m_pred = float(np.nanmean(y_pred)) if np.isfinite(y_pred).any() else np.nan

        rows_region.append({
            "model_id": model_id,
            "model_label": model_label,
            "category": ctx["category"],
            "chrom": ctx.get("chrom"),
            "start": int(ctx.get("start", -1)),
            "end": int(ctx.get("end", -1)),
            "feature_start_in_window": int(fs),
            "feature_end_in_window": int(fe),
            "roi_len": int(fe - fs),
            "mean_true_phyloP": m_true,
            "mean_pred_phyloP": m_pred,
        })

        r_pos = _pearson_ignore_nan(y_true, y_pred)
        rows_poscorr.append({
            "model_id": model_id,
            "model_label": model_label,
            "category": ctx["category"],
            "chrom": ctx.get("chrom"),
            "start": int(ctx.get("start", -1)),
            "end": int(ctx.get("end", -1)),
            "feature_start_in_window": int(fs),
            "feature_end_in_window": int(fe),
            "roi_len": int(fe - fs),
            "pos_corr_pearson": r_pos,
            "n_valid": int(np.sum(~(np.isnan(y_true) | np.isnan(y_pred)))),
        })
        kept += 1

    logging.info(f"[{name}] group={group_name}: kept {kept} regions with predictions and truth.")

    df_region = pd.DataFrame(rows_region)
    df_pos    = pd.DataFrame(rows_poscorr)

    outdir.mkdir(parents=True, exist_ok=True)

    # file tag
    tag = f"{name}_{model_label}_{group_name}"

    # save tables
    df_region.to_parquet(outdir / f"{tag}_region_rate.parquet", index=False)
    df_region.to_csv(    outdir / f"{tag}_region_rate.csv",        index=False)
    df_pos.to_parquet(  outdir / f"{tag}_position_rate.parquet",   index=False)
    df_pos.to_csv(      outdir / f"{tag}_position_rate.csv",       index=False)

    # region-level scatter + correlation
    mask = ~(df_region["mean_true_phyloP"].isna() | df_region["mean_pred_phyloP"].isna())
    if mask.sum() >= 3:
        x = df_region.loc[mask, "mean_true_phyloP"].to_numpy(dtype=np.float32)
        y = df_region.loc[mask, "mean_pred_phyloP"].to_numpy(dtype=np.float32)
        r_p = _pearson_ignore_nan(x, y)
        r_s = _spearman_ignore_nan(x, y)
        slope, intercept = _fit_slope_intercept(x, y)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_region.loc[mask],
            x="mean_true_phyloP", y="mean_pred_phyloP",
            hue="category", palette=PLOT_PALETTE, s=16, alpha=0.7
        )
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        grid = np.linspace(lo, hi, 200, dtype=np.float32)
        plt.plot(grid, grid, linewidth=1, linestyle="--", label="y = x")
        if np.isfinite(slope) and np.isfinite(intercept):
            plt.plot(grid, slope*grid + intercept, linewidth=1, linestyle=":", label=f"fit: y={slope:.2f}x+{intercept:.2f}")

        plt.title(f"Region rate correlation — {model_label}\nPearson r={r_p:.3f}  Spearman ρ={r_s:.3f}")
        plt.xlabel("Observed mean phyloP per region")
        plt.ylabel("Predicted mean phyloP per region")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / f"{tag}_region_rate_scatter.png", dpi=300)
        plt.close()

        summary = {
            "model_id": model_id,
            "model_label": model_label,
            "pearson_r": r_p,
            "spearman_rho": r_s,
            "slope": slope,
            "intercept": intercept,
            "n_regions": int(mask.sum()),
        }
        with open(outdir / f"{tag}_region_rate_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # position-level correlation distribution by category
    if not df_pos.empty:
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=df_pos, x="category", y="pos_corr_pearson",
            inner="quartile", palette=PLOT_PALETTE, cut=0
        )
        plt.title(f"Distribution of position-rate correlations — {model_label}")
        plt.xlabel("")
        plt.ylabel("Pearson r within ROI (per region)")
        plt.ylim(-1.0, 1.0)
        plt.tight_layout()
        plt.savefig(outdir / f"{tag}_position_rate_violin.png", dpi=300)
        plt.close()

        agg = (
            df_pos.groupby("category")
                .agg(
                    n_regions=("pos_corr_pearson", "size"),
                    mean_r=("pos_corr_pearson", "mean"),
                    sd_r=("pos_corr_pearson", "std"),
                )
                .reset_index()
        )
        agg.to_csv(outdir / f"{tag}_position_rate_category_summary.csv", index=False)


def compute_region_and_position_correlations(
    bigwig_file: str,
    genome_fasta: str,
    data_dir: str,
    checkpoint_dir: Optional[str],
    config_fpath: Optional[str],
    output_dir: str,
    categories: List[str],
    per_category_n: int,
    batch_size: int,
    last_step: int,
    training_chromosomes: Optional[List[str]],
    test_chromosomes: Optional[List[str]],
    model_type: Optional[str],
    training_task: Optional[str],
    baseline: str,
    model_label: str,
    seed: int = 1337,
    context_size: int = 1000,
    last_k: int = 1024,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # load model once
    if baseline == "none":
        if model_type is None or training_task is None:
            raise SystemExit("When --baseline=none, you must provide --model_type and --training_task.")
        model, tokenizer = load_model(
            checkpoint_dir,
            config_fpath,
            last_step=last_step,
            device=device,
            training_task=training_task,
            model_type=model_type,
        )
        model_id = f"{model_type}_{training_task}_step{last_step}"
    else:
        raise SystemExit("This script evaluates predicted vs observed phyloP. Use --baseline=none for trained models.")

    # ------------- mode 1: sample directly from dataset splits -------------
    if len(categories) == 1 and categories[0].lower() == "random_from_dataset":
        window_len = context_size + last_k  # e.g., 1000 + 1024 = 2024
        target_n  = per_category_n * 10 # N per split

        for split_name in ["train", "test"]:
            logging.info(f"[dataset_sample] sampling from split={split_name}")
            ctxs = _sample_random_regions_from_dataset(
                data_dir=data_dir,
                split=split_name,
                n=target_n,
                window_len=window_len,
                seed=seed if split_name == "train" else seed + 1,
            )
            
            # Fix feature spans for the sampled contexts
            for ctx in ctxs:
                ctx["feature_start_in_window"] = context_size
                ctx["feature_end_in_window"] = window_len
            
            logging.info(f"[dataset_sample] split={split_name} contexts: {len(ctxs)}")

            _run_group(
                name="dataset_random",
                group_name=split_name,
                contexts=ctxs,
                model=model,
                tokenizer=tokenizer,
                tokenized=True,  # sequences already preprocessed
                batch_size=batch_size,
                device=device,
                model_type=model_type,
                model_label=model_label,
                model_id=model_id,
                training_task=training_task,
                outdir=outdir,
                last_k=last_k,
            )
        return

    # ------------- mode 2: bed-based regions / genome random -------------
    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)

    # which chromosome groups to use
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test":     test_chromosomes,
        }
    else:
        chromosomes = list(genome.keys())
        chromosome_groups = {"all": chromosomes}

    for group_name, chroms in chromosome_groups.items():
        all_contexts: List[Dict] = []

        if len(categories) == 1 and categories[0].lower() == "random_sample":
            name = "random_sample"
            rng = np.random.default_rng(seed)
            window_len = context_size + last_k  # e.g., 1000 + 1024 = 2024
            target_n = 10_000

            chrom_list = chroms if chroms else list(genome.keys())

            caps = []
            valid_chroms = []
            for c in chrom_list:
                try:
                    L = len(genome[c])
                except KeyError:
                    continue
                cap = max(0, L - window_len + 1)
                if cap > 0:
                    valid_chroms.append(c)
                    caps.append(cap)

            if not caps:
                raise SystemExit("[random_sample] no chromosomes with sufficient length.")

            repeats_by_chrom = _load_repeats_per_chrom(valid_chroms)

            caps = np.asarray(caps, dtype=np.float64)
            probs = caps / caps.sum()
            counts = rng.multinomial(target_n, probs)

            sampled_regions: List[Dict] = []
            for c, n_c, cap_c in zip(valid_chroms, counts, caps.astype(int)):
                if n_c == 0:
                    continue

                n_kept = 0
                tries = 0
                max_tries = n_c * 10

                while n_kept < n_c and tries < max_tries:
                    s = int(rng.integers(0, cap_c, endpoint=False))
                    e = s + window_len
                    tries += 1

                    if _window_overlaps_repeats(c, s, e, repeats_by_chrom):
                        continue

                    sampled_regions.append({
                        "chrom": c,
                        "start": s,
                        "end": e,
                        "category": "random_sample",
                    })
                    n_kept += 1

                if n_kept < n_c:
                    logging.warning(
                        f"[random_sample] only obtained {n_kept}/{n_c} non-repeat windows for {c} "
                        f"after {tries} attempts; repeats may be dense."
                    )

            logging.info(f"[random_sample] drew {len(sampled_regions)} non-repeat windows of {window_len} bp")

            ctxs = _collect_contexts_using_extract_context(
                bigwig_file,      
                genome,           
                sampled_regions,
                model_type,
            )
            logging.info(f"[random_sample] valid contexts: {len(ctxs)}")
            all_contexts.extend(ctxs)

        else:
            name = "by_category"
            for cat in categories:
                logging.info(f"[{cat}] loading regions…")
                regs = _load_regions(cat, genome, bw)
                if chroms:
                    regs = [r for r in regs if r.get("chrom") in chroms]
                if not regs:
                    logging.warning(f"[{cat}] no regions found")
                    continue
                for r in regs:
                    r["category"] = cat
                sampled = _sample_regions(regs, per_category_n, seed)
                ctxs = _collect_contexts_using_extract_context(
                    bigwig_file,
                    genome,
                    sampled,
                    model_type=model_type, 
                )
                logging.info(f"[{cat}] sampled {len(sampled)} -> valid {len(ctxs)}")
                all_contexts.extend(ctxs)

        _run_group(
            name=name,
            group_name=group_name,
            contexts=all_contexts,
            model=model,
            tokenizer=tokenizer,
            tokenized=False,  # sequences need tokenization inside predict_scores_batched
            batch_size=batch_size,
            device=device,
            model_type=model_type,
            model_label=model_label,
            model_id=model_id,
            training_task=training_task,
            outdir=outdir,
            last_k=last_k,
        )

    bw.close()


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Region rate correlation and Position-rate correlation for predicted vs observed phyloP."
    )
    parser.add_argument("--bigwig_file", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    parser.add_argument("--genome_fasta", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa")
    parser.add_argument("--data_dir", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian")
    parser.add_argument("--output_dir", type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/rate_correlations/")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/mica/gamba/")
    parser.add_argument("--config_fpath", type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json")
    parser.add_argument("--per_category_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--last_step", type=int, default=44000)
    parser.add_argument("--training_chromosomes", type=str, nargs="+", 
                        default=["chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                                "chr11", "chr12", "chr13", "chr14", "chr15", "chr17", "chr18", 
                                "chr19", "chr20", "chr21", "chrX"])
    parser.add_argument("--test_chromosomes", type=str, nargs="+", 
                        default=["chr2", "chr22", "chr16", "chr3"])
    parser.add_argument("--model_type", type=str, choices=["gamba","caduceus"], default="gamba")
    parser.add_argument("--training_task", type=str, choices=["dual","cons_only","seq_only"], default="dual")
    parser.add_argument("--baseline", type=str, choices=["none"], default="none",
                        help="Use trained model only; baselines without predictions are unsupported here.")
    parser.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--context_size", type=int, default=1024,
                        help="Context size (bp) before prediction region")
    parser.add_argument("--last_k", type=int, default=1024,
                        help="Prediction region size (bp)")

    args = parser.parse_args()
    
    # model_label should be: Short tag for plots/files, e.g., seq+cons | cons-only | seq-only | seq-2-seq
    if args.training_task == "dual":
        model_label = "seq+cons" + str(args.last_step)
    elif args.training_task == "cons_only":
        model_label = "cons-only" + str(args.last_step)
    elif args.training_task == "seq_only":
        model_label = "seq-only" + str(args.last_step)

    # Derive checkpoint_dir 
    if args.model_type == "gamba":
        checkpoint_dir = args.checkpoint_dir + "/clean_dcps/CCP/"
    else:
        checkpoint_dir = args.checkpoint_dir + "/clean_caduceus_dcps/allPOSMLM"
        model_label = model_label + "-seq2seq" + str(args.last_step)

    # add model label info to output_dir
    args.output_dir = os.path.join(args.output_dir, args.model_type, model_label)

    compute_region_and_position_correlations(
        bigwig_file=args.bigwig_file,
        genome_fasta=args.genome_fasta,
        data_dir=args.data_dir,
        checkpoint_dir=checkpoint_dir,
        config_fpath=args.config_fpath,
        output_dir=args.output_dir,
        categories=args.categories,
        per_category_n=args.per_category_n,
        batch_size=args.batch_size,
        last_step=args.last_step,
        training_chromosomes=args.training_chromosomes,
        test_chromosomes=args.test_chromosomes,
        model_type=args.model_type,
        training_task=args.training_task,
        baseline=args.baseline,
        model_label=model_label,
        seed=args.seed,
        context_size=args.context_size,
        last_k=args.last_k,
    )
    logging.info("Done.")

if __name__ == "__main__":
    main()