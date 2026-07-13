#!/usr/bin/env python3
import os
import sys
import argparse
import json
import warnings
import logging
import regex as re
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# PYTHONPATH hack: make sure “utils” is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.model_utils import load_model_configs, initialize_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def _safe_token_text(t: str) -> str:
    """Make token text printable & non-empty for JSON consumers."""
    if t is None:
        return "<NULL>"
    s = t.replace("\n", "⏎").replace("\t", "⇥")
    # collapse any Unicode whitespace; if nothing remains, show a marker
    if re.sub(r"\p{Z}+", "", s) == "":
        return "␠"  # visible space marker
    return s


def _norm(s: str) -> str:
    """Light normalization to neutralize common CSV and whitespace artifacts."""
    if s is None:
        return ""
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    return s.replace("\u00A0", " ").replace("\r\n", "\n")


def compute_stats(model, tokenizer, text, temperature: float = 1.0):
    """
    Runs the model on `text` and returns:
      - nlls:  per-token negative log-likelihoods; for index i, nlls[i-1] = -log p(x_i | x_<i)
      - ent:   per-token entropies; for index i, ent[i-1] = -∑_j p_j log p_j
    """
    logger.debug(f"Computing stats for text of length {len(text)} characters")
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    ids = tokens['input_ids']
    logger.debug(f"Tokenized to {ids.size(1)} tokens")

    # Optional: detect likely overflow vs model limits
    try:
        max_ctx = getattr(model.config, "max_position_embeddings", None)
        n_tok = ids.size(1)
        if max_ctx and n_tok > max_ctx:
            logger.warning("Sequence length %d exceeds model max %d; results may be truncated/misaligned", n_tok, max_ctx)
    except Exception:
        logger.exception("Failed to check sequence length against model limits")

    with torch.no_grad():
        logits = model(**tokens).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        # Temperature-scaled log-probs for entropy computation (NLL uses original)
        if abs(temperature - 1.0) > 1e-6:
            log_probs_scaled = torch.log_softmax(logits / temperature, dim=-1)
        else:
            log_probs_scaled = log_probs

    n = ids.size(1)
    nlls, ent = [], []
    for i in range(1, n):
        tid = ids[0, i]
        nll = -log_probs[0, i-1, tid].item()  # NLL always uses original log_probs
        nlls.append(nll)

        probs = log_probs_scaled[0, i-1].exp().cpu().numpy()  # temperature-scaled for entropy
        if np.any(np.isnan(probs)) or not np.allclose(np.sum(probs), 1.0, atol=1e-3):
            logger.warning(f"Invalid probs at token position i={i}, setting entropy to 0. sum={np.sum(probs)}")
            e = 0.0
        else:
            e = -np.sum(probs * np.log(probs + 1e-12))
        ent.append(float(e))

        logger.debug(f"Token {i}: NLL={nll:.4f}, Entropy={e:.4f}")

    logger.debug(f"Computed {len(nlls)} NLL and {len(ent)} entropy values (sequence length = {n})")
    return nlls, ent


def _suffix_span_in_full_ids(full_ids, prefix_len_tokens: int, tokenizer, suffix: str,
                             max_cushion: int = 16):
    """
    Find the token span for `suffix` in the tail of FULL tokenization (prefix+prompt).
    Returns (start_postprefix, length_tokens). Uses decode verification; no substring re-tokenization.
    """
    if not suffix:
        return None, 0

    suffix_n = _norm(suffix)
    n_full = len(full_ids)

    # Backward search over the full tail; verify by decoding exact equality to suffix text.
    # Heuristic bound: allow some extra tokens to account for boundary fusion.
    max_k = min(n_full, len(suffix_n) * 4 + max_cushion)
    for k in range(1, max_k + 1):
        tail_text = tokenizer.decode(full_ids[-k:], skip_special_tokens=True)
        if tail_text == suffix_n:
            start_full = n_full - k
            start_post = start_full - prefix_len_tokens
            if start_post < 0:
                # Suffix overlaps prefix — treat as not found in post-prefix region.
                return None, 0
            return start_post, k

    return None, 0


def _build_token_stream(prefix: str,
                        prompt: str,
                        suffix: str,
                        tokenizer,
                        nlls_full: list,
                        ents_full: list):
    """
    Build a single JSON-able structure covering the FULL sequence (prefix+prompt).
    Adds is_prefix and is_suffix flags; aligns NLL/entropy with prediction_shift=1.
    """
    # Compose and tokenize the full stream
    full_text = prefix + prompt
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    full_txt = tokenizer.convert_ids_to_tokens(full_ids)

    # Token length of prefix in the SAME tokenization used for the stream
    P = len(tokenizer(prefix, add_special_tokens=False).input_ids)

    # Find suffix span directly in FULL token space and convert to post-prefix coords
    suffix_start_postprefix, S_ctx = _suffix_span_in_full_ids(full_ids, P, tokenizer, suffix)

    # Build per-token records
    tokens = []
    for i, (tid, ttext) in enumerate(zip(full_ids, full_txt)):
        pos_post = (i - P) if i >= P else None
        rec = {
            "pos_global": i,
            "pos_postprefix": pos_post,               # None for prefix tokens
            "id": int(tid),
            "text": _safe_token_text(ttext),
            "nll": (float(round(nlls_full[i-1], 6)) if i > 0 and (i-1) < len(nlls_full) else None),
            "entropy": (float(round(ents_full[i-1], 6)) if i > 0 and (i-1) < len(ents_full) else None),
            "is_prefix": 1 if i < P else 0,          # <-- highlight S1 (prefix)
            "is_suffix": 0
        }
        tokens.append(rec)

    # Mark suffix region in post-prefix space
    if suffix_start_postprefix is not None and S_ctx > 0:
        lo = suffix_start_postprefix
        hi = lo + S_ctx
        for rec in tokens:
            if rec["pos_postprefix"] is not None and lo <= rec["pos_postprefix"] < hi:
                rec["is_suffix"] = 1

    meta = {
        "prefix_len_tokens": int(P),
        "text_len_tokens": int(len(full_ids)),
        "prediction_shift": 1,
        "suffix_start_postprefix": int(suffix_start_postprefix) if suffix_start_postprefix is not None else None,
        "suffix_len_tokens_ctx": int(S_ctx),
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer))
    }

    # Optional consistency checks (disable if not wanted)
    try:
        post_len = meta["text_len_tokens"] - meta["prefix_len_tokens"]
        start = meta["suffix_start_postprefix"]; S = meta["suffix_len_tokens_ctx"]
        if start is not None:
            assert 0 <= start <= post_len
            assert 0 <= S <= (post_len - start)
            S_labeled = sum(1 for r in tokens if (r["pos_postprefix"] is not None and r["is_suffix"]))
            assert S_labeled == S, f"Label/length mismatch: labeled={S_labeled} meta={S}"
    except AssertionError as ae:
        logger.warning("Consistency check failed: %s", str(ae))

    return {"meta": meta, "tokens": tokens}


def main():
    logger.info("Starting compute_token_stats script")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Model key from config/models.yaml")
    parser.add_argument('--input_csv', required=True, help="CSV with a 'full_prompt' column (+ labels/algorithm)")
    parser.add_argument('--output_csv', required=True, help="Where to write token stats (nlls, entropies)")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for entropy softmax scaling (default: 1.0 = no scaling). T<1 sharpens distribution.")
    args = parser.parse_args()

    logger.info(f"Arguments: model={args.model}, input_csv={args.input_csv}, output_csv={args.output_csv}")

    if os.path.exists(args.output_csv):
        logger.info(f"Token stats already exist at {args.output_csv}, skipping computation")
        print(f"Token stats already exist at {args.output_csv}, skipping.")
        return

    logger.info(f"Loading model configuration for model: {args.model}")
    cfg = load_model_configs(args.model)
    logger.info("Initializing model and tokenizer")
    model, tokenizer = initialize_model(cfg)
    model.eval()
    logger.info(f"Model loaded and set to evaluation mode. Device: {model.device}")

    logger.info(f"Loading input CSV from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows from input CSV")

    df = df.dropna(subset=['full_prompt'])
    logger.info(f"After dropping rows with missing 'full_prompt': {len(df)} rows")
    df['full_prompt'] = df['full_prompt'].astype(str)

    if 'is_adversarial' in df.columns and df['is_adversarial'].dtype == object:
        logger.info("Converting 'is_adversarial' column from object to numeric")
        df['is_adversarial'] = df['is_adversarial'].map({'adversarial': 1, 'normal': 0})

    rows = []
    prefix = cfg.get('s1', '')
    logger.info(f"Using prefix: '{prefix}'")

    logger.info("Starting token stats computation for all prompts")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Computing token stats'):
        prompt_text = row['full_prompt']
        text = prefix + prompt_text
        suffix_text = row.get('suffix', '') if pd.notna(row.get('suffix', '')) else ''
        language = row.get('language', None) if 'language' in df.columns else None
        source_dataset = row.get('source_dataset', None) if 'source_dataset' in df.columns else None

        try:
            nlls, entropies = compute_stats(model, tokenizer, text, temperature=args.temperature)
            ts = _build_token_stream(prefix, prompt_text, suffix_text, tokenizer, nlls, entropies)

            out_row = {
                'full_prompt': prompt_text,
                'suffix': suffix_text,
                'token_stream': json.dumps(ts, ensure_ascii=False),
                # convenience scalar columns for quick filters/joins
                'prefix_len_tokens': ts["meta"]["prefix_len_tokens"],
                'text_len_tokens': ts["meta"]["text_len_tokens"],
                'suffix_start_postprefix': ts["meta"]["suffix_start_postprefix"],
                'suffix_len_tokens_ctx': ts["meta"]["suffix_len_tokens_ctx"],
                'is_adversarial': int(row['is_adversarial']) if pd.notna(row.get('is_adversarial')) else 0,
                'algorithm': row['algorithm'] if ('algorithm' in row and pd.notna(row['algorithm'])) else np.nan
            }
            if language is not None and pd.notna(language):
                out_row['language'] = str(language)
            if source_dataset is not None and pd.notna(source_dataset):
                out_row['source_dataset'] = str(source_dataset)

            rows.append(out_row)

            if (idx + 1) % 100 == 0:
                logger.debug(f"Processed {idx + 1}/{len(df)} prompts")

        except Exception as e:
            logger.error(f"Error processing prompt at index {idx}: {str(e)}")
            logger.debug(f"Problematic text (first 100 chars): {text[:100]}")
            continue

    logger.info(f"Successfully processed {len(rows)} prompts out of {len(df)} total")

    if len(rows) == 0:
        logger.error("No prompts were successfully processed. Exiting without creating output file.")
        return

    out_df = pd.DataFrame(rows)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving results to {args.output_csv}")
    out_df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Successfully wrote token stats for {len(out_df)} prompts to {args.output_csv}")
    print(f"Wrote token stats for {len(out_df)} prompts to {args.output_csv}")


if __name__ == '__main__':
    main()
