#!/usr/bin/env python3
"""
Reproduction script for paper #4290: BREW watermarking evaluation.

Runs BREW with paper settings (delta=6.0, BCH(31,6,7), soft watermarking,
s_max=5) on C4 with OPT-1.3B, applies 10% token-preserving synonym
substitution to watermarked texts, and computes TPR / FPR / Precision / F1.

Usage (inside the container):
    cd /repo
    python reproduce_eval.py --max-samples 100
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure /repo is on the path so that 'sage' shim and repo modules are found.
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from synonym_attack import apply_synonym_substitution


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = REPO_DIR
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "facebook" / "opt-1.3b"


# ---------------------------------------------------------------------------
# Paper-specific BREW config  (delta=6.0, soft, s_max=5, BCH(31,6,7))
# ---------------------------------------------------------------------------
PAPER_CONFIG = {
    "algorithm_name": "BREW",
    "gamma": 0.5,
    "delta": 6.0,
    "hash_key": 15485863,
    "prefix_length": 1,
    "z_threshold": 1.0,
    "bch_t": 7,
    "bch_m": 5,
    "max_shift_bit": 5,
    "scheme": "soft",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_model(model_path, hf_token):
    model_path = Path(model_path)
    if (model_path / "config.json").exists() and (model_path / "pytorch_model.bin").exists():
        print(f"Model already exists at {model_path}")
        return
    model_path.mkdir(parents=True, exist_ok=True)
    token = hf_token.strip() or None
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"Downloading facebook/opt-1.3b to {model_path} ...")
    snapshot_download(
        repo_id="facebook/opt-1.3b",
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        token=token,
    )
    print("Model download complete.")


def load_transformers_config(model_path, device, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    model.eval()

    output_embeddings = model.get_output_embeddings()
    vocab_size = int(output_embeddings.weight.shape[0]) if output_embeddings is not None else len(tokenizer)

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
        max_new_tokens=max_new_tokens,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
    )


def load_c4_samples(max_samples, offset):
    path = BASE_DIR / "dataset" / "c4" / "processed_c4.json"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))

    return [
        {
            "dataset": "c4",
            "sample_idx": offset + i,
            "prompt": row["prompt"],
            "natural_text": row["natural_text"],
        }
        for i, row in enumerate(rows)
    ]


def merge_prompt_and_continuation(prompt, continuation):
    if not prompt:
        return continuation
    if not continuation:
        return prompt
    if prompt[-1].isspace() or continuation[0].isspace() or continuation[0] in ".,;:!?)]}":
        return prompt + continuation
    return prompt + " " + continuation


def reset_watermark_state(watermark):
    if hasattr(watermark, "reset_state"):
        watermark.reset_state()
        return
    lp = getattr(watermark, "logits_processor", None)
    if lp is None:
        return
    if hasattr(lp, "codeword_queue"):
        lp.codeword_queue = []
    if hasattr(lp, "token_bit_log"):
        lp.token_bit_log = []


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_reproduction(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Download model if needed
    download_model(args.model_path, args.hf_token)

    # 2) Load model
    print("Loading model and tokenizer ...")
    tcfg = load_transformers_config(args.model_path, device, args.max_new_tokens)

    # 3) Write paper config to a temp file so AutoWatermark can load it
    config_path = BASE_DIR / "config" / "BREW_paper.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(PAPER_CONFIG, f, indent=4)

    # 4) Create BREW watermark
    watermark = AutoWatermark.load(
        "BREW",
        algorithm_config=str(config_path),
        transformers_config=tcfg,
    )

    # 5) Load samples
    samples = load_c4_samples(args.max_samples, args.sample_offset)
    print(f"Loaded {len(samples)} C4 samples")

    # 6) Run experiment
    results = []
    output_dir = BASE_DIR / "results" / "BREW_repro"
    output_dir.mkdir(parents=True, exist_ok=True)

    substitution_rate = args.substitution_rate

    for sample in samples:
        dataset_name = sample["dataset"]
        sample_idx = sample["sample_idx"]
        prompt = sample["prompt"]
        natural_text = merge_prompt_and_continuation(prompt, sample["natural_text"])

        sample_dir = output_dir / dataset_name / f"sample_{sample_idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Sample {sample_idx} ---")
        print(f"Prompt: {prompt[:100]}...")

        # Reset watermark state per sample
        reset_watermark_state(watermark)

        # (a) Generate watermarked text
        t0 = time.time()
        wm_text = watermark.generate_watermarked_text(prompt)
        gen_time = time.time() - t0
        print(f"Generated watermarked text ({len(wm_text.split())} words, {gen_time:.1f}s)")

        # (b) Apply synonym substitution to watermarked text
        attacked_wm_text = apply_synonym_substitution(
            wm_text,
            substitution_rate=substitution_rate,
            seed=args.seed + sample_idx,
        )
        n_wm_tokens = len(wm_text.split())
        n_attacked_tokens = len(attacked_wm_text.split())
        print(f"Applied {substitution_rate*100:.0f}% synonym substitution "
              f"({n_wm_tokens} -> {n_attacked_tokens} words)")

        # (c) Detect watermark on attacked watermarked text
        detect_wm = watermark.detect_watermark(prompt, attacked_wm_text)
        wm_detected = bool(detect_wm.get("is_watermarked", False))
        print(f"Watermarked (attacked) detection: {wm_detected} "
              f"(matched={detect_wm.get('matched', 0)}/{detect_wm.get('total', 0)})")

        # (d) Generate unwatermarked text
        reset_watermark_state(watermark)
        uwm_text = watermark.generate_unwatermarked_text(prompt)

        # (e) Detect watermark on unwatermarked text
        detect_uwm = watermark.detect_watermark(prompt, uwm_text)
        uwm_detected = bool(detect_uwm.get("is_watermarked", False))
        print(f"Unwatermarked detection: {uwm_detected} "
              f"(matched={detect_uwm.get('matched', 0)}/{detect_uwm.get('total', 0)})")

        # (f) Detect watermark on natural text
        detect_nat = watermark.detect_watermark(prompt, natural_text)
        nat_detected = bool(detect_nat.get("is_watermarked", False))
        print(f"Natural text detection: {nat_detected} "
              f"(matched={detect_nat.get('matched', 0)}/{detect_nat.get('total', 0)})")

        results.append({
            "sample_idx": sample_idx,
            "watermarked_detected": wm_detected,
            "watermarked_details": detect_wm,
            "unwatermarked_detected": uwm_detected,
            "unwatermarked_details": detect_uwm,
            "natural_detected": nat_detected,
            "natural_details": detect_nat,
        })

        # Save per-sample results
        with open(sample_dir / "repro_result.json", "w") as f:
            json.dump(results[-1], f, indent=4, default=str)

    # 7) Compute aggregate metrics
    n = len(results)
    tp = sum(1 for r in results if r["watermarked_detected"])
    fn = n - tp  # watermarked but not detected
    fp_uwm = sum(1 for r in results if r["unwatermarked_detected"])
    fp_nat = sum(1 for r in results if r["natural_detected"])
    fp = fp_uwm + fp_nat
    tn = 2 * n - fp  # total unwatermarked + natural = 2n

    tpr = tp / n if n > 0 else 0.0
    fpr = fp / (2 * n) if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"Samples:              {n}")
    print(f"Substitution rate:    {substitution_rate*100:.0f}%")
    print(f"TP (detected wm):     {tp}")
    print(f"FN (missed wm):       {fn}")
    print(f"FP_UWM:               {fp_uwm}")
    print(f"FP_NAT:               {fp_nat}")
    print(f"FP (total):           {fp}")
    print("-" * 60)
    print(f"TPR (Recall):         {tpr:.4f}")
    print(f"FPR:                  {fpr:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"F1:                   {f1:.4f}")
    print("=" * 60)

    # 8) Save aggregate result
    summary = {
        "paper_id": 4290,
        "num_samples": n,
        "substitution_rate": substitution_rate,
        "config": PAPER_CONFIG,
        "model": "facebook/opt-1.3b",
        "dataset": "C4",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "TPR": round(tpr, 6),
        "FPR": round(fpr, 6),
        "Precision": round(precision, 6),
        "F1": round(f1, 6),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "FP_unwatermarked": fp_uwm,
        "FP_natural": fp_nat,
        "TN": tn,
        "per_sample": results,
    }
    summary_path = output_dir / "reproduction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"\nFull results saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BREW Reproduction (Paper #4290)")
    p.add_argument("--max-samples", type=int, default=200,
                   help="Number of C4 samples to evaluate")
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--substitution-rate", type=float, default=0.10,
                   help="Synonym substitution rate (0.0 - 1.0)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_reproduction(args)
