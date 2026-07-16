#!/usr/bin/env python3
"""
unicode_properties scoring wrapper (batch support, no final_model needed)

- Loads base model + LoRA adapter (PEFT) in-memory (no merge to disk).
- Reads propagation_inputs.csv (column: text; first row is k).
- Outputs per-token loss CSV in the same format as the original pipeline:
    row0: [k]
    row1..: token losses for each document (list of floats)

NOTE:
- This script keeps the SAME single-text loss function `_calculate_loss_str`
  (imported from local score.py), but processes documents in batches for
  better throughput and less Python/tqdm overhead.
- Forward pass is still invoked per document if `_calculate_loss_str` is per-doc,
  but batching reduces overhead and allows future extension.
"""

import argparse
import csv
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure we can import local score.py (same directory)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

# We reuse loss computation from your local score.py
from score import _calculate_loss_str  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Score unicode_properties using base model + LoRA adapter (batch support).")

    p.add_argument("--score_type", choices=["loss_per_token"], default="loss_per_token",
                   help="unicode_properties only supports loss_per_token")

    p.add_argument("--base_model_id", required=True,
                   help="Base model HF id or local path, e.g. meta-llama/Llama-2-7b-hf")

    p.add_argument("--adapter_path", required=True,
                   help="LoRA adapter directory (final_adapter / final_model adapter dir)")

    p.add_argument("--path_to_inputs", required=True,
                   help="propagation_inputs.csv path (must have column 'text'; first row is k)")

    p.add_argument("--output_score_path", required=True,
                   help="output CSV path for scores")

    p.add_argument("--unicode_max_length", type=int, default=4096,
                   help="score only last N tokens (truncation_side=left)")

    p.add_argument("--attn_impl", type=str, default="flash_attention_2",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="attention implementation; eager is most stable")

    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of documents to process per outer loop batch (default=32)")

    return p.parse_args()


def load_model_and_tokenizer(base_model_id: str, adapter_path: str, attn_impl: str):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"  # keep the end for long docs

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if device_str == "cuda" else torch.float32,
        attn_implementation=attn_impl,
        device_map={"": device_str} if device_str == "cuda" else None,
    )
    model.config.use_cache = False

    # Load LoRA adapter (no merge needed for inference)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tok, device


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        attn_impl=args.attn_impl,
    )

    df = pd.read_csv(args.path_to_inputs)
    if "text" not in df.columns:
        raise ValueError(f"Input CSV must contain column 'text'. Got: {list(df.columns)}")

    # The first row is k (number of test documents), stored as a string/number in column 'text'
    k_raw = df["text"].iloc[0]
    try:
        k = int(float(str(k_raw).strip()))
    except Exception as e:
        raise ValueError(f"First row of propagation_inputs.csv should be k, but got: {k_raw}") from e

    out_dir = os.path.dirname(os.path.abspath(args.output_score_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    texts = df["text"].iloc[1:].tolist()
    # normalize to string early
    texts = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in texts]

    with open(args.output_score_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)

        # Row0: write k
        w.writerow([k])

        # Process in batches of documents
        batches = list(chunked(texts, args.batch_size))
        for batch in tqdm(batches, desc=f"Scoring docs (batch_size={args.batch_size})", unit="batch"):
            batch_rows = []
            for text in batch:
                # returns tensor shape (1, T-1) usually; we want python list
                losses = _calculate_loss_str(
                    text, model, tokenizer, device,
                    unicode_max_length=args.unicode_max_length
                ).tolist()[0]
                batch_rows.append(losses)

            # Preserve order in output (one CSV row per document)
            for losses in batch_rows:
                w.writerow(losses)

    print(f"✅ wrote scores: {args.output_score_path}")


if __name__ == "__main__":
    main()
