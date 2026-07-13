#!/usr/bin/env python3
"""Validate FineWeb NPZ split files used by the training code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tokenizers import Tokenizer


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate train/val/test NPZ files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--full-token-range-check", action="store_true")
    return parser.parse_args()


def fail(errors: List[str], message: str) -> None:
    errors.append(message)


def validate_split(path: Path, vocab_size: Optional[int], full_range: bool, errors: List[str]) -> Dict[str, object]:
    if not path.exists():
        fail(errors, f"missing file: {path}")
        return {"path": str(path), "exists": False}

    with np.load(path) as data:
        names = set(data.files)
        if names != {"tokens", "offsets"}:
            fail(errors, f"{path}: expected keys tokens/offsets, found {sorted(names)}")
            return {"path": str(path), "exists": True, "keys": sorted(names)}

        tokens = data["tokens"]
        offsets = data["offsets"]

        if tokens.ndim != 1:
            fail(errors, f"{path}: tokens must be 1D, got shape {tokens.shape}")
        if offsets.ndim != 1:
            fail(errors, f"{path}: offsets must be 1D, got shape {offsets.shape}")
        if offsets.size == 0:
            fail(errors, f"{path}: offsets must contain at least the initial zero")
        elif int(offsets[0]) != 0:
            fail(errors, f"{path}: first offset must be 0")
        if offsets.size > 0 and int(offsets[-1]) != int(tokens.shape[0]):
            fail(errors, f"{path}: final offset {int(offsets[-1])} != token count {tokens.shape[0]}")
        if offsets.size > 1 and np.any(np.diff(offsets) < 0):
            fail(errors, f"{path}: offsets must be non-decreasing")

        token_min = None
        token_max = None
        if tokens.size:
            if full_range:
                token_min = int(tokens.min())
                token_max = int(tokens.max())
            else:
                sample = tokens[: min(tokens.size, 100_000)]
                token_min = int(sample.min())
                token_max = int(sample.max())

            if token_min < 0:
                fail(errors, f"{path}: token ids must be non-negative")
            if vocab_size is not None and token_max >= vocab_size:
                fail(errors, f"{path}: token id {token_max} exceeds vocab size {vocab_size}")

        return {
            "path": str(path),
            "exists": True,
            "tokens": int(tokens.shape[0]),
            "documents": int(max(offsets.shape[0] - 1, 0)),
            "token_dtype": str(tokens.dtype),
            "offset_dtype": str(offsets.dtype),
            "sampled_token_min": token_min,
            "sampled_token_max": token_max,
        }


def main() -> None:
    args = parse_args()
    errors: List[str] = []

    vocab_size = None
    if args.tokenizer_json is not None:
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        vocab_size = int(tokenizer.get_vocab_size())

    summary = {
        "data_dir": str(args.data_dir),
        "vocab_size": vocab_size,
        "splits": {},
    }

    for split in SPLITS:
        summary["splits"][split] = validate_split(
            args.data_dir / f"{split}.npz",
            vocab_size=vocab_size,
            full_range=args.full_token_range_check,
            errors=errors,
        )

    print(json.dumps(summary, indent=2))
    if errors:
        print("\nvalidation errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nvalidation passed")


if __name__ == "__main__":
    main()
