#!/usr/bin/env python3
"""Create short, sentence-aligned prompts from FineWeb validation NPZ."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_bpe_tokenizer(path: str) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    tok.decoder = BPEDecoder(suffix="</w>")
    return tok


def iter_documents(npz_path: str):
    data = np.load(npz_path)
    tokens = data["tokens"]
    offsets = data.get("offsets")
    if offsets is not None:
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]
            yield tokens[start:end]
    else:
        yield from np.array_split(tokens, max(len(tokens) // 2048, 1))


def split_sentences(text: str) -> List[str]:
    sentences = [seg.strip() for seg in SENTENCE_SPLIT_RE.split(text) if seg.strip()]
    return sentences or [text.strip()]


def build_prompt(sentences: List[str], min_chars: int, max_chars: int) -> Dict[str, str]:
    prompt_parts: List[str] = []
    ref_parts: List[str] = []
    char_count = 0
    idx = 0

    while idx < len(sentences) and char_count < min_chars:
        prompt_parts.append(sentences[idx])
        char_count += len(sentences[idx]) + 1
        idx += 1

    while idx < len(sentences) and len(" ".join(ref_parts)) < min_chars:
        ref_parts.append(sentences[idx])
        idx += 1
        if len(" ".join(prompt_parts)) >= max_chars:
            break

    prompt_text = " ".join(prompt_parts).strip()
    reference_text = " ".join(ref_parts).strip()

    if len(prompt_text) > max_chars:
        prompt_text = prompt_text[:max_chars].rsplit(" ", 1)[0].strip()

    return {"prompt": prompt_text, "reference": reference_text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create short prompts from FineWeb val")
    parser.add_argument("--npz", required=True, help="Path to FineWeb val npz")
    parser.add_argument("--tokenizer", required=True, help="Path to BPE tokenizer json")
    parser.add_argument("--num-prompts", type=int, default=12, help="Number of prompts to output")
    parser.add_argument("--min-chars", type=int, default=80, help="Minimum characters per prompt")
    parser.add_argument("--max-chars", type=int, default=220, help="Maximum characters per prompt")
    parser.add_argument("--output", default="short_fineweb_prompts.json", help="Output JSON path")
    args = parser.parse_args()

    tokenizer = load_bpe_tokenizer(args.tokenizer)
    results: List[Dict[str, str]] = []

    for doc_tokens in iter_documents(args.npz):
        text = tokenizer.decode(doc_tokens.tolist()).strip()
        if not text:
            continue
        sentences = split_sentences(text)
        if not sentences:
            continue
        prompt_bundle = build_prompt(sentences, args.min_chars, args.max_chars)
        if prompt_bundle["prompt"]:
            results.append(prompt_bundle)
        if len(results) >= args.num_prompts:
            break

    if not results:
        raise SystemExit("No prompts could be extracted")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} prompts to {output_path}")


if __name__ == "__main__":
    main()
