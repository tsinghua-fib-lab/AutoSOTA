#!/usr/bin/env python3
"""Prepare FineWeb-Edu token splits as train/val/test NPZ files.

The output format matches the training loader used in this repository:
each split contains a flat `tokens` array and an `offsets` array with
document boundaries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FineWeb-Edu train/val/test NPZ files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/FineWeb"))
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=Path("tok/fineweb_bpe_16000.json"),
        help="Tokenizer JSON used to encode the corpus.",
    )
    parser.add_argument("--target-tokens", type=int, default=250_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--token-dtype", default="int32", choices=("int32", "uint16"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")

    source = parser.add_argument_group("input source")
    source.add_argument("--source", choices=("fineweb", "jsonl"), default="fineweb")
    source.add_argument("--input-jsonl", type=Path, help="Local JSONL input for tests or offline runs.")
    source.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    source.add_argument("--dataset-config", default="sample-10BT")
    source.add_argument("--dataset-split", default="train")
    source.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)

    filtering = parser.add_argument_group("document filters")
    filtering.add_argument("--text-field", default="text")
    filtering.add_argument("--language-field", default="language")
    filtering.add_argument("--language-score-field", default="language_score")
    filtering.add_argument("--language", default="en")
    filtering.add_argument("--min-language-score", type=float, default=0.95)
    filtering.add_argument("--min-chars", type=int, default=200)
    filtering.add_argument("--max-chars", type=int, default=8192)
    filtering.add_argument("--min-tokens", type=int, default=50)
    filtering.add_argument("--max-documents", type=int, default=None)
    filtering.add_argument("--eot-token", default="<|endoftext|>")
    filtering.add_argument("--no-append-eot", action="store_true")

    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            yield obj


def iter_fineweb(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to stream FineWeb-Edu.") from exc

    return load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
    )


def iter_documents(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    if args.source == "jsonl":
        if args.input_jsonl is None:
            raise ValueError("--input-jsonl is required when --source jsonl is used.")
        return iter_jsonl(args.input_jsonl)
    return iter_fineweb(args)


def get_text(doc: Dict[str, object], args: argparse.Namespace) -> Optional[str]:
    language = doc.get(args.language_field)
    if language is not None and language != args.language:
        return None

    score = doc.get(args.language_score_field)
    if score is not None and float(score) < args.min_language_score:
        return None

    value = doc.get(args.text_field)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if len(text) < args.min_chars or len(text) > args.max_chars:
        return None
    return text


def encode_document(tokenizer: Tokenizer, text: str, args: argparse.Namespace) -> np.ndarray:
    if not args.no_append_eot:
        text = f"{text} {args.eot_token}"
    ids = tokenizer.encode(text).ids
    return np.asarray(ids, dtype=np.dtype(args.token_dtype))


def split_doc_counts(num_docs: int, val_fraction: float, test_fraction: float) -> Dict[str, int]:
    if num_docs < 0:
        raise ValueError("num_docs must be non-negative")

    test_docs = int(num_docs * test_fraction)
    val_docs = int(num_docs * val_fraction)

    if num_docs >= 3:
        if test_fraction > 0 and test_docs == 0:
            test_docs = 1
        if val_fraction > 0 and val_docs == 0:
            val_docs = 1

    if val_docs + test_docs >= num_docs:
        overflow = val_docs + test_docs - max(num_docs - 1, 0)
        if overflow > 0:
            reduce_test = min(test_docs, overflow)
            test_docs -= reduce_test
            overflow -= reduce_test
        if overflow > 0:
            val_docs = max(0, val_docs - overflow)

    train_docs = num_docs - val_docs - test_docs
    return {"train": train_docs, "val": val_docs, "test": test_docs}


def check_output_paths(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / f"{split}.npz" for split in SPLITS if (output_dir / f"{split}.npz").exists()]
    if existing and not overwrite:
        files = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {files}. Use --overwrite to replace them.")


def collect_tokens(args: argparse.Namespace, tokenizer: Tokenizer, temp_dir: Path) -> Dict[str, object]:
    tokens_path = temp_dir / "tokens.bin"
    lengths = []
    stats = {
        "accepted_documents": 0,
        "accepted_tokens": 0,
        "skipped_filter": 0,
        "skipped_short_tokenized": 0,
    }

    with tokens_path.open("wb") as token_file:
        progress = tqdm(iter_documents(args), desc="encoding documents", unit="doc")
        for doc in progress:
            text = get_text(doc, args)
            if text is None:
                stats["skipped_filter"] += 1
                continue

            tokens = encode_document(tokenizer, text, args)
            if tokens.size < args.min_tokens:
                stats["skipped_short_tokenized"] += 1
                continue

            tokens.tofile(token_file)
            lengths.append(int(tokens.size))
            stats["accepted_documents"] += 1
            stats["accepted_tokens"] += int(tokens.size)
            progress.set_postfix(tokens=stats["accepted_tokens"], docs=stats["accepted_documents"])

            if args.max_documents is not None and stats["accepted_documents"] >= args.max_documents:
                break
            if stats["accepted_tokens"] >= args.target_tokens:
                break

    lengths_array = np.asarray(lengths, dtype=np.int64)
    np.save(temp_dir / "lengths.npy", lengths_array)
    if lengths_array.size == 0:
        raise RuntimeError("No documents passed filtering; no dataset was written.")
    return {"tokens_path": tokens_path, "lengths": lengths_array, "stats": stats}


def write_splits(
    output_dir: Path,
    tokenizer: Tokenizer,
    token_dtype: np.dtype,
    tokens_path: Path,
    lengths: np.ndarray,
    stats: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    total_tokens = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    tokens = np.memmap(tokens_path, mode="r", dtype=token_dtype, shape=(total_tokens,))
    doc_counts = split_doc_counts(len(lengths), args.val_fraction, args.test_fraction)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_splits = {}
    doc_start = 0

    for split in SPLITS:
        doc_end = doc_start + doc_counts[split]
        token_start = int(offsets[doc_start])
        token_end = int(offsets[doc_end])

        split_tokens = np.asarray(tokens[token_start:token_end], dtype=token_dtype)
        split_offsets = (offsets[doc_start : doc_end + 1] - token_start).astype(np.int64, copy=False)
        if split_offsets.size == 0:
            split_offsets = np.asarray([0], dtype=np.int64)

        out_path = output_dir / f"{split}.npz"
        np.savez_compressed(out_path, tokens=split_tokens, offsets=split_offsets)

        metadata_splits[split] = {
            "documents": int(doc_counts[split]),
            "tokens": int(split_tokens.size),
            "path": str(out_path),
        }
        print(f"wrote {out_path}: {split_tokens.size:,} tokens, {doc_counts[split]:,} documents")
        doc_start = doc_end

    metadata = {
        "dataset_name": args.dataset_name if args.source == "fineweb" else str(args.input_jsonl),
        "dataset_config": args.dataset_config if args.source == "fineweb" else None,
        "dataset_split": args.dataset_split if args.source == "fineweb" else None,
        "source": args.source,
        "tokenizer_json": str(args.tokenizer_json),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "target_tokens": int(args.target_tokens),
        "total_documents": int(len(lengths)),
        "total_tokens": total_tokens,
        "split_fractions": {"val": args.val_fraction, "test": args.test_fraction},
        "splits": metadata_splits,
        "filters": {
            "language": args.language,
            "min_language_score": args.min_language_score,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_tokens": args.min_tokens,
            "append_eot": not args.no_append_eot,
            "eot_token": args.eot_token,
        },
        "stats": stats,
    }

    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    return metadata


def main() -> None:
    args = parse_args()
    check_output_paths(args.output_dir, args.overwrite)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    if not args.no_append_eot and tokenizer.token_to_id(args.eot_token) is None:
        raise ValueError(f"EOT token {args.eot_token!r} is not present in {args.tokenizer_json}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "_prepare_tmp"
    if temp_dir.exists() and args.overwrite:
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    token_dtype = np.dtype(args.token_dtype)
    try:
        collected = collect_tokens(args, tokenizer, temp_dir)
        metadata = write_splits(
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            token_dtype=token_dtype,
            tokens_path=collected["tokens_path"],
            lengths=collected["lengths"],
            stats=collected["stats"],
            args=args,
        )
    finally:
        if temp_dir.exists() and not args.keep_temp:
            shutil.rmtree(temp_dir)

    print(
        "complete: "
        f"{metadata['total_tokens']:,} tokens across {metadata['total_documents']:,} documents "
        f"in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
