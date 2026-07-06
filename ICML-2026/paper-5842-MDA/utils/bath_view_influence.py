import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_indices_from_result(result_path: str) -> Dict[str, List[int]]:
    path = Path(result_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        # top_pos.npy / top_neg.npy: shape [top_k, 3], columns = [score, sample_index, loss]
        # 约定：传入 top_pos.npy，自动在同目录下找 top_neg.npy
        arr_pos = np.load(result_path)
        neg_path = path.parent / path.name.replace("top_pos", "top_neg")
        if not neg_path.exists():
            neg_path = path.parent / path.name.replace("pos", "neg")
        arr_neg = np.load(str(neg_path)) if neg_path.exists() else np.zeros((0, 3))
        pos = [int(x) for x in arr_pos[:, 1].tolist()]
        neg = [int(x) for x in arr_neg[:, 1].tolist()]
        return {"positive": pos, "negative": neg}

    if suffix == ".json":
        with open(result_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    elif suffix in (".pkl", ".pickle"):
        with open(result_path, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .npy, .json, .pkl, .pickle")

    pos = [int(x["sample_index"]) for x in obj.get("positive_influencers", [])]
    neg = [int(x["sample_index"]) for x in obj.get("negative_influencers", [])]
    return {
        "positive": pos,
        "negative": neg,
    }


def decode_rows(
    npy_path: str,
    tokenizer_path: str,
    indices: List[int],
    max_length: int,
    skip_special_tokens: bool,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D npy array, got shape={arr.shape}")

    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )

    n_rows = arr.shape[0]
    out = []

    for i in tqdm(range(0, len(indices), batch_size), desc="Decoding rows", unit="batch"):
        batch_indices = indices[i:i + batch_size]
        batch_token_ids = []
        valid_map = []  # [(local_idx, global_idx), ...] for valid indices

        for local_idx, global_idx in enumerate(batch_indices):
            if 0 <= global_idx < n_rows:
                token_ids = arr[global_idx, :max_length].astype(np.int64, copy=False).tolist()
                batch_token_ids.append(token_ids)
                valid_map.append((local_idx, global_idx))
            else:
                valid_map.append((local_idx, global_idx))  # still track for error reporting

        # Batch decode
        batch_texts = tok.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)

        # Build results, handling out-of-range indices
        text_idx = 0
        for local_idx, global_idx in valid_map:
            if 0 <= global_idx < n_rows:
                out.append({
                    "sample_index": global_idx,
                    "text": batch_texts[text_idx],
                    "token_count": len(batch_token_ids[text_idx]),
                })
                text_idx += 1
            else:
                out.append({
                    "sample_index": global_idx,
                    "error": f"index out of range, valid range is [0, {n_rows})",
                })

    return out


def pretty_print(group_name: str, rows: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print(group_name.upper())
    print("=" * 100)
    for item in rows:
        print(f"[sample_index] {item['sample_index']}")
        if "error" in item:
            print(f"[error] {item['error']}")
        else:
            print(f"[token_count] {item['token_count']}")
            print("[text]")
            print(item["text"])
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Decode sample_index rows from a token-id npy file using the model tokenizer."
    )
    parser.add_argument("--result-json", required=True, help="result file (json/pkl) containing positive/negative_influencers")
    parser.add_argument("--data-npy", required=True, help="2D npy file of token ids")
    parser.add_argument("--tokenizer", required=True, help="model/tokenizer path")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--output-json", default=None, help="optional path to save decoded texts as json")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for decoding")
    parser.add_argument("--topk", type=int, default=None, help="only process top-k samples for each group")
    args = parser.parse_args()

    print(f"Loading result from: {args.result_json}")
    idx_groups = load_indices_from_result(args.result_json)

    # Apply topk limit
    if args.topk is not None:
        idx_groups["positive"] = idx_groups["positive"][:args.topk]
        idx_groups["negative"] = idx_groups["negative"][:args.topk]

    print(f"Found {len(idx_groups['positive'])} positive, {len(idx_groups['negative'])} negative influencers")

    print("\nDecoding positive influencers...")
    decoded = {
        "positive": decode_rows(
            npy_path=args.data_npy,
            tokenizer_path=args.tokenizer,
            indices=idx_groups["positive"],
            max_length=args.max_length,
            skip_special_tokens=args.skip_special_tokens,
            batch_size=args.batch_size,
        ),
    }

    print("\nDecoding negative influencers...")
    decoded["negative"] = decode_rows(
        npy_path=args.data_npy,
        tokenizer_path=args.tokenizer,
        indices=idx_groups["negative"],
        max_length=args.max_length,
        skip_special_tokens=args.skip_special_tokens,
        batch_size=args.batch_size,
    )

    # pretty_print("positive influencers", decoded["positive"])
    # pretty_print("negative influencers", decoded["negative"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(decoded, f, ensure_ascii=False, indent=2)
        print(f"Saved decoded text json to: {out_path}")


if __name__ == "__main__":
    main()