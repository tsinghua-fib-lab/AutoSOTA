"""
Compute embeddings for (question + generated_answer) from multiple JSON files,
then for each item find its top-9 nearest neighbors (cosine), and save to one JSONL.

Install:
  pip install -U sentence-transformers torch numpy

Run:
  python knn_qa.py \
    --input_glob "/path/to/jsons/*.json" \
    --out_jsonl "/path/to/out/neighbors.jsonl" \
    --model "Qwen/Qwen3-Embedding-8B" \
    --embed_batch 64 \
    --query_block 256
"""

import argparse
import glob
import json
import os
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def build_text(q: str, a: str) -> str:
    q = (q or "").strip()
    a = (a or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list JSON.")
    return data


def embed_all(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    # normalize_embeddings=True => unit vectors, dot == cosine
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return emb


def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    emb: (N, D) float32, already normalized.
    Returns:
      nn_idx: (N, k) int32
      nn_score: (N, k) float32
    """
    N, D = emb.shape
    nn_idx = np.empty((N, k), dtype=np.int32)
    nn_score = np.empty((N, k), dtype=np.float32)

    # For speed: keep emb in float32, use BLAS dot.
    for start in range(0, N, query_block):
        end = min(start + query_block, N)
        q = emb[start:end]                 # (B, D)
        scores = q @ emb.T                 # (B, N)

        # Exclude self: set diagonal positions to -inf for this block
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        scores[rows, cols] = -np.inf

        # Get top-k indices per row (unsorted), then sort them by score desc
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]  # (B, k)
        top_sc = np.take_along_axis(scores, top_idx, axis=1)        # (B, k)

        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)

        nn_idx[start:end] = top_idx.astype(np.int32)
        nn_score[start:end] = top_sc.astype(np.float32)

        print(f"Processed queries [{start}:{end}) / {N}")

    return nn_idx, nn_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embed_batch", type=int, default=64)
    parser.add_argument("--query_block", type=int, default=256)
    parser.add_argument("--k", type=int, default=9)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    # 1) Load + build texts + keep meta mapping
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for p in paths:
        items = load_items(p)
        for row_in_file, obj in enumerate(items):
            q = obj.get("question", "")
            ga = obj.get("generated_answer", "")
            txt = build_text(q, ga)
            texts.append(txt)
            meta.append({
                "global_id": len(meta),                 # 0..N-1
                "source_file": os.path.basename(p),
                "row_in_file": row_in_file,
                "index": obj.get("index", None),
            })

    N = len(texts)
    print(f"Loaded {N} samples from {len(paths)} files.")

    # 2) Embed
    model = SentenceTransformer(args.model)
    emb = embed_all(model, texts, batch_size=args.embed_batch)
    print(f"Embeddings ready: shape={emb.shape}")

    # 3) KNN (top-9)
    nn_idx, nn_score = topk_neighbors_all(
        emb,
        k=args.k,
        query_block=args.query_block,
    )

    # 4) Save one JSONL: each line contains item meta + its neighbors
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in range(N):
            neighbors = [
                {"global_id": int(nn_idx[i, j]), "cosine": float(nn_score[i, j])}
                for j in range(args.k)
            ]
            row = {
                **meta[i],
                "neighbors": neighbors,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved neighbors to: {args.out_jsonl}")


if __name__ == "__main__":
    main()
