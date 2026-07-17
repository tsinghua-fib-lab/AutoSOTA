"""
Pre-compute embeddings for the full Jigsaw dataset.

This creates a single cache file with embeddings for ALL texts in row order,
which can then be sliced by index for different train/val/test splits.

The key insight: load_jigsaw_frame(sample=None) returns deterministic row order
(filtering removes some rows, but the remaining rows keep their relative order).
We embed in that order, so indices from train_test_split map directly to this array.

Usage:
    python -m src.precompute_embeddings --dataset jigsaw --model roberta-base
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from data.jigsaw import load_jigsaw_frame


DEFAULT_CACHE_DIR = Path("cache/embeddings")


def compute_text_hash(texts: list, n_canary: int = 5) -> str:
    """Compute hash of first N texts for alignment verification."""
    canary = "\n".join(str(t) for t in texts[:n_canary])
    return hashlib.md5(canary.encode()).hexdigest()[:16]


def precompute_jigsaw_embeddings(
    hf_model: str = "roberta-base",
    pooling: str = "mean",
    max_length: int = 128,
    batch_size: int = 64,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    data_path: str = "data/jigsaw/train.csv",
    device: str = None,
    force: bool = False,
):
    """
    Pre-compute embeddings for the full Jigsaw dataset.

    Saves:
    - jigsaw_{model}_{pooling}_full.npy: embeddings array (N, hidden_size)
    - jigsaw_{model}_{pooling}_full_meta.npz: metadata including text hash canary

    The embeddings are stored in the same row order as load_jigsaw_frame(sample=None),
    so train/val/test indices from any seed can directly index into this array.
    """
    from transformers import AutoTokenizer, AutoModel

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load full dataset (no sampling) - this is the canonical row order
    print(f"Loading full Jigsaw dataset from {data_path}...")
    df = load_jigsaw_frame(path=data_path, sample=None)
    texts = df['comment_text'].tolist()
    n_total = len(texts)
    print(f"Loaded {n_total:,} texts")

    # Compute canary hash for verification
    text_hash = compute_text_hash(texts)
    print(f"Text canary hash (first 5 rows): {text_hash}")

    # Check if cache already exists
    model_key = hf_model.replace('/', '_')
    cache_path = cache_dir / f"jigsaw_{model_key}_{pooling}_full.npy"
    meta_path = cache_dir / f"jigsaw_{model_key}_{pooling}_full_meta.npz"

    if cache_path.exists() and meta_path.exists() and not force:
        print(f"Cache already exists: {cache_path}")
        existing = np.load(cache_path)
        existing_meta = np.load(meta_path, allow_pickle=True)
        existing_hash = str(existing_meta.get('text_hash', ''))

        if len(existing) == n_total and existing_hash == text_hash:
            print(f"Cache valid: size={n_total:,}, hash={text_hash}")
            print("Use --force to re-embed anyway.")
            return cache_path
        else:
            print(f"Cache invalid: size {len(existing):,} vs {n_total:,}, hash '{existing_hash}' vs '{text_hash}'")
            print("Re-embedding...")

    # Load model
    print(f"Loading model: {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    encoder = AutoModel.from_pretrained(hf_model).to(device)
    encoder.eval()

    # Embed in batches
    print(f"Embedding {n_total:,} texts...")
    out_vecs = []

    with torch.no_grad():
        for i in tqdm(range(0, n_total, batch_size), desc="Embedding"):
            batch_texts = [str(t) for t in texts[i : i + batch_size]]

            # Tokenize
            toks = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}

            # Forward pass
            outputs = encoder(**toks)
            h = outputs.last_hidden_state

            # Pool to sentence embedding
            if pooling == "cls":
                vec = h[:, 0, :]
            else:  # mean pooling
                mask = toks["attention_mask"].unsqueeze(-1)
                summed = (h * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                vec = summed / counts

            out_vecs.append(vec.detach().cpu().numpy())

    embeddings = np.concatenate(out_vecs, axis=0).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings and metadata
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    np.savez(meta_path,
             n_total=n_total,
             text_hash=text_hash,
             hf_model=hf_model,
             pooling=pooling,
             max_length=max_length)
    print(f"Saved to {cache_path} ({embeddings.nbytes / 1e9:.2f} GB)")
    print(f"Metadata saved to {meta_path}")

    return cache_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for datasets")
    parser.add_argument("--dataset", type=str, default="jigsaw", help="Dataset name")
    parser.add_argument("--model", type=str, default="roberta-base", help="HuggingFace model name")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default="data/jigsaw/train.csv", help="Path to dataset CSV")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed even if cache exists")

    args = parser.parse_args()

    if args.dataset == "jigsaw":
        precompute_jigsaw_embeddings(
            hf_model=args.model,
            pooling=args.pooling,
            batch_size=args.batch_size,
            data_path=args.data_path,
            device=args.device,
            force=args.force,
        )
    else:
        print(f"Dataset {args.dataset} not yet supported for precomputation")


if __name__ == "__main__":
    main()
