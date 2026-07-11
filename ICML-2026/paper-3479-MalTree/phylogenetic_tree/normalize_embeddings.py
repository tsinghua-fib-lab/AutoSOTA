"""L2-normalize embedding vectors.

Each embedding is divided by its L2 norm so that downstream Euclidean
distances are computed on the unit sphere. This is the normalization step
before distance-matrix construction in the MalTree pipeline.

Usage:
    python normalize_embeddings.py --input fused_embeddings.json \
        --output normalized_embeddings.json
"""
import argparse
import json

import numpy as np


def load_embeddings(path):
    """Load an embeddings JSON into (embeddings array, labels, shas)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    embeddings, labels, shas = [], [], []
    for sha, value in data.items():
        if isinstance(value, dict):
            vector = value.get("embedding")
            if vector is None:
                vector = value.get("embeddings")
            label = value.get("label", value.get("family"))
        else:
            vector, label = value, None
        if vector is None:
            continue
        embeddings.append(vector)
        labels.append(label)
        shas.append(sha)

    return np.asarray(embeddings, dtype=np.float64), labels, shas


def normalize_embeddings(embeddings):
    """Divide each row by its L2 norm; zero-norm rows are left unchanged."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def save_embeddings(path, embeddings, labels, shas):
    """Write normalized embeddings back to a {sha: {embedding, label}} JSON."""
    data = {sha: {"embedding": vector.tolist(), "label": label}
            for sha, vector, label in zip(shas, embeddings, labels)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2-normalize embedding vectors")
    parser.add_argument("--input", default="fused_embeddings.json",
                        help="input embeddings JSON (default: fused_embeddings.json)")
    parser.add_argument("--output", default="normalized_embeddings.json",
                        help="output normalized embeddings JSON")
    args = parser.parse_args()

    embeddings, labels, shas = load_embeddings(args.input)
    print(f"loaded {len(shas)} embeddings of dimension "
          f"{embeddings.shape[1] if embeddings.ndim == 2 else 0}")
    normalized = normalize_embeddings(embeddings)
    save_embeddings(args.output, normalized, labels, shas)
    print(f"normalized embeddings written to {args.output}")
