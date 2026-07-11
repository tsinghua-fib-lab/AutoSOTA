"""
Distance Matrix Generation

Computes pairwise Euclidean distances between fused embeddings
for phylogenetic tree construction.
"""
import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform


def load_embeddings(input_path: str):
    """
    Load embeddings from JSON file.

    Args:
        input_path: Path to embeddings JSON file

    Returns:
        Tuple of (embeddings array, labels list, sha list)
    """
    with open(input_path, 'r') as file:
        data = json.load(file)

    print(f"Loaded {len(data)} samples", flush=True)

    embeddings = []
    labels = []
    shas = []

    for sha, value in data.items():
        if isinstance(value, dict) and 'embedding' in value:
            embeddings.append(value['embedding'])
            labels.append(value.get('label', 'Unknown'))
        elif isinstance(value, list):
            embeddings.append(value)
            labels.append('Unknown')
        else:
            embeddings.append(value)
            labels.append('Unknown')
        shas.append(sha)

    return np.array(embeddings), labels, shas


def compute_distance_matrix(embeddings, condensed=False):
    """
    Compute the pairwise Euclidean distance matrix.

    Uses scipy's vectorized ``pdist`` (orders of magnitude faster than a Python
    double loop). Memory scales as O(n^2): a full square matrix for 100k
    samples needs ~80 GB, so use ``condensed=True`` at that scale.

    Args:
        embeddings: Array of embedding vectors, shape (n, d)
        condensed: If True, return the condensed upper-triangle vector
            (length n*(n-1)/2); otherwise return the full symmetric matrix.

    Returns:
        Distance matrix (square symmetric ndarray) or condensed vector.
    """
    condensed_distances = pdist(embeddings, metric='euclidean')
    if condensed:
        return condensed_distances
    return squareform(condensed_distances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate distance matrix from embeddings')
    parser.add_argument('--input', default='fused_embeddings.json',
                        help='Path to embeddings JSON file (default: fused_embeddings.json)')
    parser.add_argument('--output', default='distance_matrix.npy',
                        help='Output path for distance matrix (default: distance_matrix.npy)')
    parser.add_argument('--output-labels', default='sample_labels.json',
                        help='Output path for sample labels/SHAs mapping')
    parser.add_argument('--condensed', action='store_true',
                        help='Save the condensed upper-triangle vector instead of '
                             'the full square matrix (needed at ~100k-sample scale)')

    args = parser.parse_args()

    # Load embeddings
    embeddings, labels, shas = load_embeddings(args.input)
    print(f"Embedding dimension: {embeddings.shape[1]}", flush=True)

    # Compute the distance matrix
    print("Computing distance matrix...", flush=True)
    distance_matrix = compute_distance_matrix(embeddings, condensed=args.condensed)

    # Save outputs
    np.save(args.output, distance_matrix)
    print(f"Distance matrix saved to {args.output}", flush=True)

    # Save SHA/label mapping for later use
    label_mapping = {i: {'sha': shas[i], 'label': labels[i]} for i in range(len(shas))}
    with open(args.output_labels, 'w') as f:
        json.dump(label_mapping, f)
    print(f"Label mapping saved to {args.output_labels}", flush=True)