"""
Embedding Fusion Module

Implements multi-modal embedding concatenation as described in Section 4.1
and Appendix I.2 of the paper.

Embedding dimensions:
- Pseudo-static (text features): 3512-d
- Dynamic (behavioral features): 1000-d
- Image (visual features from ResNet-50): 2048-d
- Fused total: 6560-d
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Embedding dimension constants from paper
PSEUDO_STATIC_DIM = 3512
DYNAMIC_DIM = 1000
IMAGE_DIM = 2048
FUSED_DIM = PSEUDO_STATIC_DIM + DYNAMIC_DIM + IMAGE_DIM  # 6560


def load_embeddings(embedding_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from JSON file.

    Args:
        embedding_path: Path to JSON file with embeddings

    Returns:
        Dictionary mapping sample SHA to embedding array
    """
    with open(embedding_path, 'r') as f:
        data = json.load(f)

    embeddings = {}
    for sha, value in data.items():
        if isinstance(value, dict) and 'embedding' in value:
            embeddings[sha] = np.array(value['embedding'])
        elif isinstance(value, list):
            embeddings[sha] = np.array(value)
        else:
            embeddings[sha] = np.array(value)

    return embeddings


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.

    Args:
        embedding: Input embedding vector

    Returns:
        L2-normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def pad_or_truncate(embedding: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad with zeros or truncate embedding to target dimension.

    Args:
        embedding: Input embedding
        target_dim: Target dimension

    Returns:
        Embedding of exactly target_dim dimensions
    """
    current_dim = len(embedding)

    if current_dim == target_dim:
        return embedding
    elif current_dim < target_dim:
        # Pad with zeros
        return np.pad(embedding, (0, target_dim - current_dim), mode='constant')
    else:
        # Truncate
        return embedding[:target_dim]


def fuse_embeddings(pseudo_static: np.ndarray,
                    dynamic: np.ndarray,
                    image: np.ndarray,
                    normalize: bool = True) -> np.ndarray:
    """
    Fuse multi-modal embeddings by concatenation.

    Implements the fusion approach from Appendix I.2.

    Args:
        pseudo_static: Pseudo-static embedding (3512-d expected)
        dynamic: Dynamic embedding (1000-d expected)
        image: Image embedding (2048-d expected)
        normalize: Whether to L2-normalize before fusion

    Returns:
        Fused embedding (6560-d)
    """
    # Ensure correct dimensions
    ps = pad_or_truncate(pseudo_static, PSEUDO_STATIC_DIM)
    dyn = pad_or_truncate(dynamic, DYNAMIC_DIM)
    img = pad_or_truncate(image, IMAGE_DIM)

    if normalize:
        ps = normalize_embedding(ps)
        dyn = normalize_embedding(dyn)
        img = normalize_embedding(img)

    # Concatenate
    fused = np.concatenate([ps, dyn, img])

    return fused


def fuse_embedding_files(pseudo_static_path: str,
                         dynamic_path: str,
                         image_path: str,
                         output_path: str,
                         normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Fuse embeddings from multiple files.

    Args:
        pseudo_static_path: Path to pseudo-static embeddings JSON
        dynamic_path: Path to dynamic embeddings JSON
        image_path: Path to image embeddings JSON
        output_path: Path to save fused embeddings
        normalize: Whether to normalize before fusion

    Returns:
        Dictionary of fused embeddings
    """
    print("Loading pseudo-static embeddings...")
    ps_embeddings = load_embeddings(pseudo_static_path)

    print("Loading dynamic embeddings...")
    dyn_embeddings = load_embeddings(dynamic_path)

    print("Loading image embeddings...")
    img_embeddings = load_embeddings(image_path)

    # Find common samples
    common_shas = set(ps_embeddings.keys()) & set(dyn_embeddings.keys()) & set(img_embeddings.keys())
    print(f"Found {len(common_shas)} samples with all three modalities")

    # Fuse embeddings
    fused_embeddings = {}
    for sha in common_shas:
        fused = fuse_embeddings(
            ps_embeddings[sha],
            dyn_embeddings[sha],
            img_embeddings[sha],
            normalize=normalize
        )
        fused_embeddings[sha] = fused.tolist()

    # Save
    print(f"Saving fused embeddings to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(fused_embeddings, f)

    print(f"Saved {len(fused_embeddings)} fused embeddings")
    return {sha: np.array(emb) for sha, emb in fused_embeddings.items()}


def fuse_with_available_modalities(embeddings_dict: Dict[str, Dict[str, np.ndarray]],
                                   modality_weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """
    Fuse embeddings using available modalities per sample.

    Handles cases where not all modalities are available for every sample.

    Args:
        embeddings_dict: Dict of dicts: {sha: {'pseudo_static': emb, 'dynamic': emb, 'image': emb}}
        modality_weights: Optional weights for each modality

    Returns:
        Dictionary of fused embeddings
    """
    if modality_weights is None:
        modality_weights = {'pseudo_static': 1.0, 'dynamic': 1.0, 'image': 1.0}

    fused = {}

    for sha, modalities in embeddings_dict.items():
        # Initialize empty components
        ps = np.zeros(PSEUDO_STATIC_DIM)
        dyn = np.zeros(DYNAMIC_DIM)
        img = np.zeros(IMAGE_DIM)

        # Fill available modalities
        if 'pseudo_static' in modalities and modalities['pseudo_static'] is not None:
            ps = pad_or_truncate(modalities['pseudo_static'], PSEUDO_STATIC_DIM)
            ps = normalize_embedding(ps) * modality_weights['pseudo_static']

        if 'dynamic' in modalities and modalities['dynamic'] is not None:
            dyn = pad_or_truncate(modalities['dynamic'], DYNAMIC_DIM)
            dyn = normalize_embedding(dyn) * modality_weights['dynamic']

        if 'image' in modalities and modalities['image'] is not None:
            img = pad_or_truncate(modalities['image'], IMAGE_DIM)
            img = normalize_embedding(img) * modality_weights['image']

        fused[sha] = np.concatenate([ps, dyn, img])

    return fused


def compute_modality_statistics(embeddings: Dict[str, np.ndarray],
                                 modality: str) -> Dict:
    """
    Compute statistics for a set of embeddings.

    Args:
        embeddings: Dictionary of embeddings
        modality: Name of modality for reporting

    Returns:
        Statistics dictionary
    """
    if not embeddings:
        return {'modality': modality, 'count': 0}

    all_embeddings = np.array(list(embeddings.values()))

    return {
        'modality': modality,
        'count': len(embeddings),
        'dimension': all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else len(all_embeddings[0]),
        'mean_norm': float(np.mean([np.linalg.norm(e) for e in all_embeddings])),
        'std_norm': float(np.std([np.linalg.norm(e) for e in all_embeddings])),
        'mean_values': all_embeddings.mean(axis=0).tolist()[:10],  # First 10 for brevity
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fuse multi-modal malware embeddings')
    parser.add_argument('--pseudo-static', required=True, help='Path to pseudo-static embeddings')
    parser.add_argument('--dynamic', required=True, help='Path to dynamic embeddings')
    parser.add_argument('--image', required=True, help='Path to image embeddings')
    parser.add_argument('--output', default='fused_embeddings.json', help='Output path')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')

    args = parser.parse_args()

    fused = fuse_embedding_files(
        args.pseudo_static,
        args.dynamic,
        args.image,
        args.output,
        normalize=not args.no_normalize
    )

    print(f"\nFusion complete!")
    print(f"Total fused embeddings: {len(fused)}")
    print(f"Fused dimension: {FUSED_DIM}")
