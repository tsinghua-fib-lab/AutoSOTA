#!/usr/bin/env python3
"""
Build ImageNet similarity matrices from pre-trained model classification heads.

This script loads pre-trained ImageNet classification heads from various models,
computes cosine similarity matrices between class embeddings, and saves them as .npy files.
"""

import os
import torch
import numpy as np
import argparse
from typing import Tuple
from scipy.stats import pearsonr, spearmanr

from utils.model_utils import load_head_weights, get_all_huggingface_models
from config.config import IMAGENET1K_HEAD_MODELS, IMAGENET21K_HEAD_MODELS, IMAGENET21K_2EXTRA_HEAD_MODELS, WEIGHTS_DIR
from utils.extract_imagenet1k_layer import compute_index_mapping, read_imagenet1k_wnids, read_imagenet21k_wnids, read_imagenet12k_wnids, read_imagenet21k_2extra_wnids12k
from utils.utils import get_text_encoder, get_label_names
from alignment.train_aligners import get_avg_image_embeddings


def compute_similarity_matrix(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute a num_classes×num_classes cosine similarity matrix between rows of the weight tensor.

    Args:
        weight (torch.Tensor): shape (num_classes, feature_dim).
    Returns:
        sim_matrix (torch.Tensor): shape (num_classes, num_classes), where
            sim_matrix[i, j] = cosine_similarity(weight[i], weight[j]).
    """
    normed = torch.nn.functional.normalize(weight, p=2, dim=1, eps=1e-12)
    sim_matrix = torch.matmul(normed, normed.t())
    return sim_matrix

def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix.
    
    Args:
        K: Kernel matrix of shape [n, n]
        
    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two kernel matrices.
    
    Args:
        K1: First kernel matrix
        K2: Second kernel matrix
        
    Returns:
        CKA score between 0 and 1
    """
    # Center the kernels
    # print("Centering kernel matrices...")
    K1_centered = center_kernel(K1)
    K2_centered = center_kernel(K2)
    
    # Compute CKA
    numerator = np.trace(K1_centered @ K2_centered)
    denominator = np.sqrt(np.trace(K1_centered @ K1_centered) * np.trace(K2_centered @ K2_centered))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_off_diagonal_correlation(K1: np.ndarray, K2: np.ndarray, method: str = 'pearson') -> Tuple[float, float]:
    """
    Compute correlation of off-diagonal elements between two matrices.
    
    Args:
        K1: First matrix
        K2: Second matrix
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    # Extract off-diagonal elements
    mask = ~np.eye(K1.shape[0], dtype=bool)
    off_diag_1 = K1[mask]
    off_diag_2 = K2[mask]
    
    if method.lower() == 'spearman':
        return spearmanr(off_diag_1, off_diag_2)
    elif method.lower() == 'pearson':
        return pearsonr(off_diag_1, off_diag_2)
    else:
        raise ValueError(f"Unsupported correlation method: {method}. Use 'pearson' or 'spearman'.")


def compute_mutual_knn_alignment(K1: np.ndarray, K2: np.ndarray, k: int) -> float:
    """
    Compute mutual K-NN alignment between two similarity matrices.
    
    For each row i, finds the k nearest neighbors (highest similarities excluding diagonal)
    in both matrices and computes the overlap ratio. Returns the average over all rows.
    
    Args:
        K1: First similarity matrix
        K2: Second similarity matrix
        k: Number of nearest neighbors to consider
        
    Returns:
        Mutual K-NN alignment score between 0 and 1
    """
    n = K1.shape[0]
    assert K1.shape == K2.shape, "Matrices must have the same shape"
    assert k < n, "k must be less than the number of classes"
    
    total_overlap = 0.0
    
    for i in range(n):
        # Get similarities for row i, excluding diagonal element
        sim1 = K1[i].copy()
        sim2 = K2[i].copy()
        
        # Set diagonal to very low value to exclude it from k-NN
        sim1[i] = -np.inf
        sim2[i] = -np.inf
        
        # Find k nearest neighbors (indices with highest similarities)
        knn_indices_1 = np.argsort(sim1)[-k:]  # Top k indices
        knn_indices_2 = np.argsort(sim2)[-k:]  # Top k indices
        
        # Compute overlap
        overlap = len(np.intersect1d(knn_indices_1, knn_indices_2))
        overlap_ratio = overlap / k
        
        total_overlap += overlap_ratio
    
    # Average over all indices
    return total_overlap / n


def get_image_cache_filename(model_name: str, use_image_representations: bool = False, few_shot_samples: int = None) -> str:
    """
    Generate a cache filename for image similarity matrix.
    
    Args:
        model_name: Name of the image model
        use_image_representations: Whether image representations are used
        few_shot_samples: Number of few-shot samples (if applicable)
        
    Returns:
        Cache filename for image similarity matrix
    """
    # Clean model names for filesystem
    clean_image_name = model_name.replace('/', '_').replace(':', '_')
    
    if use_image_representations:
        if few_shot_samples is not None:
            return f"image_similarity_{clean_image_name}_imgrep_{few_shot_samples}shot.npy"
        else:
            return f"image_similarity_{clean_image_name}_imgrep_allshots.npy"
    else:
        return f"image_similarity_{clean_image_name}_weights.npy"


def get_text_cache_filename(text_model_name: str, dataset_type: str = "imagenet1k") -> str:
    """
    Generate a cache filename for text similarity matrix.
    
    Args:
        text_model_name: Name of the text model
        dataset_type: Type of dataset ("imagenet1k", "imagenet21k", etc.)
        
    Returns:
        Cache filename for text similarity matrix
    """
    # Clean model names for filesystem
    clean_text_name = text_model_name.replace('/', '_').replace(':', '_')
    return f"text_similarity_{clean_text_name}_{dataset_type}.npy"


def get_cache_filename(model_name: str, text_model_name: str, use_image_representations: bool = False, few_shot_samples: int = None) -> str:
    """
    Generate a cache filename for similarity matrices (legacy function for backward compatibility).
    
    Args:
        model_name: Name of the image model
        text_model_name: Name of the text model
        use_image_representations: Whether image representations are used
        few_shot_samples: Number of few-shot samples (if applicable)
        
    Returns:
        Cache filename
    """
    # Clean model names for filesystem
    clean_image_name = model_name.replace('/', '_').replace(':', '_')
    clean_text_name = text_model_name.replace('/', '_').replace(':', '_')
    
    if use_image_representations:
        if few_shot_samples is not None:
            return f"similarity_matrices_{clean_image_name}_{clean_text_name}_imgrep_{few_shot_samples}shot.npz"
        else:
            return f"similarity_matrices_{clean_image_name}_{clean_text_name}_imgrep_allshots.npz"
    else:
        return f"similarity_matrices_{clean_image_name}_{clean_text_name}_weights.npz"


def load_similarity_matrices(cache_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached similarity matrices.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Tuple of (image_similarity_matrix, text_similarity_matrix)
    """
    if os.path.exists(cache_path):
        print(f"Loading cached similarity matrices from: {cache_path}")
        data = np.load(cache_path)
        return data['image_similarity_matrix'], data['text_similarity_matrix']
    else:
        return None, None


def load_image_similarity_matrix(cache_path: str) -> np.ndarray:
    """
    Load cached image similarity matrix.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Image similarity matrix or None if not found
    """
    if os.path.exists(cache_path):
        print(f"Loading cached image similarity matrix from: {cache_path}")
        return np.load(cache_path)
    else:
        return None


def load_text_similarity_matrix(cache_path: str) -> np.ndarray:
    """
    Load cached text similarity matrix.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Text similarity matrix or None if not found
    """
    if os.path.exists(cache_path):
        print(f"Loading cached text similarity matrix from: {cache_path}")
        return np.load(cache_path)
    else:
        return None


def save_similarity_matrices(cache_path: str, image_similarity_matrix: np.ndarray, text_similarity_matrix: np.ndarray):
    """
    Save similarity matrices to cache.
    
    Args:
        cache_path: Path to save the cache file
        image_similarity_matrix: Image similarity matrix
        text_similarity_matrix: Text similarity matrix
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    print(f"Saving similarity matrices to: {cache_path}")
    np.savez_compressed(cache_path, 
                       image_similarity_matrix=image_similarity_matrix,
                       text_similarity_matrix=text_similarity_matrix)


def save_image_similarity_matrix(cache_path: str, image_similarity_matrix: np.ndarray):
    """
    Save image similarity matrix to cache.
    
    Args:
        cache_path: Path to save the cache file
        image_similarity_matrix: Image similarity matrix
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    print(f"Saving image similarity matrix to: {cache_path}")
    np.save(cache_path, image_similarity_matrix)


def save_text_similarity_matrix(cache_path: str, text_similarity_matrix: np.ndarray):
    """
    Save text similarity matrix to cache.
    
    Args:
        cache_path: Path to save the cache file
        text_similarity_matrix: Text similarity matrix
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    print(f"Saving text similarity matrix to: {cache_path}")
    np.save(cache_path, text_similarity_matrix)



def compute_alignment_metrics(image_model_name, text_model_name, output_dir, device, use_image_representations, 
                             few_shot_samples, force):
    """
    Compute alignment metrics for a specific configuration.
    
    Returns:
        Dictionary containing all computed metrics
    """
    # Generate cache filenames and paths for separate matrices
    image_cache_filename = get_image_cache_filename(image_model_name, use_image_representations, few_shot_samples)
    image_cache_path = os.path.join(output_dir, image_cache_filename)
    
    # Determine dataset type for text cache
    if image_model_name in IMAGENET1K_HEAD_MODELS or use_image_representations:
        dataset_type = "imagenet1k"
    elif image_model_name in IMAGENET21K_HEAD_MODELS:
        dataset_type = "imagenet21k"
    elif image_model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
        dataset_type = "imagenet21k_2extra"
    else:
        dataset_type = "imagenet1k"  # Default fallback
    
    text_cache_filename = get_text_cache_filename(text_model_name, dataset_type)
    text_cache_path = os.path.join(output_dir, text_cache_filename)
    
    # Legacy cache for backward compatibility
    legacy_cache_filename = get_cache_filename(image_model_name, text_model_name, use_image_representations, few_shot_samples)
    legacy_cache_path = os.path.join(output_dir, legacy_cache_filename)
    
    # Try to load cached similarity matrices
    image_similarity_matrix = None
    text_similarity_matrix = None
    
    if not force:
        # First try to load individual matrices
        image_similarity_matrix = load_image_similarity_matrix(image_cache_path)
        text_similarity_matrix = load_text_similarity_matrix(text_cache_path)
        
        # If individual matrices not found, try legacy combined cache
        if image_similarity_matrix is None or text_similarity_matrix is None:
            legacy_image, legacy_text = load_similarity_matrices(legacy_cache_path)
            if legacy_image is not None and legacy_text is not None:
                print("Using legacy cached similarity matrices")
                image_similarity_matrix = legacy_image
                text_similarity_matrix = legacy_text
        
        # If we have both matrices, proceed to analysis
        if image_similarity_matrix is not None and text_similarity_matrix is not None:
            print("Using cached similarity matrices")
        else:
            # Need to compute at least one matrix
            image_similarity_matrix = None
            text_similarity_matrix = None
    
    # Initialize indices variable for use in both image and text computations
    indices = None
    
    # Determine if we need to compute image similarity matrix
    compute_image_matrix = force or image_similarity_matrix is None
    
    # Determine if we need to compute text similarity matrix
    compute_text_matrix = force or text_similarity_matrix is None

    if compute_image_matrix:
        if use_image_representations:
            # Use image representations approach
            dataset_img_repr = "imagenet1kval"
            num_classes = 1000  # ImageNet-1k has 1000 classes
            
            # print(f"Getting average image embeddings from {dataset_img_repr} dataset...")
            if few_shot_samples is not None:
                print(f"Using few-shot mode: {few_shot_samples} samples per class")
            else:
                print("Using all available samples")
                
            image_weights, image_embedding_dim = get_avg_image_embeddings(
                dataset_img_repr=dataset_img_repr, 
                image_model_name=image_model_name, 
                num_classes=num_classes, 
                device=device, 
                few_shot_samples=few_shot_samples
            )
            
            # print(f"Image embedding dimension: {image_embedding_dim}")
            # print(f"Image weights shape: {image_weights.shape}")
            # print(f"Image weights dtype: {image_weights.dtype}")
        else:
            # Load image model weights
            if image_model_name in IMAGENET1K_HEAD_MODELS:
                imagenet_pth_name = 'imagenet1k.pth'
            elif image_model_name in IMAGENET21K_HEAD_MODELS:
                imagenet_pth_name = 'imagenet21k.pth'
            elif image_model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
                imagenet_pth_name = 'imagenet21k_2extra.pth'
            else:
                raise ValueError(f"Unsupported image model: {image_model_name}. "
                                f"Supported models are: {IMAGENET1K_HEAD_MODELS + IMAGENET21K_HEAD_MODELS + IMAGENET21K_2EXTRA_HEAD_MODELS}")
            weights_path = os.path.join(WEIGHTS_DIR, image_model_name.replace('/', '_'), imagenet_pth_name)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Image model weights not found: {weights_path}")
            
            # print(f"Loading image model weights from: {weights_path}")
            weights_data = torch.load(weights_path, map_location=device)
            image_weights = weights_data['weight'].float()  # Ensure float32
            image_embedding_dim = image_weights.shape[1]
            
            # print(f"Image embedding dimension: {image_embedding_dim}")
            # print(f"Image weights shape: {image_weights.shape}")
            # print(f"Image weights dtype: {image_weights.dtype}")

            imagenet_subset_wnids = read_imagenet1k_wnids()

            if image_model_name in IMAGENET21K_HEAD_MODELS:
                imagenet_21k_wnids = read_imagenet21k_wnids()
            elif image_model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
                imagenet_21k_wnids = read_imagenet21k_2extra_wnids12k()
            
            indices, missing_wnids, index_mapping = compute_index_mapping(imagenet_subset_wnids, imagenet_21k_wnids)
            # print(f"Found {len(indices)} matching classes out of {len(imagenet_subset_wnids)} of the Imagenet subset classes")
            
            # Reduce image weights to matching indices
            # print(f"Original image weights shape: {image_weights.shape}")
            image_weights = image_weights[indices].float()

        # Compute image similarity matrix
        print("Computing image similarity matrix...")
        image_similarity_matrix = compute_similarity_matrix(image_weights)
        image_similarity_matrix = image_similarity_matrix.cpu().numpy()
        
        # Save image similarity matrix
        save_image_similarity_matrix(image_cache_path, image_similarity_matrix)
    
    # If we need indices for text computation but didn't compute image matrix, compute indices
    if compute_text_matrix and indices is None and not use_image_representations and dataset_type != "imagenet1k":
        imagenet_subset_wnids = read_imagenet1k_wnids()
        if dataset_type == "imagenet21k":
            imagenet_21k_wnids = read_imagenet21k_wnids()
        elif dataset_type == "imagenet21k_2extra":
            imagenet_21k_wnids = read_imagenet21k_2extra_wnids12k()
        
        indices, missing_wnids, index_mapping = compute_index_mapping(imagenet_subset_wnids, imagenet_21k_wnids)

    if compute_text_matrix:
        # print(f"Loading text encoder for {text_model_name}...")
        text_encoder = get_text_encoder(text_model_name, device)
        
        # print("Generating text embeddings for class names...")
        batch_size = 5000
        text_features_list = []

        if use_image_representations or dataset_type == "imagenet1k":
            all_prompts = get_label_names("imagenet1k")
        else:
            all_prompts = get_label_names("imagenet21k")
            if not use_image_representations:
                all_prompts = [all_prompts[idx] for idx in indices]
        
        # print(f"Total prompts for text embedding: {len(all_prompts)}")

        text_embeddings = text_encoder(all_prompts).float()
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
        
        text_embedding_dim = text_embeddings.shape[1]
        # print(f"Text embedding dimension: {text_embedding_dim}")
        # print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Compute text similarity matrix
        print("Computing text similarity matrix...")
        text_similarity_matrix = compute_similarity_matrix(text_embeddings)
        text_similarity_matrix = text_similarity_matrix.cpu().numpy()
        
        # Save text similarity matrix
        save_text_similarity_matrix(text_cache_path, text_similarity_matrix)
    
    # Compute CKA
    print("Computing CKA...")
    cka_score = compute_cka(image_similarity_matrix, text_similarity_matrix)

    # Zero-out diagonal elements and recompute CKA
    image_sim_off_diag = image_similarity_matrix.copy()
    text_sim_off_diag = text_similarity_matrix.copy()
    np.fill_diagonal(image_sim_off_diag, 0)
    np.fill_diagonal(text_sim_off_diag, 0)
    cka_off_diag_score = compute_cka(image_sim_off_diag, text_sim_off_diag)

    # Compute off-diagonal correlation
    print("Computing off-diagonal correlation...")
    off_diag_corr, p_value = compute_off_diagonal_correlation(image_sim_off_diag,
                                                              text_sim_off_diag,
                                                              method='pearson')

    # Off diagonal Spearman correlation
    off_diag_corr_spearman, p_value_spearman = compute_off_diagonal_correlation(
        image_sim_off_diag, text_sim_off_diag, method='spearman')

    # Compute mutual K-NN alignment for k=3, 5, 10
    print("Computing mutual K-NN alignment...")
    knn_3 = compute_mutual_knn_alignment(image_similarity_matrix, text_similarity_matrix, k=3)
    knn_5 = compute_mutual_knn_alignment(image_similarity_matrix, text_similarity_matrix, k=5)
    knn_10 = compute_mutual_knn_alignment(image_similarity_matrix, text_similarity_matrix, k=10)

    return {
        'few_shot_samples': few_shot_samples,
        'cka_score': cka_score,
        'cka_off_diag_score': cka_off_diag_score,
        'off_diag_corr': off_diag_corr,
        'p_value': p_value,
        'off_diag_corr_spearman': off_diag_corr_spearman,
        'p_value_spearman': p_value_spearman,
        'knn_3': knn_3,
        'knn_5': knn_5,
        'knn_10': knn_10
    }


def display_results_table(results_list):
    """
    Display results in tabular format.
    """
    print("\n" + "="*120)
    print("ALIGNMENT METRICS ACROSS DIFFERENT FEW-SHOT SAMPLES")
    print("="*120)
    
    # Header
    header = f"{'Samples':<8} {'CKA':<8} {'CKA(off)':<10} {'Pearson':<10} {'P-val':<10} {'Spearman':<10} {'P-val':<10} {'KNN-3':<8} {'KNN-5':<8} {'KNN-10':<8}"
    print(header)
    print("-"*120)
    
    # Data rows
    for result in results_list:
        row = (f"{result['few_shot_samples']:<8} "
               f"{result['cka_score']:<8.4f} "
               f"{result['cka_off_diag_score']:<10.4f} "
               f"{result['off_diag_corr']:<10.4f} "
               f"{result['p_value']:<10.4f} "
               f"{result['off_diag_corr_spearman']:<10.4f} "
               f"{result['p_value_spearman']:<10.4f} "
               f"{result['knn_3']:<8.4f} "
               f"{result['knn_5']:<8.4f} "
               f"{result['knn_10']:<8.4f}")
        print(row)
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Build ImageNet similarity matrices from pre-trained model classification heads")
    parser.add_argument("--output_dir", type=str, default="analysis",
                       help="Output directory for similarity matrices (default: analysis)")
    parser.add_argument("--image_model_name", type=str,
                       default='timm/beit_base_patch16_224.in22k_ft_in22k',
                       help="Name of the image model to process")
    parser.add_argument("--text_model_name", type=str,
                       help="Name of the text model to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run computation on (default: cpu)")
    parser.add_argument("--weights_dir", type=str, default="weights",
                       help="Directory to cache model weights (default: weights)")
    parser.add_argument("--use_image_representations", action="store_true",
                       help="Use average image representations instead of model weights")
    parser.add_argument("--force", action="store_true",
                       help="Force recomputation even if output files already exist")
    
    args = parser.parse_args()
    
    # inputs image_model_name text_model_name
    image_model_name = args.image_model_name
    text_model_name = args.text_model_name
    output_dir = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_image_representations = args.use_image_representations
    force = args.force
    
    # Define few_shot_samples list for iteration when using image representations
    few_shot_samples_list = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    if use_image_representations:
        # Iterate over different few_shot_samples values
        print(f"Running alignment analysis with image representations across {len(few_shot_samples_list)} different few-shot sample sizes...")
        results_list = []
        
        for few_shot_samples in few_shot_samples_list:
            print(f"\n--- Processing few_shot_samples = {few_shot_samples} ---")
            result = compute_alignment_metrics(
                image_model_name, text_model_name, output_dir, device, 
                use_image_representations, few_shot_samples, force
            )
            results_list.append(result)
            
            # Print individual results
            print(f"CKA score: {result['cka_score']:.4f}")
            print(f"CKA score (off-diagonal only): {result['cka_off_diag_score']:.4f}")
            print(f"Off-diagonal Pearson correlation: {result['off_diag_corr']:.4f}, p-value: {result['p_value']:.4f}")
            print(f"Off-diagonal Spearman correlation: {result['off_diag_corr_spearman']:.4f}, p-value: {result['p_value_spearman']:.4f}")
            print(f"Mutual K-NN alignment (k=3): {result['knn_3']:.4f}")
            print(f"Mutual K-NN alignment (k=5): {result['knn_5']:.4f}")
            print(f"Mutual K-NN alignment (k=10): {result['knn_10']:.4f}")
        
        # Display results in tabular format
        display_results_table(results_list)
    else:
        # Original behavior for non-image representations case
        print("Running alignment analysis with model weights...")
        result = compute_alignment_metrics(
            image_model_name, text_model_name, output_dir, device, 
            use_image_representations, None, force
        )
        
        # Print results
        print(f"CKA score: {result['cka_score']:.4f}")
        print(f"CKA score (off-diagonal only): {result['cka_off_diag_score']:.4f}")
        print(f"Off-diagonal Pearson correlation: {result['off_diag_corr']:.4f}, p-value: {result['p_value']:.4f}")
        print(f"Off-diagonal Spearman correlation: {result['off_diag_corr_spearman']:.4f}, p-value: {result['p_value_spearman']:.4f}")
        print(f"Mutual K-NN alignment (k=3): {result['knn_3']:.4f}")
        print(f"Mutual K-NN alignment (k=5): {result['knn_5']:.4f}")
        print(f"Mutual K-NN alignment (k=10): {result['knn_10']:.4f}")


if __name__ == '__main__':
    main()
