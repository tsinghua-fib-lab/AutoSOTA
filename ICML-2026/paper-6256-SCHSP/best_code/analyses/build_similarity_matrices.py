#!/usr/bin/env python3
"""
Build similarity matrices for probe training analysis.

This script loads concept weights from binary classification probes,
computes concept vectors, builds visual and text similarity matrices,
and saves them as .npy files.
"""

import os
import numpy as np
import torch
from typing import List, Tuple
import argparse

from utils.utils import get_label_names, get_text_encoder
from config.config import EMBEDDINGS_DIR, WEIGHTS_DIR


def load_probe_weights(weights_dir: str, dataset_name: str, balance_type: str = "undersample", num_classes: int = 10) -> torch.Tensor:
    """
    Load concept weights from binary classification probes.
    
    Args:
        weights_dir: Directory containing the probe weight files
        dataset_name: Name of the dataset (e.g., 'cifar10')
        balance_type: Balance type used during training ('undersample', 'balanced_loss', 'none')
        num_classes: Number of classes in the dataset
        
    Returns:
        Tensor of shape [num_classes, embedding_dim] containing concept vectors
    """
    concept_vectors = []
    
    for class_idx in range(num_classes):
        # Construct filename with balance_type suffix
        if balance_type == "undersample":
            suffix = "_undersample"
        elif balance_type == "balanced_loss":
            suffix = "_balanced_loss"
        elif balance_type == "none":
            suffix = "_none"
        else:
            suffix = ""
            
        weight_file = os.path.join(weights_dir, f"{dataset_name}_class{class_idx}{suffix}.pth")
        
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file not found: {weight_file}")
        
        # Load the state dict
        state_dict = torch.load(weight_file, map_location='cpu')
        weights = state_dict['weight']  # Shape: [num_output_classes, embedding_dim]
        
        # For binary classification, we expect weights to have shape [2, embedding_dim]
        # For multi-class (when class_idx was not used), it will be [num_classes, embedding_dim]
        if weights.shape[0] == 2:
            # Binary classification: concept vector = W[1] - W[0]
            # W[1] corresponds to the target class, W[0] to "not target class"
            concept_vector = weights[1] - weights[0]
        else:
            raise ValueError(f"Unexpected weight shape: {weights.shape}. Expected [2, embedding_dim] for binary classification or [10, embedding_dim] for legacy multi-class.")
        
        concept_vectors.append(concept_vector)
    
    return torch.stack(concept_vectors)


def compute_visual_similarity_matrix(concept_vectors: torch.Tensor, normalize_input_weights: bool = False) -> np.ndarray:
    """
    Compute visual similarity matrix from concept vectors.
    
    Args:
        concept_vectors: Tensor of shape [num_classes, embedding_dim]
        normalize_input_weights: Whether to normalize the input concept vectors using L2 norm
        
    Returns:
        Similarity matrix of shape [num_classes, num_classes]
    """
    # Optionally normalize input concept vectors (weights) using L2 norm
    if normalize_input_weights:
        print("  Normalizing input concept vectors (weights) using L2 norm...")
        concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=1)
        print(f"  Normalized weights range: [{concept_vectors.min().item():.3f}, {concept_vectors.max().item():.3f}]")
    
    # Normalize concept vectors for cosine similarity computation
    concept_vectors_norm = torch.nn.functional.normalize(concept_vectors, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(concept_vectors_norm, concept_vectors_norm.T)
    
    return similarity_matrix.numpy()


def compute_text_similarity_matrix(class_names: List[str], model_name: str = "clip_vitb32", device: str = 'cpu') -> np.ndarray:
    """
    Compute text similarity matrix from text embeddings of class names.
    
    Args:
        class_names: List of class names
        model_name: Name of the text encoder model (e.g., 'clip_vitb32')
        device: Device to run the model on
        
    Returns:
        Text similarity matrix of shape [num_classes, num_classes]
    """
    # Get complete text encoder (includes tokenization)
    text_encoder = get_text_encoder(model_name, device)
    
    # Create prompts for each class
    prompts = [f"a photo of a {name}" for name in class_names]
    
    # Get text embeddings (tokenization is handled internally)
    text_features = text_encoder(prompts)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(text_features, text_features.T)
    
    return similarity_matrix.cpu().numpy()


def load_training_embeddings(dataset_name: str, image_model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load training embeddings and labels for computing mean representations.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100')
        image_model_name: Name of the image model (e.g., 'clip_vitb32', 'dinov2_vitb14')
        
    Returns:
        Tuple of (embeddings, labels) tensors
    """
    train_embeddings_file = os.path.join(EMBEDDINGS_DIR, f"{dataset_name}_{image_model_name}_train.pt")
    train_labels_file = os.path.join(EMBEDDINGS_DIR, f"{dataset_name}_{image_model_name}_train_labels.pt")
    
    if not os.path.exists(train_embeddings_file):
        raise FileNotFoundError(f"Training embeddings not found: {train_embeddings_file}")
    if not os.path.exists(train_labels_file):
        raise FileNotFoundError(f"Training labels not found: {train_labels_file}")
    
    embeddings = torch.load(train_embeddings_file, map_location='cpu')
    labels = torch.load(train_labels_file, map_location='cpu')
    
    return embeddings, labels


def compute_mean_representations(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute mean representation for each class.
    
    Args:
        embeddings: Tensor of shape [num_samples, embedding_dim]
        labels: Tensor of shape [num_samples] with class labels
        num_classes: Number of classes
        
    Returns:
        Tensor of shape [num_classes, embedding_dim] containing mean representations
    """
    embedding_dim = embeddings.shape[1]
    mean_representations = torch.zeros(num_classes, embedding_dim)
    
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() == 0:
            raise ValueError(f"No samples found for class {class_idx}")
        
        class_embeddings = embeddings[class_mask]
        mean_representations[class_idx] = class_embeddings.mean(dim=0)
    
    return mean_representations


def compute_visual_similarity_matrix_from_mean_representations(dataset_name: str, image_model_name: str, num_classes: int, normalize_input_weights: bool = False) -> np.ndarray:
    """
    Compute visual similarity matrix from mean class representations.
    
    Args:
        dataset_name: Name of the dataset
        image_model_name: Name of the image model
        num_classes: Number of classes
        normalize_input_weights: Whether to normalize the input mean representations using L2 norm
        
    Returns:
        Similarity matrix of shape [num_classes, num_classes]
    """
    # Load training embeddings and labels
    print("Loading training embeddings...")
    embeddings, labels = load_training_embeddings(dataset_name, image_model_name)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Compute mean representations for each class
    print("Computing mean representations for each class...")
    mean_representations = compute_mean_representations(embeddings, labels, num_classes)
    print(f"Mean representations shape: {mean_representations.shape}")
    
    # Optionally normalize input mean representations using L2 norm
    if normalize_input_weights:
        print("  Normalizing input mean representations using L2 norm...")
        mean_representations = torch.nn.functional.normalize(mean_representations, dim=1)
        print(f"  Normalized representations range: [{mean_representations.min().item():.3f}, {mean_representations.max().item():.3f}]")
    
    # Normalize mean representations for cosine similarity computation
    mean_representations_norm = torch.nn.functional.normalize(mean_representations, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(mean_representations_norm, mean_representations_norm.T)
    
    return similarity_matrix.numpy()


def main():
    parser = argparse.ArgumentParser(description="Build similarity matrices from probe weights")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       help="Dataset name (default: cifar10)")
    parser.add_argument("--weights_dir", type=str, default="weights",
                       help="Directory containing probe weights (default: weights)")
    parser.add_argument("--output_dir", type=str, default="analysis",
                       help="Output directory for similarity matrices (default: analysis)")
    parser.add_argument("--image_model_name", type=str, default=None,
                       help="Image model name used for training the probes (optional). Supports CLIP models (clip_vitb32, clip_vitb16, clip_vitl14, clip_vitl14_336px) and DINOv2 (dinov2_vitb14). If not provided, visual similarity matrix will not be computed.")
    parser.add_argument("--text_model_name", type=str, default=None,
                       help="Text model name for encoding class names (optional). Supports CLIP models (clip_vitb32, clip_vitb16, clip_vitl14, clip_vitl14_336px) and Sentence Transformers (all-roberta-large-v1, all-mpnet-base-v2, all-MiniLM-L6-v2). If not provided, text similarity matrix will not be computed.")
    parser.add_argument("--image_sim_type", type=str, default="weights",
                       choices=["weights", "mean_representation"],
                       help="Type of visual similarity computation: 'weights' uses probe weights, 'mean_representation' uses mean class embeddings (default: weights)")
    parser.add_argument("--balance_type", type=str, default="undersample", 
                       choices=["undersample", "balanced_loss", "none"],
                       help="Balance type used during training (default: undersample)")
    parser.add_argument("--normalize_weights", action="store_true",
                       help="Normalize the concept vectors (weights) using L2 norm before computing similarity")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for text encoder model (default: cpu)")
    
    args = parser.parse_args()
    
    # Validate that at least one model is provided
    if args.image_model_name is None and args.text_model_name is None:
        print("❌ Error: At least one model must be provided!")
        print("Please specify either --image_model_name, --text_model_name, or both.")
        print("\nImage models supported:")
        print("  - CLIP: clip_vitb32, clip_vitb16, clip_vitl14, clip_vitl14_336px")
        print("  - DINOv2: dinov2_vitb14")
        print("\nText models supported:")
        print("  - CLIP: clip_vitb32, clip_vitb16, clip_vitl14, clip_vitl14_336px")
        print("  - Sentence Transformers: all-roberta-large-v1, all-mpnet-base-v2, all-MiniLM-L6-v2")
        return
    
    # Check for ImageNet-1k with image model (not supported for visual similarity)
    if args.dataset.startswith("imagenet1k") and args.image_model_name is not None:
        print("⚠️  Warning: ImageNet-1k visual similarity computation is not supported in this script!")
        print("   Visual similarity matrices for ImageNet-1k should be built using a separate script.")
        print("   Only text similarity matrix will be computed.")
        print("   Setting image_model_name to None...")
        args.image_model_name = None
    
    # Re-validate that at least one model is still provided after ImageNet check
    if args.image_model_name is None and args.text_model_name is None:
        print("❌ Error: No valid models specified!")
        print("For ImageNet-1k, please specify a text model for text similarity computation.")
        return
    
    # Update weights_dir based on image_model_name if provided (only needed for weights-based similarity)
    if args.image_model_name is not None and args.image_sim_type == "weights":
        args.weights_dir = os.path.join(WEIGHTS_DIR, args.image_model_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get class names
    class_names = get_label_names(args.dataset)
    num_classes = len(class_names)
    
    print(f"Processing {args.dataset} with {num_classes} classes")
    if args.image_model_name:
        print(f"Image model (for visual similarity): {args.image_model_name}")
        print(f"Visual similarity type: {args.image_sim_type}")
        print(f"Normalize weights: {'Yes' if args.normalize_weights else 'No'}")
    else:
        print("Image model: Not specified - visual similarity matrix will not be computed")
    if args.text_model_name:
        print(f"Text model (for text similarity): {args.text_model_name}")
    else:
        print("Text model: Not specified - text similarity matrix will not be computed")
    print(f"Class names: {class_names}")
    
    # Load probe weights and compute concept vectors (only if image model is provided)
    visual_similarity = None
    if args.image_model_name is not None:
        try:
            if args.image_sim_type == "weights":
                print("Loading probe weights...")
                concept_vectors = load_probe_weights(args.weights_dir, args.dataset, args.balance_type, num_classes)
                print(f"Loaded concept vectors with shape: {concept_vectors.shape}")
                
                # Compute visual similarity matrix from weights
                print("Computing visual similarity matrix from probe weights...")
                visual_similarity = compute_visual_similarity_matrix(concept_vectors, args.normalize_weights)
                
            elif args.image_sim_type == "mean_representation":
                print("Computing visual similarity matrix from mean class representations...")
                visual_similarity = compute_visual_similarity_matrix_from_mean_representations(
                    args.dataset, args.image_model_name, num_classes, args.normalize_weights)
            
            print(f"Visual similarity matrix shape: {visual_similarity.shape}")
            print(f"Visual similarity range: [{visual_similarity.min():.3f}, {visual_similarity.max():.3f}]")
            
        except Exception as e:
            print(f"Error computing visual similarity: {e}")
            visual_similarity = None
    else:
        print("Skipping visual similarity matrix computation (no image model specified)")

    # Compute text similarity matrix (only if text model is provided)
    text_similarity = None
    if args.text_model_name is not None:
        print("Computing text similarity matrix...")
        print(f"Using text model: {args.text_model_name}")
        try:
            text_similarity = compute_text_similarity_matrix(class_names, args.text_model_name, args.device)
            print(f"Text similarity matrix shape: {text_similarity.shape}")
            print(f"Text similarity range: [{text_similarity.min():.3f}, {text_similarity.max():.3f}]")
        except Exception as e:
            print(f"Error computing text similarity: {e}")
            text_similarity = None
    else:
        print("Skipping text similarity matrix computation (no text model specified)")
    
    # Save similarity matrices
    # Balance suffix only for visual similarity (depends on probe training)
    # Text similarity and class names don't need balance suffix (only depend on class names)
    visual_balance_suffix = f"_{args.balance_type}" if visual_similarity is not None else ""
    normalized_suffix = "_normalized" if args.normalize_weights else ""
    saved_files = []
    
    if visual_similarity is not None:
        image_model_suffix = f"_{args.image_model_name}"
        sim_type_suffix = f"_{args.image_sim_type}"
        visual_output_path = os.path.join(args.output_dir, f"{args.dataset}{visual_balance_suffix}{image_model_suffix}{sim_type_suffix}{normalized_suffix}_visual_similarity.npy")
        np.save(visual_output_path, visual_similarity)
        print(f"✅ Saved visual similarity matrix to: {visual_output_path}")
        saved_files.append(visual_output_path)
    
    if text_similarity is not None:
        text_model_suffix = f"_{args.text_model_name}"
        text_output_path = os.path.join(args.output_dir, f"{args.dataset}{text_model_suffix}_text_similarity.npy")
        np.save(text_output_path, text_similarity)
        print(f"✅ Saved text similarity matrix to: {text_output_path}")
        saved_files.append(text_output_path)
    
    # Save class names for reference
    class_names_path = os.path.join(args.output_dir, f"{args.dataset}_class_names.npy")
    np.save(class_names_path, np.array(class_names))
    print(f"✅ Saved class names to: {class_names_path}")
    saved_files.append(class_names_path)
    
    if not saved_files:
        print("❌ No similarity matrices were computed successfully!")
    else:
        print(f"\n🎉 Similarity matrix construction complete! Generated {len(saved_files)} files.")
        if visual_similarity is not None and text_similarity is not None:
            print("Both visual and text similarity matrices were computed successfully.")
        elif visual_similarity is not None:
            print("Only visual similarity matrix was computed.")
        elif text_similarity is not None:
            print("Only text similarity matrix was computed.")


if __name__ == "__main__":
    main()
