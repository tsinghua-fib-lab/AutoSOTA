#!/usr/bin/env python3
"""
Master JSON Loader for SAE Label Datasets.

Loads data from the new master JSON format where vectors are stored in separate .pt files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch


def load_master_json(master_json_path: str) -> List[Dict]:
    """
    Load and parse the master JSON file.
    
    Args:
        master_json_path: Path to the master JSON file
        
    Returns:
        List of dataset sections (each with metadata and vectors)
    """
    with open(master_json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Master JSON must contain a list, got {type(data)}")
    
    return data


def find_dataset_section(
    master_json_data: List[Dict],
    dataset_name: str,
    layer: int
) -> Optional[Dict]:
    """
    Find the dataset section matching the given dataset_name and layer.
    
    Args:
        master_json_data: Parsed master JSON data
        dataset_name: Dataset name to search for (e.g., "Goodfire/Llama-3.1-8B-Instruct-SAE-l19")
        layer: Layer number to search for
        
    Returns:
        The matching dataset section, or None if not found
    """
    for section in master_json_data:
        metadata = section.get("metadata", {})
        if metadata.get("dataset_name") == dataset_name and metadata.get("layer") == layer:
            return section
    
    return None


def load_vectors_from_pt(pt_file_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load vectors from a .pt file.
    
    Args:
        pt_file_path: Path to the .pt file
        device: Device to load tensors to (can be "auto" to auto-detect)
        
    Returns:
        Tensor of shape (num_vectors, vector_dim)
    """
    # Resolve "auto" device to actual device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Try loading with weights_only=True first (more secure)
        vectors = torch.load(pt_file_path, map_location=device, weights_only=True)
    except RuntimeError as e:
        if "don't know how to restore data location" in str(e):
            # Fall back to loading to CPU first, then moving to target device
            # This handles checkpoints saved with 'auto' or other special location tags
            print(f"⚠️  Falling back to CPU loading due to map_location compatibility issue")
            vectors = torch.load(pt_file_path, map_location='cpu', weights_only=False)
            vectors = vectors.to(device)
        else:
            raise
    
    if not isinstance(vectors, torch.Tensor):
        raise ValueError(f"Expected .pt file to contain a tensor, got {type(vectors)}")
    
    return vectors


def get_label_dataset_from_master_json(
    master_json_path: str,
    dataset_name: str,
    layer: int,
    split: str = "val",
    data_volume_path: str = "/sae-data",
    max_latents: Optional[int] = None,
    specific_latent_indices: Optional[List[int]] = None
) -> List[Dict]:
    """
    Load label dataset (existing labels) from master JSON.
    
    Args:
        master_json_path: Path to master JSON file
        dataset_name: Dataset name to use
        layer: Layer number to use
        split: Which split to use ("train", "val", "test", or "all" for all splits)
        data_volume_path: Base path where .pt files are located
        max_latents: Maximum number of latents to load (None = all)
        specific_latent_indices: Specific latent indices to load (None = all, overrides max_latents)
        
    Returns:
        List of dicts with keys: latent_index, label
    """
    # Load master JSON
    master_data = load_master_json(master_json_path)
    
    # Find the correct section
    section = find_dataset_section(master_data, dataset_name, layer)
    if section is None:
        raise ValueError(
            f"Could not find dataset section with dataset_name='{dataset_name}' and layer={layer}"
        )
    
    # Convert specific_latent_indices to a set for O(1) lookups
    specific_indices_set = None
    if specific_latent_indices is not None:
        specific_indices_set = set(specific_latent_indices)
        print(f"🎯 Filtering to {len(specific_indices_set)} specific latent indices")
    
    # Extract labels for the specified split
    vectors = section.get("vectors", [])
    result = []
    latent_count = 0
    
    for vector_entry in vectors:
        # Check if this entry belongs to the requested split
        entry_split = vector_entry.get("split")
        if split != "all" and entry_split != split:
            continue
        
        latent_index = vector_entry.get("index")
        
        # If specific indices are specified, only include those
        if specific_indices_set is not None:
            if latent_index not in specific_indices_set:
                continue
        # Otherwise, check if we've reached max_latents limit
        elif max_latents is not None and latent_count >= max_latents:
            break
        
        labels = vector_entry.get("labels", [])
        
        # For label_dataset mode, we use the existing labels
        # If there are multiple labels, we create separate entries for each
        for label in labels:
            result.append({
                "latent_index": latent_index,
                "label": label
            })
        
        latent_count += 1
    
    if specific_latent_indices is not None:
        print(f"✓ Loaded {latent_count} latents from specific indices")
    elif max_latents is not None:
        print(f"✂️  Subsampled to {latent_count} latents (max_latents={max_latents})")
    
    return result


def get_vectors_from_master_json(
    master_json_path: str,
    dataset_name: str,
    layer: int,
    split: str = "val",
    data_volume_path: str = "/sae-data",
    device: str = "cuda",
    max_latents: Optional[int] = None,
    specific_latent_indices: Optional[List[int]] = None
) -> List[Dict]:
    """
    Load SAE vectors from master JSON for label generation.
    
    Args:
        master_json_path: Path to master JSON file
        dataset_name: Dataset name to use
        layer: Layer number to use
        split: Which split to use ("train", "val", "test", or "all" for all splits)
        data_volume_path: Base path where .pt files are located
        device: Device to load vectors to
        max_latents: Maximum number of latents to load (None = all)
        specific_latent_indices: Specific latent indices to load (None = all, overrides max_latents)
        
    Returns:
        List of dicts with keys: latent_index, sae_vector
    """
    # Load master JSON
    master_data = load_master_json(master_json_path)
    
    # Find the correct section
    section = find_dataset_section(master_data, dataset_name, layer)
    if section is None:
        raise ValueError(
            f"Could not find dataset section with dataset_name='{dataset_name}' and layer={layer}"
        )
    
    metadata = section.get("metadata", {})
    filename = metadata.get("filename")
    if not filename:
        raise ValueError(f"Dataset section missing 'filename' in metadata")
    
    # Construct full path to .pt file
    pt_file_path = os.path.join(data_volume_path, filename)
    
    print(f"📥 Loading vectors from: {pt_file_path}")
    
    # Load all vectors from .pt file
    all_vectors = load_vectors_from_pt(pt_file_path, device=device)
    
    print(f"✓ Loaded tensor with shape {all_vectors.shape}")
    
    # Convert specific_latent_indices to a set for O(1) lookups
    specific_indices_set = None
    if specific_latent_indices is not None:
        specific_indices_set = set(specific_latent_indices)
        print(f"🎯 Filtering to {len(specific_indices_set)} specific latent indices")
    
    # Extract vector entries for the specified split
    vectors_list = section.get("vectors", [])
    result = []
    latent_count = 0
    
    for vector_entry in vectors_list:
        # Check if this entry belongs to the requested split
        entry_split = vector_entry.get("split")
        if split != "all" and entry_split != split:
            continue
        
        latent_index = vector_entry.get("index")
        
        # If specific indices are specified, only include those
        if specific_indices_set is not None:
            if latent_index not in specific_indices_set:
                continue
        # Otherwise, check if we've reached max_latents limit
        elif max_latents is not None and latent_count >= max_latents:
            break
        
        # Get the vector from the loaded tensor
        # Assuming vectors are stored as rows where row i corresponds to latent index i
        if latent_index >= all_vectors.shape[0]:
            raise ValueError(
                f"Latent index {latent_index} exceeds tensor size {all_vectors.shape[0]}"
            )
        
        sae_vector = all_vectors[latent_index]
        
        result.append({
            "latent_index": latent_index,
            "sae_vector": sae_vector
        })
        
        latent_count += 1
    
    if specific_latent_indices is not None:
        print(f"✓ Loaded {latent_count} latents from specific indices")
    elif max_latents is not None:
        print(f"✂️  Subsampled to {latent_count} latents (max_latents={max_latents})")
    
    return result


def get_dataset_metadata(
    master_json_path: str,
    dataset_name: str,
    layer: int
) -> Dict:
    """
    Get metadata for a specific dataset section.
    
    Args:
        master_json_path: Path to master JSON file
        dataset_name: Dataset name to use
        layer: Layer number to use
        
    Returns:
        Metadata dictionary
    """
    master_data = load_master_json(master_json_path)
    section = find_dataset_section(master_data, dataset_name, layer)
    
    if section is None:
        raise ValueError(
            f"Could not find dataset section with dataset_name='{dataset_name}' and layer={layer}"
        )
    
    return section.get("metadata", {})

