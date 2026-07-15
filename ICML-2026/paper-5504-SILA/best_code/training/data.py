#!/usr/bin/env python3
"""Dataset and data loading for SelfIE bias vector training."""

import json
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VectorLabelDataset(Dataset):
    """
    Dataset of (vector, label) pairs.
    
    Loads vectors from multiple .pt files and labels from .json file.
    Each item returns a single (vector, label) pair.
    """
    
    def __init__(
        self,
        vectors_by_dataset: Dict[str, torch.Tensor],
        vector_indices: List[int],
        dataset_names: List[str],
        labels: List[str],
        eos_token: str = "<|eot_id|>",
    ):
        """
        Args:
            vectors_by_dataset: Dict mapping dataset_name to vectors tensor (num_vectors, dim)
            vector_indices: List of vector indices for each label (indices within each dataset)
            dataset_names: List of dataset names for each label
            labels: List of label strings (one per training example)
            eos_token: EOS token to append to labels
        """
        assert len(vector_indices) == len(labels), "Mismatch between vector_indices and labels"
        assert len(dataset_names) == len(labels), "Mismatch between dataset_names and labels"
        
        self.vectors_by_dataset = vectors_by_dataset
        self.vector_indices = vector_indices
        self.dataset_names = dataset_names
        self.labels = labels
        self.eos_token = eos_token
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, str]:
        """
        Returns:
            vector: Tensor of shape (dim,)
            label: String with closing quote and EOS token appended
            vector_idx: Integer vector index
            dataset_name: String dataset name
        """
        vector_idx = self.vector_indices[idx]
        dataset_name = self.dataset_names[idx]
        label = self.labels[idx]
        
        vectors = self.vectors_by_dataset[dataset_name]
        vector = vectors[vector_idx]
        
        # Append closing quote and EOS token
        label_with_eos = label + '"' + self.eos_token
        
        return vector, label_with_eos, vector_idx, dataset_name


def collate_fn(batch):
    """
    Collate function for VectorLabelDataset.
    
    Args:
        batch: List of (vector, label, vector_idx, dataset_name) tuples
    
    Returns:
        Tuple of (stacked_vectors, list_of_labels, stacked_vector_indices, list_of_dataset_names)
    """
    vectors, labels, indices, dataset_names = zip(*batch)
    
    stacked_vectors = torch.stack(vectors, dim=0)
    stacked_indices = torch.tensor(indices, dtype=torch.long)
    
    return stacked_vectors, list(labels), stacked_indices, list(dataset_names)


def load_data(
    labels_file: str,
    seed: int = 42,
    dataset_names_to_load: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[dict]]:
    """
    Load vectors and labels from files.
    
    The JSON file contains an array of dataset sections, each with:
    - metadata: including dataset_name, filename (path to .pt file), layer, source, vector_type
    - vectors: list of {index, labels, split} dicts
    
    Args:
        labels_file: Path to .json file containing list of dataset sections
        seed: Random seed (not used here, but kept for consistency)
        dataset_names_to_load: Optional list of dataset patterns to load (supports @l{layer} suffix)
            If None, loads all datasets in the file
    
    Returns:
        Tuple of (
            dict mapping composite_key (dataset_name@l{layer}) to vectors tensor,
            list of vector data dicts with added 'composite_key' field
        )
    """
    # Parse dataset names to load, separating base name from layer suffix
    datasets_to_load_parsed: Dict[str, Optional[set]] = {}
    
    if dataset_names_to_load is not None:
        for dataset_spec in dataset_names_to_load:
            if "@l" in dataset_spec:
                base_name, layer_part = dataset_spec.rsplit("@l", 1)
                try:
                    layer = int(layer_part)
                    if base_name not in datasets_to_load_parsed:
                        datasets_to_load_parsed[base_name] = set()
                    layers_set = datasets_to_load_parsed[base_name]
                    if layers_set is not None:
                        layers_set.add(layer)
                except ValueError:
                    raise ValueError(f"Invalid layer specification in '{dataset_spec}'. Expected format: 'dataset_name@l{{number}}'")
            else:
                # No layer specified, load all layers for this dataset
                if dataset_spec not in datasets_to_load_parsed:
                    datasets_to_load_parsed[dataset_spec] = None
    
    # Load labels JSON
    print(f"Loading labels from: {labels_file}")
    with open(labels_file, "r") as f:
        file_data = json.load(f)
    
    if not isinstance(file_data, list):
        raise ValueError(f"Expected labels file to contain a list, got {type(file_data)}")
    
    if len(file_data) == 0:
        raise ValueError("Labels file contains an empty list")
    
    # Get the directory containing the labels file for resolving relative paths
    labels_dir = os.path.dirname(os.path.abspath(labels_file))
    
    # Load vectors from each dataset section
    vectors_by_dataset = {}
    all_labels_data = []
    
    for section_idx, section in enumerate(file_data):
        if not isinstance(section, dict):
            raise ValueError(f"Section {section_idx} should be a dict, got {type(section)}")
        
        if "metadata" not in section:
            raise ValueError(f"Section {section_idx} missing 'metadata' field")
        
        if "vectors" not in section:
            raise ValueError(f"Section {section_idx} missing 'vectors' field")
        
        metadata = section["metadata"]
        vectors_list = section["vectors"]
        
        # Validate metadata structure
        if not isinstance(metadata, dict):
            raise ValueError(f"Section {section_idx}: 'metadata' should be a dict, got {type(metadata)}")
        
        required_metadata_fields = ["dataset_name", "filename", "layer", "source", "vector_type"]
        for field in required_metadata_fields:
            if field not in metadata:
                raise ValueError(f"Section {section_idx}: metadata missing required field '{field}'")
        
        dataset_name = metadata["dataset_name"]
        layer = metadata["layer"]
        
        # Create composite key: dataset_name@l{layer}
        composite_key = f"{dataset_name}@l{layer}"
        
        # Check if we should load this dataset+layer combination
        should_load = False
        if dataset_names_to_load is None:
            should_load = True
        elif dataset_name in datasets_to_load_parsed:
            layers_to_load = datasets_to_load_parsed[dataset_name]
            if layers_to_load is None:
                should_load = True
            elif layer in layers_to_load:
                should_load = True
        
        if not should_load:
            print(f"  Skipping dataset '{composite_key}' (not in requested datasets)")
            continue
        
        vectors_file = metadata["filename"]
        
        # Resolve relative paths relative to the labels file directory
        if not os.path.isabs(vectors_file):
            vectors_file = os.path.join(labels_dir, vectors_file)
        
        # Load vectors for this dataset
        print(f"  Loading vectors from: {vectors_file}")
        vectors = torch.load(vectors_file, map_location="cpu", weights_only=True)
        
        if not isinstance(vectors, torch.Tensor):
            raise ValueError(f"Expected .pt file to contain a torch.Tensor, got {type(vectors)}")
        
        if vectors.ndim != 2:
            raise ValueError(f"Expected vectors to be 2D (num_vectors, dim), got shape {vectors.shape}")
        
        num_vectors, dim = vectors.shape
        print(f"    ✓ Loaded {num_vectors} vectors of dimension {dim} for dataset '{composite_key}'")
        
        # Store with composite key
        vectors_by_dataset[composite_key] = vectors
        
        # Validate vectors list
        if not isinstance(vectors_list, list):
            raise ValueError(f"Section {section_idx}: 'vectors' should be a list, got {type(vectors_list)}")
        
        # Validate each vector entry and add dataset_name
        # Only include vectors from datasets that were loaded
        for i, item in enumerate(vectors_list):
            if not isinstance(item, dict):
                raise ValueError(f"Section {section_idx}, vector item {i} should be a dict, got {type(item)}")
            if "index" not in item:
                raise ValueError(f"Section {section_idx}, vector item {i} missing 'index' field")
            if "labels" not in item:
                raise ValueError(f"Section {section_idx}, vector item {i} missing 'labels' field")
            if "split" not in item:
                raise ValueError(f"Section {section_idx}, vector item {i} missing 'split' field")
            if not isinstance(item["labels"], list):
                raise ValueError(f"Section {section_idx}, vector item {i}: 'labels' should be a list, got {type(item['labels'])}")
            
            # Check index is valid
            idx = item["index"]
            if idx < 0 or idx >= num_vectors:
                raise ValueError(f"Section {section_idx}, vector item {i}: index {idx} out of range [0, {num_vectors})")
            
            # Check split is valid
            split = item["split"]
            if split not in ["train", "val"]:
                raise ValueError(f"Section {section_idx}, vector item {i}: split must be 'train' or 'val', got '{split}'")
            
            # Add composite key to the item
            item_with_dataset = item.copy()
            item_with_dataset["dataset_name"] = composite_key  # Use composite key for consistency
            all_labels_data.append(item_with_dataset)
        
        # Print section info
        total_section_labels = sum(len(item["labels"]) for item in vectors_list)
        train_count = sum(1 for item in vectors_list if item["split"] == "train")
        val_count = sum(1 for item in vectors_list if item["split"] == "val")
        print(f"  Section {section_idx} ({composite_key}): {len(vectors_list)} vector groups ({train_count} train, {val_count} val), {total_section_labels} labels")
    
    total_labels = sum(len(item["labels"]) for item in all_labels_data)
    num_loaded = len(vectors_by_dataset)
    print(f"✓ Loaded {num_loaded} dataset(s) (with layers), {len(all_labels_data)} total vector groups, {total_labels} total labels")
    
    return vectors_by_dataset, all_labels_data


def _matches_dataset_pattern(composite_key: str, pattern: str) -> bool:
    """
    Check if a composite key matches a dataset pattern.
    
    Pattern can be:
    - "dataset_name@l{layer}" - exact match
    - "dataset_name" - matches any layer of this dataset
    
    Args:
        composite_key: Key like "dataset_name@l19"
        pattern: Pattern to match
    
    Returns:
        True if pattern matches the composite key
    """
    if pattern == composite_key:
        # Exact match
        return True
    
    # Check if pattern is a base name (no layer specified)
    if "@l" not in pattern:
        # Pattern is base name, check if composite_key starts with it
        if "@l" in composite_key:
            base_name = composite_key.rsplit("@l", 1)[0]
            return base_name == pattern
    
    return False


def subsample_by_ratios(
    labels_data: List[dict],
    dataset_ratios: Dict[str | tuple, float],
    available_datasets: set,
    seed: int = 42,
    epoch: int = 0,
    debug: bool = False,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Subsample datasets according to specified ratios.
    
    Strategy: Epoch = one pass through smallest dataset/group (at the EXAMPLE level)
    - First flatten vector groups into individual examples (one per label)
    - Groups can be specified using tuples as keys (datasets in same group are concatenated)
    - Normalize ratios to sum to 1
    - Compute target counts for each group at the example level
    - Target = (smallest_group_size / its_ratio) * each_ratio
    - Subsample larger groups randomly
    
    Args:
        labels_data: List of label items with 'dataset_name', 'labels' fields
        dataset_ratios: Dict of dataset patterns (str or tuple) to ratios
                       Tuple keys group multiple datasets (concatenated without subsampling)
        available_datasets: Set of available composite keys
        seed: Random seed
        epoch: Current epoch number (used for resampling)
        debug: Print detailed debug information
    
    Returns:
        Tuple of (subsampled_items, counts_dict mapping dataset_name to example count)
    """
    # Parse dataset_ratios, handling both string and tuple keys
    # Build group_id -> (list of dataset names, ratio)
    groups = {}  # group_id -> {"datasets": [...], "ratio": float}
    group_id = 0
    
    for pattern_or_patterns, ratio in dataset_ratios.items():
        # Normalize to tuple
        if isinstance(pattern_or_patterns, str):
            patterns = (pattern_or_patterns,)
            group_name = pattern_or_patterns  # Use pattern as group name for single-dataset groups
        else:
            patterns = pattern_or_patterns
            group_name = f"group_{group_id}"  # Generate name for multi-dataset groups
        
        # Match each pattern to actual datasets
        matched_datasets = []
        for pattern in patterns:
            matches = [key for key in available_datasets if _matches_dataset_pattern(key, pattern)]
            if not matches:
                raise ValueError(f"No datasets match pattern '{pattern}'. Available: {sorted(available_datasets)}")
            matched_datasets.extend(matches)
        
        if not matched_datasets:
            raise ValueError(f"No datasets matched for pattern(s) '{patterns}'")
        
        groups[group_name] = {"datasets": matched_datasets, "ratio": ratio}
        group_id += 1
    
    # Normalize ratios
    total_ratio = sum(g["ratio"] for g in groups.values())
    for group_info in groups.values():
        group_info["normalized_ratio"] = group_info["ratio"] / total_ratio
    
    # Extract and validate split (all items MUST have the same split)
    if not labels_data:
        raise ValueError("subsample_by_ratios() called with empty labels_data")
    
    # Check first item has split field
    if "split" not in labels_data[0]:
        raise ValueError(f"subsample_by_ratios() called with item missing 'split' field: {labels_data[0]}")
    
    split = labels_data[0]["split"]
    
    # Validate split value
    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split value in labels_data[0]: '{split}' (must be 'train' or 'val')")
    
    # Validate all items have the same split
    for i, item in enumerate(labels_data):
        if "split" not in item:
            raise ValueError(f"Item {i} in labels_data missing 'split' field: {item}")
        item_split = item["split"]
        if item_split != split:
            raise ValueError(
                f"Inconsistent splits in labels_data passed to subsample_by_ratios(): "
                f"item 0 has split='{split}', item {i} has split='{item_split}'. "
                f"This function must be called with either all 'train' or all 'val' items."
            )
    
    # CRITICAL: Flatten vector groups into individual examples FIRST
    # This ensures ratios work at the example level, not vector group level
    # (Important because different datasets may have different labels-per-vector)
    
    # Flatten: create individual (dataset_name, index, label) tuples
    flattened_examples = []  # List of {"dataset_name": str, "index": int, "label": str}
    for item in labels_data:
        dataset_name = item['dataset_name']
        vector_idx = item['index']
        for label in item['labels']:
            flattened_examples.append({
                'dataset_name': dataset_name,
                'index': vector_idx,
                'label': label,
            })
    
    # Group flattened examples by dataset, then by group
    dataset_examples = {}  # dataset_name -> list of flattened examples
    for example in flattened_examples:
        dataset_name = example['dataset_name']
        # Check if this dataset belongs to any group
        for group_info in groups.values():
            if dataset_name in group_info["datasets"]:
                if dataset_name not in dataset_examples:
                    dataset_examples[dataset_name] = []
                dataset_examples[dataset_name].append(example)
                break
    
    # Compute total examples per group
    group_examples = {}  # group_name -> list of all examples from all datasets in group
    group_counts = {}  # group_name -> total example count
    
    for group_name, group_info in groups.items():
        all_examples = []
        for dataset_name in group_info["datasets"]:
            if dataset_name in dataset_examples:
                all_examples.extend(dataset_examples[dataset_name])
        group_examples[group_name] = all_examples
        group_counts[group_name] = len(all_examples)
    
    # Find the constraining group (smallest relative to its ratio)
    min_scaled_size = float('inf')
    for group_name, count in group_counts.items():
        normalized_ratio = groups[group_name]["normalized_ratio"]
        scaled_size = count / normalized_ratio
        if scaled_size < min_scaled_size:
            min_scaled_size = scaled_size
    
    # Calculate target count for each group
    group_target_counts = {}
    for group_name in group_counts:
        normalized_ratio = groups[group_name]["normalized_ratio"]
        target_count = int(min_scaled_size * normalized_ratio)
        group_target_counts[group_name] = target_count
    
    # Subsample each group to target count (at example level)
    rng = np.random.RandomState(seed + epoch)  # Different seed per epoch
    subsampled_examples = []
    
    for group_name in sorted(group_examples.keys()):  # Sort for determinism
        examples = group_examples[group_name]
        target = group_target_counts[group_name]
        
        if len(examples) <= target:
            # Use all examples from this group
            subsampled_examples.extend(examples)
        else:
            # Subsample this group
            indices = rng.choice(len(examples), size=target, replace=False)
            subsampled_examples.extend([examples[i] for i in indices])
    
    # Now convert back to the original format (group by dataset_name and index)
    # This reconstructs vector groups from the sampled examples
    vector_group_dict = {}  # (dataset_name, index) -> list of labels
    
    for example in subsampled_examples:
        key = (example['dataset_name'], example['index'])
        if key not in vector_group_dict:
            vector_group_dict[key] = []
        vector_group_dict[key].append(example['label'])
    
    # Convert back to original item format
    subsampled_items = []
    for (dataset_name, index), labels in vector_group_dict.items():
        subsampled_items.append({
            'dataset_name': dataset_name,
            'index': index,
            'labels': labels,
            'split': split,  # Use the split from input data (validated to be consistent)
        })
    
    # Build counts dict by individual dataset name for reporting (at example level)
    dataset_counts_output = {}
    for dataset_name in dataset_examples.keys():
        # Count how many examples from this dataset made it into subsampled_examples
        count = sum(1 for example in subsampled_examples if example['dataset_name'] == dataset_name)
        dataset_counts_output[dataset_name] = count
    
    # Debug output
    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: Dataset Mixing Details (Epoch {epoch})")
        print(f"{'='*80}")
        
        for group_name in sorted(group_examples.keys()):
            group_info = groups[group_name]
            datasets_in_group = group_info["datasets"]
            original_count = group_counts[group_name]
            target_count = group_target_counts[group_name]
            
            print(f"\nGroup: {group_name}")
            print(f"  Datasets: {datasets_in_group}")
            print(f"  Ratio: {group_info['ratio']} (normalized: {group_info['normalized_ratio']:.4f})")
            print(f"  Original size: {original_count} examples")
            print(f"  Target size: {target_count} examples")
            print(f"  Action: {'Using all' if original_count <= target_count else f'Subsampling {target_count}/{original_count}'}")
            
            # Show example counts from each dataset in this group
            for dataset_name in datasets_in_group:
                if dataset_name in dataset_examples:
                    examples_from_dataset = [ex for ex in subsampled_examples if ex['dataset_name'] == dataset_name]
                    if examples_from_dataset:
                        # Show first few example labels
                        sample_examples = examples_from_dataset[:5]
                        labels_preview = [ex['label'][:50] for ex in sample_examples]
                        labels_str = ' | '.join([f'"{l}..."' for l in labels_preview])
                        print(f"  {dataset_name}: {len(examples_from_dataset)} examples")
                        print(f"    First labels: {labels_str}")
        
        print(f"\n{'='*80}")
        print(f"Total examples this epoch: {len(subsampled_examples)}")
        print(f"Total vector groups this epoch: {len(subsampled_items)}")
        print(f"{'='*80}\n")
    
    return subsampled_items, dataset_counts_output


def create_datasets(
    vectors_by_dataset: Dict[str, torch.Tensor],
    labels_data: List[dict],
    train_dataset_ratios: Optional[Dict[str | tuple, float]] = None,
    val_dataset_ratios: Optional[Dict[str | tuple, float]] = None,
    seed: int = 42,
    epoch: int = 0,
    eos_token: str = "<|eot_id|>",
    strip_labels: bool = False,
    debug: bool = False,
) -> Tuple[VectorLabelDataset, Optional[VectorLabelDataset], Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Create train and validation datasets from vectors and labels.
    
    Strategy:
    1. Filter by split property ("train" or "val")
    2. Optionally subsample by ratios (supports grouping datasets with tuples)
    3. Flatten into individual (vector, label) examples
    
    Args:
        vectors_by_dataset: Dict mapping composite keys (dataset_name@l{layer}) to vectors tensor
        labels_data: List of {index, labels, split, dataset_name} dicts
        train_dataset_ratios: Optional dict of dataset patterns (str or tuple) to mixing ratios for training
                             Tuple keys group multiple datasets (concatenated without subsampling within group)
        val_dataset_ratios: Optional dict of dataset patterns (str or tuple) to mixing ratios for validation
        seed: Random seed
        epoch: Current epoch number (for resampling larger datasets in training)
        eos_token: EOS token to append to labels
        strip_labels: Whether to strip whitespace from labels
    
    Returns:
        Tuple of (train_dataset, val_dataset, train_dataset_counts, val_dataset_counts).
        val_dataset is None if no val data exists.
        train_dataset_counts/val_dataset_counts are None if no ratios specified for that split.
    """
    # Get available datasets
    available_datasets = set(vectors_by_dataset.keys())
    
    # Filter for train split
    train_items = [item for item in labels_data if item["split"] == "train"]
    
    # Apply ratio-based subsampling for training if specified
    train_dataset_counts = None
    if train_dataset_ratios is not None:
        train_items, train_dataset_counts = subsample_by_ratios(
            train_items,
            train_dataset_ratios,
            available_datasets,
            seed=seed,
            epoch=epoch,
            debug=debug,
        )
        
        # Print subsampling info (brief version if not debug mode)
        if not debug:
            print(f"Applied train ratio-based subsampling (epoch {epoch}):")
            for dataset_name, count in sorted(train_dataset_counts.items()):
                print(f"  {dataset_name}: {count} examples")
    
    # Filter for val split
    val_items = [item for item in labels_data if item["split"] == "val"]
    
    # Apply ratio-based subsampling for validation if specified
    val_dataset_counts = None
    if val_dataset_ratios is not None:
        val_items, val_dataset_counts = subsample_by_ratios(
            val_items,
            val_dataset_ratios,
            available_datasets,
            seed=seed,
            epoch=0,  # Always use epoch 0 for validation (no resampling)
            debug=debug,
        )
        
        # Print subsampling info (brief version if not debug mode)
        if not debug:
            print(f"Applied val ratio-based subsampling:")
            for dataset_name, count in sorted(val_dataset_counts.items()):
                print(f"  {dataset_name}: {count} examples")
    
    # Flatten train split
    train_vector_indices = []
    train_dataset_names_list = []
    train_labels = []
    for item in train_items:
        vector_idx = item["index"]
        dataset_name = item["dataset_name"]
        for label in item["labels"]:
            train_vector_indices.append(vector_idx)
            train_dataset_names_list.append(dataset_name)
            train_labels.append(label.strip() if strip_labels else label)
    
    if len(train_items) == 0:
        raise ValueError("No training data found after filtering")
    
    train_dataset = VectorLabelDataset(
        vectors_by_dataset=vectors_by_dataset,
        vector_indices=train_vector_indices,
        dataset_names=train_dataset_names_list,
        labels=train_labels,
        eos_token=eos_token,
    )
    
    print(f"Train: {len(train_items)} vector groups → {len(train_dataset)} examples")
    
    # Flatten val split
    val_dataset = None
    if len(val_items) > 0:
        val_vector_indices = []
        val_dataset_names_list = []
        val_labels = []
        for item in val_items:
            vector_idx = item["index"]
            dataset_name = item["dataset_name"]
            for label in item["labels"]:
                val_vector_indices.append(vector_idx)
                val_dataset_names_list.append(dataset_name)
                val_labels.append(label.strip() if strip_labels else label)
        
        val_dataset = VectorLabelDataset(
            vectors_by_dataset=vectors_by_dataset,
            vector_indices=val_vector_indices,
            dataset_names=val_dataset_names_list,
            labels=val_labels,
            eos_token=eos_token,
        )
        
        print(f"Val: {len(val_items)} vector groups → {len(val_dataset)} examples")
    else:
        print("Val: No validation data found")
    
    return train_dataset, val_dataset, train_dataset_counts, val_dataset_counts


def create_dataloaders(
    labels_file: str,
    batch_size: int = 16,
    train_dataset_ratios: Optional[Dict[str | tuple, float]] = None,
    val_dataset_ratios: Optional[Dict[str | tuple, float]] = None,
    epoch: int = 0,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    eos_token: str = "<|eot_id|>",
    strip_labels: bool = False,
    debug: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], int, Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Create train and validation dataloaders.
    
    Args:
        labels_file: Path to .json file
        batch_size: Batch size
        train_dataset_ratios: Optional dict of dataset patterns (str or tuple) to mixing ratios for training
        val_dataset_ratios: Optional dict of dataset patterns (str or tuple) to mixing ratios for validation
        epoch: Current epoch number (for resampling larger datasets in training)
        shuffle: Whether to shuffle training data
        num_workers: Number of dataloader workers
        seed: Random seed
        eos_token: EOS token to append to labels
        strip_labels: Whether to strip whitespace from labels
    
    Returns:
        Tuple of (train_loader, val_loader, model_dim, train_dataset_counts, val_dataset_counts)
    """
    # Extract dataset names from ratios to only load what we need
    dataset_names_to_load = set()
    
    if train_dataset_ratios is not None:
        for key in train_dataset_ratios.keys():
            if isinstance(key, tuple):
                dataset_names_to_load.update(key)
            else:
                dataset_names_to_load.add(key)
    
    if val_dataset_ratios is not None:
        for key in val_dataset_ratios.keys():
            if isinstance(key, tuple):
                dataset_names_to_load.update(key)
            else:
                dataset_names_to_load.add(key)
    
    # Load data (only requested datasets if ratios specified)
    if dataset_names_to_load:
        print(f"Loading only datasets: {sorted(dataset_names_to_load)}")
        vectors_by_dataset, labels_data = load_data(labels_file, seed, list(dataset_names_to_load))
    else:
        vectors_by_dataset, labels_data = load_data(labels_file, seed)
    
    # Get model dimension (assume all datasets have same dimension)
    first_dataset_name = next(iter(vectors_by_dataset.keys()))
    model_dim = vectors_by_dataset[first_dataset_name].shape[1]
    
    # Verify all datasets have same dimension
    for dataset_name, vectors in vectors_by_dataset.items():
        if vectors.shape[1] != model_dim:
            raise ValueError(f"Dataset '{dataset_name}' has dimension {vectors.shape[1]}, expected {model_dim}")
    
    # Create datasets
    train_dataset, val_dataset, train_dataset_counts, val_dataset_counts = create_datasets(
        vectors_by_dataset, labels_data, train_dataset_ratios, val_dataset_ratios,
        seed, epoch, eos_token, strip_labels, debug
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    return train_loader, val_loader, model_dim, train_dataset_counts, val_dataset_counts
