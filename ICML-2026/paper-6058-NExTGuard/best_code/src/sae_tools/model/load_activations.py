"""
Data loading module - load SAE prediction results from files

This module provides the following functionality:
- Load SAE prediction results from .pt files
- Load and process SAE prediction results from files
"""

import os
from typing import List, Dict, Any, Tuple
import torch
import numpy as np

def load_sae_predictions_pt(file_path: str) -> Dict[str, Any]:
    """
    Read SAE prediction results from .pt files (generate_activations format).
    
    This function reads the sparse activation data from the .pt file saved by eval_sae.py, and returns the sparse format directly, without any format conversion.
    
    Args:
        file_path: .pt file path
        
    Returns:
        Dict[str, Any]: dictionary containing the following keys:
            - 'sparse_acts': torch.sparse_coo_tensor, shape: [Total_Samples * Length, num_features]
            - 'valid_token_idx': torch.Tensor (Int), shape: [Total_Samples, 2]
            - 'seq_lens': torch.Tensor (Int), shape: [Total_Samples]
            - 'shape': torch.Size
            
    Raises:
        ValueError: if the file format is incorrect or missing required keys
        FileNotFoundError: if the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the .pt file
    data = torch.load(file_path, map_location='cpu')
    
    # Check the required keys
    required_keys = ["sparse_acts", "valid_token_idx", "seq_lens"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Data format is incorrect, missing required keys: {missing_keys}")
    return data

def filter_data_by_label(
    sparse_data: Dict[str, Any],
    metadata_list: List[Dict[str, Any]],
    label_field: str = 'response_label',
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[int]]:
    """
    Filter out data without labels, and update the sparse activation data synchronously.
    
    This function will:
    1. Check if the specified label field in each metadata item is None
    2. Collect valid indices (data with labels)
    3. Filter metadata_list
    4. Synchronously filter sparse_data (including seq_lens, response_start_token_idx, sparse_acts)
    5. Re-map the token indices of sparse_acts
    
    Args:
        sparse_data: sparse activation data dictionary, containing 'sparse_acts', 'seq_lens', 'response_start_token_idx'
        metadata_list: metadata list
        label_field: the name of the label field to check, default is 'response_label'
        verbose: whether to print the filtering information
        
    Returns:
        Tuple[filtered sparse_data, filtered metadata_list, valid indices list]
        
    Raises:
        ValueError: if the length of sparse_data or metadata_list is not consistent
    """
    num_samples = len(sparse_data['seq_lens'])
    
    if len(metadata_list) != num_samples:
        raise ValueError(
            f"Data length is inconsistent: sparse_data has {num_samples} samples, "
            f"metadata_list has {len(metadata_list)} records"
        )
    
    temp_data_list = []
    valid_indices = []
    
    for i in range(num_samples):
        item = metadata_list[i].copy()
        if item.get(label_field) is not None:
            item['valid_token_idx'] = sparse_data['valid_token_idx'][i]
            temp_data_list.append(item)
            valid_indices.append(i)
    
    if verbose:
        print(f"Original data: {num_samples} records")
        print(f"Filtered (with labels): {len(temp_data_list)} records")
        print(f"Filtered out (without labels): {num_samples - len(temp_data_list)} records")
    
    # If there is no data to filter, return directly
    if len(valid_indices) == num_samples:
        return sparse_data, temp_data_list, valid_indices
    
    # Filter sparse_data based on valid indices
    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
    
    # Filter seq_lens and start_token_idx
    sparse_data_filtered = {
        'seq_lens': sparse_data['seq_lens'][valid_indices_tensor],
        'valid_token_idx': sparse_data['valid_token_idx'][valid_indices_tensor],
        'shape': sparse_data.get('shape', None),
    }
    
    # Re-build sparse_acts
    # sparse_acts is flat, shape: [Total_Tokens, num_features]
    # Calculate the token range for each sample based on seq_lens, then re-map
    sparse_acts = sparse_data['sparse_acts']
    seq_lens_old = sparse_data['seq_lens'].numpy()
    seq_lens_new = sparse_data_filtered['seq_lens'].numpy()
    
    # Calculate the token start position for each sample (in the old sparse_acts)
    sample_starts_old = np.concatenate([[0], np.cumsum(seq_lens_old[:-1])])
    sample_starts_new = np.concatenate([[0], np.cumsum(seq_lens_new[:-1])])
    
    # Extract the valid token range
    sparse_acts_indices = sparse_acts.indices()  # [2, NNZ]
    sparse_acts_values = sparse_acts.values()    # [NNZ]
    
    token_indices = sparse_acts_indices[0].numpy()  # [NNZ]
    feat_indices = sparse_acts_indices[1].numpy()   # [NNZ]
    
    # Find out which sample each token belongs to
    sample_boundaries_old = np.cumsum(seq_lens_old)
    sample_ids = np.searchsorted(sample_boundaries_old, token_indices, side='right')
    
    # Keep the tokens of valid samples
    valid_mask = np.isin(sample_ids, valid_indices)
    token_indices_valid = token_indices[valid_mask]
    feat_indices_valid = feat_indices[valid_mask]
    values_valid = sparse_acts_values[valid_mask].float().numpy()
    
    # Re-map the token indices (from 0)
    # For each valid sample, calculate the start position in the new sparse_acts
    sample_ids_valid = sample_ids[valid_mask]
    # Map the sample ID to the new sample ID (0 to len(valid_indices)-1)
    valid_indices_array = np.array(valid_indices)
    # Use searchsorted for vectorized mapping
    new_sample_ids = np.searchsorted(valid_indices_array, sample_ids_valid)
    
    # Calculate the position of each token in the new index system
    # Need to subtract the start position of the corresponding sample in the old system, then add the start position in the new system
    token_offsets_in_old = token_indices_valid - sample_starts_old[sample_ids_valid]
    token_indices_new = sample_starts_new[new_sample_ids] + token_offsets_in_old
    
    # Build the new sparse tensor
    new_num_tokens = seq_lens_new.sum()
    new_num_features = sparse_acts.shape[1]
    new_indices = np.stack([token_indices_new, feat_indices_valid], axis=0)
    
    sparse_data_filtered['sparse_acts'] = torch.sparse_coo_tensor(
        indices=torch.from_numpy(new_indices).long(),
        values=torch.from_numpy(values_valid),
        size=(new_num_tokens, new_num_features)
    ).coalesce()  # Merge the sparse tensor, ensure that indices() can be accessed normally
    
    if verbose:
        print(f"Updated sparse_data, sample number: {len(sparse_data_filtered['seq_lens'])}")
    
    return sparse_data_filtered, temp_data_list, valid_indices
