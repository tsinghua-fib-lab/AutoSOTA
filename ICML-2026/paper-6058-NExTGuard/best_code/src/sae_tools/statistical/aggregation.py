from typing import List, Optional, Dict, Any, Tuple, Callable
from numpy.typing import NDArray
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

# =======================================================
# activation value calculation tool functions
# =======================================================

def compute_max_activation(feat_acts: NDArray[Any]) -> Tuple[float, int]:
    """
    compute the maximum activation value and index of a single feature
    
    Args:
        feat_acts: feature activation value array [L]
        
    Returns:
        (max_act, max_token_idx): maximum activation value and corresponding token index
    """
    max_act = float(np.max(feat_acts))
    max_token_idx = int(np.argmax(feat_acts))
    return max_act, max_token_idx


def compute_first_activation(feat_acts: NDArray[Any]) -> Tuple[float, int]:
    """
    compute the first non-zero activation value and position of a single feature
    
    Args:
        feat_acts: feature activation value array [L]
        
    Returns:
        (first_act, first_token_idx): first non-zero activation value and corresponding token index
        if there is no non-zero value, return (0.0, -1)
    """
    non_zero_indices = np.nonzero(feat_acts)[0]
    if len(non_zero_indices) > 0:
        first_token_idx = int(non_zero_indices[0])
        first_act = float(feat_acts[first_token_idx])
        return first_act, first_token_idx
    else:
        return 0.0, -1


def compute_accumulated_activation(
    feat_acts: torch.Tensor,
    window: int = 10,
    decay: float = 0.5
) -> Tuple[float, int, torch.Tensor]:
    """
    compute the accumulated activation value (using convolution to implement sliding window)
    
    Args:
        feat_acts: feature activation value tensor [L]
        window: sliding window size
        decay: decay coefficient
        
    Returns:
        (max_act, max_token_idx, accumulated_acts): 
        maximum activation value, corresponding token index, accumulated activation value array
    """
    indices = torch.arange(window, device=feat_acts.device)
    kernel_weights = (decay ** indices)  # [1, d, d^2, ...]
    kernel_weights = torch.flip(kernel_weights, dims=[0])  # [ ..., d^2, d, 1]
    
    # conv1d kernel dim: [out_channels, in_channels, kernel_size] -> [1, 1, window]
    kernel = kernel_weights.view(1, 1, -1).to(dtype=feat_acts.dtype)
    
    # Conv1d input dim: [Batch, Channel, Length] -> [1, 1, seq_len]
    input_tensor = feat_acts.view(1, 1, -1)
    
    # fill (window-1) zero on the left of input
    accumulated_acts = F.conv1d(input_tensor, kernel, padding=window-1)
    accumulated_acts = accumulated_acts[0, 0, :feat_acts.shape[0]]
    
    max_act = float(accumulated_acts.max().item())
    max_token_idx = int(accumulated_acts.argmax().item())
    
    return max_act, max_token_idx, accumulated_acts

# =======================================================
# sparse data processing tool functions
# =======================================================

def extract_scores_from_sparse_data(
    sae_seq_indices: List[List[int]],
    sae_feat_indices: List[List[int]],
    sae_values: List[List[float]],
    feature_idx: int = 4744,
    score_type: str = "first"
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    extract the score of a specified feature from sparse SAE data (COO format).
    
    Args:
        sae_seq_indices: list of token position indices for each sample
        sae_feat_indices: list of corresponding feature IDs
        sae_values: list of corresponding activation values
        feature_idx: index of the feature to analyze
        score_type: score type, "first" or "max"
        
    Returns:
        (y_scores, first_idx_list): score array and first activation position array
    """
    y_scores = []
    first_idx_list = []
    
    for seq_idx, feat_idx, vals in zip(sae_seq_indices, sae_feat_indices, sae_values):
        # extract the activation values of the specified feature from sparse format
        feature_mask = np.array(feat_idx) == feature_idx
        if np.any(feature_mask):
            feature_seq_indices = np.array(seq_idx)[feature_mask]
            feature_values = np.array(vals)[feature_mask]
            
            # build the complete activation value sequence
            if len(feature_seq_indices) > 0:
                max_seq_len = int(np.max(feature_seq_indices)) + 1
                feat_acts_array = np.zeros(max_seq_len)
                feat_acts_array[feature_seq_indices] = feature_values
            else:
                feat_acts_array = np.array([])
        else:
            feat_acts_array = np.array([])
        
        # compute the score
        if len(feat_acts_array) > 0:
            if score_type == "first":
                score, first_token_idx = compute_first_activation(feat_acts_array)
            elif score_type == "max":
                score, first_token_idx = compute_max_activation(feat_acts_array)
            else:
                raise ValueError(f"Unsupported score_type: {score_type}")
        else:
            score = 0.0
            first_token_idx = -1
        
        y_scores.append(score)
        first_idx_list.append(first_token_idx)
    
    return np.array(y_scores), np.array(first_idx_list)

SparseReduceFn = Callable[[NDArray, NDArray, NDArray, NDArray], NDArray]

# before max pooling, transform the sorted values element-wise (length must remain unchanged).
# typical usage: decay, normalization, truncation, reweighting based on token distance.
SparseValueTransformFn = Callable[[NDArray, NDArray, NDArray, NDArray, NDArray], NDArray]


def _aggregate_sparse_coo_by_sample_feature(
    sample_ids: NDArray,
    feat_indices: NDArray,
    values: NDArray,
    token_indices: Optional[NDArray] = None,
    value_transform_fn: Optional[SparseValueTransformFn] = None,
    reduce_fn: Optional[SparseReduceFn] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    aggregate the COO list of (sample_id, feat_id, value) by (sample_id, feat_id).

    - default behavior: take max for each group (i.e. max pooling), equivalent to the old implementation of np.maximum.reduceat.
    - optional: first transform each element in the group with value_transform_fn (does not change the group, only changes the value sequence).
    - custom: pass in reduce_fn(values_sorted, reduce_indices, sample_ids_sorted, feat_indices_sorted)
      return the aggregated value array for each group (length must be equal to len(reduce_indices)).
    """
    if len(values) == 0:
        return (
            np.array([], dtype=sample_ids.dtype),
            np.array([], dtype=feat_indices.dtype),
            np.array([], dtype=values.dtype),
        )

    # 1) sort: to use reduceat, must sort by (Sample, Feature)
    # lexsort keys are in reverse order: first sort by feat, then sort by sample (main key)
    if token_indices is None:
        sort_order = np.lexsort((feat_indices, sample_ids))
    else:
        # if token-level transformation is needed, ensure the token order within the group is stable: sort by (sample, feature, token)
        sort_order = np.lexsort((token_indices, feat_indices, sample_ids))
    sample_ids = sample_ids[sort_order]
    feat_indices = feat_indices[sort_order]
    values = values[sort_order]
    if token_indices is not None:
        token_indices = token_indices[sort_order]

    # 2) find the boundary points: when sample or feature changes, it is a new group
    diff = (sample_ids[:-1] != sample_ids[1:]) | (feat_indices[:-1] != feat_indices[1:])
    reduce_indices = np.concatenate(([0], np.flatnonzero(diff) + 1))

    # 3) (optional) transform: first do custom logic, then do max pooling / custom reduce
    if value_transform_fn is not None:
        # if token_indices is not provided, give a placeholder array, for the transformation function to handle
        token_arr = token_indices if token_indices is not None else np.zeros_like(values)
        values = value_transform_fn(values, token_arr, reduce_indices, sample_ids, feat_indices)

    # 4) reduce: default max pooling; also inject custom logic
    if reduce_fn is None:
        final_values = np.maximum.reduceat(values, reduce_indices)
    else:
        final_values = reduce_fn(values, reduce_indices, sample_ids, feat_indices)

    final_rows = sample_ids[reduce_indices]
    final_cols = feat_indices[reduce_indices]
    return final_rows, final_cols, final_values



def build_sentence_feature_matrix_from_sparse(
    sparse_data: Dict[str, Any],
    value_transform_fn: Optional[Any] = None, # Updated type hints for context
    reduce_fn: Optional[Any] = None,
) -> sp.csr_matrix:
    """
    Vectorized version: Build sample feature matrix from sparse tensors.
    Avoids Python loops, usually 10-100x faster than iterative versions.
    """
    # ======================================
    # 1. Data Preparation (Zero-copy conversion)
    # ======================================
    sparse_acts = sparse_data["sparse_acts"]
    num_features = sparse_acts.shape[1]
    
    # Assume sparse_acts is on CPU; if not, .cpu() is needed beforehand
    indices = sparse_acts.indices().float().numpy() # Shape: [2, NNZ]
    values = sparse_acts.values().float().numpy()   # Shape: [NNZ]
    
    token_indices = indices[0] # Shape: [NNZ] Token index of non-zero activation (0 ~ Total_Tokens)
    feat_indices = indices[1]  # Shape: [NNZ] Feature index
    
    seq_lens = sparse_data["seq_lens"].numpy()
    num_samples = len(seq_lens)
    
    # ======================================
    # 2. Coordinate Mapping: Token Index -> Sample ID
    # ======================================
    # Calculate the end boundary index for each sample
    # E.g., seq_lens=[2, 3] -> boundaries=[2, 5]
    # tokens 0,1 fall into interval 0 (idx<2), tokens 2,3,4 fall into interval 1 (idx<5)
    sample_boundaries = np.cumsum(seq_lens)
    
    # label sample id for the token list
    sample_ids = np.searchsorted(sample_boundaries, token_indices, side='right')
    
    # ======================================
    # 3. Filtering (Interval based)
    # ======================================
    # Expected shape: [num_samples, 2], representing [start, end)
    valid_token_intervals = sparse_data["valid_token_idx"].numpy()
        
    # To calculate relative positions, we need the absolute start index of each sample
    # boundaries: [L1, L1+L2, ...] -> starts: [0, L1, L1+L2, ...]
    sample_starts = np.zeros_like(sample_boundaries)
    sample_starts[1:] = sample_boundaries[:-1]
        
    # Vectorized retrieval of sample start pos and valid intervals for each activation
    # fancy indexing: sample_ids length equals NNZ
    act_sample_starts = sample_starts[sample_ids] 
    
    # Retrieve the specific [start, end) interval for the sample this activation belongs to
    act_valid_starts = valid_token_intervals[sample_ids, 0]
    act_valid_ends = valid_token_intervals[sample_ids, 1]
    
    # Calculate relative position within the sentence
    relative_positions = token_indices - act_sample_starts
        
    # Generate mask: Keep activations where relative_pos is within [start, end)
    # Note: If start >= end (invalid interval), this condition naturally returns False
    valid_mask = (relative_positions >= act_valid_starts) & (relative_positions < act_valid_ends)
        
    # Apply mask to crop arrays
    sample_ids = sample_ids[valid_mask]
    feat_indices = feat_indices[valid_mask]
    token_indices = token_indices[valid_mask]
    values = values[valid_mask]
    
    # ======================================
    # 4. Sparse Aggregation: Default Max Pooling
    # ======================================
    # - value_transform_fn: Transforms token-level values before max pooling
    # - reduce_fn: Can replace the final aggregation if needed (not recommended; usually value_transform_fn is sufficient)
    # We now have a list of (Sample, Feature, Value), effectively a COO format.
    # However, the same (Sample, Feature) tuple may appear multiple times (multiple tokens in a sentence activating the same feature).
    # We need to take the Max of duplicate (Sample, Feature) groups.
    
    if len(values) == 0:
        return sp.csr_matrix((num_samples, num_features))

    final_rows, final_cols, final_values = _aggregate_sparse_coo_by_sample_feature(
        sample_ids=sample_ids,
        feat_indices=feat_indices,
        values=values,
        token_indices=token_indices,
        value_transform_fn=value_transform_fn,
        reduce_fn=reduce_fn,
    )
    
    # 5. Build Final CSR Matrix
    return sp.csr_matrix(
        (final_values, (final_rows, final_cols)), 
        shape=(num_samples, num_features)
    )