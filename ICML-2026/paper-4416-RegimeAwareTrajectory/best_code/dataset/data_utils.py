import numpy as np

# 3. 
def filtering_almost_constant_variables(data: np.ndarray, var_names: np.ndarray,
                                        min_relative_range: float = 1e-3,
                                        min_unique_ratio: float = 0.01,
                                        eps: float = 1e-12) -> np.ndarray:
    T, num_vars = data.shape
    # 3: Remove almost-constant variables
    # Compute max, min, and max_abs for each variable
    col_max = data.max(axis=0)         # (num_vars,)
    col_min = data.min(axis=0)         # (num_vars,)
    col_max_abs = np.maximum(np.abs(data).max(axis=0), eps)  # avoid zero

    # Scale-invariant relative range
    rel_range = (col_max - col_min) / col_max_abs  # (num_vars,)

    # Optional: use number of unique values as an additional signal
    unique_counts = np.array(
        [np.unique(data[:, j]).size for j in range(data.shape[1])],
        dtype=np.float32,
    )
    unique_ratio = unique_counts / float(T)

    # A variable is kept if it has enough relative fluctuation
    # OR it takes enough distinct values over time.
    valid_mask = (rel_range >= min_relative_range) | (unique_ratio >= min_unique_ratio)

    filtered = data[:, valid_mask]
    var_names = var_names[valid_mask]
    return filtered, var_names

# 2. 
def filter_large_values(data: np.ndarray, var_names: np.ndarray, threshold: float = 1e9) -> np.ndarray:
    """
    Filter out variables with values larger than a threshold.
    
    Args:
        data: np.ndarray of shape (T, num_vars).
        threshold: float, the value above which variables are filtered out.
    
    Returns:
        filtered: np.ndarray of shape (T, num_valid_vars).
    """
    valid_mask_large = np.all(np.abs(data) <= threshold, axis=0)
    filtered = data[:, valid_mask_large]
    var_names = var_names[valid_mask_large]

    return filtered, var_names

# Filter Strategy:
# 1. Remove variables with negative values
# 2. Remove variables with NaN or infinite values (larger than 1e9 in magnitude)
# 3. Remove almost-constant variables
def filter_series(data: np.ndarray, var_names: np.ndarray,
                  path: str,
                  min_relative_range: float = 1e-3,
                  min_unique_ratio: float = 0.01,
                  eps: float = 1e-12, mask_negative: bool = True) -> np.ndarray:
    
    data = data.astype(np.float32)
    # 1: Remove variables with negative values
    if mask_negative:
        # If any value in a variable is negative, remove that variable
        mask = np.all(data >= 0, axis=0, keepdims=False)
        filtered = data[:, mask]
        var_names = var_names[mask]
    else:
        filtered = data
        var_names = var_names # 🔑 no change

    # 2: Remove variables with NaN or infinite values (larger than 1e9 in magnitude)
    filtered, var_names = filter_large_values(filtered, var_names, threshold=1e9)
    filtered = np.clip(filtered, a_min=0, a_max=1e9)  # Ensure no negative values remain

    # 3: Remove almost-constant variables
    filtered, var_names = filtering_almost_constant_variables(filtered, var_names, min_relative_range, min_unique_ratio, eps,)

    if filtered.shape[1] <= 1:
        print(
            f"Warning/Skipping: In file '{path}', "
            f"all variables are almost constant and were removed. "
            "Returning None."
        )
        return None, None
    return filtered, var_names

import json
def _load_json_file(condition_path):
    with open(condition_path, 'r') as f:
        conditions = json.load(f)
    return conditions
def _load_system_condition(data, var_names, path: str):
    """
    Load system condition from the data file.
    data: np.ndarray, the data loaded from the file. shape = (T, num_vars), 
        The First dimension is time, and the first variable dimension is time.
    """
    assert var_names[0] == 'time', "The first variable should be 'time' representing the time dimension."
    # 🔥: using the json condition from the path
    condition_path = path.replace('.npy', '_conditions.json').replace('.csv', '_conditions.json')
    conditions = _load_json_file(condition_path)
    traj_pattern_list = [0.0] + [conditions[var_name]['trajectory_type'] for var_name in var_names[1:]]  # Time + Variables
    period_list = [0.0] + [conditions[var_name]['period'] for var_name in var_names[1:]]  # Time + Variables

    # TODO: Implement the logic to extract system condition from the data
    system_condition = {
        # Categorical conditions about the biological system
        ## About the series
        'traj_pattern': np.array(traj_pattern_list),  # [V]
        ## About the organism?
        
        # Continuous conditions can be added later
        ## Full Trajectory Boundary
        'bound_min': np.min(data, axis=0), # [V]
        'bound_max': np.max(data, axis=0), # [V]
        ## Full Trajectory Statistics
        'period': np.array(period_list),  # [V]
    }
    # Compute the condition from this data

    return system_condition