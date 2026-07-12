import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

import sys
sys.path.extend(['./', '../'])
from dataset.data_utils import _load_system_condition, filter_large_values, filtering_almost_constant_variables, filter_series

def _load_series(
    path: str,
    transform_type: str,
    min_relative_range: float = 1e-3,
    min_unique_ratio: float = 0.01,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Load one time series file and remove almost-constant variables.

    A variable is considered invalid if its relative range is too small
    (scale-invariant) and it only takes very few distinct values.

    Args:
        path: path to the time series file.
        min_relative_range: minimal (max - min) / max(|x|) to keep a variable.
        min_unique_ratio: minimal (#unique_values / T) to keep a variable.
        eps: small constant to avoid division by zero.

    Returns:
        data: np.ndarray of shape (T, num_valid_vars).

    Raises:
        ValueError: if no valid variables remain after filtering.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
        data = df.select_dtypes(include=[np.number]).to_numpy()
        name = df.select_dtypes(include=[np.number]).columns.to_numpy()
    else:
        raise ValueError(f"Unsupported file extension: {ext} for path: {path}")

    # Ensure shape is (T, num_vars)
    if data.ndim == 1:
        data = data[:, None]


    if transform_type == 'minmax':
        data, name = filter_series(data, name, path, mask_negative=False, 
                             min_relative_range=min_relative_range, min_unique_ratio=min_unique_ratio, eps=eps)
    else:  data, name = filter_series(data, name, path, min_relative_range=min_relative_range, min_unique_ratio=min_unique_ratio, eps=eps)
    if data is None: return None, None
    system_condition = _load_system_condition(data, name, path)
    return data, system_condition

class SingleBioSeriesDataset(Dataset):
    """
    Dataset base for a single BioSeries file (one data_dir).

    This class:
        - loads one time series file (one BioSeries),
        - builds sliding windows,
        - optionally treats each variable as an independent univariate series.

    File formats supported in this example:
        - .npy : numpy array of shape (T,) or (T, num_vars)
        - .csv : CSV file, time steps along rows; all numeric columns are used as variables
    """

    def __init__(
        self,
        data_path: str,
        input_window: int,
        output_window: int,
        stride: int = 1,
        variable_independent: bool = True,
        dtype: torch.dtype = torch.float32,
        transform_type: str = 'log',
        transform_eps: float = 1e-8,
        condition_names: List[str] = []
    ):
        """
        Args:
            data_path: path to a single BioSeries file.
            input_window: length of the input (history) window.
            output_window: length of the output (future) window.
            stride: step size for sliding windows over time.
            variable_independent:
                If True, each variable is treated as an independent univariate series
                (output shape: (window, 1)).
                If False, all variables are used together (multi-variate mode, reserved).
            dtype: torch dtype for returned tensors.
        """
        super().__init__()

        self.data_path = data_path
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.variable_independent = variable_independent
        self.dtype = dtype
        self.transform_type = transform_type
        self.transform_eps = transform_eps
        self.condition_names = condition_names

        self.scale = None  # Placeholder for potential scaling parameters

        # TODO: loading the system conditon from a metadata file can be added later
        # Load data for this BioSeries
        self.series, self.system_condition = _load_series(self.data_path, self.transform_type)  # shape: (T, num_vars + 1)
        if self.series is None: 
            self.index_mapping = [] # no valid series
            return 

        self.t = self.series[:, 0:1]  # time column
        self.series = self.series[:, 1:]  # data columns
        self.length, self.num_vars = self.series.shape

        # A list of index tuples describing each sample
        # For variable_independent=True: (var_index, start_index)
        # For variable_independent=False: (start_index,)  (reserved, not implemented)
        self.index_mapping: List[Tuple[int, int]] = []

        self._build_index_mapping()

        if len(self.index_mapping) == 0:
            raise ValueError(
                f"No valid windows for file {self.data_path}. "
                "Please check input_window, output_window, stride, and data length."
            )

    def _build_index_mapping(self) -> None:
        """
        Build index mapping for all available windows.

        In variable_independent mode:
            - For each variable and each valid time window, create (var_index, start_index).
        In non-variable_independent mode (reserved):
            - Only time dimension is used; all variables are kept together.
        """
        total_window = self.input_window + self.output_window
        max_start = self.length - total_window
        if max_start < 0:
            return

        if self.variable_independent:
            # Each variable is considered independent
            for var_idx in range(self.num_vars):
                for start in range(0, max_start + 1, self.stride):
                    self.index_mapping.append((var_idx, start))
        else:
            # Multi-variate mode: reserved for future implementation
            # Here we only prepare a simple time-based mapping.
            # You can extend this later if needed.
            for start in range(0, max_start + 1, self.stride):
                # Store var_idx as -1 to indicate "all variables"
                self.index_mapping.append((-1, start))

    def __len__(self) -> int:
        """
        Return number of samples (windows) in this BioSeries.
        """
        return len(self.index_mapping)

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        In variable_independent mode:
            x: shape (input_window, 1)
            y: shape (output_window, 1)

        In non-variable_independent mode (reserved):
            currently raises NotImplementedError.
        """
        var_idx, start = self.index_mapping[idx]
        total_input_end = start + self.input_window
        total_output_end = total_input_end + self.output_window

        if self.variable_independent:
            # Use one variable as a univariate time series
            x = self.series[start:total_input_end, var_idx : var_idx + 1] # [T, 1]
            y = self.series[total_input_end:total_output_end, var_idx : var_idx + 1] # [L, 1]
            x_mark = self.t[start:total_input_end] # [T, 1]
            y_mark = self.t[total_input_end:total_output_end] # [L, 1]
            conditions = [self.system_condition[name][var_idx + 1:var_idx + 2] for name in self.condition_names] # len(condition_names)
            conditions += [self.system_condition['bound_min'][0:1], self.system_condition['bound_max'][0:1]] # add time normalization conditions
            # x_mark = None
            # y_mark = None
        else:
            # Reserved for multi-variate mode (all variables together)
            # You can implement your logic here later.
            raise NotImplementedError(
                "Multi-variate mode (variable_independent=False) is reserved "
                "and not implemented yet."
            )

        # Normalization on the model side
        ###### Begin: Data normalization (standardization)
        # x_mean = np.mean(x)
        # x_std = np.std(x) + self.transform_eps
        # x = (x - x_mean) / x_std
        # y = (y - x_mean) / x_std
        ###### End: Data normalization (standardization)

        x_tensor = torch.as_tensor(x, dtype=self.dtype)
        y_tensor = torch.as_tensor(y, dtype=self.dtype)

        # dict data to return
        data = {
            'x': x_tensor,
            'y': y_tensor,
            'x_mark': torch.as_tensor(x_mark, dtype=self.dtype) if x_mark is not None else None,
            'y_mark': torch.as_tensor(y_mark, dtype=self.dtype) if y_mark is not None else None,
            'conditions': torch.as_tensor(conditions, dtype=self.dtype), # [len(condition_names) + 2, C]
            # 'conditions_names': self.condition_names,
        }
        # list data to return
        # data = [x_tensor, y_tensor,
        #         torch.as_tensor(x_mark, dtype=self.dtype) if x_mark is not None else None,
        #         torch.as_tensor(y_mark, dtype=self.dtype) if y_mark is not None else None,]
        return data
