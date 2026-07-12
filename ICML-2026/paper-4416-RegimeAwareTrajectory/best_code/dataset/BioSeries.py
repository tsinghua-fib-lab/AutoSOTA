
from typing import List, Tuple

import torch

from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from bisect import bisect_right
import numpy as np

import sys
sys.path.extend(['./', '../'])

from dataset.BaseBioDataset import SingleBioSeriesDataset

from torch.utils.data._utils.collate import default_collate

def collate_x_and_optional_mask(batch):
    """
    batch: dict of (x, y, mark_x, mark_y, conditions, conditions_names) from Dataset.__getitem__
    mark_x, mark_y can be None.
    Returns: dict with keys 'x', 'y', 'x_mark', 'y_mark', 'conditions', 'conditions_names'
    """
    xs = [b['x'] for b in batch]
    ys = [b['y'] for b in batch]
    mark_xs = [b['x_mark'] for b in batch]
    mark_ys = [b['y_mark'] for b in batch]
    conditions = [b['conditions'] for b in batch]
    # conditions_names = [b['conditions_names'] for b in batch]

    # Collate x and y
    xs = default_collate(xs)
    ys = default_collate(ys)
    conditions = default_collate(conditions)
    
    # Handle masks - only collate if all are not None, otherwise return None
    if all(m is not None for m in mark_xs): mark_xs = default_collate(mark_xs)
    else: mark_xs = None
    if all(m is not None for m in mark_ys): mark_ys = default_collate(mark_ys)
    else: mark_ys = None

    return {
        'x': xs,
        'y': ys,
        'x_mark': mark_xs,
        'y_mark': mark_ys,
        'conditions': conditions,
        # 'conditions_names': conditions_names
    }

transform_type_list = ['log', 'asinh', 'minmax']
class MultiBioSeriesDataset(Dataset):
    """
    Overall dataset that wraps a list of BioSeries datasets (dataset base list).

    This class:
        - takes a list of data_dir paths (each to a BioSeries file),
        - internally creates one SingleBioSeriesDataset per path,
        - concatenates them logically into one global dataset.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        input_window: int,
        output_window: int,
        stride: int = 1,
        variable_independent: bool = True,
        dtype: torch.dtype = torch.float32,
        transform_type: str = 'log',
        transform_eps: float = 1e-9,
        condition_names: List[str] = [],
        **kwargs,
    ):
        """
        Args:
            dataset_dirs: list of data_dir paths (each path is one BioSeries file).
            input_window: length of input (history) window.
            output_window: length of output (future) window.
            stride: step size for sliding windows (applied to each base dataset).
            variable_independent:
                If True, each variable in each BioSeries is treated as an independent
                univariate time series (variable-independent mode).
                If False, reserved for future extension.
            dtype: torch dtype for returned tensors.
        """
        super().__init__()

        self.dataset_dirs = dataset_dirs
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.variable_independent = variable_independent
        self.dtype = dtype
        self.transform_type = transform_type
        self.transform_eps = transform_eps
        self.condition_names = condition_names

        if self.transform_type not in transform_type_list:
            raise ValueError(f"Unknown transform_type: {self.transform_type}. Supported types: {transform_type_list}")

        self.scale = True  # log transform is always used

        # Create one base dataset per data_dir
        self.base_datasets: List[SingleBioSeriesDataset] = []
        # Build base datasets with a progress bar
        for i, path in enumerate(tqdm(self.dataset_dirs, desc="Building base datasets")):
            base_ds = SingleBioSeriesDataset(
                data_path=path,
                input_window=self.input_window,
                output_window=self.output_window,
                stride=self.stride,
                variable_independent=self.variable_independent,
                dtype=self.dtype,
                transform_type=self.transform_type,
                transform_eps=self.transform_eps,
                condition_names=self.condition_names,
            )
            if len(base_ds) == 0: continue # filter out empty datasets
            self.base_datasets.append(base_ds)

        # Pre-compute cumulative sizes to map global index -> (dataset_idx, local_idx)
        self.cumulative_sizes: List[int] = []
        running_total = 0
        for ds in self.base_datasets:
            running_total += len(ds)
            self.cumulative_sizes.append(running_total)
        # print("Cumulative sizes:", self.cumulative_sizes)
        self.subset_indices = None

    def __len__(self) -> int:
        """
        Return total number of samples across all base datasets.
        """
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def _get_dataset_index(self, idx: int) -> Tuple[int, int]:
        """
        Map a global index to (base_dataset_index, local_index_inside_that_dataset).
        """
        if idx < 0:
            # Support negative indices
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for MultiBioSeriesDataset")

        ds_idx = bisect_right(self.cumulative_sizes, idx)
        if ds_idx == 0: local_idx = idx
        else: local_idx = idx - self.cumulative_sizes[ds_idx - 1]
        return ds_idx, local_idx

    def __getitem__(self, idx: int):
        """
        Return a sample from the corresponding base dataset.
        """
        ds_idx, local_idx = self._get_dataset_index(idx)
        base_ds = self.base_datasets[ds_idx]
        return self.log_transform(base_ds[local_idx])

    def _transform(self, x, _type: str, Min: float=None, Max: float=None):
        """Apply transformation to the data based on the specified type."""
        if _type == 'forward':
            # check the data whether less than 0
            if self.transform_type == 'log':
                if torch.any(x < 0):
                        raise ValueError("Data contains negative values, cannot apply log transform.")
                x = torch.log(x + self.transform_eps)
                return x
            elif self.transform_type == 'asinh':
                raise NotImplementedError("asinh transform is not implemented yet.")
            elif self.transform_type == 'minmax':
                if Min is None or Max is None: raise ValueError("Min and Max must be provided for minmax transform.")
                if torch.any(x < Min) or torch.any(x > Max): raise ValueError(f"Data contains values outside the range [{Min}, {Max}].")
                x = (x - Min) / (Max - Min)
                if torch.any(x < 0) or torch.any(x > 1): raise ValueError(f"Data after minmax transform contains values outside the range [0, 1].")
                return x
        elif _type == 'inverse':
            flag = False
            if type(x) is np.ndarray:
                flag = True
                x = torch.from_numpy(x)
            
            if self.transform_type == 'log':
                y = torch.exp(x) - self.transform_eps
            elif self.transform_type == 'asinh':
                raise NotImplementedError("inverse asinh transform is not implemented yet.")
            elif self.transform_type == 'minmax':
                if Min is None or Max is None: raise ValueError("Min and Max must be provided for minmax transform.")
                y = x * (Max - Min) + Min

            if flag: y = y.numpy()
            return y
        else:
            raise ValueError(f"Unknown transform type: {_type}")

    def log_transform(self, data):
        """Apply log transformation to the data."""
        min_condition_idx = self.condition_names.index('bound_min') if 'bound_min' in self.condition_names else -1
        max_condition_idx = self.condition_names.index('bound_max') if 'bound_max' in self.condition_names else -1
        Min = data['conditions'][..., min_condition_idx, :] if min_condition_idx != -1 else None
        Max = data['conditions'][..., max_condition_idx, :] if max_condition_idx != -1 else None
        time_Min, time_Max = data['conditions'][..., -2, :], data['conditions'][..., -1, :]
        data['x'] = self._transform(data['x'], 'forward', Min=Min, Max=Max)
        data['y'] = self._transform(data['y'], 'forward', Min=Min, Max=Max)
        data['x_mark'] = self._transform(data['x_mark'], 'forward', Min=time_Min, Max=time_Max)
        data['y_mark'] = self._transform(data['y_mark'], 'forward', Min=time_Min, Max=time_Max)
        return data
    def inverse_transform(self, x, Min: float=None, Max: float=None):
        """Apply inverse log transformation to the data."""
        return self._transform(x, 'inverse', Min=Min, Max=Max)
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size: size of each batch.
            shuffle: whether to shuffle the data at every epoch.
            num_workers: number of subprocesses to use for data loading.
            pin_memory: whether to pin memory for faster transfer to GPU.
        """
        if self.subset_indices is not None:
            # If subset_indices is set, create a Subset dataset
            return DataLoader(
                Subset(self, self.subset_indices),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_x_and_optional_mask,
            )
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_x_and_optional_mask,
        )

    def subset(self, indices: List[int]) -> 'MultiBioSeriesDataset':
        """
        Create a subset of this dataset with the specified indices.

        Args:
            indices: list of global indices to include in the subset.
        """
        self.subset_indices = indices
        return self
    

class MultiBioSeriesDataset_Subset(MultiBioSeriesDataset):
    """
    A subset of the MultiBioSeriesDataset at specified indices.
    """

    def __init__(self, dataset: MultiBioSeriesDataset, indices: List[int]):
        self.dataset = Subset(dataset, indices)
        self.scale = True  # log transform is always used
        self.transform_eps = dataset.transform_eps
        self.transform_type = dataset.transform_type
        self.condition_names = dataset.condition_names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Create a DataLoader for this subset dataset.

        Args:
            batch_size: size of each batch.
            shuffle: whether to shuffle the data at every epoch.
            num_workers: number of subprocesses to use for data loading.
            pin_memory: whether to pin memory for faster transfer to GPU.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_x_and_optional_mask,
        )


import pytorch_lightning as pl
from torch.utils.data import DataLoader
class BioSeriesDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        """
        Args:
            train_dataset (Dataset): The dataset for training.
            val_dataset (Dataset): The dataset for validation.
            test_dataset (Dataset): The dataset for testing.
            batch_size (int): The batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
