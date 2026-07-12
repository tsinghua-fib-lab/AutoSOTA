"""
BioFlowMatching Dataset for BioSeries data.

This module provides dataset classes that work with the BioFlowMatching model.

Training Dataset:
    - Returns consecutive pairs: (x_t, x_{t+1}) for velocity prediction
    - Supports conditions from biological system properties
    
Test Dataset:
    - Returns input sequence and all future targets
    - Used for auto-regressive evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import pytorch_lightning as pl


class BioFlowMatchingTrainDataset(Dataset):
    """
    Training dataset for BioFlowMatching.
    
    Each sample returns:
        - x: [L, D] input sequence (positions 0 to L-1)
        - x_target: [L, D] target sequence (positions 1 to L)
        - condition_dict: dictionary of biological conditions
        
    This allows the model to learn v = x_{t+1} - x_t for each position.
    """
    def __init__(
        self,
        sequences: Union[List[np.ndarray], np.ndarray],  # List of [T, D] arrays
        conditions: Optional[List[Dict]] = None,  # List of condition dicts
        seq_length: int = 64,  # Length of each training sample
        stride: int = 1,  # Stride for sliding window
        normalize: bool = True,
    ):
        """
        Args:
            sequences: List of trajectory arrays, each [T, D]
            conditions: List of condition dicts for each trajectory
            seq_length: Length of training sequences
            stride: Stride for sliding window sampling
            normalize: Whether to normalize sequences
        """
        self.seq_length = seq_length
        self.stride = stride
        self.normalize = normalize
        
        # Process sequences
        self.samples = []
        self.sample_conditions = []
        
        for i, seq in enumerate(sequences):
            if isinstance(seq, np.ndarray):
                seq = torch.from_numpy(seq).float()
            
            if seq.dim() == 1:
                seq = seq.unsqueeze(-1)  # [T] -> [T, 1]
            
            T, D = seq.shape
            
            # Normalize if needed
            if normalize:
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True) + 1e-8
                seq = (seq - mean) / std
            # Create sliding window samples
            # We need seq_length + 1 points to get seq_length pairs
            for start in range(0, T - seq_length, stride):
                # x: positions start to start+seq_length-1
                # x_target: positions start+1 to start+seq_length
                x = seq[start:start + seq_length]
                x_target = seq[start + 1:start + seq_length + 1]
                
                self.samples.append((x, x_target))
                
                if conditions is not None:
                    self.sample_conditions.append(conditions[i])
                else:
                    self.sample_conditions.append(None)
        
        print(f"Created {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, x_target = self.samples[idx]
        condition = self.sample_conditions[idx]
        
        if condition is not None:
            # Convert condition values to tensors
            condition_dict = {}
            for key, value in condition.items():
                # Skip raw_conditions (not suitable for batching)
                if key == 'raw_conditions':
                    continue
                    
                if isinstance(value, torch.Tensor):
                    # Already a tensor
                    condition_dict[key] = value
                elif isinstance(value, np.ndarray):
                    # Numpy array
                    condition_dict[key] = torch.from_numpy(value.copy()).float()
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Python or numpy scalar -> convert to 1D tensor
                    condition_dict[key] = torch.tensor([float(value)])
                elif isinstance(value, (list, tuple)):
                    # List/tuple -> convert to tensor
                    condition_dict[key] = torch.tensor(value).float()
                else:
                    # Skip unsupported types
                    continue
            return x, x_target, condition_dict
        else:
            return x, x_target


class BioFlowMatchingTestDataset(Dataset):
    """
    Test dataset for BioFlowMatching auto-regressive evaluation.
    
    Each sample returns:
        - x_input: [input_length, D] input sequence
        - x_target: [pred_length, D] target sequence to predict
        - condition_dict: biological conditions
    """
    def __init__(
        self,
        sequences: Union[List[np.ndarray], np.ndarray],
        conditions: Optional[List[Dict]] = None,
        input_length: int = 32,
        pred_length: int = 16,
        normalize: bool = True,
    ):
        """
        Args:
            sequences: List of trajectory arrays, each [T, D]
            conditions: List of condition dicts
            input_length: Length of input sequence
            pred_length: Length of prediction horizon
        """
        self.input_length = input_length
        self.pred_length = pred_length
        self.normalize = normalize
        
        self.samples = []
        self.sample_conditions = []
        self.norm_params = []  # Store for denormalization
        
        for i, seq in enumerate(sequences):
            if isinstance(seq, np.ndarray):
                seq = torch.from_numpy(seq).float()
            
            if seq.dim() == 1:
                seq = seq.unsqueeze(-1)
            
            T, D = seq.shape
            
            if T < input_length + pred_length:
                print(f"Warning: Sequence {i} too short ({T} < {input_length + pred_length}), skipping")
                continue
            
            # Normalize
            if normalize:
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True) + 1e-8
                seq = (seq - mean) / std
                self.norm_params.append((mean, std))
            else:
                self.norm_params.append((None, None))
            
            # Take input_length as input, next pred_length as target
            x_input = seq[:input_length]
            x_target = seq[input_length:input_length + pred_length]
            
            self.samples.append((x_input, x_target))
            
            if conditions is not None:
                self.sample_conditions.append(conditions[i])
            else:
                self.sample_conditions.append(None)
        
        print(f"Created {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x_input, x_target = self.samples[idx]
        condition = self.sample_conditions[idx]
        
        if condition is not None:
            condition_dict = {}
            for key, value in condition.items():
                # Skip raw_conditions (not suitable for batching)
                if key == 'raw_conditions':
                    continue
                    
                if isinstance(value, torch.Tensor):
                    # Already a tensor
                    condition_dict[key] = value
                elif isinstance(value, np.ndarray):
                    # Numpy array
                    condition_dict[key] = torch.from_numpy(value.copy()).float()
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Python or numpy scalar -> convert to 1D tensor
                    condition_dict[key] = torch.tensor([float(value)])
                elif isinstance(value, (list, tuple)):
                    # List/tuple -> convert to tensor
                    condition_dict[key] = torch.tensor(value).float()
                else:
                    # Skip unsupported types
                    continue
            return x_input, x_target, condition_dict
        else:
            return x_input, x_target


class BioFlowMatchingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for BioFlowMatching.
    """
    def __init__(
        self,
        train_sequences: List[np.ndarray],
        val_sequences: List[np.ndarray],
        test_sequences: Optional[List[np.ndarray]] = None,
        train_conditions: Optional[List[Dict]] = None,
        val_conditions: Optional[List[Dict]] = None,
        test_conditions: Optional[List[Dict]] = None,
        seq_length: int = 64,
        input_length: int = 32,
        pred_length: int = 16,
        stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        train_AR_dataset: bool = True,
        train_ARMD_dataset: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['train_sequences', 'val_sequences', 'test_sequences',
                                          'train_conditions', 'val_conditions', 'test_conditions'])
        self.train_AR_dataset = train_AR_dataset
        self.train_ARMD_dataset = train_ARMD_dataset
        
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences if test_sequences is not None else val_sequences
        
        self.train_conditions = train_conditions
        self.val_conditions = val_conditions
        self.test_conditions = test_conditions if test_conditions is not None else val_conditions
        
        self.seq_length = seq_length
        self.input_length = input_length
        self.pred_length = pred_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.train_AR_dataset:
                print('auto regressive training dataset enabled')
                self.train_dataset = BioFlowMatchingTrainDataset(
                    self.train_sequences,
                    self.train_conditions,
                    self.seq_length,
                    self.stride,
                    self.normalize,
                )
            else:
                print('non auto regressive training dataset enabled')
                self.train_dataset = BioFlowMatchingTestDataset(
                    self.train_sequences,
                    self.train_conditions,
                    self.input_length,
                    self.pred_length if not self.train_ARMD_dataset else self.input_length,
                    self.normalize,
                )
            
            self.val_dataset = BioFlowMatchingTestDataset(
                self.val_sequences,
                self.val_conditions,
                self.input_length,
                self.pred_length,
                self.normalize,
            )

            self.test_dataset = BioFlowMatchingTestDataset(
                self.test_sequences,
                self.test_conditions,
                self.input_length,
                self.pred_length,
                self.normalize,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = BioFlowMatchingTestDataset(
                self.test_sequences,
                self.test_conditions,
                self.input_length,
                self.pred_length,
                self.normalize,
            )
            self.val_dataset = BioFlowMatchingTestDataset(
                self.val_sequences,
                self.val_conditions,
                self.input_length,
                self.pred_length,
                self.normalize,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_with_conditions,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context='fork' if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_with_conditions,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_with_conditions,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def _collate_with_conditions(self, batch):
        """
        Custom collate function to handle condition dictionaries.
        """
        if len(batch[0]) == 2:
            # No conditions
            xs, x_targets = zip(*batch)
            return torch.stack(xs), torch.stack(x_targets)
        else:
            # With conditions
            xs, x_targets, conditions = zip(*batch)
            
            # Stack tensors
            xs = torch.stack(xs)
            x_targets = torch.stack(x_targets)
            
            # Merge condition dicts
            if conditions[0] is not None and len(conditions[0]) > 0:
                condition_dict = {}
                keys = list(conditions[0].keys())
                for key in keys:
                    values = []
                    for c in conditions:
                        v = c.get(key, None)
                        if v is None:
                            continue
                        # Ensure tensor
                        if not isinstance(v, torch.Tensor):
                            if isinstance(v, np.ndarray):
                                v = torch.from_numpy(v).float()
                            elif isinstance(v, (int, float, np.integer, np.floating)):
                                v = torch.tensor([float(v)])
                            else:
                                v = torch.tensor(v).float()
                        values.append(v)
                    
                    if len(values) > 0:
                        # Try to stack, handling different shapes
                        try:
                            # Ensure all have same shape
                            if values[0].dim() == 0:
                                # Scalar tensors - add dimension
                                values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
                            condition_dict[key] = torch.stack(values)
                        except RuntimeError:
                            # If shapes don't match, just keep the first one repeated
                            condition_dict[key] = values[0].unsqueeze(0).expand(len(conditions), -1)
                
                return xs, x_targets, condition_dict
            else:
                return xs, x_targets


# ===================== Adapter for existing BioSeries Dataset =====================

def extract_sequences_and_conditions(dataset, condition_names: List[str] = None):
    """
    Extract full sequences and conditions from MultiBioSeriesDataset or MultiBioSeriesDataset_Subset.
    
    MultiBioSeriesDataset returns dict with keys:
        - 'x': [input_window, 1] - input sequence (already log/minmax transformed)
        - 'y': [output_window, 1] - output/target sequence (already transformed)
        - 'x_mark': [input_window, 1] - input time marks (transformed)
        - 'y_mark': [output_window, 1] - output time marks (transformed)
        - 'conditions': [num_conditions + 2, C] - condition tensor
            * First `len(condition_names)` rows: named conditions (e.g., bound_min, bound_max for variable)
            * Last 2 rows: time normalization bounds (time_min, time_max)
    
    Args:
        dataset: MultiBioSeriesDataset, MultiBioSeriesDataset_Subset, or torch Subset
        condition_names: List of condition names used when creating the dataset
                        (e.g., ['bound_min', 'bound_max'])
        
    Returns:
        sequences: List of numpy arrays [input_window + output_window, D]
        conditions: List of condition dicts with keys:
            - 'raw_conditions': raw condition tensor
            - 'bounds': [min, max] array for BioFlowMatching
            - 'time_bounds': [time_min, time_max] array
            - Individual condition names if provided
    """
    sequences = []
    conditions = []
    
    # Handle condition_names from dataset if not provided
    if condition_names is None:
        if hasattr(dataset, 'condition_names'):
            condition_names = dataset.condition_names
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'condition_names'):
            # For Subset wrapping
            condition_names = dataset.dataset.condition_names
    
    for i in range(len(dataset)):
        item = dataset[i]
        
        # ========== Handle MultiBioSeriesDataset dict format ==========
        if isinstance(item, dict):
            # Extract x and y sequences
            x = item['x']  # [input_window, 1]
            y = item['y']  # [output_window, 1]
            x_mark = item['x_mark']  # [input_window, 1] time marks
            y_mark = item['y_mark']  # [output_window, 1] time marks
            
            # Convert to numpy if tensor
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
            if isinstance(x_mark, torch.Tensor): x_mark = x_mark.detach().cpu().numpy()
            if isinstance(y_mark, torch.Tensor): y_mark = y_mark.detach().cpu().numpy()
            
            # Ensure 2D shape [T, D]
            if x.ndim == 1: x = x.reshape(-1, 1)
            if y.ndim == 1: y = y.reshape(-1, 1)
            if x_mark.ndim == 1: x_mark = x_mark.reshape(-1, 1)
            if y_mark.ndim == 1: y_mark = y_mark.reshape(-1, 1)
            
            # Concatenate to form full sequence: [input_window + output_window, D]
            seq = np.concatenate([x, y], axis=0)
            # TODO: Calculate the specific sequence conditions
            ## 1. calculate the specific sequence boundaries
            
            # ========== Extract conditions ==========
            cond_tensor = item.get('conditions', None)
            cond = {}
            cond['seq_bounds'] = np.array([np.min(seq, axis=0), np.max(seq, axis=0)])  # [2, D]
            cond['past_mark'] = x_mark # [input_window, 1]
            cond['future_mark'] = y_mark # [output_window, 1]
            
            if cond_tensor is not None:
                if isinstance(cond_tensor, torch.Tensor):
                    cond_tensor = cond_tensor.detach().cpu().numpy()
                
                # Store raw conditions
                cond['raw_conditions'] = cond_tensor
                
                # conditions shape: [num_conditions + 2, C]
                # Structure from SingleBioSeriesDataset:
                #   - First len(condition_names) rows: named conditions for the variable
                #   - Last 2 rows: time_min, time_max (for time normalization)
                
                num_named = len(condition_names) if condition_names else 0
                
                # Parse named conditions
                if condition_names is not None:
                    for idx, name in enumerate(condition_names):
                        if idx < cond_tensor.shape[0]:
                            # Extract scalar value from [1] or [C] shape
                            val = cond_tensor[idx]
                            cond[name] = val[0] if val.ndim > 0 and len(val) > 0 else val
                
                # Extract bounds for BioFlowMatching (bound_min, bound_max)
                if condition_names and 'bound_min' in condition_names and 'bound_max' in condition_names:
                    min_idx = condition_names.index('bound_min')
                    max_idx = condition_names.index('bound_max')
                    bound_min = cond_tensor[min_idx, :] if cond_tensor.ndim > 1 else cond_tensor[min_idx]
                    bound_max = cond_tensor[max_idx, :] if cond_tensor.ndim > 1 else cond_tensor[max_idx]
                    cond['bounds'] = np.array([bound_min, bound_max])
                elif cond_tensor.shape[0] >= 2:
                    # Fallback: assume first two are bounds
                    bound_min = cond_tensor[0, :] if cond_tensor.ndim > 1 else cond_tensor[0]
                    bound_max = cond_tensor[1, :] if cond_tensor.ndim > 1 else cond_tensor[1]
                    cond['bounds'] = np.array([bound_min, bound_max])

                if condition_names and 'traj_pattern' in condition_names:
                    traj_idx = condition_names.index('traj_pattern')
                    traj_pattern = cond_tensor[traj_idx, :] if cond_tensor.ndim > 1 else cond_tensor[traj_idx]
                    cond['traj_pattern'] = traj_pattern
                
                if condition_names and 'period' in condition_names:
                    period_idx = condition_names.index('period')
                    period = cond_tensor[period_idx, :] if cond_tensor.ndim > 1 else cond_tensor[period_idx]
                    cond['period'] = period

                # Extract time bounds (last 2 rows)
                if cond_tensor.shape[0] >= num_named + 2:
                    time_min = cond_tensor[-2, :] if cond_tensor.ndim > 1 else cond_tensor[-2]
                    time_max = cond_tensor[-1, :] if cond_tensor.ndim > 1 else cond_tensor[-1]
                    cond['time_bounds'] = np.array([time_min, time_max])
            
            sequences.append(seq)
            conditions.append(cond if cond else None)
        
        # ========== Handle tuple/list format (TFM format) ==========
        elif isinstance(item, (list, tuple)):
            # TFM format: (x0_values, x0_classes, x1_values, times_x0, times_x1)
            if len(item) >= 3:
                x0 = item[0]  # Starting values [memory, 1] or [1, 1]
                x1 = item[2]  # Target values [1, pred_len, 1] or [pred_len, 1]
                
                if isinstance(x0, torch.Tensor):
                    x0 = x0.detach().cpu().numpy()
                if isinstance(x1, torch.Tensor):
                    x1 = x1.detach().cpu().numpy()
                
                # Handle x0 shape
                if x0.ndim == 1:
                    x0 = x0.reshape(-1, 1)
                
                # Handle x1 shape (may have batch dim)
                if x1.ndim == 3:
                    x1 = x1.squeeze(0)  # [1, L, D] -> [L, D]
                if x1.ndim == 1:
                    x1 = x1.reshape(-1, 1)
                
                seq = np.concatenate([x0, x1], axis=0)
                
                # Extract conditions from x0_classes
                cond = {}
                if len(item) > 1:
                    x0_classes = item[1]
                    if isinstance(x0_classes, torch.Tensor):
                        x0_classes = x0_classes.detach().cpu().numpy()
                    cond['raw_conditions'] = x0_classes.flatten()
                    
                    # Try to parse bounds from x0_classes if it has the right structure
                    flat = x0_classes.flatten()
                    if len(flat) >= 2:
                        cond['bounds'] = np.array([float(flat[0]), float(flat[1])])
                
                sequences.append(seq)
                conditions.append(cond if cond else None)
            else:
                seq = item[0]
                if isinstance(seq, torch.Tensor):
                    seq = seq.detach().cpu().numpy()
                if seq.ndim == 1:
                    seq = seq.reshape(-1, 1)
                sequences.append(seq)
                conditions.append(None)
        
        # ========== Handle raw tensor/array ==========
        else:
            seq = item
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().cpu().numpy()
            if seq.ndim == 1:
                seq = seq.reshape(-1, 1)
            sequences.append(seq)
            conditions.append(None)
    
    return sequences, conditions


def create_bioflowmatching_datamodule_from_bio(
    train_dataset,
    val_dataset,
    test_dataset,
    seq_length: int = 64,
    input_length: int = 32,
    pred_length: int = 16,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    condition_names: List[str] = None,
    normalize: bool = True,
    train_AR_dataset: bool = True,
    train_ARMD_dataset: bool = False,
    **kwargs,
):
    """
    Create BioFlowMatching DataModule from existing MultiBioSeriesDataset.
    
    This adapter extracts sequences from the existing dataset format
    and creates appropriate training/test datasets.
    
    Args:
        train_dataset: Training dataset (MultiBioSeriesDataset or Subset)
        val_dataset: Validation dataset
        test_dataset: Test dataset
        seq_length: Length for training samples
        input_length: Input length for test
        pred_length: Prediction length for test
        stride: Sliding window stride
        batch_size: Batch size
        condition_names: List of condition names (e.g., ['bound_min', 'bound_max'])
    
    Returns:
        BioFlowMatchingDataModule
    """
    # Extract data from datasets
    print("Extracting training sequences...")
    train_sequences, train_conditions = extract_sequences_and_conditions(
        train_dataset, condition_names
    )
    
    print("Extracting validation sequences...")
    val_sequences, val_conditions = extract_sequences_and_conditions(
        val_dataset, condition_names
    )
    
    print("Extracting test sequences...")
    test_sequences, test_conditions = extract_sequences_and_conditions(
        test_dataset, condition_names
    )
    
    # Create data module
    return BioFlowMatchingDataModule(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        train_conditions=train_conditions,
        val_conditions=val_conditions,
        test_conditions=test_conditions,
        seq_length=seq_length,
        input_length=input_length,
        pred_length=pred_length,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        train_AR_dataset=train_AR_dataset,
        train_ARMD_dataset=train_ARMD_dataset,
    )

# ===================== Example Usage =====================
if __name__ == "__main__":
    print("Testing BioFlowMatching Dataset...")
    
    # Create synthetic data mimicking MultiBioSeriesDataset format
    np.random.seed(42)
    
    class MockMultiBioSeriesDataset:
        """Mock dataset that mimics MultiBioSeriesDataset format."""
        def __init__(self, n_samples=10, input_window=32, output_window=128):
            self.n_samples = n_samples
            self.input_window = input_window
            self.output_window = output_window
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            # Generate oscillating data
            t = np.linspace(0, 4*np.pi, self.input_window + self.output_window)
            freq = 0.5 + np.random.rand() * 0.5
            amp = 0.5 + np.random.rand() * 0.5
            values = amp * np.sin(freq * t)
            
            x = torch.tensor(values[:self.input_window], dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(values[self.input_window:], dtype=torch.float32).unsqueeze(-1)
            
            # Conditions: [bound_min, bound_max, time_min, time_max]
            conditions = torch.tensor([
                [-amp],      # bound_min
                [amp],       # bound_max
                [0.0],       # traj_pattern
                [freq],      # period
                [0.0],       # time_min
                [1.0],       # time_max
            ], dtype=torch.float32)
            
            return {
                'x': x,
                'y': y,
                'x_mark': torch.linspace(0, 0.5, self.input_window).unsqueeze(-1),
                'y_mark': torch.linspace(0.5, 1.0, self.output_window).unsqueeze(-1),
                'conditions': conditions,
            }
    
    # Create mock datasets
    input_window = 32
    output_window = 128
    train_ds = MockMultiBioSeriesDataset(n_samples=100, input_window=input_window, output_window=output_window)
    val_ds = MockMultiBioSeriesDataset(n_samples=20, input_window=input_window, output_window=output_window)
    test_ds = MockMultiBioSeriesDataset(n_samples=20, input_window=input_window, output_window=output_window)
    
    # Test extraction
    print("\nTesting extract_sequences_and_conditions...")
    sequences, conditions = extract_sequences_and_conditions(
        train_ds, 
        condition_names=['bound_min', 'bound_max']
    )
    print(f"Extracted {len(sequences)} sequences")
    print(f"First sequence shape: {sequences[0].shape}")
    print(f"First condition keys: {conditions[0].keys() if conditions[0] else 'None'}")
    if conditions[0] and 'bounds' in conditions[0]:
        print(f"First condition bounds: {conditions[0]['bounds']}")
    
    # Test full DataModule creation
    print("\nTesting create_bioflowmatching_datamodule_from_bio...")
    dm = create_bioflowmatching_datamodule_from_bio(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        seq_length=input_window + output_window - 1,
        input_length=input_window,
        pred_length=output_window,
        stride=1,
        batch_size=256,
        condition_names=['bound_min', 'bound_max'],
    )
    dm.setup('fit')
    
    print(f"\nTraining samples: {len(dm.train_dataset)}")
    print(f"Validation samples: {len(dm.val_dataset)}")
    
    # Test batch loading
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        if len(batch) == 2:
            x, x_target = batch
            print(f"\nTraining batch shapes: x={x.shape}, x_target={x_target.shape}")
        else:
            x, x_target, cond = batch
            print(f"\nTraining batch shapes: x={x.shape}, x_target={x_target.shape}")
            print(f"Condition keys: {cond.keys()}")
        break
    
    val_loader = dm.val_dataloader()
    for batch in val_loader:
        if len(batch) == 2:
            x_input, x_target = batch
            print(f"\nValidation batch shapes: x_input={x_input.shape}, x_target={x_target.shape}")
        else:
            x_input, x_target, cond = batch
            print(f"\nValidation batch shapes: x_input={x_input.shape}, x_target={x_target.shape}")
        break
    
    print("\n✓ All tests passed!")