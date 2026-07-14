#!/usr/bin/env python3
"""
Compute InfoGain wavelet scales for scalar or vector tracks of a dataset (e.g., QM9),
using precomputed diffusion operator tensors from an HDF5 file. Only the train split is used.
Optionally, compute scales using different divergence metrics (KL, L1, or optimal transport)
and/or a random n-sample subset of the train set.

Outputs:
- Prints progress and results, including dropped uninformative channels.
- Saves computed scales and divergence plots for the selected track in the output directory.
- Serializes the computed scales as .pt files (PyTorch tensors).

Usage examples:
python compute_infogain_scales.py \
  --config config/qm9.yaml \
  --h5_path pq_tensor_data/qm9_pq.h5 \
  --output_dir infogain_scales/ \
  --track scalar \
  --divergence kl \
  --device cuda \
  --n_samples 8192 \
  --batch_size 4096

  python3.11 /home/davejohnson/Research/vdw/code/scripts/python/compute_infogain_scales.py \
    --config /home/davejohnson/Research/vdw/code/config/yaml_files/ellipsoids_diameter/experiment.yaml \
    --h5_path /home/davejohnson/Research/vdw/data/ellipsoids/pq_tensor_data.h5 \
    --output_dir /home/davejohnson/Research/vdw/data/ellipsoids/infogain_scales/ \
    --track vector \
    --divergence l1 \
    --device cuda

python /path/to/codecode/scripts/python/compute_infogain_scales.py \
  --config /path/to/codecode/config/yaml_files/borah_qm9_vdw.yaml \
  --h5_path /path/to/codedata/QM9/pq_tensor_data_tikhonov.h5 \
  --output_dir /path/to/codedata/QM9/infogain_scales/ \
  --track vector \
  --divergence ot \
  --device cuda \
  --n_samples 8192 \
  --batch_size 4096
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
import yaml
from dataclasses import fields as dc_fields
import torch
import numpy as np
from typing import List, Optional
from data_processing.data_utilities import get_random_splits as _get_random_splits
from training.prep_dataset import load_dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset
import h5py

# from config.dataset_config import DatasetConfig
from config.train_config import TrainingConfig
from models.vdw_data_classes import VDWData, VDWDatasetHDF5
from models import infogain


# --- Helper: Load config ---
def load_config(yaml_path: str) -> TrainingConfig:
    """Load YAML into TrainingConfig and populate nested dataset_config.

    Supports experiment-style YAMLs containing 'training' and 'dataset' sections,
    as well as single-file configs that may only contain 'dataset'.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    training_cfg = TrainingConfig()

    if isinstance(cfg, dict):
        # Apply training section overrides if present
        training_section = cfg.get('training', {})
        if isinstance(training_section, dict) and training_section:
            valid_fields = {f.name for f in dc_fields(TrainingConfig)}
            filtered = {k: v for k, v in training_section.items() if k in valid_fields}
            if filtered:
                training_cfg = TrainingConfig(**filtered)

        # Apply dataset section into nested dataset_config
        dataset_section = cfg.get('dataset', {})
        if isinstance(dataset_section, dict):
            for k, v in dataset_section.items():
                if hasattr(training_cfg.dataset_config, k):
                    setattr(training_cfg.dataset_config, k, v)

    return training_cfg


# --- Helper: Load train split indices ---
def get_train_indices(
    dataset_len: int,
    train_prop: float,
    valid_prop: float,
    split_seed: int,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    """Return train-set indices consistent with data_utilities.get_random_splits.

    Args:
        dataset_len: Total number of samples.
        train_prop: Proportion for training set (0-1).
        valid_prop: Proportion for validation set (0-1).
        split_seed: Random seed shared with training pipeline.
        n_samples: Optional subsampling of the train set.

    Returns:
        NumPy array of train indices.
    """
    # Use the same helper that main_training.py relies on
    splits = _get_random_splits(
        n=dataset_len,
        seed=split_seed,
        train_prop=train_prop,
        valid_prop=valid_prop,
    )
    train_indices = np.array(splits['train'])

    # Optional subsampling for quicker InfoGain runs
    if (n_samples is not None) and (n_samples < len(train_indices)):
        rng = np.random.default_rng(split_seed)
        train_indices = rng.choice(train_indices, size=n_samples, replace=False)
    elif (n_samples is not None) and (n_samples > len(train_indices)):
        print(
            f"Requested n_samples ({n_samples}) > train set size ({len(train_indices)}); "
            f"using all {len(train_indices)} available samples."
        )

    return train_indices


# --- Helper: Filter indices based on presence in HDF5 ---
def filter_indices_in_h5(
    dataset,
    indices: np.ndarray,
    h5_path: str,
    scalar_key: str = 'P',
    vector_key: str = 'Q',
) -> np.ndarray:
    """Filter dataset indices to those whose *original* indices have P and Q in HDF5.

    Args:
        dataset: VDWDatasetHDF5 (or compatible) containing `data_list` with `original_idx`.
        indices: Array of dataset indices to validate.
        h5_path: Path to HDF5 tensor file.
    Returns:
        NumPy array containing the kept dataset indices.
    """
    # Gather available keys in the HDF5 file once.
    with h5py.File(h5_path, 'r') as h5f:
        scalar_keys = set(h5f[scalar_key].keys())
        vector_keys = set(h5f[vector_key].keys())

    kept = []
    missing = []
    # Access original_idx without triggering __getitem__ (avoid extra I/O)
    data_list = getattr(dataset, 'data_list', None)
    for idx in indices:
        try:
            orig_idx = (
                data_list[idx].original_idx
                if data_list is not None
                else dataset[idx].original_idx  # fallback (may open HDF5)
            )
        except Exception:
            orig_idx = idx  # fallback to same index

        key_str = str(orig_idx)
        if (key_str in scalar_keys) and (key_str in vector_keys):
            kept.append(idx)
        else:
            missing.append(idx)

    if missing:
        print(
            f"[InfoGain] Warning: {len(missing)} / {len(indices)} selected indices do not have both "
            f"'{scalar_key}' and '{vector_key}' operators in the HDF5 file and will be skipped."
        )

    return np.array(kept, dtype=indices.dtype)


# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description="Compute InfoGain wavelet scales for scalar and vector tracks.")

    parser.add_argument(
        '--track', type=str, default='scalar', required=True,
        choices=['scalar', 'vector'],
        help='Track to compute InfoGain scales for (scalar or vector)'
    )
    parser.add_argument(
        '--feat_key', type=str, default=None,
        help='Feature attribute key to diffuse (default: dataset scalar_feat_key for scalar track; vector_feat_key for vector track)'
    )
    parser.add_argument(
        '--divergence', type=str, default='kl',
        choices=['kl', 'l1', 'ot', 'wasserstein', 'emd'],
        help='Divergence metric to use for InfoGain (kl, l1, ot/wasserstein/emd)'
    )
    parser.add_argument(
        '--config', type=str, required=True, 
        help='Path to dataset YAML config file'
    )
    parser.add_argument(
        '--h5_path', type=str, required=True, 
        help='Path to HDF5 file with P (scalar diffusion operator) and Q (vector diffusion operator) tensor data'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True, 
        help='Directory to save output scales and plots'
    )
    parser.add_argument(
        '--n_samples', type=int, default=None, 
        help='Optional: subset number of train samples to use; if not provided, all train samples will be used'
    )
    parser.add_argument(
        '--T', type=int, default=16, 
        help='Number of diffusion steps (T)'
    )
    parser.add_argument(
        '--quantiles', type=float, nargs='+', default=(0.2, 0.4, 0.6, 0.8), 
        help='Quantiles for cumulative KL divergence thresholding'
    )
    parser.add_argument(
        '--above_zero_floor', type=float, default=None, 
        help='Optional value > 0 to replace zeros in distributions before divergence calculations'
    )
    parser.add_argument(
        '--device', type=str, default=None, 
        help='Device for computation (cpu or cuda)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, 
        help='Batch size for InfoGain calculation'
    )
    parser.add_argument(
        '--verbosity', type=int, default=0, 
        help='Verbosity level for InfoGain calculation'
    )

    args = parser.parse_args()

    # Device selection logic
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[InfoGain] No device specified. Using '{device}'.")
    elif args.device.lower() == 'cpu':
        device = 'cpu'
        print(f"[InfoGain] Device explicitly set to CPU. Using CPU.")
    elif args.device.lower().startswith('cuda'):
        if torch.cuda.is_available():
            device = args.device
            print(f"[InfoGain] Device explicitly set to CUDA. Using {device}.")
        else:
            device = 'cpu'
            print(f"[InfoGain] WARNING: CUDA requested but not available. Falling back to CPU.")
    else:
        device = 'cpu'
        print(f"[InfoGain] WARNING: Unknown device '{args.device}'. Using CPU.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Ensure the HDF5 path is stored in dataset config so load_dataset can attach P/Q
    mcfg = cfg.model_config
    dcfg = cfg.dataset_config
    if dcfg.h5_path is None:
        dcfg.h5_path = args.h5_path

    # Determine train split indices and filter by HDF5 availability
    from torch_geometric.datasets import QM9
    keys_sorted = None
    if dcfg.dataset.lower() == 'qm9':
        dataset_len = len(QM9(root=dcfg.data_dir))
        train_positions = get_train_indices(
            dataset_len=dataset_len,
            train_prop=dcfg.train_prop,
            valid_prop=dcfg.valid_prop,
            split_seed=dcfg.split_seed,
            n_samples=None,
        )
        with h5py.File(args.h5_path, 'r') as _h5:
            valid_keys = set(_h5[dcfg.scalar_operator_key].keys()) & set(_h5[dcfg.vector_operator_key].keys())
        filtered_indices = np.array([idx for idx in train_positions if str(idx) in valid_keys], dtype=int)
        removed_ct = len(train_positions) - len(filtered_indices)
        if removed_ct > 0:
            print(f"[InfoGain] Filter step removed {removed_ct} indices lacking P/Q; {len(filtered_indices)} remain.")
    else:
        with h5py.File(args.h5_path, 'r') as _h5:
            op_s, op_v = mcfg.scalar_operator_key, mcfg.vector_operator_key
            if op_s not in _h5 or op_v not in _h5:
                raise RuntimeError(f"HDF5 file missing required operator groups '{op_s}' and/or '{op_v}'.")
            valid_keys = set(_h5[op_s].keys()) & set(_h5[op_v].keys())
        if len(valid_keys) == 0:
            raise RuntimeError("No common indices found across required operator keys in HDF5 file.")
        keys_sorted = sorted(int(k) for k in valid_keys)
        dataset_len = len(keys_sorted)
        train_positions = get_train_indices(
            dataset_len=dataset_len,
            train_prop=dcfg.train_prop,
            valid_prop=dcfg.valid_prop,
            split_seed=dcfg.split_seed,
            n_samples=None,
        )
        filtered_indices = np.array([keys_sorted[i] for i in train_positions], dtype=int)

    # Optional subsampling AFTER filtering
    if args.n_samples is not None and len(filtered_indices) > args.n_samples:
        rng = np.random.default_rng(dcfg.split_seed)
        filtered_indices = rng.choice(filtered_indices, size=args.n_samples, replace=False)
        print(f"[InfoGain] Subsampled to {len(filtered_indices)} graphs for InfoGain calculation.")

    print(f"Using {len(filtered_indices)} samples from train split (seed={dcfg.split_seed})")

    # Load dataset only for the remaining indices
    cfg.dataset_config.subsample_n = None
    dataset = load_dataset(cfg, subset_indices=filtered_indices.tolist())

    # DataLoader for InfoGain (no shuffle, can use config batch_size)
    loader_batch_size = min(args.batch_size, len(dataset)) if len(dataset) > 0 else 1
    train_loader = PyGDataLoader(dataset, batch_size=loader_batch_size, shuffle=False)

    # --- Compute InfoGain scales for scalar/vector track ---
    print(f"\n[InfoGain] Computing wavelet scales for {args.track} track using '{args.divergence.upper()}' divergence...")
    # Determine feature key and vector dim
    if args.track == 'scalar':
        feat_key = args.feat_key if args.feat_key is not None else getattr(dcfg, 'scalar_feat_key', 'x')
        vec_dim = None
        diff_key = 'P'
    else:
        feat_key = args.feat_key if args.feat_key is not None else getattr(dcfg, 'vector_feat_key', 'pos')
        vec_dim = getattr(dcfg, 'vector_feat_dim', None)
        diff_key = 'Q'

    uninform_chan, scales = infogain.calc_infogain_wavelet_scales(
        pyg_graphs=train_loader,
        task=dcfg.task,
        device=device,
        divergence_metric=args.divergence,
        divergence_metric_kwargs={'above_zero_floor': args.above_zero_floor} if args.above_zero_floor is not None else {},
        diff_op_key=diff_key,
        feat_to_diffuse_key=feat_key,
        vector_feat_dim=vec_dim,
        T=args.T,
        cmltv_divergence_quantiles=args.quantiles,
        savepath_divergence_by_channel_plot=args.output_dir,
        divergence_by_channel_plot_name=f'{args.track}_{args.divergence}_curve',
        divergence_by_channel_plot_label=f'{args.track} Track ({args.divergence.upper()})',
        verbosity=args.verbosity
    )
    print(f"[InfoGain] {args.track} track scales computed:")
    print(scales)

    # Build full scales tensor with zero rows for uninformative channels
    if uninform_chan is not None and len(uninform_chan) > 0:
        print(
            f"[InfoGain] Dropped {len(uninform_chan)} uninformative {args.track} channels: {uninform_chan.tolist()}"
        )

        total_channels = scales.shape[0] + len(uninform_chan)
        full_scales = torch.zeros(
            total_channels, scales.shape[1], dtype=scales.dtype, device=scales.device
        )
        keep_mask = torch.ones(total_channels, dtype=torch.bool, device=scales.device)
        keep_mask[uninform_chan] = False
        full_scales[keep_mask] = scales
    else:
        full_scales = scales

    # Save full_scales tensor
    save_path = os.path.join(args.output_dir, f'{args.track}_{args.divergence}_infogain_scales.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(full_scales.cpu(), save_path)
    print(f"[InfoGain] Saved {args.track} scales tensor to {save_path}")

    print("\n[InfoGain] Divergence curve plots saved in:", args.output_dir)
    print("[InfoGain] Done.")

if __name__ == "__main__":
    main() 

