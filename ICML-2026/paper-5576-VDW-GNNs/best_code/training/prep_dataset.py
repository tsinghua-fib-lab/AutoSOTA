import os
import warnings
from sympy import Q
import torch
import time
import multiprocessing as mp
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Callable

from torch.utils.data import Subset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
# from accelerate import Accelerator
import h5py

from models.class_maps import ATOM_WT_TO_IDX_MAP
from models.vdw_data_classes import VDWData, VDWDatasetHDF5
# from config.dataset_config import DatasetConfig
from config.train_config import TrainingConfig
from data_processing.data_utilities import (
    AddDiracFeaturesDataset,
)
from data_processing.process_pyg_data import (
    attach_coo_tensor_attr_to_data,
    # process_edge_features_to_node_features,
)
from data_processing.ellipsoid_data_classes import (
    EllipsoidDatasetLoader,
    # get_ellipsoid_dataset_info,
)
from data_processing.ellipsoids_utils import (
    load_ellipsoid_dataset, 
)
from models.infogain import (
    process_custom_wavelet_scales_type, 
    # get_wavelet_count_from_scales,
)


def load_dataset(
    config: TrainingConfig,
    subset_indices: Optional[List[int]] = None,
    model_key: Optional[str] = None,
    print_info: bool = False,
) -> Data:
    """
    Load the dataset (on the main process) using the specified dataset class and parameters.
    If config.h5_path is set, return a VDWDatasetHDF5 instance with operator keys from config.
    If config.h5_path and config.subsample_n are set, only use those indices present in the HDF5 file, 
    and subsample up to subsample_n.
    If config.data_subset_n is set, it overrides config.subsample_n and only uses that many samples.
    The subsample_seed is used for reproducibility of the subsampling.
    
    Note: In DDP mode, only the main process (rank 0) loads and prepares the dataset,
    but CPU workers are still used for parallel processing of operations like
    attaching diffusion operators.

    Args:
        config: DatasetConfig object containing dataset parameters
        subset_indices: Optional list of indices to subset the dataset to
        model_key: Optional model key (reserved for dataset-specific behavior)
        print_info: Whether to print dataset information

    Returns:
        Dataset object with subset indices applied
    """
    data_config = config.dataset_config
    # Helper function for main process printing
    def main_print(*args, timestamp=False, indent=0, **kwargs):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if timestamp:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}]", "  " * indent, *args, **kwargs)
            else:
                print("  " * indent, *args, **kwargs)

    # Start overall data preparation timing
    prep_start_time = time.time()
    main_print(f"\n{'=' * 80}")
    main_print(f"STARTING DATA PREPARATION", timestamp=True)

    dataset_kwargs = data_config.dataset_kwargs
    
    # Get number of CPU workers to use
    num_workers = data_config.num_workers if hasattr(data_config, 'num_workers') else 1
    main_print(f"Using {num_workers} CPU workers.", indent=1)

    # Handle dataset subsampling first
    if subset_indices is not None:
        # Explicit subset list takes precedence
        subset_size = None
    else:
        subset_size = data_config.subsample_n
    
        # Initialize dataset with specified parameters
        if data_config.dataset.lower() == 'ellipsoids':
            # Refactored: delegate to utility
            dataset = load_ellipsoid_dataset(
                data_config,
                print_info=print_info,
                main_print=lambda *args, **kwargs: main_print(*args, indent=2, **kwargs)
            )
            main_print(f"Loaded {len(dataset)} samples from ellipsoid dataset", indent=2)

        elif data_config.dataset.lower() == 'macaque_reaching':
            # Data loading is done in main_training.py
            pass

        else:
            raise ValueError(f"Dataset '{data_config.dataset}' not supported")
    
    # Ensure dataset_class is defined irrespective of subsampling branch
    if 'dataset_class' not in locals():
        if data_config.dataset.lower() in ('ellipsoids', 'macaque_reaching'):
            dataset_class = None
            dataset_kwargs['transform'] = None
        else:
            raise ValueError(f"Dataset '{data_config.dataset}' not supported")

    # Load dataset
    main_print(f"Loading dataset...", indent=1)
    load_start_time = time.time()
    
    if dataset_class is not None:
        # Standard dataset loading (PyG Dataset)
        dataset = dataset_class(**dataset_kwargs)
        full_dataset_size = len(dataset)
    else:
        if data_config.dataset.lower() == 'ellipsoids':
            # Get the data directory from config
            data_dir = getattr(data_config, 'data_dir', None)
            if data_dir is None:
                raise ValueError("data_dir must be specified in config for ellipsoid datasets")
            
            # THIS IS BROKEN: Print dataset information
            # if print_info:
            #     info = get_ellipsoid_dataset_info(
            #         data_dir, 
            #         data_config.dataset_filename,
            #         scalar_feat_key=data_config.scalar_feat_key,
            #         vector_feat_key=data_config.vector_feat_key,
            #         target_key=data_config.target_key,
            #     )
            #     main_print(f"Ellipsoid dataset info:", indent=2)
            #     main_print(f"  Dataset size: {info.get('dataset_size', 'N/A')}", indent=2)
            #     main_print(f"  Nodes per sample: {info.get('num_nodes_per_sample', 'N/A')}", indent=2)
            #     main_print(f"  Vector dimension: {info.get('vector_dim', 'N/A')}", indent=2)
            #     # The pickled dataset always includes graph-level 'y' (diameter).
            #     # Print it explicitly to avoid confusion with configured targets.
            #     if 'target_dim' in info:
            #         main_print(f"  Graph target dimension: {info['target_dim']}", indent=2)
            #     # Also report the configured target selection for clarity
            #     main_print(
            #         f"  Configured task/target: {data_config.task} / {data_config.target_key}"
            #         f" (dim={data_config.target_dim})",
            #         indent=2,
            #     )
            #     if 'target_mean' in info:
            #         mean_print = info['target_mean']
            #         if isinstance(mean_print, torch.Tensor):
            #             if mean_print.numel() == 1:
            #                 mean_print = mean_print.item()
            #             else:
            #                 mean_print = mean_print.squeeze().tolist()
            #                 mean_print = f"[{', '.join(f'{x.item():.3f}' for x in mean_print)}]"
            #         main_print(f"  Target mean (first 100 samples): {mean_print:.3f}", indent=2)
            #         std_print = info['target_std']
            #         if isinstance(std_print, torch.Tensor):
            #             if std_print.numel() == 1:
            #                 std_print = std_print.item()
            #             else:
            #                 std_print = std_print.squeeze().tolist()
            #                 std_print = f"[{', '.join(f'{x.item():.3f}' for x in std_print)}]"
            #         main_print(f"  Target std (first 100 samples): {std_print:.3f}", indent=2)
            
            # Load the single ellipsoid dataset
            dataset = EllipsoidDatasetLoader(
                data_dir=data_dir,
                dataset_filename=data_config.dataset_filename,
            )
            main_print(f"Loaded {len(dataset)} samples from ellipsoid dataset", indent=2)
        elif data_config.dataset.lower() == 'macaque_reaching':
            # Data loading is done in main_training.py
            pass
        else:
            raise ValueError(f"Unknown dataset type: {data_config.dataset}")
    
    load_elapsed = time.time() - load_start_time
    load_min, load_sec = int(load_elapsed // 60), load_elapsed % 60
    main_print(f"Dataset loaded in {load_min}m {load_sec:.2f}s", indent=2)
    
    # ---------------------------------------------------------
    # Apply subsetting
    # ---------------------------------------------------------
    # Priority: by explicit indices
    if subset_indices is not None:
        main_print(f"Subsetting dataset to provided indices (n={len(subset_indices)})...", indent=1)
        dataset = Subset(dataset, subset_indices)

    # Fallback: random subsetting by size
    elif subset_size is not None and not getattr(data_config, 'h5_path', None):
        # Only perform early random subsampling when NOT using HDF5, because
        # HDF5-based runs must sample *after* checking available indices.
        main_print(f"Subsetting dataset...", indent=1)
        subset_start_time = time.time()

        # Use subsample_seed if available, otherwise use split_seed
        seed = data_config.subsample_seed if data_config.subsample_seed is not None else data_config.split_seed
        rng = np.random.default_rng(seed)

        # Generate subset indices
        total_size = len(dataset)
        subset_indices = rng.choice(
            total_size, 
            size=subset_size, 
            replace=False
        )

        dataset = Subset(dataset, subset_indices)

        subset_elapsed = time.time() - subset_start_time
        subset_min, subset_sec = int(subset_elapsed // 60), subset_elapsed % 60
        main_print(f"Original dataset size: {total_size}", indent=2)
        main_print(f"Subset size: {len(dataset)}", indent=2)
        main_print(f"Subset created in {subset_min}m {subset_sec:.2f}s", indent=2)
    
    # ------------------------------------------------------------------
    # Ensure (subset) dataset indices match HDF5 availability
    # We assume if the user wants the full dataset, they have put all samples
    # in the HDF5 file.
    # ------------------------------------------------------------------
    if data_config.subsample_n is not None and getattr(data_config, 'h5_path', None):
        print(f"Checking HDF5 'original_idx' for available indices (subsample_n={data_config.subsample_n})...")

        # Use 'original_idx' as the authoritative source of available samples.
        with h5py.File(data_config.h5_path, 'r') as _h5f:
            if 'original_idx' not in _h5f:
                raise RuntimeError(
                    "HDF5 file must contain an 'original_idx' group; "
                    "legacy inference from operator groups is no longer supported."
                )
            available_indices = {int(k) for k in _h5f['original_idx'].keys()}

        if len(available_indices) == 0:
            raise RuntimeError("No indices found in 'original_idx' group of HDF5 file.")

        available_set = set(available_indices)

        # Current dataset may be a Subset (or not).  Recover *original* indices
        if isinstance(dataset, Subset):
            current_orig_indices = list(dataset.indices)
        else:
            current_orig_indices = list(range(len(dataset)))

        # Keep only positions whose original_idx is in the HDF5 file
        keep_positions = [pos for pos, orig in enumerate(current_orig_indices) if orig in available_set]

        # Optional further subsampling: honor subsample_n *after* filtering
        if (getattr(data_config, 'subsample_n', None) is not None) \
        and (len(keep_positions) > data_config.subsample_n):
            rng = np.random.default_rng(
                data_config.subsample_seed if data_config.subsample_seed is not None else data_config.split_seed
            )
            keep_positions = sorted(rng.choice(keep_positions, size=data_config.subsample_n, replace=False))

        if len(keep_positions) < len(current_orig_indices):
            main_print(
                f"Filtering dataset from {len(current_orig_indices)} to {len(keep_positions)} samples with HDF5 data.",
                indent=1,
            )
            dataset = Subset(dataset, keep_positions)

    # ------------------------------------------------------------------
    # Lazy wrapper to attach some attributes (e.g., original_idx) 
    # on first access
    # ------------------------------------------------------------------
    class _LazyAttributeLoadingDataset(torch.utils.data.Dataset):
        """Wrap a dataset (possibly nested Subsets) and guarantee `original_idx`."""

        def __init__(self, base_ds):
            self.base_ds = base_ds

        # --------------------------------------------------
        # Helper – resolve original index through nested Subsets
        # --------------------------------------------------
        @staticmethod
        def _resolve_orig(ds, idx):
            # Recursively resolve original index through nested Subsets
            while isinstance(ds, Subset):
                idx = ds.indices[idx]
                ds = ds.dataset
            return idx

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            data = self.base_ds[idx]
            if not hasattr(data, 'original_idx'):
                data.original_idx = self._resolve_orig(self.base_ds, idx)
            return data

    dataset = _LazyAttributeLoadingDataset(dataset)
    
    # DEPRECATED: Process edge features into node features if desired
    # if config.use_edge_as_node_features:
    #     main_print(f"Processing edge features...", indent=1)
    #     edge_start_time = time.time()
        
    #     # Use multiprocessing for edge feature processing
    #     from multiprocessing import Pool
    #     with Pool(num_workers) as pool:
    #         dataset = list(pool.map(process_edge_features_to_node_features, dataset))
        
    #     # If edge_feature_key is 'bond_node_features', concatenate to x
    #     if config.edge_to_node_feature_key == 'bond_node_features':
    #         for data in dataset:
    #             if hasattr(data, 'bond_node_features'):
    #                 data.x = torch.cat([data.x, data.bond_node_features], dim=1)
    #                 delattr(data, 'bond_node_features')
        
    #     edge_elapsed = time.time() - edge_start_time
    #     edge_min, edge_sec = int(edge_elapsed // 60), edge_elapsed % 60
    #     main_print(f"Edge features processed in {edge_min}m {edge_sec:.2f}s", indent=2)
    
    # If using HDF5 for operators, return VDWDatasetHDF5
    if hasattr(data_config, 'h5_path') and data_config.h5_path:
        main_print(f"Loading HDF5 data...", indent=1)
        h5_start_time = time.time()
        
        # Convert all to VDWData for correct collating/batching
        dataset = [
            VDWData(
                **dict(d),
                operator_keys=(
                    getattr(data_config, 'scalar_operator_key', 'P'), 
                    getattr(data_config, 'vector_operator_key', 'Q')
                )
            ) \
            if not isinstance(d, VDWData) \
            else d for d in dataset
        ]
        
        # Create index map for VDWDatasetHDF5
        # The index map should map from the new indices (0 to len(dataset)-1)
        # to the original indices in the HDF5 file
        index_map = {i: d.original_idx for i, d in enumerate(dataset)}
        
        h5_elapsed = time.time() - h5_start_time
        h5_min, h5_sec = int(h5_elapsed // 60), h5_elapsed % 60
        main_print(f"HDF5 data loaded in {h5_min}m {h5_sec:.2f}s", indent=2)
        
        # Determine which operators to include based on model requirements
        op_keys = ()
        # Query HDF5 to include only present operator groups
        with h5py.File(data_config.h5_path, 'r') as _h5f:
            # Always include vector operator if not ablating vector track and present
            if not getattr(data_config, 'ablate_vector_track', False):
                vkey = getattr(data_config, 'vector_operator_key', 'Q')
                if vkey in _h5f:
                    op_keys = op_keys + (vkey,)
            # Include scalar operator P for legacy models if present
            if not getattr(data_config, 'ablate_scalar_track', False):
                skey = getattr(data_config, 'scalar_operator_key', 'P')
                if skey in _h5f:
                    op_keys = op_keys + (skey,)

        if len(op_keys) > 0:
            dataset = VDWDatasetHDF5(
                model_key=config.model_config.model_key,
                data_list=dataset,
                h5_path=data_config.h5_path,
                sparse_operators_tup=op_keys,
                index_map=index_map,
                scalar_feat_key=data_config.scalar_feat_key,
                vector_feat_key=data_config.vector_feat_key,
                attach_on_access=True,
                attributes_to_drop=data_config.attributes_to_drop,
                num_edge_features=data_config.num_edge_features,
                num_bond_types=data_config.num_bond_types,
                # Scattering configuration
                line_operator_key=getattr(data_config, 'line_operator_key', 'P_line'),
                vector_operator_key=getattr(data_config, 'vector_operator_key', 'Q'),
                line_scatter_feature_key=getattr(data_config, 'line_scatter_feature_key', 'edge_scatter'),
                vector_scatter_feature_key=getattr(data_config, 'vector_scatter_feature_key', 'vector_scatter'),
                line_scatter_kwargs={
                    'num_rbf': config.model_config.num_rbf_scatter,
                    'rbf_cutoff': config.model_config.rbf_cutoff,
                    'wavelet_scales_type': config.model_config.wavelet_scales_type,
                    'J_line': getattr(config.model_config, 'J_scalar_line', 4),
                    'include_lowpass': config.model_config.include_lowpass_wavelet,
                    'custom_line_scales': config.model_config.scalar_diffusion_scales,
                    # Scattering-order and nonlinearity controls
                    'num_orders': int(getattr(config.model_config, 'num_scattering_layers_scalar', 2) or 2),
                    'apply_first_order_nonlin': getattr(config.model_config, 'scalar_scatter_first_order_nonlin', None),
                    'apply_output_nonlin': getattr(config.model_config, 'scalar_scatter_output_nonlin', None),
                    'dtype': torch.float32,
                },
                vector_scatter_kwargs={
                    'wavelet_scales_type': config.model_config.wavelet_scales_type,
                    'J_vector': config.model_config.J_vector,
                    'include_lowpass': config.model_config.include_lowpass_wavelet,
                    'custom_vector_scales': config.model_config.vector_diffusion_scales,
                    'num_orders': int(getattr(config.model_config, 'num_scattering_layers_vector', 2) or 2),
                    'apply_first_order_nonlin': getattr(config.model_config, 'vector_scatter_first_order_nonlin', 'softplus_gate') or 'softplus_gate',
                    'apply_output_nonlin': getattr(config.model_config, 'vector_scatter_output_nonlin', None),
                    'dtype': torch.float32,
                },
            )
        
        # Skip diffusion operator processing since we're using HDF5
        proc_dataset = dataset
    
    # ------------------------------------------------------------------
    # IF NOT USING HDF5: 
    # Attach diffusion operators (sparse COO tensors) if specified
    # This alternative is more memory and processing intensive up front
    # ------------------------------------------------------------------
    elif data_config.diffusion_tensor_data_dir:
        main_print(f"Loading diffusion operators...", indent=1)
        diff_start_time = time.time()
        
        dir_path = os.path.join(
            data_config.data_dir, 
            data_config.diffusion_tensor_data_dir
        )
        tensor_keys = ()
        if not getattr(data_config, 'ablate_scalar_track', False):
            tensor_keys = tensor_keys + (data_config.scalar_operator_key,)
        if not getattr(data_config, 'ablate_vector_track', False):
            tensor_keys = tensor_keys + (data_config.vector_operator_key,)
        if data_config['graph_construction'] == 'distance_cutoff':
            # Load graph structure tensors
            tensor_keys = tensor_keys + (
                'edge_index',
                'edge_weight',
            )
            # Optional edge features
            if getattr(data_config.model_config, 'num_edge_features', None):
                tensor_keys = tensor_keys + ('edge_features',)

        if len(tensor_keys) > 0:
            proc_dataset = attach_coo_tensor_attr_to_data(
                dataset=dataset,
                dir_path=dir_path,
                tensor_keys=tensor_keys,
                num_workers=num_workers,
            )
        else:
            proc_dataset = dataset
        
        diff_elapsed = time.time() - diff_start_time
        diff_min, diff_sec = int(diff_elapsed // 60), diff_elapsed % 60
        main_print(f"Diffusion operators loaded in {diff_min}m {diff_sec:.2f}s", indent=2)
        main_print(f"Loaded operators for {len(proc_dataset)} Data objects", indent=2)

    else:
        proc_dataset = dataset


    # ------------------------------------------------------------------
    # Optional: concatenate bond-type (edge_attr) features as node scalars
    # NOTE: during diffusion, target nodes can't distinguish between their
    # type of bond with a neighbor versus its other bonds, under this setup.
    # ------------------------------------------------------------------
    # def _append_bond_features_to_x(data: Data, scalar_key: str = 'x') -> Data:
    #     """
    #     Compute per-node bond-type proportions from one-hot `edge_attr` and
    #     concatenate them to scalar features `data[scalar_key]`.
    #     Done once per Data object; guarded by `_edge_as_node_appended` flag.
    #     """
    #     try:
    #         if hasattr(data, '_edge_as_node_appended') \
    #         and getattr(data, '_edge_as_node_appended'):
    #             return data
    #         if (not hasattr(data, 'edge_attr')) or (data.edge_attr is None):
    #             return data
    #         if (not hasattr(data, 'edge_index')) or (data.edge_index is None):
    #             return data

    #         edge_attr = data.edge_attr
    #         edge_index = data.edge_index
    #         # Validate shapes
    #         # if edge_attr.dim() != 2 or edge_index.dim() != 2 or edge_index.shape[0] != 2:
    #         #     return data

    #         device = edge_attr.device
    #         num_nodes = int(data.num_nodes) \
    #             if hasattr(data, 'num_nodes') else int(edge_index.max().item() + 1)
    #         num_bond_types = int(edge_attr.shape[1])

    #         bond_counts = torch.zeros(
    #             (num_nodes, num_bond_types), 
    #             dtype=edge_attr.dtype, 
    #             device=device,
    #         )
    #         src = edge_index[0].to(torch.long)
    #         dst = edge_index[1].to(torch.long)

    #         # Accumulate for both endpoints (treat edges as undirected for counts)
    #         bond_counts.index_add_(0, src, edge_attr)
    #         bond_counts.index_add_(0, dst, edge_attr)

    #         row_sums = bond_counts.sum(dim=1, keepdim=True)
    #         row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    #         bond_props = bond_counts / row_sums

    #         # Concatenate to scalar features
    #         if hasattr(data, scalar_key) and (getattr(data, scalar_key) is not None):
    #             data[scalar_key] = torch.cat([getattr(data, scalar_key), bond_props], dim=1)
    #         else:
    #             data[scalar_key] = bond_props

    #         setattr(data, '_edge_as_node_appended', True)
    #     except Exception:
    #         return data
    #     return data

    # if hasattr(data_config, 'use_edge_as_node_features') \
    # and data_config.use_edge_as_node_features:
    #     scalar_key = data_config.scalar_feat_key
    #     # Wrap lazily so concatenation happens once per sample on first access
    #     class _AddEdgeAsNodeFeaturesDataset(torch.utils.data.Dataset):
    #         def __init__(self, base_ds: torch.utils.data.Dataset, scalar_key: str = 'x'):
    #             self.base_ds = base_ds
    #             self.scalar_key = scalar_key

    #         def __len__(self):
    #             return len(self.base_ds)

    #         def __getitem__(self, idx):
    #             d = self.base_ds[idx]
    #             return _append_bond_features_to_x(d, scalar_key=self.scalar_key)

    #     proc_dataset = _AddEdgeAsNodeFeaturesDataset(proc_dataset, scalar_key=scalar_key)


    # --------------------------------------------------------------
    # Optional: Add/concat dense Dirac indicator channels to scalar 
    # features (when requested by model config).
    # --------------------------------------------------------------
    if (
        hasattr(config, 'model_config')
        and hasattr(config, 'dataset_config')
        and isinstance(config.dataset_config.dataset, str)
        # and config.dataset_config.dataset.lower() == 'ellipsoids'
    ):
        if config.model_config.use_dirac_nodes:
            dirac_types = (
                config.model_config.dirac_types
                if getattr(config.model_config, 'dirac_types') is not None
                else ['max', 'min']
            )
            proc_dataset = AddDiracFeaturesDataset(config, proc_dataset, dirac_types=dirac_types)
            print(f"Added Dirac nodes ({dirac_types}) as scalar features.")

    if len(dataset) > len(proc_dataset):
        warnings.warn(
            f"\nDiffusion operators were computed for only a subset of "
            f"the dataset; only this subset will be used!"
        )
    
    # ------------------------------------------------------------------
    # Ensure dataset_config.vector_feat_dim matches the actual data
    # NOTE:  will fail in models where the vector features are treated as scalar features
    # and other scalar features are present (such as Diracs)
    # ------------------------------------------------------------------
    # try:
    #     vec_key = getattr(data_config, 'vector_feat_key', 'pos')
    #     _probe = proc_dataset[0] if len(proc_dataset) > 0 else None
    #     if _probe is not None and hasattr(_probe, vec_key) and getattr(_probe, vec_key) is not None:
    #         vec = getattr(_probe, vec_key)
    #         if isinstance(vec, torch.Tensor) and vec.dim() == 2:
    #             data_config.vector_feat_dim = int(vec.shape[1])
    # except Exception:
    #     pass
    
    # Print total preparation time
    prep_elapsed = time.time() - prep_start_time
    prep_min, prep_sec = int(prep_elapsed // 60), prep_elapsed % 60
    main_print(f"Complete.")
    main_print(f"Total data preparation time: {prep_min}m {prep_sec:.2f}s")
    # main_print(f"{'=' * 80}")
    
    return proc_dataset




