from typing import Dict, Tuple
import torch
import warnings
import torch.multiprocessing as mp

from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from config.train_config import TrainingConfig
from config.dataset_config import DatasetConfig
from data_processing.ellipsoids_utils import RotatedDataset


def get_train_set_vector_feat_norms_stats(
    dataset: Data,
    dataset_cfg: DatasetConfig,
    splits_dict: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """
    Compute mean and std of vector feature norms on the train split,
    if not already stored in the dataset config.
    """
    if (dataset_cfg.vector_norms_mean is not None) \
    and (dataset_cfg.vector_norms_std is not None):
        mean, std = dataset_cfg.vector_norms_mean, dataset_cfg.vector_norms_std
    else:
        train_idx = splits_dict.get('train', list(range(len(dataset))))
        if isinstance(train_idx, torch.Tensor):
            train_idx = train_idx.tolist()

        vector_feat_key = dataset_cfg.vector_feat_key
        # Each graph's vec feat is (num_nodes_in_graph, d_vector) -> get (num_nodes_in_graph,) norms
        # -> stack all nodes across all graphs to get global norm stats
        vec_feat_norms = torch.cat([
            torch.norm(dataset[i][vector_feat_key], dim=1) for i in train_idx
        ], dim=0) # (N,)
        mean, std = torch.mean(vec_feat_norms), torch.std(vec_feat_norms)
    return mean, std


def create_dataloaders(
    dataset: Data | Dict[str, list],
    splits_dict: Dict[str, torch.Tensor] | None,
    config: TrainingConfig | DatasetConfig,
) -> Tuple[Dict[str, PyGDataLoader | DataLoader], TrainingConfig]:
    """
    Create DataLoaders for each split of the dataset.
    
    Args:
        dataset: The dataset to create loaders for
        splits_dict: Dictionary mapping split names to indices
        config: TrainingConfig or DatasetConfig object containing dataloader parameters
        
    Returns:
        Tuple with (a) dictionary with DataLoaders keyed by split name ('train', 'valid', 
        'test') and (b) the updated config object.
    """
    # Shortcut: accept a prebuilt split-dict of PyG graphs (e.g., macaque path graphs)
    # In this mode, `dataset` is a dict mapping split names to lists[Data], and
    # `splits_dict` is ignored. We return DataLoaders directly over those lists.
    if isinstance(dataset, dict) \
    and all(isinstance(v, (list, tuple)) for v in dataset.values()):
        dataset_cfg = getattr(config, 'dataset_config', getattr(config, 'dataset_cfg', None))
        def _resolve_attr(attr_name, default=None):
            if hasattr(config, attr_name):
                return getattr(config, attr_name)
            if hasattr(dataset_cfg, attr_name):
                return getattr(dataset_cfg, attr_name)
            return default
        batch_size_cfg = _resolve_attr('batch_size', 128)
        valid_batch_size_cfg = _resolve_attr('valid_batch_size', batch_size_cfg)
        test_batch_size_cfg = _resolve_attr('test_batch_size', batch_size_cfg)
        drop_last_cfg = _resolve_attr('drop_last', False)
        num_workers_cfg = _resolve_attr('dataloader_num_workers', _resolve_attr('num_workers', 0))
        pin_memory_cfg = _resolve_attr('pin_memory', False)
        # print(f"pin_memory_cfg: {pin_memory_cfg}")

        # Disable pinning if any split contains sparse tensors
        if pin_memory_cfg:
            try:
                some = None
                for lst in dataset.values():
                    if len(lst) > 0:
                        some = lst[0]
                        break
                if some is not None:
                    for attr_value in vars(some).values():
                        if isinstance(attr_value, torch.Tensor) and attr_value.is_sparse:
                            pin_memory_cfg = False
                            break
            except Exception as err:
                print(f"Failed to probe sample tensors for sparsity: {err}")

        dataloader_kwargs = {
            'drop_last': drop_last_cfg,
            'num_workers': num_workers_cfg,
            'pin_memory': pin_memory_cfg,
            'collate_fn': Batch.from_data_list,
        }
        out: Dict[str, PyGDataLoader] = {}
        for split_name, graphs in dataset.items():
            lower = str(split_name).lower()
            if 'train' in lower:
                requested_bs = batch_size_cfg
            elif 'val' in lower:
                requested_bs = valid_batch_size_cfg
            elif 'test' in lower:
                requested_bs = test_batch_size_cfg
            else:
                requested_bs = batch_size_cfg
            bs = min(len(graphs), requested_bs)
            out[split_name] = PyGDataLoader(
                dataset=graphs,
                shuffle=('train' in lower),
                batch_size=bs,
                **dataloader_kwargs
            )
        return out, config

    # Resolve DataLoader parameters with TrainingConfig taking precedence over DatasetConfig
    def _resolve_attr(attr_name, default=None):
        """Return attr from TrainingConfig if available, else DatasetConfig attr, else default."""
        if hasattr(config, attr_name):
            return getattr(config, attr_name)
        # If `config` is a TrainingConfig, it contains `.dataset_config`
        if hasattr(config, 'dataset_config') and hasattr(config.dataset_config, attr_name):
            return getattr(config.dataset_config, attr_name)
        return default

    # Determine DataLoader parameters
    dataset_cfg = config.dataset_config
    batch_size_cfg = _resolve_attr('batch_size', 128)
    valid_batch_size_cfg = _resolve_attr('valid_batch_size', batch_size_cfg)
    test_batch_size_cfg = _resolve_attr('test_batch_size', batch_size_cfg)
    drop_last_cfg = _resolve_attr('drop_last', False)
    num_workers_cfg = _resolve_attr('dataloader_num_workers', _resolve_attr('num_workers', 0))
    pin_memory_cfg = _resolve_attr('pin_memory', False)

    # Build DataLoader kwargs
    dataloader_kwargs = {
        'drop_last': drop_last_cfg,
        'num_workers': num_workers_cfg,
        'pin_memory': pin_memory_cfg,
    }

    # ------------------------------------------------------------------
    # Safety: If CUDA is available and start method is 'fork' (default on Linux),
    # avoid initializing CUDA in worker subprocesses by forcing num_workers=0.
    # Users can opt-in to multiprocessing with CUDA by setting 'spawn' early
    # in their entrypoint (before any CUDA usage):
    #   import torch.multiprocessing as mp; mp.set_start_method('spawn', force=True)
    # ------------------------------------------------------------------
    try:
        start_method = mp.get_start_method(allow_none=True)
    except Exception:
        start_method = None
    if torch.cuda.is_available() and (start_method is None or start_method == 'fork'):
        if dataloader_kwargs['num_workers'] != 0:
            warnings.warn(
                "CUDA + forked DataLoader workers detected; setting num_workers=0 to avoid CUDA re-init in subprocesses. "
                "To use workers with CUDA, set spawn start method early in your entrypoint.")
            dataloader_kwargs['num_workers'] = 0

    # ------------------------------------------------------------------
    # Safety: Disable `pin_memory` when samples contain sparse tensors
    # ------------------------------------------------------------------
    # Torch (up to 2.4.1) does not implement `.pin_memory()` for sparse
    # tensors, which leads to a runtime NotImplementedError when the
    # DataLoader attempts to pin the entire Data object.  VDW datasets
    # attach sparse COO tensors `P` and `Q` to each sample, so we must
    # ensure pinning is disabled in that case.
    if dataloader_kwargs['pin_memory'] and len(dataset) > 0:
        try:
            _probe = dataset[0]
            if any(
                isinstance(getattr(_probe, _attr, None), torch.Tensor) \
                and getattr(_probe, _attr).is_sparse
                for _attr in ('P', 'Q')
            ):
                warnings.warn(
                    "pin_memory=True is incompatible with sparse tensors;"
                    " disabling pin_memory for DataLoader.")
                dataloader_kwargs['pin_memory'] = False
        except Exception:
            # Fail-safe: if probing fails, leave pin_memory as-is
            pass

    # --------------------------------------------------------------
    # Optional: rotate *only* the test split (e.g., for ellipsoid datasets)
    # --------------------------------------------------------------    
    do_rotate_test = False
    if 'ellipsoid' in getattr(dataset_cfg, 'dataset', '').lower():
        # Allow either explicit flag (rotate_test_set) or default behaviour of False
        do_rotate_test = getattr(dataset_cfg, 'rotate_test_set', True)

    # Helper dataset wrapper to lazily rotate coordinates
    if do_rotate_test:
        # Use refactored RotatedDataset from utils
        pass

    # ------------------------------------------------------------------
    # Target preprocessing
    # ------------------------------------------------------------------
    # (1) Subset targets on all data in all splits, if specified
    target_incl_idx = dataset_cfg.target_include_indices
    if target_incl_idx is not None:
        for data in dataset:
            # print(f"data.y shape:\n\tbefore: {list(data.y.shape)}")
            data.y = data.y.squeeze()[target_incl_idx]
            # print(f"\tafter: {list(data.y.shape)}")

    # (2) Apply normalization in-place to *train data only*, if specified
    # (we want valid and test metrics to be computed with de-normalized
    # output, versus raw targets, to get metrics in the original units) 
    if (dataset_cfg.target_preprocessing_type is not None):
        stats_dict = getattr(dataset_cfg, 'target_preproc_stats', None)
        if stats_dict is not None:
            center = stats_dict.get('center', None)
            scale = stats_dict.get('scale', None)
        else:
            center, scale = None, None

        # (a) Compute preprocessing stats if not already present in config
        if (stats_dict is None) \
        or ((stats_dict is not None) and (center is None or scale is None)):
            print(f"Need to compute target preprocessing stats (not found in dataset config):")
            # (i) MAD normalization
            if dataset_cfg.target_preprocessing_type == 'mad_norm':
                print(f"\tComputing MAD normalization stats from train set targets...")
                # NOTE: Target subsetting has already been applied above, so stats
                # will be computed on the correct subset of targets
                center, scale = get_train_set_targets_means_mads(
                    dataset, splits_dict
                ) # shapes (d_target,), (d_target,)
                dataset_cfg.target_preproc_stats = {
                    'center': center.tolist() \
                        if isinstance(center, torch.Tensor) else center,
                    'scale': scale.tolist() \
                        if isinstance(scale, torch.Tensor) else scale
                }
            # (ii) TODO: add other preprocessing methods
            else: 
                raise ValueError(
                    f"Target preprocessing not implemented for '{dataset_cfg.target_preprocessing_type}'"
                )
            print("\tDone.")

        # (b) Rescale targets in train split
        print(f"Rescaling targets in train split...")
        # Ensure stats are tensors for correct broadcasting with tensor targets
        center_t = center if isinstance(center, torch.Tensor) \
            else torch.as_tensor(center)
        scale_t = scale if isinstance(scale, torch.Tensor) \
            else torch.as_tensor(scale)
        for i in splits_dict['train']:
            # Note: dataset[i].y has already been subset to only 'target_incl_idx' in step (1) above
            dataset[i].y = (dataset[i].y - center_t) / scale_t  # (N,)
        print("\tDone.")
    else:
        print(f"No target preprocessing applied.")

    # --------------------------------------------------------------
    # DataLoader selection and collation
    # --------------------------------------------------------------
    if len(dataset) > 0 and _resolve_attr('using_pytorch_geo', True):
        dataloader_kwargs['collate_fn'] = Batch.from_data_list
        dataloader_class = PyGDataLoader
    else:
        dataloader_class = DataLoader

    # ------------------------------------------------------------------
    # Helper dataset wrapper to compute Euclidean edge distances once
    # ------------------------------------------------------------------
    compute_edge_distances = getattr(dataset_cfg, 'compute_edge_distances', False)
    if compute_edge_distances:
        class _WithEdgeDistances(torch.utils.data.Dataset):
            def __init__(self, base_ds: torch.utils.data.Dataset, vec_key: str = 'pos'):
                self.base_ds = base_ds
                self.vec_key = vec_key

            def __len__(self):
                return len(self.base_ds)

            def __getitem__(self, i: int):
                d = self.base_ds[i]
                try:
                    if (not hasattr(d, 'edge_weight') or getattr(d, 'edge_weight') is None) \
                    and hasattr(d, 'edge_index') and hasattr(d, self.vec_key):
                        pos = getattr(d, self.vec_key)
                        ei = d.edge_index
                        src, dst = ei[0], ei[1]
                        vec = pos[src] - pos[dst]
                        dist = torch.norm(vec, dim=-1)
                        d.edge_weight = dist
                except Exception:
                    # Best-effort; leave sample unchanged on failure
                    return d
                return d

    # Create dataloaders for each split using Subset,
    # and apply rotation to test split if requested
    dataloader_dict = {}
    for set_name, idx in splits_dict.items():
        split_ds = Subset(dataset, idx)
        # Optionally compute Euclidean edge distances once (stored in edge_weight)
        if compute_edge_distances:
            split_ds = _WithEdgeDistances(split_ds, vec_key=getattr(dataset_cfg, 'vector_feat_key', 'pos'))
        # Apply rotation lazily only to test split if enabled
        if do_rotate_test and ('test' in set_name.lower()):
            vector_attribs_to_rotate = getattr(dataset_cfg, 'vector_attribs_to_rotate')
            recompute_kwargs = {
                'vector_feat_key': getattr(dataset_cfg, 'vector_feat_key', 'pos'),
                'num_edge_features': getattr(config.model_config, 'num_edge_features', None),
                'hdf5_tensor_dtype': getattr(config.dataset_config, 'hdf5_tensor_dtype', 'float16'),
                'graph_construction': None,
                'use_mean_recentering': getattr(config.dataset_config, 'use_mean_recentering', False),
                'sing_vect_align_method': getattr(config.dataset_config, 'sing_vect_align_method', 'column_dot'),
                'local_pca_kernel_fn_kwargs': {
                    'kernel': getattr(config.dataset_config, 'local_pca_distance_kernel', 'gaussian'),
                    'gaussian_eps': getattr(config.dataset_config, 'local_pca_distance_kernel_scale', None),
                }
            }
            split_ds = RotatedDataset(
                split_ds,
                vector_attribs_to_rotate=vector_attribs_to_rotate,
                recompute_pq=True,
                recompute_kwargs=recompute_kwargs,
            )
        # Choose batch size: train uses training batch size; valid/test use evaluation batch size
        lower_name = set_name.lower()
        if 'train' in lower_name:
            requested_bs = batch_size_cfg
        elif ('val' in lower_name):
            requested_bs = valid_batch_size_cfg
        elif ('test' in lower_name):
            requested_bs = test_batch_size_cfg
        else:
            requested_bs = batch_size_cfg

        split_batch_size = min(len(split_ds), requested_bs)

        dataloader_dict[set_name] = dataloader_class(
            dataset=split_ds,
            shuffle=('train' in set_name),
            batch_size=split_batch_size,
            **dataloader_kwargs
        )
    
    return dataloader_dict, config