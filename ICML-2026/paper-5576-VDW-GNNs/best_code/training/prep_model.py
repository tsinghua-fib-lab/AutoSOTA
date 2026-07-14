import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader as PyGDataLoader
from typing import Dict, Any, Optional, Tuple, List
from config.model_config import ModelConfig
from config.dataset_config import DatasetConfig
from config.train_config import TrainingConfig
from models.vdw import VDW
from models.vdw_layer import VDWLayer
from models.vdw_modular import VDWModular
from models.vdw_macaque import VDW_macaque
from models.vdw_macaque_supcon_2 import VDW_macaque_supcon_2
from models.vanilla_nn import VanillaNN
from models.comparisons.comparison_module import ComparisonModel
from models.comparisons.egnn import EGNNModel
from models.comparisons.tfn import TFNModel
from models.comparisons.legs import LEGSModel
from models.comparisons.tnn import TNNComparisonModel
from models.comparisons.pyg_models import PyGModel
import time
from accelerate import Accelerator
from models.infogain import (
    get_wavelet_count_from_scales,
    process_custom_wavelet_scales_type,
)

# Ensure project root on path for local imports
import sys
from pathlib import Path
_TRAINING_DIR = Path(__file__).resolve().parent
_CODE_DIR = _TRAINING_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))


def process_wavelet_scales_type(scales):
    """
    Backwards-compatible helper that returns either the provided
    string identifier (e.g., 'dyadic') or a processed tensor of scales.
    """
    if scales is None:
        return None
    if isinstance(scales, str):
        return scales
    return process_custom_wavelet_scales_type(scales)
from models.class_maps import (
    SCALAR_NONLIN_FN_MAP,
    MLP_NONLIN_MODULE_MAP,
    LOSS_FN_MAP,
)


# ============================================================
#               SHARED HELPER FUNCTIONS             
# ============================================================

def _setup_base_module_kwargs(
    device: torch.device,
    config: TrainingConfig, 
    dataset_config: DatasetConfig
) -> dict:
    """
    Set up base module keyword arguments for VDW models.
    
    Args:
        config: TrainingConfig object
        dataset_config: DatasetConfig object
    Returns:
        Dictionary of base module keyword arguments
    """
    base_module_kwargs = {
        'device': device,
        'task': dataset_config.task,
        'metrics_kwargs': {
            'num_outputs': dataset_config.target_dim
        },
        'verbosity': config.verbosity
    }

    # For clustering tasks, propagate the choice of clustering metric from the
    # training config so BaseModule can select between Dunn Index, Silhouette
    # score, or a logistic linear classifier accuracy metric. When
    # config.main_metric is one of these metrics (or its validation variant)
    # and 'cluster' is in the task string, configure BaseModule accordingly.
    task_val = getattr(dataset_config, 'task', None)
    main_metric_val = getattr(config, 'main_metric', None)
    if isinstance(task_val, str) and isinstance(main_metric_val, str):
        task_lower = task_val.lower()
        main_metric_str = main_metric_val
        if main_metric_str.endswith('_valid'):
            main_metric_base = main_metric_str[:-6]
        else:
            main_metric_base = main_metric_str

        if 'cluster' in task_lower:
            metrics_kwargs = base_module_kwargs['metrics_kwargs']
            if main_metric_base == 'silhouette_score':
                metrics_kwargs['cluster_metric'] = 'silhouette_score'
            elif main_metric_base == 'logistic_linear_accuracy':
                metrics_kwargs['cluster_metric'] = 'logistic_linear_accuracy'
    
    # Map loss function key to actual PyTorch loss function
    if hasattr(config, 'loss_fn') and config.loss_fn in LOSS_FN_MAP:
        base_module_kwargs['loss_fn'] = LOSS_FN_MAP[config.loss_fn]
        # Set loss function kwargs with mean reduction for all loss functions
        if config.loss_fn == 'huber':
            base_module_kwargs['loss_fn_kwargs'] = {
                'beta': getattr(config, 'huber_delta'),
                'reduction': 'mean',
            }
        else:
            # For MSE and L1 loss, ensure mean reduction
            base_module_kwargs['loss_fn_kwargs'] = {
                'reduction': 'mean',
            }

    # If specified, pass attributes to include with predictions to base module
    if hasattr(dataset_config, 'attributes_to_include_with_preds') \
    and dataset_config.attributes_to_include_with_preds is not None:
        base_module_kwargs['attributes_to_include_with_preds'] = dataset_config.attributes_to_include_with_preds
    
    # If specified, pass target normalization statistics to base module
    # (for single vs. multi-target data, the dim of these params is
    # processed upstream, in prep_dataset.py)
    if hasattr(dataset_config, 'target_preprocessing_type') \
    and dataset_config.target_preprocessing_type is not None:
        if hasattr(config.dataset_config, 'target_preproc_stats') \
        and config.dataset_config.target_preproc_stats is not None:
            target_preproc_stats = config.dataset_config.target_preproc_stats
            stats = {
                'center': torch.tensor(target_preproc_stats['center']), 
                'scale': torch.tensor(target_preproc_stats['scale'])
            }
            base_module_kwargs['has_normalized_train_targets'] = True
            base_module_kwargs['target_preproc_stats'] = stats
        else:
            raise ValueError(
                f"Target preprocessing type specified as '{dataset_config.target_preprocessing_type}', "
                f"but no target preprocessing stats found in config!"
            )
    else:
        base_module_kwargs['has_normalized_train_targets'] = False
        base_module_kwargs['target_preproc_stats'] = None
        
    return base_module_kwargs


def _setup_mlp_kwargs(
    model_config: ModelConfig, 
    dataset_config: DatasetConfig
) -> dict:
    """
    Set up MLP keyword arguments for VDW models.
    
    Args:
        model_config: ModelConfig object
        dataset_config: DatasetConfig object
        
    Returns:
        Dictionary of MLP keyword arguments
    """
    mlp_kwargs = {
        'hidden_dims_list': model_config.mlp_hidden_dim,
        'output_dim': dataset_config.target_dim,
        'use_batch_normalization': model_config.mlp_use_batch_normalization,
    }
    
    # Propagate dropout/batch norm parameters (for VanillaNN)
    if model_config.mlp_dropout_p is not None:
        mlp_kwargs['use_dropout'] = True
        mlp_kwargs['dropout_p'] = model_config.mlp_dropout_p
        mlp_kwargs['use_batch_normalization'] = model_config.mlp_use_batch_normalization
    else:
        mlp_kwargs['use_dropout'] = False
    
    return mlp_kwargs


def _setup_ablation_flags(
    config: TrainingConfig, 
    dataset_config: DatasetConfig
) -> tuple:
    """
    Set up ablation flags and propagate them to dataset config.
    
    Args:
        config: TrainingConfig object
        dataset_config: DatasetConfig object
    Returns:
        Tuple of (ablate_vector_track, ablate_scalar_track)
    """
    ablate_vector_track = getattr(config, 'ablate_vector_track')
    ablate_scalar_track = getattr(config, 'ablate_scalar_track')
    # Propagate dynamically so data prepping sees it (even though not a dataclass field)
    setattr(dataset_config, 'ablate_vector_track', ablate_vector_track)
    setattr(dataset_config, 'ablate_scalar_track', ablate_scalar_track)
    
    return ablate_vector_track, ablate_scalar_track


def _print_model_settings(
    config: TrainingConfig, 
    scalar_custom: bool, 
    vector_custom: bool, 
    scalar_scales: Optional[torch.Tensor] = None, 
    vector_scales: Optional[torch.Tensor] = None, 
    model_name: str = "VDW", 
    acc: Optional[Accelerator] = None
) -> None:
    """
    Print model settings confirmations (only once on main process).
    
    Args:
        config: TrainingConfig object
        scalar_custom: Whether scalar scales are custom
        vector_custom: Whether vector scales are custom
        scalar_scales: Scalar scales tensor/list
        vector_scales: Vector scales tensor/list
        model_name: Name of the model for logging
        acc: Optional Accelerator object
    """
    if (acc is None) or (acc.is_main_process):
        # Print confirmation of custom diffusion scales (only once on main process)
        if scalar_custom or vector_custom:
            msg_parts = []
            if scalar_custom:
                _scales = scalar_scales
                num_scales = _scales.numel() if isinstance(_scales, torch.Tensor) else len(_scales)
                msg_parts.append(f"scalar (n_scales={num_scales})")
            if vector_custom:
                _scales = vector_scales
                num_scales = _scales.numel() if isinstance(_scales, torch.Tensor) else len(_scales)
                msg_parts.append(f"vector (n_scales={num_scales})")
            print(f"[{model_name}] Using custom diffusion scales instead of dyadic for " + ", ".join(msg_parts) + ".")

        # Print confirmation of vector track ablation
        if getattr(config, 'ablate_vector_track', False):
            print(f"[{model_name}] Ablating vector track.")
        if getattr(config, 'ablate_scalar_track', False):
            print(f"[{model_name}] Ablating scalar track.")


def _load_pretrained_weights(
    model: nn.Module,
    pretrained_dir: str,
    *,
    verbosity: int = 0,
) -> None:
    """
    Load pretrained parameters into *model* from *pretrained_dir*.

    The directory is expected to contain a weight file - first match in
    this priority order: ``model.safetensors``, ``pytorch_model.bin``,
    ``model.bin``, ``model.pt``, ``pytorch_model.pt``.  If none of those are found, the first file with extension ``.safetensors``, ``.bin`` or
    ``.pt`` is used.

    Missing or unexpected keys are ignored (``strict=False``) so transfer
    learning across tasks with different output layers works out of the box.
    """
    from pathlib import Path

    weights_dir = Path(pretrained_dir).expanduser()
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Pretrained weights directory '{weights_dir}' does not exist."
        )

    # Preferred filenames (ordered)
    candidate_files = [
        'model.safetensors',
        'pytorch_model.bin',
        'model.bin',
        'model.pt',
        'pytorch_model.pt',
    ]

    weight_file = None
    for fname in candidate_files:
        fp = weights_dir / fname
        if fp.is_file():
            weight_file = fp
            break

    # Fallback: first recognised extension
    if weight_file is None:
        for fp in weights_dir.iterdir():
            if fp.suffix in {'.safetensors', '.bin', '.pt'} and fp.is_file():
                weight_file = fp
                break

    if weight_file is None:
        raise FileNotFoundError(
            f"No weight file (.safetensors | .bin | .pt) found in '{weights_dir}'."
        )

    if verbosity > 0:
        print(f"[prep_model] Loading pretrained weights from {weight_file}")

    # Load state dict depending on file type
    if weight_file.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(str(weight_file))
        except ImportError as e:
            raise ImportError(
                "safetensors package is required to load .safetensors files. "
                "Install via 'pip install safetensors'."
            ) from e
    else:
        state_dict = torch.load(str(weight_file), map_location='cpu')

    # Strip 'module.' prefixes if model was saved under DDP
    cleaned_state = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(
        cleaned_state, 
        strict=False
    )

    if verbosity > 0:
        print(
            f"[prep_model] Pretrained weights loaded. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )


def load_pretrained_weights_if_specified(
    model: nn.Module, 
    config: TrainingConfig
) -> None:
    """Load pretrained weights if specified in config.
    
    Args:
        model: The model to load weights into
        config: TrainingConfig object
    """
    if getattr(config, 'pretrained_weights_dir', None):
        _load_pretrained_weights(
            model,
            config.pretrained_weights_dir,
            verbosity=config.verbosity,
        )


# ============================================================
#               VDW MODEL PREPARATION FUNCTIONS             
# ============================================================
def _prepare_vdw_standard_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig, 
    nn.Module, 
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a standard VDW model and potentially modify dataloaders based on model mode.

    If model_mode == 'handcrafted_scattering', the dataloaders are modified to
    contain tensors of scattering features and targets, rather than PyTorch Geometric
    Data objects. The config is modified to reflect the change in dataloader_dict.

    Args:
        config: TrainingConfig object containing model and training parameters
        dataloader_dict: Dictionary of DataLoaders for each dataset split
        acc: Optional Accelerator object for distributed training
        
    Returns:
        Tuple of (config, model, _possibly_modified_dataloader_dict)
    """
    # Get model configuration
    model_config = config.model_config
    dataset_config = config.dataset_config
    
    # Set up shared parameters using helper functions
    base_module_kwargs = _setup_base_module_kwargs(
        acc.device, 
        config, 
        dataset_config
    )
    mlp_kwargs = _setup_mlp_kwargs(model_config, dataset_config)
    ablate_vector_track, ablate_scalar_track = _setup_ablation_flags(
        config, 
        dataset_config
    )

    # -------------------------------------------------------------
    # Set per-track parameter dicts (needed before applying nonlin selections)
    # -------------------------------------------------------------
    scalar_track_kwargs = {
        'feature_key': dataset_config.scalar_feat_key,
        'diffusion_op_key': dataset_config.scalar_operator_key,
        'num_layers': model_config.num_scattering_layers_scalar,
        'filter_combos_out': model_config.filter_combos_out,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_scalar,
            'include_lowpass': True,
        },
        'J_prime': model_config.J_prime_scalar,
        'scattering_pooling_kwargs': {
            'pooling_type': model_config.pooling_type,
            'moments': model_config.moments,
            'nan_replace_value': model_config.nan_replace_value,
        },
        'diffusion_scales': model_config.infogain_scales_scalar
    }
    
    vector_track_kwargs = {
        'original_feature_key': dataset_config.vector_feat_key,
        'feature_key': dataset_config.vector_feat_key,
        'diffusion_op_key': model_config.vector_operator_key,
        'vector_dim': dataset_config.vector_feat_dim,
        'num_layers': 0 if ablate_vector_track else model_config.num_scattering_layers_vector,
        'filter_combos_out': model_config.filter_combos_out,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_vector,
            'include_lowpass': True,
        },
        'J_prime': model_config.J_prime_vector,
        'scattering_pooling_kwargs': {
            'pooling_type': model_config.pooling_type,
            'moments': model_config.moments,
            'nan_replace_value': model_config.nan_replace_value,
            'norm_p': model_config.vector_norm_p,
        },
        'diffusion_scales': model_config.infogain_scales_vector
    }

    # --------------------------------------------------
    # Apply non-linearity selections *after* kwargs exist
    # --------------------------------------------------
    chosen_scalar = getattr(model_config, 'scalar_nonlin', None)
    chosen_mlp = getattr(model_config, 'mlp_nonlin', None)
    chosen_vector_nonlin = getattr(model_config, 'vector_nonlin', None)

    if chosen_mlp in MLP_NONLIN_MODULE_MAP:
        mlp_kwargs['nonlin_fn'] = MLP_NONLIN_MODULE_MAP[chosen_mlp]
        mlp_kwargs['nonlin_fn_kwargs'] = {}

    if chosen_scalar in SCALAR_NONLIN_FN_MAP:
        scalar_track_kwargs['nonlin_fn'] = SCALAR_NONLIN_FN_MAP[chosen_scalar]
        scalar_track_kwargs['nonlin_fn_kwargs'] = {}

    if chosen_vector_nonlin:
        vector_track_kwargs['vector_nonlin_type'] = chosen_vector_nonlin

    # Only used in cross-track mode
    cross_track_kwargs = {
        'n_cross_track_combos': getattr(model_config, 'n_cross_track_combos', 16),
        'n_cross_filter_combos': getattr(model_config, 'n_cross_filter_combos', 8),
        'within_track_combine': getattr(model_config, 'within_track_combine', False),
        'cross_track_mlp_hidden_dim': getattr(model_config, 'cross_track_mlp_hidden_dim', None),
        # Wavelet recombination parameters
        'use_wavelet_recombination': getattr(model_config, 'use_wavelet_recombination', True),
        'scalar_recombination_channels': getattr(model_config, 'scalar_recombination_channels', 16),
        'vector_recombination_channels': getattr(model_config, 'vector_recombination_channels', 16),
        'recombination_hidden_dim': getattr(model_config, 'recombination_hidden_dim', 64),
        'vector_gate_hidden_dim': getattr(model_config, 'vector_gate_hidden_dim', None),
        # Wjxs batch normalization parameter - default to True
        'use_wjxs_batch_norm': True,
    }
    
    # -------------------------------------------------------------
    # Handle custom diffusion scales specified in the dataset config
    # -------------------------------------------------------------
    scalar_track_kwargs['diffusion_scales'] = process_custom_wavelet_scales_type(
        dataset_config.scalar_diffusion_scales
    )
    if not ablate_vector_track:
        vector_track_kwargs['diffusion_scales'] = process_custom_wavelet_scales_type(
            dataset_config.vector_diffusion_scales
        )

    # Print model settings confirmations (only once on main process)
    scalar_custom = scalar_track_kwargs['diffusion_scales'] is not None
    vector_custom = vector_track_kwargs['diffusion_scales'] is not None
    _print_model_settings(
        config, scalar_custom, vector_custom,
        scalar_track_kwargs['diffusion_scales'], vector_track_kwargs['diffusion_scales'],
        "VDW", acc
    )

    model = VDW(
        mode=model_config.model_mode,
        ablate_vector_track=ablate_vector_track,
        ablate_scalar_track=ablate_scalar_track,
        stream_parallelize_tracks=config.use_cuda_streams \
            and torch.cuda.is_available(),
        base_module_kwargs=base_module_kwargs,
        mlp_kwargs=mlp_kwargs,
        scalar_track_kwargs=scalar_track_kwargs,
        vector_track_kwargs=vector_track_kwargs,
        cross_track_kwargs=cross_track_kwargs,
        verbosity=config.verbosity
    )

    # Optional: load pretrained weights (transfer learning)
    # _load_pretrained_weights_if_specified(model, config)
 
    # If in handcrafted_scattering mode, compute scattering features and create new dataloaders
    if model_config.model_mode == 'handcrafted_scattering':

        if acc.is_main_process:
            print(f"\nComputing scattering features...")

        # Compute scattering features and contain in new dataloaders dict
        scattering_features = {set_name: [] for set_name in dataloader_dict.keys()}
        targets = {set_name: [] for set_name in dataloader_dict.keys()}
        
        # Process each dataloader's data into tensors of scattering features and targets
        for set_name, dataloader in dataloader_dict.items():
            if acc.is_main_process:
                print(f"   Processing {set_name} set...")
            set_start_time = time.time()
            batch_times = []
            
            # Ensure model is in eval mode for feature computation
            model.eval()
            
            with torch.no_grad():
                for batch_i, batch in enumerate(dataloader):
                    batch_start_time = time.time()
                    
                    # Move batch to correct device
                    if acc is not None:
                        batch = batch.to(acc.device)
                    else:
                        batch = batch.to(config.device)
                    
                    # In handcrafted_scattering mode, model returns tensor of scattering features
                    features = model(batch)
                    target = batch.y
                    
                    # Move tensors to CPU for storage
                    features = features.cpu()
                    target = target.cpu()
                    
                    scattering_features[set_name].append(features)
                    targets[set_name].append(target)
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
            
            # Synchronize processes before concatenating tensors
            if acc is not None:
                acc.wait_for_everyone()
            
            # Concatenate tensors
            scattering_features[set_name] = torch.cat(scattering_features[set_name], dim=0)
            targets[set_name] = torch.cat(targets[set_name], dim=0)
            
            # Synchronize processes after concatenation
            if acc is not None:
                acc.wait_for_everyone()
            
            if acc.is_main_process:
                set_time = time.time() - set_start_time
                print(f"      Complete.")
                print(f"      Total time for {set_name} set: {set_time:.2f}s")
                print(f"      Average batch time: {sum(batch_times) / len(batch_times):.2f}s")
        
        # Create new datasets and dataloaders
        scat_feats_dataset_dict = {
            set_name: TensorDataset(scattering_features[set_name], targets[set_name])
            for set_name in dataloader_dict.keys()
        }
        
        # Create new dataloaders with the same batch size and other parameters
        dataloader_dict = {
            set_name: DataLoader(
                dataset=dataset,
                batch_size=dataloader_dict[set_name].batch_size,
                shuffle=('train' in set_name),
                num_workers=dataloader_dict[set_name].num_workers,
                pin_memory=dataloader_dict[set_name].pin_memory,
                drop_last=dataloader_dict[set_name].drop_last
            ) for set_name, dataset in scat_feats_dataset_dict.items()
        }
        
        # Update config to use non-PyTorch Geometric data
        config.using_pytorch_geo = False
        config.target_key = 1  # Targets are second item in TensorDataset tuples
        
        # Create new MLP model for classification/regression
        model = VanillaNN(
            input_dim=scat_feats_dataset_dict['train'][0][0].shape[0],
            **mlp_kwargs,
            base_module_kwargs=base_module_kwargs
        )
        
        # Move model to correct device
        if acc is not None:
            model = model.to(acc.device)
        else:
            model = model.to(config.device)
    
    return config, model, dataloader_dict


def prepare_vdw_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig, 
    nn.Module, 
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a VDW-family model (standard, radial, or modular) based on
    ``config.model_config.model_key``.

    This wrapper only handles VDW variants. Comparison models are prepared by
    ``prepare_comparison_model``.

    Args:
        config: TrainingConfig object containing model and training parameters
        dataloader_dict: Dictionary of DataLoaders for each dataset split
        acc: Optional Accelerator object for distributed training
        
    Returns:
        Tuple of (config, model, _possibly_modified_dataloader_dict)
    """
    model_key = config.model_config.model_key
    
    if model_key == 'vdw':
        return _prepare_vdw_standard_model(config, dataloader_dict, acc)
    elif model_key == 'vdw_modular':
        return _prepare_vdw_modular_model(config, dataloader_dict, acc)
    elif model_key == 'vdw_layer':
        return _prepare_vdw_layer_model(config, dataloader_dict, acc)
    elif model_key == 'vdw_macaque':
        return _prepare_vdw_macaque_model(config, dataloader_dict, acc)
    elif model_key == 'vdw_supcon_2':
        return _prepare_vdw_macaque_supcon_2_model(config, dataloader_dict, acc)
    else:
        raise ValueError(
            f"Unsupported model_key for prepare_vdw_model: {model_key}. "
            "Supported values are 'vdw', 'vdw_modular', "
            "'vdw_layer'."
        )


def _prepare_vdw_layer_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None,
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader],
]:
    """
    Prepare the lightweight vector-only VDW layer stack (ScatterMLP).
    """
    model_config = config.model_config
    dataset_config = config.dataset_config

    base_module_kwargs = _setup_base_module_kwargs(acc.device, config, dataset_config)

    # Resolve diffusion scales for vector track
    custom_vector_scales = process_wavelet_scales_type(model_config.vector_diffusion_scales)

    mlp_nonlin_fn = MLP_NONLIN_MODULE_MAP.get(
        model_config.scatter_mlp_activation,
        nn.SiLU,
    )
    if not isinstance(mlp_nonlin_fn, type):
        mlp_nonlin_fn = nn.SiLU

    neighbor_k = model_config.scatter_neighbor_k
    if neighbor_k is None:
        neighbor_k = model_config.k_neighbors

    model = VDWLayer(
        base_module_kwargs=base_module_kwargs,
        vector_feat_key=dataset_config.vector_feat_key,
        vector_operator_key=model_config.vector_operator_key,
        vector_dim=dataset_config.vector_feat_dim,
        diffusion_scales=custom_vector_scales,
        scales_type=model_config.wavelet_scales_type,
        J=model_config.J_vector,
        include_lowpass=model_config.include_lowpass_wavelet,
        scatter_mlp_layers=model_config.scatter_mlp_layers,
        scatter_mlp_hidden=model_config.scatter_mlp_hidden,
        scatter_mlp_dropout=model_config.scatter_mlp_dropout,
        scatter_mlp_use_batch_norm=model_config.scatter_mlp_use_batch_norm,
        scatter_mlp_activation=mlp_nonlin_fn,
        scatter_include_zero_order=model_config.scatter_include_zero_order,
        scatter_use_residual=model_config.scatter_use_residual,
        scatter_use_neighbor_concat=model_config.scatter_use_neighbor_concat,
        scatter_neighbor_k=neighbor_k,
        scatter_neighbor_include_edge_weight=model_config.scatter_neighbor_include_edge_weight,
        k_neighbors=model_config.k_neighbors,
    )

    return config, model, dataloader_dict


def _prepare_vdw_modular_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a VDWModular model using explicit ModelConfig fields (no getattr fallbacks).
    """
    model_config = config.model_config
    dataset_config = config.dataset_config

    # Base module kwargs and ablations
    base_module_kwargs = _setup_base_module_kwargs(acc.device, config, dataset_config)
    ablate_vector_track, ablate_scalar_track = _setup_ablation_flags(config, dataset_config)

    # Resolve effective scalar feature key: if scalar and vector keys collide
    # and Diracs are enabled, dataset prep routed concatenated [pos | dirac] to 'x'.
    # _scalar_key_cfg = dataset_config.scalar_feat_key
    # _vector_key_cfg = dataset_config.vector_feat_key
    # _use_diracs = getattr(model_config, 'use_dirac_nodes', False)
    # _effective_scalar_key = 'x' if (_use_diracs and (_scalar_key_cfg == _vector_key_cfg)) else _scalar_key_cfg

    # Track kwargs
    scalar_track_kwargs = {
        'feature_key': dataset_config.scalar_feat_key,
        'diffusion_op_key': model_config.scalar_operator_key,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_scalar,
            'include_lowpass': model_config.include_lowpass_wavelet,
            'diffusion_scales': torch.tensor(model_config.scalar_diffusion_scales) \
                if model_config.scalar_diffusion_scales != 'dyadic' else 'dyadic',
        },
        # Reorganized explicit flags for clarity and compatibility with model
        'apply_scalar_first_order_nonlin': model_config.apply_scalar_first_order_nonlin,
        'scalar_scatter_first_order_nonlin': model_config.scalar_scatter_first_order_nonlin,
        # Number of scattering layers (>=1)
        'num_layers': model_config.num_scattering_layers_scalar,
    }

    vector_track_kwargs = None
    if not ablate_vector_track:
        vector_track_kwargs = {
            'feature_key': dataset_config.vector_feat_key,
            'diffusion_op_key': model_config.vector_operator_key,
            'vector_dim': dataset_config.vector_feat_dim,
            'diffusion_kwargs': {
                'scales_type': model_config.wavelet_scales_type,
                'J': model_config.J_vector,
                'include_lowpass': model_config.include_lowpass_wavelet,
                'diffusion_scales': torch.tensor(model_config.vector_diffusion_scales) \
                    if model_config.vector_diffusion_scales != 'dyadic' else 'dyadic',
            },
            # Reorganized flags controlling first-order vector gating/alignment
            'apply_vector_first_order_align_gating': model_config.apply_vector_first_order_align_gating,
            'first_order_align_gating_mode': getattr(model_config, 'vector_first_order_align_gating_mode', 'ref_align'),
            'vector_first_order_align_gating_nonlinearity': getattr(model_config, 'vector_first_order_align_gating_nonlinearity', 'softplus'),
            'replace_nonfinite_second_order': model_config.replace_nonfinite_second_order,
            # Number of scattering layers (>=1)
            'num_layers': model_config.num_scattering_layers_vector,
        }

    # Optional: pre-second-order nonlinearities / alignment flags
    # Safe getattr with defaults to avoid requiring new ModelConfig fields
    scalar_track_kwargs['apply_first_order_nonlin'] = model_config.apply_scalar_first_order_nonlin
    scalar_track_kwargs['first_order_nonlin'] = model_config.scalar_scatter_first_order_nonlin

    if vector_track_kwargs is not None:
        vector_track_kwargs['apply_vector_first_order_align_gating'] = model_config.apply_vector_first_order_align_gating

    # Mixing and neighbor kwargs from ModelConfig
    nonlin_map = MLP_NONLIN_MODULE_MAP
    mixing_kwargs = {
        'scalar_hidden_dims': model_config.scalar_wavelet_mlp_hidden,
        'scalar_dropout_p': model_config.scalar_wavelet_mlp_dropout,
        'scalar_nonlin': nonlin_map.get(
            model_config.scalar_wavelet_mlp_nonlin,
            nn.SiLU
        ),
        'W_out_scalar': model_config.W_out_scalar,
        'W_out_vector': model_config.W_out_vector,
        'use_scalar_batch_norm': model_config.use_scalar_wavelet_batch_norm,
        'use_vector_wavelet_batch_norm': model_config.use_vector_wavelet_batch_norm,
        'vector_bn_momentum': model_config.vector_bn_momentum,
        'vector_bn_eps': model_config.vector_bn_eps,
        'vector_bn_track_running_stats': model_config.vector_bn_track_running_stats,
        'vector_distance_kernel': dataset_config.local_pca_distance_kernel_scale,
        # New vector wavelet mixing gating mode and gate sigma for SimpleAffineGate
        'vector_wavelet_mixing_gate_mode': getattr(model_config, 'vector_wavelet_mixing_gate_mode', 'no_norm'),
        'vector_wavelet_mixer_gate_nonlinearity': getattr(model_config, 'vector_wavelet_mixer_gate_nonlinearity', 'sigmoid'),
        # New: single-linear vector mixer output dim override
        'vector_wavelet_mixer_linear_dim': getattr(model_config, 'vector_wavelet_mixer_linear_dim', None),
    }

    # Head kwargs are distinct from mixing kwargs; wire them explicitly
    head_kwargs = {
        'node_scalar_head_hidden': model_config.node_scalar_head_hidden,
        'node_scalar_head_nonlin': nonlin_map.get(model_config.node_scalar_head_nonlin, nn.SiLU),
        'node_scalar_head_dropout': model_config.node_scalar_head_dropout,
        'vector_gate_hidden': model_config.vector_gate_hidden,
        'vector_gate_mlp_nonlin': nonlin_map.get(model_config.vector_gate_mlp_nonlin, nn.SiLU),
        # Optional new gating behavior knobs (if present on ModelConfig)
        # Defaults are handled inside VDWModular when keys are missing
        'vector_gate_use_sigmoid': model_config.vector_gate_use_sigmoid,
        'vector_gate_init_temperature': getattr(model_config, 'vector_gate_init_temperature', 1.0),
        'use_scalar_in_vector_gate': model_config.use_scalar_in_vector_gate,
        'use_neighbor_cosines': model_config.use_neighbor_cosines,
        'use_learned_static_vector_weights': model_config.use_learned_static_vector_weights,
        'normalize_final_vector_gates': model_config.normalize_final_vector_gates,
        'vec_target_use_final_rotation_layer': model_config.vec_target_use_final_rotation_layer,
    }

    neighbor_kwargs = {
        'equal_degree': model_config.equal_degree,
        'k_neighbors': model_config.k_neighbors,
        'use_padding': model_config.neighbor_use_padding,
        'pool_stats': model_config.neighbor_pool_stats,
        'quantiles_stride': model_config.quantiles_stride
    }

    # Filter readout stats to supported ones ('mean','max','sum')
    _allowed_readout_stats = {'mean', 'max', 'sum'}
    _filtered_node_pool_stats = [s for s in model_config.neighbor_pool_stats if s in _allowed_readout_stats]
    if len(_filtered_node_pool_stats) == 0:
        _filtered_node_pool_stats = ['mean', 'max']

    readout_kwargs = {
        'type': 'agg' if model_config.equivar_pred else 'mlp',
        'mlp_hidden_dims': model_config.mlp_hidden_dim,
        'mlp_nonlin': model_config.mlp_nonlin,
        'node_pool_stats': _filtered_node_pool_stats,
    }

    model = VDWModular(
        base_module_kwargs=base_module_kwargs,
        ablate_scalar_track=ablate_scalar_track,
        ablate_vector_track=ablate_vector_track,
        scalar_track_kwargs=scalar_track_kwargs,
        vector_track_kwargs=vector_track_kwargs,
        mixing_kwargs=mixing_kwargs,
        neighbor_kwargs=neighbor_kwargs,
        head_kwargs=head_kwargs,
        readout_kwargs=readout_kwargs,
    )

    return config, model, dataloader_dict


def _prepare_vdw_macaque_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare VDW_macaque model: vector scattering from spike_data, invariants + time scalar → MLP.
    Supports both regression (kinematics) and multi-class classification (condition).
    """
    model_config = config.model_config
    dataset_config = config.dataset_config

    # Detect task type: multi-classification vs regression
    task = dataset_config.task.lower()
    is_classification = ('multi' in task) and ('class' in task)

    base_module_kwargs = _setup_base_module_kwargs(acc.device, config, dataset_config)
    
    # Branch configuration based on task type
    target_names = None
    num_classes = None
    
    if is_classification:
        # Classification: graph-level condition prediction (7 classes)
        num_classes = 7  # 7 conditions in macaque reaching dataset
        base_module_kwargs['target_name'] = 'condition_idx'
        base_module_kwargs['metrics_kwargs']['num_classes'] = num_classes
        # task_level forced to 'graph' by model class
    else:
        # Regression: detect multitask from target_key containing '+'
        target_key = dataset_config.target_key
        if isinstance(target_key, str) and ('+' in target_key):
            # Parse target names from "pos_xy+vel_xy" -> ("pos_xy", "vel_xy")
            target_names = tuple(p.strip() for p in target_key.split('+') if p.strip())
        
        # Ensure metrics track the main head (velocity: 2 dims) rather than concatenated 4
        try:
            if isinstance(base_module_kwargs.get('metrics_kwargs', None), dict):
                base_module_kwargs['metrics_kwargs']['num_outputs'] = 2
        except Exception:
            pass

    # Vector track kwargs (require dataset_config.vector_feat_key and model_config.vector_operator_key)
    vector_track_kwargs = {
        'feature_key': dataset_config.vector_feat_key,
        'diffusion_op_key': model_config.vector_operator_key,
        'vector_dim': dataset_config.vector_feat_dim,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_vector,
            'include_lowpass': model_config.include_lowpass_wavelet,
            'diffusion_scales': torch.tensor(model_config.vector_diffusion_scales) \
                if model_config.vector_diffusion_scales != 'dyadic' else 'dyadic',
        },
        'num_layers': model_config.num_scattering_layers_vector,
    }

    # Only vector mixer/head settings are used downstream
    nonlin_map = MLP_NONLIN_MODULE_MAP
    mixing_kwargs = {
        'W_out_vector': model_config.W_out_vector,
        'vector_wavelet_mixer_linear_dim': getattr(model_config, 'vector_wavelet_mixer_linear_dim', None),
        'vector_wavelet_mixing_gate_mode': getattr(model_config, 'vector_wavelet_mixing_gate_mode', 'no_norm'),
        'vector_wavelet_mixer_gate_nonlinearity': getattr(model_config, 'vector_wavelet_mixer_gate_nonlinearity', 'sigmoid'),
        'use_scalar_batch_norm': False,
    }
    head_kwargs = {
        'node_scalar_head_hidden': model_config.node_scalar_head_hidden,
        'node_scalar_head_nonlin': nonlin_map.get(model_config.node_scalar_head_nonlin, nn.SiLU),
        'node_scalar_head_dropout': model_config.node_scalar_head_dropout,
        # Neighbor cosines used by invariants helper
        'use_neighbor_cosines': getattr(model_config, 'use_neighbor_cosines', True),
    }

    # Determine whether to use vector-preserving path or invariant path
    use_vec_invariants = getattr(model_config, 'use_vec_invariants', True)
    # Determine graph aggregation mode (for graph-level tasks with invariants)
    graph_aggregation_mode = getattr(model_config, 'graph_aggregation_mode', 'pool_stats')
    # Determine scattering pipeline mode (path graphs vs spatial graph)
    scatter_path_graphs = getattr(model_config, 'scatter_path_graphs', False)
    
    model = VDW_macaque(
        base_module_kwargs=base_module_kwargs,
        vector_track_kwargs=vector_track_kwargs,
        mixing_kwargs=mixing_kwargs,
        head_kwargs=head_kwargs,
        target_names=target_names,
        use_vec_invariants=use_vec_invariants,
        graph_aggregation_mode=graph_aggregation_mode,
        task=task,
        num_classes=num_classes,
        scatter_path_graphs=scatter_path_graphs,
    )

    return config, model, dataloader_dict


def _prepare_vdw_macaque_supcon_2_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    model_config = config.model_config
    dataset_config = config.dataset_config

    def _resolve_activation(name: str) -> type[nn.Module]:
        mapping = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }
        return mapping.get(str(name).lower(), nn.ReLU)

    diffusion_scales = None
    vec_scales_cfg = getattr(model_config, 'vector_diffusion_scales')
    if isinstance(vec_scales_cfg, (list, tuple)):
        diffusion_scales = torch.tensor(vec_scales_cfg, dtype=torch.int64)

    diffusion_kwargs = {
        'scales_type': getattr(model_config, 'wavelet_scales_type', 'dyadic'),
        'diffusion_scales': diffusion_scales,
        'J': int(getattr(model_config, 'J_vector', 3)),
        'include_lowpass': bool(getattr(model_config, 'include_lowpass_wavelet', True)),
    }
    # Propagate scattering_k so LearnableMahalanobisTopK can infer its default k
    if getattr(model_config, 'scattering_k', None) is not None:
        diffusion_kwargs['scattering_k'] = int(model_config.scattering_k)

    # proj_hidden_cfg = getattr(model_config, 'projection_hidden_dim', 128)
    # if isinstance(proj_hidden_cfg, (list, tuple)):
    #     proj_hidden_dim = int(proj_hidden_cfg[0]) if proj_hidden_cfg else 128
    # else:
    #     proj_hidden_dim = int(proj_hidden_cfg)

    projection_activation = _resolve_activation(
        getattr(model_config, 'projection_activation', 'relu')
    )

    supcon_neighbor_k = int(model_config.supcon_neighbor_k)
    supcon_sampling_max_nodes = getattr(model_config, 'supcon_sampling_max_nodes', None)
    if supcon_sampling_max_nodes is None:
        supcon_sampling_max_nodes = int(config.batch_size)
    pos_pairs_per_anchor = model_config.pos_pairs_per_anchor
    neg_topk_per_positive = int(model_config.neg_topk_per_positive)
    random_negatives_per_anchor = int(model_config.random_negatives_per_anchor)
    learnable_p_kwargs = {
        'hidden_dim': [128, 128],
        'embedding_dim': 1,
        'activation': nn.ReLU,
        'use_batch_norm': False,
    }

    # Learnable Mahalanobis top-k settings
    use_learnable_topk = bool(getattr(model_config, 'use_learnable_topk', False))
    learnable_topk_k = getattr(model_config, 'learnable_topk_k', None)
    learnable_topk_proj_dim = getattr(model_config, 'learnable_topk_proj_dim', None)
    learnable_topk_temperature = float(getattr(model_config, 'learnable_topk_temperature', 1.0))
    learnable_topk_eps = float(getattr(model_config, 'learnable_topk_eps', 1e-8))

    use_distill_inference = False
    if hasattr(model_config, 'use_distill_inference'):
        use_distill_inference = bool(model_config.use_distill_inference)
    model = VDW_macaque_supcon_2(
        vector_feat_key=getattr(dataset_config, 'vector_feat_key', 'v_vel'),
        scalar_feature_key=getattr(
            dataset_config,
            'scalar_feat_key',
            'v_state',
        ),
        target_key=getattr(dataset_config, 'target_key', 'condition_idx'),
        diffusion_kwargs=diffusion_kwargs,
        num_scattering_layers=int(getattr(model_config, 'num_scattering_layers_vector', 2)),
        use_vector_batch_norm=bool(getattr(model_config, 'vector_bn_enabled', True)),
        projection_hidden_dim=getattr(model_config, 'projection_hidden_dim', [128, 128]),
        projection_embedding_dim=int(getattr(model_config, 'projection_embedding_dim', 3)),
        projection_activation=projection_activation,
        projection_residual_style=bool(getattr(model_config, 'projection_residual_style', False)),
        projection_dropout_p=getattr(model_config, 'projection_dropout_p', None),
        temperature=float(getattr(model_config, 'temperature', 0.1)),
        num_classes=int(getattr(dataset_config, 'target_dim', 7)),
        use_distill_inference=use_distill_inference,
        device=acc.device if acc is not None else None,
        verbosity=config.verbosity,
        supcon_sampling_max_nodes=supcon_sampling_max_nodes,
        supcon_neighbor_k=supcon_neighbor_k,
        pos_pairs_per_anchor=pos_pairs_per_anchor,
        neg_topk_per_positive=neg_topk_per_positive,
        random_negatives_per_anchor=random_negatives_per_anchor,
        learnable_P=bool(getattr(model_config, 'learnable_P', False)),
        learnable_p_kwargs=learnable_p_kwargs,
        learnable_p_num_views=int(getattr(model_config, 'learnable_p_num_views', 1)),
        learnable_p_view_aggregation=getattr(
            model_config, 'learnable_p_view_aggregation', 'concat'
        ),
        learnable_p_softmax_temps=getattr(
            model_config, 'learnable_p_softmax_temps', None
        ),
        learnable_p_laziness_inits=getattr(
            model_config, 'learnable_p_laziness_inits', None
        ),
        learnable_p_use_softmax=bool(
            getattr(model_config, 'learnable_p_use_softmax', False)
        ),
        learnable_p_fix_alpha=bool(
            getattr(model_config, 'learnable_p_fix_alpha', False)
        ),
        learnable_p_use_attention=bool(
            getattr(model_config, 'learnable_p_use_attention', False)
        ),
        learnable_p_attention_kwargs=getattr(
            model_config, 'learnable_p_attention_kwargs', None
        ),
        use_learnable_topk=use_learnable_topk,
        learnable_topk_k=learnable_topk_k,
        learnable_topk_proj_dim=learnable_topk_proj_dim,
        learnable_topk_temperature=learnable_topk_temperature,
        learnable_topk_eps=learnable_topk_eps,
    )
    # Attach eval-mode knobs for downstream evaluation helpers
    model.eval_mode = getattr(model_config, "eval_mode", "weighted_average")
    model.holdout_k_probe = int(getattr(model_config, "holdout_k_probe", 10))

    return config, model, dataloader_dict


# ============================================================
#               COMPARISON MODEL PREPARATION                 
# ============================================================
def prepare_comparison_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None,
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a comparison model (EGNN, TFN, LEGS) by 
    wrapping it in a BaseModule-compatible adapter.
    """
    model_config = config.model_config
    model_key = model_config.model_key
    dataset_config = config.dataset_config
    task_lower = dataset_config.task.lower()
    print(f"_prepare_comparison_model: task_lower: {task_lower}")

    # BaseModule kwargs – ensure correct loss/metric setup
    base_module_kwargs = {
        'task': dataset_config.task,
        'metrics_kwargs': {
            'num_outputs': dataset_config.target_dim,
        },
        'verbosity': config.verbosity,
    }

    # Map loss function
    if hasattr(config, 'loss_fn') and config.loss_fn in LOSS_FN_MAP:
        base_module_kwargs['loss_fn'] = LOSS_FN_MAP[config.loss_fn]
        if config.loss_fn == 'huber':
            base_module_kwargs['loss_fn_kwargs'] = {
                'beta': getattr(config, 'huber_delta'),
                'reduction': 'mean',
            }
        else:
            base_module_kwargs['loss_fn_kwargs'] = {
                'reduction': 'mean',
            }

    # Instantiate underlying PyG model
    if model_key == 'egnn':
        # If the dataset does not define atom types, pass in_dim=None to enable
        # EGNN's bias-initialized node features path
        inferred_in_dim = None
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                inferred_in_dim = nat if nat > 0 else None
            except Exception:
                inferred_in_dim = None
        ds_name = getattr(dataset_config, 'dataset', None)
        ds_is_wind = isinstance(ds_name, str) and (ds_name.lower() in ('wind', 'wind_rot'))
        pyg_model = EGNNModel(
            num_layers=model_config.comparison_model_num_layers,
            emb_dim=model_config.node_embedding_dim,
            in_dim=inferred_in_dim,
            out_dim=dataset_config.target_dim,
            # activation=model_config.mlp_nonlin,
            pool_types=model_config.pooling_type,
            # Follow config flag to control equivariant processing; the adapter/head will handle invariant targets.
            equivariant_pred=model_config.equivar_pred,
            predict_per_node=('node' in task_lower),
            vector_target=('vector' in task_lower),
            use_edge_weight=ds_is_wind,
        )
        # Wind experiments: treat wind vectors as EGNN's equivariant "pos" input,
        # while retaining graph connectivity from geographic positions.
        if ds_is_wind:
            vec_key = getattr(dataset_config, 'vector_feat_key', None)
            if isinstance(vec_key, str) and len(vec_key) > 0:
                setattr(pyg_model, 'pos_input_key', vec_key)
    elif model_key == 'tfn':
        # Infer in_dim similarly to EGNN: None -> bias path
        inferred_in_dim = None
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                inferred_in_dim = nat if nat > 0 else None
            except Exception:
                inferred_in_dim = None
        ds_name = getattr(dataset_config, 'dataset', None)
        ds_is_wind = isinstance(ds_name, str) and (ds_name.lower() in ('wind', 'wind_rot'))
        pyg_model = TFNModel(
            # Wind: wind-diff norms can approach ~55.4 when components are in [-16, 16],
            # so use a cutoff comfortably above this to avoid near-zero radial weights.
            r_max=64.0 if ds_is_wind else model_config.tfn_r_max,
            num_bessel=model_config.tfn_num_bessel,
            num_polynomial_cutoff=model_config.tfn_num_polynomial_cutoff,
            max_ell=model_config.tfn_max_ell,
            num_layers=model_config.comparison_model_num_layers,
            emb_dim=model_config.node_embedding_dim,
            hidden_irreps=None,
            mlp_dim=model_config.tfn_mlp_dim,
            in_dim=inferred_in_dim,
            out_dim=dataset_config.target_dim,
            aggr='sum',
            pool_types=model_config.pooling_type,
            gate=True,
            batch_norm=False,
            residual=True,
            # Follow config flag to control equivariant processing
            equivariant_pred=model_config.equivar_pred,
            predict_per_node=('node' in task_lower),
            vector_target=('vector' in task_lower),
            use_bias_if_no_atoms=True,
            # Wind: learn equivariant geometry in wind-vector space (lengths from wind diffs),
            # while still incorporating geographic edge strength via the precomputed edge_weight.
            use_edge_weight_as_length=(not ds_is_wind),
            use_edge_strength_feature=ds_is_wind,
            edge_strength_key="edge_weight",
            radial_mode=model_config.tfn_radial_mode,
            radial_mlp_hidden=model_config.tfn_radial_mlp_hidden,
            radial_mlp_activation=model_config.tfn_radial_mlp_activation,
            unbiased_vector_pred_head=model_config.tfn_unbiased_vector_pred_head,
            radial_kernel_gaussian_eps=dataset_config.local_pca_distance_kernel_scale
,
        )
        # Wind experiments: treat wind vectors as TFN's equivariant "pos" input,
        # while retaining graph connectivity from geographic positions.
        if ds_is_wind:
            vec_key = getattr(dataset_config, 'vector_feat_key', None)
            if isinstance(vec_key, str) and len(vec_key) > 0:
                setattr(pyg_model, 'pos_input_key', vec_key)
    elif model_key == 'legs':
        # Determine feature attribute and in_channels (using vector features as scalar channels)
        # scalar_key_cfg = dataset_config.scalar_feat_key
        # vector_key_cfg = dataset_config.vector_feat_key
        # # If scalar and vector keys collide (e.g., both 'pos'), we routed the
        # # composed scalar features to 'x' in dataset prep; use that here.
        # feat_key = 'x' \
        #     if (scalar_key_cfg == vector_key_cfg) and (model_config.use_dirac_nodes) \
        #     else scalar_key_cfg
        in_channels = dataset_config.vector_feat_dim
        if model_config.use_dirac_nodes:
            n_diracs = len(getattr(model_config, 'dirac_types', []) or [])
            in_channels += n_diracs

        pyg_model = LEGSModel(
            in_channels=in_channels,
            output_dim=dataset_config.target_dim,
            feature_attr=dataset_config.scalar_feat_key,
            J=model_config.J_scalar,
            # n_moments=model_config.moments[0] if hasattr(model_config, 'moments') else 4,
            # trainable_laziness=False,
            # apply_modulus_to_scatter=True,
            pool_types=model_config.pooling_type,
            predict_per_node=('node' in task_lower),
            mlp_head_hidden_dims=model_config.mlp_hidden_dim,
            node_mlp_dim=model_config.mlp_hidden_dim[0],
            activation=model_config.mlp_nonlin,
            mlp_dropout_p=model_config.mlp_dropout_p,
        )
    elif model_key in ('gcn', 'gin', 'gat'):
        # Shared setup for GCN/GIN/GAT backbones
        pool_types = model_config.pooling_type
        if isinstance(pool_types, str):
            pool_types = [pool_types]
        in_dim = 0
        scalar_key = getattr(dataset_config, 'scalar_feat_key', None)
        vector_key = getattr(dataset_config, 'vector_feat_key', None)
        if (scalar_key is not None) and (scalar_key == vector_key):
            in_dim = getattr(dataset_config, 'vector_feat_dim', 0) or 0
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                if in_dim == 0:
                    in_dim = nat if nat > 0 else getattr(dataset_config, 'vector_feat_dim', 0)
            except Exception:
                if in_dim == 0:
                    in_dim = getattr(dataset_config, 'vector_feat_dim', 0)
        else:
            if in_dim == 0:
                in_dim = getattr(dataset_config, 'vector_feat_dim', 0)
        pyg_model = PyGModel(
            in_dim=in_dim,
            hidden_channels=model_config.node_embedding_dim or model_config.comparison_model_hidden_channels,
            out_dim=dataset_config.target_dim,
            num_layers=2 if model_config.comparison_model_num_layers is None else model_config.comparison_model_num_layers,
            feature_attr=dataset_config.scalar_feat_key,
            pool_types=pool_types,
            predict_per_node=('node' in task_lower),
            mlp_head_hidden_dims=model_config.mlp_hidden_dim,
            mlp_dropout_p=model_config.mlp_dropout_p,
            activation=model_config.mlp_nonlin,
            backbone=model_key,
        )
    elif model_key == 'tnn':
        sample = dataloader_dict['train'].dataset[0]
        if not hasattr(sample, dataset_config.vector_feat_key):
            raise RuntimeError(f"TNN requires '{dataset_config.vector_feat_key}' on data samples.")
        if not hasattr(sample, 'tnn_operator'):
            raise RuntimeError("TNN preprocessing not found on dataset; ensure attach_tnn_operators was called.")
        pyg_model = TNNComparisonModel(
            base_module_kwargs=base_module_kwargs,
            tnn_operator=sample.tnn_operator,
            d_hat=int(getattr(sample, 'tnn_d_hat')),
            vector_feat_key=dataset_config.vector_feat_key,
            pool_types=model_config.pooling_type,
            mlp_hidden_dim=model_config.mlp_hidden_dim,
            mlp_nonlin=model_config.mlp_nonlin,
            mlp_dropout_p=model_config.mlp_dropout_p,
            target_task=dataset_config.task,
        )
    else:
        raise ValueError(f"Unsupported comparison model_key: {model_key}")

    # Wrap in ComparisonModel
    atomic_number_key = getattr(dataset_config, 'atomic_number_attrib_key', 'z')
    model = ComparisonModel(
        pyg_model=pyg_model,
        base_module_kwargs=base_module_kwargs,
        atomic_number_key=atomic_number_key,
    )

    # Move to device
    if acc is not None:
        model = model.to(acc.device)
    else:
        model = model.to(config.device)

    return config, model, dataloader_dict