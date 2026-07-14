import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch_geometric.data import Data, Batch, Dataset
from typing import Optional, Tuple, Dict, Any, List, Callable, Literal
# import h5py
from torch_geometric.nn import (
    global_mean_pool, global_max_pool, global_add_pool
)
from config.train_config import TrainingConfig
import models.base_module as bm
from models.vanilla_nn import VanillaNN
# from infogain import get_avg_infogain_wavelet_scales
from geo_scat import batch_scatter
import copy
from models.nn_utilities import (
    get_stat_moments, 
    merge_dicts_with_defaults,
)
from models import nn_utilities as nnu
from models.class_maps import (
    VECTOR_NONLIN_FN_MAP,
    GATE_FN_MAP,
    GATE_INIT_MAP,
    FUNCTION_TO_MODULE_MAP,
)

try:
    import torch_scatter
    _TORCH_SCATTER_AVAILABLE = True
except:
    print("[VDW] Warning: torch_scatter is not available. Some pooling operations will likely be slower.")
    torch_scatter = None
    _TORCH_SCATTER_AVAILABLE = False

            

class VDW(bm.BaseModule):
    """
    VDW (SO-equivariant LEarnable Scattering) model.
    Implements two parallel tracks for processing scalar and 
    vector features using diffusion operators P and Q, 
    respectively.

    This model has two 'modes':
    - 'handcrafted_scattering': the forward method returns a tensor of the 
    (readout) scattering features, for use with a (separate)
    classifier or regressor as the model (hence no backward
    training step is needed to generate the scattering features)
    - 'filter-combine': the model adds learnable weights
    to recombine wavelet-filtered features (within track and graphs
    channels) with each layer, as a form of learnable scattering

    NOTE: for graph-level tasks (where a readout function is needed)
    be sure to include the substring 'graph' in the task name
    passed in base_module_kwargs. For node-level (single graph)
    tasks, include the substring 'node' in the task name.
    """
    
    # ------------------------------------------------------------------
    # Class-level default configuration dictionaries (treated as read-only)
    # ------------------------------------------------------------------
    DEFAULT_SCALAR_KWARGS: Dict[str, Any] = {
        'diffusion_op_key': 'P',
        'feature_key': 'x',
        'filter_combos_out': (16, 8),
        'use_skip': True,
        'num_layers': 2,
        'nonlin_fn': F.relu,
        'nonlin_fn_kwargs': {},
        'diffusion_scales': None,
        'diffusion_kwargs': {
            'scales_type': 'dyadic',
            'J': 4,
            'include_lowpass': True,
            'rescale_filtrations': False,  # Disabled in favor of batch normalization
            'rescale_method': 'standardize',
        },
        'scattering_pooling_kwargs': {
            'pooling_type': 'statistical_moments',
            'moments': (1, 2, 3),
            'nan_replace_value': 0.0,
        },
    }

    DEFAULT_VECTOR_KWARGS: Dict[str, Any] = {
        'diffusion_op_key': 'Q',
        'feature_key': 'pos',
        'vector_dim': 3,
        'filter_combos_out': (16, 8),
        'use_skip': True,
        'num_layers': 2,
        'vector_nonlin_type': 'shifted_relu',
        'vector_norm_p': 2,
        'nonlin_fn_kwargs': {},
        'diffusion_scales': None,
        'diffusion_kwargs': {
            'scales_type': 'dyadic',
            'J': 4,
            'include_lowpass': True,
            'rescale_filtrations': False,  # Disabled in favor of batch normalization
            'rescale_method': 'standardize',
        },
        'scattering_pooling_kwargs': {
            'norm_p': 2,
            'pooling_type': 'statistical_moments',
            'moments': (1, 2, 3),
            'nan_replace_value': 0.0,
        },
    }

    DEFAULT_MLP_KWARGS: Dict[str, Any] = {
        'hidden_dims_list': [256, 128, 64, 32],
        'output_dim': 1,
        'use_dropout': True,
        'dropout_p': 0.2,
        'use_batch_normalization': True,
        'batch_normalization_kwargs': {
            'affine': True,
            'eps': 1e-5,
            'momentum': 0.1,
        },
        'wt_init_fn': nn.init.kaiming_normal_,
        'wt_init_fn_kwargs': {
            'nonlinearity': 'relu',
            'mode': 'fan_in',
        },
    }

    DEFAULT_BASE_KWARGS: Dict[str, Any] = {}

    def __init__(
        self,

        # Model mode: 'handcrafted_scattering' or 'filter-combine'
        mode: Literal[
            'handcrafted_scattering',
            'filter-combine',
            'cross-track',
        ] = 'handcrafted_scattering',

        # Track configurations
        scalar_track_kwargs: Dict[str, Any] = None,
        vector_track_kwargs: Dict[str, Any] = None,
        
        # MLP configuration
        mlp_kwargs: Dict[str, Any] = None,
        
        # Base module parameters
        base_module_kwargs: Dict[str, Any] = None,

        # Cross-track mode parameters (only used when mode == 'cross-track')
        cross_track_kwargs: Dict[str, Any] = None,

        # Ablation options
        ablate_vector_track: bool = False,
        ablate_scalar_track: bool = False,
        ablate_vector_wavelet_batch_norm: bool = False,

        # Parallelization options
        stream_parallelize_tracks: bool = False,

        # Normalization options
        use_input_normalization: bool = False,
        normalization_eps: float = 1e-8,

        # Verbosity of print output
        verbosity: int = 0,
    ):
        """
        Initialize the VDW model.

        Args:
            mode: Network mode, either 'handcrafted_scattering' or 'filter-combine'
            scalar_track_kwargs: Dictionary of parameters for scalar track
            vector_track_kwargs: Dictionary of parameters for vector track
            mlp_kwargs: Dictionary of keyword arguments for the final MLP
            base_module_kwargs: Additional keyword arguments for BaseModule
            stream_parallelize_tracks: Whether to process scalar and vector tracks
            use_input_normalization: Whether to normalize input to MLP
            normalization_eps: Epsilon value for numerical stability in     normalization
            verbosity: Verbosity of print output
        """
        # Prepare/merge parameter dictionaries
        (
            self.scalar_track_kwargs,
            self.vector_track_kwargs,
            self.mlp_kwargs,
            self.base_module_kwargs,
        ) = self._prepare_kwarg_dicts(
            scalar_track_kwargs,
            vector_track_kwargs,
            mlp_kwargs,
            base_module_kwargs,
        )

        super(VDW, self).__init__(**self.base_module_kwargs)

        # Apply task-based default activation choices
        self._apply_task_based_default_activations()

        # Flag indicating that this model lazily creates parameters (MLP head)
        # when running the first forward pass in 'filter-combine' mode.
        # Training utilities can use this to decide whether to run a dummy
        # forward pass before wrapping with DDP.
        self.has_lazy_parameter_initialization = (mode in ('filter-combine', 'cross-track'))

        self.mode = mode
        self.ablate_vector_track = ablate_vector_track
        self.ablate_scalar_track = ablate_scalar_track
        self.ablate_vector_wavelet_batch_norm = ablate_vector_wavelet_batch_norm
        self.stream_parallelize_tracks = stream_parallelize_tracks
        self.verbosity = verbosity
        self.use_input_normalization = use_input_normalization
        self.normalization_eps = normalization_eps
        
        # Initialize batch normalization if enabled
        self.input_batch_norm = None
        if self.use_input_normalization:
            # Will be initialized on first forward pass when we know the input dimension
            self._input_batch_norm_initialized = False
        
        # Flag to track if MLP has been initialized
        self._mlp_initialized = False
        
        # Validate track configurations & initialize within-track layers
        self._validate_and_init_track_layers(
            mode,
            scalar_layers=scalar_track_kwargs['num_layers'],
            vector_layers=vector_track_kwargs['num_layers'],
            cross_track_kwargs=cross_track_kwargs,
        )
        
        # Configure vector-track nonlinearity / gating
        self._configure_vector_nonlinearity()

        # Cross-track pathway initialization
        self._init_cross_track_layers(cross_track_kwargs)

        # Ensure scalar/vector layer ModuleLists exist when not combined
        within_track_combine_flag = self.cross_track_kwargs.get('within_track_combine', False) if self.cross_track_kwargs else False
        self._ensure_track_layer_placeholders(within_track_combine_flag)

        # Ensure required vector-specific parameters (e.g., shift) exist
        self._ensure_vector_special_params()
        
        # Initialize batch normalization for concatenated vector features (norms + cosines)
        # In cross-track mode we will lazily create **one BatchNorm1d per layer** so that
        # every (channel × wavelet) pair in that layer gets its own affine parameters.
        # This satisfies the requirement of "unique µ/σ per wavelet within each channel
        # within each layer".
        if self.mode == 'cross-track' and not self.ablate_vector_track:
            self.vector_features_bn_layers: nn.ModuleList = nn.ModuleList()
        else:
            self.vector_features_bn_layers = None
        
        # Helper lists for LazyBatchNorm1d applied right after scattering (one per scattering layer)
        self.scalar_wjxs_lazy_bn_layers: nn.ModuleList | None = None
        self.vector_wjxs_lazy_bn_layers: nn.ModuleList | None = None

        # Initialize Wjxs batch normalization layers (lazy initialization)
        # Initialize as ModuleLists if in cross-track mode, otherwise None
        if self.mode == 'cross-track':
            self.scalar_wjxs_bn_layers = nn.ModuleList()
            self.vector_wjxs_bn_layers = nn.ModuleList() if not self.ablate_vector_track else None
            # New: per-layer LazyBatchNorm1d lists applied right after scattering
            self.scalar_wjxs_lazy_bn_layers = nn.ModuleList()
            self.vector_wjxs_lazy_bn_layers = nn.ModuleList() if not self.ablate_vector_track else None
        else:
            self.scalar_wjxs_bn_layers = None
            self.vector_wjxs_bn_layers = None
            self.scalar_wjxs_lazy_bn_layers = None
            self.vector_wjxs_lazy_bn_layers = None

        # --------------------------------------------------------------
        # Optional scalar feature embedding (moved to helper for clarity)
        # --------------------------------------------------------------
        self._init_scalar_embedding()

    
    def _calculate_initial_num_filters(
        self,
        diffusion_kwargs: Dict[str, Any],
    ) -> int:
        """
        Calculate the initial number of filters based on diffusion kwargs.
        
        Args:
            diffusion_kwargs: Dictionary containing wavelet configuration parameters
            
        Returns:
            Number of initial filters
        """
        # --- Custom scales ---
        custom_scales = diffusion_kwargs.get('diffusion_scales', None)
        if custom_scales is not None:
            num_filters = custom_scales.shape[0] - 1 \
                if custom_scales.dim() == 1 \
                else custom_scales.shape[1] - 1  # if 2-D tensor, rows are channels

        # --- Dyadic scales (default, if no custom scales provided) ---
        else:
            # Dyadic wavelets: J band-pass filters
            num_filters = diffusion_kwargs['J'] + 1

        # --- Low-pass filter ---
        include_lowpass = diffusion_kwargs['include_lowpass']
        if include_lowpass:
            num_filters += 1

        return num_filters
        

    def _init_within_track_parameters(self, nonlinearity: str = 'relu', *, create_layers: bool = True):
        """Initialize learnable parameters for both tracks."""
        # Always ensure ModuleList attributes exist
        self.scalar_track_layers = nn.ModuleList()
        self.vector_track_layers = nn.ModuleList()

        if create_layers:
            # ----------------------- scalar track -----------------------
            in_channels = self._calculate_initial_num_filters(self.scalar_track_kwargs['diffusion_kwargs'])
            for layer_idx in range(self.scalar_track_kwargs['num_layers']):
                if (self.mode in ('filter-combine', 'cross-track')) and (layer_idx == 1):
                    in_channels = self.scalar_track_kwargs['J_prime']
                out_channels = self.scalar_track_kwargs['filter_combos_out'][layer_idx]
                layer = nn.Linear(in_channels, out_channels)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
                nn.init.zeros_(layer.bias)
                self.scalar_track_layers.append(layer)
                in_channels = out_channels

            # ----------------------- vector track -----------------------
            in_channels = self._calculate_initial_num_filters(self.vector_track_kwargs['diffusion_kwargs'])
        
        # Vector nonlinearity shift parameter (always needed for shifted ReLU)
        
        # If using shifted ReLU nonlinearity, initialize learnable shift parameter for vector norms
        if self.vector_track_kwargs['vector_nonlin_type'] == 'shifted_relu':
            self.vector_norm_shift = nn.Parameter(
                torch.tensor(0.1, dtype=torch.float32), 
                requires_grad=True
            )
            self.vector_track_kwargs['nonlin_fn_kwargs'] = {
                'shift': self.vector_norm_shift
            }
        
        if create_layers:
            for layer_idx in range(self.vector_track_kwargs['num_layers']):
                if (self.mode in ('filter-combine', 'cross-track')) and (layer_idx == 1):
                    # 2nd-order scattering creates J_prime wavelets
                    in_channels = self.vector_track_kwargs['J_prime']
                out_channels = self.vector_track_kwargs['filter_combos_out'][layer_idx]
                # NOTE: no bias in vector track layers -> preserve SO(d) equivariance
                layer = nn.Linear(in_channels, out_channels, bias=False)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
                self.vector_track_layers.append(layer)
                in_channels = out_channels


    def _compute_num_wavelets(
            self, 
            diffusion_kwargs: Dict[str, Any]
        ) -> int:
        """
        Compute the number of wavelets based on diffusion kwargs.
        
        Args:
            diffusion_kwargs: Dictionary containing wavelet configuration parameters
            
        Returns:
            Number of wavelets, including lowpass if specified
        """
        if diffusion_kwargs.get('scales') is None:
            # Dyadic wavelets: J + 1 wavelets
            num_wavelets = diffusion_kwargs['J'] + 1
        else:
            # Custom scales: length of scales - 1 wavelets
            num_wavelets = len(diffusion_kwargs['scales']) - 1
            
        # Add lowpass wavelet if included
        if diffusion_kwargs.get('include_lowpass', True):
            num_wavelets += 1
            
        return num_wavelets


    def _node_pooling(
        self,
        x: torch.Tensor,
        pooling_type: Tuple[str, ...],
        moments: Tuple[int],
        nan_replace_value: Optional[float] = None,
        batch_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool features across nodes for each graph in the batch.
        Useful in graph-level prediction tasks.
        - For the scalar track, this function generates statistics of each
        channel-wavelet's scattering coefficients across nodes (or original features in the skip connection).
        - For the vector track, feed this function the p-norms of each vector's wavelet scattering coefficients across nodes, to generate statistics of the norms of the vector features.

        Args:
            x: Tensor of shape (batch_num_nodes, num_channels, num_wavelets) = (N, C, W) for scalar features.
            pooling_type: Tuple of pooling types to apply
            moments: Tuple of moments to compute
            nan_replace_value: Value to replace NaNs with
            batch_index: Tensor indicating which graph each node belongs to
        Returns:
            Pooled tensor of shape (num_graphs, num_channels, 
            num_pooled_features, num_wavelets) = (B, C, S, W).
        """
        def handle_pooling_exception(
            e: Exception, 
            p_type: str, 
            num_channels: int, 
            num_wavelets: int, 
            num_moments: Optional[int] = None, 
            context: Optional[str] = None
        ) -> torch.Tensor:
            """
            Inner method to handle pooling exceptions: print error message and return
            a tensor of zeros with the appropriate shape.
            """
            msg = f"[VDW][_pool_features] EXCEPTION in pooling type '{p_type}': {e}"
            if context is not None:
                msg += f"\n  context: {context}"
            print(msg)
            if p_type == 'statistical_moments':
                return torch.zeros((num_channels, num_moments, num_wavelets), device=x.device)
            else:
                return torch.zeros((num_channels, 1, num_wavelets), device=x.device)

        def postprocess_pooled(
            pooled: torch.Tensor, 
            num_channels: int, 
            num_wavelets: int
        ) -> torch.Tensor:
            """
            Inner method to postprocess (reshape) pooled features.
            """
            pooled = pooled.view(-1, num_channels, num_wavelets)
            pooled = pooled.unsqueeze(2)  # (num_graphs, num_channels, 1, num_wavelets)
            return pooled

        # Initialize list to store results from each pooling type
        pooled_results = []

        # Get number of channels and wavelets from graph_Wjxs
        num_channels, num_wavelets = x.shape[1], x.shape[2]

        # Flatten graph_Wjxs to (num_graphs, num_channels * num_wavelets)
        flat = x.view(x.shape[0], -1)

        # Define pooling operations as lambda functions contained in a dict
        def median_pool_fallback(
            flat: torch.Tensor, 
            batch_index: torch.Tensor
        ) -> torch.Tensor:
            """
            Fallback method to compute median pooling when torch_scatter is not available.
            """
            unique_groups = batch_index.unique(sorted=True)
            medians = []
            for group in unique_groups:
                mask = (batch_index == group)
                group_flat = flat[mask]
                if group_flat.shape[0] == 0:
                    # No elements in this group, fill with NaN
                    medians.append(torch.full((1, flat.shape[1]), float('nan'), device=flat.device, dtype=flat.dtype))
                else:
                    medians.append(group_flat.median(dim=0, keepdim=True).values)
            return torch.cat(medians, dim=0)  # (num_groups, D)

        pooling_ops = {
            'mean': lambda: global_mean_pool(flat, batch_index) \
                if batch_index is not None \
                else flat.mean(dim=0, keepdim=True),
            'sum': lambda: global_add_pool(flat, batch_index) \
                if batch_index is not None \
                else flat.sum(dim=0, keepdim=True),
            'max': lambda: global_max_pool(flat, batch_index) \
                if batch_index is not None \
                else flat.max(dim=0, keepdim=True)[0],
        }
        if _TORCH_SCATTER_AVAILABLE:
            pooling_ops['median'] = lambda: torch_scatter.scatter(
                flat, batch_index.unsqueeze(1).expand_as(flat), dim=0, reduce='median') \
                if batch_index is not None \
                else flat.median(dim=0, keepdim=True).values
        else:
            pooling_ops['median'] = lambda: median_pool_fallback(flat, batch_index) \
                if batch_index is not None \
                else flat.median(dim=0, keepdim=True).values

        # Loop through pooling types and apply the appropriate pooling operation
        for p_type in pooling_type:
            if p_type == 'statistical_moments':
                try:
                    pooled = get_stat_moments(
                        x,
                        moments=moments,
                        nan_replace_value=nan_replace_value,
                        batch_index=batch_index
                    )
                    if batch_index is not None:
                        pooled = pooled.permute(1, 2, 0, 3)  # (num_graphs, num_channels, num_moments, num_wavelets)
                    else:
                        pooled = pooled.permute(1, 0, 2)
                        pooled = pooled.unsqueeze(0)
                    pooled_results.append(pooled)
                except Exception as e:
                    context = f"statistical_moments, graph_Wjxs.shape={x.shape}"
                    pooled_results.append(handle_pooling_exception(
                        e, p_type, num_channels, num_wavelets, len(moments), context=context
                    ).unsqueeze(0))
            elif p_type in pooling_ops:
                try:
                    pooled = pooling_ops[p_type]()
                    pooled = postprocess_pooled(pooled, num_channels, num_wavelets)
                    pooled_results.append(pooled)
                except Exception as e:
                    context = f"{p_type} pooling"
                    pooled_results.append(handle_pooling_exception(
                        e, p_type, num_channels, num_wavelets, context=context
                    ).unsqueeze(0))
            else:
                raise ValueError(f"Unknown pooling type: {p_type}")
            
        # Concatenate results from all pooling types in the pooling feature dimension
        if len(pooled_results) > 1:
            return torch.cat(pooled_results, dim=2)
        else:
            return pooled_results[0]


    def _pool(
        self,
        mode: Literal['node', 'channel'],
        x: torch.Tensor,
        batch_index: torch.Tensor,
        pooling_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Pool features across nodes or channels for each graph in the batch.
        
        Args:
            x: Tensor of shape (batch_num_nodes, num_channels, num_wavelets) = (N, C, W) for scalar features.
            batch_index: Tensor indicating which graph each node belongs to
            pooling_kwargs: Dictionary of pooling parameters
        Returns:
            Pooled tensor of shape (num_graphs, num_channels, 
            num_pooled_features, num_wavelets) = (B, C, S, W).
        """
        pooling_type = pooling_kwargs['pooling_type']
        moments = pooling_kwargs.get('moments', (1, 2, 3))
        nan_replace_value = pooling_kwargs.get('nan_replace_value', None)

        # Call the batch-wise pooling function
        if mode == 'node':
            pooled_features = self._node_pooling(
                x=x,
                pooling_type=pooling_type,
                moments=moments,
                nan_replace_value=nan_replace_value,
                batch_index=batch_index
            )
        elif mode == 'channel':
            pooled_features = self._channel_pool(
                x=x,
                batch_index=batch_index,
                pooling_kwargs=pooling_kwargs
            )
        return pooled_features
    

    def _channel_pool(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        pooling_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Pool features across channels for each graph in the batch.
        Useful in node-level prediction tasks.
        NOTE: for now, this function simply reshapes/views x such that 
        all node feature tensors are unrolled into a single dimension.

        Args:
            x: Tensor of shape (batch_num_nodes, num_channels, num_wavelets) = (N, C, W) for scalar features.
            batch_index: Tensor indicating which graph each node belongs to
            pooling_kwargs: Dictionary of pooling parameters
        Returns:
            Pooled tensor of shape (batch_num_nodes, num_channels*num_wavelets) = (N, C*W).
        """
        return x.view(x.shape[0], -1)


    def _vector_norm_nonlinearity(
        self,
        vectors: torch.Tensor,
        shift: Optional[float] = 0.1,
        p: int = 2,
        eps: float = 1e-8,
        nonlinearity: Callable = torch.relu
    ) -> torch.Tensor:
        """
        SO(3)-equivariant nonlinearity for vector features using a norm-based nonlinearity.
        Args:
            vectors: (..., d) tensor of vectors (d=3 for SO(3))
            shift: scalar, the shift for ReLU (if applicable; None uses self.vector_norm_shift)
            p: order of the norm (default 2)
            eps: small value to avoid division by zero
            nonlinearity: function to apply to the norm (default torch.relu, or torch.tanh, etc)
        Returns:
            (... ,d) tensor with nonlinearity applied
        This operation is SO(3)-equivariant: rotating then applying this is the same as applying then rotating.
        """
        norms = torch.norm(vectors, p=p, dim=-1, keepdim=True)  # (..., 1)
        if nonlinearity == torch.relu:
            if shift is None:
                shift = self.vector_norm_shift
            new_norms = torch.relu(norms - shift)
        else:
            new_norms = nonlinearity(norms)
            
        scale = torch.where(
            norms > eps, 
            new_norms / (norms + eps), 
            torch.zeros_like(norms)
        )
        return vectors * scale


    def _track_forward(
        self,
        track_name: str,
        batch: Batch,
        diffusion_op: torch.Tensor,
        track_kwargs: Dict[str, Any],
        layers: Optional[nn.ModuleList] = None,
    ) -> List[torch.Tensor]:
        """
        Process a single track (scalar or vector).
        Note that the input x is a (collated) scalar/vector from
        a pytorch-geometric Batch object, of shape (batch_num_nodes, 
        num_channels) for scalar features, or (batch_num_nodes * d, 
        1) for vector features, where d is the dimension of the vector.
        
        Note: InfoGain scales (if provided) are used as 'diffusion_scales' in get_Batch_Wjxs.
        """

        # Get batch index and features
        batch_index = batch.batch
        x = batch[track_kwargs['feature_key']]
        if track_name == 'vector':
            # Flatten the vector feature for multiplication by (block-diagonal) Q
            x = x.view(x.shape[0] * track_kwargs['vector_dim'], 1)  # (N*d_vector, 1)
        vector_norm_p = track_kwargs.get('vector_norm_p', 2)
        nnu.raise_if_nonfinite_tensor(x, name=f"{track_name} track input x")
        if diffusion_op is not None and isinstance(diffusion_op, torch.Tensor):
            if diffusion_op.is_sparse:
                nnu.raise_if_nonfinite_tensor(
                    diffusion_op._values(),
                    name=f"{track_name} track diffusion_op values",
                )
            else:
                nnu.raise_if_nonfinite_tensor(
                    diffusion_op,
                    name=f"{track_name} track diffusion_op",
                )

        if self.verbosity > 2:
            print(f"\nInput {track_name} features stats: min={torch.min(x).item():.4e}, max={torch.max(x).item():.4e}, has_nan={torch.isnan(x).any().item()}")

        # ---- Skip connection ----
        if track_kwargs['use_skip']:
            if 'graph' in self.task:

                # Vector track
                if track_name == 'vector':
                    # Reshape the flattened vector feature back to (N, d)
                    num_nodes, d = batch_index.shape[0], track_kwargs['vector_dim']
                    x_norm = torch.norm(
                        x.view(num_nodes, d), 
                        p=vector_norm_p, 
                        dim=1, 
                        keepdim=True
                    )  # shape (N, 1)
                    x_norm = x_norm.unsqueeze(-1)  # shape (N, 1, 1)

                    # Readout the norms of the original vector features
                    readout_x = self._pool(
                        x=x_norm,
                        batch_index=batch_index,
                        pooling_kwargs=track_kwargs['scattering_pooling_kwargs']
                    ) # shape (B, C, S, W) = (B, 1, S, 1)
                
                # Scalar track: read out the original features
                else:
                    readout_x = self._pool(
                        x=x.unsqueeze(-1), # shape (N, C, 1)
                        batch_index=batch_index,
                        pooling_kwargs=track_kwargs['scattering_pooling_kwargs']
                    ) # shape (B, C, S, W) = (B, C, 1, W)

                # Save the skip connection vector/scalar track readout
                track_outputs = [readout_x]
                if self.verbosity > 2:
                    print(f"readout_x stats: min={torch.min(readout_x).item():.4e}, max={torch.max(readout_x).item():.4e}, has_nan={torch.isnan(readout_x).any().item()}")
            
            # Single-graph tasks: save (unpooled) original features
            else:
                track_outputs = [x.detach().clone().unsqueeze(-1)]

        # No skip connection: initialize track_outputs as empty list
        else:
            track_outputs = []
        
        # ---- Loop through scattering layers ----
        for layer_i in range(track_kwargs['num_layers']):
            if self.verbosity > 1: 
                print(f"  layer_i: {layer_i}")

            # 1. (filter): apply diffusion wavelet filters
            # Use InfoGain scales if provided (diffusion_scales), else dyadic
            # At layer 0 (first-order scattering), x is the original features (N[*d], C)
            # At layer 1 (second-order scattering), x is the reshaped first-order scattering coefficients (N[*d], C*W) [or C*W' if filter-combine mode]
            Wjxs = get_Batch_Wjxs(
                x=x,
                P_sparse=diffusion_op,
                vector_dim=track_kwargs.get('vector_dim', None),
                **track_kwargs['diffusion_kwargs']
            )
            nnu.raise_if_nonfinite_tensor(Wjxs, name=f"{track_name} track Wjxs (layer {layer_i})")

            if self.verbosity > 2:
                print(f"\tWjxs stats: min={torch.min(Wjxs).item():.4e}, max={torch.max(Wjxs).item():.4e}, has_nan={torch.isnan(Wjxs).any().item()}")
            
            # Apply batch normalization to Wjxs (initialize if needed)
            if self.use_wjxs_batch_norm:
                # Check if we need to initialize batch norm for this layer
                bn_layers = self.vector_wjxs_bn_layers \
                    if (track_name == 'vector' and not self.ablate_vector_wavelet_batch_norm) \
                    else self.scalar_wjxs_bn_layers
                
                # Initialize if this layer doesn't have batch norm yet
                bn_idx = layer_i - 1  # scattering layers are 1,2,...
                if bn_layers is None or bn_idx >= len(bn_layers):
                    actual_channels = Wjxs.shape[1]
                    num_wavelets = Wjxs.shape[2]
                    vector_dim = self.vector_track_kwargs.get('vector_dim', 3)
                    self._init_wjxs_batch_norm_layers(
                        scalar_channels=(0 if track_name == 'vector' else actual_channels),
                        vector_channels=(actual_channels if track_name == 'vector' else 0),
                        num_wavelets=num_wavelets,
                        vector_dim=vector_dim,
                        device=Wjxs.device,
                        init_scalar=(track_name == 'scalar'),
                        init_vector=(track_name == 'vector'),
                    )
                
                # Apply batch normalization to Wjxs
                Wjxs = self._apply_wjxs_batch_norm(Wjxs, track_name, layer_i)
                nnu.raise_if_nonfinite_tensor(
                    Wjxs,
                    name=f"{track_name} track Wjxs post-bn (layer {layer_i})",
                )

            if (self.mode in ('filter-combine', 'cross-track')) \
            and (layer_i == 1):
                # In filter-combine mode, we approximate second-order scattering by keeping
                # only the highest-frequency wavelets (J_prime) in the second layer.
                # This is different from handcrafted_scattering mode where we strictly
                # apply lower-pass filters to all previous higher-pass filtrations: here, since we
                # learn new cross-filter combinations, the filtered features are not
                # strictly bandpass/sortable by frequency anymore.
                J_prime = track_kwargs.get('J_prime', 1)  # Get track-specific J_prime
                Wjxs = Wjxs[:, :, -J_prime:]  # Keep only last J_prime wavelets

            if self.verbosity > 1:
                print(f"\t(filter) Wjxs.shape (N[*d], C, W): {list(Wjxs.shape)}")
            # scalars shape: (batch_num_nodes, num_channels, num_wavelets)
            # vectors shape: (batch_num_nodes * d, num_channels, num_wavelets)
            
            # 2a. Handcrafted_scattering mode, layer == 1 (second-order scattering): 
            # the 'num_channels' (dim 1) is actually (num_channels * num_wavelets),
            # so we need to filter to only lower-frequency wavelets applied to
            # higher-frequency wavelets
            if (self.mode == 'handcrafted_scattering') \
            and (layer_i == 1):
                num_wavelets = Wjxs.shape[-1]
                orig_num_channels = Wjxs.shape[1] // num_wavelets
                wavelet_mask = torch.triu(
                    torch.ones(num_wavelets, num_wavelets),
                    diagonal=1
                ).bool()
                Wjxs = Wjxs.reshape(
                    Wjxs.shape[0], 
                    orig_num_channels, 
                    num_wavelets,
                    num_wavelets
                )
                Wjxs = Wjxs[:, :, wavelet_mask]
                # print(f"Wjxs.shape (2nd order scattering): {Wjxs.shape}")
                # shape: (batch_num_nodes, orig_num_channels, 
                # num_2nd_order_wavelets) [where num_2nd_order_wavelets
                # is num_wavelets * (num_wavelets - 1) / 2]

            # 2b. Filter-combine mode: cross-filter combinations step
            # learn new feature combinations across wavelet filtrations (within channels)
            elif (self.mode in ('filter-combine', 'cross-track')) and (Wjxs.shape[-1] > 1):
                Wjxs = layers[layer_i](Wjxs)

                # Print shape and stats if desired
                if self.verbosity > 1:
                    print(f"\t(combine-filters) Wjxs.shape (N[*d], C, W'): {list(Wjxs.shape)}")
                if self.verbosity > 2:
                    print(f"\tAfter filter combination stats: min={torch.min(Wjxs).item():.4e}, max={torch.max(Wjxs).item():.4e}, has_nan={torch.isnan(Wjxs).any().item()}")

            # 3. Apply nonlinearity (note vector track uses a norm-based nonlinearity, to preserve SO(d) equivariance)
            if track_name == 'vector':
                Wjxs = self.vector_track_nonlin_fn(Wjxs, p=vector_norm_p)
            else:
                Wjxs = track_kwargs['nonlin_fn'](
                    Wjxs,
                    **track_kwargs['nonlin_fn_kwargs']
                )
            nnu.raise_if_nonfinite_tensor(
                Wjxs,
                name=f"{track_name} track Wjxs post-nonlinearity (layer {layer_i})",
            )
            if self.verbosity > 2:
                print(f"\tAfter nonlinearity stats: min={torch.min(Wjxs).item():.4e}, max={torch.max(Wjxs).item():.4e}, has_nan={torch.isnan(Wjxs).any().item()}")
                # Check for all-zero outputs
                zero_mask = (torch.abs(Wjxs) < 1e-8)
                if zero_mask.all():
                    print(f"\tWARNING: All values are zero after nonlinearity!")

            # 4a. Graph-level tasks: readout/pool Wjxs across nodes for each 
            # channel in each graph and save to track_outputs list
            if 'graph' in self.task:
                if track_name == 'vector':

                    # Reshape from (num_nodes * d, num_channels, num_wavelets) to (num_nodes, d, num_channels, num_wavelets)
                    num_nodes = batch_index.shape[0]
                    d = self.base_module_kwargs.get('vector_feat_dim', 3)
                    if Wjxs.shape[0] == num_nodes * d:
                        Wjxs_reshaped = Wjxs.view(
                            num_nodes, 
                            d, 
                            Wjxs.shape[1], 
                            Wjxs.shape[2]
                        )

                    # Convert to (N, C_v", W) via p-norm over vector dim
                    Wjxs_norm = torch.norm(Wjxs_reshaped, p=vector_norm_p, dim=1)

                    # Pool the norms of the nonlinearly-processed vector features
                    readout_Wjxs = self._pool(
                        x=Wjxs_norm,
                        batch_index=batch_index,
                        pooling_kwargs=track_kwargs['scattering_pooling_kwargs']
                    ) # shape (B, C, S, W)
                else:
                    readout_Wjxs = self._pool(
                        x=Wjxs,
                        batch_index=batch_index,
                        pooling_kwargs=track_kwargs['scattering_pooling_kwargs']
                    ) # shape (B, C, S, W)

                # Save the readout of the layer
                track_outputs.append(readout_Wjxs)

                # Print stats if desired (S is num. of pooling stats/features)
                if self.verbosity > 1:
                    print(f"\t\t(readout) readout_Wjxs.shape (B, C, S, W'): {list(readout_Wjxs.shape)}")
                if self.verbosity > 2:
                    print(f"\t\tAfter readout stats: min={torch.min(readout_Wjxs).item():.4e}, max={torch.max(readout_Wjxs).item():.4e}, has_nan={torch.isnan(readout_Wjxs).any().item()}")

            # 4b. Single-graph tasks: save (unpooled) Wjxs
            else:
                track_outputs.append(Wjxs.detach().clone())

            # 5. Before next layer: reshape Wjxs by flattening across channels and wavelets and set as x for next layer
            # In handcrafted_scattering mode, this means all wavelets will be re-applied to every channels' first-order scattering coefficients 
            # from every wavelet (but then filtered down to lower-frequency wavelets)
            if (layer_i < track_kwargs['num_layers'] - 1):
                x = Wjxs.reshape(Wjxs.shape[0], -1)

                # Print shape if desired
                if self.verbosity > 1:
                    print(f"\t(reshape) x.shape (N[*d], C*W'): {list(x.shape)}")
            
        return track_outputs
        

    def forward(self, batch: Batch | Data) -> Dict[str, torch.Tensor]:
        """Forward pass through the VDW model."""
        # Get diffusion operators
        P = None
        if not self.ablate_scalar_track:
            P = getattr(batch, self.scalar_track_kwargs['diffusion_op_key'])
        
        Q = None
        if not self.ablate_vector_track:
            Q = getattr(batch, self.vector_track_kwargs['diffusion_op_key'])
        
        if self.verbosity > 2:
            if P is not None:
                print(f"P stats: has_nan={torch.isnan(P).any().item()}")
            if Q is not None:
                print(f"Q stats: has_nan={torch.isnan(Q).any().item()}")
        
        if self.mode == 'cross-track':
            return self._forward_cross_track(batch, P, Q)
        
        if self.stream_parallelize_tracks:
            # Create streams for parallel processing
            scalar_stream = torch.cuda.Stream()
            vector_stream = torch.cuda.Stream()
            
            # Process scalar track in one stream
            with torch.cuda.stream(scalar_stream):
                if self.verbosity > 1:
                    print(f"\nscalar track forward")
                scalar_outputs = self._track_forward(
                    track_name='scalar',
                    batch=batch,
                    diffusion_op=P,
                    track_kwargs=self.scalar_track_kwargs,
                    layers=self.scalar_track_layers,
                )
            
            vector_outputs = []
            if not self.ablate_vector_track:
                # Process vector track in another stream
                with torch.cuda.stream(vector_stream):
                    if self.verbosity > 1:
                        print(f"\nvector track forward")
                    vector_outputs = self._track_forward(
                        track_name='vector',
                        batch=batch,
                        diffusion_op=Q,
                        track_kwargs=self.vector_track_kwargs,
                        layers=self.vector_track_layers,
                    )
            
            # Synchronize both streams
            torch.cuda.synchronize()
        else:
            # Process tracks sequentially
            scalar_outputs = self._track_forward(
                track_name='scalar',
                batch=batch,
                diffusion_op=P,
                track_kwargs=self.scalar_track_kwargs,
                layers=self.scalar_track_layers,
            )
            if self.verbosity > 1:
                scalar_track_shapes = [out.shape for out in scalar_outputs]
                for i, shape in enumerate(scalar_track_shapes):
                    print(f"scalar track {i} shape: {shape}")

            vector_outputs = []
            if not self.ablate_vector_track:
                vector_outputs = self._track_forward(
                    track_name='vector',
                    batch=batch,
                    diffusion_op=Q,
                    track_kwargs=self.vector_track_kwargs,
                    layers=self.vector_track_layers,
                )
                if self.verbosity > 1:
                    vector_track_shapes = [out.shape for out in vector_outputs]
                    for i, shape in enumerate(vector_track_shapes):
                        print(f"vector track {i} shape: {shape}")

        # First flatten each output to 2D (preserving batch dimension)
        scalar_outputs = [out.reshape(out.shape[0], -1) for out in scalar_outputs]
        scalar_outputs = torch.cat(scalar_outputs, dim=-1)

        device = scalar_outputs.device

        if not self.ablate_vector_track and vector_outputs:
            vector_outputs = [out.reshape(out.shape[0], -1) for out in vector_outputs]
            vector_outputs = torch.cat(vector_outputs, dim=-1).to(device)
            combined = torch.cat((scalar_outputs.to(device), vector_outputs), dim=1)
        else:
            combined = scalar_outputs.to(device)
        
        nnu.raise_if_nonfinite_tensor(combined, name="combined scalar/vector features pre-mlp")
        if self.verbosity > 2:
            print(f"Combined stats: min={torch.min(combined).item():.4e}, max={torch.max(combined).item():.4e}, has_nan={torch.isnan(combined).any().item()}")
            
        if self.verbosity > 0:
            print(f"[DEBUG] forward: scalar_track_layers devices:")
            for i, layer in enumerate(self.scalar_track_layers):
                for name, param in layer.named_parameters():
                    print(f"  scalar layer {i} param {name} device: {param.device}")
            print(f"[DEBUG] forward: vector_track_layers devices:")
            for i, layer in enumerate(self.vector_track_layers):
                for name, param in layer.named_parameters():
                    print(f"  vector layer {i} param {name} device: {param.device}")
        
        # Initialize MLP on first forward pass if not already initialized
        if (self.mode in ('filter-combine', 'cross-track')):
            # Initialize MLP on first forward pass if not already initialized
            if (not self._mlp_initialized):
                # Get input dimension from reshaped tensor
                mlp_input_dim = combined.shape[-1]
                
                # Create MLP using VanillaNN
                self.mlp = VanillaNN(
                    input_dim=mlp_input_dim,
                    **self.mlp_kwargs,
                    base_module_kwargs=self.base_module_kwargs
                )
                
                # Move MLP to the same device as the input tensor
                self.mlp = self.mlp.to(combined.device)
                self._mlp_initialized = True

                # MLP is now fully initialized
                if self.verbosity > 0:
                    print(f"[VDW] MLP initialization complete")

                # Debug: print device of MLP parameters after moving
                if self.verbosity > 0:
                    print(f"[DEBUG] forward: MLP parameters after .to({combined.device}):")
                    for name, param in self.mlp.named_parameters():
                        print(f"  mlp param {name} device: {param.device}")
            
            # Pass (concatenated, batch-normalized) hidden scalar and vector features 
            # through MLP, which returns a dict of outputs
            if self.use_input_normalization:
                # Initialize batch normalization on first forward pass
                if not self._input_batch_norm_initialized:
                    self.input_batch_norm = nn.BatchNorm1d(
                        combined.shape[1],
                        eps=self.normalization_eps,
                        momentum=0.1,
                        affine=True
                    ).to(combined.device)
                    self._input_batch_norm_initialized = True
                    if self.verbosity > 2:
                        print(f"Initialized input batch normalization with eps={self.normalization_eps}")
                
                if self.verbosity > 2:
                    print(f"Before batch norm stats: min={torch.min(combined).item():.4e}, max={torch.max(combined).item():.4e}, has_nan={torch.isnan(combined).any().item()}")
                
                # Apply batch normalization
                combined = self.input_batch_norm(combined)
                
                if self.verbosity > 2:
                    print(f"After batch norm stats: min={torch.min(combined).item():.4e}, max={torch.max(combined).item():.4e}, has_nan={torch.isnan(combined).any().item()}")
            
            output_dict = self.mlp(combined)
            if isinstance(output_dict, dict) and ("preds" in output_dict):
                nnu.raise_if_nonfinite_tensor(output_dict["preds"], name="mlp output preds")
            
            if self.verbosity > 2:
                print(f"MLP output stats: min={torch.min(output_dict['preds']).item():.4e}, max={torch.max(output_dict['preds']).item():.4e}, has_nan={torch.isnan(output_dict['preds']).any().item()}")
            
            return output_dict
        else:
            return combined
        

    # ------------------------------------------------------------------
    # Scalar wavelet recombination
    # ------------------------------------------------------------------
    class ScalarWaveletRecombiner(nn.Module):
        """
        Recombines scalar features across the wavelet dimension using 2-layer MLPs.
        Each output channel is computed by a separate MLP that takes the input
        wavelet features and produces a single scalar output.
        """
        def __init__(
            self, 
            num_wavelets: int, 
            num_output_channels: int, 
            hidden_dim: int, 
            nonlin_fn: Callable = F.silu,
        ):
            """
            Args:
                num_wavelets: Number of input wavelet features (W)
                num_output_channels: Number of output channels to create (W')
                hidden_dim: Hidden dimension for the MLPs
                nonlin_fn: Nonlinearity function to use between layers (default: SiLU)
            """
            super().__init__()
            self.num_wavelets = num_wavelets
            self.num_output_channels = num_output_channels
            self.nonlin_fn = nonlin_fn
            
            # Create 2-layer MLPs for each output channel
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_wavelets, hidden_dim),
                    FUNCTION_TO_MODULE_MAP.get(self.nonlin_fn, nn.SiLU)(),
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(num_output_channels)
            ])
        
        def forward(self, scalar_inputs: torch.Tensor) -> torch.Tensor:
            """
            Args:
                scalar_inputs: Tensor of shape [N, C, W] - N nodes, C channels, W wavelets
            Returns:
                output: Tensor of shape [N, C, W'] - N nodes, C channels, W' recombined wavelets
            """
            N, C, W = scalar_inputs.shape
            assert W == self.num_wavelets, f"Expected {self.num_wavelets} wavelets, got {W}"
            x = scalar_inputs.reshape(N * C, W)
            outputs = []
            for mlp in self.mlps:
                out = mlp(x)  # [N*C, 1]
                outputs.append(out)
            out_cat = torch.cat(outputs, dim=1)  # [N*C, W']
            return out_cat.view(N, C, self.num_output_channels)


    # ------------------------------------------------------------------
    # Gated vector wavelet recombination
    # ------------------------------------------------------------------
    class GatedVectorWaveletRecombiner(nn.Module):
        """
        Recombines vector norms across the wavelet dimension using gating and reweighting.
        Since norms are rotationally invariant, this operation is equivariant.
        """
        def __init__(
            self, 
            num_wavelets: int, 
            num_output_channels: int, 
            hidden_dim: int, 
            gate_hidden_dim: int = None,
        ):
            """
            Args:
                num_wavelets: Number of input wavelet features (W)
                num_output_channels: Number of output channels to create (W')
                hidden_dim: Hidden dimension for the weight MLPs
                gate_hidden_dim: Hidden dimension for the gate MLP (default = hidden_dim)
            """
            super().__init__()
            self.num_wavelets = num_wavelets
            self.num_output_channels = num_output_channels
            gate_hidden_dim = gate_hidden_dim or hidden_dim

            # MLPs to compute scalar weights for combining wavelets into each output channel
            self.weight_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_wavelets, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, num_wavelets)
                )
                for _ in range(num_output_channels)
            ])

            # Gate MLP for modulating each input wavelet based on its norm
            self.gate_mlp = nn.Sequential(
                nn.Linear(num_wavelets, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, num_wavelets),
                nn.Sigmoid()  # output gate ∈ (0, 1)
            )

        def forward(self, norm_inputs: torch.Tensor) -> torch.Tensor:
            """
            Args:
                norm_inputs: Tensor of shape [N, C, W] - N nodes, C channels, W wavelets (already norms)
            Returns:
                output: Tensor of shape [N, C, W'] - N nodes, C channels, W' recombined wavelets
            """
            N, C, W = norm_inputs.shape
            assert W == self.num_wavelets, f"Expected {self.num_wavelets} wavelets, got {W}"
            
            # Step 1: Gating each input wavelet based on its norm
            gates = self.gate_mlp(norm_inputs)  # [N, C, W]
            gated_norms = gates * norm_inputs  # [N, C, W]
            
            # Step 2: For each output channel, compute weighted sum of gated norms
            outputs = []
            for mlp in self.weight_mlps:
                weights = mlp(norm_inputs)  # [N, C, W]
                # Weighted sum across wavelet dimension: [N, C, W] -> [N, C, 1]
                mixed = (weights * gated_norms).sum(dim=2, keepdim=True)  # [N, C, 1]
                outputs.append(mixed)
            
            # Step 3: Concatenate outputs along wavelet dimension: [N, C, W']
            return torch.cat(outputs, dim=2)  # [N, C, W']
    

    # ------------------------------------------------------------------
    # Cross-track mode forward pass
    # ------------------------------------------------------------------
    def _forward_cross_track(
        self,
        batch: Batch,
        P: torch.Tensor,
        Q: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward logic for *cross-track* mode. Supports vector track ablation
        (by substituting vector track tensors with empty tensors where needed and passing vector coordinates as scalar features instead - see _LazyAttributeLoadingDataset class in training/prep_dataset.py).

        This mode learns interactions between scalar and vector tracks at every layer (skip/zeroth-, first-, and second-order scattering layers). The overall steps for each layer are:

        1. (Optional: Layers 1 and 2) Within-track cross-filter combinations (reuses  ``scalar_track_layers`` / ``vector_track_layers`` also used in 'filter-combine' mode)
        2. Non-linearities (vector activation is norm-based)
        3. Cross-track mixing producing ``n_cross_track_combos`` channels (strictly within wavelets, if step 1 is not used)
        4. Cross-filter mixing producing ``n_cross_filter_combos`` wavelet
           channels
        5. Graph-level readout/pooling
        
        The resulting tensors are collected across layers and fed to an MLP head (same as in 'filter-combine' mode).
        """

        if (self.cross_track_layers is None) \
        or (self.cross_filter_layers is None):
            raise RuntimeError("Cross-track layers are not initialized. Check mode and initialization code.")

        batch_index = batch.batch  # (N,)

        # ------------------------------------------------------------------
        # Convenience variables and initial inputs
        # ------------------------------------------------------------------
        num_nodes = batch_index.shape[0]

        if not self.ablate_scalar_track:
            scalar_x = batch[self.scalar_track_kwargs['feature_key']]  # (N, C_s)
        else:
            scalar_x = None  # placeholder, not used

        if self.ablate_vector_track:
            # vector_flat = None
            d = 0
            vector_norm_p = 2
        else:
            vector_x = batch[self.vector_track_kwargs['feature_key']]
            d = self.vector_track_kwargs.get('vector_dim', 3)
            vector_x = vector_x.view(vector_x.shape[0] * d, 1)  # (N * d, 1)
            vector_norm_p = self.vector_track_kwargs.get('vector_norm_p', 2)

        # We might not want to combine within-track filter combinations, 
        # since we are probably interested in interactions between scalar 
        # and vector tracks, and want to learn their interactions across
        # tracks but *within* filters.
        within_track_combine = self.cross_track_kwargs.get('within_track_combine', False)

        # Cross-layer outputs to aggregate
        layers_pools: List[torch.Tensor] = []

        # Total layers = skip connection + scattering layers
        total_layers = self.scalar_track_kwargs['num_layers'] + 1

        for layer_i in range(total_layers):

            # --------------------------------------------------------------
            # 1.  Obtain per-track features for current layer
            # --------------------------------------------------------------
            # Layer 0: zeroth-order scattering layer
            if layer_i == 0:
                # ----------------------------------------------------------
                # Skip connection (zeroth-order scattering): treat inputs as
                # a single "wavelet" (W = 1)
                # ----------------------------------------------------------
                if not self.ablate_scalar_track:
                    scalar_Wjxs = scalar_x.unsqueeze(-1)  # (N, C_s, 1)
                else:
                    # Create empty tensor with zero channels so downstream
                    # dimensions match (N, 0, 1)
                    scalar_Wjxs = batch.batch.new_empty(num_nodes, 0, 1, dtype=scalar_x.dtype if scalar_x is not None else torch.float32)

                # ---------------- Vector-derived features ----------------
                if not self.ablate_vector_track:
                    # Compute vector norms so that the scalar and vector
                    # tracks have compatible channel counts for cross-track
                    # mixing later on.
                    vec_coords = vector_x.view(num_nodes, d)  # (N, d)
                    vector_norm = torch.norm(
                        vec_coords,
                        p=self.vector_track_kwargs.get('vector_norm_p', 2),
                        dim=1,
                        keepdim=True,
                    )  # (N, 1)

                    # Repeat the norm to match scalar channel count (or 1 if
                    # scalar track ablated)
                    repeat_ch = scalar_Wjxs.shape[1] if not self.ablate_scalar_track else 1
                    vector_norms = vector_norm.unsqueeze(-1).expand(-1, repeat_ch, -1)  # (N, C_rep, 1)

                    # Cosine similarity: for layer-0 we use 1s as placeholder
                    cos_sim = torch.ones_like(vector_norms)  # (N, C_rep, 1)

                    vector_features_aug = torch.cat([vector_norms, cos_sim], dim=1)  # (N, 2*C_rep, 1)
                else:
                    # Vector track ablated – use empty tensor with correct dims
                    device_ = scalar_Wjxs.device if scalar_Wjxs is not None else batch.batch.device
                    vector_features_aug = torch.empty(num_nodes, 0, 1, device=device_)

            # First- and second-order scattering layers
            else:
                # ----------------------
                # 1a. Scalar track
                # ----------------------
                if not self.ablate_scalar_track:
                    # --------------------------------------------------
                    # Apply optional scalar embedding PRIOR to the first
                    # scattering layer only (layer_i == 1).
                    # --------------------------------------------------
                    scalar_input = scalar_x
                    if (self.scalar_feat_embedding is not None) and (layer_i == 1):
                        # Ensure module is on correct device before use
                        self.scalar_feat_embedding = self.scalar_feat_embedding.to(scalar_input.device)
                        scalar_input = self.scalar_feat_embedding(scalar_input)

                    scalar_Wjxs = self._track_scattering(
                        scalar_input,
                        P,
                        track_kwargs=self.scalar_track_kwargs,
                        is_vector=False,
                        within_track_combine=within_track_combine,
                        layer_i=layer_i,
                    )
                else:
                    # Create empty tensor with zero channels matching wavelets of vector track later
                    scalar_Wjxs = None  # will set after vector track processed

                # ---------------------------
                # 1b. Vector track
                # ---------------------------
                if not self.ablate_vector_track:
                    vector_Wjxs = self._track_scattering(
                        vector_x,
                        Q,
                        track_kwargs=self.vector_track_kwargs,
                        is_vector=True,
                        within_track_combine=within_track_combine,
                        vector_dim=d,
                        layer_i=layer_i,
                    ) # shape (N*d, C_v, W)

                    # Debug: print vector_Wjxs shape
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: vector_Wjxs.shape={vector_Wjxs.shape}")

                    # 1b(i). Compute vector norms
                    # -----------------------------
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: vector_Wjxs.shape: {vector_Wjxs.shape}")
                        print(f"[DEBUG] Layer {layer_i}: num_nodes: {num_nodes}, d: {d}")
                        print(f"[DEBUG] Layer {layer_i}: Expected first dim for reshape: {num_nodes * d}")
                    vector_Wjxs_reshaped = vector_Wjxs.view(
                        num_nodes, d, vector_Wjxs.shape[1], vector_Wjxs.shape[2]
                    )  # (N, d, C_v, W)
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: vector_Wjxs_reshaped.shape: {vector_Wjxs_reshaped.shape}")
                    vector_norms = torch.norm(
                        vector_Wjxs_reshaped, 
                        p=self.vector_track_kwargs.get('vector_norm_p', 2), 
                        dim=1
                    )  # (N, C_v, W)

                    # 1b(ii). Compute cosine similarities ('angles')
                    # between original and transformed vector features
                    # ------------------------------------------------
                    pos = batch[self.vector_track_kwargs['feature_key']]  # (N, d)
                    pos_expanded = pos.unsqueeze(2).unsqueeze(3)  # (N, d, 1, 1)
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: pos.shape: {pos.shape}")
                        print(f"[DEBUG] Layer {layer_i}: pos_expanded.shape: {pos.unsqueeze(2).unsqueeze(3).shape}")
                    dot = (vector_Wjxs_reshaped * pos_expanded).sum(dim=1)  # (N, C, W)
                    vec_norms = vector_Wjxs_reshaped.norm(dim=1)  # (N, C, W)
                    pos_norms = pos.norm(dim=1, keepdim=True).unsqueeze(2)  # (N, 1, 1)
                    cos_sim = dot / (vec_norms * pos_norms + 1e-8)  # (N, C, W)
                    if self.verbosity > 1:
                        print(f'[DEBUG] Layer {layer_i} cat norms and angles: vector_norms.shape={vector_norms.shape}, cos_sim.shape={cos_sim.shape}')
                    vector_features_aug = torch.cat(
                        [vector_norms, cos_sim], 
                        dim=1
                    )  # (N, 2*C, W) [now two features per wavelet]

                    # If scalar track is ablated, create dummy zero-channel tensor with matching wavelets
                    if self.ablate_scalar_track:
                        scalar_Wjxs = torch.empty(
                            num_nodes,
                            0,
                            vector_Wjxs.shape[2],
                            device=vector_Wjxs.device,
                        )

                    # ------------------------------------------------------------
                    # DEPRECATED:Batch-norm for scattering layers only (layer_i >= 1)
                    # ------------------------------------------------------------
                    '''
                    if layer_i > 0:
                        N, C2, W = vector_features_aug.shape  # (N, 2*C_v, W)
                        flat = vector_features_aug.view(N, C2 * W)

                        # Ensure index 0 is a placeholder Identity so we have
                        # no trainable params associated with the skip layer.
                        if len(self.vector_features_bn_layers) == 0:
                            self.vector_features_bn_layers.append(nn.Identity())

                        # Lazily create BN layers up to current index
                        while len(self.vector_features_bn_layers) <= layer_i:
                            bn = nn.BatchNorm1d(
                                num_features=C2 * W,
                                eps=1e-5,
                                momentum=0.1,
                                affine=True,
                            ).to(vector_features_aug.device)
                            self.vector_features_bn_layers.append(bn)

                        flat_norm = self.vector_features_bn_layers[layer_i](flat)
                        vector_features_aug = flat_norm.view(N, C2, W)
                    '''

                    # ------------------------------------------------
                    # Prepare inputs (scalar_x / vector_x) for next layer
                    # ------------------------------------------------
                    if layer_i < total_layers - 1:
                        if not self.ablate_scalar_track:
                            scalar_x = scalar_Wjxs.reshape(scalar_Wjxs.shape[0], -1)
                        vector_x = vector_Wjxs.reshape(vector_Wjxs.shape[0], -1)
                else:
                    # Create empty tensor with zero channels
                    vector_features_aug = scalar_Wjxs.new_empty(num_nodes, 0, scalar_Wjxs.shape[2])

                    # Prepare scalar_x for next layer if needed (vector_x stays None)
                    if layer_i < total_layers - 1:
                        if not self.ablate_scalar_track:
                            scalar_x = scalar_Wjxs.reshape(scalar_Wjxs.shape[0], -1)

            # Ensure scalar and vector have matching wavelet count
            if (not self.ablate_scalar_track) and (not self.ablate_vector_track):
                if scalar_Wjxs.shape[2] != vector_norms.shape[2]:
                    raise ValueError(
                        f"Scalar and vector tracks have different numbers of wavelets "
                        f"(scalar={scalar_Wjxs.shape[2]}, vector={vector_norms.shape[2]}) "
                        f"in layer {layer_i}. Ensure diffusion parameters match."
                    )
            W = scalar_Wjxs.shape[2]

            # --------------------------------------------------------------
            # 2.  Wavelet recombination (if enabled)
            # --------------------------------------------------------------
            if self.cross_track_kwargs.get('use_wavelet_recombination', True):
                # Lazy initialize recombination layers if needed
                device = scalar_Wjxs.device
                self._init_recombination_layers(layer_i, W, device)
                if self.verbosity > 1:
                    print(f"[DEBUG] Layer {layer_i}: Recombination enabled. scalar_recombiners[{layer_i}] = {self.scalar_recombiners[layer_i] if self.scalar_recombiners else None}")
                    print(f"[DEBUG] Layer {layer_i}: Recombination enabled. vector_recombiners[{layer_i}] = {self.vector_recombiners[layer_i] if self.vector_recombiners else None}")
                
                # Apply scalar recombination
                if (not self.ablate_scalar_track) and (self.scalar_recombiners[layer_i] is not None):
                    scalar_feats = self.scalar_recombiners[layer_i](scalar_Wjxs)
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: Applied scalar recombination: {scalar_Wjxs.shape} -> {scalar_feats.shape}")
                else:
                    scalar_feats = scalar_Wjxs
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: No scalar recombination applied: {scalar_feats.shape}")
                
                # Apply vector recombination to norms
                if (self.vector_recombiners[layer_i] is not None) \
                and (not self.ablate_vector_track):
                    vector_feats = self.vector_recombiners[layer_i](vector_features_aug)
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: Applied vector recombination: {vector_features_aug.shape} -> {vector_feats.shape}")
                else:
                    vector_feats = vector_features_aug
                    if self.verbosity > 1:
                        print(f"[DEBUG] Layer {layer_i}: No vector recombination applied: {vector_feats.shape}")
                
                # Debug: print shapes after recombination
                if self.verbosity > 1:
                    print(f"[DEBUG] Layer {layer_i}: After recombination - scalar_feats.shape={scalar_feats.shape}, vector_feats.shape={vector_feats.shape}")
                
                # Update W to reflect new number of wavelets
                W = scalar_feats.shape[2]
            else:
                if self.verbosity > 1:
                    print(f"[DEBUG] Layer {layer_i}: Recombination disabled")
                # No recombination - use original features
                scalar_feats = scalar_Wjxs
                vector_feats = vector_features_aug

            # --------------------------------------------------------------
            # 3.  Cross-track channel mixing (per wavelet) + nonlinearity
            # --------------------------------------------------------------
            # Debug: print tensor shapes before concatenation
            if self.verbosity > 1:
                print(f"[DEBUG] Layer {layer_i}: scalar_feats.shape={scalar_feats.shape}, vector_feats.shape={vector_feats.shape}")
                print(f"[DEBUG] Layer {layer_i}: scalar_feats.device={scalar_feats.device}, vector_feats.device={vector_feats.device}")
            
            # Handle vector track ablation: if vector_feats has 0 channels, just use scalar_feats
            # if self.ablate_vector_track and vector_feats.shape[1] == 0:
            #     combined = scalar_feats  # only scalar
            #     if self.verbosity > 1:
            #         print(f"[DEBUG] Layer {layer_i}: Vector track ablated, using only scalar features: {combined.shape}")
            # elif self.ablate_scalar_track and scalar_feats.shape[1] == 0:
            #     combined = vector_feats  # only vector
            #     if self.verbosity > 1:
            #         print(f"[DEBUG] Layer {layer_i}: Scalar track ablated, using only vector features: {combined.shape}")

            # Only do cross-track step if both tracks are not ablated
            if not self.ablate_vector_track and not self.ablate_scalar_track:
                if self.verbosity > 1:
                    print(f'[DEBUG] Layer {layer_i}, cat scalar and vector features: scalar_feats.shape={scalar_feats.shape}, vector_feats.shape={vector_feats.shape}')
                combined = torch.cat([scalar_feats, vector_feats], dim=1)  # (N, C_tot, W)

                # Reorder to (N, W, C_tot) so Linear acts on the last dim (across track channels, within wavelets)
                combined = combined.permute(0, 2, 1)  # (N, W, C_tot)
                ct_layer = self.cross_track_layers[layer_i]
                ct_out = ct_layer(combined)  # (N, W, C_cross)

                # --- [Optional] LayerNorm (along channel dimension) ---
                # Lazily create the LayerNorm to match the *channel* dimension
                # if isinstance(self.cross_track_norm_layers[layer_i], nn.Identity):
                #     C_cross = ct_out_sw.shape[-1]  # channels
                #     self.cross_track_norm_layers[layer_i] = nn.LayerNorm(C_cross).to(ct_out_sw.device)

                # # Apply LayerNorm while channels are in the last dimension
                # ct_out_sw = self.cross_track_norm_layers[layer_i](ct_out_sw)  # (N, W, C_cross)

                # --- Nonlinearity ---
                # NOTE: after cross-track mixing, the scalar nonlinearity
                # will break equivariance past this point
                ct_out = self.scalar_track_kwargs['nonlin_fn'](
                    ct_out,
                    **self.scalar_track_kwargs.get('nonlin_fn_kwargs', {})
                )
                # Restore (N, C_cross, W) ordering for subsequent steps
                ct_out = ct_out.permute(0, 2, 1)  # (N, C_cross, W)

            # --------------------------------------------------------------
            # 4.  Cross-filter mixing across wavelets (for current layer)
            # --------------------------------------------------------------
            cf_layer = self.cross_filter_layers[layer_i]
            cf_out = cf_layer(ct_out)   # operates on last dim (wavelets) -> (N, C_cross, C_filter)

            # --------------------------------------------------------------
            # 5.  Feature pooling (for current layer)
            # --------------------------------------------------------------
            if 'graph' in self.task:
                pool_mode = 'node'
            elif 'node' in self.task:
                pool_mode = 'channel'
            else:
                raise NotImplementedError(f"Pooling/readout not implemented for task {self.task} (did you include 'graph' or 'node' in the task name?)")

            layer_pools = self._pool(
                mode=pool_mode,
                x=cf_out,
                batch_index=batch_index,
                pooling_kwargs=self.scalar_track_kwargs['scattering_pooling_kwargs'],
            )  # (B, C_cross, S, C_filter) [pool_mode='node'] 
               # (N, C_cross*C_filter)     [pool_mode='channel']
            layers_pools.append(layer_pools)

        # ------------------------------------------------------------------
        # 6. Readout (MLP)
        # (can work for both graph-level and node-level tasks)
        # ------------------------------------------------------------------
        # Flatten each layer pooling to 2-D: (B, *)
        layers_pools = [r.reshape(r.shape[0], -1) for r in layers_pools]
        combined_feats = torch.cat(layers_pools, dim=-1)

        # Optional input batch-norm
        if self.use_input_normalization:
            if not self._input_batch_norm_initialized:
                self.input_batch_norm = nn.BatchNorm1d(
                    combined_feats.shape[1],
                    eps=self.normalization_eps,
                    momentum=0.1,
                    affine=True,
                ).to(combined_feats.device)
                self._input_batch_norm_initialized = True
            combined_feats = self.input_batch_norm(combined_feats)

        # Lazy-initialize the MLP if necessary
        if not self._mlp_initialized:
            self.mlp = VanillaNN(
                input_dim=combined_feats.shape[1],
                **self.mlp_kwargs,
                base_module_kwargs=self.base_module_kwargs,
            ).to(combined_feats.device)
            self._mlp_initialized = True
            # MLP is now fully initialized
            if self.verbosity > 0:
                print(f"[VDW] MLP initialization complete (cross-track)")

        # Feed to MLP head
        output_dict = self.mlp(combined_feats)
        return output_dict

    # ------------------------------------------------------------------
    # Helper to apply per-track scattering + non-linearity
    # ------------------------------------------------------------------
    def _track_scattering(
        self,
        x_in: torch.Tensor,
        P_sparse: torch.Tensor,
        *, # keyword-only arguments (prevents positional arguments)
        track_kwargs: dict,
        is_vector: bool,
        within_track_combine: bool,
        layer_i: int,
        vector_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply wavelet filters, second-order frequency rule, optional batch
        normalization, optional filter mixing, and non-linearity for either
        the scalar or vector track.
        """
        # 1. Wavelet transform
        Wjxs = get_Batch_Wjxs(
            x=x_in,
            P_sparse=P_sparse,
            vector_dim=vector_dim,
            **track_kwargs['diffusion_kwargs'],
        )
        
        # Note: Batch normalization will be applied after all operations that might change channel count

        # 2. Second-order scattering (layer_i == 2) frequency handling
        if layer_i == 2:
            if within_track_combine:
                # Keep highest-frequency J′ wavelets
                Jp = track_kwargs.get('J_prime', 1)
                Wjxs = Wjxs[:, :, -Jp:]
            else:
                # Keep upper-triangular (high->low) combinations
                nW = Wjxs.shape[-1]
                C0 = Wjxs.shape[1] // nW
                Wjxs = Wjxs.view(Wjxs.shape[0], C0, nW, nW)
                mask = torch.triu(
                    torch.ones(nW, nW, device=Wjxs.device, dtype=torch.bool),
                    diagonal=1,
                )
                Wjxs = Wjxs[:, :, mask]

        # 3. [Optional] Apply batch normalization to SCALAR Wjxs immediately after 
        # scattering and frequency handling, but before any within-track
        # filter mixing.
        if self.use_wjxs_batch_norm and not is_vector:
            # Select per-track BN ModuleList dedicated to lazy BN instances
            # bn_layers = (
            #     self.vector_wjxs_lazy_bn_layers
            #     if is_vector else self.scalar_wjxs_lazy_bn_layers
            # )
            bn_layers = self.scalar_wjxs_lazy_bn_layers

            if bn_layers is not None:
                # Determine index in the list (skip-connection layer = 0, so scatter 1 -> idx 0)
                bn_idx = layer_i - 1

                # Resolve BN hyperparameters with sensible defaults
                eps = (self.cross_track_kwargs.get('batch_norm_eps', 1e-5)
                       if self.cross_track_kwargs is not None else 1e-5)
                momentum = (self.cross_track_kwargs.get('batch_norm_momentum', 0.1)
                            if self.cross_track_kwargs is not None else 0.1)
                affine = True  # allow learnable γ/β by default

                # Lazily create (and register) the LazyBatchNorm1d layer
                bn_layer = self._ensure_lazy_bn1d_layer(
                    module_list=bn_layers,
                    target_index=bn_idx,
                    device=Wjxs.device,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                )

                # Flatten (C, W) dims so each channel–wavelet pair has its own stats
                orig_shape = Wjxs.shape  # (N[*d], C, W)
                W_flat = Wjxs.view(orig_shape[0], -1)  # (N[*d], C*W)
                W_flat = bn_layer(W_flat)
                Wjxs = W_flat.view(orig_shape) # (N[*d], C, W)

        # 4. Optional within-track filter mixing
        # This preserves SO(d) equivariance if vector_track_layers have bias=False 
        # For V \in R^{dxW}, V' = VM, and RV' = R(VM) = (RV)M.
        # This commutes with rotation because coordinates aren't mixed: M is a 
        # scalar (rotation-invariant) operator that acts only on the scattering
        # coefficients.
        if within_track_combine and Wjxs.shape[-1] > 1:
            # Compute track layer index from layer_i (0-based index for scattering layers)
            track_layer_idx = layer_i - 1
            
            # Check that the required layer exists in the ModuleList
            track_layers = self.vector_track_layers \
                if is_vector else self.scalar_track_layers
            if track_layer_idx < len(track_layers):
                layer = track_layers[track_layer_idx]
                Wjxs = layer(Wjxs) # (N[*d], C, W')
            else:
                track_name = "vector" if is_vector else "scalar"
                raise RuntimeError(
                    f"Within-track combine is enabled but {track_name}_track_layers "
                    f"does not have layer {track_layer_idx}. "
                    f"Expected {len(track_layers)} layers but trying to access layer {track_layer_idx}. "
                    f"This may indicate a mismatch between num_layers configuration and layer initialization."
                )

        # 5. Non-linearity
        if is_vector:
            Wjxs = self.vector_track_nonlin_fn(
                Wjxs, 
                p=track_kwargs.get('vector_norm_p', 2)
            )
        else:
            Wjxs = track_kwargs['nonlin_fn'](
                Wjxs, **track_kwargs['nonlin_fn_kwargs']
            )

        return Wjxs  # (N[*d], C, W)
   
    # ------------------------------------------------------------------
    # Helper – lazy initialization for per-layer LazyBatchNorm1d
    # ------------------------------------------------------------------
    def _ensure_lazy_bn1d_layer(
        self,
        *,
        module_list: nn.ModuleList,
        target_index: int,
        device: torch.device,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> nn.LazyBatchNorm1d:
        """
        Ensure a `nn.LazyBatchNorm1d` exists at `module_list[target_index]`.
        Creates and registers it if missing, using provided hyperparameters.
        """
        # Grow the list up to target_index, appending initialized BN modules
        while len(module_list) <= target_index:
            bn_layer = nn.LazyBatchNorm1d(
                eps=eps,
                momentum=momentum,
                affine=affine,
            ).to(device)
            module_list.append(bn_layer)
        return module_list[target_index]

    # ------------------------------------------------------------------
    # Helper – get gate initial value
    # ------------------------------------------------------------------
    def _get_gate_init_value(self, gate_name: str) -> float:
        """Return a sensible initial value for the vector gate parameter."""
        return GATE_INIT_MAP.get(gate_name, 0.10)
    
    # ------------------------------------------------------------------
    # Helper – cross-track initialization
    # ------------------------------------------------------------------
    def _init_cross_track_layers(self, cross_track_kwargs: Dict[str, Any] | None) -> None:
        """Set up learnable layers for cross-track mode, or placeholders otherwise."""
        if self.mode != 'cross-track':
            # Not in cross-track mode: create placeholder attributes
            self.cross_track_kwargs = None
            self.cross_track_layers = None
            self.cross_filter_layers = None
            self.cross_track_bn_layers = None
            self.scalar_recombiners = None
            self.vector_recombiners = None
            self.use_wjxs_batch_norm = False  # Default to False for non-cross-track modes
            return

        default_kw = {
            'n_cross_track_combos': 16,
            'n_cross_filter_combos': 8,
            'within_track_combine': False,
            'batch_norm_eps': 1e-5,
            'batch_norm_momentum': 0.1,
            'cross_track_mlp_hidden_dim': None,  # None -> single Linear (legacy)
            # New recombination parameters
            'use_wavelet_recombination': True,
            'scalar_recombination_channels': 8,
            'vector_recombination_channels': 8,
            'recombination_hidden_dim': 32,
            'vector_gate_hidden_dim': None,  # defaults to recombination_hidden_dim
            # Wjxs batch normalization parameter
            'use_wjxs_batch_norm': True,  # Whether to apply batch normalization to Wjxs
        }

        # Merge user-supplied kwargs with defaults
        self.cross_track_kwargs = merge_dicts_with_defaults(
            cross_track_kwargs or {}, default_kw
        )
        
        # Set instance variable for Wjxs batch normalization
        self.use_wjxs_batch_norm = self.cross_track_kwargs['use_wjxs_batch_norm']

        total_layers = self.scalar_track_kwargs['num_layers'] + 1  # skip + scattering layers

        # Helper to map scalar-track nonlin function (callable) -> nn.Module
        def _fn_to_module(fn_call):
            mapping = {
                F.relu: nn.ReLU,
                F.silu: nn.SiLU,
                F.tanh: nn.Tanh,
                F.gelu: nn.GELU,
            }
            return mapping.get(fn_call, nn.ReLU)  # fallback

        hidden_dims = self.cross_track_kwargs.get('cross_track_mlp_hidden_dim')

        cross_track_layers_list = []
        for _ in range(total_layers):
            if not hidden_dims:
                # Legacy single Linear layer (lazy in-features)
                layer = nn.LazyLinear(self.cross_track_kwargs['n_cross_track_combos'])
            else:
                modules = []
                # First hidden – LazyLinear because in_features unknown until fwd
                h0 = hidden_dims[0]
                modules.append(nn.LazyLinear(h0))
                act_cls = _fn_to_module(self.scalar_track_kwargs['nonlin_fn'])
                modules.append(act_cls())
                prev = h0
                # Additional hidden dims (if >1)
                for h in hidden_dims[1:]:
                    modules.append(nn.Linear(prev, h))
                    modules.append(act_cls())
                    prev = h
                # Output layer
                modules.append(nn.Linear(prev, self.cross_track_kwargs['n_cross_track_combos']))
                layer = nn.Sequential(*modules)

            cross_track_layers_list.append(layer)

        self.cross_track_layers = nn.ModuleList(cross_track_layers_list)

        # Cross-filter mixing (across wavelets)
        self.cross_filter_layers = nn.ModuleList([
            nn.LazyLinear(self.cross_track_kwargs['n_cross_filter_combos'])
            for _ in range(total_layers)
        ])

        # Replace BatchNorm with placeholder identities; we will lazily
        # create Wavelet-wise LayerNorm (batch-size independent) at runtime.
        self.cross_track_norm_layers = nn.ModuleList([
            nn.Identity() for _ in range(total_layers)
        ])

        # Initialize wavelet recombination layers (lazy initialization)
        if self.cross_track_kwargs.get('use_wavelet_recombination', True):
            self.scalar_recombiners = nn.ModuleList([None] * total_layers)
            self.vector_recombiners = nn.ModuleList([None] * total_layers)
        else:
            self.scalar_recombiners = None
            self.vector_recombiners = None

    # ------------------------------------------------------------------
    # Helper – lazy initialization of recombination layers
    # ------------------------------------------------------------------
    def _init_recombination_layers(
        self,
        layer_i: int,
        num_wavelets: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Lazily initialize recombination layers for a specific layer when we know the number of wavelets."""
        if (self.scalar_recombiners is None) \
        or (self.vector_recombiners is None):
            return  # Recombination not enabled
        
        if (not self.ablate_scalar_track) and (self.scalar_recombiners[layer_i] is None):
            # Initialize scalar recombination layer
            self.scalar_recombiners[layer_i] = self.ScalarWaveletRecombiner(
                num_wavelets=num_wavelets,
                num_output_channels=self.cross_track_kwargs['scalar_recombination_channels'],
                hidden_dim=self.cross_track_kwargs['recombination_hidden_dim'],
                nonlin_fn=self.scalar_track_kwargs['nonlin_fn']
            )
            if device is not None:
                self.scalar_recombiners[layer_i] = self.scalar_recombiners[layer_i].to(device)
        
        if (self.vector_recombiners[layer_i] is None) \
        and (not self.ablate_vector_track):
            # Initialize vector recombination layer
            self.vector_recombiners[layer_i] = self.GatedVectorWaveletRecombiner(
                num_wavelets=num_wavelets,
                num_output_channels=self.cross_track_kwargs['vector_recombination_channels'],
                hidden_dim=self.cross_track_kwargs['recombination_hidden_dim'],
                gate_hidden_dim=self.cross_track_kwargs.get('vector_gate_hidden_dim')
            )
            if device is not None:
                self.vector_recombiners[layer_i] = self.vector_recombiners[layer_i].to(device)

    # ------------------------------------------------------------------
    # Helper – vector nonlinearity and gating configuration
    # ------------------------------------------------------------------
    def _configure_vector_nonlinearity(self) -> None:
        """Set up self.vector_track_nonlin_fn and any gating parameters."""

        # If the vector track is ablated, skip parameter creation
        if self.ablate_vector_track:
            self.vector_track_nonlin_fn = lambda x, p=2: x  # no-op
            return

        vector_nonlin_type = self.vector_track_kwargs.get('vector_nonlin_type')

        # Default if still unspecified
        if vector_nonlin_type is None:
            vector_nonlin_type = 'silu-gate'
            self.vector_track_kwargs['vector_nonlin_type'] = vector_nonlin_type

        # --------------------------------------------------
        # Shifted-ReLU norm-based activation (equivariant)
        # --------------------------------------------------
        if vector_nonlin_type == 'shifted_relu':
            self.vector_track_nonlin_fn = (
                lambda x, p=2: self._vector_norm_nonlinearity(
                    x,
                    shift=self.vector_norm_shift,
                    p=p,
                    nonlinearity=torch.relu,
                )
            )
            return

        # --------------------------------------------------
        # Standard norm-based activations
        # --------------------------------------------------
        if vector_nonlin_type in VECTOR_NONLIN_FN_MAP:
            fn = VECTOR_NONLIN_FN_MAP[vector_nonlin_type]
            self.vector_track_nonlin_fn = lambda x, p=2, f=fn: self._vector_norm_nonlinearity(x, p=p, nonlinearity=f)
            return

        # --------------------------------------------------
        # Gated activations ( <act>-gate )
        # --------------------------------------------------
        if vector_nonlin_type.endswith('-gate'):
            gate_name = vector_nonlin_type.replace('-gate', '')

            if gate_name not in GATE_FN_MAP:
                raise ValueError(
                    f"Unsupported gate type '{gate_name}'. Supported: {list(GATE_FN_MAP)}."
                )

            init_val = self._get_gate_init_value(gate_name)
            self.vector_gate_param = nn.Parameter(torch.tensor(init_val, dtype=torch.float32), requires_grad=True)

            gate_fn = GATE_FN_MAP[gate_name]
            self.vector_track_nonlin_fn = lambda vec, p=2, gp=self.vector_gate_param, fn=gate_fn: fn(gp) * vec
            return

        # --------------------------------------------------
        # Unknown type
        # --------------------------------------------------
        raise ValueError(
            f"Unknown vector_nonlin_type: {vector_nonlin_type}."
        )

    # ------------------------------------------------------------------
    # Helper – default activation selection based on task
    # ------------------------------------------------------------------
    def _apply_task_based_default_activations(self) -> None:
        """Set default scalar/vector activations according to task name."""
        tt = self.task.lower()
        if 'reg' in tt:
            # Regression: smooth nonlinearity, gated silu vectors
            self.scalar_track_kwargs.setdefault('nonlin_fn', F.silu)
            self.scalar_track_kwargs.setdefault('nonlin_fn_kwargs', {})
            self.vector_track_kwargs.setdefault('vector_nonlin_type', 'silu-gate')
        elif 'class' in tt:
            # Classification: default scalar ReLU; vectors use sigmoid gate
            self.vector_track_kwargs.setdefault('vector_nonlin_type', 'sigmoid-gate')
        # Otherwise keep user/default settings

    # ------------------------------------------------------------------
    # Helper – merge kwargs with defaults & preprocess
    # ------------------------------------------------------------------
    def _prepare_kwarg_dicts(
        self,
        scalar_user: Dict[str, Any] | None,
        vector_user: Dict[str, Any] | None,
        mlp_user: Dict[str, Any] | None,
        base_user: Dict[str, Any] | None,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Merge user-provided kwargs with defaults and handle InfoGain scales."""

        # Clone defaults to avoid in-place mutation then merge
        scalar_final = merge_dicts_with_defaults(
            scalar_user or {}, copy.deepcopy(self.DEFAULT_SCALAR_KWARGS)
        )
        if vector_user is not None:
            vector_final = merge_dicts_with_defaults(
                vector_user or {}, copy.deepcopy(self.DEFAULT_VECTOR_KWARGS)
            )
        else:
            vector_final = None
        mlp_final = merge_dicts_with_defaults(
            mlp_user or {}, copy.deepcopy(self.DEFAULT_MLP_KWARGS)
        )
        base_final = merge_dicts_with_defaults(
            base_user or {}, copy.deepcopy(self.DEFAULT_BASE_KWARGS)
        )

        # Custom diffusion scales -> change wavelet type to 'custom'
        if scalar_final.get('diffusion_scales') is not None:
            scalar_final['diffusion_kwargs']['scales_type'] = 'custom'
            scalar_final['diffusion_kwargs']['diffusion_scales'] = scalar_final['diffusion_scales']

        if (vector_final is not None) and (vector_final.get('diffusion_scales') is not None):
            vector_final['diffusion_kwargs']['scales_type'] = 'custom'
            vector_final['diffusion_kwargs']['diffusion_scales'] = vector_final['diffusion_scales']

        return scalar_final, vector_final, mlp_final, base_final

    # ------------------------------------------------------------------
    # Helper – validate scalar/vector layer counts and init skip layers
    # ------------------------------------------------------------------
    def _validate_and_init_track_layers(
        self,
        mode: str,
        *,
        scalar_layers: int,
        vector_layers: int,
        cross_track_kwargs: Dict[str, Any] | None,
    ) -> None:
        """Ensure layer-count consistency and create within-track Linear layers."""

        within_track_combine = cross_track_kwargs.get('within_track_combine', False) if cross_track_kwargs else False

        if mode == 'filter-combine':
            if scalar_layers != vector_layers:
                raise ValueError(
                    "Number of layers in scalar and vector tracks must match in filter-combine mode."
                )
            self._init_within_track_parameters(nonlinearity='relu', create_layers=(vector_layers>0))
            return

        if mode == 'cross-track':
            if within_track_combine:
                if scalar_layers != vector_layers:
                    raise ValueError("Scalar/vector layers mismatch for within-track combinations.")
                self._init_within_track_parameters(nonlinearity='relu', create_layers=(vector_layers>0))
            else:
                # Only need shift parameter for vector nonlinearity.
                self._init_within_track_parameters(nonlinearity='relu', create_layers=False)
            return

        # Handcrafted scattering mode
        if mode == 'handcrafted_scattering':
            if scalar_layers > 2:
                raise ValueError(
                    f"Handcrafted_scattering mode only supports up to 2 layers. Got {scalar_layers} layers."
                )
            self._init_within_track_parameters(nonlinearity='relu', create_layers=False)
            return

        raise ValueError(f"Unknown mode '{mode}'.")

    # ------------------------------------------------------------------
    # Helper – make sure track layer attributes exist as ModuleLists
    # ------------------------------------------------------------------
    def _ensure_track_layer_placeholders(self, within_track_combine: bool) -> None:
        """Guarantee that scalar_track_layers and vector_track_layers are ModuleList objects."""
        if within_track_combine:
            return  # already initialized in _init_within_track_parameters

        if getattr(self, 'scalar_track_layers', None) is None:
            self.scalar_track_layers = nn.ModuleList()

        if getattr(self, 'vector_track_layers', None) is None:
            self.vector_track_layers = nn.ModuleList()

    # ------------------------------------------------------------------
    # Helper – ensure shift and gate parameters exist
    # ------------------------------------------------------------------
    def _ensure_vector_special_params(self) -> None:
        """Guarantee vector_norm_shift or vector_gate_param exists when needed."""
        vtype = self.vector_track_kwargs.get('vector_nonlin_type', 'shifted_relu')

        # ---- Shift parameter for shifted ReLU ----
        if (vtype == 'shifted_relu') and (not hasattr(self, 'vector_norm_shift')):
            self.vector_norm_shift = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
            self.vector_track_kwargs.setdefault('nonlin_fn_kwargs', {})['shift'] = self.vector_norm_shift

        # Gate parameter creation is handled during _configure_vector_nonlinearity.
        # This helper only guarantees shift parameter existence.

    def _init_wjxs_batch_norm_layers(
        self,
        scalar_channels: int, 
        vector_channels: int,
        num_wavelets: int,
        vector_dim: int = 3,
        device: Optional[torch.device] = None,
        *,
        init_scalar: bool = True,
        init_vector: bool = True,
    ) -> None:
        """
        Initialize batch normalization layers for Wjxs if not already initialized.
        
        For scalars: Each wavelet within each channel is normalized independently
        For vectors: Each coordinate within each wavelet within each channel is normalized independently
        Set init_scalar / init_vector to control which track(s) to initialize, preventing duplicate
        scalar-layer creation when this function is invoked from the vector track.
        """
        if not self.use_wjxs_batch_norm:
            return
        
        # Initialize ModuleLists if they don't exist or are None
        if self.scalar_wjxs_bn_layers is None and init_scalar:
            self.scalar_wjxs_bn_layers = nn.ModuleList()
        if self.vector_wjxs_bn_layers is None and init_vector and not self.ablate_vector_track:
            self.vector_wjxs_bn_layers = nn.ModuleList()
        
        # ------------------------------------------------------------------
        # Scalar-track BN layers (per-wavelet) – created only when requested
        # ------------------------------------------------------------------
        if init_scalar:
            layer_bn_list = nn.ModuleList()
            for _ in range(num_wavelets):
                bn_layer = nn.BatchNorm1d(
                    scalar_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                )
                if device is not None:
                    bn_layer = bn_layer.to(device)
                layer_bn_list.append(bn_layer)
            self.scalar_wjxs_bn_layers.append(layer_bn_list)
        
        # ------------------------------------------------------------------
        # Vector-track BN layers (per-coordinate, per-wavelet) – optional
        # ------------------------------------------------------------------
        if init_vector and not self.ablate_vector_track:
            layer_bn_list = nn.ModuleList()
            for _ in range(num_wavelets):
                coord_bn_list = nn.ModuleList()
                for _ in range(vector_dim):
                    bn_layer = nn.BatchNorm1d(
                        vector_channels,
                        eps=1e-5,
                        momentum=0.1,
                        affine=True,
                    )
                    if device is not None:
                        bn_layer = bn_layer.to(device)
                    coord_bn_list.append(bn_layer)
                layer_bn_list.append(coord_bn_list)
            self.vector_wjxs_bn_layers.append(layer_bn_list)
        
        if self.verbosity > 0:
            trk = []
            if init_scalar:
                trk.append("scalar")
            if init_vector and not self.ablate_vector_track:
                trk.append("vector")
            print(
                f"[VDW] Initialized Wjxs BN for layer (tracks: {', '.join(trk)}) "
                f"scalar_channels={scalar_channels}, vector_channels={vector_channels}, "
                f"num_wavelets={num_wavelets}, vector_dim={vector_dim}"
            )

    def _apply_wjxs_batch_norm(
        self,
        Wjxs: torch.Tensor, 
        track_name: str, 
        layer_idx: int
    ) -> torch.Tensor:
        """
        Apply batch normalization to Wjxs tensor.
        
        For scalars: Each wavelet within each channel is normalized independently
        For vectors: Each coordinate within each wavelet within each channel is normalized independently
        """
        if not self.use_wjxs_batch_norm:
            return Wjxs
        
        # Get the appropriate batch normalization layers
        if track_name == 'scalar':
            bn_idx = layer_idx - 1  # scattering layers start at 1 (skip = 0)
            if self.scalar_wjxs_bn_layers is None or bn_idx < 0 or bn_idx >= len(self.scalar_wjxs_bn_layers):
                return Wjxs
            bn_layers = self.scalar_wjxs_bn_layers[bn_idx]  # List of BN layers, one per wavelet
        elif track_name == 'vector':
            bn_idx = layer_idx - 1
            if self.vector_wjxs_bn_layers is None or bn_idx < 0 or bn_idx >= len(self.vector_wjxs_bn_layers):
                return Wjxs
            bn_layers = self.vector_wjxs_bn_layers[bn_idx]  # List of lists of BN layers, one per coordinate per wavelet
        else:
            return Wjxs
        
        N, C, W = Wjxs.shape
        
        # Check if the number of wavelets matches the batch normalization layers
        if track_name == 'scalar':
            if W != len(bn_layers):
                raise ValueError(
                    f"Wjxs has {W} wavelets but batch norm has {len(bn_layers)} layers for scalar track, layer {layer_idx}. "
                    f"This indicates a mismatch in the number of wavelets between initialization and application."
                )
        elif track_name == 'vector':
            if W != len(bn_layers):
                raise ValueError(
                    f"Wjxs has {W} wavelets but batch norm has {len(bn_layers)} layers for vector track, layer {layer_idx}. "
                    f"This indicates a mismatch in the number of wavelets between initialization and application."
                )
        
        if track_name == 'scalar':
            # For scalars: normalize each wavelet independently
            # Wjxs shape: (N, C, W) -> apply BN to each wavelet separately
            normalized_wavelets = []
            for w_idx in range(W):
                # Extract wavelet w_idx: (N, C, 1) -> (N, C)
                wavelet_data = Wjxs[:, :, w_idx]  # (N, C)
                # Check if the channel count matches the batch normalization layer
                if wavelet_data.shape[1] != bn_layers[w_idx].num_features:
                    raise ValueError(
                        f"Wjxs has {wavelet_data.shape[1]} channels but batch norm expects {bn_layers[w_idx].num_features} "
                        f"for scalar track, layer {layer_idx}, wavelet {w_idx}. "
                        f"This indicates a mismatch in the number of channels between initialization and application."
                    )
                # Apply BN for this wavelet
                normalized_wavelet = bn_layers[w_idx](wavelet_data)  # (N, C)
                normalized_wavelets.append(normalized_wavelet.unsqueeze(-1))  # (N, C, 1)
            # Concatenate back: (N, C, W)
            Wjxs = torch.cat(normalized_wavelets, dim=2)
            
        elif track_name == 'vector':
            # For vectors: normalize each coordinate within each wavelet independently
            # Wjxs shape: (N*d, C, W) where N is num_nodes, d is vector_dim
            # We need to reshape to (N, d, C, W) to apply BN per coordinate
            d = self.vector_track_kwargs.get('vector_dim', 3)
            if Wjxs.shape[0] == N * d:
                Wjxs_reshaped = Wjxs.view(N, d, C, W)  # (N, d, C, W)
                
                normalized_wavelets = []
                for w_idx in range(W):
                    normalized_coords = []
                    for coord_idx in range(d):
                        # Extract coordinate coord_idx, wavelet w_idx: (N, C)
                        coord_wavelet_data = Wjxs_reshaped[:, coord_idx, :, w_idx]  # (N, C)
                        # Check if the channel count matches the batch normalization layer
                        if coord_wavelet_data.shape[1] != bn_layers[w_idx][coord_idx].num_features:
                            raise ValueError(
                                f"Wjxs has {coord_wavelet_data.shape[1]} channels but batch norm expects {bn_layers[w_idx][coord_idx].num_features} "
                                f"for vector track, layer {layer_idx}, wavelet {w_idx}, coordinate {coord_idx}. "
                                f"This indicates a mismatch in the number of channels between initialization and application."
                            )
                        # Apply BN for this coordinate and wavelet
                        normalized_coord = bn_layers[w_idx][coord_idx](coord_wavelet_data)  # (N, C)
                        normalized_coords.append(normalized_coord.unsqueeze(1))  # (N, 1, C)
                    # Concatenate coordinates: (N, d, C)
                    normalized_wavelets.append(torch.cat(normalized_coords, dim=1).unsqueeze(-1))  # (N, d, C, 1)
                # Concatenate wavelets: (N, d, C, W)
                Wjxs_reshaped = torch.cat(normalized_wavelets, dim=3)
                # Reshape back: (N*d, C, W)
                Wjxs = Wjxs_reshaped.view(N * d, C, W)
        
        if self.verbosity > 2:
            print(f"\tAfter Wjxs batch norm stats: min={torch.min(Wjxs).item():.4e}, max={torch.max(Wjxs).item():.4e}, has_nan={torch.isnan(Wjxs).any().item()}")
        
        return Wjxs

    # ------------------------------------------------------------------
    # Helper – optional scalar feature embedding initialization
    # ------------------------------------------------------------------
    def _init_scalar_embedding(self) -> None:
        """Create `self.scalar_feat_embedding` if `embedding_dim` is set.

        The embedding projects raw scalar node features into a learned
        space before the first wavelet transform.  `embedding_dim` must
        be an integer specifying the output dimensionality.
        """
        self.scalar_feat_embedding = None

        embedding_dim = self.scalar_track_kwargs.get('embedding_dim', None)
        if (embedding_dim is None) or self.ablate_scalar_track:
            return  # No embedding required

        # Enforce int type (lists/tuples are no longer accepted)
        if not isinstance(embedding_dim, int):
            raise TypeError(
                f"`embedding_dim` must be an int (got {type(embedding_dim).__name__})."
            )

        # LazyLinear initializes weights on first forward call, so we
        # don't need to know the input feature dimension here.
        self.scalar_feat_embedding = nn.LazyLinear(embedding_dim)
