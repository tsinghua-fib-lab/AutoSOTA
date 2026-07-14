# config/model_config.py
"""
This file contains the configuration for the model.
It is used to specify the model, the model architecture,
and the model initialization parameters.

Example yaml file:
model:
  model_key: 'vdw'
  model_mode: 'handcrafted_scattering'
  wavelet_scales_type: 'dyadic'
  J: 4
  mlp_hidden_dim: [256, 128, 64, 32]
  mlp_dropout_p: 0.2  # Set to None for no dropout
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any, Literal, Tuple, Union, Dict

@dataclass
class ModelConfig:
    
    # Model selection
    # Model types supported by the training pipeline.  
    #  - 'vdw':  Spectral Operator-based graph scattering network  
    #  - 'vdw_radial':  Bessel RBF version of VDW with gated message passing
    #  - 'vdw_modular': A simplified, modular version of VDW
    #  - 'mlp'  :  A simple multi-layer perceptron (used with handcrafted features)
    #  - 'egnn' | 'tfn' | 'legs' | 'gcn' | 'gin' | 'gat': comparison baselines (wrapped)
    model_key: Literal[
        'vdw', 
        'vdw_modular', 
        'mlp', 
        'egnn', 
        'tfn', 
        'legs', 
        'gcn',
        'gin',
        'gat',
        'vdw_supcon_2',
        'tnn',
    ] = 'vdw'
    model_mode: Literal['handcrafted_scattering', 'filter-combine', 'cross-track'] = 'handcrafted_scattering'
    num_scattering_layers_scalar: int = 2
    num_scattering_layers_vector: int = 2
    column_normalize_P: bool = False
    equivar_pred: bool = True
    # Diffusion scales parameters (string or list of ints)
    # These allow custom diffusion scales to be specified for each track.
    # NOTE: these will override the model_config.p_wavelet_scales_type parameter,
    # If set to the string 'dyadic' (default), standard dyadic scales
    # controlled by the J parameter will be used. Otherwise, provide either
    # 1) a list of integers that will be shared across all channels, or
    # 2) a list of lists, giving per-channel custom scales.
    # The lists will be converted to torch.LongTensor objects later in the
    # model preparation step.
    scalar_diffusion_scales: Union[Literal['dyadic'], List[int], List[List[int]]] = 'dyadic'
    vector_diffusion_scales: Union[Literal['dyadic'], List[int], List[List[int]]] = 'dyadic'
    scalar_operator_key: str = 'P'
    vector_operator_key: str = 'Q'
    scalar_feature_key: Optional[str] = None
    learnable_P: bool = False
    learnable_p_kwargs: Optional[Dict[str, Any]] = None
    learnable_p_num_views: int = 1
    learnable_p_view_aggregation: Literal['concat', 'mean', 'sum'] = 'concat'
    learnable_p_softmax_temps: Optional[List[float]] = None
    learnable_p_laziness_inits: Optional[List[float]] = None
    learnable_p_fix_alpha: bool = False
    learnable_p_use_softmax: bool = False
    learnable_p_use_attention: bool = False
    learnable_p_attention_kwargs: Optional[Dict[str, Any]] = None
    # Optional: diffusion operator key for line graph scattering on edges
    line_operator_key: str = 'P_line'
    # flat_vector_feat_key: str = 'v_flat'  # legacy key for VDW
    
    # ------------------------------------------------------------------
    # Newer VDW model architecture parameters (line, molecule)
    # ------------------------------------------------------------------
    wavelet_scales_type: Literal['dyadic', 'infogain'] = 'dyadic'
    include_lowpass_wavelet: bool = True
    # CkNN neighbor floor to avoid degenerate local PCA/O-frames; if True uses d
    cknn_min_nbrs: Optional[Union[int, bool]] = None
    J_scalar: int = 4  # Number of wavelet scales for scalar track
    J_scalar_line: int = 4  # Number of wavelet scales for line-graph scalar track
    J_vector: int = 4  # Number of wavelet scales for vector track
    J_prime_scalar: Optional[int] = None  # Number of highest-frequency wavelets to keep in second layer (scalar)
    J_prime_vector: Optional[int] = None  # Number of highest-frequency wavelets to keep in second layer (vector)
      # Nonlinearity/alignment knobs for 2nd-order scattering
    
    # Offline scattering nonlinearities (precompute stage)
    # - scalar_scatter_first_order_nonlin: 'abs' | 'relu' | None (applied before 2nd-order, but not 1st order outputs)
    # - scalar_scatter_output_nonlin: 'abs' | 'relu' | None (applied to 1st/2nd outputs)
    # - vector_scatter_first_order_nonlin: 'softplus_gate' | None (applied before 2nd-order, but not 1st order outputs)
    # - vector_scatter_output_nonlin: 'softplus_gate' | None (applied to 1st/2nd outputs)
    scalar_scatter_first_order_nonlin: Optional[Literal['abs', 'relu']] = 'abs'
    scalar_scatter_output_nonlin: Optional[Literal['abs', 'relu']] = None
    vector_scatter_first_order_nonlin: Optional[Literal['softplus_gate']] = 'softplus_gate'
    vector_scatter_output_nonlin: Optional[Literal['softplus_gate']] = None
    # When True, replace non-finite values in 2nd-order scattering with zeros.
    replace_nonfinite_second_order: bool = False

    # Legacy boolean flags: behavior can be implied by non-None vector_scatter_first_order_nonlin
    apply_scalar_first_order_nonlin: bool = True
    apply_vector_first_order_align_gating: bool = True

    # ------------------------------------------------------------------
    # VDW (v1-3) model parameters
    # ------------------------------------------------------------------
    # cross-track mode parameters
    n_cross_track_combos: int = 16
    n_cross_filter_combos: int = 8
    within_track_combine: bool = False  # If True, reuse per-track filter-combine layers inside cross-track mode

    # filter-combine mode parameters
    # (only used when model_mode == 'filter-combine')
    filter_combos_out: List[int] = field(
        default_factory=lambda: [16, 8]
    )
    
    # Cross-track per-wavelet MLP depth/width.  If None -> single Linear (legacy
    # behavior).  Otherwise the list of ints defines the hidden layer sizes
    # of an MLP applied *per wavelet* to mix scalar and vector channels before
    # cross-filter mixing.  Example: [128, 128] builds a 2-layer hidden MLP with
    # 128 units each.
    cross_track_mlp_hidden_dim: Optional[List[int]] = None

    # Wavelet recombination parameters (cross-track mode only)
    # Whether to use wavelet recombination layers in cross-track mode
    use_wavelet_recombination: bool = True
    
    # Number of output channels for scalar wavelet recombination
    scalar_recombination_channels: int = 16
    
    # Number of output channels for vector wavelet recombination  
    vector_recombination_channels: int = 16
    
    # Hidden dimension for recombination 2-layer MLPs
    recombination_hidden_dim: int = 64
    
    # Hidden dimension for vector gate MLP (defaults to recombination_hidden_dim if None)
    vector_gate_hidden_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # VDWRadial-specific parameters
    # ------------------------------------------------------------------
    num_msg_pass_layers: int = 1
    use_residual_connections: bool = True
    
    # Hidden dimension for atom and bond type embeddings in VDWRadial
    edge_embedding_dim: Optional[int] = None
    node_embedding_dim: Optional[int] = None

    # MLP hidden dims for scalar condensation
    scalar_condense_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    d_scalar_hidden: int = 64

    # MLP hidden dims for scalar gate
    scalar_gate_mlp_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    scalar_gate_nonlin: str = 'silu'
    scalar_gate_rank: int = 8

    # MLP hidden dims for vector gate
    vector_gate_mlp_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    vector_gate_mlp_nonlin: str = 'silu'
    vector_gate_rank: int = 8

    # Hidden dimension for gating MLPs in VDWRadial (legacy, may be ignored)
    gate_hidden_dim: int = 128
    gate_rank: int = 8

    # Whether to use Dirac nodes in the scattering layers
    use_dirac_nodes: bool = False
    # Types of Dirac nodes to include as indicator channels, when enabled
    dirac_types: Optional[List[Literal['max', 'min']]] = field(
        default_factory=lambda: ['max', 'min']
    )
    use_temporal_residuals: bool = True

    # ------------------------------------------------------------------
    # InfoGain wavelet scales (optional)
    # ------------------------------------------------------------------
    # If provided, these override dyadic scales for the respective track.
    # Can be None (use dyadic), a 1D list/tensor (average scales for all channels),
    # or a 2D list/tensor (per-channel scales, shape [n_channels, n_scales])
    infogain_scales_scalar: Optional[Any] = None  # e.g. List[List[int]] or torch.Tensor
    infogain_scales_vector: Optional[Any] = None  # e.g. List[List[int]] or torch.Tensor

    # ------------------------------------------------------------------
    # Pooling parameters
    # ------------------------------------------------------------------
    pooling_type: Tuple[Literal['sum', 'max', 'median', 'statistical_moments'], ...] = ('sum')
    moments: Tuple[int, ...] = (1, 2, 3)  # Statistical moments to compute (1 = mean, 2 = variance, 3 = skewness, 4 = kurtosis)
    nan_replace_value: float = 0.0  # Value to replace NaN tensor values
    vector_norm_p: int = 2  # p-norm to use for vector features

    # ------------------------------------------------------------------
    # Nonlinearity choices
    # ------------------------------------------------------------------
    # Nonlinearity applied inside the VDW scattering backbone (scalar track).
    scalar_nonlin: Literal['relu', 'silu', 'tanh'] = 'silu'

    # Nonlinearity applied between layers of the MLP head.
    #   Options correspond to torch.nn modules for easy instantiation in VanillaNN.
    mlp_nonlin: Literal['relu', 'silu', 'tanh'] = 'silu'

    # Nonlinearity used in the VDW *vector track*.
    #   Options correspond to values accepted by VDW.vector_track_kwargs['vector_nonlin_type'].
    #   Examples: 'shifted_relu', 'silu', 'silu-gate', 'relu-gate', 'sigmoid-gate', 'tanh', 'softplus'
    vector_nonlin: str = 'silu-gate'

    # Readout MLP configuration
    readout_type: Literal['deepsets', 'mlp'] = 'mlp'
    deepsets_hidden_dim: int = 128
    mlp_hidden_dim: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32]
    )
    mlp_dropout_p: Optional[float] = 0.2  # Set to None for no dropout
    mlp_use_batch_normalization: bool = False

    # ------------------------------------------------------------------
    # Comparison model hyperparameters
    # ------------------------------------------------------------------
    comparison_model_hidden_channels: int = 128  # Size of latent representation
    comparison_model_num_layers: int = 5
 
    # ------------------------------------------------------------------
    # Distance (e.g. Bessel) edge feature parameters
    # ------------------------------------------------------------------
    # Number of edge features (Bessel basis functions) per edge. If None, edge features are not computed or stored.
    num_edge_features: Optional[int] = 16

    # ------------------------------------------------------------------
    # TFN-specific defaults
    # ------------------------------------------------------------------
    tfn_r_max: float = 5.0
    tfn_num_bessel: int = 8
    tfn_num_polynomial_cutoff: int = 6
    tfn_max_ell: int = 2
    tfn_mlp_dim: int = 256
    # Radial embedding mode: also accepts kernel names ('gaussian','cosine_cutoff','epanechnikov')
    tfn_radial_mode: Literal['mlp_gates', 'bessel_cutoff', 'gaussian', 'cosine_cutoff', 'epanechnikov'] = 'mlp_gates'
    tfn_radial_mlp_hidden: List[int] = field(default_factory=lambda: [64, 64])
    tfn_radial_mlp_activation: Literal['relu', 'silu', 'swish', 'gelu'] = 'relu'
    # Equivariant vector head selection
    tfn_unbiased_vector_pred_head: bool = True

    # ------------------------------------------------------------------
    # VDWModular-specific parameters
    # ------------------------------------------------------------------
    # Within-track wavelet-mixing MLPs
    scalar_wavelet_mlp_hidden: List[int] = field(default_factory=lambda: [32, 32])
    # Replaced: vector_wavelet_mlp_hidden -> vector_wavelet_mixer_linear_dim (single int)
    vector_wavelet_mixer_linear_dim: Optional[Union[int, List[int]]] = 128
    scalar_wavelet_mlp_dropout: float = 0.0
    scalar_wavelet_mlp_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    W_out_scalar: Optional[int] = 8
    W_out_vector: Optional[Union[int, List[int]]] = 8
    use_scalar_wavelet_batch_norm: bool = False
    use_vector_wavelet_batch_norm: bool = False
    include_l2_distance_edge_feat: bool = False
    include_bond_attr_edge_feat: bool = False
    num_rbf_scatter: int = 16
    rbf_cutoff: float = 5.0

    # Gating modes for vector first-order alignment and vector wavelet mixing
    # - vector_first_order_align_gating_mode: 'ref_align' uses the
    #   reference-alignment method; 'simple_affine' uses sigma(softplus(a)*||v|| + b)
    #   with per-wavelet learnable a and b.
    # - vector_wavelet_mixing_gate_mode: 'no_norm' uses the bias-free
    #   linear + scalar gates; 'simple_affine' uses input-dependent gates based on
    #   the norms of the recombined vector wavelets.
    vector_first_order_align_gating_mode: Literal['ref_align', 'simple_affine', 'norm_only'] = 'ref_align'
    vector_first_order_align_gating_nonlinearity: Literal['sigmoid', 'softplus', 'tanh', 'identity'] = 'softplus'
    vector_wavelet_mixing_gate_mode: Literal['no_norm', 'simple_affine', 'norm_only', 'param_norm', 'param_only'] = 'no_norm'
    # Activation used for SimpleAffineGate's sigma; defaults to 'sigmoid'.
    vector_wavelet_mixer_gate_nonlinearity: Literal['sigmoid', 'softplus', 'tanh', 'identity'] = 'sigmoid'

    # Node head and vector gate MLPs
    node_scalar_head_hidden: List[int] = field(default_factory=lambda: [64, 64])
    node_scalar_head_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    node_scalar_head_dropout: float = 0.1
    vector_gate_hidden: List[int] = field(default_factory=lambda: [64, 64])
    vector_gate_mlp_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    vector_gate_use_sigmoid: bool = True
    vector_gate_init_temperature: float = 1.0
    use_scalar_in_vector_gate: bool = True
    use_neighbor_cosines: bool = True
    use_learned_static_vector_weights: bool = False
    normalize_final_vector_gates: bool = False
    # Optional: apply learned final rotation layer for (node) vector targets
    vec_target_use_final_rotation_layer: bool = False

    # ------------------------------------------------------------------
    # ScatterMLP / vdw_layer parameters
    # ------------------------------------------------------------------
    scatter_mlp_layers: int = 2
    scatter_mlp_hidden: List[int] = field(default_factory=lambda: [128, 128])
    scatter_mlp_dropout: float = 0.0
    scatter_mlp_use_batch_norm: bool = False
    scatter_mlp_activation: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    scatter_include_zero_order: bool = True
    scatter_use_residual: bool = True
    scatter_use_neighbor_concat: bool = True
    scatter_neighbor_k: Optional[int] = None
    scatter_neighbor_include_edge_weight: bool = True

    # VDW_macaque: whether to use vector invariants (norms, cosines) or preserve vectors
    # - True: convert vectors to rotation-invariant scalars, use MLP prediction head
    # - False: preserve SO(d)-equivariant vectors, use linear projection (for equivariant mode)
    use_vec_invariants: bool = True
    
    # VDW_macaque: whether to use per-trajectory path graphs (legacy mode)
    scatter_path_graphs: bool = True
    
    # VDW_macaque graph aggregation mode (for graph-level tasks with invariants)
    # - 'flatten': flatten all node features (N*F dims, preserves temporal ordering)
    # - 'pool_stats': compute mean and variance across nodes (2*F dims, dimension-reducing)
    graph_aggregation_mode: Literal['flatten', 'pool_stats'] = 'pool_stats'
    
    # VDW_macaque_supcon-specific configuration
    mode: Literal['pretrain', 'train', 'finetune'] = 'pretrain'
    save_embeddings: bool = False
    freeze_encoder: bool = False
    temperature: float = 0.1
    # Evaluation mode for held-out embeddings
    # - 'weighted_average': fixed-k Gaussian-weighted neighbor averaging (default)
    # - 'weighted_average_adaptive': adaptive-k using median train-degree per node from holdout_k_probe-nearest train neighbors
    # - 'forward_insert': temporarily insert held-out nodes and build Q rows for a forward pass
    eval_mode: Literal['weighted_average', 'weighted_average_adaptive', 'forward_insert'] = 'weighted_average'
    holdout_k_probe: int = 10
    vector_bn_momentum: float = 0.1
    vector_bn_eps: float = 1e-6
    vector_bn_track_running_stats: bool = True
    vector_bn_enabled: bool = True
    temporal_hidden_channels: Optional[List[int]] = field(default_factory=lambda: [128, 128])
    temporal_kernel_sizes: Optional[List[int]] = field(default_factory=lambda: [3, 5])
    temporal_paddings: Optional[List[int]] = None
    temporal_activation: str = 'ReLU'
    temporal_out_dim: int = 128
    projection_hidden_dim: Union[int, List[int]] = 256
    projection_embedding_dim: int = 128
    projection_activation: str = 'ReLU'
    projection_use_batch_norm: bool = True
    projection_dropout_p: Optional[float] = None
    projection_residual_style: bool = False
    classification_hidden_dims: List[int] = field(default_factory=lambda: [64])
    classification_activation: str = 'ReLU'
    classification_dropout_p: float = 0.1
    eval_cluster_metrics: List[str] = field(
        default_factory=lambda: ['svm', 'kmeans', 'spectral', 'knn']
    )
    svm_kernel: str = 'rbf'
    svm_c: float = 1.0
    svm_gamma: Any = 'scale'
    supcon_neighbor_k: int = 1
    pos_pairs_per_anchor: Optional[int] = None
    neg_topk_per_positive: int = 7
    random_negatives_per_anchor: int = 256
    supcon_sampling_max_nodes: Optional[int] = None
    # Learnable Mahalanobis top-k (VDW SupCon v2)
    use_learnable_topk: bool = False
    learnable_topk_k: Optional[int] = None
    learnable_topk_proj_dim: Optional[int] = None
    learnable_topk_temperature: float = 1.0
    learnable_topk_eps: float = 1e-8
    


    # k-NN / neighbor cosine feature options
    equal_degree: bool = False
    k_neighbors: int = 5
    cknn_dist_metric: str = "euclidean"
    scattering_k: Optional[int] = None
    cknn_delta: float = 1.5
    neighbor_use_padding: bool = True
    neighbor_pool_stats: List[str] = field(default_factory=lambda: ['max', 'mean', 'var'])
    # Controls the number of quantile stats when 'percentiles'/'quantiles' is enabled (slow)
    quantiles_stride: float = 0.2

    # ------------------------------------------------------------------
    # Message-passing parameters (molecule/line variants)
    # ------------------------------------------------------------------
    # Depth (message passing layers)
    msg_pass_num_layers: int = 2
    # Optional atom-type embedding for scalar inputs
    atom_embedding_dim: int = 128
    # Radial basis (Bessel) feature count; cutoff will use dataset_config.distance_cutoff
    num_rbf: int = 8
    num_rbf_scatter: int = 0

