"""
GAT-FM: Graph Attention Network with Flow Matching for Protein Prediction.

A framework for predicting protein expression from single-cell RNA data using:
- Graph Attention Networks (GATv2Conv) with protein-protein interaction structure
- Flow Matching for generative modeling
- Adaptive Layer Normalization (adaLN-Zero) conditioning

Based on the architecture described in GAT-Diffusion-Protein-Prediction.md
"""

__version__ = '0.1.0'

# Core model components
from .model import (
    GATFM,
    DITFM,
    GATBlock,
    DiTBlock,
    TimestepEmbedder,
    DatasetEmbedder,
    RNAProjector,
    FinalLayer,
    modulate,
)

# Graph utilities
from .graph import (
    ProteinGraph,
    load_ppi_network,
    build_protein_graph,
    apply_mask_aware_edge_pruning,
    batch_graphs_with_masks,
    create_dense_graph,
    create_sparse_identity_graph,
)

# Data loading
from .data import (
    ProteinDataset,
    MultiDatasetLoader,
    NormalizationStats,
    create_dataloader,
    load_single_dataset,
    normalize_protein_expression,
    denormalize_protein_expression,
)

# Sampling and Flow Matching
from .sampling import (
    FlowMatchingSampler,
    GuidedFlowMatchingSampler,
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    sample_time_uniform,
    sample_time_logit_normal,
    get_flow_interpolation,
    sample_conditional_ot,
    create_sample_mask,
)

# Training
from .trainer import (
    GATFMWrapper,
    Trainer,
    TrainingConfig,
    compute_pcc,
    compute_rmse,
    compute_mae,
    compute_pcc_protein,
    compute_pcc_cell,
    compute_rmse_standardized,
    compute_cmd_cell,
    compute_cmd_protein,
    standardize_for_evaluation,
    evaluate_model,
)

# Utilities
from .utils import (
    set_seed,
    get_device,
    count_parameters,
    model_summary,
    save_config,
    load_config,
    AverageMeter,
    EarlyStopping,
    Logger,
    create_ema_model,
    exponential_moving_average,
)

# Preprocessing
from .preprocess import (
    validate_adata,
    prepare_adata_for_training,
    merge_datasets,
    preprocess_for_training,
    split_adata,
)

__all__ = [
    # Version
    '__version__',
    
    # Model
    'GATFM',
    'DITFM',
    'GATBlock',
    'DiTBlock',
    'TimestepEmbedder',
    'DatasetEmbedder',
    'RNAProjector',
    'FinalLayer',
    'modulate',
    
    # Graph
    'ProteinGraph',
    'load_ppi_network',
    'build_protein_graph',
    'apply_mask_aware_edge_pruning',
    'batch_graphs_with_masks',
    'create_dense_graph',
    'create_sparse_identity_graph',
    
    # Data
    'ProteinDataset',
    'MultiDatasetLoader',
    'NormalizationStats',
    'create_dataloader',
    'load_single_dataset',
    'normalize_protein_expression',
    'denormalize_protein_expression',
    
    # Sampling
    'FlowMatchingSampler',
    'GuidedFlowMatchingSampler',
    'ConditionalFlowMatcher',
    'ExactOptimalTransportConditionalFlowMatcher',
    'sample_time_uniform',
    'sample_time_logit_normal',
    'get_flow_interpolation',
    'sample_conditional_ot',
    'create_sample_mask',
    
    # Training
    'GATFMWrapper',
    'Trainer',
    'TrainingConfig',
    'compute_pcc',
    'compute_rmse',
    'compute_mae',
    'compute_pcc_protein',
    'compute_pcc_cell',
    'compute_rmse_standardized',
    'compute_cmd_cell',
    'compute_cmd_protein',
    'standardize_for_evaluation',
    'evaluate_model',
    
    # Utils
    'set_seed',
    'get_device',
    'count_parameters',
    'model_summary',
    'save_config',
    'load_config',
    'AverageMeter',
    'EarlyStopping',
    'Logger',
    'create_ema_model',
    'exponential_moving_average',
    
    # Preprocessing
    'validate_adata',
    'prepare_adata_for_training',
    'merge_datasets',
    'preprocess_for_training',
    'split_adata',
]
