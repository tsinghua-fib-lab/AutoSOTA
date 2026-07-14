# config/dataset_config.py
"""
This file contains the default/overridable configuration for the dataset.
It is used to specify the dataset, the split parameters,
and the data loader parameters.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, List, Callable, Literal
import torch

@dataclass
class DatasetConfig:
    """Configuration for dataset initialization and behavior.
    
    h5_path: Optional path to HDF5 file storing, e.g., P and Q tensors for each sample. 
    If set, dataset loading will not attach P and Q, but expects them to be loaded 
    from HDF5 in the custom Dataset class.
    subsample_n: If set, only use a random subset of this many samples (overridden by clarg if provided).
    subsample_seed: Random seed for dataset subsampling (used for both subsample_n and data_subset_n).
    target_include_indices: List of indices to keep from the target (y) tensor. If set, the y attribute in each Data/Batch object will be subset to include only these indices.
    """
    
    # Required paths
    data_dir: str = None # must be provided by user in yaml file
    dataset_filename: Optional[str] = None
    diffusion_tensor_data_dir: Optional[str] = "pq_tensor_data"
    h5_path: Optional[str] = None  # Path to HDF5 file, e.g., for P and Q tensors
    save_processed_dataset: bool = False  # Whether to persist processed Data objects for reuse
    processed_dataset_path: Optional[str] = None  # Override path to load a previously processed dataset
    subsample_n: Optional[int] = None  # If set, only use a random subset of this many samples
    subsample_seed: Optional[int] = 123456  # Random seed for dataset subsampling

    # Dataset selection and basic parameters
    dataset: str = 'ellipsoids'
    scalar_feat_key: str = 'x'
    vector_feat_key: str = 'pos'
    geom_feat_key: Optional[str] = None
    atomic_number_attrib_key: str = 'z'
    bond_attr_key: str = 'edge_attr'
    bond_type_key: str = 'edge_type'
    num_bond_types: int = 4  # Number of bond types for embedding
    vector_feat_dim: int = 3
    # 'task' string needs (1) 'vector', if a vector target; (2) 'graph' or 'node' to indicate
    # the level of the target (e.g., graph-level or node-level), and (3) 'regression' 
    # or 'classification' to indicate the type of target 
    # examples: 'graph_regression' or 'vector_node_regression'
    task: str = 'graph_regression'  
    target_key: str = 'y'
    target_dim: int = 1  # 19 available
    num_atom_types: int = 5  # Number of atom types for embedding

    # Optional list of scalar node feature indices to keep. If set, the `x` attribute
    # of each Data/Batch object will be subset to include ONLY these feature columns
    # in the given order.
    scalar_node_feats_include_indices: Optional[List[int]] = None

    # DEPRECATED: use `scalar_node_feats_include_indices` instead.
    # node_feats_include_indices: Optional[List[int]] = None

    # Whether to use an embedding of atom types (via `z`) as scalar inputs.
    # If False, scalar features in `x` will be used directly (optionally subset).
    use_scalar_node_feat_embedding: bool = True

    # Vector feature preprocessing
    vector_norms_mean: Optional[float] = None
    vector_norms_std: Optional[float] = None
    
    # Node C_i rank deficiency handling
    rank_deficiency_strategy: Optional[Literal['Tikhonov']] = None  # Strategy for handling k < d (e.g., 'Tikhonov')
    tikhonov_epsilon: float = 0.001  # Small isotropic regularization strength if strategy == 'Tikhonov'
    # Alignment method for local PCA singular vectors across neighbors
    sing_vect_align_method: Literal['column_dot', 'procrustes'] = 'column_dot'

    # Local PCA distance kernel for weighting vector neighbors
    local_pca_distance_kernel: Literal[
        "gaussian", "cosine_cutoff", "epanechnikov",
    ] = 'cosine_cutoff'
    local_pca_distance_kernel_scale: Optional[float] = None
    use_mean_recentering: bool = False

    # Radial/continuous feature processing (don't confuse with categorical edge 'type' attributes)
    edge_rbf_key: str = 'edge_features'
    num_edge_features: int = 0

    # ------------------------------------------------------------------
    # Graph construction settings
    # ------------------------------------------------------------------
    # How to build the graph (edge_index) used for diffusion operators and
    # any downstream message-passing layers.
    #   - 'chemical_bonds'   – keep the bond graph shipped with dataset
    #   - 'distance_cutoff'  – connect any pair of atoms within `distance_cutoff` angstroms
    #                          (default and dataset-agnostic)
    graph_construction: Optional[Literal['k-nn', 'radius', 'reweight_existing_edges']] = None

    # Distance threshold for the radius-graph when `graph_construction == 'distance_cutoff'`
    distance_cutoff: float = 5.0

    # Cap on neighbours per node
    max_num_neighbors: int = 16

    # Process edge features into node features
    use_edge_as_node_features: bool = False  # Whether to process edge features into node features
    # edge_to_node_feature_key: str = 'bond_node_features'  # Key for the processed edge features

    # ------------------------------------------------------------------
    # HDF5 tensor precision
    # ------------------------------------------------------------------
    # Floating-point dtype used when saving P, Q, A and edge-feature tensors
    # to HDF5.  Accepts any torch dtype name understood by `torch.tensor`
    # constructor (e.g. 'float32', 'float16').  Default float16 halves the
    # storage requirement relative to float32 without significant loss for
    # QM-level datasets.
    hdf5_tensor_dtype: str = 'float16'

    # ------------------------------------------------------------------
    # Dataset split strategy
    # ------------------------------------------------------------------
    # Dataset split proportions for random train/valid/test splits.
    split_seed: int = 4489670
    # Number of folds for cross-validation (when experiment_type is 'kfold')
    k_folds: int = 5
    train_prop: float = 0.8
    valid_prop: float = 0.2

    # Macaque-specific: index (or indices) of the day(s) to include when running experiments
    macaque_day_index: Optional[Union[int, List[int]]] = None

    # Other data preparation parameters
    force_reload: bool = False  # whether to force reload of cached, pre-transformed data

    # DataLoader parameters
    num_workers: int = 4
    # batch_size: int = 128  # overridden by training.batch_size
    drop_last: bool = False
    pin_memory: bool = True  # doesn't work with sparse tensors in older versions of PyTorch
    using_pytorch_geo: bool = True

    # --------------------------------------------------------------
    # Optional: remove unused attributes before batching to GPU
    # --------------------------------------------------------------
    # E.g. some datasets ship attributes that VDW does not need.
    # Specify a list of attribute names to delete during the custom
    # collate function (batch-wise attachment path).  Keeps host/GPU
    # memory lower and reduces serialization overhead.
    attributes_to_drop: Optional[List[str]] = None

    # Optional: include additional attributes with predictions/targets in the model output dict
    # in case the loss function needs them
    attributes_to_include_with_preds: Optional[List[str]] = None  

    # Target subsetting
    target_include_indices: Optional[List[int]] = None  # Indices to include from the target (y) tensor

    # Target preprocessing
    # If 'mad_norm', each target property will be transformed as
    #   y_norm = (y - mean) / MAD,
    # where MAD = mean(|y - mean|). Stats are computed once on the
    # loaded dataset and stored in `target_preproc_stats`.
    target_preprocessing_type: Optional[Literal['mad_norm']] = None # 'None' means no preprocessing
    target_preproc_stats: Optional[Dict[str, Any]] = None  # {'mean': list, 'mad': list}

    # Ellipsoid-specific rotation settings
    rotate_test_set: bool = True  # Whether to rotate test set for equivariance testing
    vector_attribs_to_rotate: Optional[List[str]] = None  # List of vector attributes to rotate in test set

    # Whether to compute and store Euclidean edge distances into `edge_weight`
    # during dataset preparation when not already provided by the dataset.
    compute_edge_distances: bool = False

    # Line graph scattering
    line_scatter_feature_key: str = 'edge_scatter'
    vector_scatter_feature_key: str = 'vector_scatter'

    # Force-learning specific: when True, training will set requires_grad on
    # the coordinate attribute so that forces can be obtained via -∇E.
    requires_position_grad: bool = False

    
    def __post_init__(self):
        """Convert parameters to dataset kwargs."""
        self.dataset_kwargs = {}
        
        # Backward compatibility: migrate deprecated field if present
        # if (self.scalar_node_feats_include_indices is None) \
        # and (self.node_feats_include_indices is not None):
        #     self.scalar_node_feats_include_indices = self.node_feats_include_indices
        
        # Add dataset-specific parameters
        if self.dataset.lower() == 'ellipsoids':
            # For ellipsoids, we don't need additional kwargs as we load directly
            # But we can set some default parameters
            self.dataset_kwargs.update({
                'root': self.data_dir,  # Keep for compatibility
            })
            # Do not coerce target/task defaults here; respect YAML/CLI precedence
        # Add more dataset types here as needed

