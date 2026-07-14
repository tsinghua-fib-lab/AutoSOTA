# config/arg_parsing.py
import argparse
from pathlib import Path
import os

from yaml import parser

def get_clargs():
    """
    Parse command-line arguments for training script, which
    override any parameters in a yaml config file or default 
    values.
    """
    parser = argparse.ArgumentParser(description='Training args')
    
    # Path and environment args
    parser.add_argument(
        '--root_dir', type=Path, 
        help="Root working directory for project."
    )

    parser.add_argument(
        '--config', type=str, required=False,
        help='Filepath of yaml configuration file after root_dir. If omitted when resuming'
             ' from a snapshot, the script will fall back to the original'
             ' config saved inside the experiment directory.'
    )

    parser.add_argument(
        '--experiment_id', type=str,
        help='Identifier for the experiment run'
    )

    parser.add_argument(
        '--results_save_subdir', type=str,
        help='Subdirectory under training.save_dir for this run (e.g., "ellipsoids/HERE")'
    )

    parser.add_argument(
        '--slurm', action='store_true',
        help='Running on SLURM'
    )

    # Data args
    parser.add_argument(
        '--dataset', type=str, required=False,
        help="Dataset key, e.g. 'peptides-func' etc. If omitted, the value from the YAML configuration is used."
    )
    parser.add_argument(
        '--processed_dataset_path', type=str, required=False,
        help="Path to a pickled, pre-processed dataset graph to load instead of rebuilding."
    )

    # Optional: directory with pretrained weights
    parser.add_argument(
        '--pretrained_weights_dir', type=str, required=False,
        help='Directory containing pretrained model weights (e.g. model.safetensors) to initialize the model.'
    )

    # Model args
    parser.add_argument(
        '--model_key', type=str, 
        help="Model key, e.g. 'vdw' etc."
    )
    parser.add_argument(
        '--model_mode', type=str,
        help="Model mode, e.g. 'handcrafted_scattering' or 'filter-combine', etc."
    )
    parser.add_argument(
        '--wavelet_scales_type', type=str,
        help="P-wavelet type: 'dyadic' or 'custom' or 'precalc'"
    )

    # Scattering args
    parser.add_argument(
        '--J', type=int, 
        help='Reference number of wavelets/diffusion steps, e.g. J=4 gives T=2**J=16'
    )
    parser.add_argument(
        '--use_dirac_nodes', action='store_true',
        help='Use Dirac nodes in the scattering layers'
    )

    # Training args
    parser.add_argument(
        '--n_epochs', type=int, 
        help='Max num. of epochs to run in each fold'
    )
    parser.add_argument(
        '--validate_every_n_epochs', type=int, 
        help='Run model validation every nth epoch'
    )
    parser.add_argument(
        '--burn_in', type=int, 
        help='Min num. of epochs to run before enforcing early stopping'
    )
    parser.add_argument(
        '--patience', type=int, 
        help='If args.STOP_RULE is no_improvement, max num. of epochs'
        'without improvement in validation loss'
    )
    parser.add_argument(
        '--learn_rate', type=float, 
        help='Learning rate hyperparameter'
    )
    parser.add_argument(
        '--batch_size', type=int, 
        help='Minibatch size hyperparameter'
    )
    # parser.add_argument('--use_k_drop', action='store_true')
    parser.add_argument(
        '--verbosity', type=int, 
        help='Integer controlling volume of print output during execution'
    )
    parser.add_argument(
        '--device', type=str,
        help='Device to use for training (e.g. cuda, cpu)'
    )
    # Flags that only set True when provided; otherwise leave as None so YAML/defaults apply
    parser.add_argument(
        '--save_best_model_state', action='store_const', const=True, default=None,
        help='If provided, enable saving the best model state during training'
    )
    parser.add_argument(
        '--save_final_model_state', action='store_const', const=True, default=None,
        help='If provided, enable saving the final model state after training'
    )
    parser.add_argument(
        '--dataloader_split_batches', action='store_true', default=True,
        help='In DDP, split batches into sub-batches to ensure each process'
             'sees the same number of batches.'
    )
    parser.add_argument(
        '--subsample_n', type=int,
        help='Subsample N samples from dataset'
    )
    # parser.add_argument(
    #     '--debug_subset_n', type=int,
    #     help='Temporarily use only N samples for debugging DDP (overrides subsample_n if set)'
    # )
    parser.add_argument(
        '--subsample_seed', type=int,
        help='Random seed for dataset subsampling'
    )
    parser.add_argument(
        '--invariant_pred', action='store_true',
        help='Use invariant prediction'
    )
    parser.add_argument(
        '--ablate_vector_track', action='store_true',
        help='Ablate vector track of VDW model'
    )
    parser.add_argument(
        '--ablate_scalar_track', action='store_true',
        help='Ablate scalar track of VDW model'
    )
    parser.add_argument(
        '--scalar_feat_key', type=str,
        help='Scalar feature key, e.g. "x" or "pos" (to treat a vector feature as separate scalar features)'
    )
    parser.add_argument(
        '--vector_feat_key', type=str,
        help='Vector feature key, e.g. "pos" or "v"'
    )
    parser.add_argument(
        '--task_key', type=str,
        help='Task key, e.g. "graph_regression" or "node_regression"'
    )
    parser.add_argument(
        '--target_key', type=str,
        help='Target key, e.g. "y" or "y_node"'
    )
    parser.add_argument(
        '--target_dim', type=int, default=None,
        help='Target dimension, e.g. 1 or 3'
    )
    parser.add_argument(
        '--k_folds', type=int, default=None,
        help='Number of folds for k-fold cross-validation (default: 5)'
    )
    parser.add_argument(
        '--experiment_type', type=str, choices=['tvt', 'kfold', 'stratified_kfold'], default=None,
        help="High-level experiment mode: 'tvt' for train/valid/test, 'kfold' for k-fold CV, 'stratified_kfold' when supported"
    )
    parser.add_argument(
        '--experiment_dir', type=str,
        help='Path to experiment directory (for testing: directory containing models/, config/, etc.)'
    )
    parser.add_argument(
        '--use_wandb_logging', action='store_true',
        help='Enable Weights & Biases (wandb) logging.'
    )
    parser.add_argument(
        '--wandb_offline', action='store_true',
        help='Use wandb offline mode (for SLURM/cluster).'
    )
    parser.add_argument(
        '--wandb_log_freq', type=int,
        help='Frequency (in steps/batches) for wandb.watch logging of gradients/weights.'
    )
    parser.add_argument(
        '--checkpoint_every', type=int,
        help='Save a training checkpoint snapshot every N epochs (0 to disable).'
    )
    parser.add_argument(
        '--snapshot_path', type=str,
        help='Full path to the snapshot directory (e.g. /path/to/models/best) or snapshot file.'
    )
    # Dataset preparation flags
    parser.add_argument(
        '--compute_edge_distances', action='store_true',
        help='If set, compute and store Euclidean edge distances as edge_weight for datasets lacking them.'
    )
    parsed_args = parser.parse_args()
    
    return parsed_args

