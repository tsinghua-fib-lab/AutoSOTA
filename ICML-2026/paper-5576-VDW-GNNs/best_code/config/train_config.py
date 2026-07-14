# config/train_config.py
"""
This file contains the configuration for model training. 
Note that this class combines nested configuration classes 
(for the dataset, model, optimizer, and scheduler), so that 
only the top-level parameters need to be specified on the 
command line or in a yaml config file, and a single config 
object can be passed to the main training script.

"""
from dataclasses import dataclass, field
import warnings
from typing import Type, Optional, Tuple, Literal
from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from .optimizer_config import OptimizerConfig
from .scheduler_config import SchedulerConfig


@dataclass
class TrainingConfig:
    # Optional project root directory used to prefix relative save/data paths
    root_dir: Optional[str] = None

    # Subdirectory (relative to root_dir) where YAML configs live; used to
    # resolve relative --config paths passed on the command line.
    config_subdir: str = "code/config/yaml_files"

    # Base directory for all experiments
    save_dir: str = "experiments"

    # Directory paths (will be set by os_utilities.create_experiment_dir)
    config_save_path: Optional[str] = None
    experiment_id: Optional[str] = None  # If None, will be auto-generated
    model_save_dir: Optional[str] = None  # dir to save model snapshots and weights; if None, will be set to the 'models' subdirectory of the experiment directory; NOTE: if resuming from a snapshot, this will be set to the parent directory of the snapshot
    train_logs_save_dir: Optional[str] = None  # dir to save training logs
    results_save_dir: Optional[str] = None
    grad_track_save_path: Optional[str] = None

    # Optional directory that holds pretrained model weights to initialize a new model
    pretrained_weights_dir: Optional[str] = None
    
    # Output parameters
    verbosity: int = 0  # verbosity level of print output
    train_print_filename: Optional[str] = "train_print.txt"  # text file to save training print output
    train_logs_filename: Optional[str] = "train_logs.pkl"  # file to save epoch-by-epoch training measures
    results_filename: Optional[str] = "results.pkl"  # file to save results
    save_embeddings: bool = False  # whether to persist embeddings per split (pretrain mode only)

    # Resuming, logging, and saving parameters
    snapshot_name: Optional[str] = None  # filename of snapshot to resume training from, e.g. 'best' or 'last'
    # Which model state to return from the training loop:
    #   - 'best': load weights from the epoch with the best main_metric score
    #   - 'final': keep weights from the last epoch (no best-weight reload)
    return_which_model: Literal['best', 'final'] = 'best'
    # Deprecated legacy flag kept for backward compatibility with older YAML configs.
    # When explicitly set:
    #   - True  -> treated as return_which_model = 'best'
    #   - False -> treated as return_which_model = 'final'
    return_best: Optional[bool] = None
    save_best_model_state: bool = True  # whether to save the best model state during training
    save_final_model_state: bool = False  # whether to save the final model state (default: do not save)
    grad_track_param_names: Optional[Tuple[str]] = None
    save_grads: bool = False  # whether to save gradients (as opposed to just weights)

    # Checkpointing
    checkpoint_every: int = 0  # Save a checkpoint snapshot every N epochs (0 disables)

    # Dataset configuration
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)

    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # Optimizer configuration
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Scheduler configuration
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Training parameters
    experiment_type: Literal['tvt', 'kfold', 'stratified_kfold'] = 'tvt'
    # Inference regime for datasets that support transductive vs inductive graph construction
    experiment_inference_type: Literal['transductive', 'inductive'] = 'inductive'
    k_folds: Optional[int] = 5  # Only used in k-fold cross-validation experiments
    use_cv_test_set: bool = False  # Whether to use a test fold in cross-validation
    n_epochs: int = 100
    # Optional seed used for deterministic rotation equivariance evaluation
    rotation_seed: Optional[int] = None
    # Deprecated: use min_epochs and scheduler_burnin instead. Kept for backward compatibility.
    burnin_n_epochs: int = 10
    # Minimum number of epochs to run before enforcing the stopping rule.
    min_epochs: int = 10
    # Minimum number of epochs to run before the learning-rate scheduler is allowed to reduce LR.
    scheduler_burnin: int = 10
    # Number of warmup epochs to linearly ramp LR from 1e-6 to optimizer learn_rate.
    warmup_epochs: int = 0
    main_metric_is_better: Literal['lower', 'higher'] = 'lower'
    main_metric_rel_improv_thresh: Optional[float] = 0.001  # relative improvement threshold for stopping training
    main_metric: str = 'loss_valid'
    # Patience (in epochs) for early-stopping when the *main metric* does not
    # improve.  The counter resets whenever ``main_metric`` achieves a new
    # best value; training stops if the metric has not improved for this many
    # epochs after the burn-in period.
    no_valid_metric_improve_patience: int = 32
    stop_rule: Optional[str] = 'no_improv'

    # Loss function parameters
    #   'mse'   – mean-squared error (default)
    #   'l1'    – mean absolute error
    #   'huber' – Huber / Smooth-L1 loss controlled by ``huber_delta``
    loss_fn: Literal['mse', 'l1', 'huber'] = 'mse'
    # Threshold parameter (delta / beta) for the Huber loss.  Ignored for other losses.
    huber_delta: float = 1.0

    # Data handling and performance parameters
    using_pytorch_geo: bool = True  # whether using PyTorch Geometric data
    use_cuda_streams: bool = True  # whether to use CUDA streams (if available)
    non_blocking: bool = True  # whether to use non-blocking data transfers

    # DataLoader parameters
    batch_size: int = 128
    valid_batch_size: int = 128
    test_batch_size: int = 128
    drop_last: bool = True # helps sync GPUs in DDP by ensuring each gets the same number of batches per epoch
    num_workers: int = 0
    pin_memory: bool = True
    dataloader_num_workers: int = 2  # Number of workers for DataLoader, separate from dataset config

    # --------------------------------------------------
    # Ablation settings
    # --------------------------------------------------
    ablate_scalar_embedding: bool = False
    ablate_scalar_wavelet_batch_norm: bool = False
    ablate_vector_wavelet_batch_norm: bool = True
    ablate_second_order_wavelets: bool = False
    
    # When True in cross-track mode, disables the vector track across the entire
    # pipeline.  The training pipeline concatenates vector coordinates to scalar
    # node features and omits the Q operator.
    ablate_vector_track: bool = False

    # When True, disables the *scalar* track in VDW cross-track mode. The training
    # pipeline omits scalar node features and skips loading the diffusion operator
    # ``P`` so that only the vector track (coordinates) is used.
    ablate_scalar_track: bool = False

    # Training precision parameters
    validate_every: int = 1
    max_grad_norm: Optional[float] = None # 1.0
    mixed_precision: Literal['none', 'fp16', 'bf16'] = 'none'  # sparse cuda operations probs don't support half precisions

    # Batch normalization
    # Freeze (i.e. set to eval mode, stop running-stat updates) all BatchNorm layers
    # after this epoch number.  Set to 0 or a negative value to *disable* freezing.
    batch_norm_freeze_epoch: Optional[int] = None
    
    # Device and distributed training parameters
    device: str = 'cuda'
    dataloader_split_batches: bool = True # keep this True for DDP, to simplify global reduction operations (e.g., loss across batches) used in the control flow of train.py
    gradient_accumulation_steps: int = 1
    distributed_type: Literal['none', 'multi_gpu'] = 'none'  # Type of distributed training
    num_processes: Optional[int] = None
    local_rank: Optional[int] = None  # Local rank for distributed training
    world_size: Optional[int] = None  # World size for distributed training

    # DistributedDataParallel behavior
    ddp_find_unused_parameters: bool = False

    # Weights & Biases (wandb) logging options
    use_wandb_logging: bool = False  # Enable wandb logging of losses/metrics/gradients
    wandb_offline: bool = False  # Use wandb offline mode (for SLURM/cluster)
    wandb_log_freq: int = 2048  # Frequency (in steps/batches) for wandb.watch logging of gradients/weights

    # Parameter validation steps
    def __post_init__(self):
        """Validate configuration values."""
        # Backward compat: if legacy burnin_n_epochs is set via YAML/CLI and
        # the new fields are left at their defaults, mirror the legacy value.
        try:
            if (
                isinstance(self.burnin_n_epochs, int)
                and self.burnin_n_epochs is not None
            ):
                if (
                    'min_epochs' in self.__dict__ and 'scheduler_burnin' in self.__dict__
                    and (self.min_epochs == 10 and self.scheduler_burnin == 10)
                    and self.burnin_n_epochs != 10
                ):
                    # Only override defaults to avoid clobbering explicit values
                    self.min_epochs = int(self.burnin_n_epochs)
                    self.scheduler_burnin = int(self.burnin_n_epochs)
        except Exception:
            pass

        # Backward compatibility: map legacy return_best to return_which_model when provided.
        try:
            if self.return_best is not None:
                if self.return_best is False and self.return_which_model == 'best':
                    # User explicitly disabled returning the best model in legacy configs:
                    # interpret as "return final weights" under the new interface.
                    self.return_which_model = 'final'
                # If return_best is True, the default 'best' behavior already matches.
        except Exception:
            # Best-effort only; do not hard-fail on legacy field handling.
            pass

        # Basic validations
        if self.min_epochs > self.n_epochs:
            warnings.warn(
                f"The minimum number of epochs chosen ({self.min_epochs})"
                f" is greater than maximum number of training epochs ({self.n_epochs}).",
                category=UserWarning
            )

        if self.scheduler_burnin > self.n_epochs:
            warnings.warn(
                f"The scheduler burn-in number of epochs chosen ({self.scheduler_burnin})"
                f" is greater than maximum number of training epochs ({self.n_epochs}).",
                category=UserWarning
            )
        
        if self.main_metric_is_better not in ['lower', 'higher']:
            raise ValueError("main_metric_is_better must be either 'lower' or 'higher'")

        # Validate return_which_model
        if self.return_which_model not in ('best', 'final'):
            warnings.warn(
                f"Invalid value for return_which_model ('{self.return_which_model}'); "
                f"falling back to 'best'.",
                category=UserWarning
            )
            self.return_which_model = 'best'
        
        if self.main_metric_rel_improv_thresh is not None and self.main_metric_rel_improv_thresh <= 0:
            raise ValueError("main_metric_rel_improv_thresh must be positive")
        
        if self.mixed_precision not in ['none', 'fp16', 'bf16']:
            raise ValueError(f"mixed_precision must be one of: 'none', 'fp16', 'bf16' (got {self.mixed_precision})")
            
        # if (self.save_final_model_state) \
        # and (self.model_save_dir is None):
        #     raise ValueError(
        #         "'model_save_dir' cannot be None when 'save_final_model_state' is True. "
        #         "Please provide a valid directory path for 'model_save_dir'."
        #     )
        
        if self.distributed_type == 'none':
            # Convert 'none' to 'no' for accelerate
            self.distributed_type = 'no'
