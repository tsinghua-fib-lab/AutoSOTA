#!/usr/bin/env python3
"""Configuration system for SelfIE adapter training."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for the language model."""
    
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_map: str | Dict[str, Any] = "auto"  # "auto" for multi-GPU, "cuda" for single GPU
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    enable_gradient_checkpointing: bool = True  # Memory efficiency for large models
    load_in_8bit: bool = False  # Use 8-bit quantization (requires bitsandbytes)


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    labels_file: str = "labels.json"
    batch_size: int = 80
    shuffle: bool = True
    num_workers: int = 4
    strip_labels: bool = False  # Whether to strip whitespace from labels
    eos_token: str = "<|eot_id|>"  # EOS token to append to labels
    
    # Dataset mixing with ratios
    train_dataset_ratios: Optional[dict[str, float] | list[dict]] = None
    val_dataset_ratios: Optional[dict[str, float] | list[dict]] = None
    resample_each_epoch: bool = True
    debug_dataset_mixing: bool = False


@dataclass
class ProjectionConfig:
    """Configuration for projection architecture."""
    
    type: str = "scalar_affine"  # "scale_only", "scalar_affine", "full_rank", "scalar_affine_plus_low_rank", "low_rank_only"
    normalize_input: bool = True  # Whether to L2-normalize input vectors
    init_scale: float = 30.0  # Initial scale value
    
    # Low-rank specific
    low_rank_rank: Optional[int] = None  # e.g., 64, 128
    low_rank_init_factor: float = 0.01  # Factor for low-rank initialization std


@dataclass
class SoftPromptConfig:
    """Configuration for soft prompt template."""
    
    # Default Llama template:
    template: str = str(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        'What is the meaning of "<|reserved_special_token_0|>"?<|eot_id|>'
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        'The meaning of "<|reserved_special_token_0|>" is "'
    )
    reserved_token: str = "<|reserved_special_token_0|>"  # Token to replace with soft vectors


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Learning rates
    learning_rate: float = 1e-2
    scale_learning_rate: Optional[float] = None
    direction_learning_rate: Optional[float] = None
    bias_learning_rate: Optional[float] = None
    
    # Optimizer
    optimizer_type: str = "adamw"  # "adamw" or "sgd_momentum"
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD with momentum
    
    # Training duration
    num_epochs: int = 2
    max_steps: Optional[int] = None
    
    # Gradient settings
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 0.5
    
    # Loss
    label_smoothing: float = 0.0
    max_loss: float = 100.0
    
    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 10
    
    # Validation
    validation_every_n_steps: int = 50
    val_fraction: float = 1.0
    
    # Checkpointing
    checkpoint_every_n_steps: int = 100
    checkpoint_dir: str = "./checkpoints"
    save_final_checkpoint: bool = True
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # MLflow configuration (preferred)
    use_mlflow: bool = False  # Set to True and configure tracking_uri to use MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"  # Your MLflow tracking server
    mlflow_experiment_name: str = "selfie-adapter-training"
    
    # WandB configuration (alternative)
    use_wandb: bool = True
    wandb_project: str = "selfie-adapter-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 1
    
    # Sample generation logging
    log_sample_generations: int = 5
    log_generations_every_n_steps: int = 100
    
    # Singular value logging
    log_singular_values_every_n_steps: int = 50


@dataclass
class Config:
    """Main configuration class for SelfIE adapter training."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    soft_prompt: SoftPromptConfig = field(default_factory=SoftPromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    seed: int = 42
    experiment_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        config_dict = asdict(self)
        
        def convert_tuple_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    if isinstance(key, tuple):
                        new_key = str(list(key))
                    else:
                        new_key = key
                    new_dict[new_key] = convert_tuple_keys(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_tuple_keys(item) for item in obj]
            else:
                return obj
        
        return convert_tuple_keys(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Helper to convert list format to tuple-key dict format
        def normalize_dataset_ratios(ratios):
            if ratios is None:
                return None
            
            if isinstance(ratios, dict):
                return ratios
            
            if isinstance(ratios, list):
                result = {}
                for item in ratios:
                    if not isinstance(item, dict) or "datasets" not in item or "ratio" not in item:
                        raise ValueError(
                            "List format for dataset_ratios must contain dicts with 'datasets' and 'ratio' keys."
                        )
                    datasets = item["datasets"]
                    ratio = item["ratio"]
                    
                    if isinstance(datasets, str):
                        key = datasets
                    elif isinstance(datasets, list):
                        key = tuple(datasets)
                    else:
                        raise ValueError(f"'datasets' must be string or list, got {type(datasets)}")
                    
                    result[key] = ratio
                return result
            
            raise ValueError(f"dataset_ratios must be dict or list, got {type(ratios)}")
        
        # Normalize dataset ratios
        data_dict = config_dict.get("data", {})
        if "train_dataset_ratios" in data_dict:
            data_dict["train_dataset_ratios"] = normalize_dataset_ratios(data_dict["train_dataset_ratios"])
        if "val_dataset_ratios" in data_dict:
            data_dict["val_dataset_ratios"] = normalize_dataset_ratios(data_dict["val_dataset_ratios"])
        
        # Parse nested configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**data_dict)
        projection_config = ProjectionConfig(**config_dict.get("projection", {}))
        soft_prompt_config = SoftPromptConfig(**config_dict.get("soft_prompt", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        return cls(
            model=model_config,
            data=data_config,
            projection=projection_config,
            soft_prompt=soft_prompt_config,
            training=training_config,
            logging=logging_config,
            seed=config_dict.get("seed", 42),
            experiment_name=config_dict.get("experiment_name"),
        )
