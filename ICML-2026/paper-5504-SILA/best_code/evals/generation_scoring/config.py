#!/usr/bin/env python3
"""
Configuration classes for SAE Reflective Coherence Evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the base language model and SAE."""

    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    sae_release: str = "goodfire-llama-3.1-8b-instruct"  # SAELens release identifier
    sae_id: str = "layer_19"  # SAE identifier within the release
    sae_layer_number: int = 19  # Layer number (for master JSON lookups and hooks)
    device: str = "cuda"  # Device for model: "cuda" for single GPU, "auto" for multi-GPU distribution
    dtype: str = "bfloat16"

    @property
    def sae_layer(self) -> str:
        """Generate sae_layer string from sae_layer_number."""
        return f"model.layers.{self.sae_layer_number}"


@dataclass
class RewardConfig:
    """Configuration for the reward system."""

    # Prompt template - the label will be substituted for the underscore
    conversation_prompt_template: str = (
        "Produce a VERY SHORT conversation which exhibits '_'\n"
        "Do not include any other text in your response. Start immediately with the conversation."
    )

    # Chat formatting for instruction-tuned models
    use_chat_template: bool = True
    conversation_system_message: Optional[str] = (
        "You are a helpful AI assistant who generates EXTREMELY SHORT example conversations. "
        "The conversations are between a user and an assistant, and have the following format:\n"
        "[USER] I'm a user.\n[ASSISTANT] I'm the assistant."
    )

    # Text generation parameters
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9

    # Memory management
    aggressive_memory_cleanup: bool = True

    # Logging configuration
    num_debug_samples: int = 4
    full_debug_mode: bool = (
        False  # If True, shows ALL completions; if False, limits to num_debug_samples
    )


@dataclass
class LabelGeneratorConfig:
    """Configuration for the label generator architecture."""

    num_soft_tokens: int = 1
    template: str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        'What is the meaning of "<|reserved_special_token_0|>"?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        'The meaning of "<|reserved_special_token_0|>" is "'
    )
    max_generation_length: int = 30

    # Vector selection: "encoder" or "decoder"
    which_vector: str = "decoder"

    # Adapter checkpoint (optional - None uses identity projection for zero-bias baseline)
    adapter_checkpoint_path: Optional[str] = None  # Path to SelfIE adapter checkpoint (.pt file), or None for identity (zero bias)

    # Vector preprocessing (applied before scaling, then passed to adapter)
    # NOTE: The adapter wrapper automatically disables its internal normalization during
    # evaluation to prevent it from undoing the scale_values. This means normalization
    # happens ONCE (here, before scaling) rather than twice.
    normalize_vectors: bool = True  # Normalize SAE vectors to unit length before scaling

    # Generation parameters
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    # Label extraction method
    # If True, strips the last double quote from generated text (allows quotes inside labels)
    # If False, takes text up to first double quote (legacy behavior)
    strip_last_quote: bool = True
    
    # Reserved token for soft prompt injection
    # This token will be replaced with the soft prompt embeddings
    reserved_token: str = "<|reserved_special_token_0|>"


@dataclass
class EvaluationConfig:
    """Unified configuration for reflective coherence evaluation."""

    # Core configuration objects
    model_config: ModelConfig
    reward_config: RewardConfig
    
    # Master JSON configuration (required - we only support this format now)
    master_json_path: str  # Path to master JSON file
    master_json_dataset_name: str  # Dataset name to use from master JSON
    master_json_layer: int  # Layer number to use from master JSON
    
    # Optional fields with defaults
    label_generator_config: Optional[LabelGeneratorConfig] = None

    # Evaluation mode: "label_dataset", "label_generator", or "label_generation_only"
    evaluation_mode: str = "label_dataset"

    # Master JSON split and data path
    master_json_split: str = "val"  # Which split to use: "train", "val", "test", or "all"
    data_volume_path: str = "/sae-data"  # Base path where .pt files are located
    
    # Subsampling (optional - use None to load all data)
    max_latents: Optional[int] = None  # Maximum number of latents to load (None = all)
    specific_latent_indices: Optional[List[int]] = None  # Specific latent indices to evaluate (None = all, overrides max_latents)

    # Label generator specific settings (only used when evaluation_mode == "label_generator" or "label_generation_only")
    scale_values: List[float] = field(
        default_factory=lambda: [1.0]
    )  # Projection scales to try
    num_labels_per_scale: int = 1  # Number of labels to generate per scale value

    # Reward evaluation settings
    num_reward_samples: int = 1  # Number of reward samples per label (N >= 1)

    # Batching configuration
    label_generation_batch_size: int = 32  # Batch size for label generation
    reward_evaluation_batch_size: int = 16  # Batch size for reward evaluation

    # Parallelization settings
    num_parallel_instances: int = 1
    checkpoint_every_n_latents: int = 100  # Save progress every N latents

    # Output configuration
    output_volume_path: str = "/results"  # Path on Modal volume to save results
    run_id: Optional[str] = None  # Shared run ID for identifying shards from same run

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.evaluation_mode not in ["label_dataset", "label_generator", "label_generation_only"]:
            raise ValueError(
                f"evaluation_mode must be 'label_dataset', 'label_generator', or 'label_generation_only', got {self.evaluation_mode}"
            )

        if self.evaluation_mode in ["label_generator", "label_generation_only"]:
            if self.label_generator_config is None:
                raise ValueError(
                    "label_generator_config is required when evaluation_mode is 'label_generator' or 'label_generation_only'"
                )
            if len(self.scale_values) == 0:
                raise ValueError("scale_values must contain at least one value")

        if self.num_reward_samples < 1:
            raise ValueError("num_reward_samples must be >= 1")
        if self.num_labels_per_scale < 1:
            raise ValueError("num_labels_per_scale must be >= 1")
        if self.num_parallel_instances < 1:
            raise ValueError("num_parallel_instances must be >= 1")
        
        if self.master_json_split not in ["train", "val", "test", "all"]:
            raise ValueError(
                f"master_json_split must be 'train', 'val', 'test', or 'all', got {self.master_json_split}"
            )
        
        if self.specific_latent_indices is not None and self.max_latents is not None:
            print("⚠️  Warning: both specific_latent_indices and max_latents are set. specific_latent_indices will take precedence.")

