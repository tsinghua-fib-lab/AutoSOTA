from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class PLATEConfig(PeftConfig):
    """
    Configuration class for PLATE (Plasticity-Tunable Efficient Adapters).
    
    PLATE combines neuron selection with orthogonal input basis:
    - B: Frozen neuron selection matrix (from cosine similarity)
    - A: Trainable transformation matrix
    - Q_in: Frozen orthogonal input basis (from SRHT)
    
    The forward pass is: y = Wx + scaling * B @ A @ Q_in^T @ x
    
    Args:
        r (`int`):
            The number of selected trainable neurons.
        target_modules (`Union[List[str], str]`): 
            The names of the modules to apply PLATE to.
        plate_alpha (`float`): 
            The scaling hyperparameter for PLATE. Used in adaptive alpha computation.
        plate_dropout (`float`): 
            The dropout probability for PLATE layers.
        col_tau (`float`):
            Threshold for feature/column redundancy (used for Q_in computation).
            Must be in [0, 1]. Higher tau → smaller orthogonal complement → lower input rank.
        max_rank (`int`):
            Maximum rank for Q_in dimension.
        fan_in_fan_out (`bool`): 
            Set to True if layer stores weight like (fan_in, fan_out).
        bias (`str`): 
            Bias type for PLATE. Can be 'none', 'all' or 'plate_only'.
        modules_to_save (`List[str]`):
            List of modules to be set as trainable and saved.
        init_lora_weights (`bool`):
            Whether to initialize the weights with their default initialization.
        layers_to_transform (`Union[List[int], int]`):
            The layer indexes to transform.
        layers_pattern (`str`):
            The layer pattern name.
        rank_pattern (`dict`):
            Mapping from layer names to rank values.
        alpha_pattern (`dict`):
            Mapping from layer names to alpha values.
    """
    r: int = field(default=8, metadata={"help": "PLATE number of selected trainable neurons"})
    target_modules: Optional[Union[List[str], str, set]] = field(
        default=None,
        metadata={"help": "List of module names to replace with PLATE"}
    )
    plate_alpha: float = field(default=0.5, metadata={"help": "PLATE scaling factor (alpha) for adapter output"})
    plate_dropout: float = field(default=0.0, metadata={"help": "PLATE dropout"})
    
    # SRHT-based Q_in configuration
    num_hutch_probes: int = field(
        default=4,
        metadata={"help": "Number of random probes for Hutch++ (4 for speed, 16 for quality)"}
    )
    pool_multiple: int = field(
        default=4,
        metadata={"help": "Candidate pool size multiplier (4 optimal)"}
    )
    
    # Input subspace estimation threshold
    col_tau: float = field(default=0.90, metadata={"help": "Feature/column redundancy threshold for Q_in"})
    max_rank: int = field(default=512, metadata={"help": "Maximum rank for Q_in dimension"})
    
    # Additional PEFT config fields
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if layer stores weight like (fan_in, fan_out)"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type: 'none', 'all', or 'plate_only'"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Modules to set as trainable and save"}
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize weights"}
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={"help": "Layer indexes to transform"}
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Layer pattern name"}
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Mapping from layer names to rank values"}
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Mapping from layer names to alpha values"}
    )
    exclude_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Module names to exclude"}
    )
    col_tau_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Mapping from layer names to col_tau values for per-layer energy thresholds"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.PLATE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # Ensure pattern dicts exist
        self.rank_pattern = self.rank_pattern or {}
        self.alpha_pattern = self.alpha_pattern or {}
        self.col_tau_pattern = self.col_tau_pattern or {}

        # Validate col_tau is in [0, 1]
        if not 0 <= self.col_tau <= 1:
            raise ValueError(
                f"`col_tau` must be in [0, 1], got {self.col_tau}. "
                f"col_tau represents the energy threshold for column space selection."
            )

