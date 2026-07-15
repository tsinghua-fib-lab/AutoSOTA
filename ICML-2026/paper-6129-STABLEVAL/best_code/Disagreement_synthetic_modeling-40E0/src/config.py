"""Configuration management for the synthetic study."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml


@dataclass
class Config:
    """Configuration for synthetic disagreement study."""
    
    # Default Agent configuration
    n_agents: int = 6
    n_items: int = 500
    agent_qualities: List[float] = field(
        default_factory=lambda: [0.85, 0.80, 0.70, 0.55, 0.35, 0.20]
    )
    
    # Default Annotator configuration
    n_annotators: int = 30
    labels_per_item: int = 5
    annotator_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "normal": 18,
            "strict": 6,
            "lenient": 4,
            "adversarial": 2
        }
    )
    
    # Item ambiguity
    hard_item_prob: float = 0.2
    easy_beta_params: List[float] = field(default_factory=lambda: [2, 12])
    hard_beta_params: List[float] = field(default_factory=lambda: [6, 3])
    
    # Label configuration
    class_labels: List[int] = field(default_factory=lambda: [0, 1, 2])
    credit_mapping: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    partial_correct_prob: float = 0.35
    
    # Simulation
    n_repetitions: int = 100
    n_stability_subsamples: int = 100
    subsample_labels: int = 3
    
    # Output
    save_raw_data: bool = True
    save_confusion_matrices: bool = True
    save_posteriors: bool = True
    output_dir: str = "results"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert len(self.agent_qualities) == self.n_agents, \
            f"agent_qualities length ({len(self.agent_qualities)}) must match n_agents ({self.n_agents})"
        
        total_annotators = sum(self.annotator_distribution.values())
        assert total_annotators == self.n_annotators, \
            f"annotator_distribution sum ({total_annotators}) must match n_annotators ({self.n_annotators})"
        
        assert self.labels_per_item <= self.n_annotators, \
            "labels_per_item cannot exceed n_annotators"
        
        assert len(self.credit_mapping) == len(self.class_labels), \
            "credit_mapping must match class_labels length"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "n_agents": self.n_agents,
            "n_items": self.n_items,
            "agent_qualities": self.agent_qualities,
            "n_annotators": self.n_annotators,
            "labels_per_item": self.labels_per_item,
            "annotator_distribution": self.annotator_distribution,
            "hard_item_prob": self.hard_item_prob,
            "easy_beta_params": self.easy_beta_params,
            "hard_beta_params": self.hard_beta_params,
            "class_labels": self.class_labels,
            "credit_mapping": self.credit_mapping,
            "partial_correct_prob": self.partial_correct_prob,
            "n_repetitions": self.n_repetitions,
            "n_stability_subsamples": self.n_stability_subsamples,
            "subsample_labels": self.subsample_labels,
            "save_raw_data": self.save_raw_data,
            "save_confusion_matrices": self.save_confusion_matrices,
            "save_posteriors": self.save_posteriors,
            "output_dir": self.output_dir,
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def default(cls) -> "Config":
        """Return default configuration."""
        return cls()


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if path is None:
        return Config.default()
    return Config.from_yaml(path)