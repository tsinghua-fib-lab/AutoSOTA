"""Configuration defaults for TAP."""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class DataConfig:
    target_column: str = "target"
    task_type: str = "classification"


@dataclass
class GeneratorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MaskTemplateConfig:
    templates: Dict[str, Dict] = field(
        default_factory=lambda: {
            "explore": {"description": "generate most columns around selected anchors"},
            "conservative": {"description": "keep important columns closer to anchors"},
        }
    )


@dataclass
class InpaintConfig:
    anchor_rules: List[str] = field(
        default_factory=lambda: ["high_uncertainty", "high_error", "minority_class", "random"]
    )
    samples_per_step: int = 32
    samples_per_anchor: int = 4
    label_margin_threshold: float = 0.1
    label_p_min: float = 0.3
    num_bins: int = 7
    residual_threshold_percentile: float = 95
    commit_interval: int = 10


@dataclass
class KTOConfig:
    policy_lr: float = 3e-4
    beta_kto: float = 3.0
    lambda_D: float = 1.0
    lambda_U: float = 1.0
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    num_layers: int = 2
    ig_window_size: int = 200
    ig_quantile_q: float = 0.6
    ig_min_buf: int = 10
    tau_warmup: float = 0.0


@dataclass
class TrainConfig:
    num_steps: int = 100
    log_every: int = 10
    checkpoint_dir: str = "runs/checkpoints"
    log_dir: str = "runs/logs"
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    mask_template: MaskTemplateConfig = field(default_factory=MaskTemplateConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    kto: KTOConfig = field(default_factory=KTOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_default_config() -> Config:
    return Config()
