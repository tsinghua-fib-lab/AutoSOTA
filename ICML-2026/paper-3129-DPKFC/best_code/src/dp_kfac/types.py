from typing import Dict
from dataclasses import dataclass
import torch

Tensor = torch.Tensor
CovarianceDict = Dict[str, Tensor]


@dataclass
class KFACConfig:
    damping: float = 1e-3
    cov_ema_decay: float = 0.95
    update_freq: int = 1
    precond_steps: int = 1
    pink_noise_alpha: float = 1.0
    alpha_schedule_start: float = 0.0  # 0 means disabled
    alpha_schedule_end: float = 0.0


@dataclass
class DPConfig:
    noise_multiplier: float
    max_grad_norm: float
    delta: float = 1e-5


@dataclass
class CovariancePair:
    A: CovarianceDict
    G: CovarianceDict
