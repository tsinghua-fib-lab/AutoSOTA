from typing import Callable, Optional

import optax

from flax.struct import dataclass

from rp_ssm.types import LearningRate


@dataclass
class Config:
    """RP-GSSM config."""

    batch_size: int = 32
    num_iter: int = 1000
    seed: int = 0
    jit: bool = True
    beta_schedule: LearningRate = lambda i: 1.0
    prior_opt: Callable = optax.adam
    prior_lr: LearningRate = 1e-3
    act_lr: LearningRate = 1e-3
    rec_lr: tuple[LearningRate, ...] = (1e-3,)
    stabilize_A: Optional[str] = "scale"
    em: bool = False  # if True, perform EM (don't backprop through posterior)


@dataclass
class GenerativeConfig:
    """Generative model config."""

    batch_size: int = 32
    num_iter: int = 1000
    seed: int = 1  # must be different from Config seed
    jit: bool = True
    lr: LearningRate = 1e-3
    num_samples: int = 10  # number of latent samples used to estimate expectation for each datapoint