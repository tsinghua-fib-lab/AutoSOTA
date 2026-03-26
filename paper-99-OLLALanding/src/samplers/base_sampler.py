# OLLA_NIPS/src/samplers/base_sampler.py

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict

class BaseSampler(ABC):
    """
    Abstract base class for all sampling algorithms.
    """
    def __init__(
        self,
        step_size: float,
        num_steps: int,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.step_size = step_size
        self.num_steps = num_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed if seed is not None else torch.initial_seed())
        # Store any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def sample(
        self,
        x0: np.ndarray,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Main sampling method to be implemented by subclasses.
        """
        pass

