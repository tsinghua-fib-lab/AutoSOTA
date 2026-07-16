import logging
import time
from dataclasses import dataclass

import numpy as np
from dysts.base import BaseDyn

logger = logging.getLogger(__name__)


@dataclass
class TimeLimitEvent:
    """
    Event to check if integration is taking too long
    """

    system: BaseDyn
    max_duration: float
    terminal: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def __call__(self, t, y):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            if self.verbose:
                logger.warning(
                    f"{self.system.name} exceeded time limit: {elapsed_time:.2f}s > {self.max_duration:.2f}s"
                )
            return 0
        return 1


@dataclass
class InstabilityEvent:
    """
    Event to detect instability during numerical integration

    Ignores unbounded indices from the system
    """

    system: BaseDyn
    threshold: float
    terminal: bool = True
    verbose: bool = False

    def __call__(self, t, y):
        bounded_coords = np.abs(np.delete(y, self.system.unbounded_indices))
        if np.any(bounded_coords > self.threshold) or np.any(np.isnan(y)):
            if self.verbose:
                logger.warning(
                    f"{self.system.name} instability @ t={t:.3f}: {np.abs(y).max():.3e} > {self.threshold:.3e}"
                )
            return 0
        return 1


@dataclass
class TimeStepEvent:
    """Event that terminates integration when step size becomes too small"""

    system: BaseDyn
    min_step: float = 1e-10  # Aligned with typical atol values
    terminal: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.last_t = float("inf")

    def __call__(self, t, y):
        dt = abs(t - self.last_t)
        if dt < self.min_step:
            if self.verbose:
                logger.warning(
                    f"{self.system.name} integration terminated: step size {dt:.3e} < {self.min_step:.3e}"
                )
            return 0
        self.last_t = t
        return 1
