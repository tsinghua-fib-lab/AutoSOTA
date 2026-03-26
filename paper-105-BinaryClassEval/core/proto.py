"""Protocol definitions and shared types for the core package."""
from __future__ import annotations
from typing import Protocol, Sequence, Tuple, Dict, Any, Union, Optional, Callable
import numpy as np

# Common type definitions
Number = Union[int, float]
ArrayLike = Sequence[Number]

# Shared data structures - importing from proto package to avoid creating duplicates
from proto.config import ComputationConfig, PlotConfig, SubgroupComputedData
from proto.subgroup import SubgroupResults

# Additional shared interfaces can be defined here
class CurveData(Protocol):
    """Protocol defining the interface for curve data objects."""
    log_odds_grid: np.ndarray
    nb_curve: np.ndarray
    
# Function type definitions
PrevalenceGridFunc = Callable[[], np.ndarray]
