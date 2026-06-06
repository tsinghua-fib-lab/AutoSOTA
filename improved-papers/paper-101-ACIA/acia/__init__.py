"""Anti-Causal Invariant Abstractions (ACIA) Library"""
__version__ = "0.1.0"

from .models.architectures import (
    CausalRepresentationNetwork,
    ImprovedCausalRepresentationNetwork,
    RMNISTModel,
    BallCausalModel,
    CamelyonModel
)

from .datasets.benchmarks import (
    ColoredMNIST,
    RotatedMNIST,
    BallAgentDataset,
    Camelyon17Dataset
)

# Comment out until we verify what's in training.py and evaluation.py
from .models.training import CausalOptimizer, ctrain_model
# from .metrics.evaluation import compute_metrics
# from .visualization.plots import visualize_cmnist_results

__all__ = [
    'CausalRepresentationNetwork',
    'ColoredMNIST',
    'RotatedMNIST',
    'BallCausalModel'
]