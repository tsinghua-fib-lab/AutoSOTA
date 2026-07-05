"""
Search space models for pTNAS

Provides:
- PTNASMLP / PTNASResNet: torch_frame-based models for RelBench experiments
- DNNModel / NASBenchTabularSpace: VLDB-style models for NAS-Bench-Tabular experiments
- NASBenchEvaluator: Bridge connecting NASBenchTabularSpace with all proxy evaluators
- BaseSearchSpace: Abstract base for genetic operations

Note: GTMLP (ground truth query API) moved to tools/query_api.py
"""

from .mlp import PTNASMLP
from .resnet import PTNASResNet
from .block_mixed import PTNASBlockMixed
from .base import BaseSearchSpace
from .nas_bench_mlp import DNNModel, NASBenchTabularSpace, DATASET_CONFIGS
from .nas_bench_evaluator import NASBenchEvaluator

__all__ = [
    "PTNASMLP",
    "PTNASResNet",
    "PTNASBlockMixed",
    "BaseSearchSpace",
    "DNNModel",
    "NASBenchTabularSpace",
    "DATASET_CONFIGS",
    "NASBenchEvaluator",
]
