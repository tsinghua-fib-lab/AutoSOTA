"""
Distributed guided recompute with Ring Attention sequence parallelism.

This module provides parallel KV cache extraction, scoring, and recomputation
using Ring Attention for efficient communication.

Components:
    - DistributedConfig: Configuration for sequence parallel inference
    - DistributedExtractor: Parallel KV cache extraction
    - DistributedScorer: Distributed importance scoring with minimal communication
    - RingAttentionRecomputer: Recomputation using ring attention

Usage:
    from models.parallel import (
        DistributedConfig,
        DistributedExtractor,
        DistributedScorer,
        RingAttentionRecomputer,
    )
"""

from .config import DistributedConfig
from .extractor import DistributedExtractor
from .scorer import DistributedScorer
from .recomputer import RingAttentionRecomputer

__all__ = [
    "DistributedConfig",
    "DistributedExtractor",
    "DistributedScorer",
    "RingAttentionRecomputer",
]
