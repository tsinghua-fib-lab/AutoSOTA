"""KV Cache utilities for Qwen3-VL text model.

This module provides a modular pipeline for KV cache recomputation:

1. VLMKVCacheExtractor: Extract KV cache from VLM inputs (image + text)
2. ImportanceScorer: Compute importance scores for each token position
3. KVCacheRecomputer: Selectively recompute KV at important positions
4. KVCacheInference: Generate with recomputed cache

Example usage:
    from models.qwen.kv_cache import (
        VLMKVCacheExtractor,
        ImportanceScorer,
        KVCacheRecomputer,
        KVCacheInference,
        RecomputeConfig,
    )

    # Extract KV cache
    extractor = VLMKVCacheExtractor(model)
    kv_data = extractor.extract(inputs)

    # Compute importance scores
    scorer = ImportanceScorer(model, method="norm")
    scores = scorer.compute(kv_data)

    # Select positions and recompute
    config = RecomputeConfig(recompute_ratio=0.15)
    indices = scorer.select_positions(scores, config)

    recomputer = KVCacheRecomputer(model)
    updated_cache = recomputer.recompute(kv_data, indices)

    # Generate
    inference = KVCacheInference(model, processor)
    result = inference.generate(updated_cache, query, kv_data.seq_len)
"""

from .base import KVCacheData, ImportanceScores, RecomputeConfig
from .extractor import VLMKVCacheExtractor
from .importance_scorer import ImportanceScorer
from .recomputer import KVCacheRecomputer
from .inference import KVCacheInference
from .chunker import ImageChunker, ChunkInfo
from .chunk_prefiller import ChunkPrefiller, KVCacheConcatenator, ChunkedKVCache

__all__ = [
    # Data containers
    "KVCacheData",
    "ImportanceScores",
    "RecomputeConfig",
    "ChunkInfo",
    "ChunkedKVCache",
    # Pipeline components
    "VLMKVCacheExtractor",
    "ImportanceScorer",
    "KVCacheRecomputer",
    "KVCacheInference",
    "ImageChunker",
    "ChunkPrefiller",
    "KVCacheConcatenator",
]
