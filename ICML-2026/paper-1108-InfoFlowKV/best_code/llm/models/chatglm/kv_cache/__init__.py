"""KV Cache utilities for ChatGLM model.

This module provides a modular pipeline for KV cache recomputation:

1. KVCacheExtractor: Extract KV cache from text passages
2. ImportanceScorer: Compute importance scores (VATP, entropy, norm, etc.)
3. KVCacheRecomputer: Selectively recompute KV at important positions
4. KVCacheInference: Generate with recomputed cache

Example usage:
    from models.chatglm.kv_cache import (
        KVCacheExtractor,
        ImportanceScorer,
        KVCacheRecomputer,
        KVCacheInference,
        RecomputeConfig,
    )

    # Extract KV cache
    extractor = KVCacheExtractor(model, tokenizer, model_type="glm")
    kv_data = extractor.extract_full_context(context)

    # Compute importance scores
    scorer = ImportanceScorer(model, method="vatp")
    scores = scorer.compute(attention_weights=attn_weights, past_key_values=kv_data.past_key_values)

    # Select positions and recompute
    config = RecomputeConfig(recompute_ratio=0.15, method="vatp")
    indices = scorer.select_positions(scores, config)

    recomputer = KVCacheRecomputer(model, tokenizer, model_type="qwen")
    updated_kv = recomputer.recompute_at_positions(kv_data, indices)

    # Generate
    inference = KVCacheInference(model, tokenizer)
    result, metrics = inference.generate(query, updated_kv)
"""

from .base import KVCacheData, ImportanceScores, RecomputeConfig
from .extractor import KVCacheExtractor
from .importance_scorer import ImportanceScorer
from .recomputer import KVCacheRecomputer
from .inference import KVCacheInference

__all__ = [
    # Data containers
    "KVCacheData",
    "ImportanceScores",
    "RecomputeConfig",
    # Pipeline components
    "KVCacheExtractor",
    "ImportanceScorer",
    "KVCacheRecomputer",
    "KVCacheInference",
]
