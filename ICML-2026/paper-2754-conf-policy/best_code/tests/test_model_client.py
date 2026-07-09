import torch
from transformers.cache_utils import DynamicCache

from cpc_llm.core.model_client import ModelClient

# Synthetic cache dimensions
NUM_LAYERS = 2
NUM_HEADS = 4
SEQ_LEN = 3
HEAD_DIM = 8


def _make_kv_tensors():
    """Return deterministic (key, value) tensors with batch=1."""
    torch.manual_seed(0)
    k = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    v = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return k, v


def _get_kv_pairs(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract (key, value) pairs from a DynamicCache (any transformers version)."""
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer.keys, layer.values) for layer in cache.layers]


def _make_dynamic_cache():
    cache = DynamicCache()
    for _ in range(NUM_LAYERS):
        k, v = _make_kv_tensors()
        cache.update(k, v, len(cache))
    return cache


def _make_tuple_cache():
    layers = []
    for _ in range(NUM_LAYERS):
        k, v = _make_kv_tensors()
        layers.append((k, v))
    return tuple(layers)


class TestExpandKvCache:
    def test_dynamic_cache_input(self):
        cache = _make_dynamic_cache()
        expanded = ModelClient._expand_kv_cache(cache, batch_size=3)

        assert isinstance(expanded, DynamicCache)
        pairs = _get_kv_pairs(expanded)
        assert len(pairs) == NUM_LAYERS
        for k, v in pairs:
            assert k.shape == (3, NUM_HEADS, SEQ_LEN, HEAD_DIM)
            assert v.shape == (3, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    def test_tuple_cache_input(self):
        cache = _make_tuple_cache()
        expanded = ModelClient._expand_kv_cache(cache, batch_size=4)

        assert isinstance(expanded, DynamicCache)
        pairs = _get_kv_pairs(expanded)
        assert len(pairs) == NUM_LAYERS
        for k, v in pairs:
            assert k.shape == (4, NUM_HEADS, SEQ_LEN, HEAD_DIM)
            assert v.shape == (4, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    def test_values_are_broadcast_copies(self):
        """Each batch element should contain the same data as the original."""
        cache = _make_dynamic_cache()
        expanded = ModelClient._expand_kv_cache(cache, batch_size=5)

        orig_pairs = _get_kv_pairs(cache)
        exp_pairs = _get_kv_pairs(expanded)
        for (orig_k, orig_v), (exp_k, exp_v) in zip(orig_pairs, exp_pairs):
            for b in range(5):
                assert torch.equal(exp_k[b], orig_k[0])
                assert torch.equal(exp_v[b], orig_v[0])

    def test_expanded_tensors_are_contiguous(self):
        cache = _make_dynamic_cache()
        expanded = ModelClient._expand_kv_cache(cache, batch_size=2)

        for k, v in _get_kv_pairs(expanded):
            assert k.is_contiguous()
            assert v.is_contiguous()

    def test_batch_size_one_preserves_values(self):
        cache = _make_dynamic_cache()
        expanded = ModelClient._expand_kv_cache(cache, batch_size=1)

        orig_pairs = _get_kv_pairs(cache)
        exp_pairs = _get_kv_pairs(expanded)
        for (orig_k, orig_v), (exp_k, exp_v) in zip(orig_pairs, exp_pairs):
            assert torch.equal(orig_k, exp_k)
            assert torch.equal(orig_v, exp_v)
