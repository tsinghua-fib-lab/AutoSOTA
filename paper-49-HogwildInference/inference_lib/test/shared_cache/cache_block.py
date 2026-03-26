""" A part of key-value cache that can be concatenated with other such parts in a RoPE-friendly way; used for shared cache manipulation """
from __future__ import annotations

import contextlib
import copy
import os
import warnings
from typing import Dict, Tuple, Any, Optional
import torch
import transformers
import triton
import triton.language as tl

USE_TRITON = bool(int(os.environ.get("HOGWILD_USE_TRITON", "1")))


class CacheBlock(transformers.cache_utils.DynamicCache):
    """
    A key-value cache block that stores a sequence of tokens that can be stacked with other caches.
    The keys in CacheBlock can be quickly rotated to new position to account for RoPE.
    Designed for llama 3 & qwen 2.5 models and their descendants.

    :footnote: it is likely possible to speed up cos/sin computation in one of two ways:
        (a) cache cos/sin for shifts up to npo2(running maximum value) and reuse them
        (b) using existing formulae for sin/cos differences from values available in cache_kwargs
        cos(a - b) = cos a cos b + sin a sin b ; sin(a - b) = sin a * cos b - cos a * sin b
        (c) torch.compile with fusion

    """

    def __init__(self, *args, config: transformers.PretrainedConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.first_available_positions_by_layer = dict()

    def get_kv_with_offset(self, *, layer_idx: int, offset: int):
        """Get key-value pairs rotated so that the first value has position :offset:"""
        assert len(self.key_cache) > layer_idx, "cache must be fully populated before it can accept None args"
        key_cache, value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]
        return rotate_by_offset(keys=key_cache, offset=offset, config=self.config), value_cache

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Dict[str, Any],
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add key/value states to the cache, rotate keys so that their positions are at the end of the preceding key"""
        assert all(key in cache_kwargs for key in ('cos', 'sin', 'cache_position')), cache_kwargs
        assert len(cache_kwargs['cache_position']) == key_states.shape[-2] == value_states.shape[-2]
        # rotate keys so their new positions start immediately after the last pre-existing position
        first_available_position = self.first_available_positions_by_layer.setdefault(layer_idx, 0)
        cache_positions_before_rotation = cache_kwargs['cache_position'].tolist()
        if len(cache_positions_before_rotation) == 0:  # no new inputs
            assert key_states.shape[-2] == value_states.shape[-2] == 0
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        required_position_offset: int = first_available_position - cache_positions_before_rotation[0]
        if required_position_offset != 0 and len(cache_positions_before_rotation) > 0:
            key_states = rotate_by_offset(keys=key_states, offset=required_position_offset, config=self.config)
        self.first_available_positions_by_layer[layer_idx] = \
            cache_positions_before_rotation[-1] + required_position_offset + 1
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def update(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Act as if this were a normal cache (similar to DynamicCache). Assumes consecutive cache_position."""
        return self.append(*args, **kwargs)

    def clear(self):
        """Delete all cache entries in-place"""
        self.key_cache.clear()
        self.value_cache.clear()
        self._seen_tokens = 0
        self.first_available_positions_by_layer.clear()

    def append_from(self, other: CacheBlock, keep_other: bool = True):
        """Add tokens from other to the end of this block in-place. If not keep_other, calls other.clear()"""
        assert len(other.key_cache) == len(other.value_cache)
        for layer_idx in range(len(other.key_cache)):
            new_token_starting_position = self.first_available_positions_by_layer.setdefault(layer_idx, 0)
            other_position_span_size = other.first_available_positions_by_layer.get(layer_idx, 0)
            super().update(
                *other.get_kv_with_offset(layer_idx=layer_idx, offset=new_token_starting_position),  # keys, values
                layer_idx=layer_idx,   # cache_kwargs is omitted; we do not need it since :other: is already rotated.
            )
            self.first_available_positions_by_layer[layer_idx] += other_position_span_size
        if not keep_other:
            other.clear()


_ROTARY_CACHE: Optional[Dict[Any, Tuple[torch.Tensor, torch.Tensor]]] = None


@contextlib.contextmanager
def using_rotary_cache(cache: dict):
    """Use :cache: as a temporary storage for rotary cos/sin values to avoid recomputation of the same values"""
    global _ROTARY_CACHE
    _prev_rotary_cache = _ROTARY_CACHE
    try:
        _ROTARY_CACHE = cache
        yield _ROTARY_CACHE
    finally:
        _ROTARY_CACHE = _prev_rotary_cache


def rotate_by_offset(*, keys: torch.Tensor, offset: int, config: transformers.PretrainedConfig, unsqueeze_dim: int = 1):
    """
    Rotate a tensor of attention keys along its *last* dimension to simulate adding offset to position ids
    """
    if offset == 0:
        return keys
    if _ROTARY_CACHE is not None:
        cache_id = (offset, id(config), keys.device, keys.dtype, unsqueeze_dim)
        if cache_id not in _ROTARY_CACHE:
            _ROTARY_CACHE[cache_id] = compute_rotary_cos_sin(offset, config, unsqueeze_dim, keys.dtype, keys.device)
        cos, sin = _ROTARY_CACHE[cache_id]
    else:
        cos, sin = compute_rotary_cos_sin(offset, config, unsqueeze_dim, keys.dtype, keys.device)

    if USE_TRITON:
        return _apply_rotary_cos_sin_triton(keys, cos, sin)
    else:
        return _apply_rotary_cos_sin(keys, cos, sin)


def compute_rotary_cos_sin(
        offset: int, config: transformers.PretrainedConfig, unsqueeze_dim: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq, attention_scaling = _get_rope_init(config, device)
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if USE_TRITON:
            return _compute_rotary_cos_sin_triton(offset, inv_freq, attention_scaling, unsqueeze_dim, dtype, device)
        else:
            return _compute_rotary_cos_sin(offset, inv_freq, attention_scaling, unsqueeze_dim, dtype, device)


@torch.compile(dynamic=True, disable=bool(int(os.environ.get("HOGWILD_NO_COMPILE", False))))
def _compute_rotary_cos_sin(
        offset: int, inv_freq: torch.Tensor, attention_scaling: float, unsqueeze_dim: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Core RoPE block
    position_ids = torch.tensor(offset, device=device, dtype=torch.int64).view(1, 1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    freqs = (inv_freq_expanded.float().to(device) @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    if attention_scaling != 1:
        cos = (cos * attention_scaling)
        sin = (sin * attention_scaling)
    return cos.to(dtype).unsqueeze(unsqueeze_dim), sin.to(dtype).unsqueeze(unsqueeze_dim)


@triton.jit
def compute_rotary_cos_sin_kernel(
    inv_freq_ptr: tl.tensor,
    cos_ptr: tl.tensor,
    sin_ptr: tl.tensor,
    offset: int,
    attention_scaling: float,
    PE_DIM: int,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inv_freq = tl.load(inv_freq_ptr + offsets, offsets < PE_DIM)
    freq = offset * inv_freq
    cos = attention_scaling * tl.cos(freq)
    sin = attention_scaling * tl.sin(freq)
    tl.store(cos_ptr + offsets, cos, offsets < PE_DIM)
    tl.store(cos_ptr + offsets + PE_DIM, cos, offsets < PE_DIM)
    tl.store(sin_ptr + offsets, sin, offsets < PE_DIM)
    tl.store(sin_ptr + offsets + PE_DIM, sin, offsets < PE_DIM)


def _compute_rotary_cos_sin_triton(
    offset: int,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    unsqueeze_dim: int,
    dtype: torch.dtype,
    device: torch.device
):
    assert device.type == "cuda", "Only CUDA devices are supported"
    pe_dim = inv_freq.shape[-1]
    out_shape = [1, 1, 2 * pe_dim]
    out_shape = out_shape[:unsqueeze_dim] + [1] + out_shape[unsqueeze_dim:]
    cos = torch.zeros(out_shape, dtype=torch.float32, device=device)
    sin = torch.zeros(out_shape, dtype=torch.float32, device=device)
    grid = lambda meta: (triton.cdiv(pe_dim, meta['BLOCK_SIZE']),)
    compute_rotary_cos_sin_kernel[grid](inv_freq.float(), cos, sin, offset, attention_scaling, pe_dim, BLOCK_SIZE=32)
    return cos.to(dtype), sin.to(dtype)


@torch.compile(dynamic=None, disable=bool(int(os.environ.get("HOGWILD_NO_COMPILE", False))))
def _apply_rotary_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x_dim, pe_dim = x.shape[-1], cos.shape[-1]
    x_pe = x
    if x_dim != pe_dim:
        x_pe = x[..., -pe_dim:]

    x1, x2 = x_pe.chunk(2, dim=-1)
    x_pe_rotated_half = torch.cat((-x2, x1), dim=-1)
    y = (x_pe * cos) + (x_pe_rotated_half * sin)
    if x_dim != pe_dim:
        y = torch.cat([x[..., :-pe_dim], y], dim=-1)
    return y


@triton.jit
def apply_rotary_cos_sin_kernel(
    x_ptr: tl.tensor,
    cos_ptr: tl.tensor,
    sin_ptr: tl.tensor,
    out_ptr: tl.tensor,
    BATCH_SIZE: int,
    NUM_KV_HEADS: int,
    HEAD_DIM: int,
    L: int,
    B0: tl.constexpr,
    B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    i_range = pid_0 * B0 + tl.arange(0, B0)
    j1_range = pid_1 * B1 + tl.arange(0, B1)
    j2_range = j1_range + HEAD_DIM // 2

    x1_range = i_range[:, None] * HEAD_DIM + j1_range[None, :]
    x1_mask = (i_range[:, None] < BATCH_SIZE * NUM_KV_HEADS * L) & (j1_range[None, :] < HEAD_DIM // 2)

    x2_range = i_range[:, None] * HEAD_DIM + j2_range[None, :]
    x2_mask = (i_range[:, None] < BATCH_SIZE * NUM_KV_HEADS * L) & (j2_range[None, :] < HEAD_DIM)

    x1 = tl.load(x_ptr + x1_range, x1_mask, 0)
    x2 = tl.load(x_ptr + x2_range, x2_mask, 0)

    sin1 = tl.load(sin_ptr + j1_range, j1_range < HEAD_DIM // 2, 0)
    sin2 = tl.load(sin_ptr + j2_range, j2_range < HEAD_DIM, 0)

    cos1 = tl.load(cos_ptr + j1_range, j1_range < HEAD_DIM // 2, 0)
    cos2 = tl.load(cos_ptr + j2_range, j2_range < HEAD_DIM, 0)

    tl.store(out_ptr + x1_range, cos1 * x1 - sin1 * x2, x1_mask)
    tl.store(out_ptr + x2_range, cos2 * x2 + sin2 * x1, x2_mask)


def _apply_rotary_cos_sin_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == cos.shape[-1]
    batch_size, num_kv_heads, length, head_dim = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(batch_size * num_kv_heads * length, meta['B0']), triton.cdiv(head_dim, meta['B1']))
    apply_rotary_cos_sin_kernel[grid](x, cos, sin, out, batch_size, num_kv_heads, head_dim, length, B0=1, B1=32)
    return out


_CACHED_ROPE_PARAMS = _CACHED_ROPE_INIT = None  # this is a makeshift functools.lru_cache w/o hashing


def _get_rope_init(config: transformers.PretrainedConfig, device: torch.device):
    global _CACHED_ROPE_PARAMS, _CACHED_ROPE_INIT
    if (config, device) == _CACHED_ROPE_PARAMS:
        return _CACHED_ROPE_INIT

    config = config.get_text_config()
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
    else:
        rope_type = "default"

    if config.model_type == 'deepseek_v3':
        config_init = copy.deepcopy(config)
        assert getattr(config, "head_dim", None) is None
        assert config.rope_scaling.get('attention_factor', 1.0) == 1.0
        config_init.head_dim = config.qk_rope_head_dim
        config_init.rope_scaling['attention_factor'] = 1.0
    else:
        if config.model_type not in ("llama", "qwen2", "qwen3", "qwen3_moe", "phi3"):
            warnings.warn(f"untested model type {config.model_type}")
        config_init = config

    inv_freq, attention_scaling = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS[rope_type](config_init, device)
    _CACHED_ROPE_PARAMS, _CACHED_ROPE_INIT = (config, device), (inv_freq, attention_scaling)
    assert "dynamic" not in rope_type
    return inv_freq, attention_scaling


def test_rotate_by_offset(model_names=("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B"), atol: float = 1e-5):
    for model_name in model_names:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config, rotary_emb = model.config, model.model.rotary_emb
        for batch_size in (0, 1, 2, 3):
            for num_tokens in (0, 1, 2, 100):
                head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
                keys = torch.randn(batch_size, config.num_key_value_heads, num_tokens, head_dim)
                for offset in (-10, -2, -1, 0, 1, 30, 1024, 1025):
                    cos, sin = rotary_emb(keys, torch.tensor(offset).view(1, 1).tile(keys.shape[-2]))
                    _, ref_rotated_keys = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
                        q=keys[..., :1, :], k=keys, cos=cos, sin=sin
                    )
                    our_rotated_keys = rotate_by_offset(keys=keys, offset=offset, config=config)
                    assert torch.allclose(our_rotated_keys, ref_rotated_keys, atol=atol)


def test_cache_block(model_names=("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B"), atol: float = 1e-4):
    for model_name in model_names:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config, rotary_emb = model.config, model.model.rotary_emb

        cache_block = CacheBlock(config=config)
        def _append(key_states, value_states, cache_position):
            cos, sin = rotary_emb(key_states, cache_position.view(1, -1).tile(key_states.shape[-2]))
            cache_block.append(
                layer_idx=0,
                key_states=key_states,
                value_states=value_states,
                cache_kwargs=dict(cos=cos, sin=sin, cache_position=cache_position)
            )

        k0, v0, p0 = torch.randn(1, 8, 3, 128), torch.randn(1, 8, 3, 128), torch.tensor([9, 10, 11])
        _append(k0, v0, p0)  # save positions - rotate from position 9, 10, 11 => 0, 1, 2

        # test 1: retrieve with the same offset
        k, v = cache_block.get_kv_with_offset(layer_idx=0, offset=9)
        assert torch.allclose(k0, k, atol=1e-6) and torch.allclose(v0, v, atol=1e-6)

        # test 2: retrieve with a different offset
        k, v = cache_block.get_kv_with_offset(layer_idx=0, offset=5)
        assert (not torch.allclose(k0, k, atol=1e-6)) and torch.allclose(v0, v, atol=1e-6)
        #                  ^-- keys are rotated                 ^-- values are not rotated, hence allclose

        k = rotate_by_offset(keys=k, offset=9 - 5, config=config)
        assert torch.allclose(k0, k, atol=1e-6)

        # test 3: combine two caches written at different positions into one contigious chunk
        k1, v1, p1 = torch.randn(1, 8, 2, 128), torch.randn(1, 8, 2, 128), torch.tensor([101, 102])
        _append(k1, v1, p1)  # new values will be rotated to positions 3, 4  (because prev last position was 2)

        k, v = cache_block.get_kv_with_offset(layer_idx=0, offset=9)  # returns for positions [9, 10, 11, 12, 13]
        assert torch.allclose(v[..., :v0.shape[-2], :], v0, atol=atol)
        assert torch.allclose(v[..., v0.shape[-2]:, :], v1, atol=atol)

        assert torch.allclose(k[..., :k0.shape[-2], :], k0, atol=atol)  # first 3 values are rotated back to [9, 10, 11]
        assert not torch.allclose(k[..., k0.shape[-2]:, :], k1,
                                  atol=atol)  # last 2 values are rotated to [12, 13] != [101, 102]
        k_part_rotated = rotate_by_offset(keys=k[..., k0.shape[-2]:, :], offset=101 - 12, config=config)
        assert torch.allclose(k_part_rotated, k1, atol=atol)


def test_copy_from(model_names=("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B"), atol: float = 1e-4):
    for model_name in model_names:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config, rotary_emb = model.config, model.model.rotary_emb

        cache_block_1, cache_block_2, cache_block_3 = [CacheBlock(config=config) for _ in range(3)]

        def _append(cache_block, key_states, value_states, cache_position):
            cos, sin = rotary_emb(key_states, cache_position.view(1, -1).tile(key_states.shape[-2]))
            cache_block.append(
                layer_idx=0,
                key_states=key_states,
                value_states=value_states,
                cache_kwargs=dict(cos=cos, sin=sin, cache_position=cache_position)
            )

        _append(cache_block_1, torch.randn(1, 8, 3, 128), torch.randn(1, 8, 3, 128), torch.tensor([9, 10, 11]))
        _append(cache_block_1, torch.randn(1, 8, 2, 128), torch.randn(1, 8, 2, 128), torch.tensor([14, 15]))
        _append(cache_block_2, torch.randn(1, 8, 2, 128), torch.randn(1, 8, 2, 128), torch.tensor([17, 18]))
        _append(cache_block_2, torch.randn(1, 8, 2, 128), torch.randn(1, 8, 2, 128), torch.tensor([20, 21]))
        _append(cache_block_3, torch.randn(1, 8, 5, 128), torch.randn(1, 8, 5, 128), torch.tensor([2, 3, 4, 5, 6]))

        k1, v1 = cache_block_1.get_kv_with_offset(layer_idx=0, offset=10)
        k2, v2 = cache_block_2.get_kv_with_offset(layer_idx=0, offset=10 + 5)
        k3, v3 = cache_block_3.get_kv_with_offset(layer_idx=0, offset=10 + 5 + 4)

        cache_block_1.append_from(other=cache_block_2)
        k_both, v_both = cache_block_1.get_kv_with_offset(layer_idx=0, offset=10)
        assert cache_block_1.get_seq_length() == 5 + 4
        assert cache_block_2.get_seq_length() == 4
        assert torch.allclose(k_both, torch.cat([k1, k2], dim=-2), atol=atol)
        assert torch.allclose(v_both, torch.cat([v1, v2], dim=-2), atol=atol)

        cache_block_1.append_from(other=cache_block_3, keep_other=False)
        k_both, v_both = cache_block_1.get_kv_with_offset(layer_idx=0, offset=10)
        assert cache_block_1.get_seq_length() == 5 + 4 + 5
        assert cache_block_3.get_seq_length() == 0
        assert torch.allclose(k_both, torch.cat([k1, k2, k3], dim=-2), atol=atol)
        assert torch.allclose(v_both, torch.cat([v1, v2, v3], dim=-2), atol=atol)


if __name__ == '__main__':
    test_rotate_by_offset()
    test_cache_block()
    test_copy_from()
