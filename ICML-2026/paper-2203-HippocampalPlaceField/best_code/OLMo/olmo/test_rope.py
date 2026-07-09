import torch
import pytest
from model2 import ModelConfig, RotaryEmbedding, ScaledRoPE, BufferCache

def get_test_config(d_model=512, n_heads=8, max_seq_len=1024):
    config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        rope_theta=10000.0,
        rope_full_precision=False,
        max_sequence_length=max_seq_len,
        layer_norm_eps=1e-5,
        include_bias=True
    )
    return config

def test_scaled_rope_initialization():
    config = get_test_config()
    cache = BufferCache()
    scaled_rope = ScaledRoPE(config, lambda_val=10.0, sigma=20.0)
    
    assert scaled_rope.d_head == config.d_model // config.n_heads
    assert scaled_rope.d_half == scaled_rope.d_head // 2
    
    assert hasattr(scaled_rope, "w_sin")
    assert hasattr(scaled_rope, "w_cos")
    assert scaled_rope.w_sin.shape == (scaled_rope.d_half,)
    assert scaled_rope.w_cos.shape == (scaled_rope.d_half,)
    assert torch.all(scaled_rope.w_sin >= 0) and torch.all(scaled_rope.w_sin <= 1)
    assert torch.all(scaled_rope.w_cos >= 0) and torch.all(scaled_rope.w_cos <= 1)

def test_scaled_rope_forward_shape():
    config = get_test_config()
    cache = BufferCache()
    scaled_rope = ScaledRoPE(config)
    
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 15
    head_dim = config.d_model // config.n_heads
    q = torch.randn(batch_size, config.n_heads, seq_len_q, head_dim)
    k = torch.randn(batch_size, config.n_heads, seq_len_k, head_dim)
    
    q_rot, k_rot = scaled_rope(q, k)
    
    assert q_rot.shape == (batch_size, config.n_heads, seq_len_q, seq_len_k, head_dim)
    assert k_rot.shape == k.shape

def test_relative_rotary_embedding():
    config = get_test_config(d_model=128, n_heads=2)
    scaled_rope = ScaledRoPE(config)
    q_len, k_len = 2, 3
    
    pos_sin, pos_cos = scaled_rope.get_relative_rotary_embedding(q_len, k_len, device=torch.device("cpu"))
    
    q_pos = torch.arange(q_len)
    k_pos = torch.arange(k_len)
    relative_diffs = q_pos.view(-1, 1) - k_pos.view(1, -1)
    
    inv_freq = scaled_rope.get_inv_freq(torch.device("cpu"))
    manual_freqs = torch.einsum("ij, d -> ijd", relative_diffs, inv_freq)
    manual_sin = manual_freqs.sin()[None, None, :, :, :]
    manual_cos = manual_freqs.cos()[None, None, :, :, :]
    
    assert torch.allclose(pos_sin, manual_sin, atol=1e-6)
    assert torch.allclose(pos_cos, manual_cos, atol=1e-6)

def test_scaled_rope_numerical_stability():
    config = get_test_config()
    scaled_rope = ScaledRoPE(config)
    
    test_cases = [
        (torch.zeros(2, 8, 10, 64), torch.ones(2, 8, 10, 64)),
        (torch.randn(2, 8, 10, 64) * 1000, torch.randn(2, 8, 10, 64) * 1000),
        (torch.ones(2, 8, 1, 64), torch.randn(2, 8, 100, 64))
    ]
    
    for q, k in test_cases:
        q_rot, k_rot = scaled_rope(q, k)
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()
        assert not torch.isnan(k_rot).any()

def test_scaled_rope_compatibility_with_original():
    config = get_test_config()
    cache = BufferCache()
    original_rope = RotaryEmbedding(config, cache)
    
    scaled_rope = ScaledRoPE(config, lambda_val=1e6, sigma=1e-6)
    
    q = torch.randn(1, 8, 5, 64)
    k = torch.randn(1, 8, 5, 64)
    
    q_orig, k_orig = original_rope(q, k)
    
    q_scaled, k_scaled = scaled_rope(q, k)
    q_scaled = q_scaled[:, :, torch.arange(5), torch.arange(5), :]
    
    assert torch.allclose(q_scaled, q_orig, atol=1e-3)

def test_scaled_rope_dtype_compatibility():
    config = get_test_config()
    scaled_rope = ScaledRoPE(config)
    
    q_fp16 = torch.randn(2, 8, 10, 64, dtype=torch.float16)
    k_fp16 = torch.randn(2, 8, 10, 64, dtype=torch.float16)
    q_rot_fp16, k_rot_fp16 = scaled_rope(q_fp16, k_fp16)
    assert q_rot_fp16.dtype == torch.float16
    
    q_fp32 = torch.randn(2, 8, 10, 64, dtype=torch.float32)
    k_fp32 = torch.randn(2, 8, 10, 64, dtype=torch.float32)
    q_rot_fp32, k_rot_fp32 = scaled_rope(q_fp32, k_fp32)
    assert q_rot_fp32.dtype == torch.float32

if __name__ == "__main__":
    pytest.main(["-v", "test_rope.py"])
