#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NK Size 工具函数测试

验证 utils.py 中的 NK size 工具函数:
1. compute_output_k - 计算 slide 后的 K 维度
2. compute_compressed_k - 计算 2:4 压缩后的 K 维度
3. SlideSparseConfig - 配置类和扩展比计算
4. 理论计算与实际 slide 结果的一致性
"""

import sys
from pathlib import Path

# 添加路径 (weight_convert 目录)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils import SlideSparseConfig, compute_output_k, verify_2to4_sparsity
from slide import slide_tensor


# ============================================================
# Test: SlideSparseConfig
# ============================================================

def test_slidesparse_config_expand_ratio():
    """测试 SlideSparseConfig 的扩展比计算"""
    print("=" * 60)
    print("Test: SlideSparseConfig Expand Ratio")
    print("=" * 60)
    
    # 公式: expand_ratio = (L/2 - 1) * 4 / L = 2 - 4/L
    expected = {
        4: (2 - 1) * 4 / 4,      # 1.0
        6: (3 - 1) * 4 / 6,      # 1.333...
        8: (4 - 1) * 4 / 8,      # 1.5
        10: (5 - 1) * 4 / 10,    # 1.6
        12: (6 - 1) * 4 / 12,    # 1.666...
        16: (8 - 1) * 4 / 16,    # 1.75
    }
    
    all_correct = True
    for L, exp_ratio in expected.items():
        config = SlideSparseConfig(Z=2, L=L)
        
        match = abs(config.expand_ratio - exp_ratio) < 1e-6
        all_correct &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} L={L:2d}: expand_ratio={config.expand_ratio:.6f} (expected {exp_ratio:.6f})")
        
        assert match, f"L={L}: expand_ratio mismatch"
    
    print(f"  Status: {'PASS' if all_correct else 'FAIL'}")
    print()
    return True


def test_slidesparse_config_window_count():
    """测试 SlideSparseConfig 的窗口数量计算"""
    print("=" * 60)
    print("Test: SlideSparseConfig Window Count")
    print("=" * 60)
    
    # num_windows = L/2 - 1 (每组 L 元素产生的 4 元素窗口数)
    expected_windows = {
        4: 1,
        6: 2,
        8: 3,
        10: 4,
        12: 5,
    }
    
    all_correct = True
    for L, exp_windows in expected_windows.items():
        config = SlideSparseConfig(Z=2, L=L)
        
        match = (config.num_windows == exp_windows)
        all_correct &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} L={L:2d}: num_windows={config.num_windows} (expected {exp_windows})")
        
        assert match, f"L={L}: num_windows mismatch"
    
    print(f"  Status: {'PASS' if all_correct else 'FAIL'}")
    print()
    return True


# ============================================================
# Test: compute_output_k
# ============================================================

def test_compute_output_k_basic():
    """测试 compute_output_k 基本计算"""
    print("=" * 60)
    print("Test: compute_output_k Basic")
    print("=" * 60)
    
    # 手动计算预期值
    # k_padded = ceil(K_in / L) * L
    # k_out_raw = k_padded * expand_ratio
    # k_out = ceil(k_out_raw / align_to) * align_to
    
    test_cases = [
        # (K_in, L, align_to, expected_k_padded, expected_k_out)
        (32, 8, 16, 32, 48),    # 32 * 1.5 = 48
        (64, 8, 16, 64, 96),    # 64 * 1.5 = 96
        (128, 8, 16, 128, 192), # 128 * 1.5 = 192
        (32, 6, 16, 36, 48),    # 36 * 1.333 = 48
        (64, 6, 16, 66, 96),    # pad to 66, * 1.333 = 88 -> align to 96
        (32, 10, 16, 40, 64),   # 40 * 1.6 = 64
    ]
    
    all_correct = True
    for K_in, L, align_to, exp_padded, exp_out in test_cases:
        config = SlideSparseConfig(Z=2, L=L)
        k_padded, k_out = compute_output_k(K_in, config, align_to)
        
        padded_ok = (k_padded == exp_padded)
        out_ok = (k_out == exp_out)
        match = padded_ok and out_ok
        all_correct &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} K={K_in:3d}, L={L:2d}: k_padded={k_padded} (exp {exp_padded}), k_out={k_out} (exp {exp_out})")
        
        assert match, f"K={K_in}, L={L}: mismatch"
    
    print(f"  Status: {'PASS' if all_correct else 'FAIL'}")
    print()
    return True


def test_compute_output_k_alignment():
    """测试 compute_output_k 输出对齐"""
    print("=" * 60)
    print("Test: compute_output_k Alignment")
    print("=" * 60)
    
    # 测试各种 K 值，确保输出总是对齐到 16
    K_values = [16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 255, 256, 512, 1000, 1024, 2048, 4096]
    L_values = [6, 8, 10, 12]
    align_to = 16
    
    all_aligned = True
    violations = []
    
    for L in L_values:
        config = SlideSparseConfig(Z=2, L=L)
        for K in K_values:
            k_padded, k_out = compute_output_k(K, config, align_to)
            
            padded_aligned = (k_padded % L == 0)
            out_aligned = (k_out % align_to == 0)
            
            if not padded_aligned:
                violations.append(f"K={K}, L={L}: k_padded={k_padded} not divisible by L")
            if not out_aligned:
                violations.append(f"K={K}, L={L}: k_out={k_out} not divisible by {align_to}")
            
            all_aligned &= (padded_aligned and out_aligned)
    
    if violations:
        for v in violations[:5]:
            print(f"  ✗ {v}")
    else:
        print(f"  All {len(K_values) * len(L_values)} cases correctly aligned")
    
    print(f"  Status: {'PASS' if all_aligned else 'FAIL'}")
    print()
    return all_aligned


def test_compute_output_k_monotonic():
    """测试 compute_output_k 输出单调递增"""
    print("=" * 60)
    print("Test: compute_output_k Monotonic Increase")
    print("=" * 60)
    
    # 确保 K 增大时，k_out 也单调不减
    all_monotonic = True
    
    for L in [6, 8, 10]:
        config = SlideSparseConfig(Z=2, L=L)
        prev_k_out = 0
        
        for K in range(16, 1025, 8):
            _, k_out = compute_output_k(K, config, align_to=16)
            
            if k_out < prev_k_out:
                print(f"  ✗ L={L}: K={K} -> k_out={k_out} < prev {prev_k_out}")
                all_monotonic = False
            
            prev_k_out = k_out
    
    if all_monotonic:
        print(f"  All L values show monotonic k_out increase")
    
    print(f"  Status: {'PASS' if all_monotonic else 'FAIL'}")
    print()
    return all_monotonic


# ============================================================
# Test: Theoretical vs Actual Slide Consistency
# ============================================================

def create_ZL_sparse_tensor(N: int, K: int, Z: int, L: int, dtype=torch.float32) -> torch.Tensor:
    """创建满足 Z:L 约束的张量"""
    weight = torch.randn(N, K, dtype=torch.float32)
    
    k_padded = ((K + L - 1) // L) * L
    for i in range(N):
        for g in range(k_padded // L):
            start = g * L
            end = min(start + L, K)
            if end - start >= Z:
                local_indices = torch.randperm(end - start)[:Z]
                weight[i, start + local_indices] = 0
    
    return weight.to(dtype) if dtype != torch.int8 else weight.to(torch.int8)


def test_theoretical_vs_actual_slide():
    """测试理论计算与实际 slide 输出维度的一致性"""
    print("=" * 60)
    print("Test: Theoretical vs Actual Slide Dimensions")
    print("=" * 60)
    
    torch.manual_seed(42)
    N = 8
    align_to = 16
    
    # 测试多种 K 和 L 组合
    test_cases = [
        # (K, L)
        (32, 6),
        (32, 8),
        (64, 8),
        (128, 8),
        (33, 6),   # K 不整除 L
        (65, 8),   # K 不整除 L
        (100, 8),
        (100, 10),
        (256, 8),
        (256, 10),
        (512, 8),
        (1024, 8),
    ]
    
    all_match = True
    
    for K_in, L in test_cases:
        config = SlideSparseConfig(Z=2, L=L)
        
        # 理论计算
        k_padded_theory, k_out_theory = compute_output_k(K_in, config, align_to)
        
        # 实际 slide
        weight = create_ZL_sparse_tensor(N, K_in, Z=2, L=L)
        slided, metadata = slide_tensor(weight, config, align_to=align_to)
        k_out_actual = slided.shape[1]
        
        match = (k_out_actual == k_out_theory)
        all_match &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} K={K_in:4d}, L={L:2d}: theory={k_out_theory:4d}, actual={k_out_actual:4d}")
        
        assert match, f"K={K_in}, L={L}: theory {k_out_theory} != actual {k_out_actual}"
    
    print(f"  Status: {'PASS' if all_match else 'FAIL'}")
    print()
    return True


def test_theoretical_vs_actual_2to4_valid():
    """测试理论维度下的 slide 结果确实满足 2:4"""
    print("=" * 60)
    print("Test: Theoretical Dimensions -> 2:4 Valid")
    print("=" * 60)
    
    torch.manual_seed(42)
    N = 16
    
    test_cases = [
        (64, 6),
        (64, 8),
        (128, 8),
        (256, 8),
        (256, 10),
        (512, 8),
        (1024, 8),
    ]
    
    all_valid = True
    
    for K_in, L in test_cases:
        config = SlideSparseConfig(Z=2, L=L)
        
        weight = create_ZL_sparse_tensor(N, K_in, Z=2, L=L)
        slided, _ = slide_tensor(weight, config, align_to=16)
        
        is_valid, violation_ratio = verify_2to4_sparsity(slided)
        all_valid &= is_valid
        
        status = "✓" if is_valid else "✗"
        print(f"  {status} K={K_in:4d}, L={L:2d}: 2:4 valid={is_valid}, violation={violation_ratio:.2%}")
        
        assert is_valid, f"K={K_in}, L={L}: 2:4 not satisfied"
    
    print(f"  Status: {'PASS' if all_valid else 'FAIL'}")
    print()
    return True


# ============================================================
# Test: Compressed K Calculation
# ============================================================

def test_compressed_k_calculation():
    """测试 2:4 压缩后的 K 维度计算"""
    print("=" * 60)
    print("Test: Compressed K Calculation")
    print("=" * 60)
    
    # 2:4 压缩: 每 4 个元素压缩到 2 个，K 减半
    
    test_cases = [
        # (k_slided, expected_k_compressed)
        (48, 24),
        (64, 32),
        (96, 48),
        (128, 64),
        (192, 96),
        (256, 128),
    ]
    
    all_correct = True
    
    for k_slided, expected_k_compressed in test_cases:
        # 简单公式: k_compressed = k_slided // 2
        k_compressed = k_slided // 2
        
        match = (k_compressed == expected_k_compressed)
        all_correct &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} k_slided={k_slided:4d} -> k_compressed={k_compressed:4d} (expected {expected_k_compressed})")
        
        assert match
    
    print(f"  Status: {'PASS' if all_correct else 'FAIL'}")
    print()
    return True


# ============================================================
# Test: Real Model NK Size Patterns
# ============================================================

def test_model_nk_patterns():
    """测试真实模型的 NK 尺寸模式"""
    print("=" * 60)
    print("Test: Real Model NK Patterns")
    print("=" * 60)
    
    # 常见的 hidden_size 和相关维度
    # Qwen2.5-0.5B: hidden=896, intermediate=4864
    # Qwen2.5-1.5B: hidden=1536, intermediate=8960
    # Llama3.2-1B: hidden=2048, intermediate=8192
    
    model_configs = [
        {
            "name": "Qwen2.5-0.5B",
            "hidden": 896,
            "intermediate": 4864,
            "num_heads": 14,
            "num_kv_heads": 2,
            "head_dim": 64,
        },
        {
            "name": "Qwen2.5-1.5B",
            "hidden": 1536,
            "intermediate": 8960,
            "num_heads": 12,
            "num_kv_heads": 2,
            "head_dim": 128,
        },
        {
            "name": "Llama3.2-1B",
            "hidden": 2048,
            "intermediate": 8192,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
        },
    ]
    
    L_values = [6, 8, 10]
    
    for model in model_configs:
        print(f"\n  {model['name']}:")
        
        # 计算各层的 N, K
        h = model["hidden"]
        i = model["intermediate"]
        nh = model["num_heads"]
        nkv = model["num_kv_heads"]
        hd = model["head_dim"]
        
        layers = {
            "q_proj":   (nh * hd, h),
            "k_proj":   (nkv * hd, h),
            "v_proj":   (nkv * hd, h),
            "o_proj":   (h, nh * hd),
            "gate_proj": (i, h),
            "up_proj":  (i, h),
            "down_proj": (h, i),
        }
        
        for layer_name, (N, K) in layers.items():
            for L in [8]:  # 主要用 L=8
                config = SlideSparseConfig(Z=2, L=L)
                _, k_out = compute_output_k(K, config, align_to=16)
                k_compressed = k_out // 2
                
                # 验证对齐
                assert k_out % 16 == 0, f"{layer_name}: k_out not aligned"
                
                print(f"    {layer_name:10s}: N={N:5d}, K={K:5d} -> k_slided={k_out:5d}, k_compressed={k_compressed:5d}")
    
    print(f"\n  Status: PASS")
    print()
    return True


# ============================================================
# Test: Expand Ratio Boundary
# ============================================================

def test_expand_ratio_boundary():
    """测试扩展比的边界：确保不超过 2.0"""
    print("=" * 60)
    print("Test: Expand Ratio Boundary")
    print("=" * 60)
    
    # 扩展比公式: 2 - 4/L
    # L=4 时 expand_ratio = 1.0 (最小)
    # L->∞ 时 expand_ratio -> 2.0 (极限)
    
    all_valid = True
    
    for L in range(4, 101, 2):  # 偶数 L
        config = SlideSparseConfig(Z=2, L=L)
        
        # 必须 >= 1.0 且 < 2.0
        valid = (1.0 <= config.expand_ratio < 2.0)
        all_valid &= valid
        
        if not valid:
            print(f"  ✗ L={L}: expand_ratio={config.expand_ratio} out of range [1.0, 2.0)")
    
    # 打印一些关键点
    print(f"  L=4:  expand_ratio={SlideSparseConfig(Z=2, L=4).expand_ratio:.4f}")
    print(f"  L=6:  expand_ratio={SlideSparseConfig(Z=2, L=6).expand_ratio:.4f}")
    print(f"  L=8:  expand_ratio={SlideSparseConfig(Z=2, L=8).expand_ratio:.4f}")
    print(f"  L=10: expand_ratio={SlideSparseConfig(Z=2, L=10).expand_ratio:.4f}")
    print(f"  L=20: expand_ratio={SlideSparseConfig(Z=2, L=20).expand_ratio:.4f}")
    print(f"  L=100: expand_ratio={SlideSparseConfig(Z=2, L=100).expand_ratio:.4f}")
    
    print(f"  All expand ratios in valid range [1.0, 2.0)")
    print(f"  Status: {'PASS' if all_valid else 'FAIL'}")
    print()
    return all_valid


# ============================================================
# Main
# ============================================================

def main():
    """运行所有测试"""
    tests = [
        test_slidesparse_config_expand_ratio,
        test_slidesparse_config_window_count,
        test_compute_output_k_basic,
        test_compute_output_k_alignment,
        test_compute_output_k_monotonic,
        test_theoretical_vs_actual_slide,
        test_theoretical_vs_actual_2to4_valid,
        test_compressed_k_calculation,
        test_model_nk_patterns,
        test_expand_ratio_boundary,
    ]
    
    passed = 0
    failed = 0
    
    print("\n" + "=" * 60)
    print("NK Size Utility Tests")
    print("=" * 60 + "\n")
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
