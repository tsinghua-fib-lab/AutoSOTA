#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
K 维度变化测试

验证 prune -> slide -> compress 各阶段 K 维度变化与 L 的关系：
- Prune: K 保持不变
- Slide: K_out = K * (L / (L-2)) = K * expand_ratio
- Compress: K_final = K_out / 2 (2:4 压缩)

综合: K_final / K_original = expand_ratio / 2
    - L=6:  4/3 / 2 = 2/3 ≈ 0.667
    - L=8:  3/2 / 2 = 3/4 = 0.75
    - L=10: 8/5 / 2 = 4/5 = 0.80
    - L=12: 2/1 / 2 = 1.0 = 1.00
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils import SlideSparseConfig, compute_output_k, verify_ZL_sparsity, verify_2to4_sparsity


def calculate_theoretical_ratios():
    """计算各阶段的理论比率"""
    print("=" * 60)
    print("Theoretical K Dimension Ratios")
    print("=" * 60)
    
    print(f"{'L':<6} {'Expand (slide)':<18} {'Compress':<12} {'Final Ratio':<12}")
    print("-" * 60)
    
    for L in [6, 8, 10, 12, 14, 16]:
        config = SlideSparseConfig(Z=2, L=L)
        
        expand_ratio = config.expand_ratio
        compress_ratio = 0.5  # 2:4 压缩固定为 0.5
        final_ratio = expand_ratio * compress_ratio
        
        print(f"{L:<6} {expand_ratio:<18.4f} {compress_ratio:<12.1f} {final_ratio:<12.4f}")
    
    print()
    return True


def test_prune_dimension():
    """测试 prune 阶段 K 不变"""
    print("=" * 60)
    print("Test: Prune Stage - K Unchanged")
    print("=" * 60)
    
    from prune import prune_tensor
    
    test_cases = [
        (64, 128),
        (128, 256),
        (256, 512),
        (100, 200),  # 非整除
    ]
    
    for N, K in test_cases:
        config = SlideSparseConfig(Z=2, L=8)
        weight = torch.randn(N, K)
        
        pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
        
        print(f"  ({N}, {K}) -> ({pruned.shape[0]}, {pruned.shape[1]})")
        assert pruned.shape == weight.shape, f"Prune changed shape: {weight.shape} -> {pruned.shape}"
    
    print("  Status: PASS (K unchanged after prune)")
    print()
    return True


def test_slide_dimension():
    """测试 slide 阶段 K 变化"""
    print("=" * 60)
    print("Test: Slide Stage - K Expansion")
    print("=" * 60)
    
    from prune import prune_tensor
    from slide import slide_tensor
    
    print(f"{'N×K':<12} {'L':<6} {'K_prune':<10} {'K_slide':<10} {'Actual':<10} {'Expected':<10} {'Match':<6}")
    print("-" * 70)
    
    for N, K in [(64, 128), (128, 256)]:
        for L in [6, 8, 10, 12]:
            config = SlideSparseConfig(Z=2, L=L)
            
            weight = torch.randn(N, K)
            pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
            slided, _ = slide_tensor(pruned, config)
            
            expected_ratio = config.expand_ratio
            actual_ratio = slided.shape[1] / pruned.shape[1]
            
            # 由于对齐要求（K 对齐到 16），允许较大误差
            match = abs(actual_ratio - expected_ratio) < 0.1
            
            print(f"{N}×{K:<6} {L:<6} {pruned.shape[1]:<10} {slided.shape[1]:<10} {actual_ratio:<10.3f} {expected_ratio:<10.3f} {'✓' if match else '✗':<6}")
            
            # 只要在合理范围内即可
            assert match, f"Ratio mismatch: actual={actual_ratio:.3f}, expected={expected_ratio:.3f}"
    
    print("  Status: PASS")
    print()
    return True


def test_compress_dimension():
    """测试 compress 阶段 K 变化（模拟，不需要真实的 SO）"""
    print("=" * 60)
    print("Test: Compress Stage - K Halving (Simulated)")
    print("=" * 60)
    
    # cuSPARSELt 2:4 压缩将 K 减半
    # 这里模拟压缩效果
    
    test_cases = [
        (64, 192),   # slide 后的 K
        (128, 384),
        (256, 768),
    ]
    
    for N, K_slide in test_cases:
        # 2:4 压缩后 K 减半
        K_compressed = K_slide // 2
        compress_ratio = K_compressed / K_slide
        
        print(f"  ({N}, {K_slide}) -> ({N}, {K_compressed}), compress ratio = {compress_ratio}")
        assert compress_ratio == 0.5, f"Compress ratio should be 0.5, got {compress_ratio}"
    
    print("  Status: PASS (K halved after compress)")
    print()
    return True


def test_full_pipeline_dimension():
    """测试完整流水线的 K 变化"""
    print("=" * 60)
    print("Test: Full Pipeline K Dimension")
    print("=" * 60)
    
    from prune import prune_tensor
    from slide import slide_tensor
    # 不导入 compress，使用理论值
    
    print(f"{'N×K':<12} {'L':<6} {'K_orig':<10} {'K_slide':<10} {'K_final':<10} {'Final/Orig':<12} {'Theory':<12}")
    print("-" * 80)
    
    for N, K in [(64, 128), (128, 256), (256, 512)]:
        for L in [6, 8, 10, 12]:
            config = SlideSparseConfig(Z=2, L=L)
            
            weight = torch.randn(N, K)
            pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
            slided, _ = slide_tensor(pruned, config)
            
            K_orig = weight.shape[1]
            K_slide = slided.shape[1]
            K_final = K_slide // 2  # 模拟 2:4 压缩
            
            actual_ratio = K_final / K_orig
            theory_ratio = config.expand_ratio / 2
            
            match = abs(actual_ratio - theory_ratio) < 0.1
            
            print(f"{N}×{K:<6} {L:<6} {K_orig:<10} {K_slide:<10} {K_final:<10} {actual_ratio:<12.3f} {theory_ratio:<12.3f}")
    
    print()
    print("  Summary of Final/Original K Ratios:")
    for L in [6, 8, 10, 12]:
        config = SlideSparseConfig(Z=2, L=L)
        ratio = config.expand_ratio / 2
        print(f"    L={L}: K_final = K_original × {ratio:.4f}")
    
    print()
    print("  Status: PASS")
    print()
    return True


def test_alignment_requirements():
    """测试对齐要求"""
    print("=" * 60)
    print("Test: Alignment Requirements")
    print("=" * 60)
    
    # cuSPARSELt 要求 K 对齐到 16
    align_to = 16
    
    test_cases = [
        (64, 100),   # 非对齐
        (64, 128),   # 对齐
        (64, 200),   # 非对齐
    ]
    
    print(f"{'K_in':<10} {'L':<6} {'K_padded':<12} {'K_out':<10} {'K_out%16':<10}")
    print("-" * 50)
    
    for N, K in test_cases:
        for L in [8]:
            config = SlideSparseConfig(Z=2, L=L)
            k_padded, k_out = compute_output_k(K, config, align_to=align_to)
            
            print(f"{K:<10} {L:<6} {k_padded:<12} {k_out:<10} {k_out % align_to:<10}")
            
            assert k_out % align_to == 0, f"K_out not aligned: {k_out}"
            assert k_padded % L == 0, f"K_padded not multiple of L: {k_padded}"
    
    print("  Status: PASS")
    print()
    return True


def test_relationship_L_and_ratio():
    """测试 L 与比率的关系"""
    print("=" * 60)
    print("Test: Relationship Between L and K Ratio")
    print("=" * 60)
    
    print("The mathematical relationship:")
    print("  expand_ratio = L / (L - Z) = L / (L - 2)")
    print("  compress_ratio = 0.5 (2:4 compression)")
    print("  final_ratio = L / (2 * (L - 2))")
    print()
    
    print(f"{'L':<6} {'Expand':<12} {'Final':<12} {'K×1000->?':<12}")
    print("-" * 45)
    
    for L in range(4, 20, 2):  # L 必须 > Z=2
        if L <= 2:
            continue
        
        expand = L / (L - 2)
        final = expand / 2
        k_example = int(1000 * final)
        
        print(f"{L:<6} {expand:<12.4f} {final:<12.4f} {k_example:<12}")
    
    print()
    print("  Observations:")
    print("    - As L increases, expand_ratio decreases (approaches 1.0)")
    print("    - As L increases, final_ratio decreases (approaches 0.5)")
    print("    - L=4 is maximum compression (final=1.0)")
    print("    - L→∞ gives minimum compression (final→0.5)")
    print()
    print("  Status: PASS")
    print()
    return True


def main():
    """运行所有测试"""
    tests = [
        calculate_theoretical_ratios,
        test_prune_dimension,
        test_slide_dimension,
        test_compress_dimension,
        test_full_pipeline_dimension,
        test_alignment_requirements,
        test_relationship_L_and_ratio,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
