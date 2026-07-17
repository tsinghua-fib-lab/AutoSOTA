#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Prune 正确性测试

验证剪枝功能的正确性：
1. 剪枝后满足 Z:L 约束
2. magnitude 模式保留最重要的值
3. 数据类型正确处理
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils import SlideSparseConfig, verify_ZL_sparsity
from prune import prune_tensor, build_prune_mask_magnitude


def test_prune_basic():
    """测试基本的剪枝功能"""
    print("=" * 60)
    print("Test: Basic Pruning")
    print("=" * 60)
    
    # 创建测试数据
    torch.manual_seed(42)
    N, K = 16, 32
    weight = torch.randn(N, K)
    
    # 测试 2:8 剪枝
    config = SlideSparseConfig(Z=2, L=8)
    pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
    
    # 验证
    is_valid, valid_ratio = verify_ZL_sparsity(pruned, config.Z, config.L)
    
    print(f"  Input shape: {weight.shape}")
    print(f"  Config: {config.Z}:{config.L}")
    print(f"  Valid ratio: {valid_ratio:.2%}")
    print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
    
    assert is_valid, f"Pruning failed: valid_ratio={valid_ratio}"
    print()
    return True


def test_prune_magnitude_ordering():
    """测试 magnitude 模式是否保留最大值"""
    print("=" * 60)
    print("Test: Magnitude Ordering")
    print("=" * 60)
    
    # 创建有明确大小排序的测试数据
    N, K = 4, 8
    weight = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 应该保留 3-8，剪掉 1-2
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # 应该保留 3-8，剪掉 1-2
        [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0],  # 应该保留后 4 个
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 应该保留 0.3-0.8
    ])
    
    config = SlideSparseConfig(Z=2, L=8)
    pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
    
    # 检查每行
    for i in range(N):
        original_row = weight[i]
        pruned_row = pruned[i]
        
        # 统计保留的值
        kept_values = pruned_row[pruned_row != 0]
        pruned_values = original_row[pruned_row == 0]
        
        print(f"  Row {i}: kept={kept_values.tolist()}, pruned={pruned_values.tolist()}")
        
        # magnitude 模式应该保留较大的值
        if len(kept_values) > 0 and len(pruned_values) > 0:
            min_kept = kept_values.abs().min()
            max_pruned = pruned_values.abs().max()
            assert min_kept >= max_pruned, f"Row {i}: kept min {min_kept} < pruned max {max_pruned}"
    
    print("  Status: PASS")
    print()
    return True


def test_prune_different_L():
    """测试不同的 L 值"""
    print("=" * 60)
    print("Test: Different L Values")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 32, 64
    weight = torch.randn(N, K)
    
    for L in [6, 8, 10, 12]:
        config = SlideSparseConfig(Z=2, L=L)
        pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
        
        is_valid, valid_ratio = verify_ZL_sparsity(pruned, config.Z, config.L)
        
        # 计算实际稀疏度
        sparsity = (pruned == 0).sum().item() / pruned.numel()
        expected_sparsity = config.Z / config.L
        
        print(f"  L={L}: valid={is_valid}, sparsity={sparsity:.2%} (expected >= {expected_sparsity:.2%})")
        
        assert is_valid, f"L={L} failed"
        # 由于 padding 和边界效应，允许稍低于理论值
        assert sparsity >= expected_sparsity - 0.05, f"Sparsity too low for L={L}: {sparsity:.2%}"
    
    print("  Status: PASS")
    print()
    return True


def test_prune_dtype_preservation():
    """测试数据类型保持"""
    print("=" * 60)
    print("Test: Dtype Preservation")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 16, 32
    config = SlideSparseConfig(Z=2, L=8)
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    for dtype in dtypes:
        if dtype == torch.int8:
            weight = torch.randint(-127, 127, (N, K), dtype=dtype)
        else:
            weight = torch.randn(N, K).to(dtype)
        
        pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
        
        print(f"  {dtype}: input={weight.dtype}, output={pruned.dtype}")
        assert pruned.dtype == weight.dtype, f"Dtype mismatch: {pruned.dtype} != {weight.dtype}"
    
    print("  Status: PASS")
    print()
    return True


def test_prune_padding_handling():
    """测试 K 不能整除 L 的情况"""
    print("=" * 60)
    print("Test: Padding Handling")
    print("=" * 60)
    
    torch.manual_seed(42)
    config = SlideSparseConfig(Z=2, L=8)
    
    # 测试各种 K 值
    for K in [30, 31, 33, 35, 64, 100]:
        N = 16
        weight = torch.randn(N, K)
        
        pruned = prune_tensor(weight, config.Z, config.L, mode="magnitude")
        
        # 形状应该保持不变
        assert pruned.shape == weight.shape, f"Shape changed: {weight.shape} -> {pruned.shape}"
        
        # 仍然应该满足约束（考虑 padding）
        is_valid, valid_ratio = verify_ZL_sparsity(pruned, config.Z, config.L)
        
        print(f"  K={K}: shape={pruned.shape}, valid={is_valid}")
        assert is_valid, f"K={K} failed"
    
    print("  Status: PASS")
    print()
    return True


def test_bitnet_quant_and_prune():
    """测试 BitNet 量化+剪枝功能"""
    print("=" * 60)
    print("Test: BitNet Quant + Prune")
    print("=" * 60)
    
    from prune import quant_and_prune_tensor_bitnet
    
    torch.manual_seed(42)
    N, K = 64, 128
    
    # 模拟 BF16 权重
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    
    config = SlideSparseConfig(Z=2, L=8)
    
    # 测试不同输出格式
    for output_dtype in ["int8", "fp8_e4m3"]:
        pruned, scale = quant_and_prune_tensor_bitnet(
            weight, config.Z, config.L, mode="magnitude", output_dtype=output_dtype
        )
        
        # FP8 不支持 unique，转为 float
        pruned_float = pruned.float()
        unique_vals = torch.unique(pruned_float)
        is_ternary = all(v.item() in [-1, 0, 1] for v in unique_vals)
        
        is_valid, _ = verify_ZL_sparsity(pruned_float, config.Z, config.L)
        
        print(f"  {output_dtype}: dtype={pruned.dtype}, ternary={is_ternary}, valid={is_valid}")
        
        assert is_ternary, f"Output should be ternary for {output_dtype}"
        assert is_valid, f"Output should satisfy {config.Z}:{config.L} for {output_dtype}"
    
    print("  Status: PASS")
    print()
    return True


def test_prune_vs_bitnet():
    """测试普通剪枝和 BitNet 模式的区别"""
    print("=" * 60)
    print("Test: Prune vs BitNet Mode")
    print("=" * 60)
    
    from prune import quant_and_prune_tensor_bitnet
    
    torch.manual_seed(42)
    N, K = 32, 64
    config = SlideSparseConfig(Z=2, L=8)
    
    # INT8 输入（普通模式）
    int8_weight = torch.randint(-127, 127, (N, K), dtype=torch.int8)
    int8_pruned = prune_tensor(int8_weight, config.Z, config.L, mode="magnitude")
    
    # BF16 输入（BitNet 模式）
    bf16_weight = torch.randn(N, K, dtype=torch.bfloat16)
    bf16_pruned, scale = quant_and_prune_tensor_bitnet(
        bf16_weight, config.Z, config.L, mode="magnitude", output_dtype="int8"
    )
    
    print(f"  INT8 prune:")
    print(f"    Dtype: {int8_pruned.dtype}")
    print(f"    Unique values: {len(torch.unique(int8_pruned))}")
    
    print(f"  BitNet quant+prune:")
    print(f"    Dtype: {bf16_pruned.dtype}")
    print(f"    Unique values: {len(torch.unique(bf16_pruned))} (should be 3: -1, 0, 1)")
    print(f"    Scale: {scale.item():.4f}")
    
    # 普通剪枝保留原值，BitNet 量化为三元
    assert len(torch.unique(int8_pruned)) > 3, "INT8 prune should keep original values"
    assert len(torch.unique(bf16_pruned)) <= 3, "BitNet should produce ternary values"
    
    print("  Status: PASS")
    print()
    return True


def test_bitnet_reuses_quant_zeros():
    """
    关键测试：验证 BitNet 量化+剪枝是否正确利用了量化产生的零
    
    核心问题：
    - 如果先剪枝 BF16，再量化，稀疏度会超过预期（quant 又产生额外的零）
    - 正确做法：先量化得到 ternary，然后只对 ternary 非零元素进行剪枝
    
    验证方法：
    1. 构造一个特殊的权重，量化后在每组 L 元素中产生恰好 Z 个零
    2. 此时不需要额外剪枝，稀疏度应该恰好是 Z/L
    3. 如果实现错误，会强制剪掉 Z 个非零元素，导致稀疏度变成 2Z/L
    """
    print("=" * 60)
    print("Test: BitNet Reuses Quant Zeros (CRITICAL)")
    print("=" * 60)
    
    from prune import quant_and_prune_tensor_bitnet
    
    Z, L = 2, 8
    N, K = 4, 16  # 2 groups per row
    
    # 构造特殊权重：
    # 量化后每组 8 个元素中，恰好 2 个变成 0（小值），6 个变成 ±1（大值）
    # scale = 1 / mean(|w|) ≈ 1.25，乘以 scale 后：
    #   小值 (0.1) * 1.25 ≈ 0.125 -> round -> 0
    #   大值 (1.0) * 1.25 ≈ 1.25  -> round -> 1
    
    # 每行 16 个元素，分成 2 组
    # 每组：2 个小值 (会量化为 0) + 6 个大值 (会量化为 ±1)
    small_val = 0.1
    large_val = 1.0
    
    # 第一组: [small, small, large, large, large, large, large, large]
    # 第二组: [large, large, small, small, large, large, large, large]
    row = torch.tensor([
        small_val, small_val, large_val, -large_val, large_val, -large_val, large_val, large_val,  # 组1
        large_val, -large_val, small_val, small_val, large_val, -large_val, large_val, large_val,  # 组2
    ], dtype=torch.bfloat16)
    
    weight = row.unsqueeze(0).expand(N, -1).clone()
    
    # 先手动计算量化结果，验证假设
    weight_float = weight.float()
    scale = 1.0 / weight_float.abs().mean().clamp(min=1e-5)
    ternary_manual = (weight_float * scale).round().clamp(-1, 1)
    
    # 统计量化后的零
    quant_zeros = (ternary_manual == 0).sum().item()
    total_elements = N * K
    quant_sparsity = quant_zeros / total_elements
    
    print(f"  Input weight: mean={weight_float.abs().mean():.3f}, scale={scale:.3f}")
    print(f"  After quant: {quant_zeros} zeros out of {total_elements} = {quant_sparsity:.1%} sparsity")
    print(f"  Expected quant zeros per group: ~{Z} (because small_val * scale < 0.5)")
    
    # 执行量化+剪枝
    pruned, scale_out = quant_and_prune_tensor_bitnet(
        weight, Z, L, mode="magnitude", output_dtype="int8"
    )
    
    # 统计最终的零
    final_zeros = (pruned == 0).sum().item()
    final_sparsity = final_zeros / total_elements
    
    print(f"  After quant+prune: {final_zeros} zeros = {final_sparsity:.1%} sparsity")
    print(f"  Target sparsity: {Z}/{L} = {Z/L:.1%}")
    
    # 验证结果
    # 如果正确利用了量化产生的零，最终稀疏度应该接近 Z/L = 25%
    # 如果没有利用（先剪枝再量化），稀疏度会接近 2*Z/L = 50%
    
    expected_sparsity = Z / L
    tolerance = 0.1  # 允许 10% 的误差
    
    # 关键断言：最终稀疏度应该接近 Z/L，而不是 2*Z/L
    sparsity_ok = abs(final_sparsity - expected_sparsity) <= tolerance
    
    if not sparsity_ok:
        print(f"  ERROR: Sparsity {final_sparsity:.1%} differs from expected {expected_sparsity:.1%}")
        print(f"  This suggests quant zeros are NOT being reused!")
        
    assert sparsity_ok, \
        f"Sparsity {final_sparsity:.1%} should be close to {expected_sparsity:.1%}, " \
        f"not {2*expected_sparsity:.1%}. Quant zeros must be reused!"
    
    # 额外验证：对于量化后已经满足 Z:L 约束的组，不应该有额外剪枝
    # 检查每组的零数量
    pruned_grouped = pruned.view(-1, L)
    zeros_per_group = (pruned_grouped == 0).sum(dim=1)
    
    print(f"  Zeros per group: min={zeros_per_group.min()}, max={zeros_per_group.max()}, mean={zeros_per_group.float().mean():.2f}")
    
    # 每组应该恰好有 Z 个零（因为量化已经产生了足够的零）
    assert (zeros_per_group >= Z).all(), "Some groups have fewer than Z zeros!"
    
    print("  Status: PASS - Quant zeros are correctly reused!")
    print()
    return True


def test_bitnet_additional_prune_when_needed():
    """
    测试：当量化产生的零不足 Z 个时，需要额外剪枝
    
    验证：
    - 如果量化后某组只有 1 个零，但 Z=2，应该额外剪掉 1 个最小的非零元素
    """
    print("=" * 60)
    print("Test: BitNet Additional Prune When Needed")
    print("=" * 60)
    
    from prune import quant_and_prune_tensor_bitnet
    
    Z, L = 2, 8
    
    # 构造权重：所有值都是 1.0，量化后全部变成 1，没有天然的零
    # 此时必须强制剪枝
    weight = torch.ones(4, 16, dtype=torch.bfloat16)
    
    # 先检查量化结果
    weight_float = weight.float()
    scale = 1.0 / weight_float.abs().mean().clamp(min=1e-5)
    ternary_manual = (weight_float * scale).round().clamp(-1, 1)
    quant_zeros = (ternary_manual == 0).sum().item()
    
    print(f"  Input: all 1.0, scale={scale:.3f}")
    print(f"  After quant: {quant_zeros} zeros (all become 1)")
    
    # 执行量化+剪枝
    pruned, _ = quant_and_prune_tensor_bitnet(
        weight, Z, L, mode="magnitude", output_dtype="int8"
    )
    
    # 此时必须强制剪枝 Z 个元素
    final_zeros = (pruned == 0).sum().item()
    expected_zeros = (weight.numel() // L) * Z  # 每组 Z 个零
    
    print(f"  After quant+prune: {final_zeros} zeros")
    print(f"  Expected: {expected_zeros} zeros (Z per group)")
    
    assert final_zeros == expected_zeros, \
        f"Should have {expected_zeros} zeros but got {final_zeros}"
    
    # 验证 Z:L 约束
    is_valid, valid_ratio = verify_ZL_sparsity(pruned.float(), Z, L)
    assert is_valid, "Should satisfy Z:L constraint"
    
    print("  Status: PASS - Additional pruning works correctly!")
    print()
    return True


def main():
    """运行所有测试"""
    tests = [
        test_prune_basic,
        test_prune_magnitude_ordering,
        test_prune_different_L,
        test_prune_dtype_preservation,
        test_prune_padding_handling,
        test_bitnet_quant_and_prune,
        test_prune_vs_bitnet,
        test_bitnet_reuses_quant_zeros,
        test_bitnet_additional_prune_when_needed,
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
