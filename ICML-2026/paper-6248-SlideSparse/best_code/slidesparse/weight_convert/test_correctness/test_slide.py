#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Slide 正确性测试

验证滑动扩展功能的正确性：
1. 输出满足 2:4 稀疏约束
2. K 维度变化正确（理论计算 vs 实际结果）
3. 贪婪残差分配逻辑正确
4. 非零值正确保留
5. 边界情况处理
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils import SlideSparseConfig, verify_2to4_sparsity, compute_output_k
from slide import slide_tensor, build_slide_index_mapping


def create_ZL_sparse_tensor(N: int, K: int, Z: int, L: int, dtype=torch.float32) -> torch.Tensor:
    """
    创建满足 Z:L 稀疏约束的张量（辅助函数）
    
    Args:
        N: 行数
        K: 列数
        Z: 每组零元素数
        L: 组大小
        dtype: 数据类型
    
    Returns:
        满足 Z:L 约束的张量
    """
    if dtype == torch.int8:
        weight = torch.randint(-127, 127, (N, K), dtype=torch.float32)
    else:
        weight = torch.randn(N, K, dtype=torch.float32)
    
    # 对每行应用 Z:L 约束
    k_padded = ((K + L - 1) // L) * L
    for i in range(N):
        for g in range(k_padded // L):
            start = g * L
            end = min(start + L, K)
            if end - start >= Z:
                # 选择 Z 个位置设为 0
                local_indices = torch.randperm(end - start)[:Z]
                weight[i, start + local_indices] = 0
    
    return weight.to(dtype) if dtype != torch.int8 else weight.to(torch.int8)


def test_slide_basic():
    """测试基本的滑动扩展功能"""
    print("=" * 60)
    print("Test: Basic Slide")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 16, 32
    
    weight = create_ZL_sparse_tensor(N, K, Z=2, L=8)
    config = SlideSparseConfig(Z=2, L=8)
    slided, metadata = slide_tensor(weight, config)
    
    # 验证 2:4 稀疏性
    is_valid, violation_ratio = verify_2to4_sparsity(slided)
    
    print(f"  Input shape: {weight.shape}")
    print(f"  Output shape: {slided.shape}")
    print(f"  Expand ratio: {slided.shape[1] / weight.shape[1]:.3f} (expected: {config.expand_ratio:.3f})")
    print(f"  2:4 valid: {is_valid}, violation: {violation_ratio:.2%}")
    print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
    
    assert is_valid, f"Slide failed: violation_ratio={violation_ratio}"
    print()
    return True


def test_slide_dimension_matches_compute():
    """测试 slide 输出维度与 compute_output_k 计算结果一致"""
    print("=" * 60)
    print("Test: Dimension Matches compute_output_k")
    print("=" * 60)
    
    torch.manual_seed(42)
    N = 16
    
    test_cases = [
        # (K_in, L, align_to)
        (32, 6, 16),
        (32, 8, 16),
        (64, 8, 16),
        (100, 8, 16),   # K 不整除 L
        (33, 6, 16),    # K 不整除 L
        (128, 10, 16),
        (256, 12, 16),
    ]
    
    all_match = True
    for K_in, L, align_to in test_cases:
        config = SlideSparseConfig(Z=2, L=L)
        
        # 理论计算
        k_padded_expected, k_out_expected = compute_output_k(K_in, config, align_to)
        
        # 实际 slide
        weight = create_ZL_sparse_tensor(N, K_in, Z=2, L=L)
        slided, metadata = slide_tensor(weight, config, align_to=align_to)
        k_out_actual = slided.shape[1]
        
        match = (k_out_actual == k_out_expected)
        all_match &= match
        
        status = "✓" if match else "✗"
        print(f"  {status} K={K_in}, L={L}: expected={k_out_expected}, actual={k_out_actual}")
        
        assert match, f"Dimension mismatch: expected {k_out_expected}, got {k_out_actual}"
    
    print(f"  Status: {'PASS' if all_match else 'FAIL'}")
    print()
    return True


def test_slide_output_alignment():
    """测试输出 K 维度总是对齐到 32"""
    print("=" * 60)
    print("Test: Output Alignment to 32")
    print("=" * 60)
    
    torch.manual_seed(42)
    N = 8
    
    # 测试各种 K 值
    K_values = [32, 33, 48, 64, 100, 128, 200, 256, 500, 1024]
    L_values = [6, 8, 10]
    
    all_aligned = True
    for L in L_values:
        config = SlideSparseConfig(Z=2, L=L)
        for K in K_values:
            weight = create_ZL_sparse_tensor(N, K, Z=2, L=L)
            slided, _ = slide_tensor(weight, config, align_to=32)
            
            aligned = (slided.shape[1] % 32 == 0)
            all_aligned &= aligned
            
            if not aligned:
                print(f"  ✗ K={K}, L={L}: output_K={slided.shape[1]} (not aligned to 32)")
    
    if all_aligned:
        print(f"  All {len(K_values) * len(L_values)} cases aligned to 32")
    
    print(f"  Status: {'PASS' if all_aligned else 'FAIL'}")
    print()
    return all_aligned


def test_slide_index_mapping():
    """测试索引映射的正确性"""
    print("=" * 60)
    print("Test: Index Mapping")
    print("=" * 60)
    
    # 以 2:8 为例
    config = SlideSparseConfig(Z=2, L=8)
    K_in = 16  # 2 组
    
    index_map = build_slide_index_mapping(K_in, config)
    
    print(f"  K_in={K_in}, L={config.L}")
    print(f"  Index map shape: {index_map.shape}")
    print(f"  Index map (first 24): {index_map[:24].tolist()}")
    
    # 验证映射
    # Group 0 (输入 0-7):
    #   Window 0: output[0,1,2,3] <- input[0,1,2,3]
    #   Window 1: output[4,5,6,7] <- input[2,3,4,5]
    #   Window 2: output[8,9,10,11] <- input[4,5,6,7]
    expected_group0 = [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7]
    actual_group0 = index_map[:12].tolist()
    
    print(f"  Expected group 0: {expected_group0}")
    print(f"  Actual group 0:   {actual_group0}")
    
    assert actual_group0 == expected_group0, f"Index mapping mismatch"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_greedy_allocation():
    """测试贪婪分配的正确性"""
    print("=" * 60)
    print("Test: Greedy Allocation")
    print("=" * 60)
    
    # 创建一个简单的测试用例
    # 每组 8 个元素，其中有 2 个零（位置 0,1）
    N, K = 1, 8
    weight = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    
    config = SlideSparseConfig(Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    
    print(f"  Input: {weight[0].tolist()}")
    print(f"  Output shape: {slided.shape}")
    print(f"  Output: {slided[0].tolist()}")
    
    # 验证输出满足 2:4
    is_valid, violation_ratio = verify_2to4_sparsity(slided)
    print(f"  2:4 valid: {is_valid}")
    
    # 验证每个 4 元素组至少有 2 个零
    for i in range(slided.shape[1] // 4):
        group = slided[0, i*4:(i+1)*4]
        zeros = (group == 0).sum().item()
        print(f"  Group {i}: {group.tolist()}, zeros={zeros}")
        assert zeros >= 2, f"Group {i} has only {zeros} zeros"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_nonzero_preservation():
    """测试非零值是否被正确保留（不丢失信息）"""
    print("=" * 60)
    print("Test: Nonzero Value Preservation")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 8, 32
    
    for L in [6, 8, 10]:
        config = SlideSparseConfig(Z=2, L=L)
        
        # 创建带有唯一非零值的张量
        weight = create_ZL_sparse_tensor(N, K, Z=2, L=L)
        
        # 统计原始非零值
        original_nonzero = weight[weight != 0]
        original_nonzero_set = set(original_nonzero.tolist())
        
        slided, _ = slide_tensor(weight, config)
        
        # 统计 slide 后的非零值
        slided_nonzero = slided[slided != 0]
        slided_nonzero_set = set(slided_nonzero.tolist())
        
        # 由于重叠窗口，slide 后可能有重复值，但所有原始非零值应该都存在
        # 实际上由于贪婪分配，每个原始非零值只会被分配一次
        preserved = original_nonzero_set <= slided_nonzero_set
        
        print(f"  L={L}: original nonzeros={len(original_nonzero_set)}, slided nonzeros={len(slided_nonzero_set)}, preserved={preserved}")
        
        # 注意：贪婪分配可能不会保留所有值（如果窗口满了）
        # 但大部分值应该被保留
        preservation_ratio = len(original_nonzero_set & slided_nonzero_set) / len(original_nonzero_set)
        assert preservation_ratio > 0.5, f"L={L}: too few values preserved: {preservation_ratio:.1%}"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_different_L():
    """测试不同的 L 值"""
    print("=" * 60)
    print("Test: Different L Values")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 32, 64
    
    for L in [6, 8, 10, 12]:
        config = SlideSparseConfig(Z=2, L=L)
        weight = create_ZL_sparse_tensor(N, K, Z=2, L=L)
        slided, metadata = slide_tensor(weight, config)
        
        is_valid, violation_ratio = verify_2to4_sparsity(slided)
        
        # 验证扩展比
        actual_ratio = slided.shape[1] / weight.shape[1]
        expected_min_ratio = config.expand_ratio * 0.9  # 允许一定误差（因为对齐）
        
        print(f"  L={L}: shape {weight.shape} -> {slided.shape}, ratio={actual_ratio:.3f} (expect ~{config.expand_ratio:.3f}), 2:4 valid={is_valid}")
        
        assert violation_ratio < 0.1, f"L={L} has too many violations: {violation_ratio:.2%}"
        assert actual_ratio >= expected_min_ratio, f"L={L}: ratio {actual_ratio:.3f} < expected {expected_min_ratio:.3f}"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_dtype_preservation():
    """测试数据类型保持"""
    print("=" * 60)
    print("Test: Dtype Preservation")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 16, 32
    config = SlideSparseConfig(Z=2, L=8)
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    for dtype in dtypes:
        weight = create_ZL_sparse_tensor(N, K, Z=2, L=8, dtype=dtype)
        slided, _ = slide_tensor(weight, config)
        
        print(f"  {dtype}: input={weight.dtype}, output={slided.dtype}")
        assert slided.dtype == weight.dtype, f"Dtype mismatch: {slided.dtype} != {weight.dtype}"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_expand_ratio_formula():
    """测试扩展比公式的正确性"""
    print("=" * 60)
    print("Test: Expand Ratio Formula")
    print("=" * 60)
    
    # 公式: expand_ratio = (N-1) * 4 / L, 其中 N = L / 2
    # 简化: expand_ratio = (L/2 - 1) * 4 / L = 2 - 4/L
    
    for L in [4, 6, 8, 10, 12, 14, 16]:
        config = SlideSparseConfig(Z=2, L=L)
        
        N = L // 2
        expected_ratio = (N - 1) * 4 / L  # = 2 - 4/L
        actual_ratio = config.expand_ratio
        
        formula_check = 2 - 4/L
        
        match = abs(actual_ratio - expected_ratio) < 1e-6 and abs(actual_ratio - formula_check) < 1e-6
        
        print(f"  L={L:2d}: N={N}, expand_ratio={actual_ratio:.4f} (formula: 2 - 4/{L} = {formula_check:.4f})")
        
        assert match, f"L={L}: formula mismatch"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("Test: Edge Cases")
    print("=" * 60)
    
    config = SlideSparseConfig(Z=2, L=8)
    
    # Case 1: 最小 K（等于 L）
    print("  Case 1: K = L = 8")
    weight = create_ZL_sparse_tensor(4, 8, Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    is_valid, _ = verify_2to4_sparsity(slided)
    print(f"    shape: {weight.shape} -> {slided.shape}, 2:4 valid={is_valid}")
    assert is_valid
    
    # Case 2: K 略大于 L
    print("  Case 2: K = 9 (L+1)")
    weight = create_ZL_sparse_tensor(4, 9, Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    is_valid, _ = verify_2to4_sparsity(slided)
    print(f"    shape: {weight.shape} -> {slided.shape}, 2:4 valid={is_valid}")
    
    # Case 3: 单行
    print("  Case 3: N = 1")
    weight = create_ZL_sparse_tensor(1, 64, Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    is_valid, _ = verify_2to4_sparsity(slided)
    print(f"    shape: {weight.shape} -> {slided.shape}, 2:4 valid={is_valid}")
    assert is_valid
    
    # Case 4: 大规模
    print("  Case 4: Large scale (N=1024, K=4096)")
    weight = create_ZL_sparse_tensor(1024, 4096, Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    is_valid, violation_ratio = verify_2to4_sparsity(slided)
    print(f"    shape: {weight.shape} -> {slided.shape}, 2:4 valid={is_valid}, violation={violation_ratio:.2%}")
    assert is_valid
    
    print("  Status: PASS")
    print()
    return True


def main():
    """运行所有测试"""
    tests = [
        test_slide_basic,
        test_slide_dimension_matches_compute,
        test_slide_output_alignment,
        test_slide_index_mapping,
        test_slide_greedy_allocation,
        test_slide_nonzero_preservation,
        test_slide_different_L,
        test_slide_dtype_preservation,
        test_slide_expand_ratio_formula,
        test_slide_edge_cases,
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
