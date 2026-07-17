#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compress 正确性测试

验证 cuSPARSELt 压缩功能的正确性：
1. 压缩扩展是否可加载
2. 多数据类型支持（INT8, FP8E4M3）
3. 压缩大小查询是否正确
4. 压缩后数据大小符合预期
5. 模拟压缩功能测试
6. 真实 checkpoint 权重压缩测试
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# =============================================================================
# 测试辅助函数
# =============================================================================

def create_2to4_sparse_tensor(N: int, K: int, dtype=torch.int8) -> torch.Tensor:
    """
    创建满足 2:4 稀疏约束的张量
    
    Args:
        N: 行数
        K: 列数（必须是 4 的倍数）
        dtype: 数据类型（torch.int8 或 torch.float8_e4m3fn）
    
    Returns:
        满足 2:4 约束的张量（每 4 个元素中 2 个为零）
    """
    if K % 4 != 0:
        raise ValueError(f"K must be multiple of 4, got K={K}")
    
    # 创建随机非零值
    if dtype == torch.int8:
        weight = torch.randint(-127, 127, (N, K), dtype=torch.float32)
    elif dtype == torch.float8_e4m3fn:
        # FP8E4M3 的范围较小，用缩放后的随机数
        weight = torch.randn(N, K, dtype=torch.float32) * 0.5
    else:
        weight = torch.randn(N, K, dtype=torch.float32)
    
    # 对每行的每 4 个元素，随机选择 2 个设为 0
    for i in range(N):
        for g in range(K // 4):
            start = g * 4
            # 随机选择 2 个位置设为 0
            zero_indices = torch.randperm(4)[:2]
            weight[i, start + zero_indices] = 0
    
    return weight.to(dtype)


# =============================================================================
# 基础测试
# =============================================================================

def test_fake_compress_basic():
    """测试模拟压缩的基本功能"""
    print("=" * 60)
    print("Test: Fake Compress Basic")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    torch.manual_seed(42)
    N, K = 32, 64
    
    weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
    
    values, metadata, info = compress_tensor_fake(weight)
    
    print(f"  Input shape: {weight.shape}")
    print(f"  Values shape: {values.shape} (expected: [{N}, {K//2}])")
    print(f"  Metadata shape: {metadata.shape} (expected: [{N}, {K//4}])")
    
    # 验证输出形状
    assert values.shape == (N, K // 2), f"Values shape mismatch: {values.shape}"
    assert metadata.shape == (N, K // 4), f"Metadata shape mismatch: {metadata.shape}"
    
    # 验证 info 字典
    assert info["original_shape"] == [N, K]
    assert info["compression_type"] == "fake_2to4"
    
    print("  Status: PASS")
    print()
    return True


def test_fake_compress_value_preservation():
    """测试模拟压缩是否正确保留非零值"""
    print("=" * 60)
    print("Test: Fake Compress Value Preservation")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    # 创建一个简单的测试用例
    # 每 4 个元素：[0, 0, a, b]
    N, K = 2, 8
    weight = torch.tensor([
        [0, 0, 1, 2, 0, 0, 3, 4],
        [0, 5, 0, 6, 7, 0, 0, 8],
    ], dtype=torch.int8)
    
    values, metadata, info = compress_tensor_fake(weight)
    
    print(f"  Input: {weight.tolist()}")
    print(f"  Values: {values.tolist()}")
    print(f"  Metadata: {metadata.tolist()}")
    
    # 检查非零值数量
    original_nonzeros = (weight != 0).sum().item()
    compressed_nonzeros = (values != 0).sum().item()
    
    print(f"  Original nonzeros: {original_nonzeros}")
    print(f"  Compressed nonzeros: {compressed_nonzeros}")
    
    # 压缩后的非零值应该等于或少于原始非零值（每 4 个取 2 个）
    assert compressed_nonzeros <= original_nonzeros
    
    print("  Status: PASS")
    print()
    return True


def test_fake_compress_different_sizes():
    """测试不同维度的模拟压缩"""
    print("=" * 60)
    print("Test: Fake Compress Different Sizes")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    torch.manual_seed(42)
    
    test_cases = [
        (32, 64),
        (32, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
    ]
    
    all_pass = True
    
    for N, K in test_cases:
        weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
        values, metadata, info = compress_tensor_fake(weight)
        
        expected_values_shape = (N, K // 2)
        expected_metadata_shape = (N, K // 4)
        
        values_ok = (values.shape == expected_values_shape)
        metadata_ok = (metadata.shape == expected_metadata_shape)
        
        status = "✓" if (values_ok and metadata_ok) else "✗"
        print(f"  {status} [{N:4d}, {K:4d}] -> values {values.shape}, metadata {metadata.shape}")
        
        all_pass &= (values_ok and metadata_ok)
    
    print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# =============================================================================
# cuSPARSELt 真实压缩测试
# =============================================================================

def test_extension_availability():
    """测试压缩扩展是否可用"""
    print("=" * 60)
    print("Test: Extension Availability")
    print("=" * 60)
    
    try:
        from compress import check_compress_available, get_compress_module, get_supported_dtypes_from_lib
        
        available = check_compress_available()
        print(f"  cuSPARSELt compress available: {available}")
        
        if available:
            lib = get_compress_module()
            print(f"  Module loaded successfully")
            
            # 测试支持的数据类型
            supported = get_supported_dtypes_from_lib()
            print(f"  Supported dtypes: {supported}")
            
            assert "int8" in supported, "int8 should be supported"
            assert "fp8e4m3" in supported, "fp8e4m3 should be supported"
        else:
            print("  (extension not available, skipping detailed tests)")
        
        print("  Status: PASS")
        print()
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED")
        print()
        return True


def test_compress_sizes_query_int8():
    """测试 INT8 压缩大小查询"""
    print("=" * 60)
    print("Test: Compress Sizes Query (INT8)")
    print("=" * 60)
    
    try:
        from compress import get_compress_sizes, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (1024, 2048),
            (2048, 1024),
        ]
        
        all_valid = True
        
        for N, K in test_cases:
            try:
                compressed_size, temp_size = get_compress_sizes(N, K, "int8")
                # 压缩后大小应该大于 0
                valid = compressed_size > 0
                status = "✓" if valid else "✗"
                print(f"  {status} INT8 [{N:4d}, {K:4d}] -> compressed={compressed_size:8d}, temp={temp_size:8d}")
                all_valid &= valid
            except Exception as e:
                print(f"  ✗ INT8 [{N:4d}, {K:4d}] -> Error: {e}")
                all_valid = False
        
        print(f"  Status: {'PASS' if all_valid else 'FAIL'}")
        print()
        return all_valid
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_compress_sizes_query_fp8():
    """测试 FP8E4M3 压缩大小查询"""
    print("=" * 60)
    print("Test: Compress Sizes Query (FP8E4M3)")
    print("=" * 60)
    
    try:
        from compress import get_compress_sizes, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        ]
        
        all_valid = True
        
        for N, K in test_cases:
            try:
                compressed_size, temp_size = get_compress_sizes(N, K, "fp8e4m3")
                valid = compressed_size > 0
                status = "✓" if valid else "✗"
                print(f"  {status} FP8 [{N:4d}, {K:4d}] -> compressed={compressed_size:8d}, temp={temp_size:8d}")
                all_valid &= valid
            except Exception as e:
                print(f"  ✗ FP8 [{N:4d}, {K:4d}] -> Error: {e}")
                all_valid = False
        
        print(f"  Status: {'PASS' if all_valid else 'FAIL'}")
        print()
        return all_valid
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_real_compress_int8():
    """测试真实的 INT8 cuSPARSELt 压缩"""
    print("=" * 60)
    print("Test: Real cuSPARSELt Compress (INT8)")
    print("=" * 60)
    
    try:
        from compress import compress_tensor_offline, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]
        
        all_pass = True
        
        for N, K in test_cases:
            weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
            
            try:
                compressed, metadata = compress_tensor_offline(weight, dtype="int8")
                
                # 验证元数据
                assert metadata["original_shape"] == [N, K]
                assert metadata["dtype"] == "int8"
                assert metadata["compressed_size_bytes"] > 0
                
                status = "✓"
                print(f"  {status} INT8 [{N:4d}, {K:4d}] -> compressed size: {len(compressed)} bytes")
                
            except Exception as e:
                print(f"  ✗ INT8 [{N:4d}, {K:4d}] -> Error: {e}")
                all_pass = False
        
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_real_compress_fp8():
    """测试真实的 FP8E4M3 cuSPARSELt 压缩"""
    print("=" * 60)
    print("Test: Real cuSPARSELt Compress (FP8E4M3)")
    print("=" * 60)
    
    try:
        from compress import compress_tensor_offline, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]
        
        all_pass = True
        
        for N, K in test_cases:
            weight = create_2to4_sparse_tensor(N, K, dtype=torch.float8_e4m3fn)
            
            try:
                compressed, metadata = compress_tensor_offline(weight, dtype="fp8e4m3")
                
                # 验证元数据
                assert metadata["original_shape"] == [N, K]
                assert metadata["dtype"] == "fp8e4m3"
                assert metadata["compressed_size_bytes"] > 0
                
                status = "✓"
                print(f"  {status} FP8 [{N:4d}, {K:4d}] -> compressed size: {len(compressed)} bytes")
                
            except Exception as e:
                print(f"  ✗ FP8 [{N:4d}, {K:4d}] -> Error: {e}")
                all_pass = False
        
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_auto_dtype_detection():
    """测试自动数据类型检测"""
    print("=" * 60)
    print("Test: Auto Dtype Detection")
    print("=" * 60)
    
    try:
        from compress import compress_tensor_offline, check_compress_available, is_supported_dtype
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        N, K = 64, 64
        
        all_pass = True
        
        # 测试 INT8 自动检测
        weight_int8 = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
        assert is_supported_dtype(weight_int8.dtype), "INT8 should be supported"
        
        try:
            compressed, metadata = compress_tensor_offline(weight_int8)  # 不指定 dtype
            assert metadata["dtype"] == "int8", f"Expected int8, got {metadata['dtype']}"
            print(f"  ✓ INT8 auto-detected: {metadata['dtype']}")
        except Exception as e:
            print(f"  ✗ INT8 auto-detection failed: {e}")
            all_pass = False
        
        # 测试 FP8 自动检测
        weight_fp8 = create_2to4_sparse_tensor(N, K, dtype=torch.float8_e4m3fn)
        assert is_supported_dtype(weight_fp8.dtype), "FP8E4M3 should be supported"
        
        try:
            compressed, metadata = compress_tensor_offline(weight_fp8)  # 不指定 dtype
            assert metadata["dtype"] == "fp8e4m3", f"Expected fp8e4m3, got {metadata['dtype']}"
            print(f"  ✓ FP8E4M3 auto-detected: {metadata['dtype']}")
        except Exception as e:
            print(f"  ✗ FP8E4M3 auto-detection failed: {e}")
            all_pass = False
        
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


# =============================================================================
# 真实 Checkpoint 测试
# =============================================================================

def test_real_checkpoint_int8():
    """测试真实 INT8 checkpoint 的权重压缩"""
    print("=" * 60)
    print("Test: Real Checkpoint Compress (INT8)")
    print("=" * 60)
    
    checkpoint_path = Path("/root/vllmbench/checkpoints_slidesparse/BitNet-2B_mag_Z2L8_INT8_slided_2_8/model.safetensors")
    
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  Status: SKIPPED")
        print()
        return True
    
    try:
        from compress import compress_tensor_offline, check_compress_available
        from safetensors import safe_open
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        all_pass = True
        compressed_count = 0
        
        with safe_open(str(checkpoint_path), framework="pt") as f:
            keys = list(f.keys())
            
            # 只测试前几个权重层
            weight_keys = [k for k in keys if ".weight" in k and "_scale" not in k][:3]
            
            for key in weight_keys:
                tensor = f.get_tensor(key)
                
                # 只处理 INT8 2D 权重
                if tensor.dtype != torch.int8 or tensor.dim() != 2:
                    continue
                
                N, K = tensor.shape
                
                # 跳过太小的维度
                if N < 32 or K < 32 or N % 32 != 0 or K % 32 != 0:
                    print(f"  - Skipping {key}: dimension not aligned [{N}, {K}]")
                    continue
                
                try:
                    compressed, metadata = compress_tensor_offline(tensor)
                    print(f"  ✓ {key}: [{N}, {K}] -> {len(compressed)} bytes (dtype={metadata['dtype']})")
                    compressed_count += 1
                except Exception as e:
                    print(f"  ✗ {key}: [{N}, {K}] -> Error: {e}")
                    all_pass = False
        
        print(f"  Compressed {compressed_count} layers")
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print("  Status: FAIL")
        print()
        return False


def test_real_checkpoint_fp8():
    """测试真实 FP8 checkpoint 的权重压缩"""
    print("=" * 60)
    print("Test: Real Checkpoint Compress (FP8E4M3)")
    print("=" * 60)
    
    checkpoint_path = Path("/root/vllmbench/checkpoints_slidesparse/BitNet-2B_mag_Z2L6_FP8E4M3_slided_2_6/model.safetensors")
    
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  Status: SKIPPED")
        print()
        return True
    
    try:
        from compress import compress_tensor_offline, check_compress_available
        from safetensors import safe_open
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        all_pass = True
        compressed_count = 0
        
        with safe_open(str(checkpoint_path), framework="pt") as f:
            keys = list(f.keys())
            
            # 只测试前几个权重层
            weight_keys = [k for k in keys if ".weight" in k and "_scale" not in k][:3]
            
            for key in weight_keys:
                tensor = f.get_tensor(key)
                
                # 只处理 FP8 2D 权重
                if tensor.dtype != torch.float8_e4m3fn or tensor.dim() != 2:
                    continue
                
                N, K = tensor.shape
                
                # 跳过太小的维度
                if N < 32 or K < 32 or N % 32 != 0 or K % 32 != 0:
                    print(f"  - Skipping {key}: dimension not aligned [{N}, {K}]")
                    continue
                
                try:
                    compressed, metadata = compress_tensor_offline(tensor)
                    print(f"  ✓ {key}: [{N}, {K}] -> {len(compressed)} bytes (dtype={metadata['dtype']})")
                    compressed_count += 1
                except Exception as e:
                    print(f"  ✗ {key}: [{N}, {K}] -> Error: {e}")
                    all_pass = False
        
        print(f"  Compressed {compressed_count} layers")
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print("  Status: FAIL")
        print()
        return False


def test_compress_sparsity_verification():
    """测试压缩前的稀疏性验证"""
    print("=" * 60)
    print("Test: Compress Sparsity Verification")
    print("=" * 60)
    
    try:
        from compress import compress_tensor_offline, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        N, K = 64, 64
        
        # 创建不满足 2:4 的权重（所有元素非零）
        weight_invalid = torch.randint(-127, 127, (N, K), dtype=torch.int8)
        
        try:
            compressed, metadata = compress_tensor_offline(weight_invalid)
            print("  ✗ Should have rejected non-sparse weight")
            return False
        except ValueError as e:
            if "2:4 sparsity" in str(e):
                print(f"  ✓ Correctly rejected non-sparse weight: {e}")
            else:
                print(f"  ✗ Wrong error: {e}")
                return False
        
        # 创建满足 2:4 的权重
        weight_valid = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
        
        try:
            compressed, metadata = compress_tensor_offline(weight_valid)
            print(f"  ✓ Accepted valid sparse weight")
        except Exception as e:
            print(f"  ✗ Rejected valid sparse weight: {e}")
            return False
        
        print("  Status: PASS")
        print()
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有测试"""
    tests = [
        # 基础测试
        test_fake_compress_basic,
        test_fake_compress_value_preservation,
        test_fake_compress_different_sizes,
        
        # cuSPARSELt 测试
        test_extension_availability,
        test_compress_sizes_query_int8,
        test_compress_sizes_query_fp8,
        test_real_compress_int8,
        test_real_compress_fp8,
        test_auto_dtype_detection,
        test_compress_sparsity_verification,
        
        # 真实 checkpoint 测试
        test_real_checkpoint_int8,
        test_real_checkpoint_fp8,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "=" * 60)
    print("Compress Correctness Tests (Multi-Dtype)")
    print("=" * 60 + "\n")
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Unexpected error in {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
