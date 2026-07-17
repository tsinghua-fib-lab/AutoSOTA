#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuSPARSELt 压缩脚本

将满足 2:4 稀疏约束的权重使用 cuSPARSELt 压缩为硬件加速格式。

支持的数据类型：
- INT8:     torch.int8 -> CUDA_R_8I,      计算类型 CUSPARSE_COMPUTE_32I
- FP8E4M3:  torch.float8_e4m3fn -> CUDA_R_8F_E4M3, 计算类型 CUSPARSE_COMPUTE_32F

压缩原理：
- cuSPARSELt 的 2:4 结构化稀疏将每 4 个元素压缩为 2 个非零值 + 元数据
- K 维度减半：[N, K] -> 压缩后约 [N, K/2] + 元数据

布局约定（TN-CC 格式）：
- D = W^T × A
- W: 稀疏权重 [N, K]，行主序，转置 (opW = T)
- A: 稠密激活 [K, M]，列主序，不转置 (opA = N)  
- D: 输出 [N, M]，列主序

Usage:
    python compress.py --input /path/to/slided --output /path/to/compressed
"""

import argparse
import ctypes
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import torch

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# 使用顶层 utils 获取硬件信息和文件命名
from slidesparse.utils import (
    hw_info,
    build_filename,
    find_file,
    ensure_cusparselt_loaded,
)

# 导入 weight_convert 目录下的本地工具
from slidesparse.weight_convert.utils import (
    SlideSparseConfig,
    is_target_layer,
    load_safetensors,
    save_safetensors,
    get_all_safetensors_files,
    detect_weight_dtype,
    compute_compressed_k,
    verify_2to4_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
)


# =============================================================================
# 数据类型映射
# =============================================================================

# PyTorch dtype 到 C 字符串的映射
DTYPE_TO_STR = {
    torch.int8: "int8",
    torch.float8_e4m3fn: "fp8e4m3",
}

# 支持的数据类型
SUPPORTED_DTYPES = set(DTYPE_TO_STR.keys())


def get_dtype_str(dtype: torch.dtype) -> str:
    """将 PyTorch dtype 转换为 C 字符串"""
    if dtype not in DTYPE_TO_STR:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Supported: {', '.join(str(d) for d in SUPPORTED_DTYPES)}"
        )
    return DTYPE_TO_STR[dtype]


def is_supported_dtype(dtype: torch.dtype) -> bool:
    """检查是否为支持的数据类型"""
    return dtype in SUPPORTED_DTYPES


# =============================================================================
# 压缩扩展管理
# =============================================================================

# 压缩扩展前缀
COMPRESS_EXTENSION_PREFIX = "cusparselt_compress"

# 全局扩展模块缓存
_compress_module = None


def get_compress_module():
    """
    获取压缩扩展模块
    
    自动编译（如果需要）并加载 .so 文件。
    
    Returns:
        ctypes 加载的模块
    """
    global _compress_module
    
    if _compress_module is not None:
        return _compress_module
    
    # 确保 cuSPARSELt 库已加载
    ensure_cusparselt_loaded()
    
    # 查找已编译的 .so
    build_dir = _SCRIPT_DIR / "build"
    
    so_path = find_file(
        COMPRESS_EXTENSION_PREFIX,
        search_dir=build_dir,
        ext=".so",
    )
    
    if so_path is None:
        # 尝试编译
        print_info("Compress extension not found, building...")
        try:
            from build_compress import build_extension
            so_path = build_extension(force=False, verbose=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build compress extension: {e}\n"
                f"Please run: python3 {_SCRIPT_DIR}/build_compress.py build"
            ) from e
    
    print_info(f"Loading compress extension: {so_path.name}")
    
    # 加载 .so
    lib = ctypes.CDLL(str(so_path))
    
    # 设置函数签名
    # const char* cusparselt_compress_get_last_error()
    lib.cusparselt_compress_get_last_error.argtypes = []
    lib.cusparselt_compress_get_last_error.restype = ctypes.c_char_p
    
    # int cusparselt_get_compress_sizes(int N, int K, const char* dtype, size_t* compressed_size, size_t* temp_buffer_size)
    lib.cusparselt_get_compress_sizes.argtypes = [
        ctypes.c_int,                      # N
        ctypes.c_int,                      # K
        ctypes.c_char_p,                   # dtype
        ctypes.POINTER(ctypes.c_size_t),   # compressed_size
        ctypes.POINTER(ctypes.c_size_t),   # temp_buffer_size
    ]
    lib.cusparselt_get_compress_sizes.restype = ctypes.c_int
    
    # int cusparselt_compress_weight(const void* input, void* compressed, void* temp, int N, int K, const char* dtype, cudaStream_t stream)
    lib.cusparselt_compress_weight.argtypes = [
        ctypes.c_void_p,  # input_weight
        ctypes.c_void_p,  # compressed_weight
        ctypes.c_void_p,  # temp_buffer
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_char_p,  # dtype
        ctypes.c_void_p,  # stream (可以是 None)
    ]
    lib.cusparselt_compress_weight.restype = ctypes.c_int
    
    # int cusparselt_is_available()
    lib.cusparselt_is_available.argtypes = []
    lib.cusparselt_is_available.restype = ctypes.c_int
    
    # const char* cusparselt_get_supported_dtypes()
    lib.cusparselt_get_supported_dtypes.argtypes = []
    lib.cusparselt_get_supported_dtypes.restype = ctypes.c_char_p
    
    _compress_module = lib
    return lib


def _check_error(lib, ret: int, func_name: str):
    """检查返回值并抛出异常"""
    if ret != 0:
        err_msg = lib.cusparselt_compress_get_last_error()
        if err_msg:
            err_msg = err_msg.decode("utf-8")
        else:
            err_msg = "Unknown error"
        raise RuntimeError(f"{func_name} failed: {err_msg}")


def check_compress_available() -> bool:
    """检查压缩功能是否可用"""
    try:
        lib = get_compress_module()
        return lib.cusparselt_is_available() == 1
    except Exception as e:
        print_warning(f"Compress not available: {e}")
        return False


def get_supported_dtypes_from_lib() -> list:
    """从库获取支持的数据类型列表"""
    try:
        lib = get_compress_module()
        dtypes_str = lib.cusparselt_get_supported_dtypes().decode("utf-8")
        return dtypes_str.split(",")
    except Exception:
        return ["int8", "fp8e4m3"]


# =============================================================================
# 压缩核心函数
# =============================================================================

def get_compress_sizes(N: int, K: int, dtype: Union[torch.dtype, str] = "int8") -> Tuple[int, int]:
    """
    查询压缩后的大小和临时缓冲区大小
    
    Args:
        N: 权重行数（out_features）
        K: 权重列数（in_features，必须满足 2:4）
        dtype: 数据类型（torch.dtype 或字符串）
    
    Returns:
        (compressed_size, temp_buffer_size) in bytes
    """
    lib = get_compress_module()
    
    # 处理数据类型
    if isinstance(dtype, torch.dtype):
        dtype_str = get_dtype_str(dtype)
    else:
        dtype_str = dtype
    
    compressed_size = ctypes.c_size_t()
    temp_buffer_size = ctypes.c_size_t()
    
    ret = lib.cusparselt_get_compress_sizes(
        N, K,
        dtype_str.encode("utf-8"),
        ctypes.byref(compressed_size),
        ctypes.byref(temp_buffer_size),
    )
    _check_error(lib, ret, "cusparselt_get_compress_sizes")
    
    return compressed_size.value, temp_buffer_size.value


def compress_tensor_offline(
    weight: torch.Tensor,
    dtype: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    使用 cuSPARSELt 离线压缩 2:4 稀疏权重（用于权重转换脚本）
    
    Args:
        weight: 满足 2:4 稀疏的权重 [N, K]，INT8 或 FP8E4M3
        dtype: 数据类型字符串（自动检测如果为 None）
        verbose: 是否打印详细信息
    
    Returns:
        (compressed_weight, metadata)
        - compressed_weight: 压缩后的数据，uint8 tensor（在 CPU 上）
        - metadata: 压缩元数据
    
    Note:
        这个函数用于离线转换场景，会将压缩后的数据移回 CPU 以便保存。
        对于模型加载时的在线压缩，请使用 compress_tensor_online。
    """
    # 自动检测数据类型
    if dtype is None:
        if not is_supported_dtype(weight.dtype):
            raise ValueError(
                f"Unsupported weight dtype: {weight.dtype}. "
                f"Supported: {', '.join(str(d) for d in SUPPORTED_DTYPES)}"
            )
        dtype = get_dtype_str(weight.dtype)
    
    if weight.dim() != 2:
        raise ValueError(f"Weight must be 2D, got {weight.dim()}D")
    
    N, K = weight.shape
    
    # cuSPARSELt 对 INT8/FP8 稀疏矩阵要求 N 和 K 必须是 32 的倍数
    if N % 32 != 0:
        raise ValueError(
            f"N must be multiple of 32 for cuSPARSELt sparse matrices, got N={N}. "
            f"Use slide.py with align_to=32 to ensure proper alignment."
        )
    if K % 32 != 0:
        raise ValueError(
            f"K must be multiple of 32 for cuSPARSELt sparse matrices, got K={K}. "
            f"Use slide.py with align_to=32 to ensure proper alignment."
        )
    
    # 验证 2:4 稀疏性（转为 float 进行检查）
    weight_float = weight.float()
    is_valid, violation_ratio = verify_2to4_sparsity(weight_float)
    if not is_valid:
        raise ValueError(f"Weight does not satisfy 2:4 sparsity (violation: {violation_ratio:.2%})")
    
    # 查询压缩大小
    compressed_size, temp_buffer_size = get_compress_sizes(N, K, dtype)
    
    if verbose:
        print_info(f"  dtype={dtype}, compressed={compressed_size}, temp={temp_buffer_size}")
    
    # 分配 GPU 内存
    weight_gpu = weight.cuda().contiguous()
    compressed_gpu = torch.empty(compressed_size, dtype=torch.uint8, device="cuda")
    
    if temp_buffer_size > 0:
        temp_buffer_gpu = torch.empty(temp_buffer_size, dtype=torch.uint8, device="cuda")
        temp_ptr = temp_buffer_gpu.data_ptr()
    else:
        temp_ptr = None
    
    # 执行压缩
    lib = get_compress_module()
    
    ret = lib.cusparselt_compress_weight(
        ctypes.c_void_p(weight_gpu.data_ptr()),
        ctypes.c_void_p(compressed_gpu.data_ptr()),
        ctypes.c_void_p(temp_ptr) if temp_ptr else None,
        N, K,
        dtype.encode("utf-8"),
        None,  # stream
    )
    _check_error(lib, ret, "cusparselt_compress_weight")
    
    # 同步
    torch.cuda.synchronize()
    
    # 将压缩数据移回 CPU
    compressed_cpu = compressed_gpu.cpu()
    
    # 构建元数据
    metadata = {
        "original_shape": [N, K],
        "compressed_size_bytes": compressed_size,
        "temp_buffer_size_bytes": temp_buffer_size,
        "sparsity_pattern": "2:4",
        "layout": "TN-CC",  # W^T * A, all column-major
        "dtype": dtype,
    }
    
    return compressed_cpu, metadata


def compress_tensor_online(
    weight: torch.Tensor,
    dtype: Optional[str] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    使用 cuSPARSELt 在线压缩 2:4 稀疏权重（保持数据在 GPU 上）
    
    与 compress_tensor 的区别：
    - 不返回元数据，只返回压缩后的 tensor
    - 压缩后的数据保持在 GPU 上，避免 GPU<->CPU 拷贝
    - 用于模型加载时的在线压缩场景
    
    Args:
        weight: 满足 2:4 稀疏的权重 [N, K]，INT8 或 FP8E4M3，应在 GPU 上
        dtype: 数据类型字符串（自动检测如果为 None）
        verbose: 是否打印详细信息
    
    Returns:
        compressed_weight: 压缩后的 1D uint8 tensor，在 GPU 上
    """
    # 自动检测数据类型
    if dtype is None:
        if not is_supported_dtype(weight.dtype):
            raise ValueError(
                f"Unsupported weight dtype: {weight.dtype}. "
                f"Supported: {', '.join(str(d) for d in SUPPORTED_DTYPES)}"
            )
        dtype = get_dtype_str(weight.dtype)
    
    if weight.dim() != 2:
        raise ValueError(f"Weight must be 2D, got {weight.dim()}D")
    
    N, K = weight.shape
    
    # cuSPARSELt 对 INT8/FP8 稀疏矩阵要求 N 和 K 必须是 32 的倍数
    if N % 32 != 0:
        raise ValueError(
            f"N must be multiple of 32 for cuSPARSELt sparse matrices, got N={N}. "
            f"Use slide.py with align_to=32 to ensure proper alignment."
        )
    if K % 32 != 0:
        raise ValueError(
            f"K must be multiple of 32 for cuSPARSELt sparse matrices, got K={K}. "
            f"Use slide.py with align_to=32 to ensure proper alignment."
        )
    
    # 查询压缩大小
    compressed_size, temp_buffer_size = get_compress_sizes(N, K, dtype)
    
    if verbose:
        print_info(f"  [N={N}, K={K}] dtype={dtype}, compressed={compressed_size} bytes")
    
    # 分配 GPU 内存
    compressed_gpu = torch.empty(compressed_size, dtype=torch.uint8, device=weight.device)
    
    if temp_buffer_size > 0:
        temp_buffer_gpu = torch.empty(temp_buffer_size, dtype=torch.uint8, device=weight.device)
        temp_ptr = temp_buffer_gpu.data_ptr()
    else:
        temp_ptr = None
    
    # 执行压缩
    lib = get_compress_module()
    
    ret = lib.cusparselt_compress_weight(
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(compressed_gpu.data_ptr()),
        ctypes.c_void_p(temp_ptr) if temp_ptr else None,
        N, K,
        dtype.encode("utf-8"),
        None,  # stream
    )
    _check_error(lib, ret, "cusparselt_compress_weight")
    
    # 同步
    torch.cuda.synchronize()
    
    # 验证压缩结果大小
    if compressed_gpu.numel() != compressed_size:
        raise RuntimeError(
            f"Compression size mismatch: allocated {compressed_gpu.numel()} bytes, "
            f"expected {compressed_size} bytes. This may indicate a cuSPARSELt internal error."
        )
    
    # 直接返回 GPU 上的压缩数据
    return compressed_gpu


def compress_tensor_fake(
    weight: torch.Tensor,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    模拟 2:4 压缩（不调用 cuSPARSELt，用于测试）
    
    简单地将每 4 个元素中的 2 个非零值提取出来，并生成对应的元数据。
    
    Args:
        weight: 满足 2:4 稀疏的权重 [N, K]
        verbose: 是否打印详细信息
    
    Returns:
        (values, metadata_tensor, info)
        - values: [N, K/2] 非零值
        - metadata_tensor: [N, K/4] 稀疏位置元数据 (2 bits per element)
        - info: 压缩信息字典
    """
    N, K = weight.shape
    
    if K % 4 != 0:
        raise ValueError(f"K must be multiple of 4, got K={K}")
    
    # 重塑为 [N, K/4, 4]
    grouped = weight.view(N, K // 4, 4)
    
    # 找到每组的非零位置（每组 4 个元素中选 2 个非零值）
    values = torch.zeros(N, K // 2, dtype=weight.dtype)
    metadata_bits = torch.zeros(N, K // 4, dtype=torch.uint8)
    
    for i in range(N):
        for j in range(K // 4):
            group = grouped[i, j]
            # 找到非零位置
            nonzero_mask = (group != 0)
            nonzero_indices = torch.where(nonzero_mask)[0]
            
            # 取前 2 个非零值（或补零）
            for idx, pos in enumerate(nonzero_indices[:2]):
                values[i, j * 2 + idx] = group[pos]
                # 编码位置到 metadata（简化：直接存储索引）
                metadata_bits[i, j] |= (pos.item() << (idx * 2))
    
    # 检测数据类型
    dtype_str = get_dtype_str(weight.dtype) if is_supported_dtype(weight.dtype) else "unknown"
    
    info = {
        "original_shape": [N, K],
        "values_shape": list(values.shape),
        "metadata_shape": list(metadata_bits.shape),
        "compression_type": "fake_2to4",
        "dtype": dtype_str,
    }
    
    return values, metadata_bits, info


# =============================================================================
# Safetensors 处理
# =============================================================================

def compress_safetensors(
    input_path: Path,
    output_path: Path,
    use_real_cusparselt: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对 safetensors 文件执行 cuSPARSELt 压缩
    
    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出 safetensors 文件路径
        use_real_cusparselt: 是否使用真实的 cuSPARSELt（否则用模拟压缩）
        verbose: 是否打印详细信息
    
    Returns:
        处理报告
    """
    if verbose:
        print_info(f"Loading {input_path}")
    
    weights = load_safetensors(input_path)
    
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "use_real_cusparselt": use_real_cusparselt,
        "processed_layers": [],
        "skipped_layers": [],
        "dtype_stats": {},
    }
    
    output_weights = {}
    
    for key, tensor in weights.items():
        # 检查是否为目标层
        if not is_target_layer(key):
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
            continue
        
        # 检查是否为 2D 张量
        if tensor.dim() != 2:
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
            continue
        
        # 检查数据类型
        if not is_supported_dtype(tensor.dtype):
            if verbose:
                print_warning(f"  Skipping {key}: unsupported dtype {tensor.dtype}")
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
            continue
        
        # 获取数据类型字符串
        dtype_str = get_dtype_str(tensor.dtype)
        
        # 统计数据类型
        if dtype_str not in report["dtype_stats"]:
            report["dtype_stats"][dtype_str] = 0
        report["dtype_stats"][dtype_str] += 1
        
        N, K = tensor.shape
        
        if verbose:
            print_info(f"  Compressing {key}: [{N}, {K}] ({dtype_str})")
        
        try:
            if use_real_cusparselt:
                compressed, metadata = compress_tensor_offline(tensor, dtype=dtype_str, verbose=verbose)
                # 保存压缩数据
                output_weights[key] = compressed
                # 保存元数据（作为额外的 tensor）
                # 格式: [N, K, compressed_size_bytes, dtype_id]
                dtype_id = 0 if dtype_str == "int8" else 1  # 0=int8, 1=fp8e4m3
                output_weights[f"{key}_compress_meta"] = torch.tensor(
                    [N, K, metadata["compressed_size_bytes"], dtype_id], dtype=torch.int64
                )
            else:
                values, meta_bits, info = compress_tensor_fake(tensor, verbose=verbose)
                output_weights[key] = values
                output_weights[f"{key}_compress_meta"] = meta_bits
            
            report["processed_layers"].append({
                "key": key,
                "original_shape": [N, K],
                "compressed_shape": list(compressed.shape) if use_real_cusparselt else list(values.shape),
                "dtype": dtype_str,
            })
            
        except Exception as e:
            if verbose:
                print_error(f"  Failed to compress {key}: {e}")
            output_weights[key] = tensor
            report["skipped_layers"].append(key)
    
    # 保存
    if verbose:
        print_info(f"Saving to {output_path}")
    
    save_safetensors(output_weights, output_path)
    
    return report


def compress_checkpoint(
    input_dir: Path,
    output_dir: Path,
    use_real_cusparselt: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    压缩整个 checkpoint 目录
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        use_real_cusparselt: 是否使用真实的 cuSPARSELt
        verbose: 是否打印详细信息
    
    Returns:
        处理报告
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a directory: {input_dir}")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 找到所有 safetensors 文件
    safetensors_files = get_all_safetensors_files(input_dir)
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")
    
    if verbose:
        print_info(f"Found {len(safetensors_files)} safetensors file(s)")
    
    overall_report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_processed": 0,
        "total_layers_compressed": 0,
        "total_layers_skipped": 0,
        "dtype_stats": {},
        "file_reports": [],
    }
    
    for sf_path in safetensors_files:
        rel_path = sf_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print_header(f"Processing: {sf_path.name}")
        
        try:
            report = compress_safetensors(
                sf_path,
                out_path,
                use_real_cusparselt=use_real_cusparselt,
                verbose=verbose,
            )
            
            overall_report["files_processed"] += 1
            overall_report["total_layers_compressed"] += len(report["processed_layers"])
            overall_report["total_layers_skipped"] += len(report["skipped_layers"])
            
            # 合并 dtype 统计
            for dtype_str, count in report.get("dtype_stats", {}).items():
                if dtype_str not in overall_report["dtype_stats"]:
                    overall_report["dtype_stats"][dtype_str] = 0
                overall_report["dtype_stats"][dtype_str] += count
            
            overall_report["file_reports"].append(report)
            
            if verbose:
                print_success(f"  Compressed {len(report['processed_layers'])} layers")
                print_info(f"  Skipped {len(report['skipped_layers'])} layers")
                
        except Exception as e:
            print_error(f"  Error processing {sf_path.name}: {e}")
            overall_report["file_reports"].append({
                "input_path": str(sf_path),
                "error": str(e),
            })
    
    # 复制非 safetensors 文件（如 config.json 等）
    for item in input_dir.iterdir():
        if item.is_file() and not item.suffix == ".safetensors":
            import shutil
            shutil.copy2(item, output_dir / item.name)
            if verbose:
                print_info(f"Copied: {item.name}")
    
    return overall_report


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse cuSPARSELt 压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
压缩说明：
  2:4 结构化稀疏压缩将 K 维度减半
  
  输入：[N, K] 满足 2:4 稀疏的 INT8/FP8E4M3 权重
  输出：压缩后的数据 + 元数据

支持的数据类型：
  - INT8 (torch.int8)
  - FP8E4M3 (torch.float8_e4m3fn)

布局约定：
  TN-CC 格式（W^T * A，全列主序）

示例：
  # 压缩 INT8 checkpoint（自动推导输出路径）
  python compress.py -i checkpoints_slidesparse/BitNet-2B_mag_Z2L8_INT8_slided_2_8
  
  # 压缩 FP8 checkpoint（指定输出路径）
  python compress.py -i checkpoints_slidesparse/BitNet-2B_mag_Z2L6_FP8E4M3_slided_2_6 \\
                     -o checkpoints_slidesparse/BitNet-2B_mag_Z2L6_FP8E4M3_compressed

使用 --fake 可以跳过真实压缩进行测试
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入模型目录或 safetensors 文件",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出路径（默认: 输入路径 + _compressed 后缀）",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="使用模拟压缩（不调用 cuSPARSELt）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 检查压缩库可用性
    use_real = not args.fake
    if use_real and not check_compress_available():
        print_warning("cuSPARSELt compress not available, using fake compression")
        use_real = False
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        if input_path.is_dir():
            # 替换 _slided 后缀为 _compressed
            # 支持两种格式: xxx_slided 或 xxx_slided_Z_L
            name = input_path.name
            import re
            # 匹配 _slided 或 _slided_数字_数字
            slided_match = re.search(r"_slided(_\d+_\d+)?$", name)
            if slided_match:
                new_name = name[:slided_match.start()] + "_compressed"
            else:
                new_name = name + "_compressed"
            output_path = input_path.parent / new_name
        else:
            output_path = input_path.parent / f"{input_path.stem}_compressed"
    
    if not args.quiet:
        print_header("SlideSparse Compress")
        print_info(f"Input: {input_path}")
        print_info(f"Output: {output_path}")
        print_info(f"Mode: {'Real cuSPARSELt' if use_real else 'Fake (simulation)'}")
        print_info(f"Supported dtypes: {', '.join(get_supported_dtypes_from_lib())}")
    
    # 处理
    try:
        if input_path.is_dir():
            report = compress_checkpoint(
                input_path,
                output_path,
                use_real_cusparselt=use_real,
                verbose=not args.quiet,
            )
        else:
            # 单文件
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_dir():
                output_file = output_path / input_path.name
            else:
                output_file = output_path.with_suffix(".safetensors")
            
            report = compress_safetensors(
                input_path,
                output_file,
                use_real_cusparselt=use_real,
                verbose=not args.quiet,
            )
        
        if not args.quiet:
            print_header("Summary")
            if "files_processed" in report:
                print_success(f"Files processed: {report['files_processed']}")
                print_success(f"Total layers compressed: {report['total_layers_compressed']}")
                print_info(f"Total layers skipped: {report['total_layers_skipped']}")
                if report.get("dtype_stats"):
                    print_info(f"Dtype stats: {report['dtype_stats']}")
            else:
                print_success(f"Layers compressed: {len(report['processed_layers'])}")
                print_info(f"Layers skipped: {len(report['skipped_layers'])}")
            
            print_success("Done!")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
