#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 算法/布局搜索共享工具库

提供搜索脚本所需的通用功能：
- 模型 NK 尺寸获取
- 默认参数配置
- 数据类型验证
- 搜索元数据构建
- 结果文件命名与保存

该工具库依赖顶层 slidesparse.utils，提供搜索场景的封装。
"""

import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# 导入顶层 utils
SEARCH_DIR = Path(__file__).parent.absolute()
SLIDESPARSE_DIR = SEARCH_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.utils import (
    # 硬件信息
    hw_info,
    HardwareInfo,
    normalize_dtype,
    # 全局默认配置
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    # 编译与加载
    load_cuda_extension,
    build_cuda_extension_direct,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
    BACKEND_LDFLAGS,
    SUPPORTED_BACKENDS,
    # 文件命名
    build_filename,
    build_stem,
    build_tuned_filename,
    # 模型信息
    get_model_nk_sizes,
    get_model_nk_sizes_slided,
    get_model_nk_sizes_compressed,
    # 统一 NK 获取工具
    get_nk_list_for_search,
)

import ctypes
import torch
from typing import Callable


# =============================================================================
# CUDA 扩展编译与加载（包装顶层工具）
# =============================================================================

def build_search_extension(
    name: str,
    source_file: Path,
    build_dir: Path,
    backend: str,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    编译搜索扩展 (extern "C" 接口)。
    
    基于顶层 build_cuda_extension_direct，自动添加后端依赖库。
    
    Args:
        name: 扩展名称（如 "alg_search_cublaslt"）
        source_file: 源文件路径 (.cu)
        build_dir: 构建目录
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        force: 是否强制重新编译
        verbose: 是否显示详细输出
    
    Returns:
        编译生成的 .so 文件路径
    """
    # 构建带完整硬件信息的名称
    full_name = (
        f"{name}_{hw_info.gpu_name}_{hw_info.cc_tag}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )
    
    # 获取后端链接库
    backend_lower = backend.lower()
    if backend_lower == "cublaslt":
        extra_ldflags = ["-L/usr/lib/x86_64-linux-gnu", "-lcublasLt", "-lcublas", "-lcuda"]
    elif backend_lower == "cusparselt":
        extra_ldflags = ["-L/usr/lib/x86_64-linux-gnu", "-lcusparseLt", "-lcusparse", "-lcublas", "-lcuda"]
    else:
        raise ValueError(f"未知后端: {backend}")
    
    return build_cuda_extension_direct(
        name=full_name,
        source_file=source_file,
        build_dir=build_dir,
        extra_ldflags=extra_ldflags,
        force=force,
        verbose=verbose,
    )


def load_search_extension(
    so_path: Path,
    backend: str,
    setup_func: Callable[[ctypes.CDLL], None],
) -> ctypes.CDLL:
    """
    加载搜索扩展 (extern "C" 接口)。
    
    Args:
        so_path: .so 文件路径
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        setup_func: 设置函数签名的回调函数
    
    Returns:
        加载的 ctypes.CDLL 对象
    """
    # 预加载后端库
    backend_lower = backend.lower()
    if backend_lower == "cublaslt":
        ensure_cublaslt_loaded()
    elif backend_lower == "cusparselt":
        ensure_cusparselt_loaded()
    else:
        raise ValueError(f"未知后端: {backend}")
    
    # 加载扩展
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    
    # 调用设置函数配置签名
    setup_func(lib)
    
    return lib


# =============================================================================
# 数据准备工具
# =============================================================================

def quantize_int8(x: torch.Tensor, inplace: bool = False) -> tuple:
    """
    将 BF16/FP16 张量量化到 INT8。
    
    Args:
        x: 输入张量
        inplace: 是否原地操作（节省显存，但会修改输入）
    
    Returns:
        (quantized_tensor, scale)
    """
    abs_max = x.abs().max().item()
    scale = 127.0 / abs_max if abs_max > 0 else 1.0
    
    # 使用分步操作减少峰值显存
    if inplace:
        x.mul_(scale)
        x.round_()
        x.clamp_(-128, 127)
        q = x.to(torch.int8)
    else:
        # 原始实现，保持兼容性
        q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """转换为 FP8E4M3 格式"""
    return x.to(torch.float8_e4m3fn)


def quantize_tensor(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """
    根据 dtype 量化张量。
    
    Args:
        x: 输入张量 (BF16/FP16)
        dtype: 目标类型 ("int8" 或 "fp8e4m3")
    
    Returns:
        量化后的张量
    """
    if dtype == "int8":
        q, _ = quantize_int8(x)
        return q
    elif dtype == "fp8e4m3":
        return to_fp8_e4m3(x)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")


def get_output_torch_dtype(outdtype: str) -> torch.dtype:
    """获取输出对应的 PyTorch dtype"""
    if outdtype == "fp32":
        return torch.float32
    elif outdtype == "bf16":
        return torch.bfloat16
    elif outdtype == "int32":
        return torch.int32
    else:
        raise ValueError(f"不支持的输出类型: {outdtype}")


def analyze_algo_structure(algo_data: bytes) -> dict:
    """
    详细分析 cublasLtMatmulAlgo_t 结构体（64 字节不透明数据）。
    
    根据 cuBLAS 文档和实验推断的字段布局：
    - uint32[0] = alg_id (CUBLASLT_ALGO_CONFIG_ID)
    - uint32[1] = tile_id (CUBLASLT_ALGO_CONFIG_TILE_ID)
    - uint32[2] = stages_id (CUBLASLT_ALGO_CONFIG_STAGES_ID)
    - uint32[3] = split_k_num (CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
    - uint32[4] = reduction_scheme (CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
    - uint32[5] = cta_swizzle (CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
    - uint32[6:12] = custom options
    
    关键发现：
    - 当 split_k_num > 1 时，算法需要 workspace 存储中间结果
    - workspace 大小 ≈ M × N × split_k × dtype_size
    
    Args:
        algo_data: 64 字节的不透明算法数据
    
    Returns:
        dict 包含解析后的字段:
            - raw_u32: 原始 16 个 uint32 值
            - alg_id: 算法 ID
            - tile_id: Tile 配置 ID
            - stages_id: Pipeline stages ID
            - split_k_num: Split-K 分割数（关键！）
            - reduction_scheme: 规约方案 ID
            - cta_swizzle: CTA 重排配置
            - custom_opts: 自定义选项 (uint32[6:12])
    """
    import struct
    
    if len(algo_data) < 64:
        # 补齐到 64 字节
        algo_data = algo_data + b'\x00' * (64 - len(algo_data))
    
    u32 = struct.unpack("<16I", algo_data)
    
    return {
        "raw_u32": u32,
        "alg_id": u32[0],
        "tile_id": u32[1],
        "stages_id": u32[2],
        "split_k_num": u32[3],  # 关键！
        "reduction_scheme": u32[4],  # 关键！
        "cta_swizzle": u32[5],
        "custom_opts": u32[6:12],
    }


# =============================================================================
# 支持的数据类型
# =============================================================================

SUPPORTED_DTYPES = ["int8", "fp8e4m3"]
SUPPORTED_OUTDTYPES = ["bf16", "fp32", "int32"]


# 后端和数据类型的组合验证
def validate_dtype_outdtype_combination(
    dtype: str,
    outdtype: str,
    backend: str,
) -> str:
    """
    验证 dtype/outdtype 组合是否合法，并返回实际使用的 outdtype。
    
    规则：
    - FP8 输入: 支持 bf16 或 fp32 输出（所有后端）
    - INT8 输入 + cuBLASLt: 只支持 int32 输出（硬件限制，不支持 bf16/fp32）
    - INT8 输入 + cuSPARSELt: 支持 bf16 或 int32 输出（不支持 fp32）
    
    Args:
        dtype: 输入数据类型
        outdtype: 用户指定的输出数据类型
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
    
    Returns:
        实际使用的 outdtype
    
    Raises:
        ValueError: 不支持的组合
    """
    backend = backend.lower()
    dtype = dtype.lower()
    outdtype = outdtype.lower()
    
    if dtype in ("fp8", "fp8e4m3"):
        # FP8: 支持 bf16 或 fp32
        if outdtype not in ("bf16", "fp32"):
            raise ValueError(f"FP8 输入不支持 {outdtype} 输出。支持: bf16, fp32")
        return outdtype
    
    elif dtype == "int8":
        if backend == "cublaslt":
            # cuBLASLt INT8: 只支持 int32 输出（硬件限制）
            # 不支持 bf16/fp32，直接报错而不是 fallback
            if outdtype != "int32":
                raise ValueError(
                    f"cuBLASLt INT8 不支持 {outdtype} 输出。"
                    f"cuBLASLt 使用 CUBLAS_COMPUTE_32I，只能输出 INT32。"
                    f"如需 BF16 输出，请使用 cuSPARSELt 后端。"
                )
            return "int32"
        
        elif backend == "cusparselt":
            # cuSPARSELt INT8: 支持 bf16 或 int32（不支持 fp32）
            if outdtype == "fp32":
                raise ValueError("cuSPARSELt INT8 不支持 fp32 输出。支持: bf16, int32")
            if outdtype not in ("bf16", "int32"):
                raise ValueError(f"INT8 输入不支持 {outdtype} 输出。支持: bf16, int32")
            return outdtype
        
        else:
            raise ValueError(f"未知后端: {backend}")
    
    else:
        raise ValueError(f"不支持的输入类型: {dtype}")


def get_default_outdtype(dtype: str, backend: str) -> str:
    """
    获取给定 dtype 和 backend 的默认 outdtype。
    
    Args:
        dtype: 输入数据类型
        backend: 后端类型
    
    Returns:
        默认的 outdtype
    """
    backend = backend.lower()
    dtype = dtype.lower()
    
    if dtype in ("fp8", "fp8e4m3"):
        return "bf16"  # FP8 默认 bf16
    elif dtype == "int8":
        if backend == "cublaslt":
            return "int32"  # cuBLASLt INT8 只能 int32
        else:
            return "bf16"   # cuSPARSELt INT8 默认 bf16
    else:
        return "bf16"


# =============================================================================
# 默认配置
# =============================================================================

def default_m_list() -> List[int]:
    """
    返回默认的 M 值列表。
    
    覆盖从 decode (小 batch) 到 prefill (大 batch) 的典型场景。
    使用顶层 slidesparse.utils 中定义的 DEFAULT_M_LIST。
    """
    return list(DEFAULT_M_LIST)


def default_nk_list_bitnet_2b() -> List[Tuple[int, int]]:
    """
    BitNet-2B4T 模型的默认 NK 尺寸。
    
    基于标准 vLLM 格式的线性层尺寸。
    """
    return [
        (3840, 2560),   # qkv_proj (Wqkv)
        (2560, 2560),   # o_proj (Wo)
        (13824, 2560),  # gate_up_proj (W13)
        (2560, 6912),   # down_proj (W2)
    ]


# =============================================================================
# 输出目录与文件命名
# =============================================================================

def build_output_dir_name() -> str:
    """
    构建输出目录名称（仅包含硬件信息）。
    
    格式: {GPU}_{CC}_{PyVer}_{CUDAVer}_{arch}
    示例: RTX5080_cc120_py312_cu129_x86_64
    
    Returns:
        目录名称
    """
    return (
        f"{hw_info.gpu_name}_{hw_info.cc_tag}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )


# =============================================================================
# 搜索元数据构建
# =============================================================================

def build_search_meta(
    *,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List[Tuple[int, int]],
    model_name: str,
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建搜索结果的元数据。
    
    Args:
        dtype: 输入数据类型
        outdtype: 输出数据类型
        warmup: 预热次数
        repeat: 重复次数
        verify: 是否验证
        m_list: M 列表
        nk_list: NK 列表
        model_name: 模型名称
        layout: 布局类型
        alg_count: 算法数量
        config_count: 配置数量
        extra: 额外元数据
    
    Returns:
        元数据字典
    """
    import torch
    
    # 处理 NK 列表，确保只保留 (N, K) 对
    nk_pairs = []
    if nk_list:
        for t in nk_list:
            if len(t) >= 2:
                nk_pairs.append((t[0], t[1]))
    
    meta = {
        "gpu_name": hw_info.gpu_full_name,
        "gpu_short_name": hw_info.gpu_name,
        "compute_capability": hw_info.cc_tag,
        "arch_name": hw_info.arch_name,
        "model_name": model_name,
        "layout": layout,
        "dtype": dtype,
        "outdtype": outdtype,
        "alg_count": alg_count,
        "config_count": config_count,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": hw_info.cuda_driver_version,
        "cuda_version_runtime": hw_info.cuda_runtime_version,
        "time": datetime.datetime.now().isoformat(),
        "M_list": m_list,
        "NK_list": nk_pairs,
    }
    
    if extra:
        meta.update(extra)
    
    return meta


# =============================================================================
# Segment-K 支持检测
# =============================================================================

def supports_segment_k() -> Tuple[bool, str]:
    """
    检测当前 GPU 是否支持 Segment-K (split_k=-1)。
    
    Segment-K 仅在 SM 9.0 (Hopper) 和 SM 10.x (Blackwell) 上支持。
    在 cuSPARSELt 0.8.1 及更早版本中，对不支持的架构调用 Segment-K
    会导致 planInit 阻塞挂起。
    
    Returns:
        (supported, reason_if_not)
    """
    major = hw_info.cc_major
    
    if major in (9, 10):
        return True, ""
    else:
        return False, (
            f"Segment-K (split_k=-1) 仅在 SM 9.0/10.x (Hopper/Blackwell) 上支持，"
            f"当前架构 SM {hw_info.cc_major}.{hw_info.cc_minor} 不支持。"
        )


# =============================================================================
# dtype 兼容性探测与验证
# =============================================================================

def probe_cublaslt_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt 算法搜索扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        out = ext.search_topk(
            W, A,
            [M],
            layout,
            dtype,
            outdtype,
            1, 1,  # warmup, repeat
            False, # verify
            [],    # blacklist
            1,     # topk
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def probe_cublaslt_layout(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt layout 搜索扩展对 dtype 的支持情况。
    """
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",
            dtype,
            outdtype,
            1, 1,
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"


def probe_cusparselt_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt 算法搜索扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        W_pruned = ext.prune_24(W, layout)
        
        out = ext.search_topk(
            W_pruned, A,
            [M],
            layout,
            dtype,
            outdtype,
            1, 1,
            False,
            [],
            1,
            False,  # test_segment_k
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def probe_cusparselt_layout(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt layout 搜索扩展对 dtype 的支持情况。
    """
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",
            dtype,
            outdtype,
            1, 1,
            False,  # test_segment_k
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"


def check_dtype_support(
    ext,
    dtype: str,
    outdtype: str,
    *,
    backend: str,
    script_type: str,
    verbose: bool = True,
) -> None:
    """
    检查 dtype 是否被当前 GPU 支持（通过实际测试）。
    
    Args:
        ext: 编译好的 CUDA 扩展模块
        dtype: 要测试的数据类型
        outdtype: 输出数据类型
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        script_type: 脚本类型 ("alg_search" 或 "layout_search")
        verbose: 是否显示详细信息
    
    Raises:
        ValueError: 如果 dtype 不被支持
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"不支持的数据类型: {dtype}\n"
            f"支持的类型: {', '.join(SUPPORTED_DTYPES)}"
        )
    
    if outdtype not in SUPPORTED_OUTDTYPES:
        raise ValueError(
            f"不支持的输出数据类型: {outdtype}\n"
            f"支持的类型: {', '.join(SUPPORTED_OUTDTYPES)}"
        )
    
    # 获取对应的探测函数
    prober_map = {
        ("cublaslt", "alg_search"): probe_cublaslt_search,
        ("cublaslt", "layout_search"): probe_cublaslt_layout,
        ("cusparselt", "alg_search"): probe_cusparselt_search,
        ("cusparselt", "layout_search"): probe_cusparselt_layout,
    }
    
    prober_key = (backend.lower(), script_type.lower())
    if prober_key not in prober_map:
        raise ValueError(f"未知的 backend/script_type 组合: {prober_key}")
    
    prober = prober_map[prober_key]
    
    if verbose:
        print(f"[预测试] 检测 dtype={dtype}, outdtype={outdtype} ...", end=" ", flush=True)
    
    supported, message = prober(ext, dtype, outdtype)
    
    if supported:
        if verbose:
            print("✓", flush=True)
    else:
        if verbose:
            print("✗", flush=True)
        raise ValueError(
            f"数据类型 {dtype.upper()} 在当前 GPU ({hw_info.arch_name}) 上不可用。\n"
            f"原因: {message}"
        )


# =============================================================================
# CSV 表头构建
# =============================================================================

def build_csv_header_lines(
    *,
    model_name: str,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List[Tuple[int, int]],
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
) -> List[str]:
    """
    构建 CSV 文件的元数据头部行。
    """
    import torch
    
    return [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# CC: {hw_info.cc_tag}",
        f"# Model: {model_name}",
        f"# alg_count: {alg_count}, config_count: {config_count}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {hw_info.cuda_driver_version}, runtime: {hw_info.cuda_runtime_version}",
        f"# layout: {layout}, dtype: {dtype}, outdtype: {outdtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {m_list}",
        f"# NK_list: {nk_list}",
    ]


# =============================================================================
# Layout 配置（用于 Layout Search）
# =============================================================================

# 4 种主要 layout 配置
LAYOUT_CONFIGS = [
    {"name": "TN+CC", "opW": "T", "opA": "N", "orderW": "Col", "orderA": "Col"},
    {"name": "NT+RR", "opW": "N", "opA": "T", "orderW": "Row", "orderA": "Row"},
    {"name": "NN+RC", "opW": "N", "opA": "N", "orderW": "Row", "orderA": "Col"},
    {"name": "TT+CR", "opW": "T", "opA": "T", "orderW": "Col", "orderA": "Row"},
]

# 输出矩阵顺序
OUTPUT_ORDERS = ["Col", "Row"]


# =============================================================================
# 导出接口
# =============================================================================

__all__ = [
    # 常量
    "SUPPORTED_DTYPES",
    "SUPPORTED_OUTDTYPES",
    "LAYOUT_CONFIGS",
    "OUTPUT_ORDERS",
    # 默认配置
    "default_m_list",
    "default_nk_list_bitnet_2b",
    # 模型 NK 工具（重导出自顶层 utils）
    "get_nk_list_for_search",
    # 输出命名
    "build_output_dir_name",
    "build_tuned_filename",
    # 元数据
    "build_search_meta",
    "build_csv_header_lines",
    # Segment-K
    "supports_segment_k",
    # dtype 检测
    "check_dtype_support",
    "probe_cublaslt_search",
    "probe_cublaslt_layout",
    "probe_cusparselt_search",
    "probe_cusparselt_layout",
    # 编译与加载
    "build_search_extension",
    "load_search_extension",
    # 数据准备
    "quantize_int8",
    "to_fp8_e4m3",
    "quantize_tensor",
    "get_output_torch_dtype",
    # algo_data 解析
    "analyze_algo_structure",
    # 结果保存
    "save_alg_search_results",
    "save_layout_search_results",
    # 重导出顶层 utils
    "hw_info",
    "normalize_dtype",
    "load_cuda_extension",
    "build_cuda_extension_direct",
    "ensure_cublaslt_loaded",
    "ensure_cusparselt_loaded",
    "get_model_nk_sizes",
]


# =============================================================================
# 结果保存工具
# =============================================================================

def save_alg_search_results(
    out_dir: Path,
    model_name: str,
    dtype: str,
    outdtype: str,
    search_ret: dict,
    warmup: int,
    repeat: int,
    verify: bool,
    *,
    layout: str = "TNCCcol",
    is_sparse: bool = False,
    has_split_k: bool = False,
) -> Path:
    """
    保存算法搜索结果到 CSV 和 JSON 文件。
    
    Args:
        out_dir: 输出基础目录
        model_name: 模型名称
        dtype: 输入数据类型
        outdtype: 输出数据类型
        search_ret: 搜索结果字典，包含 results, M_list, NK_list 等
        warmup: 预热次数
        repeat: 重复次数
        verify: 是否验证
        layout: 布局类型
        is_sparse: 是否为稀疏搜索 (cuSPARSELt)
        has_split_k: 是否包含 split_k 信息
    
    Returns:
        保存结果的子目录路径
    """
    import base64
    
    subdir_name = build_output_dir_name()
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_tuned_filename("alg_search", model_name, "csv", outdtype)
    json_path = subdir / build_tuned_filename("alg_search", model_name, "json", outdtype)
    
    alg_count = search_ret.get("max_alg_count", 0)
    config_count = alg_count * (6 if search_ret.get("search_split_k") else 1)
    
    # === CSV 生成 ===
    header_lines = build_csv_header_lines(
        model_name=model_name,
        dtype=dtype,
        outdtype=outdtype,
        warmup=warmup,
        repeat=repeat,
        verify=verify,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        layout=layout,
        alg_count=alg_count,
        config_count=config_count,
    )
    
    csv_lines = list(header_lines)
    
    # 统一列格式: tops, lat_us, alg_id, split_k, workspace (top3)
    csv_lines.append("M,N,K,alg_count,config_count,tops_1,lat_us_1,alg_id_1,split_k_1,workspace_1,tops_2,lat_us_2,alg_id_2,split_k_2,workspace_2,tops_3,lat_us_3,alg_id_3,split_k_3,workspace_3")
    
    csv_rows = []
    
    for nk_idx, nk_res in enumerate(search_ret["results"]):
        N, K = nk_res["N"], nk_res["K"]
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            values = [str(M), str(N), str(K), str(m_res.get("alg_count", 0)), str(config_count)]
            
            for k in range(3):
                if k < len(results):
                    r = results[k]
                    # 统一获取 split_k：
                    # - cuSPARSELt (has_split_k=True): 直接使用 r['split_k']
                    # - cuBLASLt (has_split_k=False): 从 algo_data 解析
                    if has_split_k:
                        split_k = r.get('split_k', 1)
                    else:
                        # cuBLASLt: 从 algo_data 解析 split_k_num
                        algo_data = r.get('algo_data')
                        if algo_data:
                            algo_info = analyze_algo_structure(algo_data)
                            split_k = algo_info['split_k_num']
                        else:
                            split_k = 1
                    
                    values.extend([
                        f"{r['tops']:.6f}",
                        f"{r['lat_us']:.3f}",
                        str(r['alg_id']),
                        str(split_k),
                        str(r['workspace']),
                    ])
                else:
                    values.extend(["", "", "", "", ""])
            
            csv_rows.append((M, nk_idx, ",".join(values)))
    
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        csv_lines.append(line)
    
    csv_path.write_text("\n".join(csv_lines))
    
    # === JSON 生成 ===
    meta = build_search_meta(
        dtype=dtype,
        outdtype=outdtype,
        warmup=warmup,
        repeat=repeat,
        verify=verify,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        model_name=model_name,
        layout=layout,
        alg_count=alg_count,
        config_count=config_count,
    )
    
    if is_sparse:
        meta["supports_segment_k"] = search_ret.get("supports_segment_k", False)
        meta["search_split_k"] = search_ret.get("search_split_k", False)
    
    nk_entries = {}
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        nk_key = f"({N},{K})"
        
        m_thresholds = []
        alg_by_m = {}
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            if results:
                m_thresholds.append(M)
                r = results[0]  # 只保留 top1
                if has_split_k:
                    # cuSPARSELt: 保存 alg_id, split_k, workspace
                    alg_by_m[str(M)] = {
                        "alg_id": r["alg_id"],
                        "split_k": r.get("split_k", 1),
                        "workspace": r.get("workspace", 0),
                    }
                else:
                    # cuBLASLt: 保存 base64 编码的 algo_data 和 workspace
                    if "algo_data" in r:
                        algo_b64 = base64.b64encode(r["algo_data"]).decode('ascii')
                        alg_by_m[str(M)] = {
                            "algo_data": algo_b64,
                            "workspace": r.get("workspace", 0),
                        }
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }
    
    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    
    import json as json_mod
    json_path.write_text(json_mod.dumps(json_payload, indent=2, ensure_ascii=False))
    
    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


def save_layout_search_results(
    out_dir: Path,
    model_name: str,
    dtype: str,
    outdtype: str,
    search_ret: dict,
    warmup: int,
    repeat: int,
    verify: bool,
    *,
    layout_names: list,
    is_sparse: bool = False,
) -> Path:
    """
    保存布局搜索结果到 CSV 和 JSON 文件。
    
    Layout Search 的目的是遍历所有布局，记录每种布局的最佳算法和性能。
    
    Args:
        out_dir: 输出基础目录
        model_name: 模型名称
        dtype: 输入数据类型
        outdtype: 输出数据类型
        search_ret: 搜索结果字典
        warmup: 预热次数
        repeat: 重复次数
        verify: 是否验证
        layout_names: 布局名称列表
        is_sparse: 是否为稀疏搜索 (cuSPARSELt)
    
    Returns:
        保存结果的子目录路径
    """
    subdir_name = build_output_dir_name()
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_tuned_filename("layout_search", model_name, "csv", outdtype)
    json_path = subdir / build_tuned_filename("layout_search", model_name, "json", outdtype)
    
    num_layouts = len(layout_names)
    layout_tag = "LAYOUT_SEARCH_SPARSE24" if is_sparse else "LAYOUT_SEARCH"
    
    # === CSV 生成 ===
    header_lines = build_csv_header_lines(
        model_name=model_name,
        dtype=dtype,
        outdtype=outdtype,
        warmup=warmup,
        repeat=repeat,
        verify=verify,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        layout=layout_tag,
        alg_count=num_layouts,
        config_count=num_layouts,
    )
    
    csv_lines = list(header_lines)
    
    # CSV 列格式：统一使用 tops, lat_us, alg_id, split_k, workspace
    csv_lines.append("M,N,K,layout,tops,lat_us,alg_id,split_k,workspace")
    
    csv_rows = []
    
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            for r in results:
                # 只输出有效的布局
                if not r.get("valid", False):
                    continue
                
                values = [
                    str(M), str(N), str(K),
                    r["layout_name"],
                    f"{r['tops']:.6f}",
                    f"{r['lat_us']:.3f}",
                    str(r.get("alg_id", -1)),
                    str(r.get("split_k", 1)),
                    str(r.get("workspace", 0)),
                ]
                csv_rows.append((M, N, K, r["layout_name"], ",".join(values)))
    
    # 排序：先按 M 升序，再按 N, K 升序，最后按 layout_name
    csv_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    for _, _, _, _, line in csv_rows:
        csv_lines.append(line)
    
    csv_path.write_text("\n".join(csv_lines))
    
    # === JSON 生成：保存完整的搜索结果 ===
    # 格式与旧代码兼容：按 (N,K) 组织结果
    json_results = {}
    
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        nk_key = f"({N},{K})"
        
        json_results[nk_key] = {}
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            m_results = {}
            for r in results:
                if not r.get("valid", False):
                    continue
                
                layout_name = r["layout_name"]
                
                m_results[layout_name] = {
                    "tops": r["tops"],
                    "lat_us": r["lat_us"],
                    "alg_id": r.get("alg_id", -1),
                    "split_k": r.get("split_k", 1),
                    "workspace": r.get("workspace", 0),
                }
            
            json_results[nk_key][str(M)] = m_results
    
    meta = {
        "model": model_name,
        "dtype": dtype,
        "outdtype": outdtype,
        "gpu": hw_info.gpu_full_name,
        "gpu_short": hw_info.gpu_name,
        "cc": hw_info.cc_tag,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "m_list": search_ret["M_list"],
        "nk_list": [[nk[0], nk[1]] for nk in search_ret["NK_list"]],
        "layout_names": layout_names,
        "is_sparse": is_sparse,
    }
    
    json_payload = {
        "meta": meta,
        "results": json_results,
    }
    
    import json as json_mod
    json_path.write_text(json_mod.dumps(json_payload, indent=2, ensure_ascii=False))
    
    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# =============================================================================
# 搜索结果验证 (从 CUDA 端迁移)
# =============================================================================

def verify_gemm_result(
    W_q: torch.Tensor,
    A_q: torch.Tensor, 
    R_out: torch.Tensor,
    M: int,
    *,
    is_col_major: bool = True,
    tol: float = 0.05,
    critical_tol: float = 1.00,
) -> Dict[str, Any]:
    """
    验证 GEMM 计算结果的正确性。
    
    使用量化后的数据做 FP32 参考计算，与 CUDA 输出比较。
    
    计算公式: C = W^T @ A^T = W[N,K] @ A[M,K]^T = [N, M]
    
    Args:
        W_q: 量化后的权重矩阵 [N, K] (int8/fp8)
        A_q: 量化后的激活矩阵 [M, K] (int8/fp8)
        R_out: CUDA 输出矩阵 (bf16/fp32)
        M: batch 维度 (用于 A 切片)
        is_col_major: R_out 是否为 Column Major 存储
                      Column Major [N,M] 在 PyTorch 中存储为 [M,N]
        tol: 相对误差容限 (默认 5%)
        critical_tol: 严重误差阈值 (默认 100%)
    
    Returns:
        dict: {
            "valid": bool,          # 是否通过验证
            "max_rel_err": float,   # 最大相对误差
            "passed": bool,         # 误差 < tol
            "critical": bool,       # 误差 > critical_tol (严重错误)
            "message": str,         # 结果描述
        }
    """
    import math
    
    # 使用量化后的数据做 FP32 参考计算
    # 这样才能和 INT8/FP8 GEMM 结果对比
    A_slice = A_q.narrow(0, 0, M)
    A_fp32 = A_slice.to(torch.float32)
    W_fp32 = W_q.to(torch.float32)
    
    # 参考计算: C = W @ A^T -> [N, M]
    ref = torch.matmul(W_fp32, A_fp32.transpose(0, 1))  # [N, M]
    
    # 将输出转为 FP32 比较
    # R_out 的创建已经考虑了布局：
    #   - Column Major 时创建为 [M, N]，转置后得到 [N, M]
    #   - Row Major 时直接创建为 [N, M]
    if is_col_major:
        # R_out 是 [M, N]，转置得到 [N, M] 与 ref 对齐
        out_fp32 = R_out.to(torch.float32).t().contiguous()
    else:
        # Row Major: 直接使用
        out_fp32 = R_out.to(torch.float32)
    
    # 计算相对误差（相对于参考值的绝对值）
    ref_abs = ref.abs().clamp_min(1.0)  # 避免除以0
    rel_diff = ((out_fp32 - ref) / ref_abs).abs()
    max_rel_err = rel_diff.max().item()
    
    # 判断结果
    is_nan = math.isnan(max_rel_err)
    is_critical = max_rel_err > critical_tol or is_nan
    is_passed = max_rel_err <= tol and not is_nan
    
    if is_critical:
        message = f"严重错误: 相对误差={max_rel_err*100:.2f}% > {critical_tol*100:.0f}%"
        valid = False
    elif not is_passed:
        message = f"警告: 相对误差={max_rel_err*100:.2f}% > {tol*100:.0f}%"
        valid = True  # 可用但有警告
    else:
        message = f"通过: 相对误差={max_rel_err*100:.4f}%"
        valid = True
    
    return {
        "valid": valid,
        "max_rel_err": max_rel_err,
        "passed": is_passed,
        "critical": is_critical,
        "message": message,
    }


def verify_search_topk(
    W_q: torch.Tensor,
    A_q: torch.Tensor,
    M_list: List[int],
    topk_results: Dict[str, Any],
    *,
    is_col_major: bool = True,
    run_gemm_func: Optional[Callable] = None,
    tol: float = 0.05,
    critical_tol: float = 1.00,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    验证 search_topk 的结果。
    
    对每个 M 值的 top-1 算法执行实际计算并验证结果正确性。
    
    Args:
        W_q: 量化后的权重矩阵 [N, K]
        A_q: 量化后的激活矩阵 [max_M, K]
        M_list: M 值列表
        topk_results: search_topk 返回的结果字典
        is_col_major: 输出是否为 Column Major
        run_gemm_func: 执行 GEMM 的函数 (W, A, M) -> C
                       如果为 None，则使用 PyTorch 参考实现
        tol: 相对误差容限
        critical_tol: 严重误差阈值
        verbose: 是否打印详细信息
    
    Returns:
        dict: {
            "all_passed": bool,
            "num_verified": int,
            "num_passed": int,
            "num_critical": int,
            "results": List[dict],  # 每个 M 的验证结果
        }
    """
    N, K = W_q.shape
    results = []
    num_passed = 0
    num_critical = 0
    
    for m_idx, M in enumerate(M_list):
        # 检查是否有有效的 top-1 结果
        valid_mask = topk_results.get("valid_mask")
        if valid_mask is not None:
            if hasattr(valid_mask, "numpy"):
                is_valid = valid_mask[m_idx, 0].item() if valid_mask.dim() > 1 else valid_mask[m_idx].item()
            else:
                is_valid = valid_mask[m_idx][0] if len(valid_mask[m_idx]) > 1 else valid_mask[m_idx]
            
            if not is_valid:
                results.append({
                    "M": M,
                    "verified": False,
                    "reason": "无有效算法",
                })
                continue
        
        # 准备输出缓冲区
        outdtype = topk_results.get("outdtype", "bf16")
        out_torch_dtype = torch.bfloat16 if outdtype == "bf16" else torch.float32
        
        if is_col_major:
            # Column Major [N,M] 在 PyTorch 中存储为 [M,N]
            R_out = torch.zeros((M, N), dtype=out_torch_dtype, device=W_q.device)
        else:
            R_out = torch.zeros((N, M), dtype=out_torch_dtype, device=W_q.device)
        
        # 执行 GEMM (使用提供的函数或 PyTorch 参考)
        if run_gemm_func is not None:
            try:
                R_out = run_gemm_func(W_q, A_q, M, m_idx)
            except Exception as e:
                results.append({
                    "M": M,
                    "verified": False,
                    "reason": f"GEMM 执行失败: {e}",
                })
                continue
        else:
            # 使用 PyTorch 参考实现
            A_slice = A_q.narrow(0, 0, M).to(torch.float32)
            W_fp32 = W_q.to(torch.float32)
            ref = torch.matmul(W_fp32, A_slice.transpose(0, 1))
            if is_col_major:
                R_out = ref.t().to(out_torch_dtype)
            else:
                R_out = ref.to(out_torch_dtype)
        
        # 验证
        verify_result = verify_gemm_result(
            W_q, A_q, R_out, M,
            is_col_major=is_col_major,
            tol=tol,
            critical_tol=critical_tol,
        )
        
        if verbose:
            alg_id = -1
            if "topk_alg_id" in topk_results:
                alg_id = topk_results["topk_alg_id"][m_idx, 0].item()
            elif "topk_alg" in topk_results:
                alg_id = topk_results["topk_alg"][m_idx, 0].item()
            
            status = "✓" if verify_result["passed"] else ("✗" if verify_result["critical"] else "⚠")
            print(f"  M={M:5d} alg_id={alg_id:3d} {status} {verify_result['message']}")
        
        if verify_result["passed"]:
            num_passed += 1
        if verify_result["critical"]:
            num_critical += 1
        
        results.append({
            "M": M,
            "verified": True,
            **verify_result,
        })
    
    return {
        "all_passed": num_passed == len(results) and num_critical == 0,
        "num_verified": len([r for r in results if r.get("verified", False)]),
        "num_passed": num_passed,
        "num_critical": num_critical,
        "results": results,
    }
