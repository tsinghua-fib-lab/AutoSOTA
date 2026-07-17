#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 测试工具库

提供统一的测试基础设施，包括：
- 测试状态和结果数据类
- 测试装饰器和运行器
- 环境检测工具
- 模型查找工具
- CUDA 内存管理
- 性能测试工具

环境变量设计（两层控制）:
========================

第一层：是否启用 SlideSparse
    DISABLE_SLIDESPARSE=1  →  禁用 SlideSparse，使用 vLLM 原生路径
    DISABLE_SLIDESPARSE=0  →  启用 SlideSparse（默认）

第二层：Kernel 后端选择（三选一，互斥）
    USE_CUBLASLT=1         →  cuBLASLt kernel
    USE_CUSPARSELT=1       →  cuSPARSELt kernel (TODO)
    默认（两者都不设置）   →  CUTLASS fallback

附加选项：
    INNER_DTYPE_32=1       →  GEMM 使用高精度累加（FP8→FP32, INT8→INT32）

命令行参数映射：
===============
    --disable-slidesparse  →  DISABLE_SLIDESPARSE=1（baseline，不走 SlideSparse）
    --use-cutlass          →  USE_CUBLASLT=0, USE_CUSPARSELT=0（默认）
    --use-cublaslt         →  USE_CUBLASLT=1
    --use-cusparselt       →  USE_CUSPARSELT=1
    --inner-32             →  INNER_DTYPE_32=1
"""

import os

# ============================================================================
# Triton ptxas 兼容性设置（必须在 import triton 之前）
# ============================================================================
# 优先使用系统 CUDA ptxas（支持更新的 GPU 架构如 sm_121）
# Triton 内置的 ptxas 版本可能较旧，不支持最新架构
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

import sys
import time
import ctypes
import ctypes.util
import functools
import traceback
import argparse
import statistics
import importlib.util
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from pathlib import Path


# ============================================================================
# 路径设置
# ============================================================================

# utils.py 现在位于 slidesparse/test/utils.py
# 目录结构: PROJECT_ROOT/slidesparse/test/utils.py
TEST_DIR = Path(__file__).parent.absolute()  # slidesparse/test
SLIDESPARSE_DIR = TEST_DIR.parent            # slidesparse
PROJECT_ROOT = SLIDESPARSE_DIR.parent        # vllmbench (项目根目录)
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 硬件信息工具（从外层 utils 导入）
# ============================================================================

# 延迟导入，避免循环依赖
_hw_info = None
_build_filename = None
_build_stem = None
_build_dir_name = None
_normalize_dtype = None
# 新增：从顶层 utils 导入的模型路径工具
_get_slidesparse_checkpoints_dir = None
_resolve_slidesparse_model_path = None
_find_slidesparse_model = None
_get_sparsity_str = None
_clear_sparsity_cache = None
# 新增：统一的模型路径解析工具
_extract_model_name = None
_resolve_model_path_from_arg = None


def _import_slidesparse_utils():
    """延迟导入 slidesparse.utils"""
    global _hw_info, _build_filename, _build_stem, _build_dir_name, _normalize_dtype
    global _get_slidesparse_checkpoints_dir, _resolve_slidesparse_model_path
    global _find_slidesparse_model, _get_sparsity_str, _clear_sparsity_cache
    global _extract_model_name, _resolve_model_path_from_arg
    if _hw_info is None:
        from slidesparse.utils import (
            hw_info, build_filename, build_stem, build_dir_name, normalize_dtype,
            get_slidesparse_checkpoints_dir, resolve_slidesparse_model_path,
            find_slidesparse_model, get_sparsity_str, clear_sparsity_cache,
            extract_model_name, resolve_model_path_from_arg,
        )
        _hw_info = hw_info
        _build_filename = build_filename
        _build_stem = build_stem
        _build_dir_name = build_dir_name
        _normalize_dtype = normalize_dtype
        _get_slidesparse_checkpoints_dir = get_slidesparse_checkpoints_dir
        _resolve_slidesparse_model_path = resolve_slidesparse_model_path
        _find_slidesparse_model = find_slidesparse_model
        _get_sparsity_str = get_sparsity_str
        _clear_sparsity_cache = clear_sparsity_cache
        _extract_model_name = extract_model_name
        _resolve_model_path_from_arg = resolve_model_path_from_arg


def get_hw_info() -> "HardwareInfo":
    """
    获取硬件信息单例。
    
    Returns:
        HardwareInfo 实例（延迟初始化）
    """
    _import_slidesparse_utils()
    return _hw_info  # type: ignore[return-value]


# HardwareInfo 类型存根（仅用于类型检查）
class HardwareInfo:
    """HardwareInfo 类型存根，实际实现在 slidesparse.utils 中"""
    gpu_name: str
    gpu_full_name: str
    cc_tag: str
    cc_major: int
    cc_minor: int
    sm_code: str
    arch_name: str
    arch_suffix: str
    python_tag: str
    cuda_tag: str
    arch_tag: str
    cuda_runtime_version: str
    cuda_driver_version: str


def get_build_filename(*args, **kwargs) -> str:
    """构建标准化文件名（代理函数）"""
    _import_slidesparse_utils()
    return _build_filename(*args, **kwargs)


def get_build_stem(*args, **kwargs) -> str:
    """构建文件名主干（代理函数）"""
    _import_slidesparse_utils()
    return _build_stem(*args, **kwargs)


def get_build_dir_name(*args, **kwargs) -> str:
    """构建目录名（代理函数）"""
    _import_slidesparse_utils()
    return _build_dir_name(*args, **kwargs)


def get_normalize_dtype(dtype: str) -> str:
    """标准化数据类型名称（代理函数）"""
    _import_slidesparse_utils()
    return _normalize_dtype(dtype)


def extract_model_name(model_name_with_slide: str) -> str:
    """
    从完整模型名中提取基础模型名（代理函数）
    
    委托给顶层 slidesparse.utils.extract_model_name
    """
    _import_slidesparse_utils()
    return _extract_model_name(model_name_with_slide)


def resolve_model_path_from_arg(
    model_arg,
    quant=None,
    checkpoint_dir=None,
    fallback_model="BitNet-2B-BF16",
    warn_on_fallback=True,
):
    """
    从命令行参数解析模型路径（代理函数）
    
    委托给顶层 slidesparse.utils.resolve_model_path_from_arg
    
    Returns:
        (model_path, model_name) 元组
    """
    _import_slidesparse_utils()
    return _resolve_model_path_from_arg(
        model_arg, quant, checkpoint_dir, fallback_model, warn_on_fallback
    )


# ============================================================================
# CUDA 库加载工具（从顶层 utils 导入）
# ============================================================================

# 延迟导入，避免循环依赖
_ensure_cublaslt_loaded = None
_ensure_cusparselt_loaded = None
_load_cuda_extension = None
_SUPPORTED_BACKENDS = None
_BACKEND_LDFLAGS = None
_BACKEND_LOADERS = None


def _import_cuda_utils():
    """延迟导入 slidesparse.utils 中的 CUDA 工具"""
    global _ensure_cublaslt_loaded, _ensure_cusparselt_loaded, _load_cuda_extension
    global _SUPPORTED_BACKENDS, _BACKEND_LDFLAGS, _BACKEND_LOADERS
    if _ensure_cublaslt_loaded is None:
        from slidesparse.utils import (
            ensure_cublaslt_loaded as _ecl,
            ensure_cusparselt_loaded as _ecspl,
            load_cuda_extension as _lce,
            SUPPORTED_BACKENDS as _sb,
            BACKEND_LDFLAGS as _blf,
            BACKEND_LOADERS as _bl,
        )
        _ensure_cublaslt_loaded = _ecl
        _ensure_cusparselt_loaded = _ecspl
        _load_cuda_extension = _lce
        _SUPPORTED_BACKENDS = _sb
        _BACKEND_LDFLAGS = _blf
        _BACKEND_LOADERS = _bl


def ensure_cublaslt_loaded() -> None:
    """优先加载系统或环境变量指定的 cuBLASLt（代理函数）"""
    _import_cuda_utils()
    return _ensure_cublaslt_loaded()


def ensure_cusparselt_loaded() -> None:
    """优先加载系统或环境变量指定的 cuSPARSELt（代理函数）"""
    _import_cuda_utils()
    return _ensure_cusparselt_loaded()


def load_cuda_extension(
    script_type: str,
    backend: str,
    source_file: Path,
    build_dir: Optional[Path] = None,
    *,
    verbose: bool = True,
    force_compile: bool = False,
) -> object:
    """加载或编译 CUDA 扩展（代理函数）"""
    _import_cuda_utils()
    return _load_cuda_extension(
        script_type, backend, source_file, build_dir,
        verbose=verbose, force_compile=force_compile
    )


# 导出常量（直接从顶层导入）
def get_supported_backends():
    """获取支持的后端列表"""
    _import_cuda_utils()
    return _SUPPORTED_BACKENDS

def get_backend_ldflags():
    """获取后端链接标志"""
    _import_cuda_utils()
    return _BACKEND_LDFLAGS

def get_backend_loaders():
    """获取后端加载函数映射"""
    _import_cuda_utils()
    return _BACKEND_LOADERS


def build_output_dir_name(
    model_name: str,
    dtype: str,
    outdtype: str,
) -> str:
    """
    构建输出目录名称。
    
    格式: {GPU}_{CC}_out-{outdtype}_{model_name}_py{PyVer}_cu{CUDAVer}_{arch}
    示例: H100_cc90_out-BF16_BitNet-2B4T-FP8E4M3_py312_cu129_x86_64
    
    Args:
        model_name: 模型名称（如 "BitNet-2B4T-FP8E4M3"，已包含输入dtype后缀）
        dtype: 输入数据类型（如 "int8", "fp8e4m3"）- 已包含在 model_name 中，此参数保留用于兼容性
        outdtype: 输出数据类型（如 "bf16", "fp32"）
    
    Returns:
        目录名称
    """
    hw = get_hw_info()
    outdtype_norm = get_normalize_dtype(outdtype)
    return f"{hw.gpu_name}_{hw.cc_tag}_out-{outdtype_norm}_{model_name}_{hw.python_tag}_{hw.cuda_tag}_{hw.arch_tag}"


def build_result_filename(
    prefix: str,
    model_name: str,
    ext: str = "",
) -> str:
    """
    构建结果文件名称。
    
    格式: {prefix}_{model_name}.{ext}
    示例: alg_search_LUT_BitNet-2B4T.json, layout_search_INFO_BitNet-2B4T.json
    
    Args:
        prefix: 文件前缀（如 "alg_search_LUT", "layout_search_bench"）
        model_name: 模型名称
        ext: 文件扩展名（如 ".json", ".csv"）
    
    Returns:
        文件名称
    """
    name = f"{prefix}_{model_name}"
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        name += ext
    return name


# ============================================================================
# Segment-K 支持检测
# ============================================================================

def supports_segment_k() -> Tuple[bool, str]:
    """
    检测当前 GPU 是否支持 Segment-K (split_k=-1)。
    
    Segment-K 仅在 SM 9.0 (Hopper) 和 SM 10.x (Blackwell) 上支持。
    在 cuSPARSELt 0.8.1 及更早版本中，对不支持的架构调用 Segment-K
    会导致 planInit 阻塞挂起。
    
    Returns:
        (supported, reason_if_not) 其中:
        - supported: 是否支持 Segment-K
        - reason_if_not: 如果不支持，说明原因
    """
    hw = get_hw_info()
    major = hw.cc_major
    
    if major in (9, 10):
        return True, ""
    else:
        return False, (
            f"Segment-K (split_k=-1) 仅在 SM 9.0/10.x (Hopper/Blackwell) 上支持，"
            f"当前架构 SM {hw.cc_major}.{hw.cc_minor} 不支持。"
            f"在 cuSPARSELt 0.8.1 版本中，对不支持的架构调用会导致 planInit 阻塞挂起。"
        )


# ============================================================================
# dtype 兼容性检测（通用版本）
# ============================================================================

# 支持的 dtype 和 outdtype
SUPPORTED_DTYPES = ["int8", "fp8e4m3"]
SUPPORTED_OUTDTYPES = ["bf16", "fp32"]


def probe_cublaslt_alg_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt alg_search 扩展对 dtype 的支持情况。
    
    Args:
        ext: 编译好的 CUDA 扩展模块
        dtype: 要测试的数据类型
        outdtype: 输出数据类型
        layout: 布局类型
    
    Returns:
        (supported, message)
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        out = ext.search_topk(
            W, A,
            [M],           # M_list
            layout,
            dtype,
            outdtype,
            1, 1,          # warmup, repeat
            False,         # verify
            [],            # blacklist
            1,             # topk
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法（可能不支持）"
            
    except Exception as e:
        error_msg = str(e)
        if "CUBLAS" in error_msg:
            return False, f"cuBLASLt 不支持 dtype={dtype}: {error_msg}"
        elif "不支持的数据类型" in error_msg:
            return False, f"dtype={dtype} 不被支持"
        else:
            return False, f"dtype={dtype} 测试失败: {error_msg}"
    finally:
        torch.cuda.empty_cache()


def probe_cublaslt_layout_bench(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt layout_bench 扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",  # TN+CC+Col
            dtype,
            outdtype,
            1, 1,  # warmup, repeat
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def probe_cusparselt_alg_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt alg_search 扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        # cuSPARSELt 需要先剪枝
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
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法（可能不支持）"
            
    except Exception as e:
        error_msg = str(e)
        if "CUSPARSE_STATUS" in error_msg:
            return False, f"cuSPARSELt 不支持 dtype={dtype}: {error_msg}"
        elif "不支持的数据类型" in error_msg:
            return False, f"dtype={dtype} 不被支持"
        else:
            return False, f"dtype={dtype} 测试失败: {error_msg}"
    finally:
        torch.cuda.empty_cache()


def probe_cusparselt_layout_bench(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt layout_bench 扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",  # TN+CC+Col
            dtype,
            outdtype,
            1, 1,  # warmup, repeat
            False, # test_segment_k (探测时不测试 Segment-K)
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


# dtype 探测函数注册表
DTYPE_PROBERS = {
    ("cublaslt", "alg_search"): probe_cublaslt_alg_search,
    ("cublaslt", "layout_bench"): probe_cublaslt_layout_bench,
    ("cublaslt", "layout_search"): probe_cublaslt_layout_bench,  # layout_search 使用相同探测函数
    ("cusparselt", "alg_search"): probe_cusparselt_alg_search,
    ("cusparselt", "layout_bench"): probe_cusparselt_layout_bench,
    ("cusparselt", "layout_search"): probe_cusparselt_layout_bench,  # layout_search 使用相同探测函数
}


def check_dtype_support(
    ext,
    dtype: str,
    outdtype: str,
    arch_name: str,
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
        arch_name: 架构名称（用于显示）
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
    prober_key = (backend.lower(), script_type.lower())
    if prober_key not in DTYPE_PROBERS:
        raise ValueError(f"未知的 backend/script_type 组合: {prober_key}")
    
    prober = DTYPE_PROBERS[prober_key]
    
    if verbose:
        print(f"[预测试] 检测 dtype={dtype}, outdtype={outdtype} 在 {arch_name} 上的支持情况...", end=" ", flush=True)
    
    supported, message = prober(ext, dtype, outdtype)
    
    if supported:
        if verbose:
            print("✓", flush=True)
    else:
        if verbose:
            print("✗", flush=True)
        raise ValueError(
            f"数据类型 {dtype.upper()} 在当前 GPU ({arch_name}) 上不可用。\n"
            f"原因: {message}\n"
        )


# ============================================================================
# 搜索结果元数据构建（公共部分）
# ============================================================================

def build_search_meta(
    hw: "HardwareInfo",
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List,
    *,
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
    model_name: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    构建搜索结果的元数据（公共部分）。
    
    Args:
        hw: 硬件信息
        dtype: 输入数据类型
        outdtype: 输出数据类型
        warmup: 预热次数
        repeat: 重复次数
        verify: 是否验证
        m_list: M 列表
        nk_list: NK 列表
        layout: 布局类型
        alg_count: 算法数量
        config_count: 配置数量
        model_name: 模型名称
        extra: 额外的元数据字段
    
    Returns:
        元数据字典
    """
    import datetime
    import torch
    
    meta = {
        "gpu_name": hw.gpu_full_name,
        "gpu_short_name": hw.gpu_name,
        "compute_capability": hw.cc_tag,
        "arch_name": hw.arch_name,
        "layout": layout,
        "dtype": dtype,
        "outdtype": outdtype,
        "alg_count": alg_count,
        "config_count": config_count,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": hw.cuda_driver_version,
        "cuda_version_runtime": hw.cuda_runtime_version,
        "time": datetime.datetime.now().isoformat(),
        "M_list": m_list,
        "NK_list": list(nk_list),
    }
    
    if model_name:
        meta["model_name"] = model_name
    
    if extra:
        meta.update(extra)
    
    return meta


def build_csv_header(
    hw: "HardwareInfo",
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List,
    *,
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
    model_name: Optional[str] = None,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    """
    构建 CSV 文件的注释头部。
    
    Returns:
        注释行列表
    """
    import torch
    
    lines = [
        f"# GPU: {hw.gpu_full_name}",
        f"# CC: {hw.cc_tag}",
    ]
    
    if model_name:
        lines.append(f"# Model: {model_name}")
    
    lines.extend([
        f"# alg_count: {alg_count}, config_count: {config_count}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {hw.cuda_driver_version}, runtime: {hw.cuda_runtime_version}",
        f"# layout: {layout}, dtype: {dtype}, outdtype: {outdtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {m_list}",
        f"# NK_list: {list(nk_list)}",
    ])
    
    if extra_lines:
        lines.extend(extra_lines)
    
    return lines


# ============================================================================
# ANSI 颜色
# ============================================================================

class Colors:
    """终端颜色"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.NC}"
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.NC}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.NC}"
    
    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.NC}"
    
    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls.CYAN}{text}{cls.NC}"
    
    @classmethod
    def magenta(cls, text: str) -> str:
        return f"{cls.MAGENTA}{text}{cls.NC}"
    
    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.NC}"


# ============================================================================
# 测试状态枚举
# ============================================================================

class TestStatus(Enum):
    """测试状态"""
    PASSED = "✓"
    FAILED = "✗"
    SKIPPED = "⊘"
    WARNING = "⚠"


# ============================================================================
# 测试结果数据类
# ============================================================================

@dataclass
class TestResult:
    """单个测试的结果"""
    name: str
    status: TestStatus
    message: str = ""
    duration: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status_char = self.status.value
        if self.status == TestStatus.PASSED:
            status_str = Colors.green(f"{status_char} {self.name}")
        elif self.status == TestStatus.FAILED:
            status_str = Colors.red(f"{status_char} {self.name}")
        elif self.status == TestStatus.SKIPPED:
            status_str = Colors.yellow(f"{status_char} {self.name}")
        else:
            status_str = Colors.yellow(f"{status_char} {self.name}")
        
        if self.message:
            status_str += f": {self.message}"
        if self.duration > 0:
            status_str += f" ({self.duration:.2f}s)"
        return status_str


@dataclass  
class TestSuiteResult:
    """测试套件的结果"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success(self) -> bool:
        return self.failed == 0
    
    def add(self, result: TestResult):
        self.results.append(result)
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"测试套件: {self.name}",
            "-" * 60,
        ]
        
        passed_str = Colors.green(f"通过: {self.passed}")
        failed_str = Colors.red(f"失败: {self.failed}") if self.failed > 0 else f"失败: {self.failed}"
        skipped_str = Colors.yellow(f"跳过: {self.skipped}") if self.skipped > 0 else f"跳过: {self.skipped}"
        
        lines.append(f"{passed_str}  {failed_str}  {skipped_str}")
        lines.append(f"耗时: {self.duration:.2f}s")
        lines.append("-" * 60)
        
        for r in self.results:
            lines.append(f"  {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# 测试装饰器
# ============================================================================

def test_case(name: str = None, skip_if: Callable[[], Tuple[bool, str]] = None):
    """
    测试用例装饰器
    
    Args:
        name: 测试名称（默认使用函数名）
        skip_if: 跳过条件函数，返回 (should_skip, reason)
    
    Example:
        @test_case("导入测试")
        def test_import():
            import slidesparse
            return True, "导入成功"
            
        @test_case(skip_if=lambda: (not torch.cuda.is_available(), "需要 CUDA"))
        def test_cuda_kernel():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> TestResult:
            test_name = name or func.__name__
            
            # 检查跳过条件
            if skip_if:
                should_skip, reason = skip_if()
                if should_skip:
                    return TestResult(
                        name=test_name,
                        status=TestStatus.SKIPPED,
                        message=reason
                    )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 处理返回值
                if isinstance(result, TestResult):
                    result.duration = duration
                    return result
                elif isinstance(result, tuple) and len(result) >= 2:
                    success, message = result[0], result[1]
                    details = result[2] if len(result) > 2 else {}
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED if success else TestStatus.FAILED,
                        message=message,
                        duration=duration,
                        details=details
                    )
                elif isinstance(result, bool):
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED if result else TestStatus.FAILED,
                        duration=duration
                    )
                else:
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED,
                        duration=duration
                    )
                    
            except Exception as e:
                duration = time.time() - start_time
                return TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    message=str(e),
                    duration=duration,
                    details={"traceback": traceback.format_exc()}
                )
        
        return wrapper
    return decorator


# ============================================================================
# 环境检测工具
# ============================================================================

class EnvironmentChecker:
    """环境检测工具类"""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
    
    @classmethod
    def has_cuda(cls) -> bool:
        """检查 CUDA 是否可用"""
        if "has_cuda" not in cls._cache:
            try:
                import torch
                cls._cache["has_cuda"] = torch.cuda.is_available()
            except ImportError:
                cls._cache["has_cuda"] = False
        return cls._cache["has_cuda"]
    
    @classmethod
    def cuda_device_name(cls) -> str:
        """获取 CUDA 设备名称"""
        if not cls.has_cuda():
            return "N/A"
        if "cuda_device_name" not in cls._cache:
            import torch
            cls._cache["cuda_device_name"] = torch.cuda.get_device_name(0)
        return cls._cache["cuda_device_name"]
    
    @classmethod
    def cuda_compute_capability(cls) -> Tuple[int, int]:
        """获取 CUDA 计算能力"""
        if not cls.has_cuda():
            return (0, 0)
        if "cuda_cc" not in cls._cache:
            import torch
            cls._cache["cuda_cc"] = torch.cuda.get_device_capability(0)
        return cls._cache["cuda_cc"]
    
    @classmethod
    def supports_fp8(cls) -> bool:
        """检查是否支持 FP8 (sm_89+)"""
        cc = cls.cuda_compute_capability()
        return cc >= (8, 9)
    
    @classmethod
    def supports_int8(cls) -> bool:
        """检查是否支持 INT8 (sm_75+, Turing 及以上)"""
        cc = cls.cuda_compute_capability()
        return cc >= (7, 5)
    
    @classmethod
    def supports_cutlass_fp8(cls) -> bool:
        """
        检查 vLLM CUTLASS FP8 kernel 是否支持当前 GPU
        
        vLLM 的 CUTLASS FP8 kernel 只支持 sm_89 ~ sm_120 (Ada ~ Blackwell RTX 50xx)
        - sm_80~88: 会 fallback 到 Marlin W8A16 kernel（不是真正的 FP8 计算）
        - sm_121+: vLLM CUTLASS kernel 未编译支持
        
        转发自 slidesparse.utils.hw_info.vllm_cutlass_fp8_supported
        """
        if "supports_cutlass_fp8" not in cls._cache:
            try:
                from slidesparse.utils import hw_info
                cls._cache["supports_cutlass_fp8"] = hw_info.vllm_cutlass_fp8_supported
                cls._cache["supports_cutlass_fp8_reason"] = hw_info.supports_vllm_cutlass_fp8[1]
            except ImportError:
                # Fallback: 本地计算
                cc = cls.cuda_compute_capability()
                cls._cache["supports_cutlass_fp8"] = (8, 9) <= cc <= (12, 0)
                cls._cache["supports_cutlass_fp8_reason"] = f"sm_{cc[0]}{cc[1]}"
        return cls._cache["supports_cutlass_fp8"]
    
    @classmethod
    def supports_cutlass_fp8_reason(cls) -> str:
        """获取 FP8 CUTLASS 支持/不支持的原因"""
        cls.supports_cutlass_fp8()  # 确保缓存已填充
        return cls._cache.get("supports_cutlass_fp8_reason", "unknown")
    
    @classmethod
    def supports_cutlass_int8(cls) -> bool:
        """
        检查 vLLM CUTLASS INT8 kernel 是否支持当前 GPU
        
        vLLM 的 CUTLASS INT8 kernel 只支持 sm_75 ~ sm_90 (Turing ~ Hopper)
        sm_100+ (Blackwell) 上 vLLM 会报错 "Int8 not supported on SM1xx"
        
        转发自 slidesparse.utils.hw_info.vllm_cutlass_int8_supported
        """
        if "supports_cutlass_int8" not in cls._cache:
            try:
                from slidesparse.utils import hw_info
                cls._cache["supports_cutlass_int8"] = hw_info.vllm_cutlass_int8_supported
                cls._cache["supports_cutlass_int8_reason"] = hw_info.supports_vllm_cutlass_int8[1]
            except ImportError:
                # Fallback: 本地计算
                cc = cls.cuda_compute_capability()
                cls._cache["supports_cutlass_int8"] = (7, 5) <= cc < (10, 0)
                cls._cache["supports_cutlass_int8_reason"] = f"sm_{cc[0]}{cc[1]}"
        return cls._cache["supports_cutlass_int8"]
    
    @classmethod
    def supports_cutlass_int8_reason(cls) -> str:
        """获取 INT8 CUTLASS 支持/不支持的原因"""
        cls.supports_cutlass_int8()  # 确保缓存已填充
        return cls._cache.get("supports_cutlass_int8_reason", "unknown")
    
    @classmethod
    def supports_cublaslt(cls) -> bool:
        """
        检查 cuBLASLt 是否支持当前 GPU
        
        cuBLASLt 支持 sm_70+ (Volta 及更新架构)。
        转发自 slidesparse.utils.hw_info.cublaslt_supported
        """
        if "supports_cublaslt" not in cls._cache:
            try:
                from slidesparse.utils import hw_info
                cls._cache["supports_cublaslt"] = hw_info.cublaslt_supported
                cls._cache["supports_cublaslt_reason"] = hw_info.supports_cublaslt[1]
            except ImportError:
                # Fallback: 本地计算
                cc = cls.cuda_compute_capability()
                cls._cache["supports_cublaslt"] = cc >= (7, 0)
                cls._cache["supports_cublaslt_reason"] = f"sm_{cc[0]}{cc[1]}"
        return cls._cache["supports_cublaslt"]
    
    @classmethod
    def supports_cublaslt_reason(cls) -> str:
        """获取 cuBLASLt 支持/不支持的原因"""
        cls.supports_cublaslt()  # 确保缓存已填充
        return cls._cache.get("supports_cublaslt_reason", "unknown")
    
    @classmethod
    def supports_cusparselt(cls) -> bool:
        """
        检查 cuSPARSELt 是否支持当前 GPU
        
        cuSPARSELt 支持 sm_80+ (Ampere 及更新架构)。
        转发自 slidesparse.utils.hw_info.cusparselt_supported
        """
        if "supports_cusparselt" not in cls._cache:
            try:
                from slidesparse.utils import hw_info
                cls._cache["supports_cusparselt"] = hw_info.cusparselt_supported
                cls._cache["supports_cusparselt_reason"] = hw_info.supports_cusparselt[1]
            except ImportError:
                # Fallback: 本地计算
                cc = cls.cuda_compute_capability()
                cls._cache["supports_cusparselt"] = cc >= (8, 0)
                cls._cache["supports_cusparselt_reason"] = f"sm_{cc[0]}{cc[1]}"
        return cls._cache["supports_cusparselt"]
    
    @classmethod
    def supports_cusparselt_reason(cls) -> str:
        """获取 cuSPARSELt 支持/不支持的原因"""
        cls.supports_cusparselt()  # 确保缓存已填充
        return cls._cache.get("supports_cusparselt_reason", "unknown")
    
    @classmethod
    def is_slidesparse_enabled(cls) -> bool:
        """检查 SlideSparse 是否启用"""
        if "slidesparse_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_slidesparse_enabled
                cls._cache["slidesparse_enabled"] = is_slidesparse_enabled()
            except ImportError:
                cls._cache["slidesparse_enabled"] = False
        return cls._cache["slidesparse_enabled"]
    
    @classmethod
    def is_cublaslt_enabled(cls) -> bool:
        """检查 cuBLASLt kernel 是否启用"""
        if "cublaslt_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_cublaslt_enabled
                cls._cache["cublaslt_enabled"] = is_cublaslt_enabled()
            except ImportError:
                cls._cache["cublaslt_enabled"] = False
        return cls._cache["cublaslt_enabled"]
    
    @classmethod
    def is_cusparselt_enabled(cls) -> bool:
        """检查 cuSPARSELt kernel 是否启用"""
        if "cusparselt_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_cusparselt_enabled
                cls._cache["cusparselt_enabled"] = is_cusparselt_enabled()
            except ImportError:
                cls._cache["cusparselt_enabled"] = False
        return cls._cache["cusparselt_enabled"]
    
    @classmethod
    def is_inner_dtype_32(cls) -> bool:
        """检查 INNER_DTYPE_32 是否启用（高精度累加）"""
        if "inner_dtype_32" not in cls._cache:
            try:
                from slidesparse.core.config import is_inner_dtype_32
                cls._cache["inner_dtype_32"] = is_inner_dtype_32()
            except ImportError:
                cls._cache["inner_dtype_32"] = False
        return cls._cache["inner_dtype_32"]
    
    @classmethod
    def get_kernel_name(cls) -> str:
        """获取当前选择的 kernel 名称"""
        if not cls.is_slidesparse_enabled():
            return "vLLM 原生 (CUTLASS)"
        if cls.is_cublaslt_enabled():
            inner = "FP32/INT32" if cls.is_inner_dtype_32() else "BF16"
            return f"cuBLASLt ({inner})"
        if cls.is_cusparselt_enabled():
            inner = "FP32/INT32" if cls.is_inner_dtype_32() else "BF16"
            return f"cuSPARSELt ({inner})"
        return "CUTLASS (fallback)"
    
    @classmethod
    def get_env_info(cls) -> Dict[str, Any]:
        """获取完整环境信息"""
        info = {
            "has_cuda": cls.has_cuda(),
            "cuda_device": cls.cuda_device_name(),
            "cuda_cc": cls.cuda_compute_capability(),
            "supports_fp8": cls.supports_fp8(),
            "slidesparse_enabled": cls.is_slidesparse_enabled(),
            "cublaslt_enabled": cls.is_cublaslt_enabled(),
            "cusparselt_enabled": cls.is_cusparselt_enabled(),
            "inner_dtype_32": cls.is_inner_dtype_32(),
            "kernel_name": cls.get_kernel_name(),
        }
        
        # Python 版本
        info["python_version"] = sys.version.split()[0]
        
        # PyTorch 版本
        try:
            import torch
            info["torch_version"] = torch.__version__
        except ImportError:
            info["torch_version"] = "N/A"
        
        # vLLM 版本
        try:
            import vllm
            info["vllm_version"] = vllm.__version__
        except ImportError:
            info["vllm_version"] = "N/A"
        
        return info
    
    @classmethod
    def print_env_info(cls):
        """打印环境信息"""
        info = cls.get_env_info()
        print("=" * 70)
        print(Colors.bold("环境信息"))
        print("-" * 70)
        print(f"  Python: {info['python_version']}")
        print(f"  PyTorch: {info['torch_version']}")
        print(f"  vLLM: {info['vllm_version']}")
        print(f"  CUDA: {'可用' if info['has_cuda'] else '不可用'}")
        if info['has_cuda']:
            print(f"  GPU: {info['cuda_device']}")
            print(f"  Compute Capability: sm_{info['cuda_cc'][0]}{info['cuda_cc'][1]}")
            fp8_status = Colors.green("支持") if info['supports_fp8'] else Colors.red("不支持")
            print(f"  FP8: {fp8_status}")
        
        print("-" * 70)
        
        # SlideSparse 状态
        if info['slidesparse_enabled']:
            slidesparse_status = Colors.green("启用")
        else:
            slidesparse_status = Colors.yellow("禁用 (vLLM 原生路径)")
        print(f"  SlideSparse: {slidesparse_status}")
        
        if info['slidesparse_enabled']:
            # Kernel 选择（三选一）
            if info['cublaslt_enabled']:
                kernel_status = Colors.cyan("cuBLASLt")
            elif info['cusparselt_enabled']:
                kernel_status = Colors.magenta("cuSPARSELt")
            else:
                kernel_status = Colors.yellow("CUTLASS (fallback)")
            print(f"  Kernel 后端: {kernel_status}")
            
            # Inner dtype（仅 cuBLASLt/cuSPARSELt 时显示）
            if info['cublaslt_enabled'] or info['cusparselt_enabled']:
                inner_dtype = "FP32/INT32" if info['inner_dtype_32'] else "BF16"
                print(f"  GEMM Inner Dtype: {inner_dtype}")
        
        print("=" * 70)


# ============================================================================
# 模型查找工具
# ============================================================================

class ModelFinder:
    """模型查找工具类"""
    
    # 支持的模型类型
    MODEL_TYPES = {
        "FP8": ["FP8", "fp8", "Fp8"],
        "INT8": ["INT8", "int8", "Int8", "W8A8", "w8a8"],
    }
    
    # 推荐的小模型列表（按优先级排序）
    SMALL_MODELS = [
        "Qwen2.5-0.5B",
        "Qwen2.5-1.5B", 
        "Llama3.2-1B",
        "Qwen2.5-3B",
        "Llama3.2-3B",
    ]
    
    @classmethod
    def get_checkpoints_dir(cls) -> Path:
        """获取 checkpoints 目录路径"""
        return PROJECT_ROOT / "checkpoints"
    
    @classmethod
    def find_models(cls, model_type: str = None) -> List[Path]:
        """
        查找可用模型
        
        Args:
            model_type: 模型类型 ("FP8", "INT8", None=全部)
        
        Returns:
            模型路径列表
        """
        checkpoints_dir = cls.get_checkpoints_dir()
        if not checkpoints_dir.exists():
            return []
        
        models = []
        type_keywords = []
        if model_type and model_type in cls.MODEL_TYPES:
            type_keywords = cls.MODEL_TYPES[model_type]
        
        for item in checkpoints_dir.iterdir():
            if not item.is_dir():
                continue
            
            # 检查是否匹配类型
            if type_keywords:
                if not any(kw in item.name for kw in type_keywords):
                    continue
            
            # 检查是否有必要的文件
            if (item / "config.json").exists():
                models.append(item)
        
        return models
    
    @classmethod
    def find_small_model(cls, model_type: str = "FP8") -> Optional[Path]:
        """
        查找推荐的小模型
        
        Args:
            model_type: 模型类型
        
        Returns:
            模型路径或 None
        """
        models = cls.find_models(model_type)
        if not models:
            return None
        
        # 按推荐顺序查找
        for base_name in cls.SMALL_MODELS:
            for model_path in models:
                if base_name in model_path.name:
                    return model_path
        
        # 返回第一个找到的
        return models[0] if models else None
    
    @classmethod
    def get_test_models(cls, model_type: str = "FP8", max_count: int = 2) -> List[Path]:
        """
        获取测试用模型列表
        
        优先返回 Qwen2.5-0.5B 和 Llama3.2-1B
        """
        priority = ["Qwen2.5-0.5B", "Llama3.2-1B"]
        models = cls.find_models(model_type)
        
        result = []
        # 先添加优先模型
        for name in priority:
            for model in models:
                if name in model.name and model not in result:
                    result.append(model)
                    if len(result) >= max_count:
                        break
            if len(result) >= max_count:
                break
        
        # 如果不够，补充其他模型
        for model in models:
            if model not in result:
                result.append(model)
            if len(result) >= max_count:
                break
        
        return result
    
    @classmethod
    def get_slidesparse_checkpoints_dir(cls) -> Path:
        """
        获取 SlideSparse checkpoints 目录
        
        委托给顶层 slidesparse.utils.get_slidesparse_checkpoints_dir
        """
        _import_slidesparse_utils()
        return _get_slidesparse_checkpoints_dir()
    
    @classmethod
    def resolve_slidesparse_model_path(
        cls, 
        base_model_path: Path,
        sparsity: str = None,
    ) -> Optional[Path]:
        """
        智能解析 SlideSparse 模型路径
        
        委托给顶层 slidesparse.utils.resolve_slidesparse_model_path
        
        Args:
            base_model_path: 基础模型路径
            sparsity: 稀疏格式（如果为 None，从环境变量获取）
        
        Returns:
            SlideSparse checkpoint 路径
        """
        _import_slidesparse_utils()
        return _resolve_slidesparse_model_path(base_model_path, sparsity)
    
    @classmethod
    def find_slidesparse_model(
        cls, 
        model_type: str = "FP8",
        sparsity: str = None,
    ) -> Optional[Path]:
        """
        查找适合测试的 SlideSparse 模型
        
        委托给顶层 slidesparse.utils.find_slidesparse_model
        
        Args:
            model_type: 模型类型
            sparsity: 稀疏格式（如果为 None，从环境变量获取）
        
        Returns:
            SlideSparse checkpoint 路径
        """
        _import_slidesparse_utils()
        return _find_slidesparse_model(model_type, sparsity)
    
    @classmethod
    def resolve_model_path(cls, model_arg: str, quant: str = "FP8") -> Optional[Path]:
        """
        解析模型路径参数（委托给顶层 slidesparse.utils.resolve_model_path_from_arg）
        
        支持以下格式：
        - 完整路径: checkpoints/Llama3.2-1B-FP8
        - 模型名称: Llama3.2-1B-FP8
        - 模型名称不带量化后缀: Llama3.2-1B (会自动添加 -FP8 或 -INT8)
        
        Args:
            model_arg: 模型路径或名称
            quant: 量化类型 ("FP8" 或 "INT8")
        
        Returns:
            解析后的模型路径，如果找不到返回 None
        
        Note:
            此方法保留向后兼容，但内部委托给顶层统一实现。
            新代码应直接使用 slidesparse.utils.resolve_model_path_from_arg。
        """
        if model_arg is None:
            return None
        
        _import_slidesparse_utils()
        try:
            path, _ = _resolve_model_path_from_arg(
                model_arg, 
                quant=quant, 
                warn_on_fallback=False  # 这里不需要 fallback 警告
            )
            return path
        except ValueError:
            return None


# ============================================================================
# 资源管理工具
# ============================================================================

@contextmanager
def cuda_memory_manager():
    """CUDA 内存管理上下文"""
    try:
        yield
    finally:
        if EnvironmentChecker.has_cuda():
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@contextmanager
def suppress_vllm_logs():
    """抑制 vLLM 的日志输出"""
    old_level = os.environ.get("VLLM_LOGGING_LEVEL", "")
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    try:
        yield
    finally:
        if old_level:
            os.environ["VLLM_LOGGING_LEVEL"] = old_level
        else:
            os.environ.pop("VLLM_LOGGING_LEVEL", None)


# ============================================================================
# 测试运行器
# ============================================================================

class TestRunner:
    """测试运行器"""
    
    def __init__(self, suite_name: str, verbose: bool = True):
        self.suite_name = suite_name
        self.verbose = verbose
        self.result = TestSuiteResult(name=suite_name)
    
    def run_test(self, test_func: Callable, *args, **kwargs) -> TestResult:
        """运行单个测试"""
        if self.verbose:
            print(f"\n  运行: {test_func.__name__}...")
        
        result = test_func(*args, **kwargs)
        self.result.add(result)
        
        if self.verbose:
            print(f"    {result}")
        
        return result
    
    def run_all(self, tests: List[Callable]) -> TestSuiteResult:
        """运行所有测试"""
        self.result.start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print(Colors.bold(f"测试套件: {self.suite_name}"))
            print("=" * 60)
        
        for test in tests:
            self.run_test(test)
        
        self.result.end_time = time.time()
        
        if self.verbose:
            print("\n" + self.result.summary())
        
        return self.result


# ============================================================================
# 性能测试工具
# ============================================================================

class Benchmarker:
    """性能测试工具"""
    
    @staticmethod
    def benchmark(
        func: Callable,
        warmup: int = 10,
        repeat: int = 100,
        synchronize: bool = True
    ) -> Tuple[float, float]:
        """
        执行基准测试
        
        Returns:
            (平均时间 ms, 标准差 ms)
        """
        import torch
        
        # Warmup
        for _ in range(warmup):
            func()
        
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            func()
            if synchronize and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        return mean_time, std_time
    
    @staticmethod
    def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
        """计算 GEMM TFLOPS"""
        if time_ms <= 0:
            return 0.0
        flops = 2 * M * N * K
        return flops / (time_ms * 1e-3) / 1e12


# ============================================================================
# 跳过条件
# ============================================================================

def skip_if_no_cuda() -> Tuple[bool, str]:
    """如果没有 CUDA 则跳过"""
    return (not EnvironmentChecker.has_cuda(), "需要 CUDA")


def skip_if_no_fp8() -> Tuple[bool, str]:
    """如果不支持 FP8 则跳过"""
    if not EnvironmentChecker.has_cuda():
        return (True, "需要 CUDA")
    if not EnvironmentChecker.supports_fp8():
        cc = EnvironmentChecker.cuda_compute_capability()
        return (True, f"需要 sm_89+, 当前 sm_{cc[0]}{cc[1]}")
    return (False, "")


def skip_if_no_int8() -> Tuple[bool, str]:
    """如果不支持 INT8 则跳过"""
    if not EnvironmentChecker.has_cuda():
        return (True, "需要 CUDA")
    if not EnvironmentChecker.supports_int8():
        cc = EnvironmentChecker.cuda_compute_capability()
        return (True, f"需要 sm_75+, 当前 sm_{cc[0]}{cc[1]}")
    return (False, "")


def skip_if_no_model(model_type: str = "FP8") -> Tuple[bool, str]:
    """如果没有指定类型的模型则跳过"""
    model = ModelFinder.find_small_model(model_type)
    if model is None:
        return (True, f"未找到 {model_type} 模型")
    return (False, "")


def skip_if_slidesparse_disabled() -> Tuple[bool, str]:
    """如果 SlideSparse 禁用则跳过"""
    if not EnvironmentChecker.is_slidesparse_enabled():
        return (True, "SlideSparse 未启用 (DISABLE_SLIDESPARSE=1)")
    return (False, "")


def skip_if_cublaslt_disabled() -> Tuple[bool, str]:
    """如果 cuBLASLt 禁用则跳过"""
    if not EnvironmentChecker.is_cublaslt_enabled():
        return (True, "cuBLASLt 未启用 (设置 USE_CUBLASLT=1)")
    return (False, "")


def skip_if_cusparselt_disabled() -> Tuple[bool, str]:
    """如果 cuSPARSELt 禁用则跳过"""
    if not EnvironmentChecker.is_cusparselt_enabled():
        return (True, "cuSPARSELt 未启用 (设置 USE_CUSPARSELT=1)")
    return (False, "")


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_common_args(description: str) -> argparse.ArgumentParser:
    """
    解析通用命令行参数
    
    环境变量设计（两层）：
    ====================
    
    第一层：是否启用 SlideSparse
        --disable-slidesparse  →  DISABLE_SLIDESPARSE=1（baseline）
        默认                   →  DISABLE_SLIDESPARSE=0（启用 SlideSparse）
    
    第二层：Kernel 后端选择（三选一，互斥）
        --use-cutlass    →  CUTLASS fallback（默认）
        --use-cublaslt   →  USE_CUBLASLT=1
        --use-cusparselt →  USE_CUSPARSELT=1
    
    附加选项：
        --inner-32       →  INNER_DTYPE_32=1
        --sparsity 2_8   →  SPARSITY=2_8（仅 cuSPARSELt 时生效）
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 默认: SlideSparse + CUTLASS fallback
  %(prog)s --disable-slidesparse    # vLLM 原生路径 (baseline)
  %(prog)s --use-cublaslt           # SlideSparse + cuBLASLt
  %(prog)s --use-cublaslt --inner-32  # cuBLASLt + 高精度累加
  %(prog)s --use-cusparselt --sparsity 2_8  # SlideSparse + cuSPARSELt (2:8 稀疏)
        """
    )
    
    # 第一层：SlideSparse 开关
    parser.add_argument(
        "--disable-slidesparse", 
        action="store_true",
        help="禁用 SlideSparse，使用 vLLM 原生路径 (baseline)"
    )
    
    # 第二层：Kernel 后端选择（互斥组）
    kernel_group = parser.add_mutually_exclusive_group()
    kernel_group.add_argument(
        "--use-cutlass", 
        action="store_true",
        help="使用 CUTLASS fallback（默认）"
    )
    kernel_group.add_argument(
        "--use-cublaslt", 
        action="store_true",
        help="使用 cuBLASLt kernel"
    )
    kernel_group.add_argument(
        "--use-cusparselt", 
        action="store_true",
        help="使用 cuSPARSELt kernel"
    )
    
    # 附加选项
    parser.add_argument(
        "--inner-32", 
        action="store_true", 
        help="GEMM 使用高精度累加（FP8→FP32, INT8→INT32）"
    )
    
    # cuSPARSELt 专用选项
    parser.add_argument(
        "--sparsity",
        type=str,
        default="2_8",
        help="稀疏格式，如 2_8, 2_6, 2_10（仅 cuSPARSELt 时生效，默认 2_8）"
    )
    
    # 模型路径选项
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径或名称（如 checkpoints/Llama3.2-1B-FP8 或 Llama3.2-1B-FP8）"
    )
    
    return parser


def apply_env_args(args: argparse.Namespace, model_name: str = None) -> None:
    """
    应用环境变量参数
    
    根据命令行参数设置对应的环境变量
    
    Args:
        args: 命令行参数
        model_name: 模型名称（用于初始化 SlideSparse），默认 "Llama3.2-1B-FP8"
    """
    # 第一层：DISABLE_SLIDESPARSE
    if getattr(args, 'disable_slidesparse', False):
        os.environ["DISABLE_SLIDESPARSE"] = "1"
        # baseline 模式下，清除其他环境变量
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("INNER_DTYPE_32", None)
    else:
        os.environ["DISABLE_SLIDESPARSE"] = "0"
        
        # 第二层：Kernel 后端选择
        if getattr(args, 'use_cublaslt', False):
            os.environ["USE_CUBLASLT"] = "1"
            os.environ.pop("USE_CUSPARSELT", None)
        elif getattr(args, 'use_cusparselt', False):
            os.environ["USE_CUSPARSELT"] = "1"
            os.environ.pop("USE_CUBLASLT", None)
        else:
            # 默认或 --use-cutlass: CUTLASS fallback
            os.environ.pop("USE_CUBLASLT", None)
            os.environ.pop("USE_CUSPARSELT", None)
        
        # 附加选项：INNER_DTYPE_32
        if getattr(args, 'inner_32', False):
            os.environ["INNER_DTYPE_32"] = "1"
        else:
            os.environ.pop("INNER_DTYPE_32", None)
    
    # 清除缓存以重新读取环境变量
    EnvironmentChecker.clear_cache()
    
    # 初始化 SlideSparse（设置 model_name）
    # 只有在使用 cuBLASLt 或 cuSPARSELt 时才需要
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    use_cusparselt = getattr(args, 'use_cusparselt', False)
    if not getattr(args, 'disable_slidesparse', False) and (use_cublaslt or use_cusparselt):
        try:
            from slidesparse.core import init_slidesparse
            effective_model_name = model_name or "Llama3.2-1B-FP8"
            init_slidesparse(effective_model_name)
        except ImportError:
            pass


def get_backend_name(use_cublaslt: bool = False, use_cusparselt: bool = False, 
                     inner_32: bool = False, sparsity: str = "2_8") -> str:
    """
    根据参数获取后端名称
    
    用于测试输出显示
    """
    if use_cublaslt:
        suffix = " (高精度累加)" if inner_32 else ""
        return f"SlideSparse + cuBLASLt{suffix}"
    elif use_cusparselt:
        suffix = " (高精度累加)" if inner_32 else ""
        return f"SlideSparse + cuSPARSELt ({sparsity.replace('_', ':')}){suffix}"
    else:
        return "SlideSparse + CUTLASS"


def set_env_for_baseline() -> Dict[str, Optional[str]]:
    """
    设置环境变量为 baseline (vLLM 原生路径)
    
    Returns:
        保存的原环境变量，用于恢复
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_32": os.environ.get("INNER_DTYPE_32"),
        "SPARSITY": os.environ.get("SPARSITY"),
    }
    
    os.environ["DISABLE_SLIDESPARSE"] = "1"
    os.environ.pop("USE_CUBLASLT", None)
    os.environ.pop("USE_CUSPARSELT", None)
    os.environ.pop("INNER_DTYPE_32", None)
    os.environ.pop("SPARSITY", None)
    
    EnvironmentChecker.clear_cache()
    # 清除 sparsity 缓存
    try:
        from slidesparse.core.config import clear_sparsity_cache
        clear_sparsity_cache()
    except ImportError:
        pass
    return saved


def set_env_for_test(use_cublaslt: bool = False, use_cusparselt: bool = False,
                     inner_32: bool = False, sparsity: str = None,
                     model_name: str = None) -> Dict[str, Optional[str]]:
    """
    设置环境变量为测试配置
    
    Args:
        use_cublaslt: 使用 cuBLASLt kernel
        use_cusparselt: 使用 cuSPARSELt kernel
        inner_32: 使用高精度累加（FP8→FP32, INT8→INT32）
        sparsity: 稀疏格式（仅 cuSPARSELt 时生效），默认 "2_8"
        model_name: 模型名称（用于加载 model-specific 的 tuned kernels）
                    默认使用 "Llama3.2-1B-FP8"
    
    Returns:
        保存的原环境变量，用于恢复
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_32": os.environ.get("INNER_DTYPE_32"),
        "SPARSITY": os.environ.get("SPARSITY"),
    }
    
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    if use_cublaslt:
        os.environ["USE_CUBLASLT"] = "1"
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
    elif use_cusparselt:
        os.environ["USE_CUSPARSELT"] = "1"
        os.environ.pop("USE_CUBLASLT", None)
        os.environ["SPARSITY"] = sparsity or "2_8"
    else:
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
    
    if inner_32:
        os.environ["INNER_DTYPE_32"] = "1"
    else:
        os.environ.pop("INNER_DTYPE_32", None)
    
    EnvironmentChecker.clear_cache()
    # 清除 sparsity 缓存
    try:
        from slidesparse.core.config import clear_sparsity_cache
        clear_sparsity_cache()
    except ImportError:
        pass
    
    # 初始化 SlideSparse（设置 model_name）
    # 只有在使用 cuBLASLt 或 cuSPARSELt 时才需要
    if use_cublaslt or use_cusparselt:
        try:
            from slidesparse.core import init_slidesparse
            # 使用提供的 model_name 或默认值
            effective_model_name = model_name or "Llama3.2-1B-FP8"
            init_slidesparse(effective_model_name)
        except ImportError:
            pass
    
    return saved


def restore_env(saved: Dict[str, Optional[str]]) -> None:
    """
    恢复环境变量
    
    Args:
        saved: 之前保存的环境变量
    """
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    
    EnvironmentChecker.clear_cache()
    # 清除 sparsity 缓存
    try:
        from slidesparse.core.config import clear_sparsity_cache
        clear_sparsity_cache()
    except ImportError:
        pass
