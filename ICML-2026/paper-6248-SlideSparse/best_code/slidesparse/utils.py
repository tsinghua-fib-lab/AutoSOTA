#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 文件名统一工具库

提供统一的硬件信息获取、文件命名和模块加载功能。

命名规范
========
所有生成的文件名遵循统一格式：
    {prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}

dtype 部分是可选的，支持三种情况：
1. 单个 dtype:   cublaslt_gemm_H100_cc90_FP8E4M3_py312_cu124_x86_64.so
2. 多个 dtype:   cublaslt_gemm_H100_cc90_FP8_INT8_py312_cu124_x86_64.so
3. 无 dtype:     cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so

示例：
    cublaslt_gemm_B200_cc100_py312_cu129_x86_64.so       # 支持多种类型的 GEMM
    dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py  # 特定类型
    alg_id_LUT_A100_cc80_INT8_py311_cu121_x86_64.json   # 特定类型

组件说明：
    - prefix:    用途前缀（cublaslt_gemm, cusparselt_gemm, dequant_bias_tuned 等）
    - GPU:       GPU 简称（H100, A100, B200, GB10 等）
    - CC:        Compute Capability（cc90, cc100, cc121 等）
    - dtype:     数据类型（可选，单个或多个：FP8E4M3, INT8, BF16, FP32 等）
    - PyVer:     Python 版本（py312, py311 等）
    - CUDAVer:   CUDA 版本（cu129, cu124 等）
    - Arch:      系统架构（x86_64, aarch64 等）

主要功能
========
1. HardwareInfo: 硬件信息单例类，缓存所有硬件信息
2. FileNameBuilder: 文件名构建器
3. FileFinder: 文件查找器，支持模糊匹配
4. ModuleLoader: 模块加载器，支持 .py 和 .so

使用示例
========
>>> from slidesparse.utils import hw_info, build_filename, find_file, load_module
>>>
>>> # 获取硬件信息
>>> print(hw_info.gpu_name)  # "H100"
>>> print(hw_info.cc_tag)    # "cc90"
>>>
>>> # 构建文件名（无 dtype，用于支持多类型的扩展）
>>> name = build_filename("cublaslt_gemm", ext=".so")
>>> # -> "cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so"
>>>
>>> # 构建文件名（带单个 dtype）
>>> name = build_filename("dequant_bias_tuned", dtype="BF16", ext=".py")
>>> # -> "dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py"
>>>
>>> # 构建文件名（带多个 dtype）
>>> name = build_filename("gemm_kernel", dtype=["FP8", "INT8"], ext=".so")
>>> # -> "gemm_kernel_H100_cc90_FP8_INT8_py312_cu124_x86_64.so"
>>>
>>> # 查找文件
>>> path = find_file("cublaslt_gemm", search_dir=build_dir)
>>>
>>> # 加载模块
>>> module = load_module("cublaslt_gemm", search_dir=build_dir)
"""

import base64
import ctypes
import ctypes.util
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import cached_property

# 延迟导入 torch
_torch = None


def _get_torch():
    """延迟导入 torch，避免在不需要时加载"""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch is required but not installed")
    return _torch


# =============================================================================
# 全局默认配置
# =============================================================================

# 默认 M 值列表（用于搜索/调优）
# 覆盖从 decode (小 batch) 到 prefill (大 batch) 的典型场景
DEFAULT_M_LIST = [16, 512, 1024, 4096, 8192, 16384]

# M-quick 模式固定列表（快速测试）
M_QUICK_LIST = [16, 128, 1024, 4096, 16384]


# =============================================================================
# 数据类型标准化
# =============================================================================

# 数据类型别名映射（输入 -> 标准名称）
DTYPE_ALIASES = {
    # FP8 变体
    "fp8": "FP8E4M3",
    "fp8e4m3": "FP8E4M3",
    "fp8_e4m3": "FP8E4M3",
    "FP8": "FP8E4M3",
    "FP8E4M3": "FP8E4M3",
    "e4m3": "FP8E4M3",
    "fp8e5m2": "FP8E5M2",
    "fp8_e5m2": "FP8E5M2",
    "FP8E5M2": "FP8E5M2",
    "e5m2": "FP8E5M2",
    # INT8
    "int8": "INT8",
    "INT8": "INT8",
    "i8": "INT8",
    # INT32
    "int32": "INT32",
    "INT32": "INT32",
    "i32": "INT32",
    # BF16
    "bf16": "BF16",
    "BF16": "BF16",
    "bfloat16": "BF16",
    # FP16
    "fp16": "FP16",
    "FP16": "FP16",
    "float16": "FP16",
    "half": "FP16",
    # FP32
    "fp32": "FP32",
    "FP32": "FP32",
    "float32": "FP32",
    "float": "FP32",
}


def normalize_dtype(dtype: str) -> str:
    """
    标准化数据类型名称
    
    Args:
        dtype: 输入的数据类型名称（大小写不敏感）
        
    Returns:
        标准化的数据类型名称
        
    Raises:
        ValueError: 未知的数据类型
        
    Examples:
        >>> normalize_dtype("fp8")
        'FP8E4M3'
        >>> normalize_dtype("int8")
        'INT8'
    """
    key = dtype.lower().replace("-", "_").replace(" ", "")
    if key in DTYPE_ALIASES:
        return DTYPE_ALIASES[key]
    # 尝试原始输入
    if dtype in DTYPE_ALIASES.values():
        return dtype
    raise ValueError(f"未知的数据类型: {dtype}. 支持的类型: {set(DTYPE_ALIASES.values())}")


# #############################################################################
#
#  PART 1: CUDA 编译、链接、库加载工具
#
#  本部分提供统一的 CUDA 扩展编译和运行时库加载功能。
#
#  规范流程：
#  =========
#  【编译时】
#   1. 优先指定系统库路径 (-L/usr/lib/x86_64-linux-gnu)
#   2. 然后链接库名 (-lcusparseLt 等)
#   3. 确保链接到系统安装的新版本库，而非 pip 包的旧版本
#
#  【运行时】
#   1. 设置环境变量 (CUSPARSELT_PATH 等) 指向系统库
#   2. 预加载系统库 (RTLD_GLOBAL 模式，确保符号全局可见)
#   3. 加载自定义 .so 文件
#
#  支持的编译方式：
#  ===============
#  - build_cuda_extension():       使用 torch.utils.cpp_extension.load (PyTorch 扩展)
#  - build_cuda_extension_direct(): 直接使用 nvcc 编译 (纯 C 库，用 ctypes 加载)
#
# #############################################################################


# =============================================================================
# 系统库路径配置
# =============================================================================

# 系统库搜索路径（优先级从高到低）
SYSTEM_LIB_PATHS = {
    "x86_64": "/usr/lib/x86_64-linux-gnu",
    "aarch64": "/usr/lib/aarch64-linux-gnu",
    "default": "/usr/local/cuda/lib64",
}

def get_system_lib_path() -> str:
    """获取当前架构的系统库路径"""
    import platform
    arch = platform.machine()
    return SYSTEM_LIB_PATHS.get(arch, SYSTEM_LIB_PATHS["default"])


# =============================================================================
# NVCC 架构标志
# =============================================================================

# 支持的 GPU 架构列表
SUPPORTED_ARCHITECTURES = [
    ("80", "sm_80"),   # Ampere (A100, A10, A30)
    ("86", "sm_86"),   # Ampere (RTX 30xx)
    ("89", "sm_89"),   # Ada Lovelace (RTX 40xx)
    ("90", "sm_90"),   # Hopper (H100, H200)
    ("100", "sm_100"), # Blackwell (B100, B200)
    ("120", "sm_120"), # Blackwell (RTX 50xx)
    ("121", "sm_121"), # Blackwell (GB10)
]


def get_nvcc_arch_flags(
    min_compute: int = 80,
    max_compute: int = 121,
) -> List[str]:
    """
    生成 nvcc 架构编译选项
    
    支持从 SM 80 (Ampere) 到 SM 121 (Blackwell)
    
    Args:
        min_compute: 最小支持的 compute capability
        max_compute: 最大支持的 compute capability
        
    Returns:
        nvcc -gencode 标志列表
    """
    flags = []
    for compute, sm in SUPPORTED_ARCHITECTURES:
        cc = int(compute)
        if min_compute <= cc <= max_compute:
            flags.append(f"-gencode=arch=compute_{compute},code={sm}")
    return flags


def get_current_arch_flag() -> str:
    """
    获取当前 GPU 架构的 nvcc 编译标志
    
    Returns:
        单个 -gencode 标志，针对当前 GPU
    """
    torch = _get_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    compute = f"{prop.major}{prop.minor}"
    return f"-gencode=arch=compute_{compute},code=sm_{compute}"


# =============================================================================
# 链接库配置
# =============================================================================

# 支持的后端类型
SUPPORTED_BACKENDS = ["cublaslt", "cusparselt"]


def get_backend_ldflags(backend: str, with_lib_path: bool = True) -> List[str]:
    """
    获取后端所需的链接标志
    
    Args:
        backend: 后端名称 ("cublaslt" 或 "cusparselt")
        with_lib_path: 是否包含 -L 库路径（编译时需要，推荐 True）
        
    Returns:
        链接标志列表
    """
    lib_path = get_system_lib_path()
    
    if backend.lower() == "cublaslt":
        flags = ["-lcublasLt", "-lcublas", "-lcuda"]
    elif backend.lower() == "cusparselt":
        flags = ["-lcusparseLt", "-lcusparse", "-lcuda"]
    else:
        raise ValueError(f"未知的后端: {backend}，支持: {SUPPORTED_BACKENDS}")
    
    if with_lib_path:
        return [f"-L{lib_path}"] + flags
    return flags


# 兼容性别名（后端链接库配置）
BACKEND_LDFLAGS = {
    "cublaslt": get_backend_ldflags("cublaslt", with_lib_path=True),
    "cusparselt": get_backend_ldflags("cusparselt", with_lib_path=True),
}

# 简化版链接库（不含 -L 路径，用于 torch.utils.cpp_extension）
CUBLASLT_LDFLAGS = get_backend_ldflags("cublaslt", with_lib_path=True)
CUSPARSELT_LDFLAGS = get_backend_ldflags("cusparselt", with_lib_path=True)


# =============================================================================
# 运行时库加载
# =============================================================================

# 库加载状态
_CUBLASLT_LOADED = False
_CUSPARSELT_LOADED = False


def ensure_cublaslt_loaded() -> None:
    """
    预加载系统 cuBLASLt 库，避免符号冲突。
    
    必须在加载自定义 .so 之前完成。使用 RTLD_GLOBAL 确保符号全局可见。
    
    环境变量:
        CUBLASLT_PATH: 指定 libcublasLt.so 的完整路径（优先级最高）
        
    Raises:
        OSError: 无法找到兼容的 libcublasLt
    """
    global _CUBLASLT_LOADED
    if _CUBLASLT_LOADED:
        return

    # 构建搜索路径（优先级从高到低）
    preferred_paths = []
    
    # 1. 环境变量优先
    env_path = os.environ.get("CUBLASLT_PATH")
    if env_path:
        preferred_paths.append(env_path)

    # 2. 系统库路径
    preferred_paths.extend([
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so",
        "/usr/lib/aarch64-linux-gnu/libcublasLt.so",
        "/usr/local/cuda/lib64/libcublasLt.so",
    ])
    
    # 3. ctypes 默认搜索
    found = ctypes.util.find_library("cublasLt")
    if found:
        preferred_paths.append(found)

    # 尝试加载
    for path in dict.fromkeys(preferred_paths):  # 去重但保持优先级
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            getattr(lib, "cublasLtCreate")  # 验证库可用
            _CUBLASLT_LOADED = True
            return
        except (OSError, AttributeError):
            continue

    raise OSError(
        "无法找到兼容的 libcublasLt。\n"
        "请设置 CUBLASLT_PATH 环境变量，或确保 CUDA 已正确安装。"
    )


def ensure_cusparselt_loaded() -> None:
    """
    预加载系统 cuSPARSELt 库 (0.8.1+)，避免与 PyTorch pip 包 (0.7.x) 冲突。
    
    必须在加载自定义 .so 之前完成。使用 RTLD_GLOBAL 确保符号全局可见。
    
    环境变量:
        CUSPARSELT_PATH: 指定 libcusparseLt.so.0 的完整路径（优先级最高）
        
    Raises:
        OSError: 无法找到兼容的 libcusparseLt (需要 0.8+)
    """
    global _CUSPARSELT_LOADED
    if _CUSPARSELT_LOADED:
        return

    # 构建搜索路径（优先级从高到低）
    preferred_paths = []
    
    # 1. 环境变量优先
    env_path = os.environ.get("CUSPARSELT_PATH")
    if env_path:
        preferred_paths.append(env_path)

    # 2. 系统库路径（优先新版本目录）
    preferred_paths.extend([
        # x86_64 系统库
        "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
        "/usr/lib/x86_64-linux-gnu/libcusparseLt/12/libcusparseLt.so.0",
        "/usr/lib/x86_64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
        # aarch64 系统库
        "/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0",
        "/usr/lib/aarch64-linux-gnu/libcusparseLt/12/libcusparseLt.so.0",
        "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
        # CUDA 默认路径
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
    ])
    
    # 3. ctypes 默认搜索（可能找到 pip 包的旧版本，优先级最低）
    found = ctypes.util.find_library("cusparseLt")
    if found:
        preferred_paths.append(found)

    # 尝试加载
    for path in dict.fromkeys(preferred_paths):  # 去重但保持优先级
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            # 验证是 0.8+ 版本（此 API 在 0.7 中不存在）
            getattr(lib, "cusparseLtMatmulAlgSelectionDestroy")
            _CUSPARSELT_LOADED = True
            return
        except (OSError, AttributeError):
            continue

    raise OSError(
        "无法找到兼容的 libcusparseLt (需要 0.8+)。\n"
        "系统安装: apt install libcusparselt0 libcusparselt-dev\n"
        "或设置 CUSPARSELT_PATH 环境变量指向系统库路径。\n"
        "注意: PyTorch pip 包自带的 0.7.x 版本不兼容。"
    )


# 后端对应的库加载函数
BACKEND_LOADERS = {
    "cublaslt": ensure_cublaslt_loaded,
    "cusparselt": ensure_cusparselt_loaded,
}


# =============================================================================
# 编译辅助函数
# =============================================================================

# 默认编译选项
DEFAULT_CFLAGS = ['-O3', '-std=c++17']

DEFAULT_CUDA_CFLAGS = [
    '-O3',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
]


def should_rebuild(so_path: Path, source_paths: List[Path]) -> bool:
    """
    判断是否需要重新编译
    
    Args:
        so_path: .so 文件路径
        source_paths: 源文件路径列表
        
    Returns:
        如果 .so 不存在或比任一源文件旧，返回 True
    """
    if not so_path.exists():
        return True
    
    so_mtime = so_path.stat().st_mtime
    for src in source_paths:
        if src.exists() and src.stat().st_mtime > so_mtime:
            return True
    return False


def clean_build_artifacts(build_dir: Path, keep_extensions: Optional[List[str]] = None):
    """
    清理编译中间文件
    
    Args:
        build_dir: 构建目录
        keep_extensions: 要保留的文件扩展名列表（默认 ['.so', '.py']）
    """
    if keep_extensions is None:
        keep_extensions = ['.so', '.py']
    
    if not build_dir.exists():
        return
    
    for item in build_dir.iterdir():
        if item.suffix in keep_extensions:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


# =============================================================================
# PyTorch 扩展编译 (torch.utils.cpp_extension)
# =============================================================================

def build_cuda_extension(
    name: str,
    source_file: Path,
    build_dir: Path,
    *,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
    clean_after_build: bool = True,
) -> Path:
    """
    使用 torch.utils.cpp_extension.load 编译 CUDA 扩展
    
    生成的 .so 文件可以作为 Python 模块导入，支持 pybind11 绑定。
    适用于需要与 PyTorch Tensor 交互的 CUDA 代码。
    
    Args:
        name: 扩展名称（不含 .so 后缀）
        source_file: 源文件路径 (.cu)
        build_dir: 构建目录
        extra_cflags: 额外的 C++ 编译标志
        extra_cuda_cflags: 额外的 CUDA 编译标志
        extra_ldflags: 额外的链接标志（如 ["-lcublasLt"]）
        extra_include_paths: 额外的头文件搜索路径
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        clean_after_build: 编译后是否清理中间文件
        
    Returns:
        编译生成的 .so 文件路径
    """
    from torch.utils.cpp_extension import load
    
    source_file = Path(source_file)
    build_dir = Path(build_dir)
    
    if not source_file.exists():
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找已存在的 .so
    so_pattern = f"{name}*.so"
    existing_sos = list(build_dir.glob(so_pattern))
    
    if existing_sos and not force:
        so_path = existing_sos[0]
        if not should_rebuild(so_path, [source_file]):
            if verbose:
                print(f"✓ Using existing: {so_path.name}")
            return so_path
        elif verbose:
            print(f"⚠ Source changed, rebuilding...")
    
    if verbose:
        print(f"🔨 Building {name}...")
    
    # CUDA 路径
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    # 合并编译选项
    cflags = DEFAULT_CFLAGS + (extra_cflags or [])
    cuda_cflags = DEFAULT_CUDA_CFLAGS + get_nvcc_arch_flags() + (extra_cuda_cflags or [])
    ldflags = extra_ldflags or []
    include_paths = [os.path.join(cuda_home, 'include')] + (extra_include_paths or [])
    
    # 编译
    try:
        load(
            name=name,
            sources=[str(source_file)],
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=ldflags,
            extra_include_paths=include_paths,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        raise RuntimeError(f"编译失败: {e}") from e
    
    # 查找生成的 .so
    new_sos = list(build_dir.glob(so_pattern))
    if not new_sos:
        raise RuntimeError(f"编译完成但未找到 .so 文件: {so_pattern}")
    
    so_path = new_sos[0]
    
    if verbose:
        print(f"✓ Built: {so_path.name}")
    
    if clean_after_build:
        if verbose:
            print(f"🧹 Cleaning build artifacts...")
        clean_build_artifacts(build_dir)
    
    return so_path


# =============================================================================
# 直接 NVCC 编译 (纯 C 库，用 ctypes 加载)
# =============================================================================

def build_cuda_extension_direct(
    name: str,
    source_file: Path,
    build_dir: Path,
    *,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    直接使用 nvcc 编译 CUDA 扩展（不依赖 PyTorch）
    
    生成的 .so 是纯 C 库，通过 ctypes.CDLL 加载。
    适用于不依赖 PyTorch 的纯 CUDA 代码，编译速度快。
    
    Args:
        name: 扩展名称（不含 .so 后缀）
        source_file: 源文件路径 (.cu)
        build_dir: 构建目录
        extra_cuda_cflags: 额外的 CUDA 编译标志
        extra_ldflags: 额外的链接标志（如 ["-L/usr/lib", "-lcusparseLt"]）
        extra_include_paths: 额外的头文件搜索路径
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        
    Returns:
        编译生成的 .so 文件路径
    """
    source_file = Path(source_file)
    build_dir = Path(build_dir)
    
    if not source_file.exists():
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件
    so_path = build_dir / f"{name}.so"
    
    # 检查是否需要重新编译
    if so_path.exists() and not force:
        if not should_rebuild(so_path, [source_file]):
            if verbose:
                print(f"✓ Using existing: {so_path.name}")
            return so_path
        elif verbose:
            print(f"⚠ Source changed, rebuilding...")
    
    if verbose:
        print(f"🔨 Building {name}...")
    
    # CUDA 路径
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    
    # 构建编译命令
    cmd = [nvcc]
    cmd.extend(['-std=c++17', '-O3', '-Xcompiler', '-fPIC', '--shared'])
    cmd.extend(get_nvcc_arch_flags())
    
    if extra_cuda_cflags:
        cmd.extend(extra_cuda_cflags)
    
    # 头文件路径
    cmd.extend(['-I', os.path.join(cuda_home, 'include')])
    if extra_include_paths:
        for inc in extra_include_paths:
            cmd.extend(['-I', inc])
    
    # 源文件
    cmd.append(str(source_file))
    
    # 链接标志
    if extra_ldflags:
        cmd.extend(extra_ldflags)
    
    # 输出
    cmd.extend(['-o', str(so_path)])
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    # 执行编译
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        raise RuntimeError(f"编译失败:\n{error_msg}")
    
    if not so_path.exists():
        raise RuntimeError(f"编译完成但未找到 .so 文件: {so_path}")
    
    if verbose:
        print(f"✓ Built: {so_path.name}")
    
    return so_path


# =============================================================================
# 高级加载接口 (自动编译 + 加载)
# =============================================================================

def load_cuda_extension(
    script_type: str,
    backend: str,
    source_file: "Path",
    build_dir: "Optional[Path]" = None,
    *,
    verbose: bool = True,
    force_compile: bool = False,
) -> object:
    """
    加载或编译 PyTorch CUDA 扩展（高级接口）
    
    自动处理：
    1. 预加载系统 CUDA 库（避免版本冲突）
    2. 检查已有 .so 是否可用
    3. 必要时编译新的 .so
    4. 加载并返回模块
    
    命名规范:
        {script_type}_{backend}_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.so
        例如: alg_search_cublaslt_H100_cc90_py312_cu129_x86_64.so
    
    Args:
        script_type: 脚本类型（如 "alg_search", "layout_search"）
        backend: 后端类型（"cublaslt" 或 "cusparselt"）
        source_file: CUDA 源文件路径 (.cu)
        build_dir: 构建目录，默认为 source_file 同级的 build/
        verbose: 是否显示进度信息
        force_compile: 是否强制重新编译
    
    Returns:
        编译好的扩展模块（可调用其导出的函数）
    """
    torch = _get_torch()
    from torch.utils.cpp_extension import load
    
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"无效的后端类型: {backend}，支持: {SUPPORTED_BACKENDS}")
    
    source_file = Path(source_file)
    if not source_file.exists():
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    # Step 1: 预加载系统库
    if verbose:
        lib_name = "cuBLASLt" if backend == "cublaslt" else "cuSPARSELt"
        print(f"[1/4] 加载 {lib_name} 库...", end=" ", flush=True)
    
    BACKEND_LOADERS[backend]()
    
    if verbose:
        print("✓", flush=True)
    
    # 获取硬件信息
    hw = hw_info
    
    # 构建扩展名称
    ext_name = build_stem(f"{script_type}_{backend}")
    so_pattern = f"{ext_name}*.so"
    
    # 确定构建目录
    if build_dir is None:
        build_dir = source_file.parent / "build"
    else:
        build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: 检查已有的 .so
    existing_so = list(build_dir.glob(so_pattern))
    need_compile = force_compile
    
    if not need_compile:
        if not existing_so:
            need_compile = True
        else:
            need_compile = source_file.stat().st_mtime > existing_so[0].stat().st_mtime
    
    if not need_compile and existing_so:
        if verbose:
            print(f"[2/4] 加载 {hw.gpu_name} 扩展...", end=" ", flush=True)
        
        spec = importlib.util.spec_from_file_location(ext_name, str(existing_so[0]))
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        
        if verbose:
            print(f"✓ ({existing_so[0].name})", flush=True)
        return ext
    else:
        if verbose:
            reason = "强制" if force_compile else ("首次" if not existing_so else "源文件已更新")
            print(f"[2/4] 编译 {hw.gpu_name} 扩展（{reason}）...", end=" ", flush=True)
        
        ext = load(
            name=ext_name,
            sources=[str(source_file)],
            extra_cuda_cflags=["-O3", f"-arch={hw.sm_code}"],
            extra_ldflags=BACKEND_LDFLAGS[backend],
            verbose=False,
            build_directory=str(build_dir),
            with_cuda=True,
        )
        
        # 清理中间文件
        for pattern in [".ninja_deps", ".ninja_log", "build.ninja", "*.o"]:
            for f in build_dir.glob(pattern):
                f.unlink(missing_ok=True)
        
        if verbose:
            print("✓", flush=True)
        return ext


# #############################################################################
#
#  PART 2: 硬件信息
#
#  本部分提供统一的硬件信息获取功能。
#
#  主要内容：
#  =========
#  - HardwareInfo: 硬件信息单例类，缓存所有硬件相关信息
#  - hw_info: 全局单例实例
#  - 便捷函数: get_gpu_name, get_gpu_cc, get_sm_code 等
#
#  使用示例：
#  =========
#  >>> from slidesparse.utils import hw_info
#  >>> print(hw_info.gpu_name)     # "H100"
#  >>> print(hw_info.cc_tag)       # "cc90"
#  >>> print(hw_info.supports_fp8) # True
#
# #############################################################################


# =============================================================================
# 硬件信息类
# =============================================================================

@dataclass
class HardwareInfo:
    """
    硬件信息单例类
    
    缓存所有硬件相关信息，避免重复查询。
    所有属性使用 cached_property 实现懒加载。
    
    Attributes:
        gpu_name: GPU 简称（H100, A100 等）
        gpu_full_name: GPU 完整名称
        cc_major: Compute Capability 主版本
        cc_minor: Compute Capability 次版本
        cc_tag: CC 标签（cc90, cc100 等）
        python_tag: Python 版本标签（py312 等）
        cuda_tag: CUDA 版本标签（cu129 等）
        arch_tag: 系统架构标签（x86_64 等）
    """
    
    _instance: Optional['HardwareInfo'] = field(default=None, repr=False, init=False)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # -------------------------------------------------------------------------
    # GPU 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def gpu_full_name(self) -> str:
        """GPU 完整名称"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.name
        # 备选：nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"
    
    @cached_property
    def gpu_name(self) -> str:
        """
        GPU 简称（H100, A100, B200, RTX5080 等）
        
        处理常见格式:
        - "NVIDIA A100-SXM4-40GB" -> "A100"
        - "NVIDIA H100 PCIe" -> "H100"
        - "NVIDIA GeForce RTX 5080" -> "RTX5080"
        - "NVIDIA GeForce RTX 4090" -> "RTX4090"
        - "NVIDIA GeForce GTX 1080 Ti" -> "GTX1080Ti"
        - "NVIDIA TITAN RTX" -> "TitanRTX"
        """
        full_name = self.gpu_full_name
        if full_name == "unknown":
            return "unknown"
        
        # 移除 "NVIDIA " 前缀
        name = full_name
        if name.startswith("NVIDIA "):
            name = name[7:]
        
        # 处理 GeForce RTX/GTX 系列: "GeForce RTX 5080" -> "RTX5080"
        if name.startswith("GeForce "):
            name = name[8:]  # 移除 "GeForce "
            # 现在 name 可能是 "RTX 5080" 或 "GTX 1080 Ti"
            # 提取 RTX/GTX 前缀和型号
            parts = name.split()
            if len(parts) >= 2 and parts[0] in ("RTX", "GTX"):
                prefix = parts[0]  # RTX 或 GTX
                model = "".join(parts[1:])  # 5080 或 1080Ti
                return f"{prefix}{model}"
            # 其他 GeForce 情况
            return "".join(parts)
        
        # 处理 TITAN 系列: "TITAN RTX" -> "TitanRTX"
        if name.startswith("TITAN "):
            return "Titan" + name[6:].replace(" ", "")
        
        # 数据中心卡: "A100-SXM4-40GB" -> "A100", "H100 PCIe" -> "H100"
        # 提取第一个空格或连字符之前的部分
        for sep in [" ", "-"]:
            end_pos = name.find(sep)
            if end_pos != -1:
                name = name[:end_pos]
                break
        
        # 清理特殊字符
        if not name:
            name = full_name
            for c in [" ", "-", "/"]:
                name = name.replace(c, "_")
        
        return name
    
    @cached_property
    def cc_major(self) -> int:
        """Compute Capability 主版本"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.major
        return 0
    
    @cached_property
    def cc_minor(self) -> int:
        """Compute Capability 次版本"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.minor
        return 0
    
    @cached_property
    def cc_tag(self) -> str:
        """CC 标签（cc90, cc100, cc121 等）"""
        return f"cc{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def sm_code(self) -> str:
        """SM 代码（sm_90, sm_100 等）"""
        return f"sm_{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def gpu_memory_gb(self) -> float:
        """GPU 显存大小 (GB)"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.total_memory / (1024 ** 3)
        return 0.0
    
    # -------------------------------------------------------------------------
    # Python 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def python_version(self) -> Tuple[int, int]:
        """Python 版本 (major, minor)"""
        return (sys.version_info.major, sys.version_info.minor)
    
    @cached_property
    def python_tag(self) -> str:
        """Python 版本标签（py312, py311 等）"""
        return f"py{self.python_version[0]}{self.python_version[1]}"
    
    # -------------------------------------------------------------------------
    # CUDA 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def cuda_runtime_version(self) -> str:
        """CUDA Runtime 版本（PyTorch 编译时使用的版本）"""
        torch = _get_torch()
        try:
            return torch.version.cuda or "unknown"
        except Exception:
            return "unknown"
    
    @cached_property
    def cuda_driver_version(self) -> str:
        """CUDA Driver 版本（nvidia-smi 显示的版本）"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "CUDA Version" in line:
                        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "unknown"
    
    @cached_property
    def cuda_tag(self) -> str:
        """
        CUDA 版本标签（cu129, cu124 等）
        
        优先使用 Runtime 版本，因为这是实际编译时使用的版本。
        """
        version = self.cuda_runtime_version
        if version == "unknown":
            version = self.cuda_driver_version
        if version == "unknown":
            return "cu000"
        # "12.9" -> "cu129", "12.4" -> "cu124"
        parts = version.split(".")
        if len(parts) >= 2:
            major = parts[0]
            minor = parts[1].split(".")[0]  # 处理 "12.4.1" 这种情况
            return f"cu{major}{minor}"
        return f"cu{version.replace('.', '')}"
    
    # -------------------------------------------------------------------------
    # 系统架构
    # -------------------------------------------------------------------------
    
    @cached_property
    def arch_raw(self) -> str:
        """原始系统架构"""
        return platform.machine()
    
    @cached_property
    def arch_tag(self) -> str:
        """系统架构标签（x86_64, aarch64 等）"""
        machine = self.arch_raw
        if machine in ("x86_64", "AMD64"):
            return "x86_64"
        elif machine in ("aarch64", "arm64"):
            return "aarch64"
        return machine.lower()
    
    # -------------------------------------------------------------------------
    # 驱动信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def driver_version(self) -> str:
        """NVIDIA 驱动版本"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"
    
    # -------------------------------------------------------------------------
    # PyTorch 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def pytorch_version(self) -> str:
        """PyTorch 版本"""
        torch = _get_torch()
        return torch.__version__
    
    # -------------------------------------------------------------------------
    # 架构检测
    # -------------------------------------------------------------------------
    
    # 架构名称映射
    ARCH_INFO = {
        7: ("Volta", "volta"),         # V100 等
        8: ("Ampere", "ampere"),       # A100, A10, A30 等
        9: ("Hopper", "hopper"),       # H100, H200 等
        10: ("Blackwell", "blackwell"), # B100, B200 等
        12: ("Blackwell", "blackwell"), # GB10 等 (CC 12.x 也是 Blackwell 家族)
    }
    
    @cached_property
    def arch_name(self) -> str:
        """架构名称（Ampere, Hopper, Blackwell 等）"""
        if self.cc_major in self.ARCH_INFO:
            return self.ARCH_INFO[self.cc_major][0]
        return f"SM{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def arch_suffix(self) -> str:
        """架构后缀（ampere, hopper, blackwell 等）"""
        if self.cc_major in self.ARCH_INFO:
            return self.ARCH_INFO[self.cc_major][1]
        return f"sm{self.cc_major}{self.cc_minor}"
    
    # -------------------------------------------------------------------------
    # 功能检测
    # -------------------------------------------------------------------------
    
    @cached_property
    def supports_fp8(self) -> bool:
        """是否支持原生 FP8（CC >= 8.9，Ada/Hopper+）"""
        return (self.cc_major, self.cc_minor) >= (8, 9)
    
    @cached_property
    def supports_int8(self) -> bool:
        """是否支持原生 INT8（CC >= 8.0，Ampere+）"""
        return self.cc_major >= 8
    
    # -------------------------------------------------------------------------
    # vLLM CUTLASS Kernel 支持检测
    # -------------------------------------------------------------------------
    # 
    # vLLM 预编译的 CUTLASS kernel 有版本限制：
    # 
    # INT8 CUTLASS:
    #   - 支持范围: sm_75 ~ sm_90 (Turing ~ Hopper)
    #   - sm_100+ (Blackwell) 不支持，vLLM 报错 "Int8 not supported on SM1xx"
    #
    # FP8 CUTLASS:
    #   - 支持范围: sm_89 ~ sm_120 (Ada ~ Blackwell RTX 50xx)
    #   - sm_121+ (GB10 等) 不支持
    #   - sm_80~88 会 fallback 到 Marlin W8A16 kernel（不是真正的 FP8 计算）
    #
    # 注意：这些限制是 vLLM 预编译 kernel 的问题，不是硬件限制。
    # SlideSparse 的 cuBLASLt/cuSPARSELt 路径不受这些限制。
    # -------------------------------------------------------------------------
    
    @cached_property
    def supports_vllm_cutlass_int8(self) -> Tuple[bool, str]:
        """
        检查 vLLM CUTLASS INT8 kernel 是否支持当前 GPU
        
        vLLM 的 CUTLASS INT8 kernel 只支持 sm_75 ~ sm_90 (Turing ~ Hopper)。
        sm_100+ (Blackwell) 上 vLLM 会报错 "Int8 not supported on SM1xx"。
        
        Returns:
            (supported, reason)
        """
        cc = (self.cc_major, self.cc_minor)
        
        if cc < (7, 5):
            return False, f"sm_{self.cc_major}{self.cc_minor} < sm_75"
        
        if cc >= (10, 0):
            return False, f"sm_{self.cc_major}{self.cc_minor} >= sm_100"
        
        return True, f"sm_{self.cc_major}{self.cc_minor} 在支持范围 [sm_75, sm_90]"
    
    @cached_property
    def supports_vllm_cutlass_fp8(self) -> Tuple[bool, str]:
        """
        检查 vLLM CUTLASS FP8 kernel 是否支持当前 GPU
        
        vLLM 的 CUTLASS FP8 kernel 只支持 sm_89 ~ sm_120 (Ada ~ Blackwell RTX 50xx)。
        - sm_80~88: 会 fallback 到 Marlin W8A16 kernel（不是真正的 FP8 计算）
        - sm_121+: vLLM CUTLASS kernel 未编译支持
        
        Returns:
            (supported, reason)
        """
        cc = (self.cc_major, self.cc_minor)
        
        if cc < (8, 9):
            return False, f"sm_{self.cc_major}{self.cc_minor} < sm_89 (会 fallback 到 Marlin W8A16)"
        
        if cc > (12, 0):
            return False, f"sm_{self.cc_major}{self.cc_minor} > sm_120 (vLLM CUTLASS FP8 未编译)"
        
        return True, f"sm_{self.cc_major}{self.cc_minor} 在支持范围 [sm_89, sm_120]"
    
    @cached_property
    def vllm_cutlass_int8_supported(self) -> bool:
        """vLLM CUTLASS INT8 是否支持（便捷属性）"""
        return self.supports_vllm_cutlass_int8[0]
    
    @cached_property
    def vllm_cutlass_fp8_supported(self) -> bool:
        """vLLM CUTLASS FP8 是否支持（便捷属性）"""
        return self.supports_vllm_cutlass_fp8[0]
    
    @cached_property
    def triton_supported(self) -> Tuple[bool, str]:
        """
        检查 Triton 是否支持当前架构
        
        Returns:
            (supported, reason)
        """
        # 已知不被支持的架构
        UNSUPPORTED = {
            (12, 1): "GB10 (sm_121a) is not yet supported by Triton/ptxas",
        }
        
        if (self.cc_major, self.cc_minor) in UNSUPPORTED:
            return False, UNSUPPORTED[(self.cc_major, self.cc_minor)]
        
        return True, "Architecture is supported"
    
    @cached_property
    def needs_eager_mode(self) -> bool:
        """是否需要使用 eager mode（禁用 torch.compile）"""
        return not self.triton_supported[0]
    
    # -------------------------------------------------------------------------
    # cuBLASLt / cuSPARSELt 支持检测
    # -------------------------------------------------------------------------
    
    @cached_property
    def supports_cublaslt(self) -> Tuple[bool, str]:
        """
        检查 cuBLASLt 是否支持当前 GPU
        
        cuBLASLt 支持 sm_70+ (Volta 及更新架构)。
        
        Returns:
            (supported, reason)
        """
        cc = (self.cc_major, self.cc_minor)
        
        if cc < (7, 0):
            return False, f"sm_{self.cc_major}{self.cc_minor} < sm_70 (cuBLASLt 需要 Volta+)"
        
        return True, f"sm_{self.cc_major}{self.cc_minor} >= sm_70"
    
    @cached_property
    def cublaslt_supported(self) -> bool:
        """cuBLASLt 是否支持（便捷属性）"""
        return self.supports_cublaslt[0]
    
    @cached_property
    def supports_cusparselt(self) -> Tuple[bool, str]:
        """
        检查 cuSPARSELt 是否支持当前 GPU
        
        cuSPARSELt 支持 sm_80+ (Ampere 及更新架构)。
        
        Returns:
            (supported, reason)
        """
        cc = (self.cc_major, self.cc_minor)
        
        if cc < (8, 0):
            return False, f"sm_{self.cc_major}{self.cc_minor} < sm_80 (cuSPARSELt 需要 Ampere+)"
        
        return True, f"sm_{self.cc_major}{self.cc_minor} >= sm_80"
    
    @cached_property
    def cusparselt_supported(self) -> bool:
        """cuSPARSELt 是否支持（便捷属性）"""
        return self.supports_cusparselt[0]
    
    # -------------------------------------------------------------------------
    # 汇总信息
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "gpu": {
                "name": self.gpu_name,
                "full_name": self.gpu_full_name,
                "memory_gb": round(self.gpu_memory_gb, 1),
            },
            "compute_capability": {
                "major": self.cc_major,
                "minor": self.cc_minor,
                "tag": self.cc_tag,
                "sm_code": self.sm_code,
            },
            "cuda": {
                "runtime_version": self.cuda_runtime_version,
                "driver_version": self.cuda_driver_version,
                "tag": self.cuda_tag,
            },
            "python": {
                "version": f"{self.python_version[0]}.{self.python_version[1]}",
                "tag": self.python_tag,
            },
            "system": {
                "arch": self.arch_tag,
                "driver_version": self.driver_version,
            },
            "architecture": {
                "name": self.arch_name,
                "suffix": self.arch_suffix,
            },
            "capabilities": {
                "supports_fp8": self.supports_fp8,
                "supports_int8": self.supports_int8,
                "triton_supported": self.triton_supported[0],
                "vllm_cutlass_int8": self.vllm_cutlass_int8_supported,
                "vllm_cutlass_fp8": self.vllm_cutlass_fp8_supported,
            },
            "pytorch_version": self.pytorch_version,
        }
    
    def print_info(self):
        """打印硬件信息"""
        print("=" * 60)
        print("SlideSparse Hardware Info")
        print("=" * 60)
        print(f"GPU:           {self.gpu_full_name}")
        print(f"GPU (short):   {self.gpu_name}")
        print(f"Memory:        {self.gpu_memory_gb:.1f} GB")
        print(f"CC:            {self.cc_tag} ({self.sm_code})")
        print(f"Architecture:  {self.arch_name}")
        print(f"Python:        {self.python_tag}")
        print(f"CUDA Runtime:  {self.cuda_runtime_version}")
        print(f"CUDA Driver:   {self.cuda_driver_version}")
        print(f"CUDA Tag:      {self.cuda_tag}")
        print(f"System Arch:   {self.arch_tag}")
        print(f"Driver:        {self.driver_version}")
        print(f"PyTorch:       {self.pytorch_version}")
        print("-" * 60)
        print(f"FP8 Support:   {self.supports_fp8}")
        print(f"INT8 Support:  {self.supports_int8}")
        print(f"Triton:        {'✓' if self.triton_supported[0] else '✗ ' + self.triton_supported[1]}")
        print("=" * 60)


# 全局单例
hw_info = HardwareInfo()


# 便捷函数（hw_info 属性的快捷访问）
def get_gpu_name() -> str:
    """获取 GPU 简称"""
    return hw_info.gpu_name


def get_gpu_cc() -> str:
    """获取 CC 标签"""
    return hw_info.cc_tag


def get_python_version_tag() -> str:
    """获取 Python 版本标签"""
    return hw_info.python_tag


def get_cuda_ver() -> str:
    """获取 CUDA 版本标签"""
    return hw_info.cuda_tag


def get_arch_tag() -> str:
    """获取系统架构标签"""
    return hw_info.arch_tag


def get_sm_code() -> str:
    """获取 SM 代码"""
    return hw_info.sm_code


def print_system_info():
    """打印系统信息"""
    hw_info.print_info()


# #############################################################################
#
#  PART 3: 文件名与 IO
#
#  本部分提供统一的文件命名、查找、保存和模块加载功能。
#
#  命名规范：
#  =========
#  所有生成的文件名遵循统一格式：
#      {prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}
#
#  主要功能：
#  =========
#  - build_filename:  构建标准化文件名
#  - find_file:       查找匹配的文件
#  - load_module:     加载 Python 模块 (.py/.so)
#  - save_json/csv:   保存数据文件
#  - ensure_result_dir: 创建结果目录
#
# #############################################################################


# =============================================================================
# 文件名构建
# =============================================================================

def build_filename(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    ext: str = "",
    *,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> str:
    """
    构建标准化文件名
    
    格式: {prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}
    
    dtype 部分是可选的，支持三种情况：
    - None: 不包含 dtype，用于支持多种类型的扩展
    - str: 单个 dtype
    - List[str]: 多个 dtype，按顺序连接
    
    Args:
        prefix: 用途前缀（cublaslt_gemm, cusparselt_gemm, dequant_bias_tuned 等）
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        ext: 文件扩展名（.so, .py, .json 等），不包含点时自动添加
        gpu_name: GPU 名称，默认自动检测
        cc_tag: CC 标签，默认自动检测
        python_tag: Python 版本标签，默认自动检测
        cuda_tag: CUDA 版本标签，默认自动检测
        arch_tag: 系统架构标签，默认自动检测
        
    Returns:
        标准化的文件名
        
    Examples:
        # 无 dtype（支持多种类型的 GEMM 扩展）
        >>> build_filename("cublaslt_gemm", ext=".so")
        'cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so'
        
        # 单个 dtype
        >>> build_filename("dequant_bias_tuned", dtype="BF16", ext=".py")
        'dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py'
        
        # 多个 dtype
        >>> build_filename("gemm_kernel", dtype=["FP8", "INT8"], ext=".so")
        'gemm_kernel_H100_cc90_FP8_INT8_py312_cu124_x86_64.so'
    """
    # 使用提供的值或从硬件信息获取
    _gpu = gpu_name or hw_info.gpu_name
    _cc = cc_tag or hw_info.cc_tag
    _py = python_tag or hw_info.python_tag
    _cuda = cuda_tag or hw_info.cuda_tag
    _arch = arch_tag or hw_info.arch_tag
    
    # 构建组件列表
    components = [prefix, _gpu, _cc]
    
    # 添加数据类型（如果提供）
    if dtype:
        if isinstance(dtype, str):
            # 单个 dtype
            components.append(normalize_dtype(dtype))
        elif isinstance(dtype, (list, tuple)):
            # 多个 dtype，逐个标准化后添加
            for d in dtype:
                components.append(normalize_dtype(d))
    
    # 添加其余组件
    components.extend([_py, _cuda, _arch])
    
    # 连接组件
    name = "_".join(components)
    
    # 处理扩展名
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        name += ext
    
    return name


def build_stem(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> str:
    """
    构建文件名主干（不含扩展名）
    
    等同于 build_filename(..., ext="")
    """
    return build_filename(prefix, dtype=dtype, ext="", **kwargs)


def build_dir_name(
    prefix: Optional[str] = None,
    dtype: Optional[str] = None,
    *,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> str:
    """
    构建目录名（用于按 GPU+CC+dtype 分类的场景）
    
    格式: {GPU}_{CC}_{dtype} 或带 prefix 时 {prefix}_{GPU}_{CC}_{dtype}
    
    Args:
        prefix: 可选前缀
        dtype: 数据类型（必需）
        gpu_name: GPU 名称，默认自动检测
        cc_tag: CC 标签，默认自动检测
        
    Examples:
        >>> build_dir_name(dtype="FP8E4M3")
        'H100_cc90_FP8E4M3'
        
        >>> build_dir_name(prefix="results", dtype="INT8")
        'results_H100_cc90_INT8'
    """
    _gpu = gpu_name or hw_info.gpu_name
    _cc = cc_tag or hw_info.cc_tag
    
    components = []
    if prefix:
        components.append(prefix)
    components.extend([_gpu, _cc])
    
    if dtype:
        components.append(normalize_dtype(dtype))
    
    return "_".join(components)


def build_hw_dir_name() -> str:
    """
    构建仅包含硬件信息的目录名。
    
    格式: {GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}
    示例: RTX5080_cc120_py312_cu129_x86_64
    
    Returns:
        目录名称
    """
    return (
        f"{hw_info.gpu_name}_{hw_info.cc_tag}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )


def build_tuned_filename(
    prefix: str,
    model_name: Optional[str] = None,
    ext: str = "",
    outdtype: Optional[str] = None,
) -> str:
    """
    构建 autotune/search 生成文件的名称（统一接口）。
    
    格式:
    - 无模型无 outdtype: {prefix}.{ext}
    - 有模型无 outdtype: {prefix}_{model_name}.{ext}
    - 有模型有 outdtype: {prefix}_{model_name}_out-{outdtype}.{ext}
    
    示例:
    - build_tuned_filename("dequant_bias_tuned", ext=".py")
      -> "dequant_bias_tuned.py"
    - build_tuned_filename("dequant_bias_tuned", "BitNet-2B-BF16", ext=".py")
      -> "dequant_bias_tuned_BitNet-2B-BF16.py"
    - build_tuned_filename("alg_search", "Qwen2.5-0.5B-INT8", ext=".csv", outdtype="int32")
      -> "alg_search_Qwen2.5-0.5B-INT8_out-INT32.csv"
    
    Args:
        prefix: 文件前缀（如 "dequant_bias_tuned", "alg_search"）
        model_name: 模型名称（可选）
        ext: 文件扩展名
        outdtype: 输出数据类型（可选，用于 search 结果文件）
    
    Returns:
        文件名称
    """
    if model_name:
        name = f"{prefix}_{model_name}"
    else:
        name = prefix
    
    # 如果有 outdtype，追加到文件名
    if outdtype:
        outdtype_norm = normalize_dtype(outdtype)
        name = f"{name}_out-{outdtype_norm}"
    
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        name += ext
    
    return name


# =============================================================================
# 文件查找
# =============================================================================

def find_file(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    ext: Optional[str] = None,
    *,
    exact: bool = True,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> Optional[Path]:
    """
    查找符合命名规范的文件
    
    Args:
        prefix: 用途前缀
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        search_dir: 搜索目录
        ext: 文件扩展名（None 表示任意扩展名）
        exact: True 表示精确匹配，False 表示模糊匹配（忽略某些组件）
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        python_tag: Python 版本标签覆盖
        cuda_tag: CUDA 版本标签覆盖
        arch_tag: 系统架构标签覆盖
        
    Returns:
        找到的文件路径，未找到返回 None
        
    Examples:
        >>> find_file("cublaslt_gemm", search_dir="build", ext=".so")
        PosixPath('build/cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so')
        
        >>> find_file("dequant_bias_tuned", dtype="BF16", search_dir="build", ext=".py")
        PosixPath('build/dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py')
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return None
    
    if exact:
        # 精确匹配：构建完整文件名
        if ext:
            filename = build_filename(
                prefix, dtype=dtype, ext=ext,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            target = search_dir / filename
            return target if target.exists() else None
        else:
            # 尝试常见扩展名
            stem = build_stem(
                prefix, dtype=dtype,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            for ext_try in [".so", ".py", ".json", ".csv", ""]:
                target = search_dir / (stem + ext_try)
                if target.exists():
                    return target
            return None
    else:
        # 模糊匹配：使用 glob 模式
        _gpu = gpu_name or hw_info.gpu_name
        _cc = cc_tag or hw_info.cc_tag
        
        # 构建 dtype 模式
        if dtype is None:
            dtype_pattern = "*"
        elif isinstance(dtype, str):
            dtype_pattern = normalize_dtype(dtype)
        else:
            # 多个 dtype 连接
            dtype_pattern = "_".join(normalize_dtype(d) for d in dtype)
        
        # 模糊匹配模式：prefix_GPU_CC_[dtype_]*_py*_cu*_arch
        pattern = f"{prefix}_{_gpu}_{_cc}_{dtype_pattern}_*" if dtype else f"{prefix}_{_gpu}_{_cc}_*"
        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            pattern += ext
        
        matches = list(search_dir.glob(pattern))
        return matches[0] if matches else None


def find_files(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    ext: Optional[str] = None,
    **kwargs
) -> List[Path]:
    """
    查找所有符合条件的文件
    
    参数同 find_file，但返回所有匹配的文件列表。
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return []
    
    _gpu = kwargs.get("gpu_name") or hw_info.gpu_name
    _cc = kwargs.get("cc_tag") or hw_info.cc_tag
    
    # 构建 dtype 模式
    if dtype is None:
        dtype_pattern = "*"
    elif isinstance(dtype, str):
        dtype_pattern = normalize_dtype(dtype)
    else:
        dtype_pattern = "_".join(normalize_dtype(d) for d in dtype)
    
    # 模糊匹配模式
    pattern = f"{prefix}_{_gpu}_{_cc}_{dtype_pattern}_*" if dtype else f"{prefix}_{_gpu}_{_cc}_*"
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        pattern += ext
    
    return sorted(search_dir.glob(pattern))


def find_dir(
    dtype: Optional[str] = None,
    search_dir: Union[str, Path] = ".",
    *,
    prefix: Optional[str] = None,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> Optional[Path]:
    """
    查找符合命名规范的目录
    
    格式: {GPU}_{CC}_{dtype} 或 {prefix}_{GPU}_{CC}_{dtype}
    
    Args:
        dtype: 数据类型
        search_dir: 搜索目录
        prefix: 可选前缀
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        
    Returns:
        找到的目录路径，未找到返回 None
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return None
    
    dir_name = build_dir_name(
        prefix=prefix, dtype=dtype,
        gpu_name=gpu_name, cc_tag=cc_tag
    )
    
    target = search_dir / dir_name
    return target if target.is_dir() else None


# =============================================================================
# 模块加载
# =============================================================================

# 模块缓存
_module_cache: Dict[str, Any] = {}


def load_module(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    *,
    ext: Optional[str] = None,
    cache: bool = True,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> Any:
    """
    加载 Python 模块（.py 或 .so）
    
    自动根据当前硬件信息构建模块名并加载。
    
    Args:
        prefix: 模块前缀
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        search_dir: 搜索目录
        ext: 文件扩展名（None 表示自动检测 .so 或 .py）
        cache: 是否缓存模块
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        python_tag: Python 版本标签覆盖
        cuda_tag: CUDA 版本标签覆盖
        arch_tag: 系统架构标签覆盖
        
    Returns:
        加载的 Python 模块
        
    Raises:
        FileNotFoundError: 模块文件不存在
        ImportError: 模块加载失败
        
    Examples:
        # 无 dtype（支持多类型的 GEMM 扩展）
        >>> module = load_module("cublaslt_gemm", search_dir="build")
        >>> module.gemm(...)
        
        # 带 dtype
        >>> module = load_module("dequant_bias_tuned", dtype="BF16", search_dir="build")
    """
    search_dir = Path(search_dir)
    
    # 构建缓存键
    dtype_key = str(dtype) if dtype else "None"
    cache_key = f"{prefix}_{dtype_key}_{search_dir}_{gpu_name}_{cc_tag}_{python_tag}_{cuda_tag}_{arch_tag}"
    
    if cache and cache_key in _module_cache:
        return _module_cache[cache_key]
    
    # 查找模块文件
    module_path = None
    
    if ext:
        module_path = find_file(
            prefix, dtype=dtype, search_dir=search_dir, ext=ext,
            gpu_name=gpu_name, cc_tag=cc_tag,
            python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
        )
    else:
        # 优先 .so，然后 .py
        for try_ext in [".so", ".py"]:
            module_path = find_file(
                prefix, dtype=dtype, search_dir=search_dir, ext=try_ext,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            if module_path:
                break
    
    if not module_path:
        expected_name = build_filename(
            prefix, dtype=dtype, ext=ext or ".so/.py",
            gpu_name=gpu_name, cc_tag=cc_tag,
            python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
        )
        raise FileNotFoundError(
            f"模块不存在: {expected_name}\n"
            f"搜索路径: {search_dir.absolute()}\n"
        )
    
    # 添加目录到 sys.path
    if str(search_dir.absolute()) not in sys.path:
        sys.path.insert(0, str(search_dir.absolute()))
    
    # 加载模块
    module_name = module_path.stem
    
    if module_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # .so 文件
        module = importlib.import_module(module_name)
    
    if cache:
        _module_cache[cache_key] = module
    
    return module


def clear_module_cache():
    """清除模块缓存"""
    global _module_cache
    _module_cache.clear()


# Tuned module 缓存（按 prefix + model_name 缓存）
_tuned_module_cache: Dict[str, Any] = {}


def load_tuned_module(
    prefix: str,
    model_name: Optional[str],
    build_dir: Union[str, Path],
    *,
    ext: str = ".py",
    cache: bool = True,
    fuzzy_match: bool = True,
) -> Any:
    """
    加载 tuned kernel module。

    新的目录结构：
        build_dir/{hw_dir_name}/{prefix}_{model_name}.py

    例如:
        build/RTX5080_cc120_py312_cu129_x86_64/dequant_bias_tuned_Llama3.2-1B.py

    模糊匹配（fuzzy_match=True，默认）：
        如果 model_name 是完整名称（如 "Llama3.2-1B-FP8"），会先尝试精确匹配，
        如果失败则尝试用 base name（"Llama3.2-1B"）匹配。
        这是因为 Triton kernel 的 autotune 结果对 INT8/FP8 相同，
        生成的文件只使用 base name。

    Args:
        prefix: 模块前缀（如 "dequant_bias_tuned"）
        model_name: 模型名称（如 "Llama3.2-1B-FP8" 或 "Llama3.2-1B"）
        build_dir: kernel build 目录（如 csrc/fused_dequant_bias_triton/build）
        ext: 文件扩展名，默认 ".py"
        cache: 是否缓存模块
        fuzzy_match: 是否启用模糊匹配（默认 True）

    Returns:
        加载的 Python 模块

    Raises:
        FileNotFoundError: 模块文件不存在
        ImportError: 模块加载失败
    """
    build_dir = Path(build_dir)
    
    # 构建缓存键（使用原始 model_name）
    cache_key = f"{prefix}_{model_name}_{build_dir}"
    
    if cache and cache_key in _tuned_module_cache:
        return _tuned_module_cache[cache_key]
    
    # 进入 hw_dir_name 子目录
    hw_dir_name = build_hw_dir_name()
    search_dir = build_dir / hw_dir_name
    
    if not search_dir.exists():
        raise FileNotFoundError(
            f"Hardware-specific directory not found: {search_dir}\n"
            f"Expected format: {build_dir}/{hw_dir_name}/"
        )
    
    # 尝试查找文件的候选列表
    candidates = []
    
    # 1. 精确匹配：使用原始 model_name
    filename_exact = build_tuned_filename(prefix, model_name, ext=ext)
    candidates.append(filename_exact)
    
    # 2. 模糊匹配：使用 base name（去掉 -INT8/-FP8 等后缀）
    if fuzzy_match and model_name:
        base_name = model_base_name(model_name)
        if base_name != model_name:
            filename_base = build_tuned_filename(prefix, base_name, ext=ext)
            candidates.append(filename_base)
    
    # 尝试每个候选文件
    module_path = None
    for filename in candidates:
        candidate_path = search_dir / filename
        if candidate_path.exists():
            module_path = candidate_path
            break
    
    if module_path is None:
        # 生成友好的错误信息
        tried_files = [str(search_dir / f) for f in candidates]
        raise FileNotFoundError(
            f"Tuned kernel not found. Tried:\n"
            f"  {chr(10).join(tried_files)}\n"
            f"Run autotune for model '{model_name}' first."
        )
    
    # 添加目录到 sys.path
    if str(search_dir.absolute()) not in sys.path:
        sys.path.insert(0, str(search_dir.absolute()))
    
    # 加载模块
    module_name = module_path.stem
    
    if module_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # .so 文件
        module = importlib.import_module(module_name)
    
    if cache:
        _tuned_module_cache[cache_key] = module
    
    return module


def clear_tuned_module_cache():
    """清除 tuned module 缓存"""
    global _tuned_module_cache
    _tuned_module_cache.clear()


# =============================================================================
# 算法查表（运行时使用）
# =============================================================================

def lookup_best_cublaslt_alg(json_data: Dict, N: int, K: int, M: int) -> Optional[str]:
    """
    从 JSON 数据中查询 cuBLASLt 最佳算法配置。
    
    查询逻辑：
    1. 用 (N, K) 在 nk_entries 中找到对应条目
    2. 在 m_thresholds 中找到 <= query_M 的最大值
    3. 返回该 M 对应的 alg_by_m[m][0]（最佳配置的 base64 编码）
    
    Args:
        json_data: 加载的 JSON 数据（由 alg_search 生成）
        N: 矩阵 W 的行数
        K: 共享维度
        M: 矩阵 A 的行数（batch size）
    
    Returns:
        最佳算法的 base64 编码字符串（64B cublasLtMatmulAlgo_t 数据），
        如果找不到返回 None
    
    """
    nk_key = f"({N},{K})"
    nk_entries = json_data.get("nk_entries", {})
    
    if nk_key not in nk_entries:
        return None
    
    entry = nk_entries[nk_key]
    m_thresholds = entry.get("m_thresholds", [])
    alg_by_m = entry.get("alg_by_m", {})
    
    if not m_thresholds:
        return None
    
    # 找到 <= M 的最大阈值
    selected_m = None
    for threshold in m_thresholds:
        if threshold <= M:
            selected_m = threshold
        else:
            break
    
    if selected_m is None:
        # M 比所有阈值都小，使用最小的阈值
        selected_m = m_thresholds[0]
    
    m_key = str(selected_m)
    if m_key in alg_by_m:
        # 格式: alg_by_m[m_key] = [best_b64, 2nd_b64, 3rd_b64]
        alg_list = alg_by_m[m_key]
        if isinstance(alg_list, list) and len(alg_list) > 0:
            return alg_list[0]
    
    return None


def decode_cublaslt_algo_data(algo_data_b64: str) -> bytes:
    """
    解码 base64 编码的 cuBLASLt algo_data，返回 64 字节的原始数据。
    
    运行时使用：将返回的 bytes 直接 memcpy 到 cublasLtMatmulAlgo_t 结构体。
    
    Args:
        algo_data_b64: base64 编码的算法数据
        
    Returns:
        64 字节的原始算法数据
    """
    return base64.b64decode(algo_data_b64)


def lookup_best_cusparselt_alg(json_data: Dict, N: int, K: int, M: int) -> Optional[Dict]:
    """
    从 JSON 数据中查询 cuSPARSELt 最佳算法配置。
    
    查询逻辑：
    1. 用 (N, K) 在 nk_entries 中找到对应条目
    2. 在 m_thresholds 中找到 <= query_M 的最大值
    3. 返回该 M 对应的 alg_by_m[m][0]（最佳配置）
    
    Args:
        json_data: 加载的 JSON 数据（由 alg_search 生成）
        N: 稀疏矩阵 W 的行数
        K: 共享维度
        M: 稠密矩阵 A 的行数（batch size）
    
    Returns:
        最佳配置字典 {"alg_id": int, "split_k": int, "workspace": int}，
        如果找不到返回 None
    
    """
    nk_key = f"({N},{K})"
    nk_entries = json_data.get("nk_entries", {})
    
    if nk_key not in nk_entries:
        return None
    
    entry = nk_entries[nk_key]
    m_thresholds = entry.get("m_thresholds", [])
    alg_by_m = entry.get("alg_by_m", {})
    
    if not m_thresholds:
        return None
    
    # 找到 <= M 的最大阈值
    selected_m = None
    for threshold in m_thresholds:
        if threshold <= M:
            selected_m = threshold
        else:
            break
    
    if selected_m is None:
        # M 比所有阈值都小，使用最小的阈值
        selected_m = m_thresholds[0]
    
    m_key = str(selected_m)
    if m_key in alg_by_m:
        alg_list = alg_by_m[m_key]
        if isinstance(alg_list, list) and len(alg_list) > 0:
            first_entry = alg_list[0]
            # 支持新格式 {"alg_id": int, "split_k": int, "workspace": int} 和旧格式 int
            if isinstance(first_entry, dict):
                return first_entry
            else:
                # 兼容旧格式（仅 alg_id）
                return {"alg_id": first_entry, "split_k": 1, "workspace": 0}
    
    return None


# =============================================================================
# 文件保存
# =============================================================================

def save_json(
    data: Any,
    prefix: str,
    dtype: Optional[str] = None,
    save_dir: Union[str, Path] = ".",
    *,
    indent: int = 2,
    **kwargs
) -> Path:
    """
    保存数据为 JSON 文件
    
    Args:
        data: 要保存的数据
        prefix: 文件前缀
        dtype: 数据类型
        save_dir: 保存目录
        indent: JSON 缩进
        **kwargs: 传递给 build_filename 的参数
        
    Returns:
        保存的文件路径
    """
    import json
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = build_filename(prefix, dtype=dtype, ext=".json", **kwargs)
    filepath = save_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    return filepath


def load_json(
    prefix: str,
    dtype: Optional[str] = None,
    search_dir: Union[str, Path] = ".",
    **kwargs
) -> Any:
    """
    加载 JSON 文件
    
    Args:
        prefix: 文件前缀
        dtype: 数据类型
        search_dir: 搜索目录
        **kwargs: 传递给 find_file 的参数
        
    Returns:
        加载的数据
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    import json
    
    filepath = find_file(prefix, dtype=dtype, search_dir=search_dir, ext=".json", **kwargs)
    
    if not filepath:
        expected_name = build_filename(prefix, dtype=dtype, ext=".json", **kwargs)
        raise FileNotFoundError(f"JSON 文件不存在: {expected_name} in {search_dir}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    prefix: str,
    dtype: Optional[str] = None,
    save_dir: Union[str, Path] = ".",
    **kwargs
) -> Path:
    """
    保存数据为 CSV 文件
    
    Args:
        data: 字典列表形式的数据
        prefix: 文件前缀
        dtype: 数据类型
        save_dir: 保存目录
        **kwargs: 传递给 build_filename 的参数
        
    Returns:
        保存的文件路径
    """
    import csv
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = build_filename(prefix, dtype=dtype, ext=".csv", **kwargs)
    filepath = save_dir / filename
    
    if not data:
        filepath.touch()
        return filepath
    
    fieldnames = list(data[0].keys())
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return filepath


# =============================================================================
# 目录管理
# =============================================================================

def ensure_result_dir(
    base_dir: Union[str, Path],
    dtype: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> Path:
    """
    确保结果目录存在并返回路径
    
    创建格式为 {GPU}_{CC}_{dtype} 的子目录。
    
    Args:
        base_dir: 基础目录
        dtype: 数据类型
        prefix: 可选前缀
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        
    Returns:
        创建/已存在的目录路径
        
    Examples:
        >>> result_dir = ensure_result_dir("results", dtype="FP8E4M3")
        >>> # Creates: results/H100_cc90_FP8E4M3/
    """
    base_dir = Path(base_dir)
    dir_name = build_dir_name(prefix=prefix, dtype=dtype, gpu_name=gpu_name, cc_tag=cc_tag)
    result_dir = base_dir / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


# #############################################################################
#
#  PART 4: 模型信息管理
#
#  本部分提供模型注册表和模型信息查询功能。
#
#  主要内容：
#  =========
#  - MODEL_SIZE_GB: 模型大小参考表
#  - ModelEntry: 模型条目数据类
#  - ModelRegistry: 模型注册表（单例）
#  - 便捷函数: get_model_info, list_models, check_model_downloaded 等
#
#  使用示例：
#  =========
#  >>> from slidesparse.utils import model_registry, get_model_info
#  >>> info = get_model_info("Qwen2.5-0.5B-FP8")
#  >>> models = model_registry.list(family="Qwen2.5")
#
# #############################################################################


# =============================================================================
# 模型大小参考
# =============================================================================

# 模型大小参考（用于估算显存需求）
MODEL_SIZE_GB = {
    "0.5B": 0.9,
    "1B": 1.9,
    "1.5B": 2.1,
    "2B": 4.8,    # BitNet 2B-BF16
    "3B": 4.0,
    "7B": 8.1,
    "14B": 15.2,
}


@dataclass
class ModelEntry:
    """
    单个模型的信息条目
    
    Attributes:
        key: 模型短键名（如 "qwen2.5-7b-fp8"）
        family: 模型系列（如 "qwen", "llama"）
        version: 版本号（如 "2.5", "3.2"）
        size: 模型大小（如 "7B", "1.5B"）
        quant: 量化类型（如 "fp8", "int8"）
        hf_name: HuggingFace 模型名（如 "Qwen2.5-7B-Instruct-FP8-dynamic"）
        local_name: 本地文件夹名（如 "Qwen2.5-7B-FP8"）
        hf_org: HuggingFace 组织名（默认 "RedHatAI"）
    """
    key: str
    family: str
    version: str
    size: str
    quant: str
    hf_name: str
    local_name: str
    hf_org: str = "RedHatAI"
    
    @property
    def hf_path(self) -> Optional[str]:
        """完整的 HuggingFace 路径，本地生成的模型返回 None"""
        if self.hf_name is None or self.hf_org is None:
            return None
        return f"{self.hf_org}/{self.hf_name}"
    
    @property
    def quant_normalized(self) -> str:
        """标准化的量化类型"""
        return normalize_dtype(self.quant) if self.quant.lower() not in ("int8",) else "INT8"
    
    @property
    def estimated_gb(self) -> float:
        """估算的显存需求 (GB)"""
        size_upper = self.size.upper()
        return MODEL_SIZE_GB.get(size_upper, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "family": self.family,
            "version": self.version,
            "size": self.size,
            "quant": self.quant,
            "quant_normalized": self.quant_normalized,
            "hf_name": self.hf_name,
            "hf_path": self.hf_path,
            "local_name": self.local_name,
            "estimated_gb": self.estimated_gb,
        }


class ModelRegistry:
    """
    模型注册表
    
    管理所有支持的量化模型，提供搜索、过滤、路径构建等功能。
    
    命名规范：
        - key: {family}{version}-{size}-{quant}  例如 "qwen2.5-7b-fp8"
        - local_name: {Family}{Version}-{Size}-{QUANT}  例如 "Qwen2.5-7B-FP8"
    
    使用示例：
        >>> registry = ModelRegistry()
        >>> 
        >>> # 获取所有 FP8 模型
        >>> for entry in registry.list(quant="fp8"):
        ...     print(entry.key, entry.local_name)
        >>> 
        >>> # 获取特定模型
        >>> entry = registry.get("qwen2.5-7b-fp8")
        >>> print(entry.hf_path)
        >>> 
        >>> # 按 family 过滤
        >>> for entry in registry.list(family="llama"):
        ...     print(entry.key)
    """
    
    # 内置模型定义
    # 格式: (family, version, size, quant, hf_name, local_name)
    _BUILTIN_MODELS = [
        # Qwen2.5 INT8
        ("qwen", "2.5", "0.5B", "int8", "Qwen2.5-0.5B-Instruct-quantized.w8a8", "Qwen2.5-0.5B-INT8"),
        ("qwen", "2.5", "1.5B", "int8", "Qwen2.5-1.5B-Instruct-quantized.w8a8", "Qwen2.5-1.5B-INT8"),
        ("qwen", "2.5", "3B", "int8", "Qwen2.5-3B-Instruct-quantized.w8a8", "Qwen2.5-3B-INT8"),
        ("qwen", "2.5", "7B", "int8", "Qwen2.5-7B-Instruct-quantized.w8a8", "Qwen2.5-7B-INT8"),
        ("qwen", "2.5", "14B", "int8", "Qwen2.5-14B-Instruct-quantized.w8a8", "Qwen2.5-14B-INT8"),
        # Qwen2.5 FP8
        ("qwen", "2.5", "0.5B", "fp8", "Qwen2.5-0.5B-Instruct-FP8-dynamic", "Qwen2.5-0.5B-FP8"),
        ("qwen", "2.5", "1.5B", "fp8", "Qwen2.5-1.5B-Instruct-FP8-dynamic", "Qwen2.5-1.5B-FP8"),
        ("qwen", "2.5", "3B", "fp8", "Qwen2.5-3B-Instruct-FP8-dynamic", "Qwen2.5-3B-FP8"),
        ("qwen", "2.5", "7B", "fp8", "Qwen2.5-7B-Instruct-FP8-dynamic", "Qwen2.5-7B-FP8"),
        ("qwen", "2.5", "14B", "fp8", "Qwen2.5-14B-Instruct-FP8-dynamic", "Qwen2.5-14B-FP8"),
        # Llama3.2 INT8
        ("llama", "3.2", "1B", "int8", "Llama-3.2-1B-Instruct-quantized.w8a8", "Llama3.2-1B-INT8"),
        ("llama", "3.2", "3B", "int8", "Llama-3.2-3B-Instruct-quantized.w8a8", "Llama3.2-3B-INT8"),
        # Llama3.2 FP8
        ("llama", "3.2", "1B", "fp8", "Llama-3.2-1B-Instruct-FP8-dynamic", "Llama3.2-1B-FP8"),
        ("llama", "3.2", "3B", "fp8", "Llama-3.2-3B-Instruct-FP8-dynamic", "Llama3.2-3B-FP8"),
        # BitNet BF16 (microsoft)
        ("bitnet", "1.58", "2B", "bf16", "bitnet-b1.58-2B-4T-bf16", "BitNet-2B-BF16", "microsoft"),
        # BitNet INT8/FP8 (本地量化生成，无 HF 路径)
        ("bitnet", "1.58", "2B", "int8", None, "BitNet-2B-INT8", None),
        ("bitnet", "1.58", "2B", "fp8", None, "BitNet-2B-FP8", None),
    ]
    
    def __init__(self, hf_org: str = "RedHatAI"):
        """
        初始化模型注册表
        
        Args:
            hf_org: 默认的 HuggingFace 组织名
        """
        self.hf_org = hf_org
        self._models: Dict[str, ModelEntry] = {}
        
        # 加载内置模型
        # 支持 6 元组 (family, version, size, quant, hf_name, local_name)
        # 和 7 元组 (family, version, size, quant, hf_name, local_name, custom_hf_org)
        for model_tuple in self._BUILTIN_MODELS:
            if len(model_tuple) == 7:
                family, version, size, quant, hf_name, local_name, custom_org = model_tuple
            else:
                family, version, size, quant, hf_name, local_name = model_tuple
                custom_org = hf_org
            
            key = self._make_key(family, version, size, quant)
            self._models[key] = ModelEntry(
                key=key,
                family=family,
                version=version,
                size=size,
                quant=quant,
                hf_name=hf_name,
                local_name=local_name,
                hf_org=custom_org,
            )
    
    @staticmethod
    def _make_key(family: str, version: str, size: str, quant: str) -> str:
        """生成模型 key"""
        return f"{family}{version}-{size.lower()}-{quant.lower()}"
    
    def register(
        self,
        family: str,
        version: str,
        size: str,
        quant: str,
        hf_name: str,
        local_name: str,
        hf_org: Optional[str] = None,
    ) -> ModelEntry:
        """
        注册新模型
        
        Args:
            family: 模型系列
            version: 版本号
            size: 模型大小
            quant: 量化类型
            hf_name: HuggingFace 模型名
            local_name: 本地文件夹名
            hf_org: HuggingFace 组织名
            
        Returns:
            注册的模型条目
        """
        key = self._make_key(family, version, size, quant)
        entry = ModelEntry(
            key=key,
            family=family,
            version=version,
            size=size,
            quant=quant,
            hf_name=hf_name,
            local_name=local_name,
            hf_org=hf_org or self.hf_org,
        )
        self._models[key] = entry
        return entry
    
    def get(self, key: str) -> Optional[ModelEntry]:
        """
        获取模型条目
        
        Args:
            key: 模型 key（如 "qwen2.5-7b-fp8"）
            
        Returns:
            模型条目，不存在返回 None
        """
        return self._models.get(key.lower())
    
    def __getitem__(self, key: str) -> ModelEntry:
        """通过 key 获取模型（KeyError if not found）"""
        entry = self.get(key)
        if entry is None:
            raise KeyError(f"模型不存在: {key}")
        return entry
    
    def __contains__(self, key: str) -> bool:
        """检查模型是否存在"""
        return key.lower() in self._models
    
    def __len__(self) -> int:
        """模型数量"""
        return len(self._models)
    
    def __iter__(self):
        """迭代所有模型"""
        return iter(self._models.values())
    
    def list(
        self,
        *,
        family: Optional[str] = None,
        version: Optional[str] = None,
        size: Optional[str] = None,
        quant: Optional[str] = None,
        sort_by_size: bool = True,
    ) -> List[ModelEntry]:
        """
        列出符合条件的模型
        
        Args:
            family: 过滤模型系列（qwen, llama）
            version: 过滤版本号（2.5, 3.2）
            size: 过滤模型大小（0.5B, 7B）
            quant: 过滤量化类型（fp8, int8）
            sort_by_size: 是否按模型大小排序
            
        Returns:
            符合条件的模型列表
        """
        results = []
        
        for entry in self._models.values():
            if family and entry.family.lower() != family.lower():
                continue
            if version and entry.version != version:
                continue
            if size and entry.size.lower() != size.lower():
                continue
            if quant and entry.quant.lower() != quant.lower():
                continue
            results.append(entry)
        
        if sort_by_size:
            # 按模型大小排序
            def size_key(e: ModelEntry) -> float:
                s = e.size.upper().replace("B", "")
                try:
                    return float(s)
                except ValueError:
                    return 0.0
            results.sort(key=size_key)
        
        return results
    
    def keys(
        self,
        *,
        family: Optional[str] = None,
        quant: Optional[str] = None,
    ) -> List[str]:
        """
        获取模型 key 列表
        
        Args:
            family: 过滤模型系列
            quant: 过滤量化类型
            
        Returns:
            模型 key 列表
        """
        return [e.key for e in self.list(family=family, quant=quant)]
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """导出为字典"""
        return {k: v.to_dict() for k, v in self._models.items()}


# 全局模型注册表实例
model_registry = ModelRegistry()


# =============================================================================
# 模型路径和检查便捷函数
# =============================================================================

def get_model_registry() -> ModelRegistry:
    """获取全局模型注册表"""
    return model_registry


def get_model_info(key: str) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        key: 模型 key
        
    Returns:
        模型信息字典
        
    Raises:
        KeyError: 模型不存在
    """
    entry = model_registry.get(key)
    if entry is None:
        raise KeyError(f"模型不存在: {key}. 可用模型: {', '.join(model_registry.keys())}")
    return entry.to_dict()


def list_models(
    *,
    family: Optional[str] = None,
    quant: Optional[str] = None,
) -> List[str]:
    """
    列出模型 key
    
    Args:
        family: 过滤模型系列
        quant: 过滤量化类型
        
    Returns:
        模型 key 列表
    """
    return model_registry.keys(family=family, quant=quant)


def build_model_dir_name(
    family: str,
    version: str,
    size: str,
    quant: str,
) -> str:
    """
    构建模型目录名
    
    格式: {Family}{Version}-{Size}-{QUANT}
    例如: Qwen2.5-7B-FP8, Llama3.2-1B-INT8
    
    Args:
        family: 模型系列（qwen, llama）
        version: 版本号（2.5, 3.2）
        size: 模型大小（7B, 1B）
        quant: 量化类型（fp8, int8）
        
    Returns:
        目录名字符串
    """
    # 首字母大写
    family_cap = family.capitalize()
    size_upper = size.upper()
    quant_upper = "FP8" if quant.lower() == "fp8" else "INT8"
    return f"{family_cap}{version}-{size_upper}-{quant_upper}"


def parse_model_key(key: str) -> Dict[str, str]:
    """
    解析模型 key
    
    Args:
        key: 模型 key（如 "qwen2.5-7b-fp8"）
        
    Returns:
        解析结果字典 {"family", "version", "size", "quant"}
        
    Raises:
        ValueError: 无法解析
    """
    # 尝试从注册表获取
    entry = model_registry.get(key)
    if entry:
        return {
            "family": entry.family,
            "version": entry.version,
            "size": entry.size,
            "quant": entry.quant,
        }
    
    # 尝试手动解析: {family}{version}-{size}-{quant}
    # 例如: qwen2.5-7b-fp8, llama3.2-1b-int8
    import re
    match = re.match(r'^([a-z]+)([\d.]+)-(\d+\.?\d*b)-([a-z0-9]+)$', key.lower())
    if match:
        return {
            "family": match.group(1),
            "version": match.group(2),
            "size": match.group(3).upper(),
            "quant": match.group(4),
        }
    
    raise ValueError(f"无法解析模型 key: {key}")


def check_quant_support(quant: str) -> Tuple[bool, str]:
    """
    检查当前 GPU 是否支持指定的量化类型
    
    Args:
        quant: 量化类型（fp8, int8）
        
    Returns:
        (supported, message)
    """
    quant_lower = quant.lower()
    
    if quant_lower == "int8":
        # INT8: CC >= 8.0 (Ampere+)
        if hw_info.cc_major >= 8:
            return True, f"GPU {hw_info.gpu_name} (CC {hw_info.cc_tag}) supports INT8"
        return False, (
            f"GPU {hw_info.gpu_name} (CC {hw_info.cc_tag}) does not support efficient INT8 Tensor Core.\n"
            f"INT8 requires Ampere (CC 8.0+) or newer."
        )
    
    elif quant_lower == "fp8":
        # FP8: CC >= 8.9 (Ada/Hopper+)
        if hw_info.supports_fp8:
            return True, f"GPU {hw_info.gpu_name} (CC {hw_info.cc_tag}) supports FP8"
        return False, (
            f"GPU {hw_info.gpu_name} (CC {hw_info.cc_tag}) does not support native FP8.\n"
            f"FP8 requires Ada (CC 8.9+) or Hopper (CC 9.0+) or newer."
        )
    
    else:
        return False, f"Unknown quantization type: {quant}"


def get_model_local_path(
    key: str,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Path:
    """
    获取模型本地路径
    
    Args:
        key: 模型 key
        checkpoint_dir: checkpoints 根目录
        
    Returns:
        模型本地目录路径
        
    Raises:
        KeyError: 模型不存在
    """
    entry = model_registry[key]
    return Path(checkpoint_dir) / entry.local_name


def check_model_downloaded(
    key: str,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Tuple[bool, str]:
    """
    检查模型是否已下载
    
    Args:
        key: 模型 key
        checkpoint_dir: checkpoints 根目录
        
    Returns:
        (downloaded, message)
    """
    try:
        local_path = get_model_local_path(key, checkpoint_dir)
    except KeyError as e:
        return False, str(e)
    
    if local_path.is_dir() and (local_path / "config.json").exists():
        return True, f"Model exists: {local_path}"
    return False, f"Model not found: {local_path}"


# #############################################################################
#
#  PART 5: SlideSparse 配置与维度计算
#
#  本部分提供 SlideSparse 稀疏格式的配置和维度计算功能。
#
#  稀疏格式说明：
#  =============
#  Z:L 表示每 L 个连续元素中至少有 Z 个零
#  例如 2:8 表示每 8 个元素至少 2 个零（稀疏度 ≥ 25%）
#
#  主要功能：
#  =========
#  - SlideSparseConfig: 配置数据类
#  - compute_output_k: 计算 slided 后的 K 维度
#  - compute_compressed_k: 计算 2:4 压缩后的 K 维度
#  - get_model_nk_sizes: 提取模型的 NK 尺寸
#
# #############################################################################


# =============================================================================
# SlideSparse 配置
# =============================================================================

@dataclass
class SlideSparseConfig:
    """
    SlideSparse 转换配置
    
    稀疏格式说明：
        Z:L 表示每 L 个连续元素中至少有 Z 个零
        例如 2:8 表示每 8 个元素至少 2 个零（稀疏度 ≥ 25%）
    
    Attributes:
        Z: 每组中至少的零元素数量（当前固定为 2）
        L: 稀疏组的大小（如 6, 8, 10）
        N: 内部参数，N = L // 2
        window_size: 滑动窗口大小，固定为 4（对应 2:4 硬件）
        stride: 滑动步长，固定为 2
        num_windows: 每组内的窗口数量，= N - 1
        expand_ratio: K 维度的扩展比例
    """
    Z: int = 2
    L: int = 8
    
    # 派生参数（在 __post_init__ 中计算）
    N: int = field(init=False)
    window_size: int = field(init=False)
    stride: int = field(init=False)
    num_windows: int = field(init=False)
    expand_ratio: float = field(init=False)
    in_group_size: int = field(init=False)
    out_group_size: int = field(init=False)
    
    def __post_init__(self):
        if self.Z != 2:
            raise ValueError(f"当前仅支持 Z=2 的稀疏格式，收到 Z={self.Z}")
        if self.L % 2 != 0:
            raise ValueError(f"L 必须为偶数，收到 L={self.L}")
        if self.L < 4:
            import warnings
            warnings.warn(
                f"L={self.L} < 4，这是纯量化模式（无稀疏），slide 操作将被跳过",
                UserWarning
            )
        
        self.N = self.L // 2
        self.window_size = 4
        self.stride = 2
        self.num_windows = self.N - 1
        self.expand_ratio = (self.num_windows * self.window_size) / self.L
        self.in_group_size = self.L
        self.out_group_size = self.num_windows * self.window_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Z": self.Z, "L": self.L, "N": self.N,
            "window_size": self.window_size, "stride": self.stride,
            "num_windows": self.num_windows, "expand_ratio": self.expand_ratio,
            "in_group_size": self.in_group_size, "out_group_size": self.out_group_size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlideSparseConfig":
        return cls(Z=d["Z"], L=d["L"])
    
    def __repr__(self):
        return (f"SlideSparseConfig(Z={self.Z}, L={self.L}, N={self.N}, "
                f"expand={self.expand_ratio:.3f})")


def compute_output_k(k_in: int, config: SlideSparseConfig, align_to: int = 32) -> Tuple[int, int]:
    """
    计算滑动扩展后的 K 维度
    
    Args:
        k_in: 原始输入维度 K
        config: SlideSparse 配置
        align_to: 输出对齐要求（默认 32
    
    Returns:
        (k_padded, k_out):
            - k_padded: padding 后的输入 K（L 的倍数）
            - k_out: 滑动扩展后的输出 K（对齐到 align_to）
    """
    L = config.L
    k_padded = ((k_in + L - 1) // L) * L
    num_groups = k_padded // L
    k_out_raw = num_groups * config.out_group_size
    k_out = ((k_out_raw + align_to - 1) // align_to) * align_to
    return k_padded, k_out


def compute_compressed_k(k_slided: int) -> int:
    """
    计算 2:4 压缩后的 K 维度
    
    2:4 压缩将 K 减半（每 4 个元素压缩为 2 个值 + metadata）
    """
    return k_slided // 2


# =============================================================================
# 模型 NK Size 提取工具
# =============================================================================

# 线性层类型映射（标准 HuggingFace 格式）
LINEAR_LAYER_TYPES = {
    # Attention 层 - 映射到统一的键名
    "q_proj": "qkv",       # Q projection -> qkv（合并报告）
    "k_proj": "qkv",       # K projection -> qkv
    "v_proj": "qkv",       # V projection -> qkv
    "qkv_proj": "qkv",     # QKV 融合
    "o_proj": "wo",        # Output projection
    # MLP 层
    "gate_proj": "w13",    # Gate projection (w1) -> w13（合并报告）
    "up_proj": "w13",      # Up projection (w3) -> w13
    "gate_up_proj": "w13", # Gate+Up 融合
    "down_proj": "w2",     # Down projection (w2)
}


def get_model_nk_sizes(
    model_path: Union[str, Path],
    *,
    layer_index: int = 0,
) -> Dict[str, Tuple[int, int]]:
    """
    从 safetensor 文件提取模型线性层的 N,K 尺寸
    
    Args:
        model_path: 模型目录或 safetensor 文件路径
        layer_index: 使用哪一层的尺寸（默认第 0 层，因为所有层尺寸相同）
    
    Returns:
        Dict[str, Tuple[int, int]]: 层类型 -> (N, K) 尺寸
        键为: "qkv", "wo", "w13", "w2"
        
    Example:
        >>> sizes = get_model_nk_sizes("checkpoints/Qwen2.5-7B-INT8")
        >>> sizes
        {
            'qkv': (4608, 3584),   # Q+K+V 合并后的 N, 共同的 K
            'wo': (3584, 3584),    # Output projection
            'w13': (18944, 3584),  # Gate+Up 合并后的 N
            'w2': (3584, 9472),    # Down projection
        }
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors is required: pip install safetensors")
    
    model_path = Path(model_path)
    
    # 找到 safetensor 文件
    if model_path.is_file() and model_path.suffix == ".safetensors":
        safetensor_files = [model_path]
    elif model_path.is_dir():
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files in {model_path}")
    else:
        raise FileNotFoundError(f"Path not found: {model_path}")
    
    # 收集尺寸
    # 格式: { "qkv": {"q": (N,K), "k": (N,K), "v": (N,K)}, "wo": (N,K), ... }
    raw_sizes: Dict[str, Dict[str, Tuple[int, int]]] = {
        "qkv": {}, "wo": {}, "w13": {}, "w2": {}
    }
    
    target_layer_prefix = f".{layer_index}."  # e.g., ".0."
    
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                # 只处理目标层
                if target_layer_prefix not in key:
                    continue
                if "weight" not in key.lower() or "scale" in key.lower():
                    continue
                
                tensor = f.get_tensor(key)
                if tensor.dim() != 2:
                    continue
                
                N, K = tensor.shape
                
                # 识别层类型
                key_lower = key.lower()
                for pattern, group in LINEAR_LAYER_TYPES.items():
                    if pattern in key_lower:
                        if group == "qkv":
                            # 分别记录 q/k/v
                            if "q_proj" in key_lower or "qkv_proj" in key_lower:
                                raw_sizes["qkv"]["q"] = (N, K)
                            if "k_proj" in key_lower:
                                raw_sizes["qkv"]["k"] = (N, K)
                            if "v_proj" in key_lower:
                                raw_sizes["qkv"]["v"] = (N, K)
                            if "qkv_proj" in key_lower:
                                # 融合的 QKV
                                raw_sizes["qkv"]["qkv"] = (N, K)
                        elif group == "w13":
                            if "gate_proj" in key_lower or "gate_up_proj" in key_lower:
                                raw_sizes["w13"]["gate"] = (N, K)
                            if "up_proj" in key_lower:
                                raw_sizes["w13"]["up"] = (N, K)
                            if "gate_up_proj" in key_lower:
                                raw_sizes["w13"]["gate_up"] = (N, K)
                        else:
                            raw_sizes[group] = (N, K)
                        break
    
    # 合并尺寸
    result: Dict[str, Tuple[int, int]] = {}
    
    # QKV 处理
    qkv_data = raw_sizes["qkv"]
    if "qkv" in qkv_data:
        # 融合的 QKV
        result["qkv"] = qkv_data["qkv"]
    elif qkv_data:
        # 分离的 Q, K, V - 合并 N, K 应该相同
        q_size = qkv_data.get("q", (0, 0))
        k_size = qkv_data.get("k", (0, 0))
        v_size = qkv_data.get("v", (0, 0))
        total_n = q_size[0] + k_size[0] + v_size[0]
        common_k = q_size[1] or k_size[1] or v_size[1]
        result["qkv"] = (total_n, common_k)
    
    # W13 处理
    w13_data = raw_sizes["w13"]
    if "gate_up" in w13_data:
        result["w13"] = w13_data["gate_up"]
    elif w13_data:
        gate_size = w13_data.get("gate", (0, 0))
        up_size = w13_data.get("up", (0, 0))
        total_n = gate_size[0] + up_size[0]
        common_k = gate_size[1] or up_size[1]
        result["w13"] = (total_n, common_k)
    
    # WO 和 W2 直接使用
    if isinstance(raw_sizes["wo"], tuple):
        result["wo"] = raw_sizes["wo"]
    if isinstance(raw_sizes["w2"], tuple):
        result["w2"] = raw_sizes["w2"]
    
    return result


def get_model_nk_sizes_slided(
    nk_sizes: Dict[str, Tuple[int, int]],
    Z: int,
    L: int,
    align_to: int = 32,
) -> Dict[str, Tuple[int, int]]:
    """
    计算 slide 后的 N,K 尺寸
    
    Args:
        nk_sizes: 原始 N,K 尺寸（来自 get_model_nk_sizes）
        Z: 稀疏度分子
        L: 稀疏度分母
        align_to: 对齐要求
    
    Returns:
        Dict[str, Tuple[int, int]]: 层类型 -> slide 后的 (N, K_out)
        
    Example:
        >>> sizes = get_model_nk_sizes("checkpoints/Qwen2.5-7B-INT8")
        >>> slided = get_model_nk_sizes_slided(sizes, Z=2, L=8)
        >>> slided
        {
            'qkv': (4608, 5376),   # K 扩展 1.5x
            'wo': (3584, 5376),
            'w13': (18944, 5376),
            'w2': (3584, 14208),
        }
    """
    config = SlideSparseConfig(Z=Z, L=L)
    result = {}
    
    for layer_type, (N, K) in nk_sizes.items():
        _, k_out = compute_output_k(K, config, align_to)
        result[layer_type] = (N, k_out)
    
    return result


def get_model_nk_sizes_compressed(
    nk_sizes_slided: Dict[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """
    计算 2:4 压缩后的 N,K 尺寸
    
    Args:
        nk_sizes_slided: slide 后的 N,K 尺寸
    
    Returns:
        Dict[str, Tuple[int, int]]: 层类型 -> 压缩后的 (N, K_compressed)
    """
    result = {}
    for layer_type, (N, K) in nk_sizes_slided.items():
        result[layer_type] = (N, compute_compressed_k(K))
    return result


def print_model_nk_summary(
    model_path: Union[str, Path],
    Z: int = 2,
    L: int = 8,
    align_to: int = 32,
) -> None:
    """
    打印模型的 NK 尺寸摘要（原始、slide、压缩）
    
    Args:
        model_path: 模型路径
        Z, L: 稀疏参数
        align_to: 对齐要求
    """
    print(f"Model NK Size Summary: {model_path}")
    print(f"SlideSparse Config: Z={Z}, L={L}, align={align_to}")
    print("=" * 70)
    
    config = SlideSparseConfig(Z=Z, L=L)
    print(f"Expand ratio: {config.expand_ratio:.4f}")
    print()
    
    original = get_model_nk_sizes(model_path)
    slided = get_model_nk_sizes_slided(original, Z, L, align_to)
    compressed = get_model_nk_sizes_compressed(slided)
    
    print(f"{'Layer':<8} {'Original N,K':<18} {'Slided N,K':<18} {'Compressed N,K':<18}")
    print("-" * 70)
    
    for layer in ["qkv", "wo", "w13", "w2"]:
        if layer in original:
            orig = original[layer]
            slid = slided[layer]
            comp = compressed[layer]
            print(f"{layer:<8} {str(orig):<18} {str(slid):<18} {str(comp):<18}")


# =============================================================================
# 稀疏配置解析（环境变量 SPARSITY）
# =============================================================================

# 缓存解析结果
_sparsity_config_cache = None


def parse_sparsity_env(sparsity_str: str = None) -> Tuple[int, int, float]:
    """
    解析稀疏格式配置
    
    Args:
        sparsity_str: 稀疏格式字符串（如 "2_8"），如果为 None 则读取环境变量 SPARSITY
    
    Returns:
        (Z, L, expand_ratio) 元组:
            - Z: 每组中的零元素数量（固定为 2）
            - L: 稀疏组的大小（如 6, 8, 10）
            - expand_ratio: K 维度扩展比例 = L / (L - Z)
        
        如果未设置或格式错误，返回默认值 (2, 8, 1.333...)
    """
    if sparsity_str is None:
        sparsity_str = os.environ.get("SPARSITY", "2_8")
    
    try:
        parts = sparsity_str.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid SPARSITY format: {sparsity_str}")
        
        Z = int(parts[0])
        L = int(parts[1])
        
        if Z != 2:
            Z = 2  # 仅支持 Z=2
        
        if L < 4 or L % 2 != 0:
            L = 8  # 必须 >= 4 且为偶数
        
        expand_ratio = L / (L - Z)
        return (Z, L, expand_ratio)
        
    except (ValueError, AttributeError):
        return (2, 8, 8 / 6)


def get_sparsity_config_cached() -> Tuple[int, int, float]:
    """
    获取稀疏格式配置（带缓存）
    
    从环境变量 SPARSITY 解析，结果会被缓存
    """
    global _sparsity_config_cache
    
    if _sparsity_config_cache is None:
        _sparsity_config_cache = parse_sparsity_env()
    
    return _sparsity_config_cache


def clear_sparsity_cache() -> None:
    """清除稀疏配置缓存（用于测试时重新读取环境变量）"""
    global _sparsity_config_cache
    _sparsity_config_cache = None


def get_sparsity_str(Z: int = None, L: int = None) -> str:
    """
    获取稀疏格式字符串
    
    Args:
        Z, L: 如果提供则直接使用，否则从缓存/环境变量获取
    
    Returns:
        格式如 "2_8"、"2_6" 等
    """
    if Z is None or L is None:
        Z, L, _ = get_sparsity_config_cached()
    return f"{Z}_{L}"


# =============================================================================
# SlideSparse 模型路径解析
# =============================================================================

def get_slidesparse_checkpoints_dir() -> Path:
    """
    获取 SlideSparse checkpoints 目录
    
    Returns:
        checkpoints_slidesparse 目录的绝对路径
    """
    # 从项目根目录寻找
    project_root = Path(__file__).parent.parent
    slidesparse_dir = project_root / "checkpoints_slidesparse"
    
    if slidesparse_dir.exists():
        return slidesparse_dir
    
    # 尝试从当前工作目录
    cwd_dir = Path.cwd() / "checkpoints_slidesparse"
    if cwd_dir.exists():
        return cwd_dir
    
    # 返回默认路径（即使不存在）
    return slidesparse_dir


def resolve_slidesparse_model_path(
    base_model_path: Union[str, Path],
    sparsity: str = None,
    auto_convert: bool = True,
) -> Optional[Path]:
    """
    根据基础模型路径和稀疏配置，解析对应的 SlideSparse 模型路径
    
    命名约定:
        基础模型: checkpoints/Qwen2.5-0.5B-FP8/
        SlideSparse: checkpoints_slidesparse/Qwen2.5-0.5B-FP8-SlideSparse-2_8/
    
    Args:
        base_model_path: 基础模型路径（如 checkpoints/Qwen2.5-0.5B-FP8）
        sparsity: 稀疏配置（如 "2_8"），默认从环境变量读取
        auto_convert: 是否在找不到时自动转换（默认 True）
    
    Returns:
        SlideSparse 模型路径，如果不存在返回 None
    """
    base_path = Path(base_model_path)
    model_name = base_path.name  # e.g., "Qwen2.5-0.5B-FP8"
    
    if sparsity is None:
        sparsity = get_sparsity_str()
    
    # 构建 SlideSparse 模型名称
    slidesparse_name = f"{model_name}-SlideSparse-{sparsity}"
    slidesparse_path = get_slidesparse_checkpoints_dir() / slidesparse_name
    
    if slidesparse_path.exists() and slidesparse_path.is_dir():
        return slidesparse_path
    
    # 未找到，尝试自动转换
    if auto_convert and base_path.exists():
        converted = _try_auto_convert_specific_model(base_path, sparsity)
        if converted:
            return converted
    
    return None


def _try_auto_convert_specific_model(
    base_model_path: Path,
    sparsity: str,
) -> Optional[Path]:
    """
    尝试自动转换指定的基础模型为 SlideSparse 格式
    
    Args:
        base_model_path: 基础模型路径
        sparsity: 稀疏配置
    
    Returns:
        转换后的模型路径，失败返回 None
    """
    import sys
    import subprocess
    
    model_name = base_model_path.name
    
    print(f"\n{'='*70}")
    print(f"[SlideSparse] 未找到 {model_name}-SlideSparse-{sparsity} checkpoint")
    print(f"[SlideSparse] 发现基础模型: {model_name}")
    print(f"[SlideSparse] 自动转换中...")
    
    # 解析 sparsity
    parts = sparsity.split("_")
    if len(parts) != 2:
        print(f"[SlideSparse] 无效的稀疏配置: {sparsity}")
        return None
    
    Z, L = int(parts[0]), int(parts[1])
    
    # 调用 entry.py 进行转换
    project_root = Path(__file__).parent.parent
    entry_script = project_root / "slidesparse" / "weight_convert" / "entry.py"
    if not entry_script.exists():
        print(f"[SlideSparse] 转换脚本不存在: {entry_script}")
        return None
    
    print(f"[SlideSparse] 开始转换: {model_name} -> SlideSparse-{sparsity}")
    print(f"{'='*70}")
    
    try:
        # 使用 subprocess 调用转换脚本
        cmd = [
            sys.executable, str(entry_script),
            "--input", str(base_model_path),
            "--Z", str(Z),
            "--L", str(L),
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(entry_script.parent),
            capture_output=False,  # 显示输出
        )
        
        if result.returncode != 0:
            print(f"[SlideSparse] 转换失败 (exit code: {result.returncode})")
            return None
        
        # 转换成功，返回新路径
        slidesparse_dir = get_slidesparse_checkpoints_dir()
        expected_name = f"{model_name}-SlideSparse-{sparsity}"
        converted_path = slidesparse_dir / expected_name
        
        if converted_path.exists():
            print(f"\n[SlideSparse] 转换成功: {converted_path.name}")
            print(f"{'='*70}\n")
            return converted_path
        else:
            print(f"[SlideSparse] 转换完成但未找到输出目录: {expected_name}")
            return None
            
    except Exception as e:
        print(f"[SlideSparse] 转换异常: {e}")
        return None


def find_slidesparse_model(
    dtype: str = "FP8",
    sparsity: str = None,
    auto_convert: bool = True,
) -> Optional[Path]:
    """
    查找 SlideSparse 模型（优先选择较小的模型）
    
    如果未找到对应的 SlideSparse checkpoint 且 auto_convert=True，
    会尝试自动转换基础模型。
    
    搜索顺序: Qwen2.5-0.5B > Llama3.2-1B > Qwen2.5-1.5B > ...
    
    Args:
        dtype: 数据类型（"FP8" 或 "INT8"）
        sparsity: 稀疏配置（如 "2_8"），默认从环境变量读取
        auto_convert: 是否在找不到时自动转换（默认 True）
    
    Returns:
        找到的 SlideSparse 模型路径，如果未找到返回 None
    """
    if sparsity is None:
        sparsity = get_sparsity_str()
    
    slidesparse_dir = get_slidesparse_checkpoints_dir()
    
    # 搜索优先级（较小的模型优先）
    priority_patterns = [
        "Qwen2.5-0.5B",
        "Llama3.2-1B",
        "Qwen2.5-1.5B",
        "BitNet-2B",
        "Qwen2.5-3B",
        "Llama3.2-3B",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
    ]
    
    dtype_upper = dtype.upper()
    
    # 1. 先尝试查找已有的 SlideSparse checkpoint
    if slidesparse_dir.exists():
        for pattern in priority_patterns:
            # 构建预期的目录名
            expected_name = f"{pattern}-{dtype_upper}-SlideSparse-{sparsity}"
            model_path = slidesparse_dir / expected_name
            
            if model_path.exists() and model_path.is_dir():
                return model_path
        
        # 如果按优先级未找到，尝试模糊匹配
        for model_dir in slidesparse_dir.iterdir():
            if not model_dir.is_dir():
                continue
            name = model_dir.name
            if dtype_upper in name and f"SlideSparse-{sparsity}" in name:
                return model_dir
    
    # 2. 未找到，尝试自动转换
    if auto_convert:
        converted = _try_auto_convert_model(dtype, sparsity, priority_patterns)
        if converted:
            return converted
    
    return None


def _try_auto_convert_model(
    dtype: str,
    sparsity: str,
    priority_patterns: List[str],
) -> Optional[Path]:
    """
    尝试自动转换基础模型为 SlideSparse 格式
    
    Args:
        dtype: 数据类型
        sparsity: 稀疏配置
        priority_patterns: 优先搜索的模型模式
    
    Returns:
        转换后的模型路径，失败返回 None
    """
    import sys
    import subprocess
    
    # 获取基础 checkpoints 目录
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "checkpoints"
    
    if not checkpoints_dir.exists():
        return None
    
    dtype_upper = dtype.upper()
    
    # 查找可用的基础模型
    base_model_path = None
    base_model_name = None
    
    for pattern in priority_patterns:
        expected_name = f"{pattern}-{dtype_upper}"
        candidate = checkpoints_dir / expected_name
        if candidate.exists() and candidate.is_dir():
            base_model_path = candidate
            base_model_name = expected_name
            break
    
    if base_model_path is None:
        # 尝试模糊匹配
        for model_dir in checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            name = model_dir.name
            if dtype_upper in name:
                base_model_path = model_dir
                base_model_name = name
                break
    
    if base_model_path is None:
        return None
    
    # 找到基础模型，直接转换
    print(f"\n{'='*70}")
    print(f"[SlideSparse] 未找到 {dtype_upper} SlideSparse-{sparsity} checkpoint")
    print(f"[SlideSparse] 发现基础模型: {base_model_name}")
    print(f"[SlideSparse] 自动转换中...")
    
    # 解析 sparsity
    parts = sparsity.split("_")
    if len(parts) != 2:
        print(f"[SlideSparse] 无效的稀疏配置: {sparsity}")
        return None
    
    Z, L = int(parts[0]), int(parts[1])
    
    # 调用 entry.py 进行转换
    entry_script = project_root / "slidesparse" / "weight_convert" / "entry.py"
    if not entry_script.exists():
        print(f"[SlideSparse] 转换脚本不存在: {entry_script}")
        return None
    
    print(f"[SlideSparse] 开始转换: {base_model_name} -> SlideSparse-{sparsity}")
    print(f"{'='*70}")
    
    try:
        # 使用 subprocess 调用转换脚本
        cmd = [
            sys.executable, str(entry_script),
            "--input", str(base_model_path),
            "--Z", str(Z),
            "--L", str(L),
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(entry_script.parent),
            capture_output=False,  # 显示输出
        )
        
        if result.returncode != 0:
            print(f"[SlideSparse] 转换失败 (exit code: {result.returncode})")
            return None
        
        # 转换成功，返回新路径
        slidesparse_dir = get_slidesparse_checkpoints_dir()
        expected_name = f"{base_model_name}-SlideSparse-{sparsity}"
        converted_path = slidesparse_dir / expected_name
        
        if converted_path.exists():
            print(f"\n[SlideSparse] 转换成功: {converted_path.name}")
            print(f"{'='*70}\n")
            return converted_path
        else:
            print(f"[SlideSparse] 转换完成但未找到输出目录: {expected_name}")
            return None
            
    except Exception as e:
        print(f"[SlideSparse] 转换出错: {e}")
        return None


# =============================================================================
# 模型名称处理工具
# =============================================================================

# 量化后缀列表（用于识别和去除）
QUANT_SUFFIXES = ("-INT8", "-FP8", "-BF16", "-FP16")


def model_base_name(model_name: str) -> str:
    """
    提取模型的基础名称（去除量化后缀 -INT8/-FP8/-BF16 等）
    
    INT8 和 FP8 模型共享相同的结构（N, K 维度），只是量化方式不同。
    此函数用于获取不带量化后缀的 base name。
    
    示例:
        >>> model_base_name("Qwen2.5-0.5B-INT8")
        'Qwen2.5-0.5B'
        >>> model_base_name("Llama3.2-1B-FP8")
        'Llama3.2-1B'
        >>> model_base_name("Qwen2.5-0.5B")  # 已经是 base name
        'Qwen2.5-0.5B'
        >>> model_base_name("BitNet-2B-BF16")
        'BitNet-2B'
    
    Args:
        model_name: 模型名称（可能带有量化后缀）
    
    Returns:
        不带量化后缀的基础模型名
    """
    name = model_name
    for suffix in QUANT_SUFFIXES:
        if name.upper().endswith(suffix):
            return name[:-len(suffix)]
    return name


def model_quant_suffix(model_name: str) -> Optional[str]:
    """
    提取模型名称中的量化后缀
    
    示例:
        >>> model_quant_suffix("Qwen2.5-0.5B-INT8")
        'INT8'
        >>> model_quant_suffix("Llama3.2-1B-FP8")
        'FP8'
        >>> model_quant_suffix("Qwen2.5-0.5B")  # 无后缀
        None
    
    Args:
        model_name: 模型名称
    
    Returns:
        量化后缀（不带 -），如 "INT8"、"FP8"，如果没有则返回 None
    """
    for suffix in QUANT_SUFFIXES:
        if model_name.upper().endswith(suffix):
            return suffix[1:]  # 去掉开头的 -
    return None


def find_model_checkpoint_for_dtype(
    base_name: str,
    dtype: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    根据 base name 和 dtype 查找对应的 checkpoint 目录
    
    Args:
        base_name: 模型基础名称（不带量化后缀），如 "Qwen2.5-0.5B"
        dtype: 数据类型，"int8" 或 "fp8"
        checkpoint_dir: checkpoint 目录（默认 PROJECT_ROOT/checkpoints）
    
    Returns:
        找到的 checkpoint 目录 Path，未找到则返回 None
    
    示例:
        >>> find_model_checkpoint_for_dtype("Qwen2.5-0.5B", "int8")
        Path('/path/to/checkpoints/Qwen2.5-0.5B-INT8')
    """
    if checkpoint_dir is None:
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    # 构建目标名称
    suffix = "INT8" if dtype.lower() == "int8" else "FP8"
    target_name = f"{base_name}-{suffix}"
    target_path = checkpoint_dir / target_name
    
    if target_path.exists() and target_path.is_dir():
        return target_path
    return None


def find_any_model_checkpoint(
    base_name: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    根据 base name 查找任意一个存在的 checkpoint 目录（INT8 或 FP8）
    
    由于 INT8 和 FP8 模型的 NK 配置相同，获取 NK 时只需找到任一存在的目录即可。
    
    Args:
        base_name: 模型基础名称（不带量化后缀），如 "Qwen2.5-0.5B"
        checkpoint_dir: checkpoint 目录（默认 PROJECT_ROOT/checkpoints）
    
    Returns:
        (path, full_name): 找到的 checkpoint 目录和完整名称，未找到则返回 (None, None)
    
    示例:
        >>> find_any_model_checkpoint("Qwen2.5-0.5B")
        (Path('/path/to/checkpoints/Qwen2.5-0.5B-INT8'), 'Qwen2.5-0.5B-INT8')
    """
    if checkpoint_dir is None:
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    # 优先尝试 INT8，然后 FP8，最后 BF16
    for suffix in ["INT8", "FP8", "BF16"]:
        target_name = f"{base_name}-{suffix}"
        target_path = checkpoint_dir / target_name
        if target_path.exists() and target_path.is_dir():
            return target_path, target_name
    
    return None, None


def normalize_model_input(
    model_input: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, Optional[str]]:
    """
    标准化用户输入的模型名称
    
    无论用户输入的是 base name 还是带后缀的完整名称，都提取出：
    1. base name（用于查找 NK 配置）
    2. 用户指定的量化类型（如果有）
    
    同时验证模型是否存在（至少有一个对应的 checkpoint 目录）。
    
    Args:
        model_input: 用户输入的模型名称
        checkpoint_dir: checkpoint 目录
    
    Returns:
        (base_name, quant_hint): 
            - base_name: 模型基础名称
            - quant_hint: 用户输入中包含的量化类型（"int8"/"fp8"），或 None
    
    Raises:
        ValueError: 找不到对应的 checkpoint 目录
    
    示例:
        >>> normalize_model_input("Qwen2.5-0.5B-INT8")
        ('Qwen2.5-0.5B', 'int8')
        >>> normalize_model_input("Qwen2.5-0.5B")
        ('Qwen2.5-0.5B', None)
    """
    # 提取 base name 和量化后缀
    base = model_base_name(model_input)
    quant = model_quant_suffix(model_input)
    quant_lower = quant.lower() if quant else None
    
    # 验证模型存在
    path, _ = find_any_model_checkpoint(base, checkpoint_dir)
    if path is None:
        # 列出可用模型
        if checkpoint_dir is None:
            project_root = Path(__file__).parent.parent
            checkpoint_dir = project_root / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        available = []
        if checkpoint_dir.exists():
            # 收集所有 base name（去重）
            seen_bases = set()
            for d in sorted(checkpoint_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    b = model_base_name(d.name)
                    if b not in seen_bases:
                        seen_bases.add(b)
                        available.append(b)
        
        error_msg = f"未找到模型 '{model_input}'（base name: '{base}'）"
        if available:
            error_msg += f"\n可用的模型: {', '.join(available[:10])}"
        raise ValueError(error_msg)
    
    return base, quant_lower


def extract_model_name(model_name_with_slide: str) -> str:
    """
    从完整模型名中提取基础模型名（去除 -SlideSparse-Z_L 后缀）
    
    这是全局统一的模型名提取函数，供 core/ 和其他模块使用。
    
    命名约定:
        - model_name: 基础模型名，严格不带 -SlideSparse- 后缀
        - model_name_with_slide: 可能包含 -SlideSparse- 后缀的完整 checkpoint 名
    
    示例:
        >>> extract_model_name("Llama3.2-1B-FP8-SlideSparse-2_8")
        'Llama3.2-1B-FP8'
        >>> extract_model_name("Qwen2.5-0.5B-INT8")
        'Qwen2.5-0.5B-INT8'
        >>> extract_model_name("BitNet-2B-BF16-SlideSparse-2_10")
        'BitNet-2B-BF16'
    
    Args:
        model_name_with_slide: 可能包含 -SlideSparse- 后缀的模型名
    
    Returns:
        不带 slide 后缀的基础模型名
    """
    marker = "-SlideSparse-"
    if marker in model_name_with_slide:
        return model_name_with_slide.split(marker)[0]
    return model_name_with_slide


def resolve_model_path_from_arg(
    model_arg: Optional[str],
    quant: Optional[str] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    fallback_model: str = "BitNet-2B-BF16",
    warn_on_fallback: bool = True,
) -> Tuple[Path, str]:
    """
    从命令行参数解析模型路径（统一接口）
    
    供测试脚本、工具脚本统一使用。解决各处重复实现模型查找逻辑的问题。
    
    支持的输入格式:
      - None → 使用 fallback_model 并打印警告
      - 目录名 (如 "Qwen2.5-0.5B-FP8") → 在 checkpoint_dir 下查找
      - 完整路径 (如 "/path/to/model") → 直接使用
      - 不带量化后缀 (如 "Llama3.2-1B") → 需指定 quant 参数补全为 "Llama3.2-1B-FP8"
    
    Args:
        model_arg: 命令行 --model 参数值
        quant: 量化类型（用于补全，如 "FP8", "INT8"），默认 None
        checkpoint_dir: checkpoint 目录，默认 PROJECT_ROOT/checkpoints
        fallback_model: 当 model_arg 为 None 时的默认模型名
        warn_on_fallback: 是否在使用默认模型时打印警告
    
    Returns:
        (model_path, model_name): 
            - model_path: 模型目录的 Path 对象
            - model_name: 模型名称（目录名，如 "Qwen2.5-0.5B-FP8"）
    
    Raises:
        ValueError: 模型目录不存在
    
    Example:
        >>> # 使用默认模型（会打印警告）
        >>> path, name = resolve_model_path_from_arg(None)
        >>> print(name)  # "BitNet-2B-BF16"
        
        >>> # 指定模型名
        >>> path, name = resolve_model_path_from_arg("Qwen2.5-0.5B-FP8")
        
        >>> # 补全量化后缀
        >>> path, name = resolve_model_path_from_arg("Llama3.2-1B", quant="FP8")
    """
    # 默认 checkpoint 目录
    if checkpoint_dir is None:
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    # 如果未指定模型，使用 fallback
    if model_arg is None:
        if warn_on_fallback:
            print(f"[Warning] 未指定 --model，使用默认模型: {fallback_model}")
        
        model_path = checkpoint_dir / fallback_model
        if not model_path.exists():
            raise ValueError(
                f"默认模型 '{fallback_model}' 不存在。\n"
                f"请下载模型或使用 --model 参数指定其他模型。\n"
                f"期望路径: {model_path}"
            )
        return model_path, fallback_model
    
    # 1. 尝试作为完整路径
    direct_path = Path(model_arg)
    if direct_path.exists() and direct_path.is_dir():
        return direct_path, direct_path.name
    
    # 2. 在 checkpoint_dir 下查找
    candidates = [
        checkpoint_dir / model_arg,
    ]
    
    # 如果指定了 quant，尝试补全
    if quant:
        quant_upper = quant.upper()
        # 避免重复添加后缀
        if not model_arg.upper().endswith(f"-{quant_upper}"):
            candidates.append(checkpoint_dir / f"{model_arg}-{quant_upper}")
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name
    
    # 未找到，生成友好的错误信息
    available_models = []
    if checkpoint_dir.exists():
        available_models = sorted([
            d.name for d in checkpoint_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
    
    error_msg = f"未找到模型 '{model_arg}'。"
    if candidates:
        error_msg += f"\n尝试的路径: {[str(c) for c in candidates]}"
    if available_models:
        error_msg += f"\n可用的模型: {', '.join(available_models[:10])}"
        if len(available_models) > 10:
            error_msg += f" ... (共 {len(available_models)} 个)"
    else:
        error_msg += f"\ncheckpoints 目录 ({checkpoint_dir}) 为空或不存在。"
    
    raise ValueError(error_msg)


# =============================================================================
# 统一的 NK 尺寸获取工具（供搜索/调优脚本使用）
# =============================================================================

def get_nk_list_for_search(
    model: Optional[str] = None,
    L_max: Optional[int] = None,
    checkpoints_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[int, int]], str]:
    """
    获取用于搜索/调优的 NK 尺寸列表（统一接口）
    
    供 7 个搜索/调优脚本统一使用：
    - 3 个 Triton autotune: dequant_bias, quant_slide, quant_only
    - 4 个 CUDA search: cuBLASLt_AlgSearch/LayoutSearch, cuSPARSELt_AlgSearch/LayoutSearch
    
    策略:
    - model=None: 返回 BitNet-2B-BF16 默认 NK（打印警告）
    - L_max=None: 使用 get_model_nk_sizes 获取原始 NK
    - L_max 指定: 使用 get_model_nk_sizes_slided 获取 L=4 到 L=L_max 的所有 NK
    
    Args:
        model: 模型名称（如 "BitNet-2B-BF16", "Qwen2.5-7B-FP8"）或完整路径。
               必须与 checkpoints/ 目录下的文件夹名完全匹配。
               如果为 None，使用 BitNet-2B-BF16 默认配置（会打印警告）。
        L_max: 最大 L 值，用于 slide 稀疏。如果指定，会生成 L=4,6,8,...,L_max 的所有 NK
        checkpoints_dir: 自定义 checkpoints 目录路径（可选，默认使用项目根目录下的 checkpoints/）
    
    Returns:
        (nk_list, model_name): 
            - nk_list: [(N, K), ...] 列表，去重后的结果
            - model_name: 用于文件命名的模型名称（从找到的目录名提取）
    
    Raises:
        ValueError: 未找到模型目录、无法提取 NK 尺寸、或 NK 列表为空
    
    Example:
        >>> # 使用默认配置（会打印警告）
        >>> nk_list, name = get_nk_list_for_search()
        >>> print(name)  # "BitNet-2B-BF16"
        
        >>> # 从模型获取原始 NK
        >>> nk_list, name = get_nk_list_for_search("Qwen2.5-7B-FP8")
        >>> print(name)  # "Qwen2.5-7B-FP8"
        
        >>> # 从模型获取 slide 后的 NK (L=4 到 L=10)
        >>> nk_list, name = get_nk_list_for_search("BitNet-2B-BF16", L_max=10)
    """
    # BitNet-2B-BF16 默认 NK 尺寸
    DEFAULT_MODEL_NAME = "BitNet-2B-BF16"
    DEFAULT_NK = [
        (3840, 2560),   # qkv_proj (Wqkv)
        (2560, 2560),   # o_proj (Wo)
        (13824, 2560),  # gate_up_proj (W13)
        (2560, 6912),   # down_proj (W2)
    ]
    
    # 如果未指定模型，返回默认配置
    if model is None:
        print(f"[Warning] 未指定 --model，使用默认配置: {DEFAULT_MODEL_NAME}")
        return DEFAULT_NK, DEFAULT_MODEL_NAME
    
    # 构建搜索路径列表
    project_root = Path(__file__).parent.parent
    if checkpoints_dir:
        ckpt_dir = Path(checkpoints_dir)
    else:
        ckpt_dir = project_root / "checkpoints"
    
    search_paths = []
    
    # 1. 直接路径（用户可能传入绝对或相对路径）
    direct_path = Path(model)
    if direct_path.exists() and direct_path.is_dir():
        search_paths.append(direct_path)
    
    # 2. checkpoints 目录下精确匹配
    search_paths.append(ckpt_dir / model)
    
    # 3. 模糊匹配：如果 model 是 base name（不带后缀），尝试添加 -INT8/-FP8 后缀
    base = model_base_name(model)
    if base == model:  # model 本身就是 base name（没有量化后缀）
        for suffix in ["INT8", "FP8", "BF16"]:
            search_paths.append(ckpt_dir / f"{model}-{suffix}")
    
    # 查找有效路径
    model_path = None
    for path in search_paths:
        if path.exists() and path.is_dir():
            model_path = path
            break
    
    if model_path is None:
        # 列出可用的模型目录帮助用户（显示 base name）
        available_models = []
        if ckpt_dir.exists():
            seen_bases = set()
            for d in sorted(ckpt_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    b = model_base_name(d.name)
                    if b not in seen_bases:
                        seen_bases.add(b)
                        available_models.append(b)
        
        error_msg = f"未找到模型 '{model}'。"
        if available_models:
            error_msg += f"\n可用的模型 (base name): {', '.join(available_models)}"
        else:
            error_msg += f"\ncheckpoints 目录 ({ckpt_dir}) 为空或不存在。"
        raise ValueError(error_msg)
    
    # 提取模型名称（用于文件命名，使用实际找到的目录名）
    model_name = model_path.name
    
    # 获取原始 NK 尺寸
    try:
        nk_sizes = get_model_nk_sizes(model_path)
    except Exception as e:
        raise ValueError(f"无法从 '{model_path}' 提取 NK 尺寸: {e}")
    
    if not nk_sizes:
        raise ValueError(f"模型 '{model_name}' 的 NK 尺寸为空，请检查模型文件是否完整。")
    
    # 转换为列表格式
    nk_list = []
    layer_order = ["qkv", "wo", "w13", "w2"]
    
    if L_max is None:
        # 不使用 slide，直接返回原始 NK
        for layer_type in layer_order:
            if layer_type in nk_sizes:
                N, K = nk_sizes[layer_type]
                if (N, K) not in nk_list:
                    nk_list.append((N, K))
    else:
        # 使用 slide，生成 L=4 到 L=L_max 的所有 NK
        # L 必须是偶数且 >= 4
        if L_max < 4:
            raise ValueError(f"L_max 必须 >= 4，当前值: {L_max}")
        
        L_values = list(range(4, L_max + 1, 2))
        
        for L in L_values:
            slided_sizes = get_model_nk_sizes_slided(nk_sizes, Z=2, L=L, align_to=32)
            for layer_type in layer_order:
                if layer_type in slided_sizes:
                    N, K = slided_sizes[layer_type]
                    if (N, K) not in nk_list:
                        nk_list.append((N, K))
    
    if not nk_list:
        raise ValueError(f"模型 '{model_name}' 生成的 NK 列表为空。")
    
    return nk_list, model_name


def get_unique_n_values(nk_list: List[Tuple[int, int]]) -> List[int]:
    """
    从 NK 列表中提取唯一的 N 值（用于 Triton dequant 等按 N 维度调优的 kernel）
    
    Args:
        nk_list: [(N, K), ...] 列表
    
    Returns:
        唯一的 N 值列表，已排序
    """
    return sorted(set(N for N, K in nk_list))


def get_unique_k_values(nk_list: List[Tuple[int, int]]) -> List[int]:
    """
    从 NK 列表中提取唯一的 K 值（用于 Triton quant 等按 K 维度调优的 kernel）
    
    Args:
        nk_list: [(N, K), ...] 列表
    
    Returns:
        唯一的 K 值列表，已排序
    """
    return sorted(set(K for N, K in nk_list))


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 全局默认配置
    "DEFAULT_M_LIST",
    "M_QUICK_LIST",
    
    # 数据类型
    "normalize_dtype",
    "DTYPE_ALIASES",
    
    # =========================================================================
    # PART 1: CUDA 编译、链接、库加载工具
    # =========================================================================
    
    # 系统库路径
    "SYSTEM_LIB_PATHS",
    "get_system_lib_path",
    
    # NVCC 架构标志
    "SUPPORTED_ARCHITECTURES",
    "get_nvcc_arch_flags",
    "get_current_arch_flag",
    
    # 链接库配置
    "SUPPORTED_BACKENDS",
    "get_backend_ldflags",
    "BACKEND_LDFLAGS",
    "CUBLASLT_LDFLAGS",
    "CUSPARSELT_LDFLAGS",
    
    # 运行时库加载
    "ensure_cublaslt_loaded",
    "ensure_cusparselt_loaded",
    "BACKEND_LOADERS",
    
    # 编译辅助
    "DEFAULT_CFLAGS",
    "DEFAULT_CUDA_CFLAGS",
    "should_rebuild",
    "clean_build_artifacts",
    
    # 编译函数
    "build_cuda_extension",       # PyTorch 扩展编译
    "build_cuda_extension_direct", # 直接 nvcc 编译
    "load_cuda_extension",        # 高级加载接口
    
    # =========================================================================
    # PART 2: 硬件信息
    # =========================================================================
    "HardwareInfo",
    "hw_info",
    # 便捷函数
    "get_gpu_name",
    "get_gpu_cc",
    "get_python_version_tag",
    "get_cuda_ver",
    "get_arch_tag",
    "get_sm_code",
    "print_system_info",
    
    # =========================================================================
    # PART 3: 文件名与 IO
    # =========================================================================
    # 文件名构建
    "build_filename",
    "build_stem",
    "build_dir_name",
    "build_hw_dir_name",
    "build_tuned_filename",
    # 文件查找
    "find_file",
    "find_files",
    "find_dir",
    # 模块加载
    "load_module",
    "clear_module_cache",
    "load_tuned_module",
    "clear_tuned_module_cache",
    # 算法查表
    "lookup_best_cublaslt_alg",
    "decode_cublaslt_algo_data",
    "lookup_best_cusparselt_alg",
    # 数据保存/加载
    "save_json",
    "load_json",
    "save_csv",
    # 目录管理
    "ensure_result_dir",
    
    # =========================================================================
    # PART 4: 模型信息管理
    # =========================================================================
    "MODEL_SIZE_GB",
    "ModelEntry",
    "ModelRegistry",
    "model_registry",
    "get_model_registry",
    "get_model_info",
    "list_models",
    "build_model_dir_name",
    "parse_model_key",
    "check_quant_support",
    "get_model_local_path",
    "check_model_downloaded",
    
    # =========================================================================
    # PART 5: SlideSparse 配置与维度计算
    # =========================================================================
    "SlideSparseConfig",
    "compute_output_k",
    "compute_compressed_k",
    # 模型 NK Size 工具
    "LINEAR_LAYER_TYPES",
    "get_model_nk_sizes",
    "get_model_nk_sizes_slided",
    "get_model_nk_sizes_compressed",
    "print_model_nk_summary",
    # 统一的 NK 获取工具（供搜索/调优脚本使用）
    "get_nk_list_for_search",
    "get_unique_n_values",
    "get_unique_k_values",
    # 稀疏配置解析
    "parse_sparsity_env",
    "get_sparsity_config_cached",
    "clear_sparsity_cache",
    "get_sparsity_str",
    # SlideSparse 模型路径解析
    "get_slidesparse_checkpoints_dir",
    "resolve_slidesparse_model_path",
    "find_slidesparse_model",
    # 模型名称处理工具
    "QUANT_SUFFIXES",
    "model_base_name",
    "model_quant_suffix",
    "find_model_checkpoint_for_dtype",
    "find_any_model_checkpoint",
    "normalize_model_input",
    "extract_model_name",
    "resolve_model_path_from_arg",
]


# =============================================================================
# CLI
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SlideSparse 统一工具库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 显示硬件信息
    python -m slidesparse.utils info
    
    # 生成文件名
    python -m slidesparse.utils name cuBLASLt --dtype FP8E4M3 --ext .so
    
    # 查找文件
    python -m slidesparse.utils find cuBLASLt --dtype FP8E4M3 --dir build
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="显示硬件信息")
    info_parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    
    # name 命令
    name_parser = subparsers.add_parser("name", help="生成文件名")
    name_parser.add_argument("prefix", help="文件前缀")
    name_parser.add_argument("--dtype", help="数据类型")
    name_parser.add_argument("--ext", default="", help="文件扩展名")
    
    # find 命令
    find_parser = subparsers.add_parser("find", help="查找文件")
    find_parser.add_argument("prefix", help="文件前缀")
    find_parser.add_argument("--dtype", help="数据类型")
    find_parser.add_argument("--dir", default=".", help="搜索目录")
    find_parser.add_argument("--ext", help="文件扩展名")
    
    args = parser.parse_args()
    
    if args.command == "info":
        if args.json:
            import json
            print(json.dumps(hw_info.to_dict(), indent=2, ensure_ascii=False))
        else:
            hw_info.print_info()
    
    elif args.command == "name":
        name = build_filename(args.prefix, dtype=args.dtype, ext=args.ext)
        print(name)
    
    elif args.command == "find":
        result = find_file(args.prefix, dtype=args.dtype, search_dir=args.dir, ext=args.ext)
        if result:
            print(result)
        else:
            print(f"未找到匹配的文件", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
