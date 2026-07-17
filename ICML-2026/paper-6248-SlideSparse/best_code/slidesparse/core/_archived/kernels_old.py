# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Triton Kernel 加载模块

包含:
- Dequant + Bias kernel（FP8/INT8 共享）
- FP8 Quant kernels（quant_only, quant_slide）
- INT8 Quant kernels（quant_only, quant_slide）

torch.compile 兼容策略（方案 1+4）:
=================================
1. 模块导入时预加载：扫描 build 目录，预加载所有已编译的 tuned kernel
2. 编译期保护：如果缓存未命中且处于 torch.compile 追踪期间，抛出明确错误

这确保:
- 子进程导入模块时自动完成预加载
- 热路径上只有字典读取，无文件系统操作
- 边界情况（如新模型未预编译）给出友好错误提示

目录结构:
    build/{hw_dir_name}/{kernel_name}_tuned_{model_name}.py
    
例如:
    build/RTX5080_cc120_py312_cu129_x86_64/dequant_bias_tuned_Llama3.2-1B-FP8.py
"""

from pathlib import Path
from typing import Callable, Dict, Optional
import os

import torch
from torch.library import Library
from vllm.logger import init_logger
from vllm.platforms import current_platform

from slidesparse.utils import load_tuned_module, build_hw_dir_name

logger = init_logger(__name__)

# Custom Op Library（与 gemm_wrapper 共享同一个 library）
# 注意：这里用 FRAGMENT 模式追加到已有的 slidesparse library
_triton_lib = Library("slidesparse", "FRAGMENT")


# ============================================================================
# 目录配置
# ============================================================================

_CSRC_DIR = Path(__file__).parent.parent / "csrc"


# ============================================================================
# 模型名辅助函数
# ============================================================================

def _extract_base_model_name(model_name: str) -> str:
    """
    从完整模型名中提取基础模型名
    
    例如:
        Llama3.2-1B-FP8-SlideSparse-2_8 -> Llama3.2-1B-FP8
        Qwen2.5-0.5B-INT8-SlideSparse-2_10 -> Qwen2.5-0.5B-INT8
        Llama3.2-1B-FP8 -> Llama3.2-1B-FP8 (不变)
    """
    marker = "-SlideSparse-"
    if marker in model_name:
        return model_name.split(marker)[0]
    return model_name


# ============================================================================
# Kernel 搜索配置
# ============================================================================

# Basic kernel 文件名映射
_BASIC_KERNEL_FILES = {
    "dequant_bias": "basic_dequant_bias_triton.py",
    "quant_only":   "basic_quant_only_triton.py",
    "quant_slide":  "basic_quant_slide_triton.py",
}


def _load_basic_kernel(kernel_dir: Path, kernel_type: str) -> object:
    """加载 basic kernel 模块（无 model-specific tuning）"""
    if kernel_type not in _BASIC_KERNEL_FILES:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    basic_file = kernel_dir / _BASIC_KERNEL_FILES[kernel_type]
    if not basic_file.exists():
        raise FileNotFoundError(f"Basic kernel not found: {basic_file}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(kernel_type, basic_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger.info_once(f"Using basic kernel: {basic_file.name}")
    return module


def _search_kernel(
    kernel_dir: Path,
    kernel_type: str,
    tuned_prefix: str,
    model_name: str,
) -> object:
    """
    搜索并加载 kernel 模块
    
    搜索顺序:
    1. 首先尝试加载 tuned kernel: build/{hw_dir}/{tuned_prefix}_{base_model}.py
    2. 如果找不到，fallback 到 basic kernel: basic_{kernel_type}_triton.py
    
    Args:
        kernel_dir: kernel 目录（如 csrc/fused_dequant_bias_triton）
        kernel_type: kernel 类型（dequant_bias, quant_only, quant_slide）
        tuned_prefix: tuned 文件前缀（如 dequant_bias_tuned）
        model_name: 模型名称（可能包含 -SlideSparse-2_L 后缀）
    
    Returns:
        加载的模块对象
    """
    build_dir = kernel_dir / "build"
    
    # 提取基础模型名用于查找 tuned kernel
    base_model = _extract_base_model_name(model_name)
    
    # 1. 尝试加载 tuned kernel
    try:
        module = load_tuned_module(tuned_prefix, base_model, build_dir)
        if base_model != model_name:
            logger.info_once(f"Loaded tuned kernel for base model: {base_model} (from {model_name})")
        else:
            logger.info_once(f"Loaded tuned kernel for model: {model_name}")
        return module
    except FileNotFoundError:
        pass
    
    # 2. Fallback 到 basic kernel
    logger.warning(f"Tuned kernel not found for {base_model}, using basic kernel")
    return _load_basic_kernel(kernel_dir, kernel_type)


# ============================================================================
# Dequant + Bias Kernel（FP8/INT8 共享）
# ============================================================================

# 缓存: model_name -> kernel function
_dequant_bias_cache: Dict[str, Callable] = {}


def _load_dequant_bias_kernel(model_name: str) -> Callable:
    """加载 Triton dequant+bias kernel（按 model 懒加载）"""
    if model_name in _dequant_bias_cache:
        return _dequant_bias_cache[model_name]
    
    # 编译期保护：如果缓存未命中且在 torch.compile 追踪期间，抛出明确错误
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'dequant_bias' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_dequant_bias_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="dequant_bias",
        tuned_prefix="dequant_bias_tuned",
        model_name=model_name,
    )
    
    fn = module.dequant_bias_triton
    # 修复模块名，避免 Dynamo 尝试 import 非法名称（如 Llama3.2-1B-FP8 中的 . 会被解析为包分隔符）
    fn.__module__ = "slidesparse.core.kernels"
    _dequant_bias_cache[model_name] = fn
    logger.info_once(f"Dequant+bias kernel loaded for model: {model_name}")
    return fn


def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    model_name: str,
) -> torch.Tensor:
    """
    Dequant + Bias: output = gemm_output * scale_a * scale_b + bias
    
    支持 BF16、FP32、INT32 输入（自动检测）
    通过 torch.library custom op 调用，确保 torch.compile 兼容。
    
    Args:
        gemm_output: GEMM 输出
        scale_a: 激活 scale
        scale_b: 权重 scale  
        bias: 可选的 bias
        out_dtype: 输出类型
        model_name: 模型名称（用于加载对应的 tuned kernel）
    """
    # 将 torch.dtype 转换为字符串（custom op schema 要求）
    out_dtype_str = str(out_dtype).replace("torch.", "")
    return torch.ops.slidesparse.dequant_bias(
        gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name
    )


# ============================================================================
# FP8 Quant Only Kernel - cuBLASLt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_only_fp8_cache: Dict[str, Callable] = {}


def _load_quant_only_fp8_kernel(model_name: str) -> Callable:
    """加载 Triton FP8 quant kernel（按 model 懒加载）"""
    if model_name in _quant_only_fp8_cache:
        return _quant_only_fp8_cache[model_name]
    
    # 编译期保护
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_only_fp8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_fp8_triton
    # 修复模块名，避免 Dynamo 尝试 import 非法名称
    fn.__module__ = "slidesparse.core.kernels"
    _quant_only_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization
    
    通过 torch.library custom op 调用，确保 torch.compile 兼容。
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        
    Returns:
        qinput: [M_pad, K_pad] FP8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    return torch.ops.slidesparse.quant_only_fp8(input, model_name)


# ============================================================================
# FP8 Quant + Slide Kernel - cuSPARSELt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_slide_fp8_cache: Dict[str, Callable] = {}


def _load_quant_slide_fp8_kernel(model_name: str) -> Callable:
    """加载 Triton FP8 quant+slide kernel（按 model 懒加载）"""
    if model_name in _quant_slide_fp8_cache:
        return _quant_slide_fp8_cache[model_name]
    
    # 编译期保护
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_slide_fp8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_fp8_triton
    # 修复模块名，避免 Dynamo 尝试 import 非法名称
    fn.__module__ = "slidesparse.core.kernels"
    _quant_slide_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization + SlideSparse Slide
    
    通过 torch.library custom op 调用，确保 torch.compile 兼容。
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] FP8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
                K_slide = K * (L - 2) / (L / 2) = K * 2 * (L - 2) / L
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    return torch.ops.slidesparse.quant_slide_fp8(input, model_name, L)


# ============================================================================
# INT8 Quant Only Kernel - cuBLASLt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_only_int8_cache: Dict[str, Callable] = {}


def _load_quant_only_int8_kernel(model_name: str) -> Callable:
    """加载 Triton INT8 quant kernel（按 model 懒加载）"""
    if model_name in _quant_only_int8_cache:
        return _quant_only_int8_cache[model_name]
    
    # 编译期保护
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_only_int8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_int8_triton
    # 修复模块名，避免 Dynamo 尝试 import 非法名称
    fn.__module__ = "slidesparse.core.kernels"
    _quant_only_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_int8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization (Symmetric)
    
    通过 torch.library custom op 调用，确保 torch.compile 兼容。
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        
    Returns:
        qinput: [M_pad, K_pad] INT8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    return torch.ops.slidesparse.quant_only_int8(input, model_name)


# ============================================================================
# INT8 Quant + Slide Kernel - cuSPARSELt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_slide_int8_cache: Dict[str, Callable] = {}


def _load_quant_slide_int8_kernel(model_name: str) -> Callable:
    """加载 Triton INT8 quant+slide kernel（按 model 懒加载）"""
    if model_name in _quant_slide_int8_cache:
        return _quant_slide_int8_cache[model_name]
    
    # 编译期保护
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Triton kernel 'quant_slide_int8' for model '{model_name}' was not preloaded!\n"
            f"This error occurs during torch.compile tracing.\n"
            f"Possible causes:\n"
            f"  1. Tuned kernel not found in build directory\n"
            f"  2. init_slidesparse('{model_name}') was not called before model loading\n"
            f"Fix: Run kernel autotuning for this model, or check SLIDESPARSE_MODEL_NAME env var."
        )
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_int8_triton
    # 修复模块名，避免 Dynamo 尝试 import 非法名称
    fn.__module__ = "slidesparse.core.kernels"
    _quant_slide_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_int8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization + SlideSparse Slide (Symmetric)
    
    通过 torch.library custom op 调用，确保 torch.compile 兼容。
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] INT8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    return torch.ops.slidesparse.quant_slide_int8(input, model_name, L)


# ============================================================================
# Custom Op 注册（torch.compile 兼容）
# ============================================================================
#
# 为 Triton kernel 注册 torch.library custom op：
# - 实际实现：调用预加载的 Triton kernel
# - fake 实现：返回正确形状的空 tensor，用于 Dynamo 追踪
# ============================================================================

def _ceil16(x: int) -> int:
    """向上取整到 16 的倍数"""
    return (x + 15) // 16 * 16


def _ceil32(x: int) -> int:
    """向上取整到 32 的倍数"""
    return (x + 31) // 32 * 32


def _register_triton_custom_ops():
    """注册所有 Triton kernel 的 custom op"""
    
    # ========== dequant_bias ==========
    _triton_lib.define(
        "dequant_bias(Tensor gemm_output, Tensor scale_a, Tensor scale_b, "
        "Tensor? bias, str out_dtype_str, str model_name) -> Tensor"
    )
    
    # dtype 字符串映射
    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "int32": torch.int32,
        "int": torch.int32,
    }
    
    def _dequant_bias_impl(gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name):
        if bias is None:
            # 在 Graph 内部创建 dummy bias，确保 CUDAGraph 正确管理其生命周期
            bias = torch.zeros(
                gemm_output.shape[1],
                dtype=gemm_output.dtype,
                device=gemm_output.device
            )
        fn = _load_dequant_bias_kernel(model_name)
        out_dtype = _dtype_map.get(out_dtype_str, torch.bfloat16)
        return fn(gemm_output, scale_a, scale_b, bias, out_dtype)
    
    def _dequant_bias_fake(gemm_output, scale_a, scale_b, bias, out_dtype_str, model_name):
        M, N = gemm_output.shape
        out_dtype = _dtype_map.get(out_dtype_str, torch.bfloat16)
        return torch.empty((M, N), dtype=out_dtype, device=gemm_output.device)
    
    _triton_lib.impl("dequant_bias", _dequant_bias_impl, "CUDA")
    _triton_lib._register_fake("dequant_bias", _dequant_bias_fake)
    
    # ========== quant_only_fp8 ==========
    _triton_lib.define(
        "quant_only_fp8(Tensor input, str model_name) -> (Tensor, Tensor)"
    )
    
    def _quant_only_fp8_impl(input, model_name):
        fn = _load_quant_only_fp8_kernel(model_name)
        return fn(input)
    
    def _quant_only_fp8_fake(input, model_name):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_pad = _ceil32(K)
        fp8_dtype = current_platform.fp8_dtype()
        qinput = torch.empty((M_pad, K_pad), dtype=fp8_dtype, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_only_fp8", _quant_only_fp8_impl, "CUDA")
    _triton_lib._register_fake("quant_only_fp8", _quant_only_fp8_fake)
    
    # ========== quant_slide_fp8 ==========
    _triton_lib.define(
        "quant_slide_fp8(Tensor input, str model_name, int L) -> (Tensor, Tensor)"
    )
    
    def _quant_slide_fp8_impl(input, model_name, L):
        fn = _load_quant_slide_fp8_kernel(model_name)
        return fn(input, L)
    
    def _quant_slide_fp8_fake(input, model_name, L):
        M, K = input.shape
        M_pad = _ceil16(M)
        # K_slide = K * 2 * (L - 2) / L
        K_slide = K * 2 * (L - 2) // L
        K_slide_pad = _ceil32(K_slide)
        fp8_dtype = current_platform.fp8_dtype()
        qinput = torch.empty((M_pad, K_slide_pad), dtype=fp8_dtype, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_slide_fp8", _quant_slide_fp8_impl, "CUDA")
    _triton_lib._register_fake("quant_slide_fp8", _quant_slide_fp8_fake)
    
    # ========== quant_only_int8 ==========
    _triton_lib.define(
        "quant_only_int8(Tensor input, str model_name) -> (Tensor, Tensor)"
    )
    
    def _quant_only_int8_impl(input, model_name):
        fn = _load_quant_only_int8_kernel(model_name)
        return fn(input)
    
    def _quant_only_int8_fake(input, model_name):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_pad = _ceil32(K)
        qinput = torch.empty((M_pad, K_pad), dtype=torch.int8, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_only_int8", _quant_only_int8_impl, "CUDA")
    _triton_lib._register_fake("quant_only_int8", _quant_only_int8_fake)
    
    # ========== quant_slide_int8 ==========
    _triton_lib.define(
        "quant_slide_int8(Tensor input, str model_name, int L) -> (Tensor, Tensor)"
    )
    
    def _quant_slide_int8_impl(input, model_name, L):
        fn = _load_quant_slide_int8_kernel(model_name)
        return fn(input, L)
    
    def _quant_slide_int8_fake(input, model_name, L):
        M, K = input.shape
        M_pad = _ceil16(M)
        K_slide = K * 2 * (L - 2) // L
        K_slide_pad = _ceil32(K_slide)
        qinput = torch.empty((M_pad, K_slide_pad), dtype=torch.int8, device=input.device)
        scale_a = torch.empty((M_pad,), dtype=torch.float32, device=input.device)
        return qinput, scale_a
    
    _triton_lib.impl("quant_slide_int8", _quant_slide_int8_impl, "CUDA")
    _triton_lib._register_fake("quant_slide_int8", _quant_slide_int8_fake)
    
    logger.info_once("Triton kernel custom ops registered")


# 模块加载时注册 custom ops
_register_triton_custom_ops()


# ============================================================================
# 模块预加载
# ============================================================================

def _preload_all_kernels() -> None:
    """
    扫描 build 目录，预加载所有已编译的 tuned kernel
    
    在模块导入时调用，确保:
    1. 子进程导入模块时自动完成预加载
    2. 热路径上只有字典读取，无文件系统操作
    
    优化:
    如果环境变量 SLIDESPARSE_MODEL_NAME 存在，则只加载该模型对应的 kernel。
    这显著减少了启动时间和内存占用。如果指定模型未找到，打印警告回退到全量加载，
    以避免"配置错误但默默执行低效路径"的隐患。
    """
    try:
        hw_dir = build_hw_dir_name()
    except Exception as e:
        # 如果无法确定硬件目录（如 GPU 不可用），跳过预加载
        logger.debug(f"Skipping kernel preload: {e}")
        return
    
    # Check for target model optimization
    target_model = os.environ.get("SLIDESPARSE_MODEL_NAME")
    if target_model:
        _preload_target_model_kernels(target_model)
        return

    # Fallback to full scan (Legacy behavior)
    _preload_scan_all_kernels(hw_dir)


def _preload_target_model_kernels(target_model_name: str) -> None:
    """针对特定模型的定向预加载"""
    # 提取 base model name (因为文件名是基于 base model 的)
    # 例如: Llama3.2-1B-FP8-SlideSparse-2_8 -> Llama3.2-1B-FP8
    base_model = _extract_base_model_name(target_model_name)
    logger.info(f"Optimization enabled: Only preloading kernels for base model '{base_model}'")
    
    # 尝试加载各类 kernel
    # 注意: _load_xxx 函数内部会再次调用 _extract_base_model_name，所以传 full name 或 base name 均可
    # 但为了逻辑清晰，我们这里传 base_model
    
    # 1. Dequant Bias
    try:
        _load_dequant_bias_kernel(base_model)
    except Exception as e:
        # 不要 fallback，明确警告
        logger.warning(f"Target dequant_bias kernel for '{base_model}' not found: {e}")

    # 2. Quant Only (FP8 & INT8)
    try:
        _load_quant_only_fp8_kernel(base_model)
    except Exception as e:
        logger.warning(f"Target quant_only_fp8 kernel for '{base_model}' not found: {e}")
        
    try:
        _load_quant_only_int8_kernel(base_model)
    except Exception as e:
        logger.warning(f"Target quant_only_int8 kernel for '{base_model}' not found: {e}")

    # 3. Quant Slide (FP8 & INT8)
    try:
        _load_quant_slide_fp8_kernel(base_model)
    except Exception as e:
        logger.warning(f"Target quant_slide_fp8 kernel for '{base_model}' not found: {e}")

    try:
        _load_quant_slide_int8_kernel(base_model)
    except Exception as e:
        logger.warning(f"Target quant_slide_int8 kernel for '{base_model}' not found: {e}")


def _preload_scan_all_kernels(hw_dir: str) -> None:
    """全量扫描 build 目录加载所有 kernel (原始逻辑)"""
    preloaded_count = 0
    
    # 1. 扫描 dequant_bias kernels
    dequant_build_dir = _CSRC_DIR / "fused_dequant_bias_triton" / "build" / hw_dir
    if dequant_build_dir.exists():
        for kernel_file in dequant_build_dir.glob("dequant_bias_tuned_*.py"):
            # 从文件名提取 model_name: dequant_bias_tuned_{model_name}.py
            stem = kernel_file.stem  # e.g., "dequant_bias_tuned_Llama3.2-1B-FP8"
            prefix = "dequant_bias_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_dequant_bias_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload dequant_bias for {model_name}: {e}")
    
    # 2. 扫描 quant_only kernels（同时加载 FP8 和 INT8）
    quant_only_build_dir = _CSRC_DIR / "quant_only_triton" / "build" / hw_dir
    if quant_only_build_dir.exists():
        for kernel_file in quant_only_build_dir.glob("quant_only_tuned_*.py"):
            stem = kernel_file.stem
            prefix = "quant_only_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_quant_only_fp8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_only_fp8 for {model_name}: {e}")
                try:
                    _load_quant_only_int8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_only_int8 for {model_name}: {e}")
    
    # 3. 扫描 quant_slide kernels（同时加载 FP8 和 INT8）
    quant_slide_build_dir = _CSRC_DIR / "fused_quant_slide_triton" / "build" / hw_dir
    if quant_slide_build_dir.exists():
        for kernel_file in quant_slide_build_dir.glob("quant_slide_tuned_*.py"):
            stem = kernel_file.stem
            prefix = "quant_slide_tuned_"
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                try:
                    _load_quant_slide_fp8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_slide_fp8 for {model_name}: {e}")
                try:
                    _load_quant_slide_int8_kernel(model_name)
                    preloaded_count += 1
                except Exception as e:
                    logger.debug(f"Failed to preload quant_slide_int8 for {model_name}: {e}")
    
    if preloaded_count > 0:
        logger.info(f"Preloaded {preloaded_count} Triton kernels from {hw_dir}")


# 模块导入时执行预加载
_preload_all_kernels()


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Search helper
    "_search_kernel",
    "_load_basic_kernel",
    "_BASIC_KERNEL_FILES",
    "_CSRC_DIR",
    
    # Preload
    "_preload_all_kernels",
    
    # Dequant kernel（共享）
    "_load_dequant_bias_kernel",
    "dequant_bias_kernel",
    
    # FP8 Quant kernels
    "_load_quant_only_fp8_kernel",
    "quant_only_fp8_kernel",
    "_load_quant_slide_fp8_kernel",
    "quant_slide_fp8_kernel",
    
    # INT8 Quant kernels
    "_load_quant_only_int8_kernel",
    "quant_only_int8_kernel",
    "_load_quant_slide_int8_kernel",
    "quant_slide_int8_kernel",
]
