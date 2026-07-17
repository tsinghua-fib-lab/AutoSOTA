#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Tools 工具库

本模块提供 tools 目录下脚本的专用工具函数。

注意：
    - 硬件信息、文件名、模型管理等通用功能请使用顶层 slidesparse.utils 模块
    - 本模块仅包含 tools 目录专用的功能

主要功能
========
1. 颜色输出和打印工具
2. vLLM 命令构建
3. 结果目录管理
4. 测试参数计算

使用示例
========
>>> from slidesparse.tools.utils import (
...     print_header, print_info, print_success,
...     build_result_dir, get_vllm_env_vars
... )
"""

import os
import re
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

# 确保可以导入顶层 slidesparse
_TOOLS_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _TOOLS_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    hw_info,
    model_registry,
    check_quant_support,
    get_model_local_path,
    check_model_downloaded,
    build_stem,
    resolve_slidesparse_model_path,
    get_slidesparse_checkpoints_dir,
)


# =============================================================================
# 路径常量
# =============================================================================

# 项目根目录
PROJECT_ROOT = _PROJECT_ROOT

# Checkpoints 目录
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 默认 HuggingFace 组织
HF_ORG = "RedHatAI"


# =============================================================================
# 颜色和打印工具
# =============================================================================

class Colors:
    """终端颜色代码"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def disable(cls):
        """禁用颜色输出"""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.CYAN = cls.MAGENTA = cls.NC = ""


# ANSI 转义码正则表达式
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    """
    去除字符串中的 ANSI 转义码
    
    Args:
        text: 可能包含 ANSI 转义码的字符串
        
    Returns:
        去除转义码后的纯文本
    """
    return _ANSI_ESCAPE_RE.sub('', text)


def print_header(msg: str) -> None:
    """打印标题"""
    print()
    print(f"{Colors.CYAN}{'=' * 60}{Colors.NC}")
    print(f"{Colors.CYAN}  {msg}{Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.NC}")
    print()


def print_subheader(msg: str) -> None:
    """打印子标题"""
    print()
    print(f"{Colors.MAGENTA}{'-' * 60}{Colors.NC}")
    print(f"{Colors.MAGENTA}  {msg}{Colors.NC}")
    print(f"{Colors.MAGENTA}{'-' * 60}{Colors.NC}")


def print_info(msg: str) -> None:
    """打印信息"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def print_success(msg: str) -> None:
    """打印成功"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def print_warning(msg: str) -> None:
    """打印警告"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")


def print_error(msg: str) -> None:
    """打印错误"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


# =============================================================================
# 结果目录管理
# =============================================================================

def get_hw_folder_name(dtype: Optional[str] = None) -> str:
    """
    获取完整的硬件信息文件夹名
    
    格式: {GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}
    
    Args:
        dtype: 数据类型（如 "INT8", "FP8"）
    
    Returns:
        完整的硬件标识字符串
    """
    # 直接复用外层 build_stem，去掉前导下划线
    return build_stem("", dtype=dtype).lstrip("_")


def build_result_dir(
    base_name: str,
    *,
    test_mode: Optional[str] = None,
    dtype: Optional[str] = None,
    create: bool = True,
) -> Path:
    """
    构建结果目录路径
    
    格式 (有 test_mode): {base_name}_results/{test_mode}/{完整硬件名}/
    格式 (无 test_mode): {base_name}_results/{完整硬件名}/
    
    例如:
        throughput_bench_results/prefill/H100_cc90_py312_cu124_x86_64/
        accuracy_quickbench_results/H100_cc90_INT8_py312_cu124_x86_64/
    
    Args:
        base_name: 基础名称（如 "throughput_bench", "accuracy_quickbench"）
        test_mode: 测试模式子目录（如 "prefill", "decode"），可选
        dtype: 数据类型（如 "INT8", "FP8"），可选
        create: 是否创建目录
        
    Returns:
        结果目录路径
    """
    # 完整硬件信息文件夹名: {GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}
    hw_folder = get_hw_folder_name(dtype=dtype)
    
    # 构建路径
    result_base = _TOOLS_DIR / f"{base_name}_results"
    
    if test_mode:
        result_dir = result_base / test_mode / hw_folder
    else:
        result_dir = result_base / hw_folder
    
    if create:
        result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir


def get_result_dir(
    base_name: str,
    *,
    test_mode: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Optional[Path]:
    """
    获取现有的结果目录
    
    Args:
        base_name: 基础名称
        test_mode: 测试模式子目录
        dtype: 数据类型（如 "INT8", "FP8"），可选
        
    Returns:
        结果目录，不存在返回 None
    """
    hw_folder = get_hw_folder_name(dtype=dtype)
    result_base = _TOOLS_DIR / f"{base_name}_results"
    
    if test_mode:
        result_dir = result_base / test_mode / hw_folder
    else:
        result_dir = result_base / hw_folder
    
    return result_dir if result_dir.exists() else None


# =============================================================================
# GPU 工具函数
# =============================================================================

def get_gpu_devices_for_tp(tp_size: int, gpu_id: str) -> Tuple[str, int]:
    """
    根据 TP 大小和 GPU_ID 计算实际使用的 GPU 列表
    
    Args:
        tp_size: 目标 tensor parallelism 大小
        gpu_id: GPU ID 列表字符串 (逗号分隔)
        
    Returns:
        (gpu_devices, actual_tp_size) 元组
    """
    if tp_size <= 1:
        first_gpu = gpu_id.split(",")[0]
        return first_gpu, 1
    
    gpu_list = [g.strip() for g in gpu_id.split(",")]
    available_count = len(gpu_list)
    
    if available_count < tp_size:
        print_warning(f"GPU_ID 指定 {available_count} 个 GPU，但 TP={tp_size}，使用所有可用 GPU")
        return gpu_id, available_count
    elif available_count > tp_size:
        selected = ",".join(gpu_list[:tp_size])
        print_warning(f"GPU_ID 指定 {available_count} 个 GPU，但 TP={tp_size}，使用前 {tp_size} 个: {selected}")
        return selected, tp_size
    else:
        return gpu_id, tp_size


# =============================================================================
# vLLM 环境变量和命令构建
# =============================================================================

def get_vllm_env_vars(
    *,
    log_level: str = "WARNING",
    disable_compile: bool = False,
    extra_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    获取 vLLM 运行所需的环境变量
    
    注意: 此函数不设置 CUDA_VISIBLE_DEVICES，调用方应自行设置
    
    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        disable_compile: 是否禁用 torch.compile
        extra_vars: 额外的环境变量
        
    Returns:
        环境变量字典
    """
    env = {}
    
    # vLLM 日志级别
    env["VLLM_LOGGING_LEVEL"] = log_level
    
    # Triton 缓存（避免污染）
    env["TRITON_CACHE_DIR"] = str(_TOOLS_DIR / ".triton_cache")
    
    # 禁用 torch.compile（用于不支持的架构）
    if disable_compile or not hw_info.triton_supported[0]:
        env["VLLM_TORCH_COMPILE_LEVEL"] = "0"
    
    # 额外变量
    if extra_vars:
        env.update(extra_vars)
    
    return env


def check_triton_support_and_warn() -> bool:
    """
    检查 Triton 支持并打印警告
    
    Returns:
        True 如果支持，False 如果需要 eager mode
    """
    supported, reason = hw_info.triton_supported
    if not supported:
        print_warning(f"GPU 架构不被 Triton 支持: {reason}")
        print_warning("将使用 eager mode (禁用 torch.compile)")
        return False
    return True


def check_quant_support_or_exit(quant: str) -> None:
    """
    检查量化支持，不支持则退出
    
    Args:
        quant: 量化类型（fp8, int8）
    """
    supported, msg = check_quant_support(quant)
    if not supported:
        print_error(f"{quant.upper()} 测试被拒绝!")
        print_error(msg)
        sys.exit(1)
    print_success(f"{quant.upper()} 支持检查通过")


# =============================================================================
# HuggingFace CLI 工具
# =============================================================================

def check_hf_cli() -> bool:
    """
    检查 HuggingFace CLI 是否可用
    
    Returns:
        True 如果可用
    """
    # 尝试 hf version 命令 (新版和 0.36.0 都支持)
    try:
        result = subprocess.run(
            ["hf", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # 尝试 huggingface-cli version 命令
    try:
        result = subprocess.run(
            ["huggingface-cli", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return False


def get_hf_download_cmd() -> str:
    """
    获取 HuggingFace 下载命令
    
    Returns:
        "hf download" 或 "huggingface-cli download"
    """
    # 优先使用 hf 命令
    try:
        result = subprocess.run(
            ["hf", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "hf download"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return "huggingface-cli download"


def download_model(
    key: str,
    checkpoint_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    下载模型
    
    Args:
        key: 模型 key
        checkpoint_dir: checkpoints 目录
        force: 是否强制重新下载
        
    Returns:
        (success, message)
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    # 获取模型信息
    entry = model_registry.get(key)
    if entry is None:
        return False, f"模型不存在: {key}"
    
    local_dir = checkpoint_dir / entry.local_name
    
    # 检查是否已存在
    if not force and local_dir.is_dir() and (local_dir / "config.json").exists():
        return True, f"模型已存在: {local_dir}"
    
    # 检查 HF CLI
    if not check_hf_cli():
        return False, "HuggingFace CLI 未安装，请运行: pip install -U huggingface_hub"
    
    # 创建目录
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载
    hf_cmd = get_hf_download_cmd()
    cmd = f"{hf_cmd} {entry.hf_path} --local-dir {local_dir}"
    
    print_info(f"下载命令: {cmd}")
    
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            return True, f"下载成功: {local_dir}"
        else:
            return False, f"下载失败 (exit code {result.returncode})"
    except Exception as e:
        return False, f"下载出错: {e}"


# =============================================================================
# 模型检查和列表
# =============================================================================

def print_model_status(
    checkpoint_dir: Optional[Path] = None,
    quant_filter: Optional[str] = None,
    family_filter: Optional[str] = None,
) -> Tuple[int, int]:
    """
    打印模型下载状态
    
    Args:
        checkpoint_dir: checkpoints 目录
        quant_filter: 量化类型过滤
        family_filter: 模型系列过滤
        
    Returns:
        (downloaded_count, missing_count)
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    print_header("模型下载状态")
    
    downloaded = 0
    missing = 0
    
    # 按量化类型分组显示
    for quant in ["int8", "fp8", "bf16"]:
        if quant_filter and quant != quant_filter.lower():
            continue
        
        # 检查该类型是否有模型
        models = model_registry.list(quant=quant, family=family_filter)
        if not models:
            continue
        
        print(f"\n{quant.upper()} 模型:")
        print("-" * 40)
        
        for entry in model_registry.list(quant=quant, family=family_filter):
            local_dir = checkpoint_dir / entry.local_name
            
            if local_dir.is_dir() and (local_dir / "config.json").exists():
                # 计算大小
                try:
                    size_bytes = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
                    size_gb = size_bytes / (1024 ** 3)
                    size_str = f"{size_gb:.1f} GB"
                except Exception:
                    size_str = "unknown"
                
                print(f"  {Colors.GREEN}✓{Colors.NC} {entry.local_name} - {size_str}")
                downloaded += 1
            else:
                print(f"  {Colors.RED}✗{Colors.NC} {entry.local_name} - not downloaded")
                missing += 1
    
    print()
    print("-" * 40)
    print(f"总计: {Colors.GREEN}{downloaded} 已下载{Colors.NC}, {Colors.RED}{missing} 缺失{Colors.NC}")
    
    # 显示 checkpoints 目录大小
    if checkpoint_dir.exists():
        try:
            result = subprocess.run(
                ["du", "-sh", str(checkpoint_dir)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                size = result.stdout.strip().split()[0]
                print_info(f"Checkpoints 目录大小: {size}")
        except Exception:
            pass
    
    return downloaded, missing


# =============================================================================
# 打印硬件信息
# =============================================================================

def print_hardware_info() -> None:
    """打印硬件信息表格"""
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Hardware Information                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ GPU:              {hw_info.gpu_full_name:<42}│"[:65] + "│")
    print(f"│ GPU (short):      {hw_info.gpu_name:<42}│")
    print(f"│ Memory:           {hw_info.gpu_memory_gb:.1f} GB{'':<36}│")
    print(f"│ CC:               {hw_info.cc_tag} ({hw_info.arch_name}){'':<30}│"[:65] + "│")
    print(f"│ SM Code:          {hw_info.sm_code:<42}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ CUDA Runtime:     {hw_info.cuda_runtime_version:<42}│")
    print(f"│ CUDA Driver:      {hw_info.cuda_driver_version:<42}│")
    print(f"│ Driver:           {hw_info.driver_version:<42}│")
    print(f"│ PyTorch:          {hw_info.pytorch_version:<42}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ Triton:           {'✓ supported' if hw_info.triton_supported[0] else '✗ ' + hw_info.triton_supported[1][:30]:<42}│"[:65] + "│")
    print(f"│ FP8 Support:      {'✓' if hw_info.supports_fp8 else '✗':<42}│")
    print(f"│ INT8 Support:     {'✓' if hw_info.supports_int8 else '✗':<42}│")
    print("└─────────────────────────────────────────────────────────────┘")


# =============================================================================
# 导出
# =============================================================================

# =============================================================================
# Benchmark 专用工具函数
# =============================================================================

def build_backend_result_dir(
    base_name: str,
    stage: str,
    backend: str,
    dtype: str,
    sparsity: Optional[str] = None,
    *,
    create: bool = True,
) -> Path:
    """
    构建 Backend 对应的结果目录
    
    格式:
        {base_name}_results/{stage}/{hw_folder}/{backend}/
        {base_name}_results/{stage}/{hw_folder}/cusparselt/{sparsity}/
    
    Args:
        base_name: 基础名称（如 "throughput_benchmark"）
        stage: "prefill" 或 "decode"
        backend: "cutlass" / "cublaslt" / "cusparselt"
        dtype: "FP8" 或 "INT8"
        sparsity: 稀疏配置 (仅 cusparselt)
        create: 是否创建目录
    
    Returns:
        结果目录路径
    """
    hw_folder = get_hw_folder_name(dtype=dtype)
    
    result_base = _TOOLS_DIR / f"{base_name}_results"
    
    if backend == "cusparselt" and sparsity:
        result_dir = result_base / stage / hw_folder / backend / sparsity
    else:
        result_dir = result_base / stage / hw_folder / backend
    
    if create:
        result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir


def get_checkpoint_path(
    model_key: str,
    backend: str,
    sparsity: Optional[str] = None,
    *,
    auto_convert: bool = True,
) -> Optional[Path]:
    """
    获取模型 checkpoint 路径
    
    对于 cuSPARSELt backend，如果 sparse checkpoint 不存在且 auto_convert=True，
    会尝试自动将 dense checkpoint 转换为 sparse 格式。
    
    Args:
        model_key: 模型 key
        backend: "cutlass" / "cublaslt" / "cusparselt"
        sparsity: 稀疏配置 (仅 cusparselt)
        auto_convert: 是否自动转换 (默认 True)
    
    Returns:
        checkpoint 路径，如果不存在返回 None
    """
    entry = model_registry.get(model_key)
    if entry is None:
        return None
    
    if backend in ("cutlass", "cublaslt"):
        # Dense checkpoint
        model_path = get_model_local_path(model_key, CHECKPOINT_DIR)
        if model_path.exists() and (model_path / "config.json").exists():
            return model_path
        return None
    
    elif backend == "cusparselt":
        # Sparse checkpoint
        model_path = get_model_local_path(model_key, CHECKPOINT_DIR)
        if not model_path.exists():
            return None
        
        # sparsity 必须指定
        if sparsity is None:
            return None
        
        # 使用顶层工具查找/转换 SlideSparse checkpoint
        # resolve_slidesparse_model_path 默认 auto_convert=True，会自动转换
        return resolve_slidesparse_model_path(model_path, sparsity, auto_convert=auto_convert)
    
    return None


__all__ = [
    # 路径
    "PROJECT_ROOT",
    "CHECKPOINT_DIR",
    "HF_ORG",
    # 颜色
    "Colors",
    # 打印
    "print_header",
    "print_subheader",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    # 目录
    "get_hw_folder_name",
    "build_result_dir",
    "get_result_dir",
    "build_backend_result_dir",
    # Checkpoint
    "get_checkpoint_path",
    # GPU
    "get_gpu_devices_for_tp",
    # vLLM
    "get_vllm_env_vars",
    "check_triton_support_and_warn",
    "check_quant_support_or_exit",
    # HuggingFace
    "check_hf_cli",
    "get_hf_download_cmd",
    "download_model",
    # 模型
    "print_model_status",
    # 硬件
    "print_hardware_info",
    # 其他
    "strip_ansi",
]
