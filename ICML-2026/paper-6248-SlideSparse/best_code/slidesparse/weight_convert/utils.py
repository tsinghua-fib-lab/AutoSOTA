#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Weight Convert 共享工具函数

提供 safetensor 读写、配置管理、目标层检测等通用功能。

注意: SlideSparseConfig、LINEAR_LAYER_TYPES 等核心定义已移至顶层 utils.py，
      本模块从顶层导入并复用。
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import torch
import numpy as np

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 从顶层 utils 导入核心定义（避免重复）
from slidesparse.utils import (
    SlideSparseConfig,
    LINEAR_LAYER_TYPES,
    compute_output_k,
    compute_compressed_k,
)

# 尝试导入 safetensors
try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Run: pip install safetensors")

# 尝试导入 numba 加速
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# 目录常量
# =============================================================================

CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
CHECKPOINT_SLIDESPARSE_DIR = _PROJECT_ROOT / "checkpoints_slidesparse"
BUILD_DIR = _SCRIPT_DIR / "build"


# =============================================================================
# 目标层检测（基于顶层 LINEAR_LAYER_TYPES）
# =============================================================================

# 旧版 BitNet 格式的目标层名称（补充到 LINEAR_LAYER_TYPES 的映射）
# 这些是 BitNet 特有的命名，与标准 HuggingFace 格式不同
_BITNET_LAYER_MAP = {
    "wqkv": "qkv",
    "wo": "wo",
    "w13": "w13",
    "w2": "w2",
}


def is_target_layer(key: str, format_type: str = "auto") -> bool:
    """
    判断 key 是否为目标层（需要进行 SlideSparse 转换）
    
    使用顶层 LINEAR_LAYER_TYPES 定义，统一管理层类型映射。
    
    Args:
        key: 权重名称
        format_type: 格式类型
            - "auto": 自动检测（vLLM + BitNet）
            - "vllm": vLLM compressed-tensor 格式
            - "bitnet": 旧版 BitNet 格式
    
    Returns:
        是否为目标层
    """
    key_lower = key.lower()
    
    # 必须是权重，不是 scale 或其他参数
    if "weight" not in key_lower:
        return False
    if "scale" in key_lower or "zero" in key_lower:
        return False
    
    # 检查 vLLM 格式（使用顶层 LINEAR_LAYER_TYPES）
    if format_type in ("auto", "vllm"):
        for pattern in LINEAR_LAYER_TYPES.keys():
            if pattern in key_lower:
                return True
    
    # 检查 BitNet 格式
    if format_type in ("auto", "bitnet"):
        for pattern in _BITNET_LAYER_MAP.keys():
            if pattern in key_lower:
                return True
    
    return False


def get_layer_type(key: str) -> Optional[str]:
    """
    获取层类型
    
    使用顶层 LINEAR_LAYER_TYPES 定义，返回统一的层类型标识。
    
    Returns:
        层类型: "qkv", "wo", "w13", "w2", None
    """
    key_lower = key.lower()
    
    # 优先检查融合层（更具体的模式）
    if "qkv_proj" in key_lower or "wqkv" in key_lower:
        return "qkv"
    if "gate_up_proj" in key_lower or "w13" in key_lower:
        return "w13"
    
    # 检查标准 vLLM 格式
    for pattern, layer_type in LINEAR_LAYER_TYPES.items():
        if pattern in key_lower:
            return layer_type
    
    # 检查 BitNet 格式
    for pattern, layer_type in _BITNET_LAYER_MAP.items():
        if pattern in key_lower:
            return layer_type
    
    return None


# =============================================================================
# Safetensor 读写工具
# =============================================================================

def load_safetensors(
    path: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    加载 safetensors 文件
    
    Args:
        path: safetensors 文件路径
        device: 加载到的设备
    
    Returns:
        权重字典 {name: tensor}
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors not installed")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return safetensors_load_file(str(path), device=device)


def save_safetensors(
    tensors: Dict[str, torch.Tensor],
    path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    保存为 safetensors 文件
    
    Args:
        tensors: 权重字典 {name: tensor}
        path: 输出路径
        metadata: 可选的元数据
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors not installed")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    safetensors_save_file(tensors, str(path), metadata=metadata)


def get_safetensors_metadata(path: Union[str, Path]) -> Dict[str, str]:
    """获取 safetensors 文件的元数据"""
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors not installed")
    
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def get_safetensors_keys(path: Union[str, Path]) -> List[str]:
    """获取 safetensors 文件中的所有 key"""
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors not installed")
    
    with safe_open(str(path), framework="pt") as f:
        return list(f.keys())


def load_single_tensor(
    path: Union[str, Path],
    key: str,
    device: str = "cpu",
) -> torch.Tensor:
    """从 safetensors 文件中加载单个张量"""
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors not installed")
    
    with safe_open(str(path), framework="pt", device=device) as f:
        return f.get_tensor(key)


# =============================================================================
# 模型路径工具
# =============================================================================

def get_model_safetensors_path(model_dir: Union[str, Path]) -> Path:
    """
    获取模型目录中的 safetensors 文件路径
    
    支持：
        - model.safetensors (单文件)
        - model-00001-of-00002.safetensors (分片)
    """
    model_dir = Path(model_dir)
    
    # 单文件格式
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return single_file
    
    # 分片格式
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if shard_files:
        return shard_files[0]
    
    raise FileNotFoundError(f"No safetensors file found in {model_dir}")


def get_all_safetensors_files(model_dir: Union[str, Path]) -> List[Path]:
    """获取模型目录中的所有 safetensors 文件"""
    model_dir = Path(model_dir)
    
    # 单文件格式
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]
    
    # 分片格式
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if shard_files:
        return shard_files
    
    return []


def get_output_model_dir(
    input_model_dir: Union[str, Path],
    output_base: Union[str, Path] = CHECKPOINT_SLIDESPARSE_DIR,
    suffix: str = "-SlideSparse",
) -> Path:
    """
    根据输入模型目录生成输出目录路径
    
    Args:
        input_model_dir: 输入模型目录
        output_base: 输出基目录
        suffix: 目录名后缀
    
    Returns:
        输出目录路径
    """
    input_model_dir = Path(input_model_dir)
    output_base = Path(output_base)
    
    model_name = input_model_dir.name
    output_name = f"{model_name}{suffix}"
    
    return output_base / output_name


# =============================================================================
# 配置文件工具
# =============================================================================

def load_model_config(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """加载模型的 config.json"""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_config(config: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """保存模型配置"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_slidesparse_config(
    config: SlideSparseConfig,
    output_dir: Union[str, Path],
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """保存 SlideSparse 配置"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    if extra_info:
        data["extra"] = extra_info
    
    with open(output_dir / "slidesparse_config.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_slidesparse_config(model_dir: Union[str, Path]) -> SlideSparseConfig:
    """加载 SlideSparse 配置"""
    config_path = Path(model_dir) / "slidesparse_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"slidesparse_config.json not found in {model_dir}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return SlideSparseConfig.from_dict(data)


def copy_non_weight_files(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    复制非权重文件（config.json, tokenizer 等）
    
    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        exclude_patterns: 排除的文件模式
    
    Returns:
        复制的文件列表
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if exclude_patterns is None:
        exclude_patterns = ["*.safetensors", "*.bin", "*.pt", ".cache"]
    
    copied = []
    for item in src_dir.iterdir():
        # 检查是否应该排除
        should_exclude = False
        for pattern in exclude_patterns:
            if item.match(pattern):
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        dst_path = dst_dir / item.name
        if item.is_file():
            shutil.copy2(item, dst_path)
            copied.append(item.name)
        elif item.is_dir() and item.name != ".cache":
            shutil.copytree(item, dst_path, dirs_exist_ok=True)
            copied.append(item.name + "/")
    
    return copied


# =============================================================================
# 数据类型工具
# =============================================================================

def detect_weight_dtype(tensor: torch.Tensor) -> str:
    """
    检测权重的数据类型
    
    Returns:
        "fp8_e4m3", "fp8_e5m2", "int8", "bf16", "fp16", "fp32"
    """
    dtype = tensor.dtype
    
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    elif dtype == torch.float8_e5m2:
        return "fp8_e5m2"
    elif dtype == torch.int8:
        return "int8"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        return str(dtype)


def can_represent_ternary(dtype_str: str) -> bool:
    """
    检查数据类型是否可以表示三元值 (-1, 0, 1)
    
    FP8 E4M3 可以精确表示 -1, 0, 1
    FP8 E5M2 也可以
    INT8 当然可以
    """
    return dtype_str in ("fp8_e4m3", "fp8_e5m2", "int8", "bf16", "fp16", "fp32")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """将字符串转换为 torch.dtype"""
    mapping = {
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "int8": torch.int8,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return mapping[dtype_str]


# =============================================================================
# 稀疏验证工具
# =============================================================================

def verify_ZL_sparsity(
    tensor: torch.Tensor,
    Z: int,
    L: int,
) -> Tuple[bool, float]:
    """
    验证张量是否满足 Z:L 稀疏约束
    
    Args:
        tensor: 需要验证的权重张量 [N, K]
        Z: 每组最少零元素数
        L: 组大小
    
    Returns:
        (is_valid, valid_ratio)
    """
    N, K = tensor.shape
    
    if K % L != 0:
        # 需要先 padding
        pad_cols = (L - (K % L)) % L
        tensor_padded = torch.cat([
            tensor,
            torch.zeros(N, pad_cols, dtype=tensor.dtype, device=tensor.device)
        ], dim=1)
    else:
        tensor_padded = tensor
    
    # 重塑为 [num_groups, L]
    grouped = tensor_padded.view(-1, L)
    
    # 统计每组的零元素数量
    zero_counts = (grouped == 0).sum(dim=1)
    
    # 检查是否每组至少 Z 个零
    valid_groups = (zero_counts >= Z).sum().item()
    total_groups = grouped.shape[0]
    
    valid_ratio = valid_groups / total_groups if total_groups > 0 else 0
    is_valid = valid_ratio == 1.0
    
    return is_valid, valid_ratio


def verify_2to4_sparsity(tensor: torch.Tensor, tolerance: float = 0.0) -> Tuple[bool, float]:
    """
    验证张量是否满足 2:4 稀疏约束
    
    Args:
        tensor: 需要验证的权重张量 [N, K]，K 应为 4 的倍数
        tolerance: 允许的违规组比例
    
    Returns:
        (is_valid, violation_ratio)
    """
    N, K = tensor.shape
    
    if K % 4 != 0:
        print(f"警告: K={K} 不是 4 的倍数，无法验证 2:4 稀疏")
        return False, 1.0
    
    # 重塑为 [N, K/4, 4] 进行分组检查
    grouped = tensor.view(N, K // 4, 4)
    
    # 统计每组的零元素数量
    zero_counts = (grouped == 0).sum(dim=2)
    
    # 检查是否每组至少 2 个零
    violations = zero_counts < 2
    num_violations = violations.sum().item()
    total_groups = N * (K // 4)
    
    violation_ratio = num_violations / total_groups if total_groups > 0 else 0
    is_valid = violation_ratio <= tolerance
    
    return is_valid, violation_ratio


# =============================================================================
# 打印工具
# =============================================================================

def print_header(msg: str, char: str = "=", width: int = 70) -> None:
    """打印标题"""
    print(char * width)
    print(msg)
    print(char * width)


def print_info(msg: str) -> None:
    """打印信息"""
    print(f"[INFO] {msg}")


def print_success(msg: str) -> None:
    """打印成功信息"""
    print(f"[✓] {msg}")


def print_warning(msg: str) -> None:
    """打印警告"""
    print(f"[WARNING] {msg}")


def print_error(msg: str) -> None:
    """打印错误"""
    print(f"[ERROR] {msg}")


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 常量
    "CHECKPOINT_DIR",
    "CHECKPOINT_SLIDESPARSE_DIR",
    "BUILD_DIR",
    "HAS_SAFETENSORS",
    "HAS_NUMBA",
    
    # 配置类（从顶层 utils 重导出）
    "SlideSparseConfig",
    
    # 层类型定义（从顶层 utils 重导出）
    "LINEAR_LAYER_TYPES",
    
    # 目标层检测
    "is_target_layer",
    "get_layer_type",
    
    # Safetensor 工具
    "load_safetensors",
    "save_safetensors",
    "get_safetensors_metadata",
    "get_safetensors_keys",
    "load_single_tensor",
    
    # 路径工具
    "get_model_safetensors_path",
    "get_all_safetensors_files",
    "get_output_model_dir",
    
    # 配置文件工具
    "load_model_config",
    "save_model_config",
    "save_slidesparse_config",
    "load_slidesparse_config",
    "copy_non_weight_files",
    
    # 数据类型工具
    "detect_weight_dtype",
    "can_represent_ternary",
    "get_torch_dtype",
    
    # 维度计算（从顶层 utils 重导出）
    "compute_output_k",
    "compute_compressed_k",
    
    # 稀疏验证
    "verify_ZL_sparsity",
    "verify_2to4_sparsity",
    
    # 打印工具
    "print_header",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
]
