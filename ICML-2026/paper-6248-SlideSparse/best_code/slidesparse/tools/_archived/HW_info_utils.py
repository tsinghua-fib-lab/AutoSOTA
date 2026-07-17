#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SlideSparse Benchmark 工具函数集
用于获取硬件信息、CUDA 版本等基准测试所需的系统信息
"""

import subprocess
import re
import json
import sys
from typing import Tuple, Dict, Any, Optional


# ============================================================================
# 架构检测
# ============================================================================

# 架构名称映射：compute capability major -> (arch_name, arch_suffix)
ARCH_INFO = {
    7: ("Volta", "volta"),         # V100 等
    8: ("Ampere", "ampere"),       # A100, A10, A30 等
    9: ("Hopper", "hopper"),       # H100, H200 等
    10: ("Blackwell", "blackwell"), # B100, B200 等
    12: ("Blackwell", "blackwell"), # GB10 等 (CC 12.x 也是 Blackwell 家族)
}


def get_torch():
    """延迟导入 torch，避免在不需要时加载"""
    try:
        import torch
        return torch
    except ImportError:
        return None


# ============================================================================
# CUDA 版本信息获取
# ============================================================================

def get_nvidia_smi_cuda_version() -> str:
    """获取 nvidia-smi 显示的 CUDA Version（驱动支持的最高 CUDA 版本）"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 解析 "CUDA Version: 13.0" 这样的格式
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 备选方案：通过 cudaDriverGetVersion API
    torch = get_torch()
    if torch is not None:
        try:
            driver_version = torch.cuda.driver_version()
            if driver_version:
                major = driver_version // 1000
                minor = (driver_version % 1000) // 10
                return f"{major}.{minor}"
        except Exception:
            pass
    
    return "unknown"


def get_cuda_runtime_version() -> str:
    """获取 CUDA runtime 版本（PyTorch 编译时使用的版本）"""
    torch = get_torch()
    if torch is not None:
        try:
            # torch.version.cuda 是 PyTorch 编译时使用的 CUDA toolkit 版本
            return torch.version.cuda or "unknown"
        except Exception:
            pass
    return "unknown"


def get_driver_version() -> str:
    """获取 NVIDIA 驱动版本"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unknown"


# ============================================================================
# GPU 信息获取
# ============================================================================

def get_gpu_full_name() -> str:
    """获取 GPU 完整名称"""
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        try:
            prop = torch.cuda.get_device_properties(0)
            return prop.name
        except Exception:
            pass
    
    # 备选：使用 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return "unknown"


def get_gpu_short_name() -> str:
    """
    获取 GPU 名称（简化版，如 A100, H100, B200）。
    
    常见格式处理:
    - "NVIDIA A100-SXM4-40GB" -> "A100"
    - "NVIDIA H100 PCIe" -> "H100"
    - "NVIDIA B200" -> "B200"
    """
    full_name = get_gpu_full_name()
    if full_name == "unknown":
        return "unknown"
    
    short_name = full_name
    
    # 移除 "NVIDIA " 前缀
    nvidia_pos = full_name.find("NVIDIA ")
    if nvidia_pos != -1:
        short_name = full_name[nvidia_pos + 7:]
    
    # 提取第一个空格或连字符之前的部分作为 GPU 型号
    for sep in [" ", "-"]:
        end_pos = short_name.find(sep)
        if end_pos != -1:
            short_name = short_name[:end_pos]
            break
    
    # 如果提取失败，使用清理后的完整名称（替换空格和特殊字符）
    if not short_name:
        short_name = full_name
        for c in [" ", "-", "/"]:
            short_name = short_name.replace(c, "_")
    
    return short_name


def get_gpu_memory_gb() -> float:
    """获取 GPU 显存大小 (GB)"""
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        try:
            prop = torch.cuda.get_device_properties(0)
            return prop.total_memory / (1024 ** 3)
        except Exception:
            pass
    
    # 备选：使用 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            mem_mb = float(result.stdout.strip().split('\n')[0])
            return mem_mb / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return 0.0


def get_compute_capability() -> Tuple[int, int]:
    """获取 GPU 计算能力 (major, minor)"""
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        try:
            prop = torch.cuda.get_device_properties(0)
            return prop.major, prop.minor
        except Exception:
            pass
    
    # 备选：使用 nvidia-smi (只能获取 major.minor 格式)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            cc_str = result.stdout.strip().split('\n')[0]
            parts = cc_str.split('.')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return 0, 0


def get_gpu_count() -> int:
    """获取 GPU 数量"""
    torch = get_torch()
    if torch is not None:
        try:
            return torch.cuda.device_count()
        except Exception:
            pass
    
    # 备选：使用 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return 0


# ============================================================================
# 架构检测
# ============================================================================

def detect_arch() -> Tuple[str, str, str]:
    """
    检测 GPU 架构。
    
    返回:
        (arch_name, arch_suffix, sm_code) 其中:
        - arch_name: "Ampere", "Hopper", "Blackwell" 等（用于显示）
        - arch_suffix: "ampere", "hopper", "blackwell" 等（用于文件命名）
        - sm_code: "sm_80", "sm_90" 等（用于 nvcc 编译）
    """
    major, minor = get_compute_capability()
    sm_code = f"sm_{major}{minor}"
    
    if major in ARCH_INFO:
        name, suffix = ARCH_INFO[major]
        return name, suffix, sm_code
    
    # 未知架构
    return f"SM{major}{minor}", f"sm{major}{minor}", sm_code


def check_triton_arch_support() -> Tuple[bool, str]:
    """
    检查当前 GPU 架构是否被 Triton/ptxas 支持。
    
    某些新架构（如 GB10 的 sm_121a）可能不被当前版本的 ptxas 支持，
    导致 torch.compile 失败。
    
    返回:
        (supported, reason) 其中:
        - supported: True 如果架构被支持
        - reason: 如果不支持，说明原因
    """
    major, minor = get_compute_capability()
    
    # 已知不被 Triton 支持的架构
    # GB10 (CC 12.1) 的 sm_121a 目前不被大多数 ptxas 版本支持
    UNSUPPORTED_ARCHS = {
        (12, 1): "GB10 (sm_121a) is not yet supported by Triton/ptxas",
    }
    
    if (major, minor) in UNSUPPORTED_ARCHS:
        return False, UNSUPPORTED_ARCHS[(major, minor)]
    
    # CC >= 12.0 的架构可能需要较新版本的工具链
    if major >= 12:
        # 尝试检测 ptxas 是否支持
        try:
            result = subprocess.run(
                ["ptxas", "--list-gpu-code"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                sm_code = f"sm_{major}{minor}"
                if sm_code not in result.stdout and f"sm_{major}{minor}a" not in result.stdout:
                    return False, f"ptxas does not support {sm_code}"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # 无法检测，保守起见认为不支持
            return False, f"Cannot verify ptxas support for sm_{major}{minor}"
    
    return True, "Architecture is supported"


def needs_eager_mode() -> bool:
    """
    检查是否需要使用 eager mode（禁用 torch.compile）。
    
    当 GPU 架构不被 Triton 支持时返回 True。
    """
    supported, _ = check_triton_arch_support()
    return not supported


# ============================================================================
# PyTorch 信息
# ============================================================================

def get_pytorch_version() -> str:
    """获取 PyTorch 版本"""
    torch = get_torch()
    if torch is not None:
        try:
            return torch.__version__
        except Exception:
            pass
    return "unknown"


# ============================================================================
# 综合信息获取
# ============================================================================

def get_hardware_info() -> Dict[str, Any]:
    """
    获取完整的硬件信息字典。
    
    返回包含以下字段的字典:
    - gpu_name: GPU 完整名称
    - gpu_short_name: GPU 简短名称
    - gpu_count: GPU 数量
    - gpu_memory_gb: GPU 显存 (GB)
    - compute_capability: 计算能力字符串 (如 "8.0")
    - cc_major: 计算能力主版本
    - cc_minor: 计算能力次版本
    - arch_name: 架构名称
    - arch_suffix: 架构后缀（用于文件命名）
    - sm_code: SM 代码
    - cuda_driver_version: CUDA 驱动版本
    - cuda_runtime_version: CUDA runtime 版本
    - driver_version: NVIDIA 驱动版本
    - pytorch_version: PyTorch 版本
    - folder_name: 建议的文件夹名称 (如 "A100_cc80")
    """
    cc_major, cc_minor = get_compute_capability()
    arch_name, arch_suffix, sm_code = detect_arch()
    gpu_short = get_gpu_short_name()
    
    info = {
        "gpu_name": get_gpu_full_name(),
        "gpu_short_name": gpu_short,
        "gpu_count": get_gpu_count(),
        "gpu_memory_gb": round(get_gpu_memory_gb(), 1),
        "compute_capability": f"{cc_major}.{cc_minor}",
        "cc_major": cc_major,
        "cc_minor": cc_minor,
        "arch_name": arch_name,
        "arch_suffix": arch_suffix,
        "sm_code": sm_code,
        "cuda_driver_version": get_nvidia_smi_cuda_version(),
        "cuda_runtime_version": get_cuda_runtime_version(),
        "driver_version": get_driver_version(),
        "pytorch_version": get_pytorch_version(),
        "folder_name": f"{gpu_short}_cc{cc_major}{cc_minor}",
    }
    
    return info


def print_hardware_info_table(info: Optional[Dict[str, Any]] = None) -> None:
    """以表格形式打印硬件信息"""
    if info is None:
        info = get_hardware_info()
    
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Hardware Information                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ GPU:              {info['gpu_name']:<42}│")
    print(f"│ GPU Count:        {info['gpu_count']:<42}│")
    print(f"│ GPU Memory:       {info['gpu_memory_gb']:.1f} GB{'':<36}│")
    print(f"│ Compute Cap:      {info['compute_capability']} ({info['arch_name']}){'':<30}│"[:65] + "│")
    print(f"│ SM Code:          {info['sm_code']:<42}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ CUDA Driver:      {info['cuda_driver_version']:<42}│")
    print(f"│ CUDA Runtime:     {info['cuda_runtime_version']:<42}│")
    print(f"│ NVIDIA Driver:    {info['driver_version']:<42}│")
    print(f"│ PyTorch:          {info['pytorch_version']:<42}│")
    print("└─────────────────────────────────────────────────────────────┘")


# ============================================================================
# 量化格式支持检测 (原生支持预拦截)
# ============================================================================
# 设计思路:
# - INT8 原生支持: CC >= 8.0 (Ampere+), 拦截 V100 等老卡
# - FP8 原生支持:  CC >= 8.9 (Ada/Hopper+), 拦截 A100 等不支持原生 FP8 的卡
# 这两个函数是对称的，都是为了避免 vLLM 的 fallback 机制污染 Benchmark 数据
# ============================================================================

def check_int8_support() -> Tuple[bool, str]:
    """
    检测 GPU 是否支持原生 INT8 GEMM 计算。
    
    INT8 原生支持要求:
    - Ampere (A100/A10): CC 8.0+
    - Ada (4090/L40S): CC 8.9+
    - Hopper (H100/H200): CC 9.0+
    - Blackwell (B100/B200): CC 10.0+
    
    Volta (V100) 等老架构 (CC < 8.0) 不支持高效的 INT8 Tensor Core GEMM。
    
    注意: 这里只检测硬件原生支持，不检测 vLLM kernel 是否覆盖了该架构。
    kernel 覆盖问题（如 SM100 上的 cutlass_scaled_mm 不支持）由运行时试错机制处理。
    
    返回:
        (is_supported, message): 是否支持及说明信息
    """
    major, minor = get_compute_capability()
    gpu_name = get_gpu_full_name()
    
    # INT8 原生支持: CC >= 8.0 (Ampere 开始有高效的 INT8 Tensor Core)
    if major >= 8:
        return True, f"GPU {gpu_name} (CC {major}.{minor}) supports native INT8 GEMM"
    
    return False, (
        f"GPU {gpu_name} (CC {major}.{minor}) does not support efficient INT8 Tensor Core GEMM.\n"
        f"INT8 quantization requires Ampere (A100) or newer architecture.\n"
        f"Suggestion: Use A100/H100/B100 or newer GPUs for INT8 tests."
    )


def check_int8_support_exit_if_not() -> None:
    """
    检测 INT8 支持，如果不支持则打印错误并退出。
    用于在 shell 脚本中预检测拦截。
    """
    supported, message = check_int8_support()
    if not supported:
        major, minor = get_compute_capability()
        gpu_name = get_gpu_full_name()
        print(f"\n⛔ [FATAL ERROR] INT8 test execution refused!")
        print(f"Detected GPU: {gpu_name} (Compute Capability {major}.{minor})")
        print(f"This GPU does not support efficient INT8 Tensor Core GEMM.")
        print(f"INT8 quantization requires Ampere (CC 8.0+) or newer architecture.")
        print(f"Please use A100/H100/B100 or newer GPUs for INT8 tests.\n")
        sys.exit(1)
    else:
        print(f"✓ INT8 support check passed: {message}")


def check_fp8_support() -> Tuple[bool, str]:
    """
    检测 GPU 是否支持原生 FP8 GEMM 计算。
    
    FP8 原生支持要求:
    - Hopper (H100/H200): CC 9.0+
    - Ada (4090/L40S): CC 8.9+
    - Blackwell (B100/B200): CC 10.0+
    
    如果不支持，vLLM 会回退到 Marlin (W8A16) 模式，这会污染 Benchmark 数据。
    
    返回:
        (is_supported, message): 是否支持及说明信息
    """
    major, minor = get_compute_capability()
    gpu_name = get_gpu_full_name()
    
    # FP8 原生支持: CC >= 8.9 (Ada) 或 CC >= 9.0 (Hopper) 或 CC >= 10.0 (Blackwell)
    # 实际上 CC 8.9 是 Ada (4090/L40S)，8.0 是 Ampere (A100)
    if major > 8 or (major == 8 and minor >= 9):
        return True, f"GPU {gpu_name} (CC {major}.{minor}) supports native FP8 GEMM"
    
    return False, (
        f"GPU {gpu_name} (CC {major}.{minor}) does not support native FP8 GEMM.\n"
        f"vLLM will fallback to Marlin (W8A16) mode, which will pollute Benchmark data.\n"
        f"Suggestion: Use H100/H200/L40S/B100/B200 or skip FP8 tests."
    )


def check_fp8_support_exit_if_not() -> None:
    """
    检测 FP8 支持，如果不支持则打印错误并退出。
    用于在 shell 脚本中预检测拦截。
    """
    supported, message = check_fp8_support()
    if not supported:
        major, minor = get_compute_capability()
        gpu_name = get_gpu_full_name()
        print(f"\n⛔ [FATAL ERROR] FP8 test execution refused!")
        print(f"Detected GPU: {gpu_name} (Compute Capability {major}.{minor})")
        print(f"This GPU does not support native FP8 GEMM computation.")
        print(f"vLLM might fallback to Marlin (W8A16) mode, which will pollute Benchmark data.")
        print(f"Please use H100/H200/L40S/B100/B200 or skip FP8 tests.\n")
        sys.exit(1)
    else:
        print(f"✓ FP8 support check passed: {message}")


# ============================================================================
# CLI 接口
# ============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SlideSparse Benchmark Utilities - Hardware Info Tool"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON format"
    )
    parser.add_argument(
        "--field", type=str, default=None,
        help="Get specific field value (e.g., folder_name, gpu_short_name, compute_capability)"
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Print hardware info as formatted table"
    )
    parser.add_argument(
        "--check-int8", action="store_true",
        help="Check if GPU supports native INT8 GEMM (exit 1 if not supported, for V100 etc.)"
    )
    parser.add_argument(
        "--check-fp8", action="store_true",
        help="Check if GPU supports native FP8 GEMM (exit 1 if not supported, for A100 etc.)"
    )
    parser.add_argument(
        "--check-triton-support", action="store_true",
        help="Check if GPU architecture is supported by Triton/ptxas. "
             "Outputs 'needs_eager' if torch.compile should be disabled, 'supported' otherwise."
    )
    
    args = parser.parse_args()
    
    # INT8 支持检测 (预检测拦截 - 拦截 V100 等老卡)
    if args.check_int8:
        check_int8_support_exit_if_not()
        sys.exit(0)
    
    # FP8 支持检测 (预检测拦截 - 拦截 A100 等不支持原生 FP8 的卡)
    if args.check_fp8:
        check_fp8_support_exit_if_not()
        sys.exit(0)
    
    # Triton 架构支持检测
    if args.check_triton_support:
        supported, reason = check_triton_arch_support()
        if supported:
            print("supported")
        else:
            print("needs_eager")
            # 输出原因到 stderr，不影响 stdout 的解析
            print(f"Reason: {reason}", file=sys.stderr)
        sys.exit(0)
    
    info = get_hardware_info()
    
    if args.field:
        # 输出特定字段
        if args.field in info:
            print(info[args.field])
        else:
            print(f"Unknown field: {args.field}", file=sys.stderr)
            print(f"Available fields: {', '.join(info.keys())}", file=sys.stderr)
            sys.exit(1)
    elif args.json:
        # JSON 格式输出
        print(json.dumps(info, indent=2))
    elif args.table:
        # 表格形式输出
        print_hardware_info_table(info)
    else:
        # 默认输出表格
        print_hardware_info_table(info)


if __name__ == "__main__":
    main()
