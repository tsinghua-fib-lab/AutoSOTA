#!/usr/bin/env python3
"""
自动调优 + 生成固定配置 Slide Kernel 的脚本

Usage:
    python3 autotune_slide_kernels.py <input_file> [-o output_file]

示例:
    # 调优 slide_L6_fp8.py，自动生成 slide_L6_fp8_autotuned.py
    python3 autotune_slide_kernels.py slide_L6_fp8.py
    
    # 调优并指定输出文件名
    python3 autotune_slide_kernels.py slide_L8_int8.py -o my_tuned_kernel.py
    
    # 调优 codegen_unified.py 生成的任意文件
    python3 autotune_slide_kernels.py my_custom_slide_kernel.py

功能:
1. 读取输入文件，解析 L, DTYPE 等配置
2. 对 K_in=2560, 6912 和各种 M 值进行 autotune
3. 生成不需要运行时 autotune 的固定配置 kernel 文件
"""

import torch
import sys
import os
import re
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path


# =============================================================================
# 调优测试参数
# =============================================================================

# 目标 K_in 值
K_IN_VALUES = [2560, 6912]

# 目标 M 值
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def load_module_from_file(file_path: str):
    """动态加载 Python 模块"""
    file_path = os.path.abspath(file_path)
    module_name = Path(file_path).stem
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_config_from_file(file_path: str) -> dict:
    """从文件中解析配置参数"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    config = {}
    
    # 解析 L
    match = re.search(r'^L\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        config['L'] = int(match.group(1))
    
    # 解析 L_PAD
    match = re.search(r'^L_PAD\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        config['L_PAD'] = int(match.group(1))
    
    # 解析 N
    match = re.search(r'^N\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        config['N'] = int(match.group(1))
    
    # 解析 NUM_WINDOWS
    match = re.search(r'^NUM_WINDOWS\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        config['NUM_WINDOWS'] = int(match.group(1))
    
    # 解析 EXPAND_RATIO
    match = re.search(r'^EXPAND_RATIO\s*=\s*([\d.]+)', content, re.MULTILINE)
    if match:
        config['EXPAND_RATIO'] = float(match.group(1))
    
    # 解析 DTYPE
    match = re.search(r'^DTYPE\s*=\s*["\'](\w+)["\']', content, re.MULTILINE)
    if match:
        config['DTYPE'] = match.group(1)
    
    return config


def run_tuning(module, config: dict) -> dict:
    """运行调优，返回结果字典 {num_groups: {M: config_dict}}"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return None
    
    L = config['L']
    fused_quant_slide = module.fused_quant_slide
    kernel = module._fused_quant_slide_kernel
    
    print(f"\nTesting K_in values: {K_IN_VALUES}")
    print(f"Testing M values: {M_VALUES}")
    
    results = {}
    max_K = max(K_IN_VALUES)
    max_M = max(M_VALUES)
    
    # 预分配输入 buffer
    x_buffer = torch.randn(max_M, max_K, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for K_in in K_IN_VALUES:
        # 计算 num_groups
        K_in_padded = ((K_in + L - 1) // L) * L
        num_groups = K_in_padded // L
        
        if num_groups not in results:
            results[num_groups] = {}
        
        print(f"\n[K_in={K_in}, num_groups={num_groups}]")
        
        for M in M_VALUES:
            print(f"  M={M:<5} ... ", end="", flush=True)
            x = x_buffer[:M, :K_in].contiguous()
            
            try:
                # 运行 kernel 触发 autotune
                y, scale = fused_quant_slide(x)
                torch.cuda.synchronize()
                
                # 获取最优配置
                best_config = None
                cache_key = (num_groups,)
                best_config = kernel.cache.get(cache_key)
                
                if best_config is None:
                    for key, cfg in kernel.cache.items():
                        if isinstance(key, tuple) and len(key) >= 1 and key[0] == num_groups:
                            best_config = cfg
                            break
                
                if best_config:
                    cfg_dict = {
                        'BLOCK_GROUPS': best_config.kwargs['BLOCK_GROUPS'],
                        'num_warps': best_config.num_warps,
                        'num_stages': best_config.num_stages,
                    }
                    print(f"Done. {cfg_dict}")
                    results[num_groups][M] = cfg_dict
                else:
                    cfg_dict = {
                        'BLOCK_GROUPS': 256,
                        'num_warps': 8,
                        'num_stages': 3,
                    }
                    print(f"(default) {cfg_dict}")
                    results[num_groups][M] = cfg_dict
                    
            except Exception as e:
                print(f"Error: {e}")
                results[num_groups][M] = {
                    'BLOCK_GROUPS': 256,
                    'num_warps': 8,
                    'num_stages': 3,
                }
    
    return results


def analyze_results(results: dict) -> dict:
    """
    分析调优结果，为每个 num_groups 选择最优配置
    """
    best_configs = {}
    
    for num_groups, m_configs in results.items():
        # 统计每个配置出现的次数
        cfg_counts = {}
        for M, cfg in m_configs.items():
            cfg_key = (cfg['BLOCK_GROUPS'], cfg['num_warps'], cfg['num_stages'])
            cfg_counts[cfg_key] = cfg_counts.get(cfg_key, 0) + 1
        
        # 选择出现次数最多的配置
        best_cfg_key = max(cfg_counts.keys(), key=lambda k: cfg_counts[k])
        best_configs[num_groups] = {
            'BLOCK_GROUPS': best_cfg_key[0],
            'num_warps': best_cfg_key[1],
            'num_stages': best_cfg_key[2],
        }
        
        print(f"num_groups={num_groups}: best config = {best_configs[num_groups]}")
    
    return best_configs


def generate_config_selector(best_configs: dict) -> str:
    """生成配置选择函数代码"""
    lines = []
    lines.append("def _get_config(num_groups: int):")
    lines.append('    """根据 num_groups 返回最优配置 (BLOCK_GROUPS, num_warps, num_stages)"""')
    
    sorted_groups = sorted(best_configs.keys())
    
    if not sorted_groups:
        lines.append("    return 256, 8, 3  # default")
        return "\n".join(lines)
    
    for i, ng in enumerate(sorted_groups):
        cfg = best_configs[ng]
        ret = f"{cfg['BLOCK_GROUPS']}, {cfg['num_warps']}, {cfg['num_stages']}"
        
        if i == 0:
            lines.append(f"    if num_groups <= {ng}:")
            lines.append(f"        return {ret}")
        else:
            lines.append(f"    elif num_groups <= {ng}:")
            lines.append(f"        return {ret}")
    
    # 默认配置（大于所有测试的 num_groups）
    last_cfg = best_configs[sorted_groups[-1]]
    lines.append(f"    else:")
    lines.append(f"        return {last_cfg['BLOCK_GROUPS']}, {last_cfg['num_warps']}, {last_cfg['num_stages']}")
    
    return "\n".join(lines)


def generate_kernel_code(input_file: str, config: dict, best_configs: dict) -> str:
    """生成完整的固定配置 Kernel 代码"""
    
    # 读取原始文件
    with open(input_file, 'r') as f:
        original_content = f.read()
    
    L = config['L']
    L_PAD = config.get('L_PAD', 8)
    N = config.get('N', L // 2)
    NUM_WINDOWS = config.get('NUM_WINDOWS', N - 1)
    EXPAND_RATIO = config.get('EXPAND_RATIO', (NUM_WINDOWS * 4) / L)
    DTYPE = config.get('DTYPE', 'int8')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 生成配置选择器
    config_selector = generate_config_selector(best_configs)
    
    # 确定输出数据类型
    if DTYPE == 'fp8':
        dtype_torch = 'torch.float8_e4m3fn'
    else:
        dtype_torch = 'torch.int8'
    
    # 提取原始 kernel 函数体
    kernel_match = re.search(
        r'@triton\.jit\s*\ndef _fused_quant_slide_kernel\([^)]+\):\s*\n(.*?)(?=\ndef |\Z)',
        original_content,
        re.DOTALL
    )
    
    if kernel_match:
        kernel_body = kernel_match.group(1)
    else:
        raise ValueError("Cannot find kernel function in input file")
    
    # 生成代码
    kernel_code = f'''"""
Fused {DTYPE.upper()} Quantization + SlideSparse Slide Kernel (Auto-tuned, No Runtime Autotune)

Auto-generated by autotune_slide_kernels.py at {timestamp}
Source: {os.path.basename(input_file)}

Configuration:
    L = {L}  (2:{L} sparsity)
    L_PAD = {L_PAD}  (for vectorization)
    dtype = {DTYPE}
    expand_ratio = {EXPAND_RATIO:.3f}x

DO NOT EDIT MANUALLY
"""

import torch
import triton
import triton.language as tl


# Configuration
L = {L}
L_PAD = {L_PAD}
N = {N}
NUM_WINDOWS = {NUM_WINDOWS}
EXPAND_RATIO = {EXPAND_RATIO}
DTYPE = "{DTYPE}"

{"# FP8 E4M3 max value" if DTYPE == 'fp8' else ""}
{"FP8_E4M3_MAX = 448.0" if DTYPE == 'fp8' else ""}

{config_selector}


@triton.jit
def _fused_quant_slide_kernel(
    x_ptr, y_ptr, scale_ptr,
    M, K_in_orig, K_in, K_out, num_groups,
    stride_x, stride_y,
    BLOCK_GROUPS: tl.constexpr,
):
{kernel_body}

def fused_quant_slide(x: torch.Tensor):
    """
    Fused {DTYPE.upper()} Quantization + SlideSparse Slide
    
    Args:
        x: Input tensor [M, K_in], bf16/fp16/fp32
    
    Returns:
        y: Output tensor [M, K_out_padded] ({DTYPE}), padded to 16-byte alignment
        scale: Scale tensor [M], fp32
    """
    assert x.dim() == 2
    M, K_in_orig = x.shape
    
    # 计算 padded K (不实际 pad，在 kernel 里用 mask 处理)
    K_in = ((K_in_orig + L - 1) // L) * L
    
    num_groups = K_in // L
    K_out = num_groups * NUM_WINDOWS * 4
    
    # Pad output to 16-byte alignment
    K_out_padded = ((K_out + 15) // 16) * 16
    y = torch.empty(M, K_out_padded, dtype={dtype_torch}, device=x.device)
    if K_out_padded > K_out:
        y[:, K_out:].zero_()
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # 获取最优配置
    BLOCK_GROUPS, num_warps, num_stages = _get_config(num_groups)
    
    # 传递 K_in_orig 给 kernel，在 kernel 里用 mask 处理越界访问
    _fused_quant_slide_kernel[(M,)](
        x, y, scale, M, K_in_orig, K_in, K_out, num_groups, x.stride(0), y.stride(0),
        BLOCK_GROUPS=BLOCK_GROUPS, num_warps=num_warps, num_stages=num_stages,
    )
    return y, scale


def get_config():
    return {{'L': L, 'L_PAD': L_PAD, 'N': N, 'NUM_WINDOWS': NUM_WINDOWS, 'EXPAND_RATIO': EXPAND_RATIO, 'DTYPE': DTYPE}}


__all__ = ["fused_quant_slide", "get_config"]
'''
    return kernel_code


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune slide kernel and generate fixed-config version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 autotune_slide_kernels.py slide_L6_fp8.py
    python3 autotune_slide_kernels.py slide_L8_int8.py -o my_tuned.py
    python3 autotune_slide_kernels.py my_custom_kernel.py
        """
    )
    
    parser.add_argument('input_file', type=str,
                        help='Input kernel file (generated by codegen_unified.py)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: <input>_autotuned.py)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.getcwd(), input_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    # 确定输出文件
    if args.output:
        output_file = args.output
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.path.dirname(input_file), output_file)
    else:
        base_name = Path(input_file).stem
        output_file = os.path.join(os.path.dirname(input_file), f"{base_name}_autotuned.py")
    
    print("=" * 80)
    print("SlideSparse Kernel Auto-Tuning Script")
    print("=" * 80)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    
    # Step 1: 解析配置
    print("\nStep 1: Parsing configuration from input file...")
    config = parse_config_from_file(input_file)
    print(f"  L = {config.get('L', 'N/A')}")
    print(f"  DTYPE = {config.get('DTYPE', 'N/A')}")
    print(f"  L_PAD = {config.get('L_PAD', 'N/A')}")
    print(f"  NUM_WINDOWS = {config.get('NUM_WINDOWS', 'N/A')}")
    
    if 'L' not in config:
        print("Error: Cannot find L configuration in input file")
        return 1
    
    # Step 2: 加载模块并运行调优
    print("\nStep 2: Loading module and running autotune...")
    try:
        module = load_module_from_file(input_file)
    except Exception as e:
        print(f"Error loading module: {e}")
        return 1
    
    results = run_tuning(module, config)
    if results is None:
        return 1
    
    # Step 3: 分析结果
    print("\nStep 3: Analyzing results...")
    best_configs = analyze_results(results)
    
    # Step 4: 生成代码
    print("\nStep 4: Generating kernel code...")
    kernel_code = generate_kernel_code(input_file, config, best_configs)
    
    with open(output_file, "w") as f:
        f.write(kernel_code)
    
    print(f"\nGenerated kernel saved to: {output_file}")
    print(f"File size: {len(kernel_code)} bytes")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nYou can now use the auto-tuned kernel:")
    print(f"  from {Path(output_file).stem} import fused_quant_slide")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
