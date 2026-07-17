#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 离线权重转换入口脚本

完整流程：检查模型 → Prune → Slide → Compress → 保存

将 HuggingFace 的 compressed-tensor 格式模型（FP8/INT8）转换为
支持 cuSPARSELt 2:4 稀疏加速的格式。

模型名称约定
============
本脚本使用的模型名称需要明确指定（精确匹配，不支持模糊匹配）：

1. --model 参数接受两种格式：
   - 目录名（如 "Qwen2.5-0.5B-FP8"）：直接在 checkpoints/ 下查找
   - Registry key（如 "qwen2.5-0.5b-fp8"）：通过 model_registry 解析

2. 输出目录命名规则：
   - 格式：{输入目录名}-SlideSparse-{Z}_{L}
   - 示例：Qwen2.5-0.5B-FP8-SlideSparse-2_8

与其他脚本的区别：
- model_download.py / throughput_benchmark.py：使用 registry key（如 qwen2.5-0.5b-fp8）
- offline_autotune_algsearch.py：使用 base name（如 Qwen2.5-0.5B），自动按 dtype 扩展
- weight_convert_entry.py（本脚本）：使用完整目录名或 registry key

Usage:
    # 处理单个模型（默认只做 prune+slide，用于在线压缩）
    python weight_convert_entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8
    
    # 处理指定目录
    python weight_convert_entry.py --input /path/to/model --Z 2 --L 8
    
    # 完整流程：prune+slide+compress（离线压缩，输出目录加 -compressed 后缀）
    python weight_convert_entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8 --compress
    
    # 处理所有已下载的 FP8 模型
    python weight_convert_entry.py --all --quant fp8 --Z 2 --L 8
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# 导入本地模块
from utils import (
    SlideSparseConfig,
    CHECKPOINT_DIR,
    CHECKPOINT_SLIDESPARSE_DIR,
    get_model_safetensors_path,
    get_all_safetensors_files,
    get_output_model_dir,
    load_model_config,
    save_model_config,
    save_slidesparse_config,
    copy_non_weight_files,
    load_safetensors,
    save_safetensors,
    is_target_layer,
    detect_weight_dtype,
    verify_ZL_sparsity,
    verify_2to4_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
)

from prune import prune_tensor, quant_and_prune_tensor_bitnet
from slide import slide_tensor
from compress import compress_tensor_offline, compress_tensor_fake, check_compress_available

# 添加项目路径以导入 slidesparse.utils
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from slidesparse.utils import (
        model_registry,
        check_model_downloaded,
        get_model_local_path,
        list_models,
    )
    HAS_MODEL_REGISTRY = True
except ImportError:
    HAS_MODEL_REGISTRY = False
    print_warning("slidesparse.utils not available, --model option disabled")


# =============================================================================
# 完整流程处理
# =============================================================================

def process_single_tensor(
    tensor: torch.Tensor,
    config: SlideSparseConfig,
    mode: str = "magnitude",
    skip_prune: bool = False,
    skip_slide: bool = False,
    skip_compress: bool = False,
    use_real_cusparselt: bool = True,
    bitnet_mode: bool = False,
    output_dtype: str = "int8",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    处理单个权重张量的完整流程
    
    Args:
        tensor: 输入权重 [N, K]
        config: SlideSparse 配置
        mode: 剪枝模式
        skip_prune: 跳过剪枝
        skip_slide: 跳过滑动
        skip_compress: 跳过压缩
        use_real_cusparselt: 使用真实 cuSPARSELt
        bitnet_mode: 是否使用 BitNet 量化+剪枝融合模式（用于 BF16 输入）
        output_dtype: BitNet 模式下的输出数据类型 ("int8", "fp8_e4m3")
        verbose: 详细输出
    
    Returns:
        {
            "tensor": 处理后的张量,
            "metadata": 元数据张量（如果有）,
            "info": 处理信息,
        }
    """
    original_shape = list(tensor.shape)
    original_dtype = detect_weight_dtype(tensor)
    
    result = {
        "original_shape": original_shape,
        "original_dtype": original_dtype,
        "stages": [],
    }
    
    current_tensor = tensor
    scale_tensor = None  # BitNet 模式会产生 scale
    
    # Stage 1: Prune (或 BitNet 模式下的 Quant+Prune)
    if not skip_prune:
        if bitnet_mode or original_dtype in ("bf16", "fp16", "fp32"):
            # BitNet 模式：量化 + 剪枝融合
            if verbose:
                print_info(f"  Stage 1: Quant+Pruning (BitNet {config.Z}:{config.L}, mode={mode})")
            
            current_tensor, scale_tensor = quant_and_prune_tensor_bitnet(
                tensor, config.Z, config.L, mode, output_dtype
            )
            
            # 验证剪枝结果
            tensor_for_verify = current_tensor.float()
            is_valid, valid_ratio = verify_ZL_sparsity(tensor_for_verify, config.Z, config.L)
            
            result["stages"].append({
                "name": "quant_prune",
                "shape": list(current_tensor.shape),
                "output_dtype": output_dtype,
                "ZL_valid": is_valid,
                "ZL_valid_ratio": valid_ratio,
            })
            
            if verbose:
                status = "✓" if is_valid else f"✗ ({valid_ratio:.2%})"
                print_info(f"    Output dtype: {output_dtype}")
                print_info(f"    {config.Z}:{config.L} validation: {status}")
        else:
            # 标准剪枝模式（FP8/INT8 输入）
            if verbose:
                print_info(f"  Stage 1: Pruning ({config.Z}:{config.L}, mode={mode})")
            
            current_tensor = prune_tensor(
                current_tensor, config.Z, config.L, mode
            )
            
            # 验证剪枝结果
            tensor_for_verify = current_tensor.float() if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else current_tensor
            is_valid, valid_ratio = verify_ZL_sparsity(tensor_for_verify, config.Z, config.L)
            
            result["stages"].append({
                "name": "prune",
                "shape": list(current_tensor.shape),
                "ZL_valid": is_valid,
                "ZL_valid_ratio": valid_ratio,
            })
            
            if verbose:
                status = "✓" if is_valid else f"✗ ({valid_ratio:.2%})"
                print_info(f"    {config.Z}:{config.L} validation: {status}")
    
    # Stage 2: Slide
    if not skip_slide:
        if verbose:
            print_info(f"  Stage 2: Sliding (expand_ratio={config.expand_ratio:.3f})")
        
        current_tensor, slide_meta = slide_tensor(current_tensor, config)
        
        # 验证 2:4 稀疏性
        tensor_for_verify = current_tensor.float() if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else current_tensor
        is_valid, violation_ratio = verify_2to4_sparsity(tensor_for_verify)
        
        result["stages"].append({
            "name": "slide",
            "shape": list(current_tensor.shape),
            "2to4_valid": is_valid,
            "2to4_violation_ratio": violation_ratio,
            "slide_metadata": slide_meta,
        })
        
        if verbose:
            status = "✓" if is_valid else f"✗ (violation: {violation_ratio:.2%})"
            print_info(f"    Shape: {original_shape} -> {list(current_tensor.shape)}")
            print_info(f"    2:4 validation: {status}")
    
    # Stage 3: Compress
    metadata_tensor = None
    if not skip_compress:
        if verbose:
            print_info(f"  Stage 3: Compressing (cuSPARSELt={'real' if use_real_cusparselt else 'fake'})")
        
        # 需要转换为 INT8
        if current_tensor.dtype != torch.int8:
            if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                tensor_int8 = current_tensor.float().round().clamp(-127, 127).to(torch.int8)
            else:
                tensor_int8 = current_tensor.round().clamp(-127, 127).to(torch.int8)
        else:
            tensor_int8 = current_tensor
        
        if use_real_cusparselt:
            compressed, compress_meta = compress_tensor_offline(tensor_int8, verbose=False)
            result["stages"].append({
                "name": "compress",
                "compressed_size": compressed.shape[0],
                "compress_metadata": compress_meta,
            })
            current_tensor = compressed
        else:
            values, meta_tensor, compress_info = compress_tensor_fake(tensor_int8, verbose=False)
            result["stages"].append({
                "name": "compress_fake",
                "values_shape": list(values.shape),
                "meta_shape": list(meta_tensor.shape),
                "compress_info": compress_info,
            })
            current_tensor = values
            metadata_tensor = meta_tensor
        
        if verbose:
            if use_real_cusparselt:
                print_info(f"    Compressed: {compressed.shape[0]} bytes")
            else:
                print_info(f"    Values: {list(values.shape)}, Meta: {list(meta_tensor.shape)}")
    
    result["tensor"] = current_tensor
    result["metadata_tensor"] = metadata_tensor
    result["scale_tensor"] = scale_tensor  # BitNet 模式产生的 scale
    result["final_shape"] = list(current_tensor.shape)
    
    return result


def process_model(
    input_dir: Path,
    output_dir: Path,
    config: SlideSparseConfig,
    mode: str = "magnitude",
    skip_prune: bool = False,
    skip_slide: bool = False,
    skip_compress: bool = False,
    use_real_cusparselt: bool = True,
    bitnet_mode: bool = False,
    output_dtype: str = "int8",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    处理完整模型
    
    Args:
        input_dir: 输入模型目录
        output_dir: 输出目录
        config: SlideSparse 配置
        mode: 剪枝模式
        skip_*: 跳过对应阶段
        use_real_cusparselt: 使用真实 cuSPARSELt
        bitnet_mode: 是否使用 BitNet 模式（用于 BF16/FP16 模型）
        output_dtype: BitNet 模式输出数据类型
        verbose: 详细输出
    
    Returns:
        处理报告
    """
    start_time = time.time()
    
    if verbose:
        print_header(f"Processing: {input_dir.name}")
        print_info(f"Config: {config}")
        print_info(f"Mode: {mode}")
        if bitnet_mode:
            print_info(f"BitNet Mode: enabled (output_dtype={output_dtype})")
        print_info(f"Output: {output_dir}")
        print()
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制非权重文件
    if verbose:
        print_info("Copying non-weight files...")
    copied_files = copy_non_weight_files(input_dir, output_dir)
    if verbose:
        print_info(f"  Copied: {', '.join(copied_files[:5])}{'...' if len(copied_files) > 5 else ''}")
    
    # 获取所有 safetensors 文件
    safetensors_files = get_all_safetensors_files(input_dir)
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")
    
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "mode": mode,
        "skip_prune": skip_prune,
        "skip_slide": skip_slide,
        "skip_compress": skip_compress,
        "use_real_cusparselt": use_real_cusparselt,
        "files": [],
        "total_layers_processed": 0,
        "total_layers_skipped": 0,
    }
    
    # 处理每个 safetensors 文件
    for sf_path in safetensors_files:
        if verbose:
            print()
            print_info(f"Processing file: {sf_path.name}")
        
        file_report = {
            "file": sf_path.name,
            "layers": [],
            "skipped": [],
        }
        
        # 加载权重
        weights = load_safetensors(sf_path)
        output_weights = {}
        
        for key, tensor in weights.items():
            # 检查是否为目标层
            if not is_target_layer(key):
                output_weights[key] = tensor
                file_report["skipped"].append(key)
                continue
            
            # 检查是否为 2D 张量
            if tensor.dim() != 2:
                output_weights[key] = tensor
                file_report["skipped"].append(key)
                continue
            
            if verbose:
                dtype_str = detect_weight_dtype(tensor)
                print_info(f"\nLayer: {key}")
                print_info(f"  Input: shape={list(tensor.shape)}, dtype={dtype_str}")
            
            # 处理张量
            result = process_single_tensor(
                tensor,
                config,
                mode=mode,
                skip_prune=skip_prune,
                skip_slide=skip_slide,
                skip_compress=skip_compress,
                use_real_cusparselt=use_real_cusparselt,
                bitnet_mode=bitnet_mode,
                output_dtype=output_dtype,
                verbose=verbose,
            )
            
            # 保存处理后的张量
            output_weights[key] = result["tensor"]
            
            # 如果有元数据张量，也保存
            if result["metadata_tensor"] is not None:
                meta_key = key.replace(".weight", ".weight_meta")
                output_weights[meta_key] = result["metadata_tensor"]
            
            # 如果有 scale 张量（BitNet 模式），也保存
            if result.get("scale_tensor") is not None:
                scale_key = key.replace(".weight", ".weight_scale")
                output_weights[scale_key] = result["scale_tensor"]
            
            file_report["layers"].append({
                "key": key,
                "result": {
                    "original_shape": result["original_shape"],
                    "final_shape": result["final_shape"],
                    "stages": result["stages"],
                }
            })
        
        # 保存输出
        output_sf_path = output_dir / sf_path.name
        if verbose:
            print()
            print_info(f"Saving: {output_sf_path}")
        
        save_safetensors(output_weights, output_sf_path)
        
        report["files"].append(file_report)
        report["total_layers_processed"] += len(file_report["layers"])
        report["total_layers_skipped"] += len(file_report["skipped"])
    
    # 保存 SlideSparse 配置
    save_slidesparse_config(config, output_dir, extra_info={
        "mode": mode,
        "skip_prune": skip_prune,
        "skip_slide": skip_slide,
        "skip_compress": skip_compress,
    })
    
    elapsed_time = time.time() - start_time
    report["elapsed_time"] = elapsed_time
    
    # 保存处理报告
    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    if verbose:
        print()
        print_header("Summary")
        print_success(f"Processed: {report['total_layers_processed']} layers")
        print_info(f"Skipped: {report['total_layers_skipped']} layers")
        print_info(f"Time: {elapsed_time:.2f}s")
        print_info(f"Report: {report_path}")
    
    return report


# =============================================================================
# 模型查找（基于顶层 utils 的 model_registry）
# =============================================================================

def find_model_dir(model_key: str) -> Optional[Path]:
    """
    根据模型标识查找本地目录
    
    支持以下两种输入格式（必须精确匹配）：
    1. 目录名：Qwen2.5-0.5B-FP8（完整的 checkpoint 目录名）
    2. Registry key：qwen2.5-0.5b-fp8（model_registry 中注册的 key）
    
    注意：不支持模糊匹配，避免不确定性。
    
    Args:
        model_key: 模型标识，可以是：
            - 完整目录名（如 "Qwen2.5-0.5B-FP8"）
            - registry key（如 "qwen2.5-0.5b-fp8"）
    
    Returns:
        模型目录路径，如果不存在返回 None
    """
    # 方式 1: 尝试作为目录名直接查找（精确匹配）
    direct_path = CHECKPOINT_DIR / model_key
    if direct_path.is_dir():
        return direct_path
    
    # 方式 2: 尝试通过 model_registry 查找（精确匹配）
    if HAS_MODEL_REGISTRY:
        try:
            # 使用顶层的 get_model_local_path
            local_path = get_model_local_path(model_key, CHECKPOINT_DIR)
            if local_path.is_dir():
                return local_path
        except (KeyError, AttributeError):
            pass
    
    return None


def list_available_models(quant_filter: Optional[str] = None) -> List[Path]:
    """
    列出所有已下载的模型
    
    使用顶层 model_registry 列出模型，然后检查本地是否存在。
    
    Args:
        quant_filter: 量化类型过滤（"fp8", "int8"）
    
    Returns:
        模型目录列表
    """
    models = []
    
    if not CHECKPOINT_DIR.exists():
        return models
    
    # 如果有 model_registry，优先使用它
    if HAS_MODEL_REGISTRY:
        try:
            # 使用顶层的 list_models 获取所有已注册模型
            registered_keys = list_models(quant=quant_filter) if quant_filter else list_models()
            for key in registered_keys:
                try:
                    local_path = get_model_local_path(key, CHECKPOINT_DIR)
                    if local_path.is_dir() and (local_path / "config.json").exists():
                        if get_all_safetensors_files(local_path):
                            models.append(local_path)
                except (KeyError, AttributeError):
                    pass
            if models:
                return sorted(models)
        except Exception:
            pass
    
    # 回退：直接扫描目录
    for d in sorted(CHECKPOINT_DIR.iterdir()):
        if not d.is_dir():
            continue
        
        # 检查是否有 config.json
        if not (d / "config.json").exists():
            continue
        
        # 检查是否有 safetensors 文件
        if not get_all_safetensors_files(d):
            continue
        
        # 过滤
        if quant_filter:
            name_lower = d.name.lower()
            if quant_filter == "fp8" and "fp8" not in name_lower:
                continue
            if quant_filter == "int8" and "int8" not in name_lower:
                continue
        
        models.append(d)
    
    return models


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse 离线权重转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
完整流程:
  1. Prune:    应用 Z:L 稀疏约束 (如 2:8)
  2. Slide:    2:L → 2:4 滑动窗口映射 (K 扩展)
  3. Compress: cuSPARSELt 2:4 压缩 (K 减半)

示例:
  # 处理单个模型（默认只做 prune+slide，用于在线压缩）
  python weight_convert_entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8
  
  # 处理单个模型（通过路径）
  python weight_convert_entry.py --input /path/to/model --Z 2 --L 8
  
  # 完整流程（含压缩，输出目录加 -compressed 后缀）
  python weight_convert_entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8 --compress
  
  # 处理所有 FP8 模型
  python weight_convert_entry.py --all --quant fp8 --Z 2 --L 8

维度变化 (2:8, expand_ratio=1.5):
  原始:     [N, K]
  Slide后:  [N, K×1.5]
  压缩后:   [N, K×0.75]
        """
    )
    
    # 输入选项组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model", "-m",
        type=str,
        help="模型 key 或目录名（如 qwen2.5-0.5b-fp8）",
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="输入模型目录路径",
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="处理所有已下载的模型",
    )
    input_group.add_argument(
        "--list",
        action="store_true",
        help="列出可用模型",
    )
    
    # 配置选项
    parser.add_argument(
        "--Z",
        type=int,
        default=2,
        help="稀疏度分子（默认: 2）",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=8,
        help="稀疏度分母（默认: 8）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["magnitude", "random"],
        default="magnitude",
        help="剪枝模式（默认: magnitude）",
    )
    
    # 过滤选项
    parser.add_argument(
        "--quant",
        type=str,
        choices=["fp8", "int8"],
        help="量化类型过滤（用于 --all）",
    )
    
    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录（默认: checkpoints_slidesparse/）",
    )
    
    # 阶段控制
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="跳过剪枝阶段",
    )
    parser.add_argument(
        "--skip-slide",
        action="store_true",
        help="跳过滑动阶段",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="启用压缩阶段（默认跳过，输出目录加 -compressed 后缀）",
    )
    parser.add_argument(
        "--fake-compress",
        action="store_true",
        help="使用模拟压缩（不调用 cuSPARSELt）",
    )
    
    # BitNet 选项
    parser.add_argument(
        "--bitnet",
        action="store_true",
        help="启用 BitNet 模式（量化+剪枝融合，用于 BF16/FP16 模型）",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        choices=["int8", "fp8_e4m3"],
        default="int8",
        help="BitNet 模式输出数据类型（默认: int8）",
    )
    
    # 其他选项
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    # 处理 --list
    if args.list:
        models = list_available_models(args.quant)
        print_header("Available Models")
        for m in models:
            print(f"  - {m.name}")
        print()
        print_info(f"Total: {len(models)} models")
        return 0
    
    # 创建配置
    config = SlideSparseConfig(Z=args.Z, L=args.L)
    
    # 确定要处理的模型
    if args.all:
        models = list_available_models(args.quant)
        if not models:
            print_error("No models found")
            return 1
        
        if not args.quiet:
            print_header(f"Processing {len(models)} models")
            for m in models:
                print(f"  - {m.name}")
            print()
    
    elif args.model:
        model_dir = find_model_dir(args.model)
        if model_dir is None:
            print_error(f"Model not found: {args.model}")
            print_info("Use --list to see available models")
            return 1
        models = [model_dir]
    
    elif args.input:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print_error(f"Directory not found: {args.input}")
            return 1
        models = [input_path]
    
    else:
        print_error("No input specified")
        return 1
    
    # 检查 cuSPARSELt（仅在启用压缩时需要）
    use_real_cusparselt = not args.fake_compress
    skip_compress = not args.compress  # 默认跳过压缩
    if use_real_cusparselt and args.compress:
        if not check_compress_available():
            print_warning("cuSPARSELt not available, using fake compression")
            use_real_cusparselt = False
    
    # 处理每个模型
    success_count = 0
    failed_models = []
    
    for model_dir in models:
        try:
            # 确定输出目录（如果启用压缩，加上 -compressed 后缀）
            suffix = f"-SlideSparse-{args.Z}_{args.L}"
            if args.compress:
                suffix += "-compressed"
            if args.output:
                output_dir = Path(args.output) / f"{model_dir.name}{suffix}"
            else:
                output_dir = get_output_model_dir(model_dir, suffix=suffix)
            
            report = process_model(
                model_dir,
                output_dir,
                config,
                mode=args.mode,
                skip_prune=args.skip_prune,
                skip_slide=args.skip_slide,
                skip_compress=skip_compress,
                use_real_cusparselt=use_real_cusparselt,
                bitnet_mode=args.bitnet,
                output_dtype=args.output_dtype,
                verbose=not args.quiet,
            )
            
            success_count += 1
            
        except Exception as e:
            print_error(f"Failed to process {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append(model_dir.name)
    
    # 最终总结
    if len(models) > 1 and not args.quiet:
        print()
        print_header("Final Summary")
        print_success(f"Processed: {success_count}/{len(models)} models")
        if failed_models:
            print_error(f"Failed: {', '.join(failed_models)}")
    
    return 0 if not failed_models else 1


if __name__ == "__main__":
    sys.exit(main())
