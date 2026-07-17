#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_03_inference.py - 端到端推理输出对比

对于相同的 prompt，分别用 vLLM 原生路径 和 SlideSparse 后端运行推理，
并排打印输出让用户直观比较精度差异。（eager mode）

对比路径:
=========
    [vLLM 原生路径]     DISABLE_SLIDESPARSE=1     ← baseline
                              vs
    [SlideSparse 后端]  根据参数选择不同 kernel    ← test

使用方法:
    python3 test_03_inference.py --use-cutlass              # 默认: vs CUTLASS fallback
    python3 test_03_inference.py --use-cublaslt             # vs cuBLASLt
    python3 test_03_inference.py --use-cublaslt --inner-32  # cuBLASLt + 高精度累加

    python3 test_03_inference.py --use-cusparselt --sparsity 2_4
    python3 test_03_inference.py --use-cusparselt --sparsity 2_6
    python3 test_03_inference.py --use-cusparselt --sparsity 2_8
    python3 test_03_inference.py --use-cusparselt --sparsity 2_10


    python3 test_03_inference.py --use-cusparselt --sparsity 2_4 --inner-32
    python3 test_03_inference.py --use-cusparselt --sparsity 2_6 --inner-32
    python3 test_03_inference.py --use-cusparselt --sparsity 2_8 --inner-32
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EnvironmentChecker,
    ModelFinder,
    Colors,
    cuda_memory_manager,
    parse_common_args,
    get_backend_name,
    set_env_for_baseline,
    set_env_for_test,
    restore_env,
)


# ============================================================================
# 测试配置
# ============================================================================

# 测试提示词
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
]


# ============================================================================
# 核心功能：vLLM 原生路径 vs SlideSparse 后端 输出对比
# ============================================================================

def run_comparison_inference(
    model_path: Path,
    prompts: List[str],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_32: bool = False,
    sparsity: str = "2_8",
    max_tokens: int = 48,
    verbose: bool = True,
    baseline_model_path: Path = None,
) -> List[Tuple[str, str, str]]:
    """
    运行 vLLM 原生路径 和 SlideSparse 后端推理并对比输出
    
    Args:
        model_path: 测试模型路径（SlideSparse 后端使用）
        prompts: 提示词列表
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt
        use_cusparselt: SlideSparse 后端是否使用 cuSPARSELt
        inner_32: 是否使用高精度累加（FP8→FP32, INT8→INT32）
        sparsity: 稀疏格式（仅 cuSPARSELt 时生效）
        max_tokens: 最大生成 token 数
        verbose: 是否打印详细信息
        baseline_model_path: baseline 模型路径（若不同于 model_path）
    
    Returns:
        List of (prompt, baseline_output, test_output)
        如果 baseline 不可用，baseline_output 为 None
    """
    from vllm import LLM, SamplingParams
    
    results = []
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity)
    
    # 检测 CUTLASS 是否支持当前 GPU
    cutlass_supported = EnvironmentChecker.supports_cutlass_fp8()
    
    # 确定 baseline 模型路径
    # 对于 cuSPARSELt，baseline 使用原始 FP8 模型；test 使用 SlideSparse checkpoint
    baseline_path = baseline_model_path if baseline_model_path else model_path
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪采样确保可复现
        max_tokens=max_tokens,
    )
    
    if verbose:
        print("\n" + "=" * 80)
        if cutlass_supported:
            print(Colors.bold("vLLM 原生路径 vs SlideSparse 推理输出对比"))
        else:
            print(Colors.bold(f"{backend_name} 推理测试"))
        print("=" * 80)
        if use_cusparselt and baseline_model_path:
            print(f"基准模型: {baseline_path.name}")
            print(f"测试模型: {model_path.name}")
        else:
            print(f"模型: {model_path.name}")
        if cutlass_supported:
            print(f"基准: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)")
        else:
            cc = EnvironmentChecker.cuda_compute_capability()
            print(Colors.yellow(f"注意: CUTLASS 不支持当前 GPU (sm_{cc[0]}{cc[1]})，跳过 baseline"))
        print(f"测试: {backend_name}")
        print(f"采样: temperature=0.0, max_tokens={max_tokens}")
        print("=" * 80)
    
    baseline_texts = None
    
    # 1. 运行 vLLM 原生路径 (基准)
    # 对于 cuSPARSELt，使用原始 FP8 模型作为 baseline
    if cutlass_supported:
        if verbose:
            print(f"\n{Colors.cyan('[1/2] 运行 vLLM 原生路径 (基准)...')}")
        
        saved_env = set_env_for_baseline()
        
        try:
            with cuda_memory_manager():
                llm_baseline = LLM(
                    model=str(baseline_path),
                    max_model_len=256,
                    gpu_memory_utilization=0.45,
                    disable_log_stats=True,
                    enforce_eager=True,
                )
                
                outputs_baseline = llm_baseline.generate(prompts, sampling_params)
                baseline_texts = [o.outputs[0].text.strip() for o in outputs_baseline]
                
                del llm_baseline
        except RuntimeError as e:
            if "Error Internal" in str(e) or "cutlass" in str(e).lower():
                if verbose:
                    print(Colors.yellow(f"  CUTLASS 运行失败: {e}"))
                    print(Colors.yellow("  跳过 baseline 对比"))
                baseline_texts = None
            else:
                raise
        
        restore_env(saved_env)
    
    # 2. 运行 SlideSparse 后端 (测试)
    if verbose:
        step = "[1/1]" if baseline_texts is None else "[2/2]"
        print(f"\n{Colors.cyan(f'{step} 运行 {backend_name}...')}")
    
    # 从 model_path 提取 model_name 用于加载 model-specific kernels
    model_name = model_path.name  # e.g., "Qwen2.5-0.5B-FP8"
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_32, sparsity, model_name)
    
    with cuda_memory_manager():
        llm_test = LLM(
            model=str(model_path),
            max_model_len=256,
            gpu_memory_utilization=0.45,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        outputs_test = llm_test.generate(prompts, sampling_params)
        test_texts = [o.outputs[0].text.strip() for o in outputs_test]
        
        del llm_test
    
    restore_env(saved_env)
    
    # 3. 打印对比结果
    if verbose:
        print("\n" + "=" * 80)
        if baseline_texts is not None:
            print(Colors.bold("输出对比"))
        else:
            print(Colors.bold(f"{backend_name} 输出"))
        print("=" * 80)
        
        for i, prompt in enumerate(prompts):
            test_out = test_texts[i]
            baseline_out = baseline_texts[i] if baseline_texts else None
            
            print(f"\n{Colors.bold(f'[Prompt {i+1}]')} {prompt}")
            print("-" * 80)
            
            if baseline_out is not None:
                print(f"{Colors.blue('vLLM 原生:')} {baseline_out}")
                print()
                print(f"{Colors.green(f'{backend_name}:')} {test_out}")
                
                # 比较是否完全相同
                if baseline_out == test_out:
                    print(f"\n  {Colors.green('✓ 输出完全一致')}")
                else:
                    print(f"\n  {Colors.yellow('⚠ 输出有差异（FP8 精度正常）')}")
                
                results.append((prompt, baseline_out, test_out))
            else:
                # 无 baseline，只显示 test 输出
                print(f"{Colors.green(f'{backend_name}:')} {test_out}")
                results.append((prompt, None, test_out))
        
        print("\n" + "=" * 80)
        
        # 统计
        if baseline_texts is not None:
            identical = sum(1 for _, b, t in results if b == t)
            print(f"统计: {identical}/{len(results)} 个输出完全一致")
        else:
            print(f"已完成 {len(results)} 个推理测试 (无 baseline 对比)")
        print("=" * 80)
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = parse_common_args("端到端推理输出对比")
    args = parser.parse_args()
    
    # 注意：这里不调用 apply_env_args(args)
    # 因为 run_comparison_inference 会自己管理环境变量
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 根据参数决定测试的 SlideSparse 后端
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    use_cusparselt = getattr(args, 'use_cusparselt', False)
    inner_32 = getattr(args, 'inner_32', False)
    sparsity = getattr(args, 'sparsity', '2_8')
    
    # ========== 预拦截：CUTLASS 支持检测 ==========
    # 当用户不指定 --use-cublaslt 或 --use-cusparselt 时，默认使用 CUTLASS
    # 如果当前 GPU 不支持 vLLM CUTLASS FP8，需要提示用户切换后端
    if not use_cublaslt and not use_cusparselt:
        cutlass_supported = EnvironmentChecker.supports_cutlass_fp8()
        if not cutlass_supported:
            reason = EnvironmentChecker.supports_cutlass_fp8_reason()
            print(Colors.yellow("\n" + "=" * 70))
            print(Colors.yellow("预拦截: vLLM CUTLASS FP8 不支持当前 GPU"))
            print(Colors.yellow("=" * 70))
            print(Colors.yellow(f"原因: {reason}"))
            print(Colors.yellow("\n请使用以下替代方案:"))
            print(Colors.cyan("  python3 test_03_inference.py --use-cublaslt"))
            print(Colors.cyan("  python3 test_03_inference.py --use-cusparselt --sparsity 2_8"))
            print(Colors.yellow("=" * 70 + "\n"))
            sys.exit(0)
    
    # 查找模型
    baseline_model_path = None
    model_arg = getattr(args, 'model', None)
    
    # 对于 cuSPARSELt 路径，需要查找 SlideSparse 转换后的 checkpoint
    if use_cusparselt:
        # 先找原始 FP8 模型作为 baseline
        if model_arg:
            baseline_model_path = ModelFinder.resolve_model_path(model_arg, "FP8")
        else:
            baseline_model_path = ModelFinder.find_small_model("FP8")
        if baseline_model_path is None:
            print(Colors.red("错误: 未找到 FP8 模型 (用于 baseline)"))
            sys.exit(1)
        
        # 再找 SlideSparse checkpoint 作为测试模型
        # 如果用户指定了模型，基于该模型查找对应的 SlideSparse checkpoint
        if model_arg:
            # 基于用户指定的模型查找对应的 SlideSparse checkpoint
            model_path = ModelFinder.resolve_slidesparse_model_path(
                baseline_model_path, sparsity
            )
        else:
            # 自动查找
            model_path = ModelFinder.find_slidesparse_model("FP8", sparsity)
            if model_path is None:
                model_path = ModelFinder.resolve_slidesparse_model_path(
                    baseline_model_path, sparsity
                )
        if model_path is None:
            print(Colors.red(f"错误: 未找到 FP8 SlideSparse-{sparsity} 模型"))
            print(Colors.yellow(f"请确保 checkpoints_slidesparse/ 目录下存在对应的 checkpoint"))
            print(Colors.yellow(f"期望路径: checkpoints_slidesparse/{baseline_model_path.name}-SlideSparse-{sparsity}"))
            sys.exit(1)
        print(Colors.cyan(f"Baseline: {baseline_model_path.name}"))
        print(Colors.cyan(f"Test:     {model_path.name}"))
    else:
        # CUTLASS 或 cuBLASLt 路径：baseline 和 test 使用同一个模型
        # 优先使用 --model 参数
        if model_arg:
            model_path = ModelFinder.resolve_model_path(model_arg, "FP8")
            if model_path is None:
                print(Colors.red(f"错误: 未找到模型 {model_arg}"))
                sys.exit(1)
        else:
            model_path = ModelFinder.find_small_model("FP8")
            if model_path is None:
                print(Colors.red("错误: 未找到 FP8 模型"))
                sys.exit(1)
    
    # 运行输出对比
    run_comparison_inference(
        model_path=model_path,
        prompts=TEST_PROMPTS,
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        sparsity=sparsity,
        max_tokens=64,
        verbose=True,
        baseline_model_path=baseline_model_path,
    )
    
    sys.exit(0)
