#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_04_throughput.py - 吞吐量对比测试（torch.compile mode）

分离测试 Prefill 和 Decode 两个阶段的吞吐量：
  - Prefill: 长输入 + 短输出（测试 prompt 处理速度）
  - Decode: 短输入 + 长输出（测试 token 生成速度）

对比路径:
=========
    [vLLM 原生路径]     DISABLE_SLIDESPARSE=1     ← baseline
                              vs
    [SlideSparse 后端]  根据参数选择不同 kernel    ← test

使用方法:

    python3 test_04_throughput.py --use-cutlass
    python3 test_04_throughput.py --use-cublaslt
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_8 


    python3 test_04_throughput.py --use-cusparselt --sparsity 2_4
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_6
    
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_4 --inner-32
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_6 --inner-32
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_8 --inner-32

    启用 SlideSparse 计时诊断: 必须开启eager mode, 计时器本身也会带来明显开销
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cutlass --eager
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cublaslt --eager 
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cusparselt --sparsity 2_8 --eager
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

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

# Prefill 测试配置：
# - M_prefill = num_prompts * prompt_len
# - 目标 M=2048，设置 num_prompts=2, prompt_len=1024 → M=2048
# - 测试约 100 次，warmup=5
PREFILL_CONFIG = {
    "num_prompts": 2,      # 每批 2 个 prompt
    "prompt_len": 1024,    # 每个 prompt 1024 tokens → M_prefill = 2048
    "output_len": 1,       # 最小化 decode 开销
    "warmup": 5,           # 预热次数
    "repeat": 100,         # 测试重复次数
}

# Decode 测试配置：
# - M_decode = num_prompts (batch size)
# - 目标 M=32，设置 num_prompts=32
# - output_len=100 → 约 3200 个 decode step
DECODE_CONFIG = {
    "num_prompts": 32,     # batch size = 32 → M_decode = 32
    "prompt_len": 4,      # 短 prompt，快速跳过 prefill
    "output_len": 100,     # 生成 100 tokens/prompt → 3200 decode steps
    "warmup": 2,           # 预热次数
}


# ============================================================================
# 辅助函数
# ============================================================================

def generate_dummy_prompt(length: int, seed: int = 0) -> str:
    """
    生成指定 token 长度的 dummy prompt
    
    Args:
        length: 目标 token 长度（近似）
        seed: 随机种子，不同 seed 生成不同内容，避免 prefix caching
    """
    import random
    rng = random.Random(seed)
    
    # 使用多样化的词汇，每个词约 1-2 tokens
    words = [
        "Hello", "world", "this", "is", "a", "test", "prompt", "for", "benchmark",
        "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
        "Python", "programming", "language", "machine", "learning", "artificial",
        "intelligence", "deep", "neural", "network", "transformer", "attention",
        "model", "training", "inference", "optimization", "performance", "speed",
    ]
    
    # 随机打乱词汇顺序，确保不同 seed 生成不同的 prompt
    shuffled_words = words.copy()
    rng.shuffle(shuffled_words)
    
    # 生成足够长度的文本
    result_words = []
    for i in range(length):
        result_words.append(shuffled_words[i % len(shuffled_words)])
    
    return " ".join(result_words)


@dataclass
class PhaseResult:
    """单阶段测试结果"""
    total_time_s: float
    input_tokens: int
    output_tokens: int
    
    @property
    def throughput_tps(self) -> float:
        """主要吞吐量指标 (tokens/s)"""
        return self.input_tokens / self.total_time_s if self.total_time_s > 0 else 0


# ============================================================================
# Profile 统计管理
# ============================================================================

def reset_profile_stats():
    """重置 SlideSparse profile 统计（如果启用）"""
    try:
        from slidesparse.core.SlideSparseLinearMethod_FP8 import reset_profile_stats as _reset
        _reset()
    except ImportError:
        pass


def print_profile_stats_with_label(label: str):
    """打印 SlideSparse profile 统计（带标签）"""
    try:
        from slidesparse.core.SlideSparseLinearMethod_FP8 import (
            print_profile_stats as _print_stats,
            _flush_pending_events,
        )
        # 先 flush pending events
        _flush_pending_events()
        print(f"\n{'#' * 80}")
        print(f"# {label}")
        print(f"{'#' * 80}")
        _print_stats()
    except ImportError:
        pass


# ============================================================================
# 核心测试函数
# ============================================================================

def run_phase_test(
    model_path: Path,
    num_prompts: int,
    prompt_len: int,
    output_len: int,
    warmup: int = 2,
    repeat: int = 1,
    reset_profile: bool = True,
    enforce_eager: bool = False,
) -> PhaseResult:
    """
    运行单阶段吞吐量测试
    
    Args:
        model_path: 模型路径
        num_prompts: 请求数量（每次 generate 调用的 batch size）
        prompt_len: 输入 prompt 的 token 长度
        output_len: 输出的 token 长度
        warmup: 预热次数
        repeat: 正式测试重复次数（多次调用 generate）
        reset_profile: 是否在正式测试前重置 profile 统计
    
    Returns:
        PhaseResult
    """
    from vllm import LLM, SamplingParams
    import torch
    
    with cuda_memory_manager():
        llm = LLM(
            model=str(model_path),
            max_model_len=prompt_len + output_len + 64,
            gpu_memory_utilization=0.8,
            disable_log_stats=True,
            enforce_eager=enforce_eager,
        )
        
        # 生成 prompts（每个 prompt 使用不同 seed，避免 prefix caching）
        # 为每次重复生成不同的 prompts
        all_prompts = []
        for r in range(repeat):
            batch_prompts = [
                generate_dummy_prompt(prompt_len, seed=r * num_prompts + i)
                for i in range(num_prompts)
            ]
            all_prompts.append(batch_prompts)
        
        # Warmup 用不同的 prompts（seed 从 100000 开始）
        warmup_prompts = [generate_dummy_prompt(prompt_len, seed=100000+i) for i in range(num_prompts)]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=output_len,
            ignore_eos=True,  # 强制生成指定数量的 token
        )
        
        # Warmup（预热不计入统计）
        for _ in range(warmup):
            _ = llm.generate(warmup_prompts, sampling_params)
        torch.cuda.synchronize()
        
        # 重置 profile 统计（warmup 后、正式测试前）
        if reset_profile:
            reset_profile_stats()
        
        # 正式测试
        total_input_tokens = 0
        total_output_tokens = 0
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for r in range(repeat):
            outputs = llm.generate(all_prompts[r], sampling_params)
            total_input_tokens += sum(len(o.prompt_token_ids) for o in outputs)
            total_output_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        del llm
    
    return PhaseResult(
        total_time_s=elapsed,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def format_speedup(speedup: float) -> str:
    """格式化加速比显示"""
    speedup_str = f"{speedup:.3f}x"
    if speedup > 1.02:
        return Colors.green(speedup_str + " ↑")
    elif speedup < 0.98:
        return Colors.red(speedup_str + " ↓")
    else:
        return Colors.yellow(speedup_str + " ≈")


# ============================================================================
# 吞吐量对比测试
# ============================================================================

def run_throughput_comparison(
    model_path: Path,
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_32: bool = False,
    sparsity: str = None,
    verbose: bool = True,
    baseline_model_path: Path = None,
    enforce_eager: bool = False,
) -> Dict[str, Any]:
    """
    运行完整的吞吐量对比测试（分离 Prefill 和 Decode）
    
    Args:
        model_path: 测试模型路径
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt
        use_cusparselt: SlideSparse 后端是否使用 cuSPARSELt
        inner_32: 是否使用高精度累加（FP8→FP32, INT8→INT32）
        sparsity: 稀疏配置 (如 "2_8", "2_6")，仅 cuSPARSELt 有效
        verbose: 是否打印详细信息
        baseline_model_path: baseline 模型路径（若不同于 model_path）
        enforce_eager: 是否强制使用 eager mode
    
    Returns:
        对比结果字典
    """
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity)
    results = {}
    
    # 确定 baseline 模型路径
    baseline_path = baseline_model_path if baseline_model_path else model_path
    
    # 预检测 CUTLASS FP8 是否支持当前 GPU
    # baseline 使用 vLLM 原生路径（CUTLASS），如果不支持则跳过 baseline
    cutlass_supported = EnvironmentChecker.supports_cutlass_fp8()
    
    if verbose:
        print("\n" + "=" * 90)
        if cutlass_supported:
            print(Colors.bold("vLLM 原生路径 vs SlideSparse 吞吐量对比"))
        else:
            print(Colors.bold(f"SlideSparse 吞吐量测试 ({backend_name})"))
        print("=" * 90)
        if baseline_model_path:
            print(f"基准模型: {baseline_path.name}")
            print(f"测试模型: {model_path.name}")
        else:
            print(f"模型: {model_path.name}")
        if cutlass_supported:
            print(f"基准: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)")
        else:
            reason = EnvironmentChecker.supports_cutlass_fp8_reason()
            print(Colors.yellow(f"注意: CUTLASS FP8 不支持当前 GPU ({reason})，跳过 baseline"))
        print(f"测试: {backend_name}")
        print("=" * 90)
    
    # ========== Prefill 测试 ==========
    # M_prefill = num_prompts * prompt_len = 2 * 1024 = 2048
    if verbose:
        print(f"\n{Colors.cyan('━' * 50)}")
        print(Colors.bold("Prefill 吞吐量测试"))
        print(f"配置: {PREFILL_CONFIG['num_prompts']} prompts × "
              f"{PREFILL_CONFIG['prompt_len']} input tokens × "
              f"{PREFILL_CONFIG['output_len']} output token × "
              f"{PREFILL_CONFIG['repeat']} repeats")
        print(f"M_prefill = {PREFILL_CONFIG['num_prompts'] * PREFILL_CONFIG['prompt_len']}")
        print(f"{Colors.cyan('━' * 50)}")
    
    # 基准: vLLM 原生路径 (仅当 CUTLASS 支持时)
    baseline_prefill_tps = None
    if cutlass_supported:
        if verbose:
            print(f"\n  {Colors.blue('[基准] vLLM 原生路径...')}")
        saved_env = set_env_for_baseline()
        try:
            baseline_prefill = run_phase_test(
                baseline_path,
                num_prompts=PREFILL_CONFIG["num_prompts"],
                prompt_len=PREFILL_CONFIG["prompt_len"],
                output_len=PREFILL_CONFIG["output_len"],
                warmup=PREFILL_CONFIG["warmup"],
                repeat=PREFILL_CONFIG["repeat"],
                reset_profile=True,
                enforce_eager=enforce_eager,
            )
            baseline_prefill_tps = baseline_prefill.input_tokens / baseline_prefill.total_time_s
        except Exception as e:
            if verbose:
                print(f"    {Colors.yellow(f'跳过 (运行时错误): {e}')}")
        restore_env(saved_env)
    else:
        if verbose:
            print(f"\n  {Colors.yellow('[基准] 跳过 (CUTLASS FP8 不支持当前 GPU)')}")
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] {backend_name}...')}")
    # 从 model_path 提取 model_name 用于加载 model-specific kernels
    model_name = model_path.name  # e.g., "Qwen2.5-0.5B-FP8"
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_32, sparsity, model_name)
    test_prefill = run_phase_test(
        model_path,
        num_prompts=PREFILL_CONFIG["num_prompts"],
        prompt_len=PREFILL_CONFIG["prompt_len"],
        output_len=PREFILL_CONFIG["output_len"],
        warmup=PREFILL_CONFIG["warmup"],
        repeat=PREFILL_CONFIG["repeat"],
        reset_profile=True,
        enforce_eager=enforce_eager,
    )
    test_prefill_tps = test_prefill.input_tokens / test_prefill.total_time_s
    # 打印 Prefill 阶段的 profile 统计
    print_profile_stats_with_label(f"Prefill Profile ({backend_name}) - M={PREFILL_CONFIG['num_prompts'] * PREFILL_CONFIG['prompt_len']}")
    restore_env(saved_env)
    
    if baseline_prefill_tps is not None:
        prefill_speedup = test_prefill_tps / baseline_prefill_tps if baseline_prefill_tps > 0 else 0
        results["prefill"] = {
            "baseline_tps": baseline_prefill_tps,
            "test_tps": test_prefill_tps,
            "speedup": prefill_speedup,
        }
        if verbose:
            print(f"\n  结果:")
            print(f"    vLLM 原生路径: {baseline_prefill_tps:>10.1f} tok/s")
            print(f"    {backend_name}: {test_prefill_tps:>10.1f} tok/s")
            print(f"    加速比: {format_speedup(prefill_speedup)}")
    else:
        results["prefill"] = {"test_tps": test_prefill_tps}
        if verbose:
            print(f"\n  结果:")
            print(f"    {backend_name}: {test_prefill_tps:>10.1f} tok/s")
            print(f"    (无 baseline 对比)")
    
    # ========== Decode 测试 ==========
    # M_decode = num_prompts = 32
    if verbose:
        print(f"\n{Colors.cyan('━' * 50)}")
        print(Colors.bold("Decode 吞吐量测试"))
        print(f"配置: {DECODE_CONFIG['num_prompts']} prompts × "
              f"{DECODE_CONFIG['prompt_len']} input tokens × "
              f"{DECODE_CONFIG['output_len']} output tokens")
        print(f"M_decode = {DECODE_CONFIG['num_prompts']}")
        print(f"{Colors.cyan('━' * 50)}")
    
    # 基准: vLLM 原生路径 (仅当 CUTLASS 支持时)
    baseline_decode_tps = None
    if cutlass_supported:
        if verbose:
            print(f"\n  {Colors.blue('[基准] vLLM 原生路径...')}")
        saved_env = set_env_for_baseline()
        try:
            baseline_decode = run_phase_test(
                baseline_path,
                num_prompts=DECODE_CONFIG["num_prompts"],
                prompt_len=DECODE_CONFIG["prompt_len"],
                output_len=DECODE_CONFIG["output_len"],
                warmup=DECODE_CONFIG["warmup"],
                repeat=1,  # Decode 只需运行一次（output_len 已经很长）
                reset_profile=True,
                enforce_eager=enforce_eager,
            )
            baseline_decode_tps = baseline_decode.output_tokens / baseline_decode.total_time_s
        except Exception as e:
            if verbose:
                print(f"    {Colors.yellow(f'跳过 (运行时错误): {e}')}")
        restore_env(saved_env)
    else:
        if verbose:
            print(f"\n  {Colors.yellow('[基准] 跳过 (CUTLASS FP8 不支持当前 GPU)')}")
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] {backend_name}...')}")
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_32, sparsity, model_name)
    test_decode = run_phase_test(
        model_path,
        num_prompts=DECODE_CONFIG["num_prompts"],
        prompt_len=DECODE_CONFIG["prompt_len"],
        output_len=DECODE_CONFIG["output_len"],
        warmup=DECODE_CONFIG["warmup"],
        repeat=1,
        reset_profile=True,
        enforce_eager=enforce_eager,
    )
    test_decode_tps = test_decode.output_tokens / test_decode.total_time_s
    # 打印 Decode 阶段的 profile 统计
    print_profile_stats_with_label(f"Decode Profile ({backend_name}) - M={DECODE_CONFIG['num_prompts']}")
    restore_env(saved_env)
    
    if baseline_decode_tps is not None:
        decode_speedup = test_decode_tps / baseline_decode_tps if baseline_decode_tps > 0 else 0
        results["decode"] = {
            "baseline_tps": baseline_decode_tps,
            "test_tps": test_decode_tps,
            "speedup": decode_speedup,
        }
        if verbose:
            print(f"\n  结果:")
            print(f"    vLLM 原生路径: {baseline_decode_tps:>10.1f} tok/s")
            print(f"    {backend_name}: {test_decode_tps:>10.1f} tok/s")
            print(f"    加速比: {format_speedup(decode_speedup)}")
    else:
        results["decode"] = {"test_tps": test_decode_tps}
        if verbose:
            print(f"\n  结果:")
            print(f"    {backend_name}: {test_decode_tps:>10.1f} tok/s")
            print(f"    (无 baseline 对比)")
    
    # ========== 总结 ==========
    if verbose:
        print(f"\n{'=' * 90}")
        print(Colors.bold("总结 - 吞吐量对比表格"))
        print(f"{'=' * 90}")
        
        # 表头
        print(f"\n  {'阶段':<12} │ {'Baseline (tok/s)':<18} │ {backend_name + ' (tok/s)':<18} │ {'加速比':<12}")
        print(f"  {'─' * 12}─┼─{'─' * 18}─┼─{'─' * 18}─┼─{'─' * 12}")
        
        # Prefill 行
        prefill_data = results.get("prefill", {})
        if "speedup" in prefill_data:
            baseline_str = f"{prefill_data['baseline_tps']:>14.1f}"
            test_str = f"{prefill_data['test_tps']:>14.1f}"
            speedup_str = format_speedup(prefill_data['speedup'])
        else:
            baseline_str = f"{'N/A':>14}"
            test_str = f"{prefill_data.get('test_tps', 0):>14.1f}"
            speedup_str = "N/A"
        print(f"  {'Prefill':<12} │ {baseline_str:<18} │ {test_str:<18} │ {speedup_str:<12}")
        
        # Decode 行
        decode_data = results.get("decode", {})
        if "speedup" in decode_data:
            baseline_str = f"{decode_data['baseline_tps']:>14.1f}"
            test_str = f"{decode_data['test_tps']:>14.1f}"
            speedup_str = format_speedup(decode_data['speedup'])
        else:
            baseline_str = f"{'N/A':>14}"
            test_str = f"{decode_data.get('test_tps', 0):>14.1f}"
            speedup_str = "N/A"
        print(f"  {'Decode':<12} │ {baseline_str:<18} │ {test_str:<18} │ {speedup_str:<12}")
        
        print(f"  {'─' * 12}─┴─{'─' * 18}─┴─{'─' * 18}─┴─{'─' * 12}")
        
        # 综合加速比
        if "speedup" in prefill_data and "speedup" in decode_data:
            avg_speedup = (prefill_data['speedup'] + decode_data['speedup']) / 2
            print(f"\n  平均加速比: {format_speedup(avg_speedup)}")
        
        print(f"{'=' * 90}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = parse_common_args("吞吐量对比测试（分离 Prefill/Decode）")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="强制使用 eager mode（禁用 torch.compile）"
    )
    args = parser.parse_args()
    
    # 注意：不调用 apply_env_args(args)
    # 因为 run_throughput_comparison 会自己管理环境变量
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 根据参数决定测试的 SlideSparse 后端
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    use_cusparselt = getattr(args, 'use_cusparselt', False)
    inner_32 = getattr(args, 'inner_32', False)
    sparsity = getattr(args, 'sparsity', None)
    
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
            print(Colors.cyan("  python3 test_04_throughput.py --use-cublaslt"))
            print(Colors.cyan("  python3 test_04_throughput.py --use-cusparselt --sparsity 2_8"))
            print(Colors.yellow("=" * 70 + "\n"))
            sys.exit(0)
    
    # 查找模型
    baseline_model_path = None
    model_arg = getattr(args, 'model', None)
    
    if use_cusparselt:
        # 先找原始 FP8 模型作为 baseline
        if model_arg:
            baseline_model_path = ModelFinder.resolve_model_path(model_arg, "FP8")
        else:
            baseline_model_path = ModelFinder.find_small_model("FP8")
        if baseline_model_path is None:
            print(Colors.red("错误: 未找到 FP8 模型 (用于 baseline)"))
            sys.exit(1)
        
        # cuSPARSELt 需要 SlideSparse checkpoint（已预先稀疏化）
        # 如果用户指定了模型，基于该模型查找对应的 SlideSparse checkpoint
        if model_arg:
            model_path = ModelFinder.resolve_slidesparse_model_path(
                baseline_model_path, sparsity
            )
        else:
            model_path = ModelFinder.find_slidesparse_model("FP8", sparsity)
            if model_path is None:
                model_path = ModelFinder.resolve_slidesparse_model_path(
                    baseline_model_path, sparsity
                )
        if model_path is None:
            print(Colors.red(f"错误: 未找到 SlideSparse checkpoint (sparsity={sparsity or '2_8'})"))
            print(Colors.yellow(f"期望路径: checkpoints_slidesparse/{baseline_model_path.name}-SlideSparse-{sparsity or '2_8'}"))
            sys.exit(1)
        print(Colors.cyan(f"Baseline: {baseline_model_path.name}"))
        print(Colors.cyan(f"Test:     {model_path.name}"))
    else:
        # CUTLASS 或 cuBLASLt 路径：使用普通 FP8 checkpoint
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
    
    # 获取 eager 参数
    enforce_eager = getattr(args, 'eager', False)
    
    # 运行吞吐量对比
    run_throughput_comparison(
        model_path=model_path,
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        sparsity=sparsity,
        verbose=True,
        baseline_model_path=baseline_model_path,
        enforce_eager=enforce_eager,
    )
    
    sys.exit(0)
