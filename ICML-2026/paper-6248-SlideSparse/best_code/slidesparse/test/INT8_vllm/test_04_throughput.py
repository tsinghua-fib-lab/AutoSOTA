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

架构说明:
=========
使用 subprocess 隔离每个测试阶段，解决 CUDA Context 无法彻底清理导致的 OOM 问题。
每个测试（baseline/test × prefill/decode）都在独立的子进程中运行。

使用方法:

    python3 test_04_throughput.py --use-cutlass 
    python3 test_04_throughput.py --use-cublaslt
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_8


    python3 test_04_throughput.py --use-cusparselt --sparsity 2_4
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_6
    
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_4 --inner-32
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_6 --inner-32
    python3 test_04_throughput.py --use-cusparselt --sparsity 2_8 --inner-32

    --show-subprocess-output 打印子进程输出

    启用 SlideSparse 计时诊断: 必须开启eager mode, 计时器本身也会带来明显开销
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cutlass --eager
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cublaslt --eager 
    SLIDESPARSE_PROFILE=1 python3 test_04_throughput.py --use-cusparselt --sparsity 2_8 --eager
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EnvironmentChecker,
    ModelFinder,
    Colors,
    parse_common_args,
    get_backend_name,
    extract_model_name,
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
# Subprocess 隔离运行
# ============================================================================

_SUBPROCESS_SCRIPT_TEMPLATE = '''
import os
import sys
import time
import json
import random

# 设置环境变量
for k, v in {env_vars}.items():
    os.environ[k] = v
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import torch
from vllm import LLM, SamplingParams

# Profile 统计管理（仅 eager mode + SLIDESPARSE_PROFILE=1 时有效）
def reset_profile_stats():
    """重置 SlideSparse profile 统计（如果启用）"""
    try:
        from slidesparse.core.SlideSparseLinearMethod_FP8 import reset_profile_stats as _reset
        _reset()
    except ImportError:
        pass

def print_profile_stats_with_label(label):
    """打印 SlideSparse profile 统计（带标签）"""
    try:
        from slidesparse.core.SlideSparseLinearMethod_FP8 import (
            print_profile_stats as _print_stats,
            _flush_pending_events,
        )
        # 先 flush pending events
        _flush_pending_events()
        print("\\n" + "#" * 80)
        print("# " + label)
        print("#" * 80)
        _print_stats()
    except ImportError:
        pass

def generate_dummy_prompt(length, seed=0):
    rng = random.Random(seed)
    words = [
        "Hello", "world", "this", "is", "a", "test", "prompt", "for", "benchmark",
        "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
        "Python", "programming", "language", "machine", "learning", "artificial",
        "intelligence", "deep", "neural", "network", "transformer", "attention",
        "model", "training", "inference", "optimization", "performance", "speed",
    ]
    shuffled = words.copy()
    rng.shuffle(shuffled)
    return " ".join([shuffled[i % len(shuffled)] for i in range(length)])

# 配置
model_path = "{model_path}"
config = {config}
enforce_eager = {enforce_eager}

# 创建 LLM
# 计算目标 M 值：Prefill 阶段 M = num_prompts * prompt_len，Decode 阶段 M = num_prompts
target_M = config["num_prompts"] * config["prompt_len"] if config["output_len"] == 1 else config["num_prompts"]

# vLLM 配置约束:
# 1. max_model_len >= prompt_len + output_len (否则无法处理请求)
# 2. max_num_batched_tokens >= max_model_len (配置验证要求)
#
# 实际 M 值控制逻辑:
# - max_num_batched_tokens 只是“允许的上限”，不是强制值
# - 实际 M 由 max_num_seqs 和请求状态决定:
#   - Prefill: M = max_num_seqs * prompt_len = num_prompts * prompt_len = target_M ✅
#   - Decode:  M = max_num_seqs (每序列生成 1 token) = num_prompts = target_M ✅
# - enable_chunked_prefill=False 确保 prompt 不被分片
#
# 因此，即使 effective_max_num_batched_tokens > target_M，实际 M 仍然等于 target_M
MODEL_LEN_BUFFER = 16  # 预留 buffer 处理 BOS token 等额外开销
min_model_len = config["prompt_len"] + config["output_len"] + MODEL_LEN_BUFFER
effective_max_model_len = min_model_len
effective_max_num_batched_tokens = max(target_M, min_model_len)  # 仅用于通过配置验证

llm = LLM(
    model=model_path,
    max_model_len=effective_max_model_len,
    gpu_memory_utilization=0.8,
    disable_log_stats=True,
    enforce_eager=enforce_eager,
    max_num_batched_tokens=effective_max_num_batched_tokens,
    max_num_seqs=config["num_prompts"],  # ← 这才是控制实际 M 的关键参数
    enable_chunked_prefill=False,
)

# 生成 prompts
repeat = config.get("repeat", 1)
all_prompts = []
for r in range(repeat):
    batch = [generate_dummy_prompt(config["prompt_len"], seed=r * config["num_prompts"] + i) 
             for i in range(config["num_prompts"])]
    all_prompts.append(batch)

warmup_prompts = [generate_dummy_prompt(config["prompt_len"], seed=100000 + i) 
                  for i in range(config["num_prompts"])]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=config["output_len"],
    ignore_eos=True,
)

# Warmup
for _ in range(config["warmup"]):
    _ = llm.generate(warmup_prompts, sampling_params)
torch.cuda.synchronize()

# 重置 profile 统计（warmup 后、正式测试前）
reset_profile_stats()

# 正式测试
total_input = 0
total_output = 0
torch.cuda.synchronize()
start = time.perf_counter()

for r in range(repeat):
    outputs = llm.generate(all_prompts[r], sampling_params)
    total_input += sum(len(o.prompt_token_ids) for o in outputs)
    total_output += sum(len(o.outputs[0].token_ids) for o in outputs)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# 打印 profile 统计（仅当 SLIDESPARSE_PROFILE=1 且 enforce_eager=True 时有效）
if enforce_eager and os.environ.get("SLIDESPARSE_PROFILE") == "1":
    phase_label = "{phase_name}"
    M_value = config["num_prompts"] * config["prompt_len"] if config["output_len"] == 1 else config["num_prompts"]
    print_profile_stats_with_label(phase_label + " Profile - M=" + str(M_value))

# 输出结果 JSON（通过特殊标记让主进程识别）
result = {{
    "total_time_s": elapsed,
    "input_tokens": total_input,
    "output_tokens": total_output,
}}
print("SUBPROCESS_RESULT_JSON:" + json.dumps(result))
'''


def run_phase_in_subprocess(
    phase_name: str,
    model_path: Path,
    config: dict,
    env_vars: dict,
    enforce_eager: bool = False,
    verbose: bool = True,
    show_subprocess_output: bool = False,
) -> Optional[PhaseResult]:
    """
    在独立子进程中运行测试阶段
    
    解决 CUDA Context 无法彻底清理导致的 OOM 问题。
    
    Args:
        phase_name: 阶段名称（用于日志）
        model_path: 模型路径
        config: 测试配置
        env_vars: 环境变量
        enforce_eager: 是否使用 eager mode
        verbose: 是否打印详细信息
        show_subprocess_output: 是否显示子进程的完整输出（调试用）
    
    Returns:
        PhaseResult 或 None（如果失败）
    """
    # 构建脚本
    script = _SUBPROCESS_SCRIPT_TEMPLATE.format(
        env_vars=repr(env_vars),
        model_path=str(model_path),
        config=repr(config),
        enforce_eager=enforce_eager,
        phase_name=phase_name,
    )
    
    # 准备环境
    env = os.environ.copy()
    env.update(env_vars)
    env["VLLM_LOGGING_LEVEL"] = "ERROR"
    # 传递 SLIDESPARSE_PROFILE 环境变量（如果设置了）
    if "SLIDESPARSE_PROFILE" in os.environ:
        env["SLIDESPARSE_PROFILE"] = os.environ["SLIDESPARSE_PROFILE"]
    
    if verbose:
        print(f"    启动子进程...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,  # 10 分钟超时
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"    {Colors.red('子进程失败')}")
                # 如果启用了 show_subprocess_output，打印完整输出
                if show_subprocess_output:
                    if result.stdout.strip():
                        print(Colors.yellow("      [stdout]"))
                        for line in result.stdout.split("\n"):
                            if line.strip():
                                print(f"      {line}")
                    if result.stderr.strip():
                        print(Colors.yellow("      [stderr]"))
                        for line in result.stderr.split("\n"):
                            if line.strip():
                                print(f"      {line}")
                else:
                    # 默认只打印最后几行错误
                    stderr_lines = result.stderr.strip().split('\n')
                    for line in stderr_lines[-10:]:
                        print(f"      {line}")
            return None
        
        # 打印子进程的输出
        # 1. 如果 show_subprocess_output=True，打印所有输出（调试用）
        # 2. 如果 SLIDESPARSE_PROFILE=1 且 enforce_eager，打印 profile 统计
        should_print = show_subprocess_output or (os.environ.get("SLIDESPARSE_PROFILE") == "1" and enforce_eager)
        if verbose and should_print:
            for line in result.stdout.split("\n"):
                if not line.startswith("SUBPROCESS_RESULT_JSON:"):
                    if line.strip():
                        print(f"      {line}")
            # 也打印 stderr（如果有内容且不是纯空白）
            if show_subprocess_output and result.stderr.strip():
                print(Colors.yellow("      [stderr]"))
                for line in result.stderr.split("\n"):
                    if line.strip():
                        print(f"      {line}")
        
        # 解析结果
        for line in result.stdout.split("\n"):
            if line.startswith("SUBPROCESS_RESULT_JSON:"):
                data = json.loads(line[len("SUBPROCESS_RESULT_JSON:"):])
                return PhaseResult(
                    total_time_s=data["total_time_s"],
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                )
        
        if verbose:
            print(f"    {Colors.red('未找到结果')}")
        return None
        
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"    {Colors.red('子进程超时')}")
        return None
    except Exception as e:
        if verbose:
            print(f"    {Colors.red(f'子进程异常: {e}')}")
        return None


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
    show_subprocess_output: bool = False,
) -> Dict[str, Any]:
    """
    运行完整的吞吐量对比测试（分离 Prefill 和 Decode）
    
    使用 subprocess 隔离每个测试阶段，避免 CUDA Context 残留导致的 OOM。
    
    Args:
        model_path: 测试模型路径
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt
        use_cusparselt: SlideSparse 后端是否使用 cuSPARSELt
        inner_32: 是否使用高精度累加（FP8→FP32, INT8→INT32）
        sparsity: 稀疏配置 (如 "2_8", "2_6")，仅 cuSPARSELt 有效
        verbose: 是否打印详细信息
        baseline_model_path: baseline 模型路径（若不同于 model_path）
        enforce_eager: 是否强制使用 eager mode
        show_subprocess_output: 是否显示子进程的完整输出（调试用）
    
    Returns:
        对比结果字典
    """
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_32, sparsity)
    results = {}
    
    # 确定 baseline 模型路径
    baseline_path = baseline_model_path if baseline_model_path else model_path
    # 提取基础模型名（去除 -SlideSparse- 后缀）用于查找 GEMM 配置和 Triton kernel
    # 例如 Llama3.2-1B-INT8-SlideSparse-2_10 -> Llama3.2-1B-INT8
    model_name = extract_model_name(model_path.name)
    
    # 预检测 CUTLASS INT8 是否支持当前 GPU
    cutlass_supported = EnvironmentChecker.supports_cutlass_int8()
    
    # 构建环境变量
    baseline_env = {"DISABLE_SLIDESPARSE": "1"}
    
    test_env = {"DISABLE_SLIDESPARSE": "0", "SLIDESPARSE_MODEL_NAME": model_name}
    if use_cublaslt:
        test_env["USE_CUBLASLT"] = "1"
    if use_cusparselt:
        test_env["USE_CUSPARSELT"] = "1"
    if inner_32:
        test_env["INNER_DTYPE_32"] = "1"
    if sparsity:
        test_env["SPARSITY"] = sparsity
    
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
            reason = EnvironmentChecker.supports_cutlass_int8_reason()
            print(Colors.yellow(f"注意: CUTLASS INT8 不支持当前 GPU ({reason})，跳过 baseline"))
        print(f"测试: {backend_name}")
        print(Colors.cyan("架构: 使用 subprocess 隔离每个测试阶段"))
        print("=" * 90)
    
    # ========== Prefill 测试 ==========
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
        baseline_prefill = run_phase_in_subprocess(
            "baseline_prefill",
            baseline_path,
            PREFILL_CONFIG,
            baseline_env,
            enforce_eager,
            verbose,
            show_subprocess_output,
        )
        if baseline_prefill:
            baseline_prefill_tps = baseline_prefill.input_tokens / baseline_prefill.total_time_s
    else:
        if verbose:
            print(f"\n  {Colors.yellow('[基准] 跳过 (CUTLASS INT8 不支持当前 GPU)')}")
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"\n  {Colors.green(f'[测试] {backend_name}...')}")
    test_prefill = run_phase_in_subprocess(
        "test_prefill",
        model_path,
        PREFILL_CONFIG,
        test_env,
        enforce_eager,
        verbose,
        show_subprocess_output,
    )
    
    if test_prefill:
        test_prefill_tps = test_prefill.input_tokens / test_prefill.total_time_s
        
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
    else:
        if verbose:
            print(f"\n  {Colors.red('Prefill 测试失败')}")
    
    # ========== Decode 测试 ==========
    if verbose:
        print(f"\n{Colors.cyan('━' * 50)}")
        print(Colors.bold("Decode 吞吐量测试"))
        print(f"配置: {DECODE_CONFIG['num_prompts']} prompts × "
              f"{DECODE_CONFIG['prompt_len']} input tokens × "
              f"{DECODE_CONFIG['output_len']} output tokens")
        print(f"M_decode = {DECODE_CONFIG['num_prompts']}")
        print(f"{Colors.cyan('━' * 50)}")
    
    # Decode 配置（repeat=1）
    decode_config = {**DECODE_CONFIG, "repeat": 1}
    
    # 基准: vLLM 原生路径 (仅当 CUTLASS 支持时)
    baseline_decode_tps = None
    if cutlass_supported:
        if verbose:
            print(f"\n  {Colors.blue('[基准] vLLM 原生路径...')}")
        baseline_decode = run_phase_in_subprocess(
            "baseline_decode",
            baseline_path,
            decode_config,
            baseline_env,
            enforce_eager,
            verbose,
            show_subprocess_output,
        )
        if baseline_decode:
            baseline_decode_tps = baseline_decode.output_tokens / baseline_decode.total_time_s
    else:
        if verbose:
            print(f"\n  {Colors.yellow('[基准] 跳过 (CUTLASS INT8 不支持当前 GPU)')}")
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"\n  {Colors.green(f'[测试] {backend_name}...')}")
    test_decode = run_phase_in_subprocess(
        "test_decode",
        model_path,
        decode_config,
        test_env,
        enforce_eager,
        verbose,
        show_subprocess_output,
    )
    
    if test_decode:
        test_decode_tps = test_decode.output_tokens / test_decode.total_time_s
        
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
    else:
        if verbose:
            print(f"\n  {Colors.red('Decode 测试失败')}")
    
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
            test_str = f"{prefill_data.get('test_tps', 0):>14.1f}" if prefill_data else f"{'FAIL':>14}"
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
            test_str = f"{decode_data.get('test_tps', 0):>14.1f}" if decode_data else f"{'FAIL':>14}"
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
    parser.add_argument(
        "--show-subprocess-output",
        action="store_true",
        help="显示子进程的完整输出（调试用，包含模型加载日志等）"
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
    # 如果当前 GPU 不支持 vLLM CUTLASS INT8，需要提示用户切换后端
    if not use_cublaslt and not use_cusparselt:
        cutlass_supported = EnvironmentChecker.supports_cutlass_int8()
        if not cutlass_supported:
            reason = EnvironmentChecker.supports_cutlass_int8_reason()
            print(Colors.yellow("\n" + "=" * 70))
            print(Colors.yellow("预拦截: vLLM CUTLASS INT8 不支持当前 GPU"))
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
        # 先找原始 INT8 模型作为 baseline
        if model_arg:
            baseline_model_path = ModelFinder.resolve_model_path(model_arg, "INT8")
        else:
            baseline_model_path = ModelFinder.find_small_model("INT8")
        if baseline_model_path is None:
            print(Colors.red("错误: 未找到 INT8 模型 (用于 baseline)"))
            sys.exit(1)
        
        # cuSPARSELt 需要 SlideSparse checkpoint（已预先稀疏化）
        # 如果用户指定了模型，基于该模型查找对应的 SlideSparse checkpoint
        if model_arg:
            model_path = ModelFinder.resolve_slidesparse_model_path(
                baseline_model_path, sparsity
            )
        else:
            model_path = ModelFinder.find_slidesparse_model("INT8", sparsity)
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
        # CUTLASS 或 cuBLASLt 路径：使用普通 INT8 checkpoint
        if model_arg:
            model_path = ModelFinder.resolve_model_path(model_arg, "INT8")
            if model_path is None:
                print(Colors.red(f"错误: 未找到模型 {model_arg}"))
                sys.exit(1)
        else:
            model_path = ModelFinder.find_small_model("INT8")
            if model_path is None:
                print(Colors.red("错误: 未找到 INT8 模型"))
                sys.exit(1)
    
    # 获取 eager 参数
    enforce_eager = getattr(args, 'eager', False)
    show_subprocess_output = getattr(args, 'show_subprocess_output', False)
    
    # 运行吐量量对比
    run_throughput_comparison(
        model_path=model_path,
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_32=inner_32,
        sparsity=sparsity,
        verbose=True,
        baseline_model_path=baseline_model_path,
        enforce_eager=enforce_eager,
        show_subprocess_output=show_subprocess_output,
    )
    
    sys.exit(0)
