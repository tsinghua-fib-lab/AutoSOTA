#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse vLLM Throughput Benchmark 脚本

用于精确测试 W8A8 量化模型在不同 M (GEMM batch size) 下的 Prefill/Decode 性能。

核心设计思想：
  - Prefill 测试: 控制 M_prefill = max_num_seqs × prompt_length，最小化 Decode 开销
  - Decode 测试:  控制 M_decode = max_num_seqs，最小化 Prefill 开销
  - 动态计算 max-model-len 以最大化 KV Cache 利用率 (Tight Fit 策略)
  - 禁用 Chunked Prefill 以获得纯净的性能数据

使用方法:
    python3 throughput_bench.py [选项]

示例:
    python3 throughput_bench.py --model qwen2.5-0.5b-fp8 --prefill --M 16,32,64,128,256
    python3 throughput_bench.py --model llama3.2-1b-fp8 --decode --M 1,2,4,8,16
    python3 throughput_bench.py --all --prefill --M 16,256
    python3 throughput_bench.py --all --decode --M 1,16
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import signal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# 确保可以导入 slidesparse
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    HardwareInfo,
    model_registry,
    list_models,
    check_quant_support,
    get_model_local_path,
)
from slidesparse.tools.utils import (
    Colors,
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
    strip_ansi,
    CHECKPOINT_DIR,
    build_result_dir,
    get_vllm_env_vars,
    check_triton_support_and_warn,
    print_hardware_info,
    get_gpu_devices_for_tp,
)


# ============================================================================
# 全局配置参数
# ============================================================================

# Prefill 测试配置
DEFAULT_M_LIST_PREFILL = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_PREFILL = 128  # Prefill 重复次数

# Decode 测试配置
DEFAULT_M_LIST_DECODE = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
N_DECODE = 512  # Decode 生成的 token 数

# Prompt length 配置
PROMPT_LENGTH_CAP_PREFILL = 1024  # Prefill 模式下 prompt_length 的上限
PROMPT_LENGTH_FIXED_DECODE = 16   # Decode 模式下固定的 prompt_length

# max-model-len 计算的 Buffer
MODEL_LEN_BUFFER = 128

# 日志级别
VLLM_LOG_LEVEL = "WARNING"

# GPU 配置
GPU_ID = "0,1"
GPU_MEMORY_UTILIZATION = 0.8
TENSOR_PARALLEL_SIZE = 1

# 全局状态 (用于信号处理)
_OUTPUT_DIR: Path | None = None


# ============================================================================
# 信号处理
# ============================================================================

def _signal_handler(signum, frame):
    """处理中断信号 (SIGINT/SIGTERM)"""
    print()
    print("=" * 46)
    print("测试被中断!")
    if _OUTPUT_DIR is not None:
        print(f"结果目录: {_OUTPUT_DIR}")
    print("=" * 46)
    sys.exit(130)


def _setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class TestParams:
    """测试参数"""
    prompt_length: int
    max_num_seqs: int
    num_prompts: int
    output_len: int
    max_model_len: int
    n_prefill: int
    n_decode: int
    m_prefill: int
    m_decode: int


# ============================================================================
# 辅助函数
# ============================================================================

def parse_m_list(m_str: str) -> list[int]:
    """解析 M 值列表字符串"""
    return [int(x.strip()) for x in m_str.split(",")]


def calculate_test_params(m_value: int, test_mode: str, n_repeat: Optional[int] = None) -> TestParams:
    """
    根据测试模式和 M 值计算所有测试参数
    
    Args:
        m_value: M 值
        test_mode: 测试模式 (prefill/decode)
        n_repeat: 重复次数 (覆盖默认值)
        
    Returns:
        TestParams 数据类实例
    """
    if test_mode == "prefill":
        # Prefill 测试: M_prefill = max_num_seqs × prompt_length
        n_prefill_val = n_repeat if n_repeat else N_PREFILL
        
        if m_value <= PROMPT_LENGTH_CAP_PREFILL:
            prompt_length = m_value
            max_num_seqs = 1
        else:
            prompt_length = PROMPT_LENGTH_CAP_PREFILL
            max_num_seqs = m_value // prompt_length
        
        num_prompts = n_prefill_val * max_num_seqs
        output_len = 1  # 最小化 Decode
        max_model_len = prompt_length + output_len + MODEL_LEN_BUFFER
        
        m_prefill = max_num_seqs * prompt_length
        m_decode = max_num_seqs
        
        return TestParams(
            prompt_length=prompt_length,
            max_num_seqs=max_num_seqs,
            num_prompts=num_prompts,
            output_len=output_len,
            max_model_len=max_model_len,
            n_prefill=n_prefill_val,
            n_decode=0,
            m_prefill=m_prefill,
            m_decode=m_decode,
        )
    else:
        # Decode 测试: M_decode = max_num_seqs (batch size)
        n_decode_val = n_repeat if n_repeat else N_DECODE
        
        prompt_length = PROMPT_LENGTH_FIXED_DECODE
        max_num_seqs = m_value
        num_prompts = max_num_seqs
        output_len = n_decode_val
        max_model_len = prompt_length + output_len + MODEL_LEN_BUFFER
        
        m_prefill = max_num_seqs * prompt_length
        m_decode = max_num_seqs
        
        return TestParams(
            prompt_length=prompt_length,
            max_num_seqs=max_num_seqs,
            num_prompts=num_prompts,
            output_len=output_len,
            max_model_len=max_model_len,
            n_prefill=1,
            n_decode=n_decode_val,
            m_prefill=m_prefill,
            m_decode=m_decode,
        )


def run_single_m_test(
    model_key: str,
    m_value: int,
    test_mode: str,
    result_json_dir: Path,
    log_file: Path,
    *,
    n_repeat: Optional[int] = None,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    tp_size: int = TENSOR_PARALLEL_SIZE,
    gpu_id: str = GPU_ID,
    enforce_eager: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    运行单个 M 值的吞吐测试
    
    Returns:
        成功返回 True，失败返回 False
    """
    # 获取模型信息
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"模型不存在: {model_key}")
        return False
    
    model_path = get_model_local_path(model_key, CHECKPOINT_DIR)
    
    # 检查模型是否已下载
    if not model_path.exists() or not (model_path / "config.json").exists():
        print_warning(f"模型未下载，跳过: {entry.local_name}")
        return False
    
    # 计算测试参数
    params = calculate_test_params(m_value, test_mode, n_repeat)
    
    # 计算实际使用的 GPU
    gpu_devices, actual_tp = get_gpu_devices_for_tp(tp_size, gpu_id)
    
    # 结果文件名: {model_local_name}_M{m_value}.json (硬件信息已在文件夹名中)
    result_file = result_json_dir / f"{entry.local_name}_M{m_value}.json"
    
    # max-num-batched-tokens (禁用 chunking)
    max_num_batched_tokens = params.max_num_seqs * params.max_model_len
    
    # 显示测试参数
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    测试参数                                  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ 模型:  {entry.local_name}")
    print(f"│ 模式:  {test_mode} test")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ GEMM M 维度:")
    print(f"│   M_prefill     = {params.m_prefill} (= {params.max_num_seqs} x {params.prompt_length})")
    print(f"│   M_decode      = {params.m_decode}")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ vLLM 参数:")
    print(f"│   --input-len       = {params.prompt_length}")
    print(f"│   --output-len      = {params.output_len}")
    print(f"│   --num-prompts     = {params.num_prompts}")
    print(f"│   --max-num-seqs    = {params.max_num_seqs}")
    print(f"│   --max-model-len   = {params.max_model_len}")
    if actual_tp > 1:
        print(f"│   --tensor-parallel = {actual_tp}")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 迭代次数:")
    print(f"│   N_prefill     = {params.n_prefill}")
    print(f"│   N_decode      = {params.n_decode}")
    if enforce_eager:
        print("├─────────────────────────────────────────────────────────────┤")
        print("│ 编译模式:")
        print("│   --enforce-eager   = true (torch.compile 已禁用)")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    
    # 构建环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    env.update(get_vllm_env_vars(log_level=VLLM_LOG_LEVEL))
    
    # 构建命令
    cmd = [
        "vllm", "bench", "throughput",
        "--model", str(model_path),
        "--dataset-name", "random",
        "--input-len", str(params.prompt_length),
        "--output-len", str(params.output_len),
        "--num-prompts", str(params.num_prompts),
        "--max-num-seqs", str(params.max_num_seqs),
        "--max-model-len", str(params.max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--disable-log-stats",
        "--output-json", str(result_file),
    ]
    
    if actual_tp > 1:
        cmd.extend(["--tensor-parallel-size", str(actual_tp)])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    # 记录到日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"========== M={m_value} ==========\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Params: prompt_len={params.prompt_length}, output_len={params.output_len}, ")
        f.write(f"num_prompts={params.num_prompts}, max_num_seqs={params.max_num_seqs}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("\n")
    
    # Dry-run 模式
    if dry_run:
        print_info("[DRY-RUN] 将执行的命令:")
        print(f"CUDA_VISIBLE_DEVICES={gpu_devices} " + " ".join(cmd))
        print()
        # 生成模拟结果
        with open(result_file, "w") as f:
            json.dump({
                "requests_per_second": 0,
                "tokens_per_second": 0,
                "elapsed_time": 0,
                "num_requests": 0,
            }, f)
        return True
    
    # 执行测试
    print_info("开始测试...")
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 记录输出到日志 (去除 ANSI 转义码)
        with open(log_file, "a", encoding="utf-8") as f:
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(strip_ansi(result.stdout))
                f.write("\n")
            if result.stderr:
                f.write("STDERR:\n")
                f.write(strip_ansi(result.stderr))
                f.write("\n")
        
        if result.returncode == 0 and result_file.exists():
            print_success(f"测试完成! 耗时: {duration:.1f}s")
            
            # 解析并显示结果
            with open(result_file, "r") as f:
                data = json.load(f)
            
            req_per_s = data.get("requests_per_second", 0)
            tok_per_s = data.get("tokens_per_second", 0)
            elapsed = data.get("elapsed_time", 0)
            num_req = data.get("num_requests", 0)
            
            print()
            print(f"{Colors.GREEN}测试结果:{Colors.NC}")
            print(f"  Requests/s:   {req_per_s:.2f}")
            print(f"  Tokens/s:     {tok_per_s:.2f}")
            print(f"  Total Reqs:   {num_req}")
            print(f"  Elapsed:      {elapsed:.2f}s")
            
            # 计算单次操作的性能
            if test_mode == "prefill" and params.n_prefill > 0:
                total_prefill_tokens = params.m_prefill * params.n_prefill
                if elapsed > 0:
                    prefill_tps = total_prefill_tokens / elapsed
                    print()
                    print("  [Prefill 分析]")
                    print(f"  Total Prefill Tokens: {total_prefill_tokens}")
                    print(f"  Prefill Tokens/s:     {prefill_tps:.2f}")
            elif test_mode == "decode" and params.n_decode > 0:
                decode_tokens = params.m_decode * params.n_decode
                if elapsed > 0:
                    decode_tps = decode_tokens / elapsed
                    print()
                    print("  [Decode 分析]")
                    print(f"  Total Decode Tokens:  {decode_tokens}")
                    print(f"  Decode Tokens/s:      {decode_tps:.2f}")
            
            return True
        else:
            print_error(f"测试失败: M={m_value} (exit code: {result.returncode})")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: Test failed for M={m_value}\n")
            
            # 打印错误摘要
            if result.stderr:
                print()
                print(f"{Colors.RED}─── 错误输出 (最后 10 行) ───{Colors.NC}")
                for line in result.stderr.splitlines()[-10:]:
                    print(line)
                print(f"{Colors.RED}─────────────────────────────────{Colors.NC}")
            
            return False
            
    except Exception as e:
        print_error(f"执行异常: {e}")
        return False


def generate_model_csv(
    model_name: str,
    m_list: list[int],
    test_mode: str,
    result_json_dir: Path,
    output_dir: Path,
    n_repeat: Optional[int] = None,
):
    """生成单个模型的 CSV 结果"""
    # CSV 文件名: {model_name}_{test_mode}.csv (硬件信息已在文件夹名中)
    csv_file = output_dir / f"{model_name}_{test_mode}.csv"
    
    print()
    print_subheader(f"生成 CSV: {model_name}")
    
    # CSV 表头
    if test_mode == "prefill":
        header = "M_prefill,prompt_len,max_num_seqs,num_prompts,N_prefill,requests_per_s,tokens_per_s,elapsed_time_s"
    else:
        header = "M_decode,prompt_len,max_num_seqs,num_prompts,N_decode,output_len,requests_per_s,tokens_per_s,elapsed_time_s"
    
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        
        for m_value in m_list:
            # 结果文件名: {model_name}_M{m_value}.json
            result_file = result_json_dir / f"{model_name}_M{m_value}.json"
            
            if result_file.exists():
                params = calculate_test_params(m_value, test_mode, n_repeat)
                
                try:
                    with open(result_file, "r") as rf:
                        data = json.load(rf)
                    req_s = data.get("requests_per_second", 0)
                    tok_s = data.get("tokens_per_second", 0)
                    elapsed = data.get("elapsed_time", 0)
                except Exception:
                    req_s = tok_s = elapsed = 0
                
                if test_mode == "prefill":
                    f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                           f"{params.num_prompts},{params.n_prefill},"
                           f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
                else:
                    f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                           f"{params.num_prompts},{params.n_decode},{params.output_len},"
                           f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
    
    print_success(f"CSV 保存到: {csv_file}")
    
    # 显示 CSV 预览
    print()
    print("预览:")
    print("-" * 46)
    with open(csv_file, "r") as f:
        print(f.read())
    print("-" * 46)


def run_model_benchmark(
    model_key: str,
    m_list: list[int],
    test_mode: str,
    **kwargs,
) -> int:
    """
    测试单个模型的所有 M 值
    
    Returns:
        0=成功, 1=普通错误, 2=精度不支持
    """
    global _OUTPUT_DIR
    
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"模型不存在: {model_key}")
        return 1
    
    model_path = get_model_local_path(model_key, CHECKPOINT_DIR)
    
    if not model_path.exists() or not (model_path / "config.json").exists():
        print_warning(f"模型未下载，跳过: {entry.local_name}")
        return 1
    
    # 根据模型的 dtype 构建输出目录
    output_dir = build_result_dir("throughput_bench", test_mode=test_mode, dtype=entry.quant.upper())
    _OUTPUT_DIR = output_dir  # 更新全局状态
    result_json_dir = output_dir / "json"
    result_json_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "benchmark.log"
    
    print_header(f"测试模型: {entry.local_name}")
    
    total_tests = len(m_list)
    failed_tests = 0
    first_error = False
    
    for i, m_value in enumerate(m_list, 1):
        print()
        print("=" * 46)
        print(f"[{i}/{total_tests}] 测试 M={m_value}")
        print("=" * 46)
        
        success = run_single_m_test(
            model_key, m_value, test_mode,
            result_json_dir, log_file, **kwargs
        )
        
        if not success:
            failed_tests += 1
            
            if not first_error:
                first_error = True
                print()
                print_warning("=" * 50)
                print_warning(f"⚠️  {entry.quant.upper()} 模型测试失败: {entry.local_name}")
                print_warning("    跳过该模型的剩余测试")
                print_warning(f"    将跳过所有其他 {entry.quant.upper()} 模型")
                print_warning("=" * 50)
                
                with open(log_file, "a") as f:
                    f.write(f"SKIP: {entry.quant.upper()} test failed\n")
                
                return 2
    
    # 生成 CSV 结果
    generate_model_csv(
        entry.local_name, m_list, test_mode,
        result_json_dir, output_dir, kwargs.get("n_repeat")
    )
    
    print()
    print_info(f"模型 {entry.local_name} 完成: {total_tests} 测试, {failed_tests} 失败")
    return 0


def test_models(
    quant_filter: str | None,
    family_filter: str | None,
    specific_model: str | None,
    m_list: list[int],
    test_mode: str,
    **kwargs,
) -> tuple[int, int]:
    """批量测试模型"""
    success_count = 0
    failed_count = 0
    
    if specific_model:
        entry = model_registry.get(specific_model)
        if entry is None:
            print_error(f"模型不存在: {specific_model}")
            return 0, 1
        
        if not check_quant_support(entry.quant):
            print_error(f"GPU 不支持原生 {entry.quant.upper()} GEMM，跳过测试")
            return 0, 1
        
        result = run_model_benchmark(
            specific_model, m_list, test_mode, **kwargs
        )
        
        if result == 0:
            success_count = 1
        else:
            failed_count = 1
    else:
        quant_types = [quant_filter] if quant_filter else ["int8", "fp8"]
        
        for quant in quant_types:
            if not check_quant_support(quant):
                print_warning(f"GPU 不支持原生 {quant.upper()} GEMM，跳过所有 {quant.upper()} 模型")
                continue
            
            print_header(f"测试 {quant.upper()} 模型")
            
            models = model_registry.list(family=family_filter, quant=quant)
            skip_remaining = False
            
            for entry in models:
                if skip_remaining:
                    print_warning(f"跳过 {entry.key} (之前的测试失败)")
                    failed_count += 1
                    continue
                
                result = run_model_benchmark(
                    entry.key, m_list, test_mode, **kwargs
                )
                
                if result == 0:
                    success_count += 1
                elif result == 2:
                    failed_count += 1
                    skip_remaining = True
                else:
                    failed_count += 1
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse vLLM Throughput Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式详情:

[Prefill 测试模式]
  目标: 测试不同 M_prefill 下的 Prefill 性能
  策略: 仅测试 Prefill，无 Decode (output_len=1)
  默认 M_list: """ + ",".join(map(str, DEFAULT_M_LIST_PREFILL[:8])) + """...

[Decode 测试模式]
  目标: 测试不同 M_decode (batch size) 下的 Decode 性能
  策略: 运行一次 Prefill，然后进行 N_decode 次 Decode 迭代
  默认 M_list: """ + ",".join(map(str, DEFAULT_M_LIST_DECODE)) + """

示例:
  %(prog)s --model qwen2.5-7b-fp8 --prefill
  %(prog)s --model qwen2.5-7b-fp8 --decode
  %(prog)s --all --prefill --M 16,32,64,128 --N 64
  %(prog)s --int8 --qwen --decode --dry-run
"""
    )
    
    # 模型选择
    model_group = parser.add_argument_group("模型选择")
    model_group.add_argument("-a", "--all", action="store_true", help="测试所有模型")
    model_group.add_argument("-i", "--int8", action="store_true", help="仅测试 INT8 模型")
    model_group.add_argument("-f", "--fp8", action="store_true", help="仅测试 FP8 模型")
    model_group.add_argument("-q", "--qwen", action="store_true", help="仅测试 Qwen2.5 系列")
    model_group.add_argument("-l", "--llama", action="store_true", help="仅测试 Llama3.2 系列")
    model_group.add_argument("-m", "--model", type=str, metavar="NAME", help="测试指定模型")
    model_group.add_argument("-c", "--check", action="store_true", help="检查模型下载状态")
    
    # 测试模式
    mode_group = parser.add_argument_group("测试模式")
    mode_group.add_argument("--prefill", action="store_true", help="Prefill 测试模式 [默认]")
    mode_group.add_argument("--decode", action="store_true", help="Decode 测试模式")
    
    # 参数覆盖
    param_group = parser.add_argument_group("参数覆盖")
    param_group.add_argument("--M", type=str, metavar="LIST", help="覆盖 M 值列表 (逗号分隔)")
    param_group.add_argument("--N", type=int, metavar="NUM", help="覆盖重复次数")
    
    # 编译选项
    compile_group = parser.add_argument_group("编译选项")
    compile_group.add_argument("--eager", action="store_true", help="强制使用 eager mode")
    compile_group.add_argument("--compile", action="store_true", help="强制启用 torch.compile")
    
    # 硬件选项
    hw_group = parser.add_argument_group("硬件选项")
    hw_group.add_argument("--tp", type=int, default=TENSOR_PARALLEL_SIZE, help=f"TP 大小 (默认: {TENSOR_PARALLEL_SIZE})")
    hw_group.add_argument("--gpu-id", type=str, default=GPU_ID, help=f"GPU ID 列表 (默认: {GPU_ID})")
    hw_group.add_argument("--gpu-mem", type=float, default=GPU_MEMORY_UTILIZATION, help=f"GPU 内存利用率 (默认: {GPU_MEMORY_UTILIZATION})")
    
    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument("--dry-run", action="store_true", help="只显示命令不执行")
    
    args = parser.parse_args()
    
    # 检查模式
    if args.check:
        from slidesparse.tools.utils import print_model_status
        print_model_status(CHECKPOINT_DIR)
        return 0
    
    # 检查 vllm 是否安装
    if not shutil.which("vllm"):
        print_error("vllm 未安装或不在 PATH 中")
        return 1
    
    # 确定测试模式
    test_mode = "decode" if args.decode else "prefill"
    
    # 确定 M 值列表
    if args.M:
        m_list = parse_m_list(args.M)
    else:
        m_list = DEFAULT_M_LIST_DECODE if test_mode == "decode" else DEFAULT_M_LIST_PREFILL
    
    # 确定过滤条件
    quant_filter = None
    family_filter = None
    
    if args.int8 and not args.fp8:
        quant_filter = "int8"
    elif args.fp8 and not args.int8:
        quant_filter = "fp8"
    elif args.all:
        quant_filter = None
    elif not args.model:
        parser.print_help()
        return 0
    
    if args.qwen and not args.llama:
        family_filter = "qwen"
    elif args.llama and not args.qwen:
        family_filter = "llama"
    
    # 确定是否需要 eager mode
    enforce_eager = args.eager
    if not args.eager and not args.compile:
        if not check_triton_support_and_warn():
            print_warning("检测到不支持 torch.compile 的 GPU 架构")
            print_warning("自动启用 eager mode")
            enforce_eager = True
    
    # 获取硬件信息
    hw_info = HardwareInfo()
    
    # 设置信号处理
    _setup_signal_handlers()
    
    # 显示配置信息
    print_header(f"SlideSparse vLLM Throughput Benchmark ({test_mode.upper()})")
    print()
    print_hardware_info()
    print()
    
    print("测试配置:")
    print(f"  测试模式:         {test_mode}")
    print(f"  M 值列表:         {m_list}")
    print(f"  重复次数 (N):     {args.N or (N_DECODE if test_mode == 'decode' else N_PREFILL)}")
    print(f"  GPU 内存利用率:   {args.gpu_mem}")
    if enforce_eager:
        print(f"  编译模式:         Eager")
    print()
    print(f"输出目录结构: throughput_bench_results/{test_mode}/{{GPU}}_{{CC}}_{{dtype}}_{{PyVer}}_{{CUDAVer}}_{{Arch}}/")
    print("=" * 46)
    
    # 执行测试
    success_count, failed_count = test_models(
        quant_filter=quant_filter,
        family_filter=family_filter,
        specific_model=args.model,
        m_list=m_list,
        test_mode=test_mode,
        n_repeat=args.N,
        gpu_memory_util=args.gpu_mem,
        tp_size=args.tp,
        gpu_id=args.gpu_id,
        enforce_eager=enforce_eager,
        dry_run=args.dry_run,
    )
    
    # 显示结果
    print()
    print_header("Benchmark 完成!")
    print("结果目录结构:")
    print(f"  throughput_bench_results/{test_mode}/")
    print("    ├── {GPU}_{CC}_INT8_{PyVer}_{CUDAVer}_{Arch}/")
    print("    │   ├── benchmark.log")
    print("    │   ├── json/")
    print("    │   │   └── {Model}_M{N}.json")
    print(f"    │   └── {{Model}}_{test_mode}.csv")
    print("    └── {GPU}_{CC}_FP8_{PyVer}_{CUDAVer}_{Arch}/")
    print("        ├── ...")
    print()
    print(f"成功: {success_count}, 失败: {failed_count}")
    print("=" * 46)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
