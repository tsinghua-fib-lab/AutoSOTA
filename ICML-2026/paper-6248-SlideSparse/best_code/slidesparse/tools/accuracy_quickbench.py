#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse vLLM Accuracy Quick Benchmark 脚本

用于快速测试 W8A8 量化模型的精度，使用 vllm run-batch 命令进行推理。

核心设计思想：
  - 使用预定义的 prompt 文件进行批量推理
  - 固定输出长度 (max_tokens) 以确保测试效率
  - 保留 FP8/INT8 硬件支持检测和回退机制
  - 输出结果以 JSON 格式保存，便于人工检查

使用方法:
    python3 accuracy_quickbench.py [选项]

示例:
    python3 accuracy_quickbench.py --model qwen2.5-0.5b-int8
    python3 accuracy_quickbench.py --model llama3.2-1b-fp8
    python3 accuracy_quickbench.py --all
    python3 accuracy_quickbench.py --int8 --qwen
    python3 accuracy_quickbench.py --fp8 --llama
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

# 输出长度配置
MAX_OUTPUT_TOKENS = 64

# max-model-len 配置 (需要足够容纳 prompt + output)
MAX_MODEL_LEN = 512

# 日志级别 (减少 vLLM 输出)
VLLM_LOG_LEVEL = "WARNING"

# GPU 设备编号 (逗号分隔，支持多 GPU)
GPU_ID = "0,1"

# GPU 内存利用率 (0.0-1.0)
GPU_MEMORY_UTILIZATION = 0.8

# Tensor Parallelism 配置
TENSOR_PARALLEL_SIZE = 2

# Temperature 配置
TEMPERATURE = 0.0

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
# 辅助函数
# ============================================================================

def get_prompt_input_file() -> Path:
    """获取 prompt 输入文件路径"""
    return _SCRIPT_DIR / "accuracy_quickbench_prompts.jsonl"


def extract_responses_to_txt(json_file: Path, txt_file: Path, prompts_file: Path):
    """
    从 JSON 输出文件提取回答内容到 txt 文件
    
    Args:
        json_file: vllm run-batch 输出的 JSON 文件
        txt_file: 提取结果的输出文件
        prompts_file: 原始 prompts 文件 (用于获取问题内容)
    """
    if not json_file.exists():
        print_warning(f"JSON 文件不存在: {json_file}")
        return
    
    print_info(f"提取回答到: {txt_file}")
    
    # 构建 custom_id -> prompt 的映射
    prompts_map = {}
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    cid = entry.get("custom_id", "")
                    body = entry.get("body", {})
                    msgs = body.get("messages", [])
                    if msgs:
                        prompts_map[cid] = msgs[0].get("content", "N/A")
                except json.JSONDecodeError:
                    pass
    
    # 读取 JSON 输出并提取回答
    line_num = 0
    with open(json_file, "r", encoding="utf-8") as f_in, \
         open(txt_file, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                entry = json.loads(line.strip())
                custom_id = entry.get("custom_id", f"req-{line_num:03d}")
                
                # 从 prompts_map 获取原始 prompt
                prompt = prompts_map.get(custom_id, "N/A")
                
                # 获取回答内容
                response = entry.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                
                if choices:
                    content = choices[0].get("message", {}).get("content", "[No content]")
                else:
                    error = entry.get("error", {})
                    if error:
                        content = f"[Error: {error}]"
                    else:
                        content = "[No response]"
                
                f_out.write(f"=== {custom_id} ===\n")
                f_out.write(f"Q: {prompt}\n")
                f_out.write(f"A: {content}\n")
                f_out.write("\n")
                
            except json.JSONDecodeError as e:
                f_out.write(f"=== Line {line_num} ===\n")
                f_out.write(f"[JSON Parse Error: {e}]\n\n")
    
    print_success(f"成功提取 {line_num} 条回答")


def run_accuracy_test(
    model_key: str,
    *,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    max_model_len: int = MAX_MODEL_LEN,
    temperature: float = TEMPERATURE,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    tp_size: int = TENSOR_PARALLEL_SIZE,
    gpu_id: str = GPU_ID,
    enforce_eager: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    运行单个模型的精度测试
    
    Args:
        model_key: 模型 key
        max_tokens: 最大输出 token 数
        max_model_len: 最大模型长度
        temperature: 采样温度
        gpu_memory_util: GPU 内存利用率
        tp_size: Tensor Parallelism 大小
        gpu_id: GPU ID 列表
        enforce_eager: 强制使用 eager mode
        dry_run: 只显示命令不执行
        
    Returns:
        成功返回 True，失败返回 False
    """
    global _OUTPUT_DIR
    
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
    
    # 检查 prompt 输入文件
    prompt_file = get_prompt_input_file()
    if not prompt_file.exists():
        print_error(f"Prompt 输入文件不存在: {prompt_file}")
        return False
    
    # 根据模型的 dtype 构建输出目录
    output_dir = build_result_dir("accuracy_quickbench", dtype=entry.quant.upper())
    _OUTPUT_DIR = output_dir  # 更新全局状态，用于信号处理
    log_file = output_dir / "benchmark.log"
    
    print_header(f"测试模型: {entry.local_name}")
    
    # 输出文件名: {model_local_name}.json (硬件信息已在文件夹名中)
    output_file = output_dir / f"{entry.local_name}.json"
    response_txt = output_dir / f"{entry.local_name}_responses.txt"
    
    # 计算实际使用的 GPU
    gpu_devices, actual_tp = get_gpu_devices_for_tp(tp_size, gpu_id)
    
    # 显示测试参数
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    测试参数                                  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ 模型:           {entry.local_name}")
    print(f"│ 量化类型:       {entry.quant.upper()}")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ 输入文件:       {prompt_file}")
    print(f"│ 输出文件:       {output_file}")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ vLLM 参数:")
    print(f"│   --max-model-len   = {max_model_len}")
    print(f"│   --temperature     = {temperature}")
    print(f"│   --max-tokens      = {max_tokens} (via generation-config)")
    if actual_tp > 1:
        print(f"│   --tensor-parallel = {actual_tp}")
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
    generation_config = json.dumps({
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    })
    
    cmd = [
        "vllm", "run-batch",
        "--model", str(model_path),
        "--served-model-name", "model",
        "--input-file", str(prompt_file),
        "--output-file", str(output_file),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--override-generation-config", generation_config,
        "--disable-log-stats",
    ]
    
    if actual_tp > 1:
        cmd.extend(["--tensor-parallel-size", str(actual_tp)])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    # 记录到日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"========== {entry.local_name} ==========\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Output: {output_file}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("\n")
    
    # Dry-run 模式
    if dry_run:
        print_info("[DRY-RUN] 将执行的命令:")
        print(f"CUDA_VISIBLE_DEVICES={gpu_devices} " + " ".join(cmd))
        print()
        # 生成模拟结果
        with open(output_file, "w") as f:
            f.write(json.dumps({"status": "dry-run", "model": entry.local_name}) + "\n")
        return True
    
    # 执行测试
    print_info("开始精度测试...")
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
        
        if result.returncode == 0 and output_file.exists():
            print_success(f"测试完成! 耗时: {duration:.1f}s")
            print_success(f"输出保存到: {output_file}")
            
            # 显示输出预览
            print()
            print(f"{Colors.GREEN}输出预览 (前 5 条):{Colors.NC}")
            print("-" * 46)
            with open(output_file, "r") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(line.rstrip()[:100] + ("..." if len(line) > 100 else ""))
            print("-" * 46)
            
            # 统计输出条目数
            with open(output_file, "r") as f:
                output_count = sum(1 for _ in f)
            print_info(f"输出条目数: {output_count}")
            
            # 提取回答内容
            extract_responses_to_txt(output_file, response_txt, prompt_file)
            
            return True
        else:
            print_error(f"测试失败: {entry.local_name} (exit code: {result.returncode})")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: Test failed for {entry.local_name}\n")
            
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


def test_models(
    quant_filter: str | None,
    family_filter: str | None,
    specific_model: str | None,
    **kwargs,
) -> tuple[int, int]:
    """
    批量测试模型
    
    Args:
        quant_filter: 量化类型过滤 (int8, fp8)
        family_filter: 模型系列过滤 (qwen, llama)
        specific_model: 指定模型 key
        **kwargs: 传递给 run_accuracy_test 的其他参数
        
    Returns:
        (成功数, 失败数) 元组
    """
    success_count = 0
    failed_count = 0
    
    # 确定要测试的模型
    if specific_model:
        entry = model_registry.get(specific_model)
        if entry is None:
            print_error(f"模型不存在: {specific_model}")
            return 0, 1
        
        # 检查量化格式支持
        if not check_quant_support(entry.quant):
            print_error(f"GPU 不支持原生 {entry.quant.upper()} GEMM，跳过测试")
            return 0, 1
        
        if run_accuracy_test(specific_model, **kwargs):
            success_count = 1
        else:
            failed_count = 1
    else:
        # 批量测试
        quant_types = []
        if quant_filter:
            quant_types = [quant_filter]
        else:
            quant_types = ["int8", "fp8"]
        
        for quant in quant_types:
            # 检查量化格式支持
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
                
                if run_accuracy_test(entry.key, **kwargs):
                    success_count += 1
                else:
                    failed_count += 1
                    # 试错机制: 一个模型失败，跳过该精度的剩余模型
                    skip_remaining = True
                    print()
                    print_warning("=" * 50)
                    print_warning(f"⚠️  {quant.upper()} 模型测试失败: {entry.local_name}")
                    print_warning(f"    将跳过所有剩余的 {quant.upper()} 模型")
                    print_warning("=" * 50)
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse vLLM Accuracy Quick Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用模型:

  INT8 模型 (quantized.w8a8):
""" + "\n".join(f"    - {key}" for key in list_models(quant="int8")) + """

  FP8 模型 (FP8-dynamic):
""" + "\n".join(f"    - {key}" for key in list_models(quant="fp8")) + """

示例:
  %(prog)s --model qwen2.5-0.5b-int8   # 测试指定模型
  %(prog)s --all                        # 测试所有模型
  %(prog)s --int8 --qwen                # 测试 Qwen INT8 模型
  %(prog)s --fp8 --llama --dry-run      # Dry-run 模式
"""
    )
    
    # 模型选择
    model_group = parser.add_argument_group("模型选择")
    model_group.add_argument(
        "-a", "--all", action="store_true",
        help="测试所有模型 (INT8 + FP8)"
    )
    model_group.add_argument(
        "-i", "--int8", action="store_true",
        help="仅测试 INT8 模型"
    )
    model_group.add_argument(
        "-f", "--fp8", action="store_true",
        help="仅测试 FP8 模型"
    )
    model_group.add_argument(
        "-q", "--qwen", action="store_true",
        help="仅测试 Qwen2.5 系列"
    )
    model_group.add_argument(
        "-l", "--llama", action="store_true",
        help="仅测试 Llama3.2 系列"
    )
    model_group.add_argument(
        "-m", "--model", type=str, metavar="NAME",
        help="测试指定模型"
    )
    model_group.add_argument(
        "-c", "--check", action="store_true",
        help="检查模型下载状态"
    )
    
    # 输出选项
    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help=f"最大输出 token 数 (默认: {MAX_OUTPUT_TOKENS})"
    )
    output_group.add_argument(
        "--max-model-len", type=int, default=MAX_MODEL_LEN,
        help=f"最大模型长度 (默认: {MAX_MODEL_LEN})"
    )
    output_group.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"采样温度 (默认: {TEMPERATURE})"
    )
    
    # 编译选项
    compile_group = parser.add_argument_group("编译选项")
    compile_group.add_argument(
        "--eager", action="store_true",
        help="强制使用 eager mode (禁用 torch.compile)"
    )
    compile_group.add_argument(
        "--compile", action="store_true",
        help="强制启用 torch.compile (覆盖自动检测)"
    )
    
    # 硬件选项
    hw_group = parser.add_argument_group("硬件选项")
    hw_group.add_argument(
        "--tp", type=int, default=TENSOR_PARALLEL_SIZE,
        help=f"Tensor Parallelism 大小 (默认: {TENSOR_PARALLEL_SIZE})"
    )
    hw_group.add_argument(
        "--gpu-id", type=str, default=GPU_ID,
        help=f"GPU ID 列表 (默认: {GPU_ID})"
    )
    hw_group.add_argument(
        "--gpu-mem", type=float, default=GPU_MEMORY_UTILIZATION,
        help=f"GPU 内存利用率 (默认: {GPU_MEMORY_UTILIZATION})"
    )
    
    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "--dry-run", action="store_true",
        help="只显示命令不执行"
    )
    
    args = parser.parse_args()
    
    # 检查模式
    if args.check:
        from slidesparse.tools.utils import print_model_status
        print_model_status(CHECKPOINT_DIR)
        return 0
    
    # 检查 vllm 是否安装
    if not shutil.which("vllm"):
        print_error("vllm 未安装或不在 PATH 中")
        print_info("请确保 vLLM 已正确安装")
        return 1
    
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
        # 自动检测
        if not check_triton_support_and_warn():
            print_warning("检测到不支持 torch.compile 的 GPU 架构")
            print_warning("自动启用 eager mode (--enforce-eager)")
            enforce_eager = True
    
    # 获取硬件信息
    hw_info = HardwareInfo()
    
    # 设置信号处理
    _setup_signal_handlers()
    
    # 显示配置信息
    print_header("SlideSparse vLLM Accuracy Quick Benchmark")
    print()
    print_hardware_info()
    print()
    
    prompt_file = get_prompt_input_file()
    print("测试配置:")
    print(f"  Prompt 输入文件:  {prompt_file}")
    print(f"  最大输出 Token:   {args.max_tokens}")
    print(f"  最大模型长度:     {args.max_model_len}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  GPU 内存利用率:   {args.gpu_mem}")
    print(f"  vLLM 日志级别:    {VLLM_LOG_LEVEL}")
    if enforce_eager:
        print(f"  编译模式:         Eager (torch.compile 已禁用)")
    print()
    print("输出目录结构: accuracy_quickbench_results/{GPU}_{CC}_{dtype}_{PyVer}_{CUDAVer}_{Arch}/")
    print("=" * 46)
    
    # 执行测试
    success_count, failed_count = test_models(
        quant_filter=quant_filter,
        family_filter=family_filter,
        specific_model=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
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
    print("  accuracy_quickbench_results/")
    print("    ├── {GPU}_{CC}_INT8_{PyVer}_{CUDAVer}_{Arch}/")
    print("    │   ├── benchmark.log")
    print("    │   ├── {Model}.json")
    print("    │   └── {Model}_responses.txt")
    print("    └── {GPU}_{CC}_FP8_{PyVer}_{CUDAVer}_{Arch}/")
    print("        ├── ...")
    print()
    print(f"成功: {success_count}, 失败: {failed_count}")
    print()
    print("查看结果:")
    print("  - *_responses.txt 包含易读的 Q&A 对")
    print("  - *.json 包含完整的 API 响应详情")
    print("=" * 46)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
