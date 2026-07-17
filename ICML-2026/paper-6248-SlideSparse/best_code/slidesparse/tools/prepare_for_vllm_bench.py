#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 端到端准备脚本

自动化执行从模型下载到性能测试的完整流程，设计用于无人值守的长时间运行。

流水线任务：
============
Task 1: 模型下载（8个模型：4个 INT8 + 4个 FP8）
Task 2: 模型转换（每个模型生成 2:4, 2:6, 2:8, 2:10 的 SlideSparse 版本）
Task 3: 离线粗调优（CUDA cuBLASLt + Triton quant_only）
Task 4: 离线细调优（Triton Dequant + Quant Slide + cuSPARSELt）
Task 5: 简单端到端 Benchmark（验证性测试）
Task 6: 完整 Prefill Benchmark
Task 7: 完整 Decode Benchmark

Usage:
    # 执行所有任务
    python3 prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1
    
    # 只执行模型下载和转换
    python3 prepare_for_vllm_bench.py --task 1,1,0,0,0,0,0
    
    # 跳过下载，只做 benchmark
    python3 prepare_for_vllm_bench.py --task 0,0,0,0,1,1,1
    
    # 查看任务配置（不执行）
    python3 prepare_for_vllm_bench.py --info

    
    # 1. 创建 tmux 会话并进入
    tmux new -s vllm_bench

    # 2. 在 tmux 里运行脚本
    cd /root/vllmbench/slidesparse/tools && python3 prepare_for_vllm_bench.py --task 0,0,1,1,0,0,0 --gpu 0 
    
    # 3. 脚本开始运行后，可以按下以下组合键将其放到后台运行
    # Ctrl+B 后 按 D 退出保持运行
    或者
    tmux detach

    # 4. 重新连接 tmux 会话
    tmux attach -t vllm_bench
    # 查看日志
    cat /root/vllmbench/slidesparse/tools/prepare_bench_*.log

    tmux kill-session -t vllm_bench
"""

import argparse
import os
import sys
import subprocess
import signal
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# 路径设置
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import hw_info
from slidesparse.tools.utils import (
    Colors,
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
    strip_ansi,
)


# =============================================================================
# 任务配置（写死在此，一般不需要修改）
# =============================================================================

@dataclass
class TaskConfig:
    """任务配置"""
    
    # Task 1: 模型下载 - 4个 base model × 2 种量化 = 8 个模型
    DOWNLOAD_MODELS: List[str] = field(default_factory=lambda: [
        "llama3.2-1b-int8", "llama3.2-1b-fp8",
        "llama3.2-3b-int8", "llama3.2-3b-fp8",
        "qwen2.5-7b-int8", "qwen2.5-7b-fp8",
        "qwen2.5-14b-int8", "qwen2.5-14b-fp8",
    ])
    
    # Task 2: 模型转换 - 稀疏配置
    CONVERT_MODELS: List[str] = field(default_factory=lambda: [
        "llama3.2-1b-int8", "llama3.2-1b-fp8",
        "llama3.2-3b-int8", "llama3.2-3b-fp8",
        "qwen2.5-7b-int8", "qwen2.5-7b-fp8",
        "qwen2.5-14b-int8", "qwen2.5-14b-fp8",
    ])
    CONVERT_SPARSITIES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (2, 4), (2, 6), (2, 8), (2, 10)
    ])
    
    # Task 3: 离线粗调优 - cuBLASLt + Triton quant_only
    TUNE_COARSE_MODELS: List[str] = field(default_factory=lambda: [
        "Llama3.2-1B", "Llama3.2-3B", "Qwen2.5-7B", "Qwen2.5-14B"
    ])
    TUNE_COARSE_M_LIST: List[int] = field(default_factory=lambda: [256, 1024, 4096, 16384, 32768])
    TUNE_COARSE_DTYPE: str = "all"
    TUNE_COARSE_LMAX: int = 10
    TUNE_COARSE_WARMUP: int = 25
    TUNE_COARSE_REPEAT: int = 50
    TUNE_COARSE_KERNELS: str = "1,0,0,0,1"  # cuBLAS + Triton quant_only
    
    # Task 4: 离线细调优 - cuSPARSELt + Triton Dequant/QuantSlide
    TUNE_FINE_MODELS: List[str] = field(default_factory=lambda: [
        "Llama3.2-1B", "Llama3.2-3B", "Qwen2.5-7B", "Qwen2.5-14B"
    ])
    TUNE_FINE_M_LIST: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    ])
    TUNE_FINE_DTYPE: str = "all"
    TUNE_FINE_LMAX: int = 10
    TUNE_FINE_WARMUP: int = 25
    TUNE_FINE_REPEAT: int = 50
    TUNE_FINE_KERNELS: str = "0,1,1,1,0"  # cuSPARSELt + Triton Dequant/QuantSlide
    
    # Task 5: 简单端到端 Benchmark
    BENCH_SIMPLE_MODELS: List[str] = field(default_factory=lambda: [
        "llama3.2-1b-int8", "llama3.2-1b-fp8"
    ])
    BENCH_SIMPLE_M_QUICK: bool = True
    BENCH_SIMPLE_SPARSITIES: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_10"])
    BENCH_SIMPLE_BACKENDS: str = "all"
    BENCH_SIMPLE_STAGES: str = "all"
    
    # Task 6: 完整 Prefill Benchmark
    BENCH_PREFILL_MODELS: List[str] = field(default_factory=lambda: [
        "llama3.2-1b-int8", "llama3.2-1b-fp8",
        "llama3.2-3b-int8", "llama3.2-3b-fp8",
        "qwen2.5-7b-int8", "qwen2.5-7b-fp8",
        "qwen2.5-14b-int8", "qwen2.5-14b-fp8",
    ])
    BENCH_PREFILL_M_LIST: List[int] = field(default_factory=lambda: [
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    ])
    BENCH_PREFILL_BACKENDS: str = "cublaslt,cusparselt"
    BENCH_PREFILL_SPARSITIES: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_8", "2_10"])
    BENCH_PREFILL_STAGES: str = "prefill"
    
    # Task 7: 完整 Decode Benchmark
    BENCH_DECODE_MODELS: List[str] = field(default_factory=lambda: [
        "llama3.2-1b-int8", "llama3.2-1b-fp8",
        "llama3.2-3b-int8", "llama3.2-3b-fp8",
        "qwen2.5-7b-int8", "qwen2.5-7b-fp8",
        "qwen2.5-14b-int8", "qwen2.5-14b-fp8",
    ])
    BENCH_DECODE_M_LIST: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    BENCH_DECODE_BACKENDS: str = "cublaslt,cusparselt"
    BENCH_DECODE_SPARSITIES: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_8", "2_10"])
    BENCH_DECODE_STAGES: str = "decode"


# 全局配置实例
CONFIG = TaskConfig()

# 任务名称
TASK_NAMES = [
    "模型下载",
    "模型转换 (SlideSparse)",
    "离线粗调优 (cuBLAS + quant_only)",
    "离线细调优 (cuSPARSE + Triton)",
    "简单端到端 Benchmark",
    "完整 Prefill Benchmark",
    "完整 Decode Benchmark",
]

# 脚本路径
SCRIPTS = {
    "download": _SCRIPT_DIR / "model_download.py",
    "convert": _SLIDESPARSE_ROOT / "weight_convert" / "weight_convert_entry.py",
    "tune": _SCRIPT_DIR / "offline_autotune_algsearch.py",
    "benchmark": _SCRIPT_DIR / "throughput_benchmark.py",
}


# =============================================================================
# 进程保护
# =============================================================================

def setup_process_protection():
    """设置进程保护（防止被 OOM Killer 杀掉）"""
    try:
        # 设置 OOM score 为 -1000（最低优先级被杀）
        with open(f'/proc/{os.getpid()}/oom_score_adj', 'w') as f:
            f.write('-1000')
        print_info("已设置 OOM 保护 (oom_score_adj=-1000)")
    except (PermissionError, FileNotFoundError):
        print_warning("无法设置 OOM 保护（需要 root 权限）")


def setup_gpu_environment(gpu_id: str = "0"):
    """设置 GPU 环境变量"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print_info(f"已设置 CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # 使用系统 ptxas 替代 Triton 内置版本
    # 解决新架构（如 GB10/sm_121a）在旧版 Triton 中不支持的问题
    system_ptxas = "/usr/local/cuda/bin/ptxas"
    if os.path.exists(system_ptxas):
        os.environ["TRITON_PTXAS_PATH"] = system_ptxas
        print_info(f"已设置 TRITON_PTXAS_PATH={system_ptxas}")


# =============================================================================
# 日志管理
# =============================================================================

class TeeLogger:
    """
    同时输出到控制台和日志文件的 Logger
    """
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8', buffering=1)  # 行缓冲
        
    def write(self, message):
        self.terminal.write(message)
        # 写入日志时去除 ANSI 颜色码
        self.log.write(strip_ansi(message))
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 主日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = log_dir / f"prepare_bench_{timestamp}.log"
        
        # 任务状态文件
        self.status_file = log_dir / f"prepare_bench_{timestamp}_status.json"
        
        # 原始 stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_logger = None
        
    def start(self):
        """开始日志记录"""
        # 写入日志头
        with open(self.main_log, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SlideSparse Prepare Benchmark Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Hardware:\n")
            f.write(f"  GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})\n")
            f.write(f"  Python: {hw_info.python_tag}\n")
            f.write(f"  CUDA: {hw_info.cuda_tag}\n")
            f.write(f"  Arch: {hw_info.arch_tag}\n")
            f.write("\n")
        
        # 设置 TeeLogger
        self._tee_logger = TeeLogger(self.main_log)
        sys.stdout = self._tee_logger
        sys.stderr = self._tee_logger
        
        print_info(f"日志文件: {self.main_log}")
        
    def stop(self):
        """停止日志记录"""
        if self._tee_logger:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._tee_logger.close()
            
    def save_status(self, status: Dict[str, Any]):
        """保存任务状态"""
        import json
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
            
    def log_task_start(self, task_id: int, task_name: str):
        """记录任务开始"""
        print()
        print("=" * 70)
        print(f"TASK {task_id + 1}: {task_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
    def log_task_end(self, task_id: int, task_name: str, success: bool, duration: float):
        """记录任务结束"""
        print()
        print("-" * 70)
        status = f"{Colors.GREEN}SUCCESS{Colors.NC}" if success else f"{Colors.RED}FAILED{Colors.NC}"
        print(f"TASK {task_id + 1}: {task_name} - {status}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print("-" * 70)
        print()


# =============================================================================
# 命令执行
# =============================================================================

def run_command(
    cmd: List[str],
    name: str,
    timeout: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> Tuple[bool, str, float]:
    """
    执行命令并捕获输出
    
    Args:
        cmd: 命令列表
        name: 命令名称
        timeout: 超时时间（秒）
        cwd: 工作目录
        
    Returns:
        (success, output, duration)
    """
    start_time = time.time()
    
    print_info(f"执行: {' '.join(cmd)}")
    if cwd:
        print_info(f"工作目录: {cwd}")
    print()
    
    try:
        # 实时输出，同时捕获
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end='')  # 实时输出
                output_lines.append(line)
            elif process.poll() is not None:
                break
                
        # 读取剩余输出
        remaining = process.stdout.read()
        if remaining:
            print(remaining, end='')
            output_lines.append(remaining)
            
        duration = time.time() - start_time
        success = process.returncode == 0
        output = ''.join(output_lines)
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start_time
        return False, f"[TIMEOUT] 超过 {timeout} 秒", duration
        
    except Exception as e:
        duration = time.time() - start_time
        return False, f"[ERROR] {str(e)}\n{traceback.format_exc()}", duration


# =============================================================================
# 任务执行器
# =============================================================================

class TaskRunner:
    """任务执行器"""
    
    def __init__(self, log_manager: LogManager, global_results_ref: Dict):
        self.log_manager = log_manager
        self.results = global_results_ref  # 直接引用全局结果字典
        
    def run_task_1_download(self) -> bool:
        """Task 1: 模型下载"""
        script = SCRIPTS["download"]
        
        total_success = 0
        total_fail = 0
        
        for model_key in CONFIG.DOWNLOAD_MODELS:
            print_subheader(f"下载: {model_key}")
            
            cmd = [
                sys.executable, str(script),
                "--model", model_key,
            ]
            
            success, output, duration = run_command(cmd, f"download {model_key}")
            
            if success:
                print_success(f"{model_key} 下载完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} 下载失败")
                total_fail += 1
                
        print()
        print_info(f"下载统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_2_convert(self) -> bool:
        """Task 2: 模型转换（支持断点续传，跳过已存在的模型）"""
        script = SCRIPTS["convert"]
        
        total_success = 0
        total_fail = 0
        total_skip = 0
        
        # 模型名称映射（小写key -> 目录名大小写）
        model_name_map = {
            "llama3.2-1b-int8": "Llama3.2-1B-INT8",
            "llama3.2-1b-fp8": "Llama3.2-1B-FP8",
            "llama3.2-3b-int8": "Llama3.2-3B-INT8",
            "llama3.2-3b-fp8": "Llama3.2-3B-FP8",
            "qwen2.5-7b-int8": "Qwen2.5-7B-INT8",
            "qwen2.5-7b-fp8": "Qwen2.5-7B-FP8",
            "qwen2.5-14b-int8": "Qwen2.5-14B-INT8",
            "qwen2.5-14b-fp8": "Qwen2.5-14B-FP8",
        }
        
        for model_key in CONFIG.CONVERT_MODELS:
            for Z, L in CONFIG.CONVERT_SPARSITIES:
                # 检查目标目录是否已存在
                model_dir_name = model_name_map.get(model_key.lower(), model_key)
                target_dir = _PROJECT_ROOT / "checkpoints_slidesparse" / f"{model_dir_name}-SlideSparse-{Z}_{L}"
                
                if target_dir.exists() and (target_dir / "model.safetensors").exists():
                    print_info(f"跳过已存在: {model_key} {Z}_{L}")
                    total_skip += 1
                    continue
                
                print_subheader(f"转换: {model_key} -> SlideSparse-{Z}_{L}")
                
                cmd = [
                    sys.executable, str(script),
                    "--model", model_key,
                    "--Z", str(Z),
                    "--L", str(L),
                ]
                
                success, output, duration = run_command(
                    cmd, f"convert {model_key} {Z}_{L}",
                    cwd=script.parent
                )
                
                if success:
                    print_success(f"{model_key} {Z}_{L} 转换完成 ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{model_key} {Z}_{L} 转换失败")
                    total_fail += 1
                    
        print()
        print_info(f"转换统计: 成功 {total_success}, 跳过 {total_skip}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_3_tune_coarse(self) -> bool:
        """Task 3: 离线粗调优"""
        script = SCRIPTS["tune"]
        
        models_str = ",".join(CONFIG.TUNE_COARSE_MODELS)
        m_list_str = ",".join(map(str, CONFIG.TUNE_COARSE_M_LIST))
        
        cmd = [
            sys.executable, str(script),
            "--model", models_str,
            "--dtype", CONFIG.TUNE_COARSE_DTYPE,
            "--m_list", m_list_str,
            "--Lmax", str(CONFIG.TUNE_COARSE_LMAX),
            "--warmup", str(CONFIG.TUNE_COARSE_WARMUP),
            "--repeat", str(CONFIG.TUNE_COARSE_REPEAT),
            "--kernels", CONFIG.TUNE_COARSE_KERNELS,
        ]
        
        success, output, duration = run_command(cmd, "coarse tune")
        return success
    
    def run_task_4_tune_fine(self) -> bool:
        """Task 4: 离线细调优"""
        script = SCRIPTS["tune"]
        
        models_str = ",".join(CONFIG.TUNE_FINE_MODELS)
        m_list_str = ",".join(map(str, CONFIG.TUNE_FINE_M_LIST))
        
        cmd = [
            sys.executable, str(script),
            "--model", models_str,
            "--dtype", CONFIG.TUNE_FINE_DTYPE,
            "--m_list", m_list_str,
            "--Lmax", str(CONFIG.TUNE_FINE_LMAX),
            "--warmup", str(CONFIG.TUNE_FINE_WARMUP),
            "--repeat", str(CONFIG.TUNE_FINE_REPEAT),
            "--kernels", CONFIG.TUNE_FINE_KERNELS,
        ]
        
        success, output, duration = run_command(cmd, "fine tune")
        return success
    
    def run_task_5_bench_simple(self) -> bool:
        """Task 5: 简单端到端 Benchmark"""
        script = SCRIPTS["benchmark"]
        
        total_success = 0
        total_fail = 0
        
        for model_key in CONFIG.BENCH_SIMPLE_MODELS:
            print_subheader(f"Benchmark: {model_key}")
            
            cmd = [
                sys.executable, str(script),
                "--model", model_key,
                "--backend", CONFIG.BENCH_SIMPLE_BACKENDS,
                "--stage", CONFIG.BENCH_SIMPLE_STAGES,
                "--sparsity", ",".join(CONFIG.BENCH_SIMPLE_SPARSITIES),
            ]
            
            if CONFIG.BENCH_SIMPLE_M_QUICK:
                cmd.extend(["--M", "quick"])
            
            success, output, duration = run_command(cmd, f"bench {model_key}")
            
            if success:
                print_success(f"{model_key} Benchmark 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} Benchmark 失败")
                total_fail += 1
                
        print()
        print_info(f"Benchmark 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_6_bench_prefill(self) -> bool:
        """Task 6: 完整 Prefill Benchmark"""
        script = SCRIPTS["benchmark"]
        
        total_success = 0
        total_fail = 0
        
        m_list_str = ",".join(map(str, CONFIG.BENCH_PREFILL_M_LIST))
        
        for model_key in CONFIG.BENCH_PREFILL_MODELS:
            print_subheader(f"Prefill Benchmark: {model_key}")
            
            cmd = [
                sys.executable, str(script),
                "--model", model_key,
                "--backend", CONFIG.BENCH_PREFILL_BACKENDS,
                "--stage", CONFIG.BENCH_PREFILL_STAGES,
                "--sparsity", ",".join(CONFIG.BENCH_PREFILL_SPARSITIES),
                "--M", m_list_str,
            ]
            
            success, output, duration = run_command(cmd, f"prefill {model_key}")
            
            if success:
                print_success(f"{model_key} Prefill 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} Prefill 失败")
                total_fail += 1
                
        print()
        print_info(f"Prefill 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_7_bench_decode(self) -> bool:
        """Task 7: 完整 Decode Benchmark"""
        script = SCRIPTS["benchmark"]
        
        total_success = 0
        total_fail = 0
        
        m_list_str = ",".join(map(str, CONFIG.BENCH_DECODE_M_LIST))
        
        for model_key in CONFIG.BENCH_DECODE_MODELS:
            print_subheader(f"Decode Benchmark: {model_key}")
            
            cmd = [
                sys.executable, str(script),
                "--model", model_key,
                "--backend", CONFIG.BENCH_DECODE_BACKENDS,
                "--stage", CONFIG.BENCH_DECODE_STAGES,
                "--sparsity", ",".join(CONFIG.BENCH_DECODE_SPARSITIES),
                "--M", m_list_str,
            ]
            
            success, output, duration = run_command(cmd, f"decode {model_key}")
            
            if success:
                print_success(f"{model_key} Decode 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} Decode 失败")
                total_fail += 1
                
        print()
        print_info(f"Decode 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_all(self, task_mask: List[bool]) -> None:
        """执行所有任务（结果直接写入 self.results）"""
        task_runners = [
            self.run_task_1_download,
            self.run_task_2_convert,
            self.run_task_3_tune_coarse,
            self.run_task_4_tune_fine,
            self.run_task_5_bench_simple,
            self.run_task_6_bench_prefill,
            self.run_task_7_bench_decode,
        ]
        
        for i, (enabled, runner, name) in enumerate(zip(task_mask, task_runners, TASK_NAMES)):
            if not enabled:
                print_info(f"跳过 Task {i+1}: {name}")
                self.results[i] = {"skipped": True}
                continue
                
            self.log_manager.log_task_start(i, name)
            
            start_time = time.time()
            try:
                success = runner()
            except Exception as e:
                print_error(f"Task {i+1} 异常: {e}")
                traceback.print_exc()
                success = False
                
            duration = time.time() - start_time
            
            self.results[i] = {
                "success": success,
                "duration": duration,
                "skipped": False,
            }
            
            self.log_manager.log_task_end(i, name, success, duration)
            
            # 保存状态（方便中途查看进度）
            self.log_manager.save_status({
                "current_task": i,
                "results": self.results,
                "last_update": datetime.now().isoformat(),
            })


# =============================================================================
# 信号处理
# =============================================================================

_GLOBAL_LOG_MANAGER: Optional[LogManager] = None
_GLOBAL_RESULTS: Dict = {}


def signal_handler(signum, frame):
    """处理中断信号"""
    print()
    print("=" * 70)
    print(f"{Colors.YELLOW}收到中断信号 (signal {signum}){Colors.NC}")
    print("=" * 70)
    
    if _GLOBAL_LOG_MANAGER:
        _GLOBAL_LOG_MANAGER.save_status({
            "interrupted": True,
            "signal": signum,
            "results": _GLOBAL_RESULTS,
            "timestamp": datetime.now().isoformat(),
        })
        print_info(f"状态已保存: {_GLOBAL_LOG_MANAGER.status_file}")
        _GLOBAL_LOG_MANAGER.stop()
        
    sys.exit(130)


# =============================================================================
# 主函数
# =============================================================================

def parse_task_mask(mask_str: str) -> List[bool]:
    """解析任务 mask 字符串"""
    parts = mask_str.split(",")
    if len(parts) != 7:
        raise ValueError(f"任务 mask 必须有 7 个值（当前 {len(parts)} 个）: {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def print_config_info():
    """打印配置信息"""
    print_header("任务配置")
    print()
    
    print(f"{Colors.CYAN}Task 1: 模型下载{Colors.NC}")
    print(f"  模型: {CONFIG.DOWNLOAD_MODELS}")
    print()
    
    print(f"{Colors.CYAN}Task 2: 模型转换{Colors.NC}")
    print(f"  模型: {CONFIG.CONVERT_MODELS}")
    print(f"  稀疏度: {CONFIG.CONVERT_SPARSITIES}")
    print()
    
    print(f"{Colors.CYAN}Task 3: 离线粗调优{Colors.NC}")
    print(f"  模型: {CONFIG.TUNE_COARSE_MODELS}")
    print(f"  M 列表: {CONFIG.TUNE_COARSE_M_LIST}")
    print(f"  Kernels: {CONFIG.TUNE_COARSE_KERNELS} (cuBLAS + quant_only)")
    print(f"  Warmup/Repeat: {CONFIG.TUNE_COARSE_WARMUP}/{CONFIG.TUNE_COARSE_REPEAT}")
    print()
    
    print(f"{Colors.CYAN}Task 4: 离线细调优{Colors.NC}")
    print(f"  模型: {CONFIG.TUNE_FINE_MODELS}")
    print(f"  M 列表: {CONFIG.TUNE_FINE_M_LIST}")
    print(f"  Kernels: {CONFIG.TUNE_FINE_KERNELS} (cuSPARSE + Triton)")
    print(f"  Warmup/Repeat: {CONFIG.TUNE_FINE_WARMUP}/{CONFIG.TUNE_FINE_REPEAT}")
    print()
    
    print(f"{Colors.CYAN}Task 5: 简单 Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.BENCH_SIMPLE_MODELS}")
    print(f"  Backend: {CONFIG.BENCH_SIMPLE_BACKENDS}")
    print(f"  稀疏度: {CONFIG.BENCH_SIMPLE_SPARSITIES}")
    print()
    
    print(f"{Colors.CYAN}Task 6: Prefill Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.BENCH_PREFILL_MODELS}")
    print(f"  M 列表: {CONFIG.BENCH_PREFILL_M_LIST}")
    print(f"  Backend: {CONFIG.BENCH_PREFILL_BACKENDS}")
    print()
    
    print(f"{Colors.CYAN}Task 7: Decode Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.BENCH_DECODE_MODELS}")
    print(f"  M 列表: {CONFIG.BENCH_DECODE_M_LIST}")
    print(f"  Backend: {CONFIG.BENCH_DECODE_BACKENDS}")
    print()


def main():
    global _GLOBAL_LOG_MANAGER, _GLOBAL_RESULTS
    
    parser = argparse.ArgumentParser(
        description="SlideSparse 端到端准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--task", type=str, default="1,1,1,1,1,1,1",
        help='任务 mask，格式 "1,1,1,1,1,1,1" (默认: 全部执行)'
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="使用的 GPU ID (默认: 0)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="仅显示配置信息，不执行"
    )
    parser.add_argument(
        "--no-protect", action="store_true",
        help="禁用进程保护"
    )
    
    args = parser.parse_args()
    
    # 解析任务 mask
    try:
        task_mask = parse_task_mask(args.task)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    # 显示配置信息
    print_header("SlideSparse 端到端准备脚本")
    print()
    print(f"  GPU:     {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:  {hw_info.python_tag}")
    print(f"  CUDA:    {hw_info.cuda_tag}")
    print()
    print("  任务列表:")
    for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} Task {i+1}: {name}")
    print()
    
    if args.info:
        print_config_info()
        return 0
    
    # 设置 GPU 环境
    setup_gpu_environment(args.gpu)
    
    # 设置进程保护
    if not args.no_protect:
        setup_process_protection()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建日志管理器
    log_manager = LogManager(_SCRIPT_DIR)
    _GLOBAL_LOG_MANAGER = log_manager
    log_manager.start()
    
    try:
        # 执行任务
        runner = TaskRunner(log_manager, _GLOBAL_RESULTS)
        runner.run_all(task_mask)
        results = _GLOBAL_RESULTS
        
        # 打印最终总结
        print()
        print_header("最终总结")
        print()
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        total_duration = 0
        
        for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
            r = results.get(i, {})
            
            if r.get("skipped"):
                print(f"  Task {i+1}: {name} - {Colors.YELLOW}SKIPPED{Colors.NC}")
                skip_count += 1
            elif r.get("success"):
                d = r.get("duration", 0)
                total_duration += d
                print(f"  Task {i+1}: {name} - {Colors.GREEN}SUCCESS{Colors.NC} ({d:.1f}s)")
                success_count += 1
            else:
                d = r.get("duration", 0)
                total_duration += d
                print(f"  Task {i+1}: {name} - {Colors.RED}FAILED{Colors.NC} ({d:.1f}s)")
                fail_count += 1
                
        print()
        print(f"  总计: {Colors.GREEN}{success_count} 成功{Colors.NC}, ", end="")
        if fail_count > 0:
            print(f"{Colors.RED}{fail_count} 失败{Colors.NC}, ", end="")
        else:
            print(f"{fail_count} 失败, ", end="")
        print(f"{skip_count} 跳过")
        print(f"  总耗时: {total_duration:.1f} 秒 ({total_duration/3600:.2f} 小时)")
        print()
        
        # 保存最终状态
        log_manager.save_status({
            "completed": True,
            "results": results,
            "summary": {
                "success": success_count,
                "failed": fail_count,
                "skipped": skip_count,
                "total_duration": total_duration,
            },
            "timestamp": datetime.now().isoformat(),
        })
        
        print_info(f"日志文件: {log_manager.main_log}")
        print_info(f"状态文件: {log_manager.status_file}")
        
        return 0 if fail_count == 0 else 1
        
    finally:
        log_manager.stop()


if __name__ == "__main__":
    sys.exit(main())
