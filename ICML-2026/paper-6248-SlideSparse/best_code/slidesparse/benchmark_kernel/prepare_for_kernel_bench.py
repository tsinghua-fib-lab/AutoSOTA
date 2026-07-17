#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark Preparation Script

Automate the complete workflow for cuBLASLt/cuSPARSELt kernel-level benchmark.

Pipeline Tasks:
===============
Task 1: cuBLASLt Model test (8 models, 5 precisions)
Task 2: cuBLASLt Square test (square matrix, 5 precisions)
Task 3: cuSPARSELt Model high-sparsity test (2_4, 2_6, 2_8, 2_10)
Task 4: cuSPARSELt Square high-sparsity test (2_4, 2_6, 2_8, 2_10)
Task 5: cuSPARSELt Model low-sparsity test (2_12, 2_14, 2_16, 2_inf)
Task 6: cuSPARSELt Square low-sparsity test (2_12, 2_14, 2_16, 2_inf)

M list: [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
Precision config:
    - Model mode: fp8e4m3, int8 (only these 2)
    - Square mode: fp16, bf16, int8, fp8e4m3, fp4e2m1 (all 5)
4 models (Kernel test only needs different model structures, INT8/FP8 have same NK dimensions):
    - Llama3.2-1B-INT8/FP8 pick one
    - Llama3.2-3B-INT8/FP8 pick one
    - Qwen2.5-7B-INT8/FP8 pick one
    - Qwen2.5-14B-INT8/FP8 pick one

Usage:
    # Execute all tasks
    python3 prepare_for_kernel_bench.py --task 1,1,1,1,1,1
    
    # Only execute cuBLASLt tests
    python3 prepare_for_kernel_bench.py --task 1,1,0,0,0,0
    
    # Only execute high-sparsity cuSPARSELt tests
    python3 prepare_for_kernel_bench.py --task 0,0,1,1,0,0
    
    # View task config (don't execute)
    python3 prepare_for_kernel_bench.py --info
    
    # Run in tmux
    tmux new -s kernel_bench
    cd /root/vllmbench/slidesparse/benchmark_kernel && python3 prepare_for_kernel_bench.py --task 0,0,0,0,1,1 --gpu 0

    tmux detach

    # 4. Reconnect tmux session
    tmux attach -t kernel_bench

    tmux kill-session -t kernel_bench
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
# Path Setup
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
# Task Configuration (hardcoded, usually no need to modify)
# =============================================================================

@dataclass
class TaskConfig:
    """Task configuration"""
    
    # Common M list
    M_LIST: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    ])
    
    # 4 models (Kernel test only needs different structures, INT8/FP8 have same NK dims, use INT8 here)
    MODELS: List[str] = field(default_factory=lambda: [
        "Llama3.2-1B-INT8",
        "Llama3.2-3B-INT8",
        "Qwen2.5-7B-INT8",
        "Qwen2.5-14B-INT8",
    ])
    
    # High sparsity config (standard + medium)
    HIGH_SPARSITY: List[str] = field(default_factory=lambda: [
        "2_4", "2_6", "2_8", "2_10"
    ])
    
    # Low sparsity config (deep sparse)
    LOW_SPARSITY: List[str] = field(default_factory=lambda: [
        "2_12", "2_14", "2_16", "2_inf"
    ])
    
    # Precision config
    # Square mode: test all precisions (fp16, bf16, int8, fp8e4m3, fp4e2m1)
    SQUARE_DTYPE: List[str] = field(default_factory=lambda: ["all"])
    # Model mode: only test FP8 and INT8 (reduce test count)
    MODEL_DTYPE: List[str] = field(default_factory=lambda: ["fp8e4m3", "int8"])
    
    # Test parameters
    WARMUP: int = 25
    REPEAT: int = 50


# Global config instance
CONFIG = TaskConfig()

# Task names
TASK_NAMES = [
    "cuBLASLt Model Test",
    "cuBLASLt Square Test",
    "cuSPARSELt Model High-Sparsity (2_4~2_10)",
    "cuSPARSELt Square High-Sparsity (2_4~2_10)",
    "cuSPARSELt Model Low-Sparsity (2_12~2_inf)",
    "cuSPARSELt Square Low-Sparsity (2_12~2_inf)",
]

# Script path
BENCHMARK_ENTRY = _SCRIPT_DIR / "benchmark_entry.py"


# =============================================================================
# Process Protection
# =============================================================================

def setup_process_protection():
    """Setup process protection (prevent OOM Killer)"""
    try:
        with open(f'/proc/{os.getpid()}/oom_score_adj', 'w') as f:
            f.write('-1000')
        print_info("OOM protection set (oom_score_adj=-1000)")
    except (PermissionError, FileNotFoundError):
        print_warning("Cannot set OOM protection (requires root)")


def setup_gpu_environment(gpu_id: str = "0", memory_fraction: float = 0.95):
    """
    Setup GPU environment variables
    
    Args:
        gpu_id: GPU device ID
        memory_fraction: GPU memory usage ratio (0.0-1.0), default 95%
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print_info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # Set PyTorch CUDA memory allocation strategy
    # PYTORCH_CUDA_ALLOC_CONF controls memory allocation behavior
    # max_split_size_mb prevents memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Set GPU memory usage ratio (via torch)
    try:
        import torch
        if torch.cuda.is_available():
            # Set memory usage ratio
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print_info(f"GPU memory usage ratio set: {memory_fraction*100:.0f}%")
    except Exception as e:
        print_warning(f"Cannot set GPU memory ratio: {e}")


# =============================================================================
# Log Management
# =============================================================================

class TeeLogger:
    """Logger that outputs to both console and log file"""
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8', buffering=1)
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(strip_ansi(message))
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class LogManager:
    """Log manager"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "kernel_bench_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = self.log_dir / f"kernel_bench_{timestamp}.log"
        self.status_file = self.log_dir / f"kernel_bench_{timestamp}_status.json"
        
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_logger = None
        
    def start(self):
        """Start logging"""
        with open(self.main_log, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SlideSparse Kernel Benchmark Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("Hardware:\n")
            f.write(f"  GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})\n")
            f.write(f"  Python: {hw_info.python_tag}\n")
            f.write(f"  CUDA: {hw_info.cuda_tag}\n")
            f.write(f"  Arch: {hw_info.arch_tag}\n")
            f.write("\n")
        
        self._tee_logger = TeeLogger(self.main_log)
        sys.stdout = self._tee_logger
        sys.stderr = self._tee_logger
        
        print_info(f"Log file: {self.main_log}")
        
    def stop(self):
        """Stop logging"""
        if self._tee_logger:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._tee_logger.close()
            
    def save_status(self, status: Dict[str, Any]):
        """Save task status"""
        import json
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
            
    def log_task_start(self, task_id: int, task_name: str):
        """Log task start"""
        print()
        print("=" * 70)
        print(f"TASK {task_id + 1}: {task_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
    def log_task_end(self, task_id: int, task_name: str, success: bool, duration: float):
        """Log task end"""
        print()
        print("-" * 70)
        status = f"{Colors.GREEN}SUCCESS{Colors.NC}" if success else f"{Colors.RED}FAILED{Colors.NC}"
        print(f"TASK {task_id + 1}: {task_name} - {status}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print("-" * 70)
        print()


# =============================================================================
# Command Execution
# =============================================================================

def run_command(
    cmd: List[str],
    name: str,
    cwd: Optional[Path] = None,
) -> Tuple[bool, str, float]:
    """
    Execute command with real-time output
    
    Args:
        cmd: Command and argument list
        name: Command description name (for logging)
        cwd: Working directory
    
    Returns:
        (success, output, duration)
    """
    start_time = time.time()
    
    print_info(f"Executing: {' '.join(cmd)}")
    if cwd:
        print_info(f"Working dir: {cwd}")
    print()
    
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        
        assert process.stdout is not None
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end='')
                output_lines.append(line)
            elif process.poll() is not None:
                break
                
        remaining = process.stdout.read()
        if remaining:
            print(remaining, end='')
            output_lines.append(remaining)
            
        duration = time.time() - start_time
        success = process.returncode == 0
        output = ''.join(output_lines)
        
        return success, output, duration
        
    except Exception as e:
        duration = time.time() - start_time
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass
        return False, f"[ERROR] {str(e)}\n{traceback.format_exc()}", duration


# =============================================================================
# Task Runner
# =============================================================================

class TaskRunner:
    """Task runner"""
    
    def __init__(self, log_manager: LogManager, global_results_ref: Dict):
        self.log_manager = log_manager
        self.results = global_results_ref
        
    def _build_base_cmd(self, backend: str, dtype: str) -> List[str]:
        """Build base command
        
        Args:
            backend: Backend type (cublaslt/cusparselt)
            dtype: Data type (fp16, bf16, int8, fp8e4m3, fp4e2m1, all)
        """
        m_list_str = ",".join(map(str, CONFIG.M_LIST))
        return [
            sys.executable, str(BENCHMARK_ENTRY),
            "--dtype", dtype,
            "--warmup", str(CONFIG.WARMUP),
            "--repeat", str(CONFIG.REPEAT),
            "--m_list", m_list_str,
            "--backend", backend,
        ]
    
    def run_task_1_cublaslt_model(self) -> bool:
        """Task 1: cuBLASLt Model test (only FP8/INT8)"""
        total_success = 0
        total_fail = 0
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuBLASLt Model: {model}")
            
            for dtype in CONFIG.MODEL_DTYPE:
                cmd = self._build_base_cmd("cublaslt", dtype=dtype)
                cmd.extend(["--model", model])
                
                success, output, duration = run_command(cmd, f"cublaslt {model} {dtype}")
                
                if success:
                    print_success(f"{model} [{dtype}] done ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{model} [{dtype}] failed")
                    total_fail += 1
                
        print()
        print_info(f"cuBLASLt Model stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_task_2_cublaslt_square(self) -> bool:
        """Task 2: cuBLASLt Square test (all precisions)"""
        print_subheader("cuBLASLt Square Test")
        
        total_success = 0
        total_fail = 0
        
        for dtype in CONFIG.SQUARE_DTYPE:
            cmd = self._build_base_cmd("cublaslt", dtype=dtype)
            cmd.extend(["--model", "square"])
            
            success, output, duration = run_command(cmd, f"cublaslt square {dtype}")
            
            if success:
                print_success(f"Square [{dtype}] test done ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"Square [{dtype}] test failed")
                total_fail += 1
        
        print_info(f"cuBLASLt Square stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_task_3_cusparselt_model_high(self) -> bool:
        """Task 3: cuSPARSELt Model high-sparsity test (FP8/INT8 only)"""
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.HIGH_SPARSITY)
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuSPARSELt Model High-Sparsity: {model}")
            
            for dtype in CONFIG.MODEL_DTYPE:
                cmd = self._build_base_cmd("cusparselt", dtype=dtype)
                cmd.extend([
                    "--model", model,
                    "--sparsity", sparsity_str,
                ])
                
                success, output, duration = run_command(cmd, f"cusparselt high {model} {dtype}")
                
                if success:
                    print_success(f"{model} [{dtype}] high-sparsity done ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{model} [{dtype}] high-sparsity failed")
                    total_fail += 1
                
        print()
        print_info(f"cuSPARSELt Model high-sparsity stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_task_4_cusparselt_square_high(self) -> bool:
        """Task 4: cuSPARSELt Square high-sparsity test (all precisions)"""
        print_subheader("cuSPARSELt Square High-Sparsity Test")
        
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.HIGH_SPARSITY)
        
        for dtype in CONFIG.SQUARE_DTYPE:
            cmd = self._build_base_cmd("cusparselt", dtype=dtype)
            cmd.extend([
                "--model", "square",
                "--sparsity", sparsity_str,
            ])
            
            success, output, duration = run_command(cmd, f"cusparselt square high {dtype}")
            
            if success:
                print_success(f"Square [{dtype}] high-sparsity test done ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"Square [{dtype}] high-sparsity test failed")
                total_fail += 1
        
        print_info(f"cuSPARSELt Square high-sparsity stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_task_5_cusparselt_model_low(self) -> bool:
        """Task 5: cuSPARSELt Model low-sparsity test (FP8/INT8 only)"""
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.LOW_SPARSITY)
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuSPARSELt Model Low-Sparsity: {model}")
            
            for dtype in CONFIG.MODEL_DTYPE:
                cmd = self._build_base_cmd("cusparselt", dtype=dtype)
                cmd.extend([
                    "--model", model,
                    "--sparsity", sparsity_str,
                ])
                
                success, output, duration = run_command(cmd, f"cusparselt low {model} {dtype}")
                
                if success:
                    print_success(f"{model} [{dtype}] low-sparsity done ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{model} [{dtype}] low-sparsity failed")
                    total_fail += 1
                
        print()
        print_info(f"cuSPARSELt Model low-sparsity stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_task_6_cusparselt_square_low(self) -> bool:
        """Task 6: cuSPARSELt Square low-sparsity test (all precisions)"""
        print_subheader("cuSPARSELt Square Low-Sparsity Test")
        
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.LOW_SPARSITY)
        
        for dtype in CONFIG.SQUARE_DTYPE:
            cmd = self._build_base_cmd("cusparselt", dtype=dtype)
            cmd.extend([
                "--model", "square",
                "--sparsity", sparsity_str,
            ])
            
            success, output, duration = run_command(cmd, f"cusparselt square low {dtype}")
            
            if success:
                print_success(f"Square [{dtype}] low-sparsity test done ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"Square [{dtype}] low-sparsity test failed")
                total_fail += 1
        
        print_info(f"cuSPARSELt Square low-sparsity stats: {total_success} success, {total_fail} failed")
        return total_fail == 0
    
    def run_all(self, task_mask: List[bool]) -> None:
        """Execute all tasks"""
        task_runners = [
            self.run_task_1_cublaslt_model,
            self.run_task_2_cublaslt_square,
            self.run_task_3_cusparselt_model_high,
            self.run_task_4_cusparselt_square_high,
            self.run_task_5_cusparselt_model_low,
            self.run_task_6_cusparselt_square_low,
        ]
        
        for i, (enabled, runner, name) in enumerate(zip(task_mask, task_runners, TASK_NAMES)):
            if not enabled:
                print_info(f"Skip Task {i+1}: {name}")
                self.results[i] = {"skipped": True}
                continue
                
            self.log_manager.log_task_start(i, name)
            
            start_time = time.time()
            try:
                success = runner()
            except Exception as e:
                print_error(f"Task {i+1} exception: {e}")
                traceback.print_exc()
                success = False
                
            duration = time.time() - start_time
            
            self.results[i] = {
                "success": success,
                "duration": duration,
                "skipped": False,
            }
            
            self.log_manager.log_task_end(i, name, success, duration)
            
            self.log_manager.save_status({
                "current_task": i,
                "results": self.results,
                "last_update": datetime.now().isoformat(),
            })


# =============================================================================
# Signal Handling
# =============================================================================

_GLOBAL_LOG_MANAGER: Optional[LogManager] = None
_GLOBAL_RESULTS: Dict = {}


def signal_handler(signum, frame):
    """Handle interrupt signal"""
    print()
    print("=" * 70)
    print(f"{Colors.YELLOW}Received interrupt signal (signal {signum}){Colors.NC}")
    print("=" * 70)
    
    if _GLOBAL_LOG_MANAGER:
        _GLOBAL_LOG_MANAGER.save_status({
            "interrupted": True,
            "signal": signum,
            "results": _GLOBAL_RESULTS,
            "timestamp": datetime.now().isoformat(),
        })
        print_info(f"Status saved: {_GLOBAL_LOG_MANAGER.status_file}")
        _GLOBAL_LOG_MANAGER.stop()
        
    sys.exit(130)


# =============================================================================
# Main Function
# =============================================================================

def parse_task_mask(mask_str: str) -> List[bool]:
    """Parse task mask string"""
    parts = mask_str.split(",")
    if len(parts) != 6:
        raise ValueError(f"Task mask must have 6 values (got {len(parts)}): {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def print_config_info():
    """Print configuration info"""
    print_header("Task Configuration")
    print()
    
    print(f"{Colors.CYAN}Common config:{Colors.NC}")
    print(f"  M list: {CONFIG.M_LIST}")
    print(f"  Model precision: {CONFIG.MODEL_DTYPE} (FP8/INT8 only)")
    print(f"  Square precision: {CONFIG.SQUARE_DTYPE} (all)")
    print(f"  Warmup/Repeat: {CONFIG.WARMUP}/{CONFIG.REPEAT}")
    print()
    
    print(f"{Colors.CYAN}Model list ({len(CONFIG.MODELS)}):{Colors.NC}")
    for model in CONFIG.MODELS:
        print(f"  - {model}")
    print()
    
    print(f"{Colors.CYAN}Task 1: cuBLASLt Model Test{Colors.NC}")
    print(f"  {len(CONFIG.MODELS)} models x {len(CONFIG.MODEL_DTYPE)} precisions ({CONFIG.MODEL_DTYPE})")
    print()
    
    print(f"{Colors.CYAN}Task 2: cuBLASLt Square Test{Colors.NC}")
    print(f"  Square matrix test (M=N=K), {CONFIG.SQUARE_DTYPE}")
    print()
    
    print(f"{Colors.CYAN}Task 3: cuSPARSELt Model High-Sparsity{Colors.NC}")
    print(f"  Sparsity: {CONFIG.HIGH_SPARSITY}, Precision: {CONFIG.MODEL_DTYPE}")
    print()
    
    print(f"{Colors.CYAN}Task 4: cuSPARSELt Square High-Sparsity{Colors.NC}")
    print(f"  Sparsity: {CONFIG.HIGH_SPARSITY}, Precision: {CONFIG.SQUARE_DTYPE}")
    print()
    
    print(f"{Colors.CYAN}Task 5: cuSPARSELt Model Low-Sparsity{Colors.NC}")
    print(f"  Sparsity: {CONFIG.LOW_SPARSITY}, Precision: {CONFIG.MODEL_DTYPE}")
    print()
    
    print(f"{Colors.CYAN}Task 6: cuSPARSELt Square Low-Sparsity{Colors.NC}")
    print(f"  Sparsity: {CONFIG.LOW_SPARSITY}, Precision: {CONFIG.SQUARE_DTYPE}")
    print()
    
    # Estimate memory requirement
    print(f"{Colors.YELLOW}⚠ Memory estimate:{Colors.NC}")
    max_m = max(CONFIG.M_LIST)
    # Max matrix: max_m × max_m × 2bytes (BF16) × 3 (W, A, R)
    max_mem_gb = (max_m * max_m * 2 * 3) / (1024**3)
    print(f"  Max single matrix (M={max_m}): ~{max_m*max_m*2/1024/1024:.1f} MB (BF16)")
    print(f"  Max total memory (W+A+R): ~{max_mem_gb:.1f} GB")
    print()


def main():
    global _GLOBAL_LOG_MANAGER, _GLOBAL_RESULTS
    
    parser = argparse.ArgumentParser(
        description="SlideSparse Kernel Benchmark 准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--task", type=str, default="1,1,1,1,1,1",
        help='Task mask, format "1,1,1,1,1,1" (default: execute all)'
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU ID to use (default: 0)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Only show config info, don't execute"
    )
    parser.add_argument(
        "--no-protect", action="store_true",
        help="Disable process protection"
    )
    parser.add_argument(
        "--memory-fraction", type=float, default=0.95,
        help="GPU memory usage ratio (0.0-1.0), default: 0.95 (95%)"
    )
    
    args = parser.parse_args()
    
    try:
        task_mask = parse_task_mask(args.task)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    print_header("SlideSparse Kernel Benchmark Preparation Script")
    print()
    print(f"  GPU:     {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:  {hw_info.python_tag}")
    print(f"  CUDA:    {hw_info.cuda_tag}")
    print(f"  Memory:  {args.memory_fraction*100:.0f}%")
    print()
    print("  Task list:")
    for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} Task {i+1}: {name}")
    print()
    
    if args.info:
        print_config_info()
        return 0
    
    setup_gpu_environment(args.gpu, args.memory_fraction)
    
    if not args.no_protect:
        setup_process_protection()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_manager = LogManager(_SCRIPT_DIR)
    _GLOBAL_LOG_MANAGER = log_manager
    log_manager.start()
    
    try:
        runner = TaskRunner(log_manager, _GLOBAL_RESULTS)
        runner.run_all(task_mask)
        results = _GLOBAL_RESULTS
        
        print()
        print_header("Final Summary")
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
        print(f"  Total: {Colors.GREEN}{success_count} success{Colors.NC}, ", end="")
        if fail_count > 0:
            print(f"{Colors.RED}{fail_count} failed{Colors.NC}, ", end="")
        else:
            print(f"{fail_count} failed, ", end="")
        print(f"{skip_count} skipped")
        print(f"  Total time: {total_duration:.1f} seconds ({total_duration/3600:.2f} hours)")
        print()
        
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
        
        print_info(f"Log file: {log_manager.main_log}")
        print_info(f"Status file: {log_manager.status_file}")
        
        # Output result directory info
        print()
        print_info("Results saved at:")
        print(f"  - cuBLASLt:   {_SCRIPT_DIR / 'cuBLASLt' / 'alg_search_results'}")
        print(f"  - cuSPARSELt: {_SCRIPT_DIR / 'cuSPARSELt' / 'alg_search_results'}")
        
        return 0 if fail_count == 0 else 1
        
    finally:
        log_manager.stop()


if __name__ == "__main__":
    sys.exit(main())
