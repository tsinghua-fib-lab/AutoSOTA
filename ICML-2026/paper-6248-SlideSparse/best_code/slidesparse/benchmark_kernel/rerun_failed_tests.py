#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
重跑失败测试的脚本

用法:
    # 查看所有可用任务
    python3 rerun_failed_tests.py --list
    
    # 重跑所有失败的测试
    python3 rerun_failed_tests.py --all
    
    # 只重跑 cuBLASLt FP4
    python3 rerun_failed_tests.py --cublaslt-fp4
    
    # 只重跑 cuSPARSELt INT8 失败的测试
    python3 rerun_failed_tests.py --cusparselt-int8
    
    # 重跑特定模型
    python3 rerun_failed_tests.py --cublaslt-fp4 --models Llama3.2-1B-INT8,Qwen2.5-7B-FP8
    
    # Dry run（只打印命令，不执行）
    python3 rerun_failed_tests.py --all --dry-run
"""

import argparse
import os
import sys
import subprocess
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# 添加项目路径以导入 utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from slidesparse.benchmark_kernel.utils import IncrementalResultSaver

# =============================================================================
# 配置
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
BENCHMARK_ENTRY = SCRIPT_DIR / "benchmark_entry.py"
LOG_DIR = SCRIPT_DIR / "rerun_logs"

# M 列表（最大到 16384）
M_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
M_LIST_STR = ",".join(str(m) for m in M_LIST)

# FP4 初始 M 列表（从最大开始，失败时递减）
# 策略：如果失败，去掉最后一个 M 值重试，直到成功或 M_LIST 只剩到 1024
M_LIST_FP4_INITIAL = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
M_LIST_FP4_MIN_MAX = 1024  # 最小的 max M 值，低于这个就放弃

# 所有模型
ALL_MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
]

# cuSPARSELt INT8 失败的测试
# 注意：Qwen2.5-7B 的测试在第二次运行时已经成功完成，不需要重跑
# 只剩下 SQUARE_2_6 一个测试未完成
CUSPARSELT_INT8_FAILED = [
    # (model, sparsity_list)
    # ("Qwen2.5-7B-INT8", "2_12,2_16,2_inf"),  # 已完成，不需要重跑
    # ("Qwen2.5-7B-FP8", "2_12,2_16,2_inf"),   # 已完成，不需要重跑
    ("square", "2_6"),  # 唯一未完成的
]


# =============================================================================
# 日志
# =============================================================================

class Logger:
    def __init__(self, log_file: Path):
        LOG_DIR.mkdir(exist_ok=True)
        self.log_file = log_file
        self.f = open(log_file, 'w', encoding='utf-8')
        self._log(f"=== Rerun Failed Tests ===")
        self._log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"M_list: {M_LIST}")
        self._log("")
    
    def _log(self, msg: str):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    
    def info(self, msg: str):
        self._log(f"[INFO] {msg}")
    
    def success(self, msg: str):
        self._log(f"[SUCCESS] {msg}")
    
    def error(self, msg: str):
        self._log(f"[ERROR] {msg}")
    
    def cmd(self, cmd: str):
        self._log(f"[CMD] {cmd}")
    
    def close(self):
        self._log("")
        self._log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.f.close()


# =============================================================================
# 运行命令
# =============================================================================

def reset_gpu():
    """重置 GPU 状态，清理 CUDA context"""
    try:
        # 使用 nvidia-smi 重置 GPU（需要没有其他进程使用 GPU）
        # 这是最彻底的方式
        result = subprocess.run(
            ["nvidia-smi", "--gpu-reset", "-i", "0"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[GPU] GPU reset successful")
            return True
        else:
            # 如果无法重置，至少尝试清理 CUDA 缓存
            print(f"[GPU] GPU reset failed (code {result.returncode}), trying Python cleanup...")
    except Exception as e:
        print(f"[GPU] nvidia-smi reset failed: {e}")
    
    # 备用方案：通过 Python 清理
    try:
        cleanup_code = """
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
gc.collect()
print('[GPU] Python CUDA cleanup done')
"""
        subprocess.run(
            ["python3", "-c", cleanup_code],
            capture_output=True,
            text=True,
            timeout=10
        )
    except Exception as e:
        print(f"[GPU] Python cleanup failed: {e}")
    
    return False


def _convert_base64_to_bytes(obj: Any) -> Any:
    """
    递归将 JSON 中的 base64 编码的 algo_data 字符串转换回 bytes
    
    这是 IncrementalResultSaver._convert_bytes_to_base64() 的逆操作
    只转换 algo_data 字段，其他字段保持不变
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == 'algo_data' and isinstance(v, str):
                # 将 base64 字符串解码回 bytes
                try:
                    result[k] = base64.b64decode(v)
                except Exception:
                    result[k] = v  # 如果解码失败，保持原样
            else:
                result[k] = _convert_base64_to_bytes(v)
        return result
    elif isinstance(obj, list):
        return [_convert_base64_to_bytes(item) for item in obj]
    else:
        return obj


def convert_progress_to_final(
    progress_file: Path, 
    logger: Optional[Logger] = None,
    delete_after_convert: bool = True,
) -> bool:
    """
    将 progress.json 转换为最终的 .csv 和 .json 格式
    
    直接复用 IncrementalResultSaver 的 finalize() 逻辑，确保格式完全一致
    
    Args:
        progress_file: progress.json 文件路径
        logger: 日志记录器
        delete_after_convert: 转换成功后是否删除原始 progress 文件
                              True = 转换后删除（默认，用于重跑模式）
                              False = 只转换不删除（安全模式，用于抢救数据）
    
    注意事项:
    1. progress.json 中的 algo_data 已经是 base64 字符串，需要解码回 bytes
    2. progress.json 中的 m_results 键是字符串，需要转换为整数
    
    Returns:
        True 转换成功，False 转换失败或无有效数据
    """
    if not progress_file.exists():
        return False
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        # 检查是否有有效结果
        results = progress.get('results', [])
        if not results:
            if logger:
                logger.info(f"  Progress 文件无有效结果: {progress_file.name}")
            return False
        
        # 提取元数据
        meta = progress['meta']
        
        # 过滤出成功的 M 值，并做必要的类型转换
        successful_m_set = set()
        filtered_results = []
        
        for nk_result in results:
            m_results = nk_result.get('m_results', {})
            # 将 m_results 的键从字符串转换为整数，并解码 base64 algo_data
            filtered_m_results = {}
            
            for m_str, m_data in m_results.items():
                # 只保留有结果且无错误的 M
                if m_data.get('results') and not m_data.get('error'):
                    m_int = int(m_str)
                    successful_m_set.add(m_int)
                    # 解码 algo_data (base64 字符串 -> bytes)
                    converted_m_data = _convert_base64_to_bytes(m_data)
                    filtered_m_results[m_int] = converted_m_data
            
            if filtered_m_results:
                filtered_results.append({
                    'nk_id': nk_result['nk_id'],
                    'N': nk_result['N'],
                    'K': nk_result['K'],
                    'm_results': filtered_m_results,
                    'skipped': nk_result.get('skipped', False),
                    'skip_reason': nk_result.get('skip_reason'),
                })
        
        if not successful_m_set:
            if logger:
                logger.info(f"  Progress 文件无成功的 M 值: {progress_file.name}")
            return False
        
        successful_m_list = sorted(successful_m_set)
        
        # 从文件名解析 sparsity（如果有）
        filename = progress_file.name
        sparsity = None
        if '_2_' in filename:
            # 例如 alg_search_SQUARE_2_6.progress.json
            parts = filename.replace('.progress.json', '').split('_')
            for i, p in enumerate(parts):
                if p == '2' and i + 1 < len(parts):
                    sparsity = f"2_{parts[i+1]}"
                    break
        
        # 使用 IncrementalResultSaver 重新生成文件
        # 确定输出目录
        out_dir = progress_file.parent.parent.parent  # 去掉 hw_folder/dtype_folder
        
        saver = IncrementalResultSaver(
            out_dir=out_dir,
            model_name=meta['model_name'],
            dtype=meta['dtype'],
            backend=meta['backend'],
            mode=meta['mode'],
            warmup=meta['warmup'],
            repeat=meta['repeat'],
            m_list=successful_m_list,  # 使用过滤后的 M 列表
            nk_list=[(nk[0], nk[1]) for nk in meta['NK_list']],
            sparsity=sparsity,
        )
        
        # 添加过滤后的结果
        for nk_res in filtered_results:
            saver.add_nk_result(nk_res, save_progress=False)
        
        # 更新统计信息（添加注释说明这是部分结果）
        saver.search_stats['note'] = 'Partial results converted from progress file due to CUDA errors'
        
        # 最终保存，根据参数决定是否删除 progress 文件
        saver.finalize(delete_progress=delete_after_convert)
        
        if logger:
            logger.success(f"  转换成功! M_list={successful_m_list}")
            logger.info(f"    输出目录: {saver.subdir}")
            if delete_after_convert:
                logger.info(f"    已删除 progress 文件: {progress_file.name}")
            else:
                logger.info(f"    保留 progress 文件: {progress_file.name}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"  转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_benchmark(
    model: str,
    dtype: str,
    backend: str,
    m_list: List[int],
    sparsity: Optional[str] = None,
    logger: Optional[Logger] = None,
    dry_run: bool = False,
) -> bool:
    """运行单个 benchmark 测试"""
    m_list_str = ",".join(str(m) for m in m_list)
    
    cmd = [
        sys.executable, str(BENCHMARK_ENTRY),
        "--model", model,
        "--dtype", dtype,
        "--backend", backend,
        "--m_list", m_list_str,
        "--warmup", "25",
        "--repeat", "50",
    ]
    
    if sparsity:
        cmd.extend(["--sparsity", sparsity])
    
    cmd_str = " ".join(cmd)
    
    if logger:
        logger.cmd(cmd_str)
    
    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return True
    
    try:
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        if result.returncode == 0:
            if logger:
                logger.success(f"{model} {dtype} {backend} {sparsity or ''}")
            return True
        else:
            if logger:
                logger.error(f"{model} {dtype} {backend} {sparsity or ''} (code={result.returncode})")
            return False
    except Exception as e:
        if logger:
            logger.error(f"{model} {dtype} {backend} {sparsity or ''}: {e}")
        return False


def run_benchmark_with_retry(
    model: str,
    dtype: str,
    backend: str,
    sparsity: Optional[str] = None,
    logger: Optional[Logger] = None,
    dry_run: bool = False,
) -> bool:
    """
    运行 benchmark，如果失败则尝试从 progress 文件恢复部分结果
    
    策略：
    1. 运行完整的 M_LIST
    2. 如果失败，检查是否有 progress.json
    3. 如果有，将 progress 转换为最终结果（保留成功的部分）
    """
    is_fp4 = dtype.lower() in ("fp4e2m1", "fp4")
    
    # 确定 progress 文件路径
    if backend.lower() == "cublaslt":
        base_dir = SCRIPT_DIR / "cuBLASLt/alg_search_results/RTX5080_cc120_py312_cu129_x86_64"
        dtype_dir = "FP4" if is_fp4 else dtype.upper()
    else:
        base_dir = SCRIPT_DIR / "cuSPARSELt/alg_search_results/RTX5080_cc120_py312_cu129_x86_64"
        dtype_dir = "FP4" if is_fp4 else dtype.upper()
    
    progress_dir = base_dir / dtype_dir
    
    # 构建 progress 文件名
    model_name = model.upper() if model.lower() == "square" else model
    if sparsity:
        progress_filename = f"alg_search_{model_name}_{sparsity}.progress.json"
    else:
        progress_filename = f"alg_search_{model_name}.progress.json"
    
    progress_file = progress_dir / progress_filename
    
    # 删除旧的 progress 文件（如果存在）
    if progress_file.exists() and not dry_run:
        if logger:
            logger.info(f"  删除旧的 progress 文件: {progress_filename}")
        progress_file.unlink()
    
    # 重置 GPU
    reset_gpu()
    time.sleep(2)
    
    # 运行测试
    m_list = M_LIST_FP4_INITIAL if is_fp4 else M_LIST
    if logger:
        logger.info(f"  M_LIST: {m_list}")
    
    success = run_benchmark(
        model, dtype, backend, m_list,
        sparsity=sparsity, logger=logger, dry_run=dry_run
    )
    
    if success:
        return True
    
    # 失败了，尝试从 progress 恢复
    if dry_run:
        return False
    
    if logger:
        logger.info(f"  测试失败，尝试从 progress 文件恢复...")
    
    # 等待一下让文件系统同步
    time.sleep(1)
    
    if progress_file.exists():
        if convert_progress_to_final(progress_file, logger):
            if logger:
                logger.success(f"  从 progress 恢复成功（部分结果）")
            return True  # 算作成功，因为我们保存了部分结果
        else:
            if logger:
                logger.error(f"  从 progress 恢复失败")
            return False
    else:
        if logger:
            logger.error(f"  没有找到 progress 文件")
        return False


# =============================================================================
# 任务函数
# =============================================================================

def run_cublaslt_fp4(
    models: Optional[List[str]] = None,
    logger: Optional[Logger] = None,
    dry_run: bool = False,
) -> int:
    """重跑 cuBLASLt FP4 测试（带递减 M_LIST 重试）"""
    if logger:
        logger.info("=" * 60)
        logger.info("Task: cuBLASLt FP4 (with adaptive M_LIST retry)")
        logger.info(f"初始 M_LIST: {M_LIST_FP4_INITIAL}")
        logger.info(f"最小 max_M: {M_LIST_FP4_MIN_MAX}")
        logger.info("=" * 60)
    
    target_models = models or ALL_MODELS
    success_count = 0
    
    for model in target_models:
        if logger:
            logger.info(f"Running: {model}")
        
        if run_benchmark_with_retry(model, "fp4e2m1", "cublaslt", logger=logger, dry_run=dry_run):
            success_count += 1
    
    # SQUARE
    if not models:  # 只有在跑全部时才跑 SQUARE
        if logger:
            logger.info("Running: SQUARE")
        if run_benchmark_with_retry("square", "fp4e2m1", "cublaslt", logger=logger, dry_run=dry_run):
            success_count += 1
    
    total = len(target_models) + (1 if not models else 0)
    if logger:
        logger.info(f"cuBLASLt FP4 完成: {success_count}/{total}")
    
    return success_count


def run_cusparselt_int8(
    logger: Optional[Logger] = None,
    dry_run: bool = False,
) -> int:
    """重跑 cuSPARSELt INT8 失败的测试"""
    if logger:
        logger.info("=" * 60)
        logger.info("Task: cuSPARSELt INT8 (failed tests)")
        logger.info("=" * 60)
    
    # 先删除 progress 文件
    progress_dir = SCRIPT_DIR / "cuSPARSELt/alg_search_results/RTX5080_cc120_py312_cu129_x86_64/INT8"
    if progress_dir.exists() and not dry_run:
        for f in progress_dir.glob("*.progress.json"):
            if logger:
                logger.info(f"Removing: {f.name}")
            f.unlink()
    
    success_count = 0
    for model, sparsity in CUSPARSELT_INT8_FAILED:
        if logger:
            logger.info(f"Running: {model} sparsity={sparsity}")
        
        # 每次测试前重置 GPU
        reset_gpu()
        time.sleep(5)
        
        if run_benchmark(model, "int8", "cusparselt", M_LIST, sparsity=sparsity, logger=logger, dry_run=dry_run):
            success_count += 1
    
    if logger:
        logger.info(f"cuSPARSELt INT8 完成: {success_count}/{len(CUSPARSELT_INT8_FAILED)}")
    
    return success_count


def list_tasks():
    """列出所有可用任务"""
    print("=" * 60)
    print("可用任务:")
    print("=" * 60)
    print()
    print("1. cuBLASLt FP4 (--cublaslt-fp4)")
    print("   - 所有 8 个模型 + SQUARE")
    print(f"   - M 列表: {M_LIST_FP4_INITIAL}")
    print("   - 策略: 如果失败，从 progress.json 恢复已成功的部分结果")
    print()
    print("2. cuSPARSELt INT8 失败测试 (--cusparselt-int8)")
    print("   - SQUARE: 2_6 (唯一未完成的)")
    print(f"   - M 列表: {M_LIST}")
    print("   - 注: Qwen2.5-7B 的 INT8 测试已在第二次运行时完成")
    print()
    print("3. cuSPARSELt FP4 失败测试 (--cusparselt-fp4)")
    print("   - Qwen2.5 + FP4 + 2_6 失败的测试")
    print("   - 用于 GB10 上的转换")
    print()
    print("转换模式 (--convert-only):")
    print("  只转换已有的 progress.json 文件，不重新运行测试")
    print("  默认保留原始 progress 文件，使用 --delete 删除")
    print()
    print("过滤选项:")
    print("  --hw GB10       只处理 GB10 硬件目录")
    print("  --hw A100       只处理 A100 硬件目录")
    print("  --sparsity 2_6  只处理包含 2_6 稀疏度的文件")
    print()
    print("运行示例:")
    print("  # 重跑模式（会重新运行测试）")
    print("  python3 rerun_failed_tests.py --all")
    print("  python3 rerun_failed_tests.py --cublaslt-fp4")
    print("  python3 rerun_failed_tests.py --cusparselt-int8")
    print("  python3 rerun_failed_tests.py --cublaslt-fp4 --models Llama3.2-1B-INT8")
    print()
    print("  # 转换模式（只转换，不重新运行）")
    print("  python3 rerun_failed_tests.py --convert-only --cublaslt-fp4 --hw GB10")
    print("  python3 rerun_failed_tests.py --convert-only --cusparselt-fp4 --hw GB10 --sparsity 2_6")
    print("  python3 rerun_failed_tests.py --convert-only --all --hw GB10")
    print("  python3 rerun_failed_tests.py --convert-only --all --hw GB10 --delete  # 转换后删除")
    print()
    print("  # Dry run（只打印命令，不执行）")
    print("  python3 rerun_failed_tests.py --all --dry-run")


# =============================================================================
# Main
# =============================================================================

def convert_all_progress_files(
    backend: str,
    dtype: str,
    logger: Optional[Logger] = None,
    delete_after_convert: bool = False,
    hw_filter: Optional[str] = None,
    sparsity_filter: Optional[str] = None,
) -> int:
    """
    转换指定目录下的所有 progress.json 文件为最终格式
    
    Args:
        backend: "cublaslt" 或 "cusparselt"
        dtype: 数据类型（如 "fp4e2m1", "int8", "fp16" 等）
        logger: 日志记录器
        delete_after_convert: 转换成功后是否删除原始文件（默认 False，安全模式）
        hw_filter: 硬件过滤器（如 "GB10", "A100"），只转换匹配的硬件目录
        sparsity_filter: 稀疏度过滤器（如 "2_6"），只转换匹配的稀疏度文件
    
    Returns:
        成功转换的文件数
    """
    if backend.lower() == "cublaslt":
        base_dir = SCRIPT_DIR / "cuBLASLt/alg_search_results"
    else:
        base_dir = SCRIPT_DIR / "cuSPARSELt/alg_search_results"
    
    # 确定 dtype 目录名
    dtype_dir = dtype.upper()
    if dtype.lower() == "fp4e2m1":
        dtype_dir = "FP4"
    elif dtype.lower() == "int8":
        dtype_dir = "INT8"
    
    # 查找指定 dtype 目录下的所有 progress 文件
    # 路径格式: base_dir/hw_folder/dtype_dir/*.progress.json
    pattern = f"*/{dtype_dir}/*.progress.json"
    progress_files = list(base_dir.glob(pattern))
    
    # 应用过滤器
    if hw_filter:
        progress_files = [f for f in progress_files if hw_filter in str(f)]
    if sparsity_filter:
        progress_files = [f for f in progress_files if f"_{sparsity_filter}" in f.name or f.name.endswith(f"{sparsity_filter}.progress.json")]
    
    if logger:
        logger.info(f"找到 {len(progress_files)} 个 progress 文件需要转换")
        logger.info(f"  delete_after_convert={delete_after_convert}")
        if hw_filter:
            logger.info(f"  硬件过滤: {hw_filter}")
        if sparsity_filter:
            logger.info(f"  稀疏度过滤: {sparsity_filter}")
        for f in progress_files:
            logger.info(f"  - {f.relative_to(base_dir)}")
    
    success_count = 0
    for progress_file in progress_files:
        if logger:
            logger.info(f"转换: {progress_file.name}")
        
        if convert_progress_to_final(progress_file, logger, delete_after_convert=delete_after_convert):
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description="重跑失败的测试")
    parser.add_argument("--list", action="store_true", help="列出所有可用任务")
    parser.add_argument("--all", action="store_true", help="重跑所有失败的测试")
    parser.add_argument("--cublaslt-fp4", action="store_true", help="重跑 cuBLASLt FP4")
    parser.add_argument("--cusparselt-int8", action="store_true", help="重跑 cuSPARSELt INT8 失败的测试")
    parser.add_argument("--convert-only", action="store_true", 
                        help="只转换已有的 progress.json 文件，不重新运行测试")
    parser.add_argument("--cusparselt-fp4", action="store_true", 
                        help="转换 cuSPARSELt FP4 失败的测试 (Qwen2.5 + 2_6)")
    parser.add_argument("--delete", action="store_true",
                        help="转换后删除原始 progress 文件（默认保留）")
    parser.add_argument("--hw", type=str, default=None,
                        help="硬件过滤器（如 GB10, A100），只处理匹配的目录")
    parser.add_argument("--sparsity", type=str, default=None,
                        help="稀疏度过滤器（如 2_6），只处理匹配的文件")
    parser.add_argument("--models", type=str, help="指定模型列表（逗号分隔）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    
    args = parser.parse_args()
    
    if args.list:
        list_tasks()
        return
    
    if not (args.all or args.cublaslt_fp4 or args.cusparselt_int8 or args.cusparselt_fp4):
        parser.print_help()
        print("\n请指定要执行的任务！")
        return
    
    # 解析模型列表
    models = args.models.split(",") if args.models else None
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"rerun_{timestamp}.log"
    logger = Logger(log_file)
    
    logger.info(f"Log file: {log_file}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Convert only: {args.convert_only}")
    logger.info(f"Delete after convert: {args.delete}")
    if args.hw:
        logger.info(f"Hardware filter: {args.hw}")
    if args.sparsity:
        logger.info(f"Sparsity filter: {args.sparsity}")
    
    total_success = 0
    total_tasks = 0
    
    try:
        # 如果只是转换模式
        if args.convert_only:
            if args.all or args.cublaslt_fp4:
                logger.info("=" * 60)
                logger.info("Converting cuBLASLt FP4 progress files...")
                logger.info("=" * 60)
                success = convert_all_progress_files(
                    "cublaslt", "fp4e2m1", logger,
                    delete_after_convert=args.delete,
                    hw_filter=args.hw,
                    sparsity_filter=args.sparsity,
                )
                total_success += success
                logger.info(f"转换完成: {success} 个文件")
            
            if args.all or args.cusparselt_int8:
                logger.info("=" * 60)
                logger.info("Converting cuSPARSELt INT8 progress files...")
                logger.info("=" * 60)
                success = convert_all_progress_files(
                    "cusparselt", "int8", logger,
                    delete_after_convert=args.delete,
                    hw_filter=args.hw,
                    sparsity_filter=args.sparsity,
                )
                total_success += success
                logger.info(f"转换完成: {success} 个文件")
            
            if args.all or args.cusparselt_fp4:
                logger.info("=" * 60)
                logger.info("Converting cuSPARSELt FP4 progress files...")
                logger.info("=" * 60)
                success = convert_all_progress_files(
                    "cusparselt", "fp4e2m1", logger,
                    delete_after_convert=args.delete,
                    hw_filter=args.hw,
                    sparsity_filter=args.sparsity,
                )
                total_success += success
                logger.info(f"转换完成: {success} 个文件")
        else:
            # 正常的重跑模式
            # cuBLASLt FP4
            if args.all or args.cublaslt_fp4:
                success = run_cublaslt_fp4(models=models, logger=logger, dry_run=args.dry_run)
                total_success += success
                total_tasks += len(models) if models else len(ALL_MODELS) + 1
            
            # cuSPARSELt INT8
            if args.all or args.cusparselt_int8:
                success = run_cusparselt_int8(logger=logger, dry_run=args.dry_run)
                total_success += success
                total_tasks += len(CUSPARSELT_INT8_FAILED)
        
        logger.info("")
        logger.info("=" * 60)
        if args.convert_only:
            logger.info(f"总计: {total_success} 个文件转换成功")
        else:
            logger.info(f"总计: {total_success}/{total_tasks} 成功")
        logger.info("=" * 60)
        
    finally:
        logger.close()
    
    print(f"\n日志保存到: {log_file}")


if __name__ == "__main__":
    main()
