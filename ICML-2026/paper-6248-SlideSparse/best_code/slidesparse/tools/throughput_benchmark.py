#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse vLLM Throughput Benchmark Script (official vllm bench)

Precise testing of W8A8 quantized model performance across different Backend/Sparsity/M for Prefill/Decode.

Test dimension hierarchy:
  Model -> Backend -> Sparsity -> Stage -> M

Backend support:
  - cutlass:    SlideSparse CUTLASS fallback (baseline)
  - cublaslt:   SlideSparse cuBLASLt dense GEMM
  - cusparselt: SlideSparse cuSPARSELt 2:N sparse GEMM (requires pre-sparsified checkpoint)

Core design principles:
  - Prefill test: Control M_prefill = max_num_seqs x prompt_length, minimize Decode overhead
  - Decode test:  Control M_decode = max_num_seqs, minimize Prefill overhead
  - Dynamically calculate max-model-len to maximize KV Cache utilization (Tight Fit strategy)
  - Disable Chunked Prefill for clean performance data

Usage examples:
    # Default test (using DEFAULT_MODEL_LIST)
    python3 throughput_benchmark.py
    
    # Test specific model with all backends
    python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8
    
    # Test cutlass backend only (baseline)
    python3 throughput_benchmark.py --model fp8 --backend cutlass
    
    # Test cublaslt backend only
    python3 throughput_benchmark.py --model fp8 --backend cublaslt
    
    # Test cusparselt with specific sparsities
    python3 throughput_benchmark.py --model fp8 --backend cusparselt --sparsity 2_4,2_8
    
    # Quick test (fewer M values)
    python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8 --M quick
    
    # Dry-run validation
    python3 throughput_benchmark.py --model fp8 --dry-run
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
from typing import Optional, Dict, List, Tuple

# Ensure slidesparse can be imported
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    HardwareInfo,
    model_registry,
    check_quant_support,
    get_model_local_path,
    hw_info,
    build_stem,
    extract_model_name,
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
    get_vllm_env_vars,
    check_triton_support_and_warn,
    print_hardware_info,
    get_hw_folder_name,
    build_backend_result_dir,
    get_checkpoint_path,
)


# ============================================================================
# Backend Support Detection
# ============================================================================

def check_backend_support(backend: str, quant: str) -> Tuple[bool, str]:
    """
    Check if specified backend supports current GPU and quantization type
    
    Args:
        backend: "cutlass" / "cublaslt" / "cusparselt"
        quant: "fp8" / "int8"
    
    Returns:
        (supported, reason)
    """
    quant_upper = quant.upper()
    
    if backend == "cutlass":
        # CUTLASS is SlideSparse's fallback path (internally calls vLLM's cutlass_scaled_mm)
        if quant_upper == "INT8":
            supported, reason = hw_info.supports_vllm_cutlass_int8
            if not supported:
                return False, f"vLLM CUTLASS INT8 not supported: {reason}"
        elif quant_upper == "FP8":
            supported, reason = hw_info.supports_vllm_cutlass_fp8
            if not supported:
                return False, f"vLLM CUTLASS FP8 not supported: {reason}"
        return True, "OK"
    
    elif backend == "cublaslt":
        # cuBLASLt supports sm_70+
        supported, reason = hw_info.supports_cublaslt
        if not supported:
            return False, f"cuBLASLt not supported: {reason}"
        return True, "OK"
    
    elif backend == "cusparselt":
        # cuSPARSELt supports sm_80+
        supported, reason = hw_info.supports_cusparselt
        if not supported:
            return False, f"cuSPARSELt not supported: {reason}"
        return True, "OK"
    
    return False, f"Unknown backend: {backend}"


# ============================================================================
# Global Configuration Parameters
# ============================================================================

# Default model list (used when --model not specified)
DEFAULT_MODEL_LIST = [
    "llama3.2-1b-int8",
    "llama3.2-1b-fp8",
]

# Prefill test config
DEFAULT_M_LIST_PREFILL = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_PREFILL = 128  # Prefill repeat count

# Decode test config
DEFAULT_M_LIST_DECODE = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
N_DECODE = 256  # Decode tokens to generate

# Quick test M list
QUICK_M_LIST = [16, 128, 256]

# Prompt length config
PROMPT_LENGTH_CAP_PREFILL = 1024  # Prompt length cap in Prefill mode
PROMPT_LENGTH_FIXED_DECODE = 16   # Fixed prompt length in Decode mode

# max-model-len buffer
MODEL_LEN_BUFFER = 16

# Default sparsity list (cusparselt only)
DEFAULT_SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10", "2_12"]

# Supported backends (order is default test order)
DEFAULT_BACKEND_LIST = ["cutlass", "cublaslt", "cusparselt"]

# Log level (WARNING reduces overhead, change to INFO for debugging)
VLLM_LOG_LEVEL = "WARNING"

# GPU config
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
GPU_MEMORY_UTILIZATION = 0.8

# Global state (for signal handling)
_CURRENT_OUTPUT_DIR: Path | None = None
_GLOBAL_LOG_FILE: Path | None = None


# ============================================================================
# Log Management
# ============================================================================

class TeeLogger:
    """
    Logger that outputs to both console and log file
    
    Usage:
        with TeeLogger(log_file) as logger:
            # All print output will be recorded
            print("test")
    """
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.original_stdout = None
        self.original_stderr = None
        self.file = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file = open(self.log_file, "a", encoding="utf-8")
        sys.stdout = _TeeStream(self.original_stdout, self.file)
        sys.stderr = _TeeStream(self.original_stderr, self.file)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.file:
            self.file.close()
        return False


class _TeeStream:
    """Wrapper that writes to two streams simultaneously"""
    
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    
    def write(self, data):
        self.stream1.write(data)
        # Remove ANSI color codes when writing to file
        self.stream2.write(strip_ansi(data))
    
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()
    
    def isatty(self):
        return self.stream1.isatty()


def create_log_file(result_base: Path, args) -> Path:
    """
    Create log file
    
    Args:
        result_base: Result directory base path
        args: Command line arguments
    
    Returns:
        Log file path
    """
    logs_dir = result_base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp as filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_{timestamp}.log"
    
    # Write header info
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"SlideSparse vLLM Throughput Benchmark Log\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Original command line
        f.write("Original command:\n")
        f.write(f"  {' '.join(sys.argv)}\n\n")
        
        # Parsed arguments
        f.write("Command line arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  --{key.replace('_', '-')}: {value}\n")
        f.write("\n")
        
        # Hardware info
        f.write("Hardware info:\n")
        f.write(f"  GPU: {hw_info.gpu_name}\n")
        f.write(f"  Compute Capability: {hw_info.cc_tag}\n")
        f.write(f"  VRAM: {hw_info.gpu_memory_gb:.1f} GB\n")
        f.write(f"  CUDA: {hw_info.cuda_runtime_version}\n")
        f.write(f"  Python: {hw_info.python_tag}\n")
        f.write("\n")
        
        # Backend env vars (initial state)
        f.write("Backend env vars (initial state):\n")
        f.write(f"  DISABLE_SLIDESPARSE: {os.environ.get('DISABLE_SLIDESPARSE', 'not set')}\n")
        f.write(f"  USE_CUBLASLT: {os.environ.get('USE_CUBLASLT', 'not set')}\n")
        f.write(f"  USE_CUSPARSELT: {os.environ.get('USE_CUSPARSELT', 'not set')}\n")
        f.write(f"  SPARSITY: {os.environ.get('SPARSITY', 'not set')}\n")
        f.write(f"  INNER_DTYPE_32: {os.environ.get('INNER_DTYPE_32', 'not set')}\n")
        f.write("\n")
        f.write("=" * 70 + "\n\n")
    
    return log_file


# ============================================================================
# Signal Handling
# ============================================================================

def _signal_handler(signum, frame):
    """Handle interrupt signal (SIGINT/SIGTERM)"""
    print()
    print("=" * 60)
    print("Test interrupted!")
    if _CURRENT_OUTPUT_DIR is not None:
        print(f"Current result dir: {_CURRENT_OUTPUT_DIR}")
    print("=" * 60)
    sys.exit(130)


def _setup_signal_handlers():
    """Setup signal handlers"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestParams:
    """Test parameters"""
    prompt_length: int
    max_num_seqs: int
    num_prompts: int
    output_len: int
    max_model_len: int
    n_prefill: int
    n_decode: int
    m_prefill: int
    m_decode: int


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    models: List[str]
    backends: List[str]
    sparsities: List[str]
    stages: List[str]
    m_list_prefill: List[int]
    m_list_decode: List[int]
    n_repeat: Optional[int]
    inner_32: bool
    enforce_eager: bool
    dry_run: bool
    gpu_memory_util: float
    gpu_id: str


# ============================================================================
# Environment Variable Management
# ============================================================================

def set_backend_env(
    backend: str,
    sparsity: Optional[str] = None,
    inner_32: bool = False,
    model_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Set environment variables for backend
    
    Args:
        backend: "cutlass" / "cublaslt" / "cusparselt"
        sparsity: Sparsity config (cusparselt only)
        inner_32: Use high-precision accumulation
        model_name: Model name for loading model-specific tuned kernels
    
    Returns:
        Saved original env vars for restoration
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_32": os.environ.get("INNER_DTYPE_32"),
        "SPARSITY": os.environ.get("SPARSITY"),
        "SLIDESPARSE_MODEL_NAME": os.environ.get("SLIDESPARSE_MODEL_NAME"),
        "SLIDESPARSE_MODEL_NAME_WITH_SLIDE": os.environ.get("SLIDESPARSE_MODEL_NAME_WITH_SLIDE"),
    }
    
    # All backends go through SlideSparse (DISABLE_SLIDESPARSE=0)
    # - cutlass:    Don't set USE_CUBLASLT/USE_CUSPARSELT -> SlideSparse CUTLASS fallback
    # - cublaslt:   USE_CUBLASLT=1
    # - cusparselt: USE_CUSPARSELT=1 + SPARSITY
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    if backend == "cutlass":
        # CUTLASS: Clear other backend env vars, use SlideSparse default CUTLASS path
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
        os.environ.pop("SLIDESPARSE_MODEL_NAME", None)  # CUTLASS doesn't need this
        os.environ.pop("SLIDESPARSE_MODEL_NAME_WITH_SLIDE", None)
    elif backend == "cublaslt":
        os.environ["USE_CUBLASLT"] = "1"
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
        # cuBLASLt needs model_name to load tuned kernels
        if model_name:
            # Set full checkpoint name (may have -SlideSparse- suffix)
            os.environ["SLIDESPARSE_MODEL_NAME_WITH_SLIDE"] = model_name
            # Set base model name (remove -SlideSparse-2_L suffix) for kernel lookup
            base_model = extract_model_name(model_name)
            os.environ["SLIDESPARSE_MODEL_NAME"] = base_model
    elif backend == "cusparselt":
        os.environ["USE_CUSPARSELT"] = "1"
        os.environ.pop("USE_CUBLASLT", None)
        if sparsity:
            os.environ["SPARSITY"] = sparsity
        # cuSPARSELt needs model_name to load tuned kernels
        if model_name:
            # Set full checkpoint name (may have -SlideSparse- suffix)
            os.environ["SLIDESPARSE_MODEL_NAME_WITH_SLIDE"] = model_name
            # Set base model name (remove -SlideSparse-2_L suffix) for kernel lookup
            base_model = extract_model_name(model_name)
            os.environ["SLIDESPARSE_MODEL_NAME"] = base_model
    
    if inner_32:
        os.environ["INNER_DTYPE_32"] = "1"
    else:
        os.environ.pop("INNER_DTYPE_32", None)
    
    return saved


def restore_env(saved: Dict[str, Optional[str]]) -> None:
    """Restore environment variables"""
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


# ============================================================================
# Helper Functions
# ============================================================================

def _truncate(s: str, max_len: int = 45) -> str:
    """Truncate string to fit display box"""
    return s[:max_len-3] + "..." if len(s) > max_len else s


def parse_m_list(m_str: Optional[str], stage: str) -> List[int]:
    """
    Parse M value list string
    
    Args:
        m_str: M value string (comma-separated) or "quick" or None
        stage: "prefill" or "decode"
    
    Returns:
        M value list
    """
    if m_str is None:
        return DEFAULT_M_LIST_PREFILL if stage == "prefill" else DEFAULT_M_LIST_DECODE
    
    if m_str.lower() == "quick":
        return QUICK_M_LIST
    
    return [int(x.strip()) for x in m_str.split(",")]


def parse_model_list(model_arg: Optional[str]) -> List[str]:
    """
    Parse model list argument
    
    Args:
        model_arg: --model argument value
    
    Returns:
        Model key list
    """
    if model_arg is None:
        return DEFAULT_MODEL_LIST
    
    model_arg_lower = model_arg.lower()
    
    if model_arg_lower == "all":
        # Return all models
        return list(model_registry.keys())
    
    if model_arg_lower in ("fp8", "int8"):
        # Filter by quantization type
        return [e.key for e in model_registry.list(quant=model_arg_lower)]
    
    # Specific model name
    if model_registry.get(model_arg_lower):
        return [model_arg_lower]
    
    # Try to return as-is
    return [model_arg]


def parse_backend_list(backend_arg: Optional[str]) -> List[str]:
    """Parse backend list argument"""
    if backend_arg is None or backend_arg.lower() == "all":
        return DEFAULT_BACKEND_LIST.copy()
    
    backends = [b.strip().lower() for b in backend_arg.split(",")]
    for b in backends:
        if b not in DEFAULT_BACKEND_LIST:
            print_warning(f"Unknown backend: {b}, will be ignored")
    return [b for b in backends if b in DEFAULT_BACKEND_LIST]


def parse_sparsity_list(sparsity_arg: Optional[str]) -> List[str]:
    """Parse sparsity list argument"""
    if sparsity_arg is None:
        return DEFAULT_SPARSITY_LIST.copy()
    
    return [s.strip() for s in sparsity_arg.split(",")]


def parse_stage_list(stage_arg: Optional[str]) -> List[str]:
    """Parse stage list argument"""
    if stage_arg is None or stage_arg.lower() == "all":
        return ["prefill", "decode"]
    
    return [stage_arg.lower()]


def calculate_test_params(m_value: int, test_mode: str, n_repeat: Optional[int] = None) -> TestParams:
    """
    Calculate all test parameters based on test mode and M value
    
    Args:
        m_value: M value
        test_mode: Test mode (prefill/decode)
        n_repeat: Repeat count (overrides default)
        
    Returns:
        TestParams dataclass instance
    """
    if test_mode == "prefill":
        # Prefill test: M_prefill = max_num_seqs x prompt_length
        n_prefill_val = n_repeat if n_repeat else N_PREFILL
        
        if m_value <= PROMPT_LENGTH_CAP_PREFILL:
            prompt_length = m_value
            max_num_seqs = 1
        else:
            prompt_length = PROMPT_LENGTH_CAP_PREFILL
            # Use ceiling division to ensure M_prefill >= m_value
            max_num_seqs = (m_value + prompt_length - 1) // prompt_length
        
        num_prompts = n_prefill_val * max_num_seqs
        output_len = 1  # Minimize Decode
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
        # Decode test: M_decode = max_num_seqs (batch size)
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


# ============================================================================
# Core Test Functions
# ============================================================================

def run_single_m_test(
    model_key: str,
    m_value: int,
    test_mode: str,
    backend: str,
    result_json_dir: Path,
    log_file: Path,
    checkpoint_path: Path,
    *,
    sparsity: Optional[str] = None,
    n_repeat: Optional[int] = None,
    inner_32: bool = False,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    gpu_id: str = GPU_ID,
    enforce_eager: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Run throughput test for single M value
    
    Returns:
        True on success, False on failure
    """
    # Get model info
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"Model not found: {model_key}")
        return False
    
    # Calculate test params
    params = calculate_test_params(m_value, test_mode, n_repeat)
    
    # Result filename
    result_file = result_json_dir / f"{entry.local_name}_M{m_value}.json"
    
    # - max_num_batched_tokens: 设置为目标 M 值，控制每次迭代处理的最大 token 数
    #   - Prefill: M = max_num_seqs * prompt_length
    #   - Decode:  M = max_num_seqs (batch size)
    if test_mode == "prefill":
        max_num_batched_tokens = params.m_prefill  # = max_num_seqs * prompt_length
    else:
        max_num_batched_tokens = params.m_decode   # = max_num_seqs
    
    # vLLM configuration constraints:
    # 1. max_model_len >= prompt_len + output_len (otherwise can't process request)
    # 2. max_num_batched_tokens >= max_model_len (config validation requirement)
    #
    # Actual M value control logic:
    # - max_num_batched_tokens is just "allowed upper limit", not enforced
    # - Actual M determined by max_num_seqs and request state:
    #   - Prefill: M = max_num_seqs * prompt_len ✅
    #   - Decode:  M = max_num_seqs (1 token per sequence) ✅
    # - --no-enable-chunked-prefill ensures prompt not chunked
    #
    # Therefore, even if max_num_batched_tokens > target_M, actual M still equals target_M
    min_model_len = params.prompt_length + params.output_len
    effective_max_model_len = min_model_len
    
    # If max_num_batched_tokens < min_model_len, need to relax for config validation
    if max_num_batched_tokens < min_model_len:
        max_num_batched_tokens = min_model_len
    
    # Build backend display name
    if backend == "cutlass":
        backend_display = "CUTLASS (SlideSparse fallback)"
    elif backend == "cusparselt" and sparsity:
        backend_display = f"cuSPARSELt ({sparsity.replace('_', ':')})"
    else:
        backend_display = "cuBLASLt"
    
    # cuBLASLt INT8 output fixed as INT32
    if backend == "cublaslt" and entry.quant.lower() == "int8":
        backend_display += " [INT32 output]"
    elif inner_32:
        backend_display += " [inner32]"
    
    # Display test params
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Test Parameters                          │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ Model:    {_truncate(entry.local_name, 48):<48}│")
    print(f"│ Backend:  {_truncate(backend_display, 48):<48}│")
    print(f"│ Stage:    {test_mode:<48}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ GEMM M dimension (precise control):")
    print(f"│   Target M       = {m_value}")
    print(f"│   M_prefill      = {params.m_prefill} (= {params.max_num_seqs} x {params.prompt_length})")
    print(f"│   M_decode       = {params.m_decode}")
    print(f"│   batched_tokens = {max_num_batched_tokens} (key param for M control)")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ vLLM params:")
    print(f"│   --input-len              = {params.prompt_length}")
    print(f"│   --output-len             = {params.output_len}")
    print(f"│   --num-prompts            = {params.num_prompts}")
    print(f"│   --max-num-seqs           = {params.max_num_seqs}")
    print(f"│   --max-model-len          = {effective_max_model_len}")
    print(f"│   --max-num-batched-tokens = {max_num_batched_tokens}")
    print(f"│   --no-enable-chunked-prefill")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ Iteration count:")
    print(f"│   N_prefill = {params.n_prefill}")
    print(f"│   N_decode  = {params.n_decode}")
    if enforce_eager:
        print("├─────────────────────────────────────────────────────────────┤")
        print("│ Compile Mode: --enforce-eager")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    
    # 设置环境变量（从 checkpoint_path 提取 model_name）
    model_name = checkpoint_path.name if checkpoint_path else None
    saved_env = set_backend_env(backend, sparsity, inner_32, model_name)
    
    try:
        # Build environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.update(get_vllm_env_vars(log_level=VLLM_LOG_LEVEL))
        
        # slidesparse not pip installed, need to add project root to PYTHONPATH
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{_PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(_PROJECT_ROOT)
        
        # Build command
        # Key parameters:
        # - --max-num-batched-tokens: Precisely control M value per iteration
        # - --max-num-seqs: Control max sequences per iteration
        # - --no-enable-chunked-prefill: Disable chunked prefill, ensure prompt not chunked
        cmd = [
            "vllm", "bench", "throughput",
            "--model", str(checkpoint_path),
            "--dataset-name", "random",
            "--input-len", str(params.prompt_length),
            "--output-len", str(params.output_len),
            "--num-prompts", str(params.num_prompts),
            "--max-num-seqs", str(params.max_num_seqs),
            "--max-model-len", str(effective_max_model_len),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--no-enable-chunked-prefill",  # Disable chunked prefill for precise M control
            "--gpu-memory-utilization", str(gpu_memory_util),
            "--disable-log-stats",
            "--output-json", str(result_file),
        ]
        
        if enforce_eager:
            cmd.append("--enforce-eager")
        
        # Log to file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"========== M={m_value} ==========\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backend: {backend_display}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Params: prompt_len={params.prompt_length}, output_len={params.output_len}, ")
            f.write(f"num_prompts={params.num_prompts}, max_num_seqs={params.max_num_seqs}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("\n")
        
        # Dry-run mode
        if dry_run:
            print_info("[DRY-RUN] Command to execute:")
            env_str = f"CUDA_VISIBLE_DEVICES={gpu_id} DISABLE_SLIDESPARSE=0"
            if backend == "cutlass":
                pass  # CUTLASS: No extra env vars needed
            elif backend == "cublaslt":
                env_str += " USE_CUBLASLT=1"
            elif backend == "cusparselt":
                env_str += f" USE_CUSPARSELT=1 SPARSITY={sparsity}"
            if inner_32:
                env_str += " INNER_DTYPE_32=1"
            print(f"{env_str} {' '.join(cmd)}")
            print()
            # Generate mock result
            with open(result_file, "w") as f:
                json.dump({
                    "requests_per_second": 0,
                    "tokens_per_second": 0,
                    "elapsed_time": 0,
                    "num_requests": 0,
                }, f)
            return True
        
        # Execute test
        print_info("Starting test...")
        start_time = datetime.now()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print output to console (will be captured by TeeLogger to global log)
        if result.stdout:
            print("\n─── STDOUT ───")
            print(result.stdout)
        if result.stderr:
            print("\n─── STDERR ───")
            print(result.stderr)
        
        # Also log to per-model log
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
            print_success(f"Test completed! Duration: {duration:.1f}s")
            
            # Parse and display results
            with open(result_file, "r") as f:
                data = json.load(f)
            
            req_per_s = data.get("requests_per_second", 0)
            tok_per_s = data.get("tokens_per_second", 0)
            elapsed = data.get("elapsed_time", 0)
            num_req = data.get("num_requests", 0)
            
            print()
            print(f"{Colors.GREEN}Test Results:{Colors.NC}")
            print(f"  Requests/s:   {req_per_s:.2f}")
            print(f"  Tokens/s:     {tok_per_s:.2f}")
            print(f"  Total Reqs:   {num_req}")
            print(f"  Elapsed:      {elapsed:.2f}s")
            
            # Analysis
            if test_mode == "prefill" and params.n_prefill > 0:
                total_prefill_tokens = params.m_prefill * params.n_prefill
                if elapsed > 0:
                    prefill_tps = total_prefill_tokens / elapsed
                    print()
                    print("  [Prefill Analysis]")
                    print(f"  Total Prefill Tokens: {total_prefill_tokens}")
                    print(f"  Prefill Tokens/s:     {prefill_tps:.2f}")
            elif test_mode == "decode" and params.n_decode > 0:
                decode_tokens = params.m_decode * params.n_decode
                if elapsed > 0:
                    decode_tps = decode_tokens / elapsed
                    print()
                    print("  [Decode Analysis]")
                    print(f"  Total Decode Tokens:  {decode_tokens}")
                    print(f"  Decode Tokens/s:      {decode_tps:.2f}")
            
            return True
        else:
            print_error(f"Test failed: M={m_value} (exit code: {result.returncode})")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: Test failed for M={m_value}\n")
            
            return False
            
    except Exception as e:
        print_error(f"Execution exception: {e}")
        return False
    finally:
        restore_env(saved_env)


def generate_model_csv(
    model_name: str,
    m_list: List[int],
    test_mode: str,
    result_json_dir: Path,
    output_dir: Path,
    n_repeat: Optional[int] = None,
):
    """Generate CSV results for single model"""
    csv_file = output_dir / f"{model_name}_{test_mode}.csv"
    
    print()
    print_subheader(f"Generating CSV: {model_name}")
    
    # CSV header
    if test_mode == "prefill":
        header = "M_prefill,prompt_len,max_num_seqs,num_prompts,N_prefill,requests_per_s,tokens_per_s,elapsed_time_s"
    else:
        header = "M_decode,prompt_len,max_num_seqs,num_prompts,N_decode,output_len,requests_per_s,tokens_per_s,elapsed_time_s"
    
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        
        for m_value in m_list:
            result_file = result_json_dir / f"{model_name}_M{m_value}.json"
            params = calculate_test_params(m_value, test_mode, n_repeat)
            
            if result_file.exists():
                try:
                    with open(result_file, "r") as rf:
                        data = json.load(rf)
                    req_s = data.get("requests_per_second", 0)
                    tok_s = data.get("tokens_per_second", 0)
                    elapsed = data.get("elapsed_time", 0)
                except Exception:
                    req_s = tok_s = elapsed = -1
            else:
                req_s = tok_s = elapsed = -1
            
            # 写入 CSV（包括失败的测试）
            if test_mode == "prefill":
                f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                       f"{params.num_prompts},{params.n_prefill},"
                       f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
            else:
                f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                       f"{params.num_prompts},{params.n_decode},{params.output_len},"
                       f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
    
    print_success(f"CSV saved to: {csv_file}")
    
    # Show CSV preview
    print()
    print("Preview:")
    print("-" * 60)
    with open(csv_file, "r") as f:
        print(f.read())
    print("-" * 60)


# ============================================================================
# Advanced Test Functions
# ============================================================================

def run_stage_benchmark(
    model_key: str,
    backend: str,
    stage: str,
    m_list: List[int],
    checkpoint_path: Path,
    *,
    sparsity: Optional[str] = None,
    n_repeat: Optional[int] = None,
    inner_32: bool = False,
    enforce_eager: bool = False,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    gpu_id: str = GPU_ID,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Run all M value tests for single (model, backend, sparsity, stage) combination
    
    Returns:
        (success_count, fail_count)
    """
    global _CURRENT_OUTPUT_DIR
    
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"Model not found: {model_key}")
        return (0, 1)
    
    # Build output directory
    output_dir = build_backend_result_dir("throughput_benchmark", stage, backend, entry.quant.upper(), sparsity)
    _CURRENT_OUTPUT_DIR = output_dir
    
    result_json_dir = output_dir / "json"
    result_json_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "benchmark.log"
    
    # Build title
    if backend == "cutlass":
        title = f"{entry.local_name} | CUTLASS | {stage}"
    elif backend == "cusparselt" and sparsity:
        title = f"{entry.local_name} | cuSPARSELt ({sparsity}) | {stage}"
    else:
        title = f"{entry.local_name} | cuBLASLt | {stage}"
    
    print_header(title)
    print_info(f"Checkpoint: {checkpoint_path}")
    print_info(f"Output: {output_dir}")
    
    success_count = 0
    fail_count = 0
    
    for i, m_value in enumerate(m_list, 1):
        print()
        print("=" * 60)
        print(f"[{i}/{len(m_list)}] Testing M={m_value}")
        print("=" * 60)
        
        success = run_single_m_test(
            model_key, m_value, stage, backend,
            result_json_dir, log_file, checkpoint_path,
            sparsity=sparsity,
            n_repeat=n_repeat,
            inner_32=inner_32,
            gpu_memory_util=gpu_memory_util,
            gpu_id=gpu_id,
            enforce_eager=enforce_eager,
            dry_run=dry_run,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 生成 CSV 结果
    generate_model_csv(
        entry.local_name, m_list, stage,
        result_json_dir, output_dir, n_repeat
    )
    
    print()
    print_info(f"Completed: {success_count} success, {fail_count} failed")
    
    return (success_count, fail_count)


def run_full_benchmark(config: BenchmarkConfig) -> Tuple[int, int]:
    """
    Run full benchmark
    
    Iterate: Model -> Backend -> Sparsity -> Stage -> M
    
    Returns:
        (total_success, total_fail)
    """
    total_success = 0
    total_fail = 0
    
    for model_key in config.models:
        entry = model_registry.get(model_key)
        if entry is None:
            print_warning(f"Model not found, skipping: {model_key}")
            total_fail += 1
            continue
        
        # Check hardware support
        supported, msg = check_quant_support(entry.quant)
        if not supported:
            print_warning(f"Hardware doesn't support {entry.quant.upper()}, skipping: {entry.local_name}")
            print_warning(f"  Reason: {msg}")
            continue
        
        for backend in config.backends:
            # ========== Pre-check: Backend support ==========
            # Before running tests, check if current GPU supports backend + quant combination
            backend_supported, backend_reason = check_backend_support(backend, entry.quant)
            if not backend_supported:
                print_warning(f"Backend not supported, skipping: {entry.local_name} + {backend}")
                print_warning(f"  Reason: {backend_reason}")
                continue
            
            if backend == "cusparselt":
                # cuSPARSELt: iterate all sparsities
                for sparsity in config.sparsities:
                    # Get sparse checkpoint
                    checkpoint_path = get_checkpoint_path(model_key, backend, sparsity)
                    if checkpoint_path is None:
                        print_warning(f"Sparse checkpoint not found, skipping: {entry.local_name} ({sparsity})")
                        continue
                    
                    for stage in config.stages:
                        m_list = config.m_list_prefill if stage == "prefill" else config.m_list_decode
                        
                        success, fail = run_stage_benchmark(
                            model_key, backend, stage, m_list, checkpoint_path,
                            sparsity=sparsity,
                            n_repeat=config.n_repeat,
                            inner_32=config.inner_32,
                            enforce_eager=config.enforce_eager,
                            gpu_memory_util=config.gpu_memory_util,
                            gpu_id=config.gpu_id,
                            dry_run=config.dry_run,
                        )
                        total_success += success
                        total_fail += fail
            else:
                # cutlass / cuBLASLt: use dense checkpoint
                checkpoint_path = get_checkpoint_path(model_key, "cublaslt", None)
                if checkpoint_path is None:
                    print_warning(f"Dense checkpoint not found, skipping: {entry.local_name}")
                    continue
                
                for stage in config.stages:
                    m_list = config.m_list_prefill if stage == "prefill" else config.m_list_decode
                    
                    success, fail = run_stage_benchmark(
                        model_key, backend, stage, m_list, checkpoint_path,
                        sparsity=None,
                        n_repeat=config.n_repeat,
                        inner_32=config.inner_32,
                        enforce_eager=config.enforce_eager,
                        gpu_memory_util=config.gpu_memory_util,
                        gpu_id=config.gpu_id,
                        dry_run=config.dry_run,
                    )
                    total_success += success
                    total_fail += fail
    
    return (total_success, total_fail)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse vLLM Throughput Benchmark (refactored version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test dimension hierarchy:
  Model -> Backend -> Sparsity -> Stage -> M

Backend description:
  cutlass    - SlideSparse CUTLASS fallback (as baseline)
  cublaslt   - SlideSparse cuBLASLt dense GEMM (using original checkpoint)
  cusparselt - SlideSparse cuSPARSELt sparse GEMM (using pre-sparsified checkpoint)

Sparsity description:
  2_4  - 2:4 sparse (50% sparsity)
  2_6  - 2:6 sparse (67% sparsity)
  2_8  - 2:8 sparse (75% sparsity)
  2_10 - 2:10 sparse (80% sparsity)
  2_12 - 2:12 sparse (83% sparsity)

Examples:
  python3 throughput_benchmark.py                     # Use default model list
  python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8  # Test specific model
  python3 throughput_benchmark.py --model fp8         # Test all FP8 models
  python3 throughput_benchmark.py --model all         # Test all models
  python3 throughput_benchmark.py --backend cutlass   # Test CUTLASS only (baseline)
  python3 throughput_benchmark.py --backend cublaslt  # Test cuBLASLt only
  python3 throughput_benchmark.py --backend cusparselt --sparsity 2_8
  python3 throughput_benchmark.py --stage prefill     # Test Prefill only
  python3 throughput_benchmark.py --M quick           # Quick test
  python3 throughput_benchmark.py --dry-run           # Dry-run validation
"""
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "-m", "--model", type=str, metavar="NAME",
        help="Model selection: specific name / fp8 / int8 / all (default: DEFAULT_MODEL_LIST)"
    )
    
    # Backend selection
    backend_group = parser.add_argument_group("Backend Selection")
    backend_group.add_argument(
        "-b", "--backend", type=str, metavar="NAME",
        help="Backend: cutlass / cublaslt / cusparselt / all (default: all)"
    )
    backend_group.add_argument(
        "--sparsity", type=str, metavar="LIST",
        help=f"Sparsity 列表 (仅 cusparselt): 逗号分隔 (默认: {','.join(DEFAULT_SPARSITY_LIST)})"
    )
    
    # Stage selection
    stage_group = parser.add_argument_group("Stage Selection")
    stage_group.add_argument(
        "-s", "--stage", type=str, metavar="NAME",
        help="Stage: prefill / decode / all (default: all)"
    )
    
    # Parameter overrides
    param_group = parser.add_argument_group("Parameter Overrides")
    param_group.add_argument(
        "--M", type=str, metavar="LIST",
        help="M value list: comma-separated / quick (default: stage-specific DEFAULT_M_LIST)"
    )
    param_group.add_argument(
        "--N", type=int, metavar="NUM",
        help="Override repeat count"
    )
    param_group.add_argument(
        "--inner-32", action="store_true",
        help="High-precision accumulation (FP8->FP32, INT8->INT32)"
    )
    
    # Compile options
    compile_group = parser.add_argument_group("Compile Options")
    compile_group.add_argument(
        "--eager", action="store_true",
        help="Force eager mode"
    )
    
    # Hardware options
    hw_group = parser.add_argument_group("Hardware Options")
    hw_group.add_argument(
        "--gpu-id", type=str, default=GPU_ID,
        help=f"GPU ID (default: {GPU_ID})"
    )
    hw_group.add_argument(
        "--gpu-mem", type=float, default=GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization (default: {GPU_MEMORY_UTILIZATION})"
    )
    
    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "--dry-run", action="store_true",
        help="Show commands only, don't execute"
    )
    other_group.add_argument(
        "--list-models", action="store_true",
        help="List all available models"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print_header("Available Model List")
        for entry in model_registry.list():
            local_path = get_model_local_path(entry.key, CHECKPOINT_DIR)
            status = "✓" if local_path.exists() else "✗"
            print(f"  {status} {entry.key:<25} ({entry.local_name})")
        return 0
    
    # Check if vllm is installed
    if not shutil.which("vllm"):
        print_error("vllm not installed or not in PATH")
        return 1
    
    # Parse arguments
    models = parse_model_list(args.model)
    backends = parse_backend_list(args.backend)
    sparsities = parse_sparsity_list(args.sparsity)
    stages = parse_stage_list(args.stage)
    
    # M list (parse separately by stage)
    m_list_prefill = parse_m_list(args.M, "prefill")
    m_list_decode = parse_m_list(args.M, "decode")
    
    # Determine if eager mode is needed
    enforce_eager = args.eager
    if not args.eager:
        if not check_triton_support_and_warn():
            print_warning("Detected GPU architecture that doesn't support torch.compile")
            print_warning("Auto-enabling eager mode")
            enforce_eager = True
    
    # Build config
    config = BenchmarkConfig(
        models=models,
        backends=backends,
        sparsities=sparsities,
        stages=stages,
        m_list_prefill=m_list_prefill,
        m_list_decode=m_list_decode,
        n_repeat=args.N,
        inner_32=args.inner_32,
        enforce_eager=enforce_eager,
        dry_run=args.dry_run,
        gpu_memory_util=args.gpu_mem,
        gpu_id=args.gpu_id,
    )
    
    # Setup signal handlers
    _setup_signal_handlers()
    
    # Display config info
    print_header("SlideSparse vLLM Throughput Benchmark")
    print()
    print_hardware_info()
    print()
    
    print("Test config:")
    print(f"  Models:             {models}")
    print(f"  Backends:         {backends}")
    if "cusparselt" in backends:
        print(f"  Sparsities:       {sparsities}")
    print(f"  Stages:           {stages}")
    print(f"  M_prefill:        {m_list_prefill}")
    print(f"  M_decode:         {m_list_decode}")
    if args.N:
        print(f"  N_repeat:         {args.N}")
    if args.inner_32:
        print(f"  Inner dtype:      FP32/INT32")
    print(f"  GPU mem util:     {args.gpu_mem}")
    if enforce_eager:
        print(f"  Compile mode:     Eager")
    if args.dry_run:
        print(f"  Mode:             DRY-RUN")
    print()
    print("Output directory structure:")
    print("  throughput_benchmark_results/{stage}/{hw_folder}/{backend}/[{sparsity}/]")
    print("=" * 60)
    
    # Create log file
    result_base = _SCRIPT_DIR / "throughput_benchmark_results"
    result_base.mkdir(parents=True, exist_ok=True)
    log_file = create_log_file(result_base, args)
    global _GLOBAL_LOG_FILE
    _GLOBAL_LOG_FILE = log_file
    
    print_info(f"Log file: {log_file}")
    print()
    
    # Use TeeLogger to output to both console and log file
    with TeeLogger(log_file):
        # Execute tests
        total_success, total_fail = run_full_benchmark(config)
        
        # Show results
        print()
        print_header("Benchmark completed!")
        print()
        print(f"Total: {Colors.GREEN}{total_success} success{Colors.NC}, ", end="")
        if total_fail > 0:
            print(f"{Colors.RED}{total_fail} failed{Colors.NC}")
        else:
            print(f"{total_fail} failed")
        print("=" * 60)
    
    print()
    print_info(f"Log saved: {log_file}")
    
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
