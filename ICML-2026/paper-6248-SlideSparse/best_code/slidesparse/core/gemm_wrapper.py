# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse GEMM Wrapper Module

Contains:
- cuBLASLt GEMM ctypes wrapper (FP8 + INT8)
- cuSPARSELt GEMM ctypes wrapper (FP8 + INT8)
- AlgorithmConfigManager: GEMM algorithm config manager (offline tuning lookup)
- torch.library Custom Op registration (for torch.compile tracing)
- Custom Op call wrapper functions

Architecture:
=============
ctypes wrappers directly call CUDA extension libraries (.so files),
but torch.compile cannot trace ctypes calls.

By registering custom ops via torch.library:
- Real impl: calls ctypes wrapper
- Fake impl: returns empty tensor with correct shape for tracing

torch.compile uses fake impl during tracing, real impl during execution.

Algorithm Config Lookup:
========================
AlgorithmConfigManager loads offline-tuned JSON configs at startup,
looks up optimal algorithm config by (model, N, K, M) at runtime,
transparently passes to CUDA Kernel. Falls back to default if not found.
"""

import base64
import bisect
import ctypes
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.library import Library

from vllm.logger import init_logger

from slidesparse.utils import (
    find_file,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
    build_hw_dir_name,
    extract_model_name,
)
from .kernels import _CSRC_DIR

logger = init_logger(__name__)


# ============================================================================
# Search Results Directory
# ============================================================================

_SEARCH_DIR = Path(__file__).parent.parent / "search"


# ============================================================================
# GEMM Library Build Configuration
# ============================================================================

_GEMM_BUILD_CONFIG = {
    "cublaslt":   ("build_cublaslt.py",   ["build"]),
    "cusparselt": ("build_cusparselt.py", ["build"]),
}


def _build_gemm_library(kernel_dir: Path, backend: str) -> None:
    """Auto-build GEMM .so library when not found"""
    import subprocess
    
    if backend not in _GEMM_BUILD_CONFIG:
        raise ValueError(f"Unknown GEMM backend: {backend}")
    
    script_name, extra_args = _GEMM_BUILD_CONFIG[backend]
    script_path = kernel_dir / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Build script not found: {script_path}")
    
    logger.info(f"Auto-building {backend} GEMM library from {script_path}...")
    
    try:
        subprocess.run(
            ["python3", str(script_path)] + extra_args,
            cwd=kernel_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"{backend} GEMM library build completed")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to build {backend} GEMM library:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        ) from e


# ============================================================================
# Extension Cache
# ============================================================================

_gemm_extensions = {}


# ============================================================================
# AlgorithmConfigManager: GEMM Algorithm Config Manager
# ============================================================================

class AlgorithmConfigManager:
    """
    GEMM Algorithm Config Manager
    
    Design points:
    1. Eager init at module import (avoid torch.compile tracing setattr)
    2. Auto-loads all model JSONs from current hw folder in __init__ (KB-level)
    3. Uses bisect binary search for M thresholds, O(log n)
    4. torch.compile compatible: only pure reads at runtime, no side effects
    5. Base64 pre-decoded at load time, avoids runtime decode overhead
    """
    
    def __init__(self):
        # cuBLASLt: {model_name: {(N,K): {"m_thresholds": [...], "alg_by_m": {m_str: (algo_bytes, workspace)}}}}
        # Note: alg_by_m values pre-decoded to (bytes, int) tuple at load time
        self._cublaslt_configs: Dict[str, Dict] = {}
        # cuSPARSELt: {model_name: {(N,K): {"m_thresholds": [...], "alg_by_m": {m_str: (alg_id, split_k, workspace)}}}}
        self._cusparselt_configs: Dict[str, Dict] = {}
        # Model name convention:
        # - _model_name: base model name (strictly without -SlideSparse- suffix), for lookup and kernel loading
        # - _model_name_with_slide: full checkpoint name (may have -SlideSparse- suffix)
        self._model_name: Optional[str] = None
        self._model_name_with_slide: Optional[str] = None
        self._loaded = False
        self._cublaslt_warned_models: set = set()  # Avoid cuBLASLt repeated warnings
        self._cusparselt_warned_models: set = set()  # Avoid cuSPARSELt repeated warnings
        
        # Eager load configs (done at module import, not hot path)
        self.load_all_configs()
    
    def load_all_configs(self) -> None:
        """Load all JSON configs from current hardware directory"""
        if self._loaded:
            return
        
        hw_folder = build_hw_dir_name()  # e.g., RTX5080_cc120_py312_cu129_x86_64
        
        # 优化: 如果设置了 SLIDESPARSE_MODEL_NAME，只加载该模型的配置
        # 注意: SLIDESPARSE_MODEL_NAME 现在是严格不带 slide 后缀的基础名
        target_model_name = os.environ.get("SLIDESPARSE_MODEL_NAME")
        
        if target_model_name:
            logger.info(f"Optimization enabled: Only loading GEMM configs for model '{target_model_name}'")
        
        # cuBLASLt 配置
        cublaslt_dir = _SEARCH_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / hw_folder
        if cublaslt_dir.exists():
            for json_file in cublaslt_dir.glob("alg_search_*.json"):
                self._load_single_config("cublaslt", json_file, target_model_name)
        
        # cuSPARSELt 配置
        cusparselt_dir = _SEARCH_DIR / "cuSPARSELt_AlgSearch" / "alg_search_results" / hw_folder
        if cusparselt_dir.exists():
            for json_file in cusparselt_dir.glob("alg_search_*.json"):
                self._load_single_config("cusparselt", json_file, target_model_name)
        
        self._loaded = True
        logger.info(f"Loaded algorithm configs: cuBLASLt={len(self._cublaslt_configs)}, "
                   f"cuSPARSELt={len(self._cusparselt_configs)} models")
    
    def _load_single_config(self, backend: str, json_path: Path, target_model_name: Optional[str] = None) -> None:
        """Load single JSON config file and pre-decode Base64 data"""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            model_name = data["meta"]["model_name"]
            
            # Filter: skip if target_model_name specified and current doesn't match
            if target_model_name and model_name != target_model_name:
                return

            nk_entries = data.get("nk_entries", {})
            
            # Convert to internal format {(N,K): entry}, pre-decode base64
            parsed = {}
            for nk_str, entry in nk_entries.items():
                # "(3072,2048)" -> (3072, 2048)
                n, k = eval(nk_str)
                
                # Pre-decode all configs in alg_by_m
                decoded_alg_by_m = {}
                for m_key, algo_config in entry.get("alg_by_m", {}).items():
                    if backend == "cublaslt":
                        # cuBLASLt: pre-decode base64 -> bytes
                        algo_data_b64 = algo_config.get("algo_data", "")
                        algo_bytes = base64.b64decode(algo_data_b64) if algo_data_b64 else None
                        workspace = algo_config.get("workspace", 0)
                        decoded_alg_by_m[m_key] = (algo_bytes, workspace)
                    else:
                        # cuSPARSELt: extract integer configs directly
                        alg_id = algo_config.get("alg_id", -1)
                        split_k = algo_config.get("split_k", -1)
                        workspace = algo_config.get("workspace", 0)
                        decoded_alg_by_m[m_key] = (alg_id, split_k, workspace)
                
                parsed[(n, k)] = {
                    "m_thresholds": entry["m_thresholds"],
                    "alg_by_m": decoded_alg_by_m,
                }
            
            if backend == "cublaslt":
                self._cublaslt_configs[model_name] = parsed
            else:
                self._cusparselt_configs[model_name] = parsed
                
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    
    def set_model(self, model_name_with_slide: str) -> None:
        """Set current model (called by init_slidesparse)
        
        Auto-extracts base model name for lookup and kernel loading.
        Also sets env vars so subprocess (e.g., vLLM EngineCore) can access.
        
        Note: Should be called before torch.compile (typically in init_slidesparse).
        
        Env var naming convention:
        - SLIDESPARSE_MODEL_NAME: base model name (strictly without -SlideSparse- suffix)
        - SLIDESPARSE_MODEL_NAME_WITH_SLIDE: full checkpoint name (may have suffix)
        
        Args:
            model_name_with_slide: full model name (may contain -SlideSparse-2_L suffix)
        """
        self._model_name_with_slide = model_name_with_slide
        self._model_name = extract_model_name(model_name_with_slide)
        
        # Set env vars (for subprocess use)
        os.environ["SLIDESPARSE_MODEL_NAME"] = self._model_name
        os.environ["SLIDESPARSE_MODEL_NAME_WITH_SLIDE"] = self._model_name_with_slide
        
        if self._model_name != model_name_with_slide:
            logger.info(f"Model name mapping: {model_name_with_slide} -> {self._model_name}")
    
    def get_model_name_with_slide(self) -> Optional[str]:
        """Get full checkpoint name (may have -SlideSparse- suffix)
        
        Returns instance var first, then reads from env var (supports subprocess).
        
        torch.compile compatible:
        - If _model_name_with_slide set, returns directly (no write)
        - If reading from env var, caches to instance var only in non-compile mode
        """
        if self._model_name_with_slide is not None:
            return self._model_name_with_slide
        # Subprocess scenario: read from env var
        env_model = os.environ.get("SLIDESPARSE_MODEL_NAME_WITH_SLIDE")
        if env_model:
            # Cache only when not in torch.compile tracing (avoid setattr side effect)
            if not torch.compiler.is_compiling():
                self._model_name_with_slide = env_model
                self._model_name = extract_model_name(env_model)
            return env_model
        return None
    
    def get_model_name(self) -> Optional[str]:
        """Get base model name (strictly without -SlideSparse- suffix, for lookup and kernel loading)
        
        Most commonly used interface, returned name directly maps to:
        - model_name in GEMM config JSON
        - Triton kernel filename suffix
        
        Returns instance var first, then reads from env var (supports subprocess).
        
        torch.compile compatible:
        - If _model_name set, returns directly (no write)
        - If reading from env var, caches to instance var only in non-compile mode
        """
        if self._model_name is not None:
            return self._model_name
        # Subprocess scenario: read from env var
        env_name = os.environ.get("SLIDESPARSE_MODEL_NAME")
        if env_name:
            # Cache only when not in torch.compile tracing
            if not torch.compiler.is_compiling():
                self._model_name = env_name
            return env_name
        # If no model_name, try to extract from full name
        full_model = self.get_model_name_with_slide()
        if full_model:
            name = extract_model_name(full_model)
            # Cache only when not in torch.compile tracing
            if not torch.compiler.is_compiling():
                self._model_name = name
            return name
        return None
    
    def lookup_cublaslt(self, N: int, K: int, M: int) -> Tuple[Optional[bytes], int]:
        """
        Look up optimal cuBLASLt config
        
        Strategy: use config for M_upper (smallest threshold >= M)
        
        Reason: cublasLtMatmulAlgo_t's split_k and workspace are M-related.
        Configs with ws>0 can only execute M_exec <= M_from, so must use M_upper config.
        
        Interval partitioning (left-open right-closed):
          (0, M_list[0]]        -> M_list[0] config
          (M_list[0], M_list[1]]-> M_list[1] config
          ...
          > M_list[-1]          -> fallback to default heuristic
        
        Args:
            N: weight N dimension
            K: inner dimension
            M: activation M dimension
        
        Returns:
            (algo_data_bytes, workspace)
            Returns (None, 0) if not found
        """
        model_name = self.get_model_name()
        if model_name is None:
            return (None, 0)
        
        model_config = self._cublaslt_configs.get(model_name)
        if model_config is None:
            # 只在非 torch.compile 追踪时打印警告（避免 set.add 副作用）
            if not torch.compiler.is_compiling():
                if model_name not in self._cublaslt_warned_models:
                    logger.warning(f"No cuBLASLt config for model '{model_name}', "
                                 f"using default algorithm")
                    self._cublaslt_warned_models.add(model_name)
            return (None, 0)
        
        nk_entry = model_config.get((N, K))
        if nk_entry is None:
            return (None, 0)
        
        # M_upper strategy: find M_upper = min{M_i in thresholds | M_i >= M}
        thresholds = nk_entry["m_thresholds"]
        idx = bisect.bisect_left(thresholds, M)
        
        # If M exceeds max threshold, fallback to default heuristic
        if idx >= len(thresholds):
            # Only print warning when not in torch.compile tracing
            if not torch.compiler.is_compiling():
                logger.warning(
                    f"cuBLASLt: M={M} exceeds max searched M={thresholds[-1]} for "
                    f"(N={N}, K={K}), falling back to default heuristic"
                )
            return (None, 0)
        
        m_key = str(thresholds[idx])
        algo_tuple = nk_entry["alg_by_m"].get(m_key)
        if algo_tuple is None:
            return (None, 0)
        
        # Directly return pre-decoded (algo_bytes, workspace)
        return algo_tuple
    
    def lookup_cusparselt(self, N: int, K: int, M: int) -> Tuple[int, int, int]:
        """
        Look up optimal cuSPARSELt config
        
        Uses M_upper strategy: find M_upper = min{M_i | M_i >= M}, use that config.
        This ensures configs with split_k > 1 can execute correctly (sufficient workspace).
        
        Args:
            N: weight N dimension
            K: inner dimension (K' after slide)
            M: activation M dimension
        
        Returns:
            (alg_id, split_k, workspace)
            Returns (-1, -1, 0) if not found
        """
        model_name = self.get_model_name()
        if model_name is None:
            return (-1, -1, 0)
        
        model_config = self._cusparselt_configs.get(model_name)
        if model_config is None:
            # 只在非 torch.compile 追踪时打印警告（避免 set.add 副作用）
            if not torch.compiler.is_compiling():
                if model_name not in self._cusparselt_warned_models:
                    logger.warning(f"No cuSPARSELt config for model '{model_name}', "
                                 f"using default algorithm")
                    self._cusparselt_warned_models.add(model_name)
            return (-1, -1, 0)
        
        nk_entry = model_config.get((N, K))
        if nk_entry is None:
            return (-1, -1, 0)
        
        thresholds = nk_entry["m_thresholds"]
        
        # M_upper strategy: use bisect_left to find first position >= M
        # This ensures sufficient workspace (ws ~ M * N * split_k * dtype_size when split_k > 1)
        idx = bisect.bisect_left(thresholds, M)
        if idx >= len(thresholds):
            # M exceeds all offline-searched M values, use largest config
            # Note: if largest config uses split_k > 1, may fail to execute
            # Only print warning when not in torch.compile tracing
            if not torch.compiler.is_compiling():
                logger.warning(f"cuSPARSELt: M={M} exceeds max offline M={thresholds[-1]}, "
                             f"using largest config (may fail if split_k > 1)")
            idx = len(thresholds) - 1
        
        m_key = str(thresholds[idx])
        algo_tuple = nk_entry["alg_by_m"].get(m_key)
        if algo_tuple is None:
            return (-1, -1, 0)
        
        # Directly return pre-processed (alg_id, split_k, workspace)
        return algo_tuple


# Global singleton (eager init at module load, avoids setattr during torch.compile tracing). Create instance at module top level, not lazy-load on hot path
_algo_config_manager: AlgorithmConfigManager = AlgorithmConfigManager()


# ============================================================================
# WorkspaceManager: Static Workspace Manager
# ============================================================================

class WorkspaceManager:
    """
    Static Workspace Manager
    
    Design points:
    1. Pre-allocate fixed-size workspace, avoid dynamic allocation per GEMM call
    2. Multi-GPU support (independent workspace per device)
    3. Dynamic expansion if algorithm needs larger workspace (rare)
    4. torch.compile/CUDAGraph safe: workspace tensor created on first use,
       subsequent calls only return reference to existing tensor
    
    Default sizes (override via env vars):
    - cuBLASLt:   32 MB (SLIDESPARSE_CUBLASLT_WORKSPACE_MB)
    - cuSPARSELt: 64 MB (SLIDESPARSE_CUSPARSELT_WORKSPACE_MB)
    """
    
    # 默认 workspace 大小
    _CUBLASLT_DEFAULT_SIZE = 32 * 1024 * 1024   # 32 MB
    _CUSPARSELT_DEFAULT_SIZE = 64 * 1024 * 1024  # 64 MB
    
    # {device: torch.Tensor}
    _cublaslt_workspaces: Dict[torch.device, torch.Tensor] = {}
    _cusparselt_workspaces: Dict[torch.device, torch.Tensor] = {}
    
    @classmethod
    def _get_cublaslt_max_size(cls) -> int:
        """Get cuBLASLt workspace max size"""
        env_mb = os.environ.get("SLIDESPARSE_CUBLASLT_WORKSPACE_MB")
        if env_mb:
            try:
                return int(env_mb) * 1024 * 1024
            except ValueError:
                pass
        return cls._CUBLASLT_DEFAULT_SIZE
    
    @classmethod
    def _get_cusparselt_max_size(cls) -> int:
        """Get cuSPARSELt workspace max size"""
        env_mb = os.environ.get("SLIDESPARSE_CUSPARSELT_WORKSPACE_MB")
        if env_mb:
            try:
                return int(env_mb) * 1024 * 1024
            except ValueError:
                pass
        return cls._CUSPARSELT_DEFAULT_SIZE
    
    @classmethod
    def get_cublaslt_workspace(cls, device: torch.device, required_size: int = 0) -> torch.Tensor:
        """
        获取 cuBLASLt workspace
        
        Args:
            device: CUDA 设备
            required_size: 算法需要的最小 workspace 大小
        
        Returns:
            workspace tensor (至少 required_size 字节)
        """
        # Ensure device is correct type
        if not isinstance(device, torch.device):
            device = torch.device(device)
        
        max_size = cls._get_cublaslt_max_size()
        target_size = max(max_size, required_size)
        
        workspace = cls._cublaslt_workspaces.get(device)
        
        # Create new if doesn't exist or too small
        if workspace is None or workspace.numel() < target_size:
            workspace = torch.empty(target_size, dtype=torch.uint8, device=device)
            cls._cublaslt_workspaces[device] = workspace
            logger.debug(f"Created cuBLASLt workspace: {target_size / 1024 / 1024:.1f} MB on {device}")
        
        return workspace
    
    @classmethod
    def get_cusparselt_workspace(cls, device: torch.device, required_size: int = 0) -> torch.Tensor:
        """
        获取 cuSPARSELt workspace
        
        Args:
            device: CUDA 设备
            required_size: 算法需要的最小 workspace 大小
        
        Returns:
            workspace tensor (至少 required_size 字节)
        """
        # Ensure device is correct type
        if not isinstance(device, torch.device):
            device = torch.device(device)
        
        max_size = cls._get_cusparselt_max_size()
        target_size = max(max_size, required_size)
        
        workspace = cls._cusparselt_workspaces.get(device)
        
        # Create new if doesn't exist or too small
        if workspace is None or workspace.numel() < target_size:
            workspace = torch.empty(target_size, dtype=torch.uint8, device=device)
            cls._cusparselt_workspaces[device] = workspace
            logger.debug(f"Created cuSPARSELt workspace: {target_size / 1024 / 1024:.1f} MB on {device}")
        
        return workspace


def get_algo_config_manager() -> AlgorithmConfigManager:
    """
    Get algorithm config manager (eager init)
    
    Note:
    - Instance created at module import, avoids torch.compile tracing setattr
    - Config loading also done at module import
    - Runtime calls just return existing instance
    """
    return _algo_config_manager


# ============================================================================
# cuBLASLt GEMM Wrapper
# ============================================================================

class cuBLASLtGemmWrapper:
    """cuBLASLt GEMM ctypes wrapper (supports FP8 and INT8)"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cublaslt_gemm_get_last_error.argtypes = []
        self._lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # FP8 GEMM 签名: 
        # int fn(W, A, D, workspace_ptr, workspace_size, M, N, K, inner_dtype, algo_data, stream)
        gemm_sig = ([ctypes.c_void_p] * 3 +                # W, A, D
                   [ctypes.c_void_p, ctypes.c_size_t] +   # workspace_ptr, workspace_size
                   [ctypes.c_int64] * 3 +                 # M, N, K
                   [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]) # inner_dtype, algo_data, stream
        
        self._lib.cublaslt_fp8_mm.argtypes = gemm_sig
        self._lib.cublaslt_fp8_mm.restype = ctypes.c_int
        
        # INT8 GEMM 签名（同上）
        self._lib.cublaslt_int8_mm.argtypes = gemm_sig
        self._lib.cublaslt_int8_mm.restype = ctypes.c_int
        
    def cublaslt_fp8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
        algo_data: Optional[bytes] = None,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuBLASLt FP8 GEMM
        
        Computes: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] FP8, weight (row-major, not transposed)
            qinput: [M_pad, K_pad] FP8, quantized activation
            inner_dtype: GEMM output precision ("bf16" or "fp32")
            algo_data: algorithm config data (64 bytes), None for default heuristic
            algo_workspace: workspace size specified in algorithm config
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        # Use static Workspace (avoid allocation per call)
        workspace = WorkspaceManager.get_cublaslt_workspace(qinput.device, algo_workspace)
        workspace_size = workspace.numel()
        
        # Process algo_data
        algo_ptr = None
        if algo_data is not None and len(algo_data) == 64:
            algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
        
        ret = self._lib.cublaslt_fp8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            workspace.data_ptr(), workspace_size,
            M_pad, N, K_pad,
            inner_dtype.encode(),
            ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
            torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"cublaslt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        
        return output
    
    def cublaslt_int8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
        algo_data: Optional[bytes] = None,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuBLASLt INT8 GEMM
        
        Computes: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] INT8, weight (row-major, not transposed)
            qinput: [M_pad, K_pad] INT8, quantized activation
            inner_dtype: ignored, output always INT32
            algo_data: algorithm config data (64 bytes), None for default heuristic
            algo_workspace: workspace size specified in algorithm config
            
        Returns:
            output: [M_pad, N] INT32
            
        Note:
            cuBLASLt INT8 GEMM only supports INT32 output, inner_dtype is ignored.
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        # INT8 GEMM output always INT32
        output = torch.empty((M_pad, N), dtype=torch.int32, device=qinput.device)
        
        # Use static Workspace (avoid allocation per call)
        workspace = WorkspaceManager.get_cublaslt_workspace(qinput.device, algo_workspace)
        workspace_size = workspace.numel()
        
        # Process algo_data
        algo_ptr = None
        if algo_data is not None and len(algo_data) == 64:
            algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
        
        ret = self._lib.cublaslt_int8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            workspace.data_ptr(), workspace_size,
            M_pad, N, K_pad,
            "int32".encode(),
            ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
            torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"cublaslt_int8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


# ============================================================================
# cuSPARSELt GEMM Wrapper
# ============================================================================

class cuSPARSELtGemmWrapper:
    """cuSPARSELt 2:4 Sparse GEMM ctypes wrapper (supports FP8 and INT8)"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cusparselt_gemm_get_last_error.argtypes = []
        self._lib.cusparselt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # FP8 GEMM 签名: int fn(W_compressed, A, D, M, N, K, inner_dtype, alg_id, split_k, workspace_ptr, workspace_size, stream)
        gemm_sig = ([ctypes.c_void_p] * 3 + 
                   [ctypes.c_int64] * 3 + 
                   [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        self._lib.cusparselt_fp8_mm.argtypes = gemm_sig
        self._lib.cusparselt_fp8_mm.restype = ctypes.c_int
        
        # INT8 GEMM 签名（同上）
        self._lib.cusparselt_int8_mm.argtypes = gemm_sig
        self._lib.cusparselt_int8_mm.restype = ctypes.c_int
        
    def cusparselt_fp8_mm(
        self,
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
        alg_id: int = -1,
        split_k: int = -1,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuSPARSELt 2:4 Sparse FP8 GEMM
        
        Computes: output[M_pad, N] = qinput[M_pad, K_slide_pad] @ weight_decompressed.T
        
        Args:
            weight_compressed: [compressed_size] uint8 1D, compressed weight
            qinput: [M_pad, K_slide_pad] FP8, quantized+slide activation
            N: weight N dimension
            K_slide: weight K_slide dimension (after slide expansion)
            inner_dtype: GEMM output precision ("bf16" or "fp32")
            alg_id: algorithm ID, -1 for default
            split_k: split_k setting, -1 for not set
            algo_workspace: workspace size specified in algorithm config
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad = qinput.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        # Use static Workspace (avoid allocation per call)
        workspace = WorkspaceManager.get_cusparselt_workspace(qinput.device, algo_workspace)

        ret = self._lib.cusparselt_fp8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(),
            alg_id, split_k, 
            workspace.data_ptr(), workspace.numel(),
            torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cusparselt_gemm_get_last_error()
            raise RuntimeError(f"cusparselt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        return output
    
    def cusparselt_int8_mm(
        self,
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
        alg_id: int = -1,
        split_k: int = -1,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuSPARSELt 2:4 Sparse INT8 GEMM
        
        Computes: output[M_pad, N] = qinput[M_pad, K_slide_pad] @ weight_decompressed.T
        
        Args:
            weight_compressed: [compressed_size] uint8 1D, compressed weight
            qinput: [M_pad, K_slide_pad] INT8, quantized+slide activation
            N: weight N dimension
            K_slide: weight K_slide dimension (after slide expansion)
            inner_dtype: GEMM output precision ("bf16" or "int32", "fp32" not supported)
            alg_id: algorithm ID, -1 for default
            split_k: split_k setting, -1 for not set
            algo_workspace: workspace size specified in algorithm config
            
        Returns:
            output: [M_pad, N] BF16/INT32
            
        Note:
            cuSPARSELt INT8 GEMM does not support FP32 output, only BF16 or INT32.
        """
        M_pad = qinput.shape[0]
        
        # INT8 does not support FP32 output
        if inner_dtype == "fp32":
            raise ValueError(
                "cuSPARSELt INT8 GEMM does not support FP32 output. "
                "Use 'bf16' (default) or 'int32'."
            )
        
        out_dtype = torch.int32 if inner_dtype == "int32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        # Use static Workspace (avoid allocation per call)
        workspace = WorkspaceManager.get_cusparselt_workspace(qinput.device, algo_workspace)
        
        ret = self._lib.cusparselt_int8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(),
            alg_id, split_k, 
            workspace.data_ptr(), workspace.numel(),
            torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cusparselt_gemm_get_last_error()
            raise RuntimeError(f"cusparselt_int8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


# ============================================================================
# Extension Loading
# ============================================================================

def _get_gemm_extension(backend: str):
    """
    Get GEMM extension (lazy load)
    
    Loads ctypes-wrapped CUDA extension (pure C lib via ctypes.CDLL)
    
    Note: SlideSparseFp8LinearOp.__init__ preloads extension,
    so torch.compile tracing directly hits cache.
    """
    if backend in _gemm_extensions:
        return _gemm_extensions[backend]
    
    if backend == "cublaslt":
        # Preload system cuBLASLt library (RTLD_GLOBAL mode)
        ensure_cublaslt_loaded()
        # Directory name is cublaslt_gemm
        kernel_dir = _CSRC_DIR / "cublaslt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cublaslt_gemm"
        wrapper_class = cuBLASLtGemmWrapper
    elif backend == "cusparselt":
        # Preload system cuSPARSELt library (0.8.1+, RTLD_GLOBAL mode)
        ensure_cusparselt_loaded()
        # Directory name is cusparselt_gemm
        kernel_dir = _CSRC_DIR / "cusparselt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cusparselt_gemm"
        wrapper_class = cuSPARSELtGemmWrapper
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Find .so file
    so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
    
    if so_path is None:
        # Not found, try auto build
        _build_gemm_library(kernel_dir, backend=backend)
        so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
        if so_path is None:
            raise FileNotFoundError(
                f"{backend} GEMM extension not found after build.\n"
                f"Build may have failed. Please check the logs."
            )
    
    # Create ctypes wrapper (pass .so path)
    wrapper = wrapper_class(so_path)
    _gemm_extensions[backend] = wrapper
    logger.info_once(f"{backend} GEMM extension loaded: {so_path.name}")
    return wrapper


# ============================================================================
# torch.library Custom Ops
# ============================================================================

# Create SlideSparse library
_slidesparse_lib = Library("slidesparse", "FRAGMENT")


# ----------------------------------------------------------------------------
# FP8 Custom Ops
# ----------------------------------------------------------------------------

def _register_cublaslt_fp8_custom_op():
    """Register cuBLASLt FP8 GEMM custom op"""
    
    # Define op schema
    _slidesparse_lib.define(
        "cublaslt_fp8_mm(Tensor weight, Tensor qinput, str inner_dtype) -> Tensor"
    )
    
    # Real implementation
    def _cublaslt_fp8_mm_impl(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt FP8 GEMM actual execution (with algorithm config lookup)"""
        # Compile-time guard: if called during torch.compile tracing, init is wrong
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "cublaslt_fp8_mm CUDA implementation called during torch.compile tracing!\n"
                "This indicates an initialization error. Ensure that:\n"
                "  1. init_slidesparse(model_name) is called before model loading\n"
                "  2. SlideSparseFp8LinearOp is instantiated before torch.compile"
            )
        
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded.\n"
                "This can happen if:\n"
                "  1. _get_gemm_extension('cublaslt') was not called during init\n"
                "  2. The .so file failed to load\n"
                "Ensure SlideSparseFp8LinearOp is instantiated before calling this op."
            )
        
        # Lookup optimal algorithm config
        M, K = qinput.shape
        N = weight.shape[0]
        mgr = get_algo_config_manager()
        algo_data, algo_workspace = mgr.lookup_cublaslt(N, K, M)
        
        return ext.cublaslt_fp8_mm(weight, qinput, inner_dtype, algo_data, algo_workspace)
    
    # Fake implementation (for torch.compile tracing)
    def _cublaslt_fp8_mm_fake(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt FP8 GEMM fake impl - returns empty tensor with correct shape"""
        M_pad = qinput.shape[0]
        N = weight.shape[0]
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # Register implementation
    _slidesparse_lib.impl("cublaslt_fp8_mm", _cublaslt_fp8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cublaslt_fp8_mm", _cublaslt_fp8_mm_fake)
    
    logger.info_once("cuBLASLt FP8 custom op registered: slidesparse::cublaslt_fp8_mm")


def _register_cusparselt_fp8_custom_op():
    """Register cuSPARSELt FP8 GEMM custom op"""
    
    # Define op schema
    _slidesparse_lib.define(
        "cusparselt_fp8_mm(Tensor weight_compressed, Tensor qinput, "
        "int N, int K_slide, str inner_dtype) -> Tensor"
    )
    
    # Real implementation
    def _cusparselt_fp8_mm_impl(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt FP8 GEMM actual execution (with algorithm config lookup)"""
        # Compile-time guard
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "cusparselt_fp8_mm CUDA implementation called during torch.compile tracing!\n"
                "This indicates an initialization error. Ensure that:\n"
                "  1. init_slidesparse(model_name) is called before model loading\n"
                "  2. SlideSparseFp8LinearOp is instantiated before torch.compile"
            )
        
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded.\n"
                "This can happen if:\n"
                "  1. _get_gemm_extension('cusparselt') was not called during init\n"
                "  2. The .so file failed to load\n"
                "Ensure SlideSparseFp8LinearOp is instantiated before calling this op."
            )
        
        # Lookup optimal algorithm config
        M = qinput.shape[0]
        mgr = get_algo_config_manager()
        alg_id, split_k, algo_workspace = mgr.lookup_cusparselt(N, K_slide, M)
        
        return ext.cusparselt_fp8_mm(weight_compressed, qinput, N, K_slide, inner_dtype,
                                     alg_id, split_k, algo_workspace)
    
    # Fake implementation (for torch.compile tracing)
    def _cusparselt_fp8_mm_fake(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt FP8 GEMM fake impl - returns empty tensor with correct shape"""
        M_pad = qinput.shape[0]
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # Register implementation
    _slidesparse_lib.impl("cusparselt_fp8_mm", _cusparselt_fp8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cusparselt_fp8_mm", _cusparselt_fp8_mm_fake)
    
    logger.info_once("cuSPARSELt FP8 custom op registered: slidesparse::cusparselt_fp8_mm")


# ----------------------------------------------------------------------------
# INT8 Custom Ops
# ----------------------------------------------------------------------------

def _register_cublaslt_int8_custom_op():
    """Register cuBLASLt INT8 GEMM custom op"""
    
    # Define op schema
    _slidesparse_lib.define(
        "cublaslt_int8_mm(Tensor weight, Tensor qinput, str inner_dtype) -> Tensor"
    )
    
    # Real implementation
    def _cublaslt_int8_mm_impl(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt INT8 GEMM actual execution (output fixed INT32, with algorithm config lookup)"""
        # Compile-time guard
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "cublaslt_int8_mm CUDA implementation called during torch.compile tracing!\n"
                "This indicates an initialization error. Ensure that:\n"
                "  1. init_slidesparse(model_name) is called before model loading\n"
                "  2. SlideSparseInt8LinearOp is instantiated before torch.compile"
            )
        
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded.\n"
                "This can happen if:\n"
                "  1. _get_gemm_extension('cublaslt') was not called during init\n"
                "  2. The .so file failed to load\n"
                "Ensure SlideSparseInt8LinearOp is instantiated before calling this op."
            )
        
        # Lookup optimal algorithm config
        M, K = qinput.shape
        N = weight.shape[0]
        mgr = get_algo_config_manager()
        algo_data, algo_workspace = mgr.lookup_cublaslt(N, K, M)
        
        return ext.cublaslt_int8_mm(weight, qinput, inner_dtype, algo_data, algo_workspace)
    
    # Fake implementation (for torch.compile tracing)
    def _cublaslt_int8_mm_fake(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt INT8 GEMM fake impl - returns INT32 tensor with correct shape"""
        M_pad = qinput.shape[0]
        N = weight.shape[0]
        # INT8 GEMM output fixed INT32
        return torch.empty((M_pad, N), dtype=torch.int32, device=qinput.device)
    
    # Register implementation
    _slidesparse_lib.impl("cublaslt_int8_mm", _cublaslt_int8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cublaslt_int8_mm", _cublaslt_int8_mm_fake)
    
    logger.info_once("cuBLASLt INT8 custom op registered: slidesparse::cublaslt_int8_mm")


def _register_cusparselt_int8_custom_op():
    """Register cuSPARSELt INT8 GEMM custom op"""
    
    # Define op schema
    _slidesparse_lib.define(
        "cusparselt_int8_mm(Tensor weight_compressed, Tensor qinput, "
        "int N, int K_slide, str inner_dtype) -> Tensor"
    )
    
    # Real implementation
    def _cusparselt_int8_mm_impl(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt INT8 GEMM actual execution (with algorithm config lookup)"""
        # Compile-time guard
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "cusparselt_int8_mm CUDA implementation called during torch.compile tracing!\n"
                "This indicates an initialization error. Ensure that:\n"
                "  1. init_slidesparse(model_name) is called before model loading\n"
                "  2. SlideSparseInt8LinearOp is instantiated before torch.compile"
            )
        
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded.\n"
                "This can happen if:\n"
                "  1. _get_gemm_extension('cusparselt') was not called during init\n"
                "  2. The .so file failed to load\n"
                "Ensure SlideSparseInt8LinearOp is instantiated before calling this op."
            )
        
        # Lookup optimal algorithm config
        M = qinput.shape[0]
        mgr = get_algo_config_manager()
        alg_id, split_k, algo_workspace = mgr.lookup_cusparselt(N, K_slide, M)
        
        return ext.cusparselt_int8_mm(weight_compressed, qinput, N, K_slide, inner_dtype,
                                      alg_id, split_k, algo_workspace)
    
    # Fake implementation (for torch.compile tracing)
    def _cusparselt_int8_mm_fake(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt INT8 GEMM fake impl - returns empty tensor with correct shape"""
        M_pad = qinput.shape[0]
        # INT8 does not support FP32, only BF16 or INT32
        out_dtype = torch.int32 if inner_dtype == "int32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # Register implementation
    _slidesparse_lib.impl("cusparselt_int8_mm", _cusparselt_int8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cusparselt_int8_mm", _cusparselt_int8_mm_fake)
    
    logger.info_once("cuSPARSELt INT8 custom op registered: slidesparse::cusparselt_int8_mm")


# Register custom ops at module load
# Note: only registers op schema and impl, does not load ctypes library
# ctypes library loaded in LinearOp.__init__
_register_cublaslt_fp8_custom_op()
_register_cusparselt_fp8_custom_op()
_register_cublaslt_int8_custom_op()
_register_cusparselt_int8_custom_op()


# ============================================================================
# Custom Op Call Wrapper Functions
# ============================================================================

def cublaslt_fp8_mm_op(
    weight: torch.Tensor,
    qinput: torch.Tensor,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuBLASLt FP8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op，而非直接调用 ctypes。
    """
    return torch.ops.slidesparse.cublaslt_fp8_mm(weight, qinput, inner_dtype)


def cusparselt_fp8_mm_op(
    weight_compressed: torch.Tensor,
    qinput: torch.Tensor,
    N: int,
    K_slide: int,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuSPARSELt FP8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op，而非直接调用 ctypes。
    """
    return torch.ops.slidesparse.cusparselt_fp8_mm(
        weight_compressed, qinput, N, K_slide, inner_dtype
    )


def cublaslt_int8_mm_op(
    weight: torch.Tensor,
    qinput: torch.Tensor,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuBLASLt INT8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op。
    注意：输出固定为 INT32，inner_dtype 参数被忽略。
    """
    return torch.ops.slidesparse.cublaslt_int8_mm(weight, qinput, inner_dtype)


def cusparselt_int8_mm_op(
    weight_compressed: torch.Tensor,
    qinput: torch.Tensor,
    N: int,
    K_slide: int,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuSPARSELt INT8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op。
    注意：不支持 FP32 输出，只能使用 "bf16" 或 "int32"。
    """
    return torch.ops.slidesparse.cusparselt_int8_mm(
        weight_compressed, qinput, N, K_slide, inner_dtype
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Algorithm config manager
    "AlgorithmConfigManager",
    "get_algo_config_manager",
    
    # Wrapper classes
    "cuBLASLtGemmWrapper",
    "cuSPARSELtGemmWrapper",
    
    # Extension loading
    "_get_gemm_extension",
    
    # FP8 Custom Op wrapper functions
    "cublaslt_fp8_mm_op",
    "cusparselt_fp8_mm_op",
    
    # INT8 Custom Op wrapper functions
    "cublaslt_int8_mm_op",
    "cusparselt_int8_mm_op",
]
