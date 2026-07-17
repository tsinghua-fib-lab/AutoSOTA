# SPDX-License-Identifier: Apache-2.0
# SlideSparse: Sparse Acceleration for LLM Inference
"""
SlideSparse Plugin Module - vLLM Integration
"""

__version__ = "0.1.0"

# Core initialization functions (imported from core)
from .core import (
    init_slidesparse,
    get_algo_config_manager,
)

# Export core utilities from unified utility library
from .utils import (
    # Hardware info
    hw_info,
    HardwareInfo,
    
    # Filename building
    build_filename,
    build_stem,
    build_dir_name,
    
    # File finding
    find_file,
    find_files,
    find_dir,
    
    # Module loading
    load_module,
    
    # Data save/load
    save_json,
    load_json,
    save_csv,
    
    # Directory management
    ensure_result_dir,
    
    # Convenience functions
    get_gpu_name,
    get_gpu_cc,
    get_python_version_tag,
    get_cuda_ver,
    get_arch_tag,
    get_sm_code,
    normalize_dtype,
    print_system_info,
)
