"""
Agent Cognitive Attack Framework - Core Modules

This package contains the modular architecture for the Agent Cognitive Attack framework.

Structure:
    src/
    ├── core/      - Configuration and utilities
    ├── mab/       - MAB experiment engine
    ├── fitting/   - Cognitive parameter fitting
    └── analysis/  - Metrics and visualization

Usage:
    from src.mab import JailbreakEnvironment, JailbreakProbeRunner
    from src.fitting import CognitiveFitter
    from src.analysis import MetricsCalculator, DriftPlotter, RadarPlotter

Version: 2.0
"""

__version__ = "2.0"
__author__ = "Agent Cognitive Attack Team"

from src.core.config import GlobalConfig, CONFIG, get_model_config, print_available_platforms
from src.core.settings import get_api_config, get_runtime_config, print_config_summary
from src.core.utils import setup_logger, safe_save_csv, load_csv_with_validation, set_debug_mode, get_log_level

# Import from submodules
from src.mab import JailbreakEnvironment, JailbreakProbeRunner, LLMClient
from src.fitting import CognitiveFitter, batch_fit_cognitive_model

__all__ = [
    "GlobalConfig",
    "CONFIG",
    "get_model_config",
    "print_available_platforms",
    "get_api_config",
    "get_runtime_config",
    "print_config_summary",
    "setup_logger",
    "safe_save_csv",
    "load_csv_with_validation",
    "set_debug_mode",
    "get_log_level",
    # MAB modules
    "JailbreakEnvironment",
    "JailbreakProbeRunner",
    "LLMClient",
    # Fitting modules
    "CognitiveFitter",
    "batch_fit_cognitive_model",
]
