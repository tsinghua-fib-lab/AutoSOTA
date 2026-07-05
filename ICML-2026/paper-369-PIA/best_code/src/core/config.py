"""
Global configuration management
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Try to import the release settings module, use a simple fallback if unavailable.
try:
    from .settings import get_api_config, load_dotenv, PLATFORM_CONFIGS
    ENV_CONFIG_AVAILABLE = True
except ImportError:
    ENV_CONFIG_AVAILABLE = False


@dataclass
class GlobalConfig:
    """Global configuration"""

    # Default cognitive parameters
    default_params: Dict[str, float] = field(default_factory=lambda: {
        'alpha_pos': 0.1,
        'alpha_neg': 0.1,
        'rho': 1.0,
        'theta': 0.0,
        'lambda': 0.0,
        'phi': 0.0,
        'beta': 2.0,
        'R_perc': 1.0,
        'lambda_LA': 1.0,
    })


    # Parameter bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'alpha_pos': (0.0, 1.0),
        'alpha_neg': (0.0, 1.0),
        'rho': (0.1, 10.0),
        'theta': (-5.0, 5.0),
        'lambda': (-2.0, 2.0),
        'phi': (0.0, 1.0),
        'beta': (0.5, 10.0),
        'R_perc': (0.1, 100.0),
        'lambda_LA': (0.1, 5.0),
    })


    # Fitting strategy mapping
    strategy_map: Dict[str, List[str]] = field(default_factory=lambda: {
        'BASELINE': ['lambda', 'beta', 'phi', 'theta'],
        'OPTIMISM': ['alpha_pos', 'alpha_neg'],
        'PUNISHMENT': ['alpha_neg'],  # Renamed from OPTIMISM-NEG
        'STIMULUS': ['rho'],
        'MAGNITUDE': ['rho'],  # Renamed from STIMULUS-MATH, also measures rho
        'THREAT': ['theta', 'alpha_neg'],
        'AUTHORITY': ['theta'],
        'SYCOPHANCY': ['theta'],
        'REGRET': ['alpha_pos', 'alpha_neg'],
        'FIT_BASELINE': ['lambda', 'beta', 'theta', 'lambda_LA', 'alpha_pos', 'alpha_neg'],
        'FIT_PERCEPTION': ['R_perc', 'rho', 'theta'],
        'FIT_FEAR': ['alpha_neg', 'lambda_LA', 'theta'],
        'FIT_GREED': ['alpha_pos', 'alpha_neg', 'theta'],
    })

    # Experiment default configuration
    default_trials: int = 2
    default_output_dir: str = "./logs/cigt-smoke"
    default_max_workers: int = 1


# Global configuration instance
CONFIG = GlobalConfig()


def get_model_config(source: str, model_name: str) -> Dict[str, str]:
    """
    Get model API configuration

    Priority: provider environment variables, optional local settings file,
    external config module, then defaults.

    Args:
        source: API source (siliconflow, ollama, etc.)
        model_name: Model name (reserved parameter for compatibility)

    Returns:
        Dictionary containing BASE_URL and API_KEY
    """
    # Use release settings system.
    if ENV_CONFIG_AVAILABLE:
        load_dotenv()

        # Use enhanced configuration retrieval
        config = get_api_config(source, model_name)

        # If configuration status is normal, return directly
        if config.get('status') != 'missing_key':
            return {
                "BASE_URL": config['BASE_URL'],
                "API_KEY": config['API_KEY']
            }

    # Fallback to environment variables.
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    # Try to import from external config.py
    try:
        from config import get_model_config as external_get
        return external_get(source, model_name)
    except ImportError:
        pass

    # Return configuration (may contain placeholders)
    return {
        "BASE_URL": base_url,
        "API_KEY": api_key or "placeholder-key"
    }


def print_available_platforms():
    """
    Print available platform configurations
    """
    if not ENV_CONFIG_AVAILABLE:
        print("Settings module unavailable")
        return

    print("\n" + "=" * 70)
    print("Available Platform Configurations")
    print("=" * 70)

    for source, config in PLATFORM_CONFIGS.items():
        print(f"\n{source.upper()}:")
        print(f"  Environment variable prefix: {config['env_prefix']}")
        print(f"  Default address: {config['default_base_url']}")
        print(f"  Description: {config['description']}")
        print(f"  Example:")
        print(f"    {config['env_prefix']}_<provider key variable>")
        print(f"    {config['env_prefix']}_BASE_URL={config['default_base_url']}")

    print("\n" + "=" * 70)
