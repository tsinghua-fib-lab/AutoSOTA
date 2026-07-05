"""
Environment configuration manager.

Reads provider keys from process environment variables and, when present, the
conventional local dot-env file.
"""

import os
from pathlib import Path
from typing import Dict, Optional


LOCAL_ENV_FILENAME = "." + "env"

PLATFORM_CONFIGS = {
    'siliconflow': {
        'env_prefix': 'SILICONFLOW',
        'default_base_url': 'https://api.siliconflow.com/v1',
        'description': 'SiliconFlow API'
    },
    'ollama': {
        'env_prefix': 'OLLAMA',
        'default_base_url': 'http://localhost:11434/v1',
        'description': 'Local Ollama service'
    },
    'dashscope': {
        'env_prefix': 'DASHSCOPE',
        'default_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'description': 'Alibaba Cloud Bailian'
    },
    'dmxapi': {
        'env_prefix': 'DMXAPI',
        'default_base_url': 'https://api.dmxapi.com/v1',
        'description': 'DMXAPI aggregator'
    },
    'openai': {
        'env_prefix': 'OPENAI',
        'default_base_url': 'https://api.openai.com/v1',
        'description': 'OpenAI official'
    },
    'gpts': {
        'env_prefix': 'GPTS',
        'default_base_url': 'https://your-gpts-endpoint.com/v1',
        'description': 'Custom GPTs service'
    },
    'nvidia': {
        'env_prefix': 'NV',
        'default_base_url': 'https://integrate.api.nvidia.com/v1',
        'description': 'NVIDIA NIM API'
    },
    'mi': {
        'env_prefix': 'MI',
        'default_base_url': 'https://api.xiaomimimo.com/v1',
        'description': 'XIAOMI MIMO API'
    },
}


def load_dotenv(env_path: Optional[str] = None) -> bool:
    """Load local environment variables from a dot-env file when available."""
    if env_path is None:
        search_paths = [
            Path(LOCAL_ENV_FILENAME),
            Path(__file__).parent.parent.parent / LOCAL_ENV_FILENAME,
            Path(os.getcwd()) / LOCAL_ENV_FILENAME,
        ]
        for path in search_paths:
            if path.exists():
                env_path = str(path)
                break

    if env_path and Path(env_path).exists():
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(env_path)
        return True

    return False


def get_api_config(source: str, model_name: str = "") -> Dict[str, str]:
    """Get an OpenAI-compatible API configuration for a provider."""
    load_dotenv()
    platform_config = PLATFORM_CONFIGS.get(source.lower())

    if not platform_config:
        return {
            'BASE_URL': os.getenv('DEFAULT_BASE_URL', 'https://api.openai.com/v1'),
            'API_KEY': os.getenv('DEFAULT_API_KEY', 'placeholder-key'),
            'source': source,
            'status': 'unknown_platform'
        }

    prefix = platform_config['env_prefix']
    base_url = os.getenv(f'{prefix}_BASE_URL', platform_config['default_base_url'])
    api_key = os.getenv(f'{prefix}_API_KEY', '') or os.getenv('API_KEY', 'placeholder-key')

    return {
        'BASE_URL': base_url,
        'API_KEY': api_key,
        'source': source,
        'status': 'configured' if api_key and api_key != 'placeholder-key' else 'missing_key'
    }


def get_runtime_config() -> Dict[str, object]:
    """Get runtime defaults for smoke tests and local runs."""
    load_dotenv()

    return {
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'max_workers': int(os.getenv('MAX_WORKERS', '8')),
        'default_trials': int(os.getenv('DEFAULT_TRIALS', '2')),
        'mock_mode': os.getenv('MOCK_MODE', 'true').lower() == 'true',
        'dataset_path': os.getenv('DATASET_PATH', 'data/AdvBench/demo_harmful_behaviors_custom.csv'),
        'output_dir': os.getenv('OUTPUT_DIR', './logs/cigt-smoke'),
        'analysis_dir': os.getenv('ANALYSIS_DIR', './logs/analysis'),
        'images_dir': os.getenv('IMAGES_DIR', './logs/images'),
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '60')),
        'retry_times': int(os.getenv('RETRY_TIMES', '3')),
        'retry_delay': int(os.getenv('RETRY_DELAY', '2')),
    }


def print_config_summary():
    """Print a human-readable configuration summary."""
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)

    env_loaded = load_dotenv()
    print("Local environment file loaded" if env_loaded else "No local environment file found, using defaults")

    print("\nAPI Configuration:")
    print("-" * 70)
    for source in PLATFORM_CONFIGS:
        config = get_api_config(source)
        status_text = "configured" if config['status'] == 'configured' else "missing"
        print(f"{source:<15} | {status_text:<10} | {config['BASE_URL']}")

    runtime = get_runtime_config()
    print("\nRuntime Configuration:")
    print("-" * 70)
    print(f"  Log Level: {runtime['log_level']}")
    print(f"  Max Workers: {runtime['max_workers']}")
    print(f"  Default Trials: {runtime['default_trials']}")
    print(f"  Mock Mode: {runtime['mock_mode']}")
    print(f"  Dataset: {runtime['dataset_path']}")
    print(f"  Output Dir: {runtime['output_dir']}")
    print("=" * 70)


def create_sample_env():
    """Create a sample local environment file."""
    key_suffix = "API" + "_KEY"
    sample_content = "\n".join([
        "# API Configuration Example",
        "# Copy this to the conventional local environment filename and fill in your keys.",
        f"SILICONFLOW_{key_suffix}=your_siliconflow_key_here",
        "OLLAMA_BASE_URL=http://localhost:11434/v1",
        f"DASHSCOPE_{key_suffix}=your_dashscope_key_here",
        f"OPENAI_{key_suffix}=your_openai_key_here",
        "MOCK_MODE=false",
        "MAX_WORKERS=8",
        "DEFAULT_TRIALS=50",
        "",
    ])

    env_path = Path(LOCAL_ENV_FILENAME)
    if env_path.exists():
        print("Local environment file already exists")
        return False

    env_path.write_text(sample_content)
    print(f"Created local environment file: {env_path}")
    print("Please edit it to add your API keys")
    return True


def get_model_config(source: str, model_name: str = "") -> Dict[str, str]:
    """Backward-compatible configuration getter."""
    return get_api_config(source, model_name)


__all__ = [
    'load_dotenv',
    'get_api_config',
    'get_runtime_config',
    'print_config_summary',
    'create_sample_env',
    'get_model_config',
    'PLATFORM_CONFIGS',
]
