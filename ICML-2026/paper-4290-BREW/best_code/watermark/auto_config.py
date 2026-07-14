import importlib
from typing import Any, Optional
from utils.transformers_config import TransformersConfig

CONFIG_MAPPING_NAMES = {
    "BREW": "watermark.BREW.BREWConfig",
}

def config_name_from_alg_name(name: str) -> Optional[str]:
    if name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[name]
    raise ValueError(f"Invalid algorithm name: {name}")

class AutoConfig:
    def __init__(self):
        raise EnvironmentError("Use AutoConfig.load(...)")

    @classmethod
    def load(cls, algorithm_name: str, transformers_config: TransformersConfig, algorithm_config_path=None, **kwargs) -> Any:
        config_name = config_name_from_alg_name(algorithm_name)
        module_name, class_name = config_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        if algorithm_config_path is None:
            algorithm_config_path = f"config/{algorithm_name}.json"
        return config_class(algorithm_config_path, transformers_config, **kwargs)
