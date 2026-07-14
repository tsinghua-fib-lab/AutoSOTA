import importlib
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES = {
    "BREW": "watermark.BREW.BREW",
}

def watermark_name_from_alg_name(name):
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    raise ValueError(f"Invalid algorithm name: {name}")

class AutoWatermark:
    def __init__(self):
        raise EnvironmentError("Use AutoWatermark.load(...)")

    @staticmethod
    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(algorithm_name, transformers_config, algorithm_config_path=algorithm_config, **kwargs)
        return watermark_class(watermark_config)
