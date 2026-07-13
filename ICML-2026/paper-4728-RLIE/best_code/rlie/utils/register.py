from .logger_config import LoggerConfig

logger_name = "RLIE - Register"


class Register:
    def __init__(self, name: str):
        self.name = name
        self.entries = {}

    def register(self, key: str):
        def decorator(class_builder):
            self.entries[key] = class_builder
            return class_builder

        return decorator

    def build(self, type: str):
        logger = LoggerConfig.get_logger(logger_name)
        if type not in self.entries and "default" not in self.entries:
            if type == "vllm":
                raise ImportError(
                    "vLLM is not imported. Please install the optional vLLM dependencies."
                )
            raise ValueError(
                f"Entry {type} not found in registry {self.name}. Available entries: {', '.join(self.entries.keys())}"
            )
        if type not in self.entries:
            logger.warning(
                f"Entry {type} not found in registry {self.name}. Using default entry."
            )
        return self.entries[type] if type in self.entries else self.entries["default"]
