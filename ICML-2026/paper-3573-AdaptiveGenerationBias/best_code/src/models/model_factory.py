from typing import TYPE_CHECKING

from src.configs import ModelConfig


if TYPE_CHECKING:
    from .model import BaseModel

from .open_ai import OpenAIGPT

# from .hf_model import HFModel
from .together_model import TogetherModel
from .anthropic_model import AnthropicModel

from .openrouter_model import OpenRouterModel
from .local_replace import LocalReplaceModel


def get_model(config: ModelConfig) -> "BaseModel":
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    if config.provider == "openrouter":
        return OpenRouterModel(config)
    elif config.provider == "hf":
        pass
        # return HFModel(config)
    elif config.provider == "together":
        return TogetherModel(config)
    elif config.provider == "anthropic":
        return AnthropicModel(config)
    elif config.provider == "gcp":
        raise NotImplementedError("GCP models were purposefully disabled as their API is bad.")
        # return GCPModel(config)
    elif config.provider == "local_replace":
        return LocalReplaceModel(config)
    else:
        raise NotImplementedError
