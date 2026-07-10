from typing import Optional
from class_registry import ClassRegistry

from AgentTailor.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = "gpt-4o"
        elif model_name == 'mock':
            model = cls.registry.get(model_name)
        elif model_name == 'DeepSeekChat':
            print("loading for deepseek modes")
            model = cls.registry.get(model_name)
        else:  # OpenAI-compatible: model id string (e.g. gpt-4o) -> GPTChat
            model = cls.registry.get('GPTChat', model_name)
        return model
