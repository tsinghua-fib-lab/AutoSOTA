import os

AVAILABLE_MODELS = {
    "llava_llada": "LlavaLLaDAModelLM, LlavaLLaDAConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
