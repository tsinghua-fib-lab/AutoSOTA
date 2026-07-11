# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, Type

from verbalized_sampling.methods import Method

from .base import BaseLLM
from .embed import get_embedding_model
from .google import GoogleLLM
from .litellm import LiteLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM
from .vllm import VLLMOpenAI

__all__ = ["get_model", "get_embedding_model"]

LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "vllm": VLLMOpenAI,
    "openrouter": OpenRouterLLM,
    "litellm": LiteLLM,
    "openai": OpenAILLM,
    "google": GoogleLLM,
}


def get_model(
    model_name: str,
    method: Method,
    config: dict,
    use_vllm: bool = False,
    num_workers: int = 128,
    strict_json: bool = False,
) -> BaseLLM:
    """Get a model instance."""
    if "claude" in model_name:
        if os.environ.get("ANTHROPIC_API_KEY") is None:
            print("ANTHROPIC_API_KEY is not set, falling back to openrouter")
            model_class = LLM_REGISTRY["openrouter"]
        else:
            model_class = LLM_REGISTRY["litellm"]
    elif "gpt" in model_name or "o3" in model_name or "openai/" in model_name.lower():
        model_class = LLM_REGISTRY["openai"]
    elif "deepseek" in model_name.lower():
        model_class = LLM_REGISTRY["openai"]
    elif "llama" in model_name.lower():
        model_class = LLM_REGISTRY["vllm"]
    # elif ("gemini" in model_name) and (os.environ.get("GEMINI_API_KEY") is not None):
    #     model_class = LLM_REGISTRY["litellm"]
    else:
        model_class = LLM_REGISTRY["vllm" if use_vllm else "openrouter"]

    # print("Model class: ", model_class)
    # print("Model name: ", model_name)
    return model_class(
        model_name=model_name, config=config, num_workers=num_workers, strict_json=strict_json
    )
