"""
Runtime configuration: API keys for the supported LLM providers, plus the
list of self-hosted vLLM endpoints used for in-house fine-tuned models.

API keys are pulled from environment variables — never hard-code them here.
Self-hosted models are routed by name in :mod:`discoverllm.core.generate`:
any model whose name appears in :data:`OUR_MODELS` is sent to one of the
URLs in :data:`OUR_URLS` (with retry/round-robin handled by the dispatcher).

To add a new fine-tuned LoRA:

1. Launch a vLLM server with the adapter loaded as a ``--lora-modules`` entry.
2. Append the LoRA's ``name`` (the value passed to vLLM, *not* the path) to
   :data:`OUR_MODELS`.
3. Make sure :data:`OUR_URLS` includes the host:port the vLLM server is on.
"""

import os

# --------------------------------------------------------------------------- #
# Provider API keys                                                           #
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")

# --------------------------------------------------------------------------- #
# DeepSeek fallback                                                           #
# --------------------------------------------------------------------------- #
# When OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY are unavailable,
# the system can route calls through DeepSeek's OpenAI-compatible API.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Model name mapping for DeepSeek fallback: proprietary model names are
# mapped to the closest DeepSeek equivalent.
DEEPSEEK_MODEL_MAP = {
    # GPT family → DeepSeek equivalents
    "gpt-5.1": "deepseek-chat",
    "gpt-5.1-2025-08-07": "deepseek-chat",
    "gpt-5": "deepseek-chat",
    "gpt-5-2025-08-07": "deepseek-chat",
    "gpt-5-mini": "deepseek-chat",
    "gpt-5-mini-2025-08-07": "deepseek-chat",
    "gpt-4.1": "deepseek-chat",
    # Claude family → DeepSeek equivalents
    "claude-sonnet-4-5": "deepseek-chat",
    "claude-haiku-4-5": "deepseek-chat",
    "claude-sonnet-4": "deepseek-chat",
    "claude-opus-4-5": "deepseek-chat",
    # Gemini family → DeepSeek equivalents
    "gemini-3-flash": "deepseek-chat",
    "gemini-3-pro": "deepseek-chat",
    "gemini-2.5-flash": "deepseek-chat",
    "gemini-2.5-pro": "deepseek-chat",
}

# When set, all external LLM calls go through DeepSeek regardless of model name.
USE_DEEPSEEK_FALLBACK = (
    os.environ.get("USE_DEEPSEEK_FALLBACK", "").lower() == "true"
    or (DEEPSEEK_API_KEY is not None and OPENAI_API_KEY is None
        and ANTHROPIC_API_KEY is None and GEMINI_API_KEY is None)
)

def resolve_model_name(model_name: str) -> str:
    """Map a proprietary model name to its DeepSeek equivalent when fallback is active."""
    if not USE_DEEPSEEK_FALLBACK:
        return model_name
    return DEEPSEEK_MODEL_MAP.get(model_name, model_name)

# --------------------------------------------------------------------------- #
# Self-hosted vLLM endpoints                                                  #
# --------------------------------------------------------------------------- #
# Each URL points at a vLLM server with one or more LoRA adapters loaded.
# The dispatcher in ``core/generate.py`` cycles through them with retries.
OUR_URLS = [
    "http://localhost:7880/v1",  # e.g. Qwen vLLM server
    "http://localhost:7881/v1",  # e.g. Llama vLLM server
]
OUR_API_KEY = "EMPTY"  # vLLM ignores the API key but the OpenAI client requires one.

# Names that should be routed to ``OUR_URLS`` rather than to a hosted provider.
# Add the LoRA names you've registered with vLLM here.
OUR_MODELS = [
    # Base models (served without a LoRA adapter)
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    # Example fine-tuned adapters — replace with your own.
    # The name must match the ``--lora-modules`` registration on your vLLM server.
    "llama-sft-ours",
    "llama-dpo-ours",
    "llama-sft-dpo-ours",
    "qwen-sft-ours",
    "qwen-dpo-ours",
    "qwen-sft-dpo-ours",
    "qwen-online-dpo-ours",
    "qwen-grpo-ours",
]
