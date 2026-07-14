# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenRouter API client — single source of truth for all API calls.

Provides retry logic, rate-limit handling, JSON-mode support, and
vision (image) inputs.  Used by both the evaluator and generator modules.
"""

import base64
import os
import time
from typing import Any, Dict, Optional

import requests


# Model name mapping

MODEL_MAPPING: Dict[str, str] = {
    # OpenAI
    "gpt35_turbo": "openai/gpt-3.5-turbo",
    "gpt4": "openai/gpt-4",
    "gpt_4o": "openai/gpt-4o",
    "gpt4_turbo": "openai/gpt-4-turbo",
    "gpt5": "openai/gpt-5",
    "gpt5_1": "openai/gpt-5.1",
    "gpt5_2": "openai/gpt-5.2",
    # Anthropic
    "claude35_sonnet": "anthropic/claude-3.5-sonnet",
    "claude35_haiku": "anthropic/claude-3.5-haiku",
    "claude37_sonnet": "anthropic/claude-3.7-sonnet",
    "claude45_sonnet": "anthropic/claude-sonnet-4.5",
    "claude45_opus": "anthropic/claude-opus-4.5",
    # Google Gemini
    "gemini25_flash": "google/gemini-2.5-flash",
    "gemini25_pro": "google/gemini-2.5-pro",
    "gemini3_flash": "google/gemini-3-flash-preview",
    "gemini3_pro": "google/gemini-3-pro-preview",
    # Google Gemma
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    # Qwen
    "qwen3_32b": "qwen/qwen3-32b",
    "qwen3_235b": "qwen/qwen3-235b-a22b",
}


def resolve_model_name(name: str) -> str:
    """Resolve a friendly model name to an OpenRouter identifier.

    If *name* is already a full identifier (contains ``/``), it is returned
    unchanged.
    """
    return MODEL_MAPPING.get(name, name)


# OpenRouterClient


class OpenRouterClient:
    """Client for the OpenRouter API with retry logic and rate-limit handling.

    Args:
        api_key: OpenRouter API key.  Defaults to the ``OPENROUTER_API_KEY``
            environment variable.
        base_url: API base URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY env var or pass api_key."
            )
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


    def predict(
        self,
        prompt: str,
        model: str = "anthropic/claude-3.5-sonnet",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make a single text prediction via the OpenRouter API.

        Supports JSON-mode via *response_format* (``{"type": "json_object"}``
        or ``{"type": "json_schema", ...}``).  Falls back gracefully when
        the upstream model does not support structured output.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )

                # Cascading JSON-mode fallback on 400 or 429
                if resp.status_code in (400, 429) and "response_format" in payload:
                    cur = payload.get("response_format", {})
                    if cur.get("type") == "json_schema":
                        payload["response_format"] = {"type": "json_object"}
                        resp = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=payload,
                            timeout=60,
                        )
                    if resp.status_code in (400, 429):
                        del payload["response_format"]
                        resp = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=payload,
                            timeout=60,
                        )

                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data:
                    raise RuntimeError(f"API error: {data}")
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    raise RuntimeError(
                        f"API returned null content (model={model}). "
                        f"finish_reason={data['choices'][0].get('finish_reason')}"
                    )
                return content

            except Exception as e:
                if self._is_rate_limit(e):
                    wait = retry_delay * (2 ** (attempt + 2))
                    time.sleep(wait)
                    if attempt < max_retries - 1:
                        continue
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay * (2 ** attempt))

        raise RuntimeError("Unexpected error in API request")


    def predict_with_image(
        self,
        prompt: str,
        image_path: str,
        model: str = "google/gemini-flash-1.5",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make a prediction with an image input (vision models)."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data:
                    raise RuntimeError(f"API error: {data}")
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if self._is_rate_limit(e):
                    wait = retry_delay * (2 ** (attempt + 3))
                    time.sleep(wait)
                    if attempt < max_retries - 1:
                        continue
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay * (2 ** attempt))

        raise RuntimeError("Unexpected error in API request")


    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        s = str(exc).lower()
        return "rate" in s or "limit" in s or "429" in s
