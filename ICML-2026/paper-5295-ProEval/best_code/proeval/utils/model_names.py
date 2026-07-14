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

"""Mapping from CSV column names to OpenRouter model identifiers.

Usage::

    from proeval.utils.model_names import MODEL_NAME_MAP

    openrouter_id = MODEL_NAME_MAP.get(csv_name, csv_name)
"""

MODEL_NAME_MAP = {
    # OpenAI
    "gpt35_turbo": "openai/gpt-3.5-turbo",
    "gpt4": "openai/gpt-4",
    "gpt_4o": "openai/gpt-4o",
    "gpt4_turbo": "openai/gpt-4-turbo",
    "gpt5": "openai/gpt-5",
    "gpt5_pro": "openai/gpt-5-pro",
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
