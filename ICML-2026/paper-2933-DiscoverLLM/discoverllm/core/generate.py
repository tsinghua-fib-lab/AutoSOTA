import inspect
import json
import os
import random
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import openai
import requests
from anthropic import Anthropic, AnthropicBedrock
from google import genai
from google.genai import types as genai_types

from discoverllm.config import (
    ANTHROPIC_API_KEY,
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    OUR_API_KEY,
    OUR_MODELS,
    OUR_URLS,
    TOGETHER_API_KEY,
    USE_DEEPSEEK_FALLBACK,
    resolve_model_name,
)
from discoverllm.utils import process_json, process_yaml


# --------------------------------------------------------------------------- #
# Lazy provider clients                                                       #
# --------------------------------------------------------------------------- #
# We construct clients on first use rather than at module import. This means
# importing :mod:`discoverllm.core.generate` no longer requires every API key
# to be set — useful for tests, type checkers, and any code path that only
# uses one provider.
@lru_cache(maxsize=1)
def _openai_client():
    return openai.Client(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=1)
def _together_client():
    return openai.OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")


@lru_cache(maxsize=1)
def _anthropic_client():
    return AnthropicBedrock(
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region="us-west-2",
    )


@lru_cache(maxsize=1)
def _anthropic_counting_client():
    return Anthropic(api_key=ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def _gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


@lru_cache(maxsize=1)
def _deepseek_client():
    """OpenAI-compatible client pointed at DeepSeek's API."""
    return openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def generate_chat_deepseek(
    model_name: str,
    system_prompt: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192
) -> str:
    """Generate text via DeepSeek's OpenAI-compatible chat completions API.

    Uses the standard ``/v1/chat/completions`` endpoint. Maps proprietary
    model names to DeepSeek equivalents via :func:`resolve_model_name`.

    DeepSeek V4 Pro is a reasoning model — when reasoning is produced, the
    actual answer text appears in ``content``. If ``content`` is empty but
    ``reasoning_content`` is present (e.g. ``max_tokens`` exhausted during
    reasoning), we raise a clear error so callers can retry with a larger
    budget.
    """
    resolved = resolve_model_name(model_name)

    # For DeepSeek reasoning models, disable the reasoning_effort / thinking
    # so the model directly produces content rather than internal reasoning.
    # This is more predictable for judge-style prompts and avoids empty
    # ``content`` due to reasoning-token exhaustion.
    extra_body = None
    if resolved and "deepseek" in resolved.lower():
        # DeepSeek parameter to suppress reasoning / chain-of-thought.
        # The model still produces the answer — it just skips the CoT.
        extra_body = {"think": False}

    msgs: List[Dict[str, str]] = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)

    output = _deepseek_client().chat.completions.create(
        model=resolved,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    content = output.choices[0].message.content or ""

    # Fallback: if content is empty but reasoning_content exists, use it.
    if not content and hasattr(output.choices[0].message, "reasoning_content"):
        reasoning = output.choices[0].message.reasoning_content
        if reasoning:
            content = reasoning

    return content


def _append_usage_jsonl(filename: str, **fields: Any) -> None:
    """Append one provider-usage record to ``./<filename>`` (creates if missing).

    Always sets a ``timestamp`` field; callers pass everything else as kwargs.
    """
    fields.setdefault("timestamp", datetime.now().isoformat())
    mode = "a" if os.path.exists(filename) else "w"
    with open(filename, mode) as f:
        f.write(json.dumps(fields) + "\n")


def save_anthropic_usage(
    model_name: str,
    system: str | None,
    input_messages: List[Dict[str, str]],
    output_message: str,
    elapsed_time: float,
    caller_function_name: str,
) -> None:
    """
    Record one Anthropic API call's token usage to ``anthropic_usage.jsonl``.

    Anthropic doesn't return token counts in the response — we make a separate
    ``count_tokens`` call (cheap but not free) for the input + output.
    """
    count_kwargs = {"model": model_name, "messages": input_messages}
    if system is not None:
        count_kwargs["system"] = system
    input_tokens = _anthropic_counting_client().messages.count_tokens(**count_kwargs).input_tokens
    output_tokens = _anthropic_counting_client().messages.count_tokens(
        model=model_name,
        messages=[{"role": "assistant", "content": output_message}],
    ).input_tokens

    _append_usage_jsonl(
        "anthropic_usage.jsonl",
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_time=elapsed_time,
        caller_function_name=caller_function_name,
    )


def save_gemini_usage(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_time: float,
    caller_function_name: str,
) -> None:
    """
    Record one Gemini API call's token usage to ``gemini_usage.jsonl``.

    Gemini returns token counts on the response object, so the caller passes
    them in directly.
    """
    _append_usage_jsonl(
        "gemini_usage.jsonl",
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_time=elapsed_time,
        caller_function_name=caller_function_name,
    )


def generate(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    verbose: bool = False
) -> str:
    """
    Generate text using various LLM providers based on model name.

    Args:
        model_name: Name of the model to use
        system_prompt: Optional system prompt
        user_prompt: User prompt/input text
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 1000)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Generated text response

    Raises:
        NotImplementedError: If model provider is not implemented
    """
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    return generate_chat(
        model_name=model_name,
        system_prompt=system_prompt,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose
    )


def generate_chat_anthropic(
    model_name: str,
    system_prompt: Optional[str],
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192,
    caller_function_name: str = ''
) -> str:
    """
    Generate text using Anthropic models.

    Args:
        model_name: Name of the model to use
        system_prompt: Optional system prompt
        messages: List of messages
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 1000)
        max_retries: Maximum number of retry attempts (default: 3)
        verbose: Whether to print verbose output
    """
    model_dictionary = {
        "claude-sonnet-4-5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-haiku-4-5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0"
    }

    # Cleaner approach: assemble kwargs, add 'thinking' only if needed
    kwargs = {
        "model": model_dictionary[model_name],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if system_prompt is not None:
        kwargs["system"] = system_prompt
    if "4-5" in model_name:
        kwargs["thinking"] = {"type": "disabled"}

    start_time = time.time()
    output = _anthropic_client().messages.create(**kwargs)
    elapsed_time = time.time() - start_time

    output_text = ""
    thinking_text = ""

    for content in output.content:
        # check if has thinking attribute
        if hasattr(content, 'thinking'):
            thinking_text += " " + content.thinking
        elif hasattr(content, 'text'):
            output_text += " " + content.text

    thinking_text = thinking_text.strip()
    output_text = output_text.strip()

    if thinking_text != "" or output_text != "":
        save_anthropic_usage(
            model_name,
            system_prompt,
            messages,
            thinking_text + " " + output_text,
            elapsed_time,
            caller_function_name
        )
    return output_text


def generate_chat_openai(
    model_name: str,
    system_prompt: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192
) -> str:
    """
    Generate text using OpenAI models.

    Args:
        model_name: Name of the model to use
        system_prompt: Optional system prompt
        messages: List of messages
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 1000)
        max_retries: Maximum number of retry attempts (default: 3)
        verbose: Whether to print verbose output
    """
    if system_prompt is not None:
        messages.insert(0, {"role": "developer", "content": system_prompt})

    if model_name.startswith("gpt-5.1") or model_name.startswith("gpt-5.2"):
        output = _openai_client().responses.create(
            model=model_name,
            input=messages,
            reasoning={
                "effort": "none"
            }
        )
        output_text = output.output_text
    elif model_name.startswith("gpt-5.4") or model_name.startswith("gpt-5.3"):
        output = _openai_client().responses.create(
            model=model_name,
            input=messages,
            reasoning={
                "effort": "low"
            }
        )
        output_text = output.output_text
    elif model_name.startswith("gpt-5"):
        output = _openai_client().responses.create(
            model=model_name,
            input=messages,
            reasoning={
                "effort": "minimal"
            }
        )
        output_text = output.output_text
    else:
        output = _openai_client().chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        output_text = output.choices[0].message.content
    return output_text


def generate_chat_other(
    model_name: str,
    system_prompt: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192
) -> str:
    """
    Generate text using Together, Our Models (vLLM), or local HuggingFace.

    Falls back to HuggingFace transformers when vLLM is unreachable and the
    model is in ``OUR_MODELS``.
    """
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})

    if model_name in OUR_MODELS:
        # Try vLLM first
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OUR_API_KEY}"
        }

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        output_text = None
        for our_url in OUR_URLS:
            try:
                response = requests.post(
                    f"{our_url}/chat/completions",
                    headers=headers, json=payload,
                    timeout=5,
                )
                response.raise_for_status()
                result = response.json()
                output_text = result["choices"][0]["message"]["content"]
                break
            except requests.exceptions.RequestException:
                continue

        if output_text is None:
            # vLLM is not available — fall back to local HuggingFace inference.
            output_text = _generate_chat_hf(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    else:
        output = _together_client().chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        output_text = output.choices[0].message.content
    return output_text


# --------------------------------------------------------------------------- #
# HuggingFace local-inference fallback                                        #
# --------------------------------------------------------------------------- #
# Map canonical HF model names to local filesystem paths so the fallback can
# load models that have already been downloaded to /models.
_HF_MODEL_PATH_MAP: Dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "/models/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B": "/models/Qwen3-8B",
}

_HF_PIPELINE_CACHE: Dict[str, Any] = {}
_HF_PIPELINE_LOCK = None

def _get_hf_pipeline_lock():
    global _HF_PIPELINE_LOCK
    if _HF_PIPELINE_LOCK is None:
        import threading as _thr
        _HF_PIPELINE_LOCK = _thr.Lock()
    return _HF_PIPELINE_LOCK

def _resolve_hf_path(model_name: str) -> str:
    """Return the local filesystem path for *model_name*, or the name unchanged."""
    import os as _os
    candidate = _HF_MODEL_PATH_MAP.get(model_name, model_name)
    if _os.path.isdir(candidate) and _os.path.isfile(_os.path.join(candidate, "config.json")):
        return candidate
    return model_name


def _get_hf_pipeline(model_name: str):
    """Lazily load a HuggingFace text-generation pipeline for *model_name*."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if model_name not in _HF_PIPELINE_CACHE:
        with _get_hf_pipeline_lock():
            if model_name in _HF_PIPELINE_CACHE:
                return _HF_PIPELINE_CACHE[model_name]
            model_path = _resolve_hf_path(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            local_files_only=True,
        )
        _HF_PIPELINE_CACHE[model_name] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    return _HF_PIPELINE_CACHE[model_name]


def _generate_chat_hf(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> str:
    """Generate text using a local HuggingFace model."""
    pipe = _get_hf_pipeline(model_name)

    # Use the tokenizer's chat template if available, else manual format
    tokenizer = pipe.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

    do_sample = temperature > 0.0
    result = pipe(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]
    # Strip the prompt from the output
    if isinstance(generated, str) and generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def generate_chat_gemini(
    model_name: str,
    system_prompt: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192,
    caller_function_name: str = ''
) -> str:
    """
    Generate text using Google Gemini models.

    Args:
        model_name: Name of the model to use (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro')
        system_prompt: Optional system prompt
        messages: List of messages with 'role' and 'content' keys
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 8192)
        caller_function_name: Name of the calling function for usage tracking

    Returns:
        Generated text response
    """
    if _gemini_client() is None:
        raise ValueError("GEMINI_API_KEY not set. Please set the GEMINI_API_KEY environment variable.")

    # Convert messages to Gemini format
    # Gemini uses 'user' and 'model' roles, with content as parts
    gemini_contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Map OpenAI-style roles to Gemini roles
        if role == "assistant":
            role = "model"
        elif role == "system":
            # System messages should be handled via system_instruction
            # If there's a system message in the list, prepend it to system_prompt
            if system_prompt:
                system_prompt = content + "\n\n" + system_prompt
            else:
                system_prompt = content
            continue

        gemini_contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=content)]
            )
        )

    # Build config with optional system instruction
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=genai_types.ThinkingConfig(thinkingBudget=0)
    )
    if system_prompt is not None:
        config.system_instruction = system_prompt

    start_time = time.time()
    response = _gemini_client().models.generate_content(
        model=model_name,
        contents=gemini_contents,
        config=config
    )
    elapsed_time = time.time() - start_time

    # Extract token usage from response metadata
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

    # Save usage data
    if input_tokens > 0 or output_tokens > 0:
        save_gemini_usage(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_time=elapsed_time,
            caller_function_name=caller_function_name
        )

    return response.text


def generate_chat(
    model_name: str,
    system_prompt: Optional[str],
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    verbose: bool = False
) -> str:
    """
    Generate text using various LLM providers based on model name.

    Args:
        model_name: Name of the model to use
        system_prompt: Optional system prompt
        messages: List of messages
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 1000)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Generated text response

    Raises:
        NotImplementedError: If model provider is not implemented
    """
    # Find the first function name in the call stack that is not "generate", "generate_and_process", or "generate_and_process_yaml"
    skip_names = {"generate", "generate_and_process", "generate_and_process_yaml"}
    caller_function_name = None
    for frame_info in inspect.stack():
        func_name = frame_info.function
        if func_name not in skip_names:
            caller_function_name = func_name
            break
    # Defensive fallback in case all are skipped (should not happen)
    if caller_function_name is None:
        caller_function_name = "unknown"


    # When DeepSeek fallback is active, route ALL external provider calls
    # (GPT, Claude, Gemini) through DeepSeek's OpenAI-compatible API.
    # vLLM/Together routing (generate_chat_other) is never redirected.
    if USE_DEEPSEEK_FALLBACK and ("gpt" in model_name or "claude" in model_name or "gemini" in model_name):
        return generate_chat_deepseek(
            model_name=model_name,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    for attempt in range(max_retries):
        try:
            output_text = None

            # GPT MODELS
            if "gpt" in model_name:
                output_text = generate_chat_openai(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            # CLAUDE MODELS
            elif "claude" in model_name:
                output_text = generate_chat_anthropic(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    caller_function_name=caller_function_name
                )

            # GEMINI MODELS
            elif "gemini" in model_name:
                output_text = generate_chat_gemini(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    caller_function_name=caller_function_name
                )

            # OTHER MODELS
            else:
                output_text = generate_chat_other(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            if verbose:
                print(f"<{caller_function_name.upper()} ({model_name}))>")
                print(f"{output_text}")
                print("-" * 40)
                print()

            return output_text

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except (openai.APIConnectionError, openai.InternalServerError):
            if attempt == max_retries - 1:
                raise
            # Shorter wait for connection/server errors
            wait_time = 1 + random.uniform(0, 1)
            print(f"API connection/server error, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            # Handle vLLM HTTP request errors
            wait_time = 1 + random.uniform(0, 1)
            print(f"vLLM request error: {e}, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

    raise NotImplementedError(f"Model provider for {model_name} not implemented yet.")

EMPTY_GENERATION_ERROR = "Generated output is empty or whitespace-only - stopping generation"
def generate_and_process(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    processing_function: Callable,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    verbose: bool = False,
    validation_function: Callable = None
) -> Any:
    """
    Generate text using an LLM and process it using a processing function with retry logic.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            output = generate(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=3,  # API-level retries
                verbose=verbose
            ).strip()

            if output == "":
                raise ValueError(EMPTY_GENERATION_ERROR + "\n[INPUT] " + user_prompt)

            processed = processing_function(output)
            if validation_function is not None:
                validation_function(processed)
            return processed, output

        except ValueError as e:
            last_error = e
            if last_error.args and last_error.args[0] == EMPTY_GENERATION_ERROR:
                raise last_error
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Processing failed, retrying in {wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                if verbose:
                    print(f"Error details: {str(e)}")
                time.sleep(wait_time)
            else:
                print(f"❌ Processing failed after {max_retries} attempts")
                print(f"Final error: {str(last_error)}")
                if hasattr(last_error, 'args') and last_error.args:
                    # Try to extract output from error message if available
                    error_str = str(last_error)
                    if "Output was:" in error_str or "Generated output" in error_str:
                        print("Error details already include output")
                raise last_error

    raise last_error if last_error else ValueError("Failed to process generated output")

def generate_and_process_yaml(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    verbose: bool = False,
    validation_function: Callable = None
) -> Dict[str, Any]:
    """
    Generate text using an LLM and process it as YAML with retry logic.

    Args:
        model_name: Name of the model to use
        system_prompt: Optional system prompt
        user_prompt: User prompt/input text
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 8192)
        max_retries: Maximum number of retry attempts for YAML processing (default: 3)
        verbose: Whether to print verbose output and retry messages

    Returns:
        Parsed YAML dict

    Raises:
        ValueError: If YAML processing fails after all retries
    """
    return generate_and_process(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        processing_function=process_yaml,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        validation_function=validation_function
    )


def generate_and_process_json(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    verbose: bool = False,
    validation_function: Callable = None
) -> Dict[str, Any]:
    """
    Generate text using an LLM and process it as JSON with retry logic.
    """
    return generate_and_process(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        processing_function=process_json,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        validation_function=validation_function
    )
