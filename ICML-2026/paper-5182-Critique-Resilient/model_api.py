import atexit
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import os
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Global session for connection pooling
_http_session = None
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def get_http_session() -> requests.Session:
    """Get or create a shared HTTP session for connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
    return _http_session

def _cleanup_http_session():
    """Close the global HTTP session on program exit."""
    global _http_session
    if _http_session is not None:
        _http_session.close()
        _http_session = None

# Register cleanup handler to close session on exit
atexit.register(_cleanup_http_session)


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 3,
    backoff_seconds: int = 1,
    **kwargs,
) -> requests.Response:
    last_exc = None
    last_response = None
    for attempt in range(max_retries):
        try:
            resp = session.request(method, url, **kwargs)
            last_response = resp
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
    if last_response is not None:
        return last_response
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Request failed for {method} {url}")


OPENAI_API_URL = "https://api.openai.com/v1"
ANTHROPIC_INTERNAL_NAMES = {
    "claude-opus-4.1": "claude-opus-4-1",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-opus-4": "claude-opus-4",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-3.7-sonnet": "claude-3-7-sonnet",
    "claude-sonnet-3.7": "claude-3-7-sonnet",
    "claude-3.5-haiku": "claude-3-5-haiku",
    "claude-haiku-3.5": "claude-3-5-haiku",
    "claude-3-haiku": "claude-3-haiku",
    "claude-haiku-3": "claude-3-haiku"
}

ANTHROPIC_MAX_TOKENS = {
    "claude-opus-4-1": 32000,
    "claude-opus-4-5-20251101": 64000,
    "claude-opus-4": 32000,
    "claude-sonnet-4": 64000,
    "claude-3-7-sonnet": 64000,
    "claude-3-5-haiku": 8192,
    "claude-3-haiku": 4096,
}

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_FILES_URL = f"{OPENAI_API_URL}/files"
OPENAI_BATCH_URL = f"{OPENAI_API_URL}/batches"
OPENAI_CONTENT_URL = f"{OPENAI_API_URL}/files/{{output_file_id}}/content"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EFFORT_RATIOS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
} # OpenRouter effort to token ratio

# Disable Anthropic batch API usage when True.
DISABLE_ANTHROPIC_BATCH_PROCESSING = True

def _validate_response_format(response_format):
    """Validate response_format parameter is in the correct format."""
    if response_format is None:
        return
    if not isinstance(response_format, dict):
        raise ValueError(f"response_format must be a dict, got {type(response_format).__name__}")
    if "type" not in response_format:
        raise ValueError("response_format dict must contain 'type' key")
    valid_types = {"json_object", "json_schema", "text"}
    if response_format["type"] not in valid_types:
        raise ValueError(f"response_format type must be one of {valid_types}, got '{response_format['type']}'")


# OpenRouter formula to compute max tokens from effort level
def anthropic_effort_to_tokens(model: str, effort: str):
    max_tokens = ANTHROPIC_MAX_TOKENS[model]

    return int(max(1024, min(32000, EFFORT_RATIOS[effort] * max_tokens)))

def _get_anthropic_max_tokens(model: str) -> int:
    max_tokens = ANTHROPIC_MAX_TOKENS.get(model)
    if max_tokens is None:
        for key in ANTHROPIC_MAX_TOKENS:
            if model.startswith(key):
                max_tokens = ANTHROPIC_MAX_TOKENS[key]
                break
    if max_tokens is None:
        raise ValueError(f"Model '{model}' not found in ANTHROPIC_MAX_TOKENS")
    return max_tokens


def _normalize_message_content(content: object) -> str:
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def _anthropic_messages_from_messages(messages: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
    system_parts = []
    out_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _normalize_message_content(msg.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        out_messages.append({"role": role, "content": content})
    system = "\n".join(system_parts) if system_parts else None
    return out_messages, system


def _query_openai_single(model: str, messages: List[Dict], response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format:
        payload["response_format"] = response_format
    if reasoning:
        payload["reasoning_effort"] = reasoning
    if api_kwargs:
        for k, v in api_kwargs.items():
            if k not in ["reasoning"]:
                payload[k] = v

    session = get_http_session()
    resp = _request_with_retries(
        session,
        "POST",
        f"{OPENAI_API_URL}/chat/completions",
        headers=headers,
        json=payload,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error: {resp.status_code} - {resp.text}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI error: No choices in response: {data}")
    message_obj = choices[0].get("message", {})
    content = message_obj.get("content")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "".join(text_parts)
    return content or ""

def _query_anthropic_single(model: str, messages: List[Dict], response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    anthropic_messages, system = _anthropic_messages_from_messages(messages)
    max_tokens = _get_anthropic_max_tokens(model)
    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning in ["medium", "high"]:
        thinking_tokens = anthropic_effort_to_tokens(model, reasoning)
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}
    if response_format:
        payload["response_format"] = response_format
    if api_kwargs:
        for key, value in api_kwargs.items():
            if key not in ["reasoning", "thinking"]:
                payload[key] = value

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    session = get_http_session()
    resp = _request_with_retries(
        session,
        "POST",
        ANTHROPIC_MESSAGES_URL,
        headers=headers,
        json=payload,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic error: {resp.status_code} - {resp.text}")
    data = resp.json()
    content = data.get("content") or []
    for item in content:
        if item.get("type") == "text":
            return item.get("text", "")
    raise RuntimeError(f"Anthropic error: No text content found in response: {data}")

def query_llm(model: str, messages: List[Dict], response_format: Optional[Dict] = None, temperature: Optional[float] = None, api_kwargs: Optional[Dict] = None, reasoning: Optional[str] = None) -> str:
    """
    Query a single LLM endpoint.
    """
    _validate_response_format(response_format)

    if api_kwargs:
        if api_kwargs.get("reasoning") is not None:
            raise ValueError("Do not set 'reasoning' in api_kwargs; use the reasoning parameter instead.")

        if api_kwargs.get("temperature") is not None:
            raise ValueError("Do not set 'temperature' in api_kwargs; use the temperature parameter instead.")

    if model.startswith("openai/"):
        return _query_openai_single(model.replace("openai/", ""), messages, response_format, temperature, api_kwargs, reasoning)

    if "gemini" in model:
        return _query_gemini_single(model, messages, response_format, temperature, api_kwargs, reasoning)

    if 'anthropic' in model and temperature is not None and reasoning is not None:
        raise ValueError("Cannot set both temperature and reasoning in Anthropic requests")

    if "anthropic" in model:
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model.replace("anthropic/", ""), model.replace("anthropic/", ""))
        return _query_anthropic_single(norm_model, messages, response_format, temperature, api_kwargs, reasoning)

    json_kwargs = {}

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    json_kwargs["model"] = model
    json_kwargs["messages"] = messages
    if temperature is not None:
        json_kwargs["temperature"] = temperature

    if response_format:
        json_kwargs["response_format"] = response_format

    # Add reasoning to api_kwargs for OpenRouter
    if reasoning in ["medium", "high"]:
        json_kwargs["reasoning"] = {"effort": reasoning, "exclude": True}
    if api_kwargs:
        for key, value in api_kwargs.items():
            json_kwargs[key] = value

    session = get_http_session()
    response = _request_with_retries(
        session,
        "POST",
        OPENROUTER_API_URL,
        headers=headers,
        json=json_kwargs,
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error: {response.status_code} - {response.text}")

    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"OpenRouter error: No choices found in response: {data}")

    return data["choices"][0]["message"]["content"]


def query_llm_single(model: str, message: str, prompt: str = "You are a helpful assistant.", response_format: Optional[Dict] = None, temperature: Optional[float] = None, api_kwargs: Optional[Dict] = None, reasoning: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    return query_llm(model, messages, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs, reasoning=reasoning)


def _build_batch_requests(model: str, messages_list: List[str], prompt: str, response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str] = None) -> Tuple[List[Tuple[str, Dict]], List[str]]:
    """
    Build batch requests for the Anthropic Claude API.
    Returns: (batch_requests, custom_ids)
    """
    batch_requests = []
    custom_ids = []

    for message in messages_list:
        custom_id = f"msg-{uuid.uuid4()}"
        custom_ids.append(custom_id)

        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": message},
            ]
        }

        if temperature is not None:
            body["temperature"] = temperature

        if prompt:
            body["system"] = prompt
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model, model)
        body["max_tokens"] = _get_anthropic_max_tokens(norm_model)
        # Enable thinking if reasoning is set
        if reasoning in ["medium", "high"]:
            effort = reasoning
            thinking_tokens = anthropic_effort_to_tokens(norm_model, effort)
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}
        if response_format:
            body["response_format"] = response_format
        if api_kwargs:
            for key, value in api_kwargs.items():
                if key not in ["reasoning", "thinking"]:
                    body[key] = value

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_batch_results(results_text: str, custom_ids: List[str]) -> List[str]:
    """
    Map Anthropic .jsonl results to input order using custom_id.
    """
    responses_map = {}

    for line in results_text.strip().splitlines():
        result_obj = json.loads(line)
        cid = result_obj["custom_id"]
        result = result_obj["result"]
        if result["type"] == "succeeded":
            sub_result = result["message"]["content"]

            # Find the first result with type 'text'
            for item in sub_result:
                if item["type"] == "text":
                    responses_map[cid] = item["text"]
                    break
            else:
                raise RuntimeError(f"No text content found in Anthropic response: {result}")
        else:
            raise RuntimeError(f"Anthropic batch request failed: {result['type']}")

    return [responses_map[cid] for cid in custom_ids]


def _query_anthropic_batch(model: str, messages_list: List[str], prompt: str, response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str] = None) -> List[str]:
    """
    Helper for Anthropic Claude batch API.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    batch_requests, custom_ids = _build_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    batch_payload = {"requests": [
        {"custom_id": cid, "params": body} for cid, body in batch_requests
    ]}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    session = get_http_session()
    resp = _request_with_retries(
        session,
        "POST",
        ANTHROPIC_API_URL,
        headers=headers,
        json=batch_payload,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic error: Batch creation failed: {resp.status_code} - {resp.text}")
    batch_id = resp.json()["id"]
    poll_url = f"{ANTHROPIC_API_URL}/{batch_id}"

    # TQDM progress bar for polling
    with tqdm(total=1, desc="Polling Claude batch", bar_format='{desc}: {elapsed} [{bar}]') as pbar:
        while True:
            time.sleep(5)
            poll_resp = _request_with_retries(session, "GET", poll_url, headers=headers)
            if poll_resp.status_code != 200:
                raise RuntimeError(f"Anthropic error: Batch poll failed: {poll_resp.status_code} - {poll_resp.text}")
            poll_data = poll_resp.json()
            status = poll_data.get("processing_status")

            if status == "ended":
                results_url = poll_data.get("results_url")
                pbar.update(1)
                break
            pbar.set_postfix_str(f"Status: {status}")

    results_resp = _request_with_retries(session, "GET", results_url, headers=headers)
    if results_resp.status_code != 200:
        raise RuntimeError(f"Anthropic error: Batch results download failed: {results_resp.status_code} - {results_resp.text}")

    return _map_batch_results(results_resp.text, custom_ids)


def _build_openai_batch_requests(model: str, messages_list: List[str], prompt: str, response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str] = None) -> Tuple[List[Tuple[str, Dict]], List[str]]:
    """
    Build batch requests for the OpenAI chat completions endpoint.
    Returns: (batch_requests, custom_ids)
    """
    batch_requests = []
    custom_ids = []

    for message in messages_list:
        custom_id = f"msg-{uuid.uuid4()}"
        custom_ids.append(custom_id)

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
        }
        if temperature is not None:
            body["temperature"] = temperature

        # OpenAI chat/completions batches do not accept the "reasoning" param; ignore if provided.
        if response_format:
            body["response_format"] = response_format
        if api_kwargs:
            for key, value in api_kwargs.items():
                if key != "reasoning":
                    body[key] = value

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_openai_batch_results(results_text: str, custom_ids: List[str]) -> List[str]:
    """
    Map OpenAI .jsonl results to input order using custom_id.
    """
    responses_map = {}

    for line in results_text.strip().splitlines():
        result_obj = json.loads(line)
        cid = result_obj.get("custom_id")
        if not cid:
            continue

        response = result_obj.get("response")
        error = result_obj.get("error")

        if response and response.get("status_code") == 200:
            body = response.get("body", {})
            choices = body.get("choices") or []

            text_content = None
            if choices:
                message_obj = choices[0].get("message", {})
                content = message_obj.get("content")

                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    text_content = "".join(text_parts).strip() if text_parts else None
                else:
                    text_content = content

            if not text_content:
                raise RuntimeError(f"OpenAI batch request returned no text content for custom_id {cid}")
            responses_map[cid] = text_content
        else:
            raise RuntimeError(f"OpenAI batch request failed for custom_id {cid}: {error or 'Unknown error'}")

    if len(responses_map) != len(custom_ids):
        missing = [cid for cid in custom_ids if cid not in responses_map]
        raise RuntimeError(f"OpenAI batch missing responses for custom_ids: {missing}")

    return [responses_map[cid] for cid in custom_ids]


def _query_openai_batch(model: str, messages_list: List[str], prompt: str, response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str] = None) -> List[str]:
    """
    Helper for OpenAI batch API using the chat completions endpoint.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    batch_requests, custom_ids = _build_openai_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    batch_lines = []

    for cid, body in batch_requests:
        batch_line = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {**body, "reasoning_effort": reasoning} if reasoning else body,
        }
        batch_lines.append(json.dumps(batch_line))

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        for line in batch_lines:
            f.write(line + "\n")
        batch_file_path = f.name

    headers_auth = {"Authorization": f"Bearer {api_key}"}
    session = get_http_session()
    try:
        with open(batch_file_path, "rb") as file_handle:
            files = {"file": (os.path.basename(batch_file_path), file_handle)}
            data = {"purpose": "batch"}
            upload_resp = _request_with_retries(
                session,
                "POST",
                OPENAI_FILES_URL,
                headers=headers_auth,
                files=files,
                data=data,
            )

        if upload_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch file upload failed: {upload_resp.status_code} - {upload_resp.text}")

        input_file_id = upload_resp.json()["id"]

        create_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        create_payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        create_resp = _request_with_retries(
            session,
            "POST",
            OPENAI_BATCH_URL,
            headers=create_headers,
            json=create_payload,
        )

        if create_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch creation failed: {create_resp.status_code} - {create_resp.text}")

        batch_id = create_resp.json()["id"]
        poll_url = f"{OPENAI_BATCH_URL}/{batch_id}"

        with tqdm(total=1, desc="Polling OpenAI batch", bar_format="{desc}: {elapsed} [{bar}]") as pbar:
            while True:
                poll_resp = _request_with_retries(session, "GET", poll_url, headers=create_headers)
                if poll_resp.status_code != 200:
                    raise RuntimeError(f"OpenAI batch poll failed: {poll_resp.status_code} - {poll_resp.text}")

                poll_data = poll_resp.json()
                status = poll_data.get("status")

                if status == "completed":
                    output_file_id = poll_data.get("output_file_id")
                    if not output_file_id:
                        raise RuntimeError(f"OpenAI batch completed without output_file_id: {poll_data}")
                    pbar.update(1)
                    break
                if status in {"failed", "cancelled", "expired"}:
                    raise RuntimeError(f"OpenAI batch ended with status '{status}': {poll_data}")

                pbar.set_description(f"Polling OpenAI batch (status: {status})")
                time.sleep(5)

        content_url = OPENAI_CONTENT_URL.format(output_file_id=output_file_id)
        results_resp = _request_with_retries(session, "GET", content_url, headers=headers_auth)
        if results_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch results download failed: {results_resp.status_code} - {results_resp.text}")

        return _map_openai_batch_results(results_resp.text, custom_ids)

    finally:
        if os.path.exists(batch_file_path):
            os.remove(batch_file_path)


def _single_query_worker(args):
    model, message, prompt, response_format, temperature, api_kwargs, reasoning = args
    return query_llm_single(model, message, prompt=prompt, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs, reasoning=reasoning)


def _get_gemini_client():
    if not genai:
        raise RuntimeError("google-genai is not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key, project=project)


def _normalize_gemini_model(model: str) -> str:
    return model.replace("google/", "")


def _gemini_contents_from_messages(messages: list, prompt: str):
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_instruction = text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    if prompt and not system_instruction:
        system_instruction = prompt
    return contents, system_instruction


def _query_gemini_single(model: str, messages: List[Dict], response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str]) -> str:
    client = _get_gemini_client()
    model = _normalize_gemini_model(model)
    contents, system_instruction = _gemini_contents_from_messages(messages, prompt=None)
    thinking_config = None
    if reasoning in ["medium", "high"]:
        thinking_config = genai_types.ThinkingConfig(include_thoughts=False, thinking_level=reasoning)
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=thinking_config,
        system_instruction=system_instruction,
    )
    #help(client.models.generate_content)
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
        #system_instruction=system_instruction,
    )
    parts = resp.candidates[0].content.parts if resp.candidates else []
    text_parts = [p.text for p in parts if getattr(p, "text", None)]
    return "".join(text_parts)


def _query_gemini_batch(model: str, messages_list: List[str], prompt: str, response_format: Optional[Dict], temperature: Optional[float], api_kwargs: Optional[Dict], reasoning: Optional[str]) -> List[str]:
    client = _get_gemini_client()
    model = _normalize_gemini_model(model)
    request_key_to_index: Dict[str, int] = {}
    requests_file_entries = []
    for idx, msg in enumerate(messages_list):
        contents, system_instruction = _gemini_contents_from_messages([{"role": "user", "content": msg}], prompt)
        request_payload: Dict[str, object] = {"contents": contents}
        thinking_config = None
        if reasoning in ["medium", "high"]:
            thinking_config = genai_types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=reasoning,
            )
        generation_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=thinking_config,
        ).model_dump(by_alias=True, exclude_none=True, mode="json")
        if generation_config:
            request_payload["generationConfig"] = generation_config
        if system_instruction:
            request_payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        request_key = f"request-{idx}"
        request_key_to_index[request_key] = idx
        requests_file_entries.append({"key": request_key, "request": request_payload})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            for entry in requests_file_entries:
                tmp_file.write(json.dumps(entry))
                tmp_file.write("\n")
        uploaded_file = client.files.upload(
            file=tmp_path,
            config=genai_types.UploadFileConfig(
                display_name=f"gemini-batch-{uuid.uuid4().hex}",
                mime_type="jsonl",
            ),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    batch = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": f"gemini-batch-{uuid.uuid4().hex}"},
    )

    with tqdm(total=1, desc="Polling Gemini batch", bar_format="{desc}: {elapsed} [{bar}]") as pbar:
        while True:
            job = client.batches.get(name=batch.name)
            state = getattr(job, "state", None)
            state_name = state.name if state else "UNKNOWN"
            if state_name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                if state_name != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError(f"Gemini batch ended with state {state_name}")
                pbar.update(1)
                break
            pbar.set_postfix_str(f"State: {state_name}")
            time.sleep(5)

    if job.error:
        raise RuntimeError(f"Gemini batch error: {job.error}")
    responses = []

    # Handle different response formats
    if job.dest and getattr(job.dest, "file_name", None):
        # File-based response format
        file_name = job.dest.file_name
        logger.debug(f"Processing Gemini batch file-based response from {file_name}")
        content_bytes = client.files.download(file=file_name)
        text = content_bytes.decode("utf-8")
        line_count = len(text.strip().splitlines())
        logger.debug(f"Parsing {line_count} JSONL responses from Gemini batch file")
        responses = [None] * len(messages_list)
        for idx, line in enumerate(text.strip().splitlines()):
            try:
                obj = json.loads(line)
                resp = obj.get("response") or obj.get("inlineResponse")
                if not resp:
                    error = obj.get("error") or obj.get("status")
                    if error:
                        raise RuntimeError(f"Gemini batch response error: {error}")
                    raise RuntimeError("Gemini batch response missing response field")
                candidates = resp.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini batch response missing candidates")
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if p.get("text")]
                if not text_parts:
                    raise RuntimeError("Gemini batch response missing text content")
                response_text = "".join(text_parts)
                key = obj.get("key")
                if key and key in request_key_to_index:
                    responses[request_key_to_index[key]] = response_text
                elif key:
                    raise RuntimeError(f"Gemini batch response has unknown key: {key}")
                else:
                    responses[idx] = response_text
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini batch response line {idx}: {e}")
                raise RuntimeError(f"Failed to parse Gemini batch response: {e}")
        if any(r is None for r in responses):
            missing = [str(i) for i, r in enumerate(responses) if r is None]
            raise RuntimeError(f"Gemini batch response missing outputs for indexes: {', '.join(missing)}")
        logger.debug(f"Successfully parsed {len(responses)} responses from Gemini batch file")

    elif job.dest and getattr(job.dest, "inlined_responses", None):
        # Inlined response format
        logger.debug("Processing Gemini batch inlined response format")
        inline_count = len(job.dest.inlined_responses)
        logger.debug(f"Processing {inline_count} inlined responses from Gemini batch")
        for inline_response in job.dest.inlined_responses:
            resp = getattr(inline_response, "response", None)
            if resp and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                text_parts = [p.text for p in parts if getattr(p, "text", None)]
                responses.append("".join(text_parts))
            elif getattr(inline_response, "error", None):
                raise RuntimeError(f"Gemini batch request failed: {inline_response.error}")
            else:
                raise RuntimeError("Gemini batch request returned no response")
        logger.debug(f"Successfully processed {len(responses)} inlined responses from Gemini batch")
    else:
        logger.error(f"Gemini batch job has unrecognized destination format: {job.dest}")
        raise RuntimeError(f"Gemini batch job has unrecognized destination format: {job.dest}")

    if len(responses) != len(messages_list):
        raise RuntimeError(f"Gemini batch returned {len(responses)} responses but expected {len(messages_list)}")
    return responses


def query_llm_parallel(
    model: str,
    messages_list: List[str],
    prompt: str = "You are a helpful assistant.",
    response_format: Optional[Dict] = None,
    temperature: Optional[float] = None,
    api_kwargs: Optional[Dict] = None,
    reasoning: Optional[str] = None,
    max_workers: int = 8,
    show_progress: bool = True,
    desc: Optional[str] = None,
) -> List[str]:
    """
    Parallel query using query_llm_single for each message. Avoids provider batch APIs.
    """
    if not messages_list:
        return []

    _validate_response_format(response_format)

    if api_kwargs:
        if api_kwargs.get("reasoning") is not None:
            raise ValueError("Do not set 'reasoning' in api_kwargs; use the reasoning parameter instead.")

        if api_kwargs.get("temperature") is not None:
            raise ValueError("Do not set 'temperature' in api_kwargs; use the temperature parameter instead.")

    worker_args = [(model, message, prompt, response_format, temperature, api_kwargs, reasoning) for message in messages_list]
    if max_workers is None or max_workers < 1 or len(worker_args) == 1:
        results_iter = (_single_query_worker(args) for args in worker_args)
        if show_progress and len(worker_args) > 1:
            return list(tqdm(results_iter, total=len(worker_args), desc=desc or f"Processing {model}", unit="query"))
        return list(results_iter)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(_single_query_worker, worker_args)
        if show_progress:
            return list(tqdm(results_iter, total=len(worker_args), desc=desc or f"Processing {model}", unit="query"))
        return list(results_iter)


def query_llm_batch(model: str, messages_list: List[str], prompt: str = "You are a helpful assistant.", response_format: Optional[Dict] = None, temperature: Optional[float] = None, api_kwargs: Optional[Dict] = None, reasoning: Optional[str] = None, max_workers: int = 8) -> List[str]:
    """
    Batch query for LLMs: only Anthropic Claude batch API is used. All other models use parallel processing (multiprocessing).
    reasoning: None, "medium", or "high". If set, enables Claude 'thinking' or OpenRouter 'reasoning'.
    max_workers: number of parallel workers for non-Anthropic models.
    """
    if not messages_list:
        return []

    _validate_response_format(response_format)

    if api_kwargs:
        if api_kwargs.get("reasoning") is not None:
            raise ValueError("Do not set 'reasoning' in api_kwargs; use the reasoning parameter instead.")

        if api_kwargs.get("temperature") is not None:
            raise ValueError("Do not set 'temperature' in api_kwargs; use the temperature parameter instead.")

    if "gemini" in model:
        return _query_gemini_batch(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning)
    if 'anthropic' in model:
        if temperature is not None and reasoning is not None:
            raise ValueError("Cannot set both temperature and reasoning in Anthropic requests")
        if DISABLE_ANTHROPIC_BATCH_PROCESSING:
            return query_llm_parallel(
                model,
                messages_list,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                api_kwargs=api_kwargs,
                reasoning=reasoning,
                max_workers=max_workers,
                show_progress=True,
                desc=f"Processing {model}",
            )
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model.replace('anthropic/', ''), model.replace('anthropic/', ''))
        return _query_anthropic_batch(norm_model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    if 'openai' in model:
        return _query_openai_batch(model.replace('openai/', ''), messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)

    worker_args = [(model, message, prompt, response_format, temperature, api_kwargs, reasoning) for message in messages_list]
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(_single_query_worker, worker_args),
            total=len(worker_args),
            desc=f"Processing {model}",
            unit="query"
        ))
    return results
