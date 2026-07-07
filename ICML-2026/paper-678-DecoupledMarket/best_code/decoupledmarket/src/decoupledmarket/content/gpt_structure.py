import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Callable, Dict, Optional

import google.generativeai as genai
import openai
from openai import OpenAI
from timeout_decorator import timeout

from decoupledmarket.database_utils import round_two_decimal
from zai import ZhipuAiClient
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _record_api_usage(response: Any) -> None:
    try:
        from decoupledmarket.performance_monitor import get_monitor

        monitor = get_monitor()
        if response is not None and getattr(response, "usage", None) is not None:
            usage = response.usage
            monitor.record_api_call(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", None),
            )
        else:
            monitor.record_api_call(0, 0)
    except Exception:
        pass


def _load_env_file(path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into process env without overriding existing vars."""
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                if key:
                    os.environ.setdefault(key, value)
    except Exception:
        pass


# Auto-load local .env files (project root + current working dir).
_load_env_file(os.path.join(os.getcwd(), ".env"))
_load_env_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def _print_env_status() -> None:
    """Print non-sensitive API env status for quick diagnostics."""
    if os.getenv("PRINT_API_ENV_STATUS", "1") != "1":
        return

    keys = [
        "OPENAI_API_KEY",
        "GLM_API_KEY",
        "DEEPINFRA_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
    ]
    print("[ENV] API key status:")
    for key in keys:
        value = os.getenv(key, "")
        status = "SET" if value else "NOT_SET"
        print(f"[ENV] {key}: {status}")


_print_env_status()


# =============================================================================
# Rate Limiter and Retry Configuration
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter with per-provider tracking."""

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._last_request: Dict[str, float] = {}
        self._min_intervals: Dict[str, float] = {
            "openai": 0.1,        # 10 requests per second max
            "deepinfra": 0.5,     # 2 requests per second (stricter for 429 avoidance)
            "bigmodel": 0.3,      # ~3 requests per second
            "deepseek": 0.2,      # 5 requests per second
            "gemini": 0.2,        # 5 requests per second
            "default": 0.1,
        }

    def _get_lock(self, provider: str) -> threading.Lock:
        if provider not in self._locks:
            self._locks[provider] = threading.Lock()
        return self._locks[provider]

    def wait_if_needed(self, provider: str) -> None:
        """Wait if necessary to respect rate limits."""
        lock = self._get_lock(provider)
        with lock:
            min_interval = self._min_intervals.get(provider, self._min_intervals["default"])
            last_time = self._last_request.get(provider, 0)
            now = time.time()
            elapsed = now - last_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            self._last_request[provider] = time.time()


class RetryHandler:
    """Exponential backoff retry handler for API calls."""

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute_with_retry(self, func: Callable, provider: str = "default", *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check for rate limit or server errors
                is_rate_limit = "429" in error_str or "rate" in error_str or "limit" in error_str
                is_server_error = "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str
                is_auth_error = "401" in error_str or "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str

                # Don't retry auth errors
                if is_auth_error:
                    logger.error(f"[{provider}] Authentication error: {e}")
                    raise

                if is_rate_limit or is_server_error:
                    # Exponential backoff with jitter
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    logger.warning(f"[{provider}] Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    # For other errors, shorter retry
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (attempt + 1)
                        logger.warning(f"[{provider}] Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)

        raise last_exception


# Global instances
_rate_limiter = RateLimiter()
_retry_handler = RetryHandler(
    max_retries=int(os.getenv("API_MAX_RETRIES", "5")),
    base_delay=float(os.getenv("API_BASE_DELAY", "1.0")),
    max_delay=float(os.getenv("API_MAX_DELAY", "60.0"))
)

# Provider mapping for rate limiting
_PROVIDER_MAP = {
    "api.openai.com": "openai",
    "api.deepinfra.com": "deepinfra",
    "open.bigmodel.cn": "bigmodel",
    "api.deepseek.com": "deepseek",
    "generativelanguage.googleapis.com": "gemini",
}


def _get_provider_from_url(base_url: str) -> str:
    """Extract provider name from base URL."""
    for url_part, provider in _PROVIDER_MAP.items():
        if url_part in (base_url or ""):
            return provider
    return "default"


# =============================================================================
# End Rate Limiter Configuration
# =============================================================================


proxy_url = os.getenv("LOCAL_PROXY_URL", "http://127.0.0.1")
proxy_port = os.getenv("LOCAL_PROXY_PORT", "7890")
use_local_proxy = os.getenv("USE_LOCAL_PROXY", "0") == "1"
if use_local_proxy:
    os.environ["http_proxy"] = f"{proxy_url}:{proxy_port}"
    os.environ["https_proxy"] = f"{proxy_url}:{proxy_port}"


def temp_sleep(seconds: int = 1) -> None:
    time.sleep(seconds)


def _require_api_key(env_name: str) -> str:
    api_key = os.getenv(env_name, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key in environment variable: {env_name}")
    return api_key


def gemini(prompt: str) -> str:
    provider = "gemini"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        google_api_key = _require_api_key("GOOGLE_API_KEY")
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name="gemini-3-pro-preview")
        response = model.generate_content(prompt)
        _record_api_usage(None)
        return response.text

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"Gemini ERROR: {e}")
        return "Gemini ERROR"


def qwen_request(prompt: str) -> str:
    provider = "deepinfra"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(
            api_key=_require_api_key("Qwen_API_KEYs"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[{"role": "user", "content": prompt}],
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"Qwen ERROR: {e}")
        return "Qwen ERROR"


def deepseek_v3_2(prompt: str) -> str:
    provider = "deepinfra"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(
            api_key=_require_api_key("Deepseek_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"Deepseek ERROR: {e}")
        return "Deepseek ERROR"


def Chat5_request(prompt: str, model_name: str) -> str:
    provider = "openai"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(api_key=_require_api_key("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"ChatGPT5 {model_name} ERROR: {e}")
        return "ChatGPT5 ERROR"


def ChatGPT_request(prompt: str, model_name: str) -> str:
    provider = "openai"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(api_key=_require_api_key("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"ChatGPT {model_name} ERROR: {e}")
        return "ChatGPT ERROR"


def deepseek3v1(prompt: str) -> str:
    provider = "deepseek"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(
            api_key=_require_api_key("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    return _retry_handler.execute_with_retry(_call, provider)


def llama3_request(prompt: str) -> str:
    provider = "deepinfra"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        client = OpenAI(
            api_key=_require_api_key("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        _record_api_usage(response)
        return response.choices[0].message.content

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"llama ERROR: {e}")
        return "llama ERROR"


def glm5_request(prompt: str) -> str:
    provider = "bigmodel"
    _rate_limiter.wait_if_needed(provider)

    def _call():
        glm_api_key = _require_api_key("GLM_API_KEY")
        client = ZhipuAiClient(api_key=glm_api_key)
        response = client.chat.completions.create(
            model="GLM-4.7",
            messages=[{"role": "user", "content": prompt}],
            # Disabled thinking mode for faster response
            max_tokens=4096,
            temperature=0.7
        )
        _record_api_usage(response)
        logger.debug(f"GLM response received")
        message = response.choices[0].message
        if isinstance(message, str):
            return message
        return getattr(message, "content", "") or str(message)

    try:
        return _retry_handler.execute_with_retry(_call, provider)
    except Exception as e:
        logger.error(f"GLM-4.7 ERROR: {e}")
        return "GLM-4.7 ERROR"

_GLM_RATE_LOCK = threading.Lock()
_GLM_NEXT_ALLOWED_TS = 0.0


@timeout(50)
def send_request(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    return ChatGPT_request(prompt, model_name).strip()


def _request_by_model(agent_model: str, prompt: str) -> str:
    model = (agent_model or "").strip().lower()

    # Disable Gemini (network issues in China)
    disable_gemini = os.getenv("DISABLE_GEMINI", "1") == "1"
    if disable_gemini and model == "gemini":
        fallback = os.getenv("DISABLE_GEMINI_TARGET", "gpt-4o-mini").strip().lower()
        logger.info(f"[Gemini disabled] Falling back to {fallback}")
        model = fallback

    # Disable GLM if configured
    disable_glm = os.getenv("DISABLE_GLM", "0") == "1"
    if disable_glm and model == "glm-5":
        fallback = os.getenv("DISABLE_GLM_TARGET", "gpt-3.5-turbo").strip().lower()
        logger.info(f"[GLM disabled] Falling back to {fallback}")
        model = fallback

    if model in {"gpt-4o-mini", "gpt-3.5-turbo"}:
        return ChatGPT_request(prompt, model).strip()
    if model in {"gpt-5", "gpt-5-mini"}:
        return Chat5_request(prompt, model).strip()
    if model == "qwen":
        return qwen_request(prompt).strip()
    if model == "llama3":
        return llama3_request(prompt).strip()
    if model == "gemini":
        return gemini(prompt).strip()
    if model == "deepseek":
        return deepseek_v3_2(prompt).strip()
    if model == "glm-5":
        return glm5_request(prompt).strip()
    return ChatGPT_request(prompt, agent_model).strip()


def _extract_json_output(text: str) -> Optional[str]:
    cleaned = re.sub(r"\s{3,}", "\n", text).replace("\n", "\\n")
    begin = cleaned.find("{")
    end = cleaned.rfind("}")
    if begin < 0 or end < 0 or end <= begin:
        return None
    payload = cleaned[begin : end + 1]
    try:
        return json.loads(payload)["output"]
    except Exception:
        return None


def _build_json_prompt(prompt: str, example_output: str, special_instruction: str) -> str:
    final_prompt = '"""' + "\n" + prompt + "\n" + '"""' + "\n"
    final_prompt += f"Output the response to the prompt above in JSON. {special_instruction}\n"
    final_prompt += "Only output in this format:\n"
    final_prompt += '{"output": "' + str(example_output) + '"}'
    return final_prompt


def llm_safe_generate_response(
    persona: Any,
    prompt: str,
    example_output: str,
    special_instruction: str,
    repeat: int = 3,
    fail_safe_response: str = "error",
    func_validate: Optional[Callable[[str, str], Any]] = None,
    func_clean_up: Optional[Callable[[str, str], Any]] = None,
    verbose: bool = False,
    virtual_date: Optional[int] = None,
    iteration: Optional[int] = None,
):
    original_prompt = prompt

    # Check if OpenCLAW memory is disabled
    disable_openclaw = os.getenv("DISABLE_OPENCLAW", "0") == "1"

    if not disable_openclaw:
        try:
            from openclaw_memory import get_memory_snippet_for_prompt

            agent_id = getattr(persona, "person_id", -1)
            memory_snippet = get_memory_snippet_for_prompt(
                agent_id=agent_id,
                query=original_prompt,
                max_items=5,
            )
            if memory_snippet:
                prompt = (
                    "Below is the agent's long-term memory across previous episodes.\n"
                    "Use it as background context to keep behavior consistent, "
                    "but do NOT repeat it verbatim:\n"
                    f"{memory_snippet}\n\n"
                    "Now consider the current situation:\n"
                    + prompt
                )
        except Exception as e:
            print(f"[OpenCLAW] Error retrieving memory for agent {getattr(persona, 'person_id', -1)}: {e}")

    final_prompt = _build_json_prompt(prompt, example_output, special_instruction)
    if verbose:
        print("CHAT GPT PROMPT")
        print(final_prompt)

    for _ in range(repeat):
        try:
            raw = _request_by_model(getattr(persona, "agent_model", ""), final_prompt)
            output = _extract_json_output(raw)
            if output is None:
                continue
            if func_validate and func_validate(output, prompt=final_prompt):
                cleaned = func_clean_up(output, prompt=final_prompt) if func_clean_up else output

                # Only store memory if OpenCLAW is enabled
                if not disable_openclaw:
                    try:
                        from openclaw_memory import append_memory_entry


                        iter_num = iteration if iteration is not None else getattr(persona, "iteration", None)

                        append_memory_entry(
                            agent_id=getattr(persona, "person_id", -1),
                            event_type="llm_call",
                            prompt=original_prompt,
                            response=str(cleaned),
                            meta={
                                "agent_model": getattr(persona, "agent_model", ""),
                                "virtual_date": vdate,
                                "iteration": iter_num,
                                "episode_id": f"day_{vdate}_iter_{iter_num}" if vdate is not None and iter_num is not None else None,
                            },
                        )
                    except Exception as e:
                        print(f"[OpenCLAW] Error storing memory for agent {getattr(persona, 'person_id', -1)}: {e}")
                return cleaned
        except Exception as e:
            print(f"{getattr(persona, 'agent_model', '')} connection error: {e}")
    return fail_safe_response if fail_safe_response not in (None, "error") else False


def ChatGPT_safe_generate_response(
    persona: Any,
    prompt: str,
    example_output: str,
    special_instruction: str,
    repeat: int = 3,
    fail_safe_response: str = "error",
    func_validate: Optional[Callable[[str, str], Any]] = None,
    func_clean_up: Optional[Callable[[str, str], Any]] = None,
    verbose: bool = False,
    virtual_date: Optional[int] = None,
    iteration: Optional[int] = None,
):
    original_prompt = prompt

    # Check if OpenCLAW memory is disabled
    disable_openclaw = os.getenv("DISABLE_OPENCLAW", "0") == "1"

    if not disable_openclaw:
        try:
            from openclaw_memory import get_memory_snippet_for_prompt

            agent_id = getattr(persona, "person_id", -1)
            memory_snippet = get_memory_snippet_for_prompt(
                agent_id=agent_id,
                query=original_prompt,
                max_items=5,
            )
            if memory_snippet:
                prompt = (
                    "Below is the agent's long-term memory across previous episodes.\n"
                    "Use it as background context to keep behavior consistent, "
                    "but do NOT repeat it verbatim:\n"
                    f"{memory_snippet}\n\n"
                    "Now consider the current situation:\n"
                    + prompt
                )
        except Exception as e:
            print(f"[OpenCLAW] Error retrieving memory for agent {getattr(persona, 'person_id', -1)}: {e}")

    final_prompt = _build_json_prompt(prompt, example_output, special_instruction)
    if verbose:
        print("CHAT GPT PROMPT")
        print(final_prompt)

    for _ in range(repeat):
        try:
            raw = _request_by_model(getattr(persona, "agent_model", ""), final_prompt)
            output = _extract_json_output(raw)
            if output is None:
                continue

            if func_validate and func_validate(output, prompt=final_prompt):
                cleaned = func_clean_up(output, prompt=final_prompt) if func_clean_up else output

                # Only store memory if OpenCLAW is enabled
                if not disable_openclaw:
                    try:
                        from openclaw_memory import append_memory_entry


                        iter_num = iteration if iteration is not None else getattr(persona, "iteration", None)

                        append_memory_entry(
                            agent_id=getattr(persona, "person_id", -1),
                            event_type="llm_call",
                            prompt=original_prompt,
                            response=str(cleaned),
                            meta={
                                "agent_model": getattr(persona, "agent_model", ""),
                                "virtual_date": vdate,
                                "iteration": iter_num,
                                "episode_id": f"day_{vdate}_iter_{iter_num}" if vdate is not None and iter_num is not None else None,
                            },
                        )
                    except Exception as e:
                        print(f"[OpenCLAW] Error storing memory for agent {getattr(persona, 'person_id', -1)}: {e}")
                return cleaned
        except Exception as e:
            print(f"{getattr(persona, 'agent_model', '')} connection error: {e}")
    return fail_safe_response if fail_safe_response not in (None, "error") else False


def GPT_request(prompt: str, gpt_parameter: Dict[str, Any]) -> str:
    temp_sleep()
    try:
        response = openai.Completion.create(
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"],
        )
        return response.choices[0].text
    except Exception:
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input: Any, prompt_lib_file: str) -> str:
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(round_two_decimal(i)) for i in curr_input]

    with open(prompt_lib_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    for count, value in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", value)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(
    prompt: str,
    gpt_parameter: Dict[str, Any],
    repeat: int = 5,
    fail_safe_response: str = "error",
    func_validate: Optional[Callable[[str, str], Any]] = None,
    func_clean_up: Optional[Callable[[str, str], Any]] = None,
    verbose: bool = False,
):
    if verbose:
        print(prompt)

    for _ in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate and func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response
    return fail_safe_response

def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    text = text.replace("\n", " ") if text else "this is blank"
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
