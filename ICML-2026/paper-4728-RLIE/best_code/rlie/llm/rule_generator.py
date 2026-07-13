import json
import logging
import time
from typing import Dict, List

from ollama import Client as OllamaClient
from openai import APIError, AuthenticationError, OpenAI

from rlie.llm.api_client_pool import APIClientPool
from rlie.llm.parse_failure_recorder import record_attempt, record_failure, record_usage
from rlie.utils.logger_config import LoggerConfig


LOGGER = logging.getLogger("rlie.generation")
GENERATION_LOGGER_NAME = "RLIE - Generation"


FALLBACK_RULES = [
    "If a post mentions work-related deadlines, pressure, workload, or burnout, label as stress; otherwise no stress.",
    "If a post describes physical symptoms (headache, insomnia, fatigue, anxiety attacks) in the context of life demands, label as stress; otherwise no stress.",
    "If a post discusses financial hardship, debt, or inability to pay bills, label as stress; otherwise no stress.",
    "If a post mentions relationship problems, family conflicts, or social isolation, label as stress; otherwise no stress.",
    "If a post describes academic pressure, exam anxiety, or school-related overwhelming demands, label as stress; otherwise no stress.",
    "If a post expresses feelings of being overwhelmed, trapped, or unable to cope with daily responsibilities, label as stress; otherwise no stress.",
    "If a post uses casual, humorous, or neutral language about daily events without emotional distress signals, label as no stress; otherwise stress.",
    "If a post describes positive coping strategies, successful problem resolution, or social support received, label as no stress; otherwise stress.",
    "If a post mentions health concerns or medical issues within a stressful life context, label as stress; otherwise no stress.",
    "If a post describes life transitions (moving, job change, loss of loved one) with negative emotional impact, label as stress; otherwise no stress.",
]


def _parse_ollama_model(model_name: str) -> str:
    return model_name.split(":", 1)[1] if ":" in model_name else model_name.partition(" ")[2] or model_name


def _should_use_responses_api(model_name: str, force_flag: bool = False) -> bool:
    return force_flag or str(model_name or "").lower().startswith("gpt-5")


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    output = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def extract_hypotheses(text: str, num_hypotheses: int) -> List[str]:
    import re

    logger = LoggerConfig.get_logger(GENERATION_LOGGER_NAME)
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    logger.info("Text provided %s", text)
    hypotheses = pattern.findall(text)

    if len(hypotheses) == 0:
        logger.info("No hypotheses are generated.")
        return []

    deduped: List[str] = []
    seen = set()
    for hypothesis in hypotheses:
        cleaned = hypothesis.strip()
        if cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)

    if len(deduped) != num_hypotheses:
        logger.info(
            "Expected %d hypotheses, but got %d.",
            num_hypotheses,
            len(deduped),
        )

    return deduped[:num_hypotheses]


class RuleGenerator:
    def __init__(self, config: Dict):
        self.model_name = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.num_predict = int(config.get("num_predict", 4096))
        self.max_retries = max(1, int(config.get("max_retries", 3)))
        self.min_backoff = float(config.get("min_backoff", 1.0))
        self.max_backoff = float(config.get("max_backoff", 30.0))
        self.request_timeout = float(config.get("request_timeout", 120))
        self.request_delay = float(config.get("request_delay", 0.0))
        self._fallback_index = 0
        self.output_dir = config.get("output_dir")

        model_lower = self.model_name.lower()
        self._use_ollama = model_lower.startswith("ollama")
        client_cfg = config.get("client", {})
        request_timeout = float(config.get("request_timeout", 120))
        disable_flag = config.get("disable_thinking")
        # thinking tri-state:
        #   None  -> do not send enable_thinking
        #   True  -> force enable_thinking=true
        #   False -> force enable_thinking=false
        if disable_flag is True:
            self._thinking_mode = False
        elif disable_flag is False:
            self._thinking_mode = True
        else:
            self._thinking_mode = None
        self._reasoning_effort = config.get("reasoning_effort")
        self._chat_template_kwargs = config.get("chat_template_kwargs")
        self._use_responses_api = _should_use_responses_api(
            self.model_name,
            bool(config.get("use_responses_api", False)),
        )
        self._client_pool = None

        if self._use_ollama:
            host = client_cfg.get("base_url", "http://localhost:11434")
            self.model = _parse_ollama_model(self.model_name)
            self.client = OllamaClient(host=host, timeout=request_timeout)
        else:
            base_url = client_cfg.get("base_url")
            api_keys = client_cfg.get("api_keys") or client_cfg.get("api_key")
            if isinstance(api_keys, str):
                api_keys = [api_keys]
            if not base_url or not api_keys:
                raise ValueError("Rule generation requires client.base_url and api_key(s)")
            self.model = self.model_name
            if len(api_keys) == 1:
                self.client = OpenAI(base_url=base_url, api_key=api_keys[0])
            else:
                self._client_pool = APIClientPool(base_url=base_url, api_keys=api_keys)
                self.client = None

    def generate(self, messages: List[Dict[str, str]], expected: int) -> List[str]:
        LOGGER.info("Calling %s model %s for %d expected hypotheses", "local" if self._use_ollama else "remote", self.model, expected)
        prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
        record_attempt(self.output_dir, "generation")
        last_response: str = ""
        try:
            if self._use_ollama:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.num_predict,
                    },
                    think=False,
                    timeout=self.request_timeout,
                )
                content = (response.get("message") or {}).get("content", "")
                last_response = content or ""
            else:
                backoff = self.min_backoff
                for attempt in range(1, self.max_retries + 1):
                    kwargs = {
                        "timeout": self.request_timeout,
                    }
                    try:
                        # 根据重试次数选择客户端：首次尝试用常规 client，重试时轮换 key
                        if attempt == 1:
                            client = (
                                self._client_pool.get_sync_client()
                                if self._client_pool
                                else self.client
                            )
                        else:
                            client = (
                                self._client_pool.get_sync_client_for_retry(attempt)
                                if self._client_pool
                                else self.client
                            )
                        if self._use_responses_api:
                            kwargs.update(
                                {
                                    "model": self.model,
                                    "input": messages,
                                    "max_output_tokens": self.num_predict,
                                    "temperature": self.temperature,
                                    "store": False,
                                }
                            )
                            if self._reasoning_effort:
                                kwargs["reasoning"] = {"effort": self._reasoning_effort}
                            response = client.responses.create(**kwargs)
                            last_response = _extract_response_text(response)
                        else:
                            kwargs.update(
                                {
                                    "model": self.model,
                                    "messages": messages,
                                    "temperature": self.temperature,
                                }
                            )
                            extra_body = {}
                            if self._thinking_mode is not None:
                                extra_body["enable_thinking"] = self._thinking_mode
                            if self._reasoning_effort:
                                extra_body["reasoning_effort"] = self._reasoning_effort
                            if self._chat_template_kwargs:
                                extra_body.setdefault("chat_template_kwargs", self._chat_template_kwargs)
                            if extra_body:
                                kwargs["extra_body"] = extra_body
                            response = client.chat.completions.create(**kwargs)
                            last_response = (response.choices[0].message.content or "") if response.choices else ""
                        record_usage(self.output_dir, "generation", getattr(response, "usage", None))
                        # 添加请求间延迟，避免突发并发导致 API 限流
                        if self.request_delay > 0:
                            time.sleep(self.request_delay)
                        break
                    except APIError as exc:
                        if self._maybe_toggle_thinking(exc):
                            continue
                        if attempt >= self.max_retries:
                            LOGGER.warning(
                                "Generation request failed after %d attempts [gen]: model=%s; messages=%s; error=%s; using fallback rules",
                                attempt,
                                self.model,
                                messages,
                                exc,
                            )
                            record_failure(
                                output_dir=self.output_dir,
                                stage="generation",
                                model=self.model,
                                attempts=attempt,
                                error=str(exc),
                                prompt=prompt_text,
                                response=last_response,
                                round_idx=None,
                                rule=None,
                                sample_idx=None,
                            )
                            return self._fallback(expected)
                        LOGGER.debug(
                            "Generation request failed (attempt %d/%d) [gen]: model=%s; error=%s; backing off %.1fs",
                            attempt,
                            self.max_retries,
                            self.model,
                            exc,
                            backoff,
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 2, self.max_backoff)
                        continue
                else:  # pragma: no cover
                    LOGGER.warning("Generation request exhausted retries unexpectedly; using fallback")
                    record_failure(
                        output_dir=self.output_dir,
                        stage="generation",
                        model=self.model,
                        attempts=self.max_retries,
                        error="exhausted retries",
                        prompt=prompt_text,
                        response=last_response,
                        round_idx=None,
                        rule=None,
                        sample_idx=None,
                    )
                    return self._fallback(expected)
                if self._use_responses_api:
                    content = last_response
                else:
                    if not response.choices:
                        LOGGER.warning("Remote model returned no choices; falling back")
                        record_failure(
                            output_dir=self.output_dir,
                            stage="generation",
                            model=self.model,
                            attempts=self.max_retries,
                            error="no choices returned",
                            prompt=prompt_text,
                            response=str(response),
                            round_idx=None,
                            rule=None,
                            sample_idx=None,
                        )
                        return self._fallback(expected)
                    content = response.choices[0].message.content or ""
        except (AuthenticationError, APIError) as exc:
            LOGGER.error("Model request failed: %s", exc)
            record_failure(
                output_dir=self.output_dir,
                stage="generation",
                model=self.model,
                attempts=self.max_retries,
                error=str(exc),
                prompt=prompt_text,
                response=last_response,
                round_idx=None,
                rule=None,
                sample_idx=None,
            )
            return self._fallback(expected)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Unexpected generation failure: %s", exc)
            record_failure(
                output_dir=self.output_dir,
                stage="generation",
                model=self.model,
                attempts=self.max_retries,
                error=str(exc),
                prompt=prompt_text,
                response=last_response,
                round_idx=None,
                rule=None,
                sample_idx=None,
            )
            return self._fallback(expected)

        rules = extract_hypotheses(content, expected)
        LOGGER.info("Model produced %d hypotheses", len(rules))
        if not rules:
            LOGGER.warning("Hypothesis extraction failed; using fallback hypotheses")
            record_failure(
                output_dir=self.output_dir,
                stage="generation",
                model=self.model,
                attempts=self.max_retries,
                error="hypothesis extraction returned 0",
                prompt=prompt_text,
                response=content or last_response,
                round_idx=None,
                rule=None,
                sample_idx=None,
            )
            return self._fallback(expected)
        return rules

    def _fallback(self, expected: int) -> List[str]:
        LOGGER.warning("Using fallback hypotheses due to generation failure")
        if expected <= 0:
            return []
        rules = []
        for _ in range(expected):
            rule = FALLBACK_RULES[self._fallback_index % len(FALLBACK_RULES)]
            self._fallback_index += 1
            rules.append(rule)
        return rules

    def _maybe_toggle_thinking(self, error: Exception) -> bool:
        message = str(error).lower()
        if "enable_thinking" not in message:
            return False
        if "must be set to false" in message and self._thinking_mode is True:
            LOGGER.info("模型 %s 要求关闭思考模式，自动改为 enable_thinking=false", self.model_name)
            self._thinking_mode = False
            return True
        if "unknown parameter" in message and self._thinking_mode is not None:
            LOGGER.info("模型 %s 不支持 enable_thinking，自动移除该参数", self.model_name)
            self._thinking_mode = None
            return True
        return False
