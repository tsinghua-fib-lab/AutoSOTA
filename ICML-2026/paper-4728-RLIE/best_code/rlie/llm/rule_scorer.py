import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from ollama import Client as OllamaClient
from openai import APITimeoutError, APIError, AsyncOpenAI, OpenAI, RateLimitError
from tqdm.auto import tqdm

from rlie.llm.prompt import BasePrompt
from rlie.datasets.data_loader import _normalize_label
from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.llm.api_client_pool import APIClientPool
from rlie.llm.parsing import (
    allowed_answers,
    append_answer_constraint,
    derive_label_aliases,
    extract_response_text,
    parse_ollama_model,
    parse_prediction_response,
    should_use_responses_api,
)
from rlie.llm import ruleset_eval
from rlie.utils import concurrent_config
from rlie.llm.parse_failure_recorder import record_attempt, record_failure, record_usage


LOGGER = logging.getLogger("rlie.rule_scorer")


class RuleScorer:
    """Score rule applicability with the configured LLM backend."""

    def __init__(
        self,
        config: Dict,
        prompt: BasePrompt,
        label_mapping: Dict[str, int],
        *,
        label_values: Optional[Sequence[str]] = None,
        disable_progress: Optional[bool] = None,
    ):
        if "model" not in config:
            raise ValueError("Evaluation config must include a model name")

        self.prompt = prompt
        self.model_name = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.request_timeout = float(config.get("request_timeout", 120))
        self.num_predict = int(config.get("num_predict", 64))
        self.output_dir = config.get("output_dir")

        # 并发度优先级：_concurrent_budget（注入） > max_concurrent_per_key（新） > max_concurrent（旧，已弃用）
        if "_concurrent_budget" in config:
            self.max_concurrent = max(1, int(config["_concurrent_budget"]))
            LOGGER.debug("使用注入的并发预算: %d", self.max_concurrent)
        elif "max_concurrent_per_key" in config:
            # 如果配置了 max_concurrent_per_key，需要根据 API key 数量计算
            # 但此时我们可能还不知道 key 数量，先记录一个占位值，后面按真实 key 数调整
            per_key = int(config["max_concurrent_per_key"])
            self.max_concurrent = max(1, per_key)
            LOGGER.debug("使用 max_concurrent_per_key 占位值: %d (待按真实 key 数调整)", self.max_concurrent)
        else:
            # 向后兼容：使用旧的 max_concurrent
            self.max_concurrent = max(
                1,
                int(
                    config.get(
                        "max_concurrent",
                        config.get("max_workers", 1),
                    )
                ),
            )
            LOGGER.debug("使用旧的 max_concurrent 配置: %d", self.max_concurrent)
        self.max_retries = max(1, int(config.get("max_retries", 3)))
        self.max_parse_retries = max(1, int(config.get("max_parse_retries", self.max_retries)))
        self.min_backoff = float(config.get("min_backoff", 1.0))
        self.max_backoff = float(config.get("max_backoff", 30.0))
        self.request_delay = float(config.get("request_delay", 0.0))
        # 全局关闭进度条，避免多线程刷新干扰；如需打开可传入 disable_progress=False
        if disable_progress is None:
            self._disable_tqdm = True
        else:
            self._disable_tqdm = bool(disable_progress)
        # thinking tri-state:
        #   None  -> do not send enable_thinking
        #   True  -> force enable_thinking=true
        #   False -> force enable_thinking=false
        disable_flag = config.get("disable_thinking")
        if disable_flag is True:
            self._thinking_mode = False
        elif disable_flag is False:
            self._thinking_mode = True
        else:
            self._thinking_mode = None
        self._reasoning_effort = config.get("reasoning_effort")
        self._chat_template_kwargs = config.get("chat_template_kwargs")
        self._use_responses_api = should_use_responses_api(
            self.model_name,
            bool(config.get("use_responses_api", False)),
        )

        self._provider = "ollama" if self.model_name.lower().startswith("ollama") else "remote"
        client_cfg = config.get("client", {})

        if self._provider == "ollama":
            host = client_cfg.get("base_url", "http://localhost:11434")
            self.model = parse_ollama_model(self.model_name)
            self.client = OllamaClient(host=host, timeout=self.request_timeout)
            self._client_pool = None
        else:
            base_url = client_cfg.get("base_url")
            api_keys = client_cfg.get("api_keys") or client_cfg.get("api_key")
            if isinstance(api_keys, str):
                api_keys = [api_keys]
            if not base_url or not api_keys:
                raise ValueError("Remote evaluation requires client.base_url and api_key(s)")

            if "_concurrent_budget" not in config and "max_concurrent_per_key" in config:
                retry_overhead = float(config.get("retry_overhead", 0.15))
                self.max_concurrent = concurrent_config.calculate_total_concurrent_limit(
                    config,
                    len(api_keys),
                    retry_overhead,
                )
                LOGGER.debug(
                    "按真实 key 数调整 max_concurrent=%d (keys=%d, retry_overhead=%.2f)",
                    self.max_concurrent,
                    len(api_keys),
                    retry_overhead,
                )

            self.model = self.model_name
            self._base_url = base_url
            if len(api_keys) == 1:
                key = api_keys[0]
                self.client = OpenAI(base_url=base_url, api_key=key)
                self.async_client = AsyncOpenAI(base_url=base_url, api_key=key)
                self._client_pool = None
            else:
                self._client_pool = APIClientPool(base_url, api_keys)
                self.client = None
                self.async_client = None

        # All tasks treated as binary for parsing; not applicable will be counted as negative.
        self._use_three_state = False
        self.metric_labels = sorted({int(idx) for idx in label_mapping.values()})
        self._use_async_remote = bool(config.get("use_async_remote", False))

        self._build_label_sets(label_mapping, label_values)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        for logger_name in ("openai", "openai._base_client", "openai._http"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            logging.getLogger(logger_name).propagate = False

        LOGGER.info(
            "Rule checker initialized with %s model %s (max_concurrent=%d)",
            "local" if self._provider == "ollama" else "remote",
            self.model,
            self.max_concurrent,
        )

    def _build_label_sets(
        self,
        label_mapping: Dict[str, int],
        label_values: Optional[Sequence[str]] = None,
    ):
        self.canonical_positive: List[str] = []
        self.canonical_negative: List[str] = []

        if label_values and len(label_values) >= 2:
            pos_value = label_values[0]
            neg_value = label_values[1]
        else:
            # fallback to mapping order
            positives = [text for text, idx in label_mapping.items() if idx == 1]
            negatives = [text for text, idx in label_mapping.items() if idx == 0]
            pos_value = positives[0] if positives else "positive"
            neg_value = negatives[0] if negatives else "negative"

        self.canonical_positive = [pos_value]
        self.canonical_negative = [neg_value]

        self.positive_labels = { _normalize_label(pos_value) }
        self.negative_labels = { _normalize_label(neg_value) }

        self.positive_synonyms = {"truthful", "yes", "positive", "1"}
        self.negative_synonyms = {"deceptive", "no", "negative", "0"}
        self.neutral_synonyms = {
            "not applicable",
            "na",
            "n a",
            "neutral",
            "unknown",
            "无法判断",
            "不适用",
            "not sure",
        }

        for label_text in self.positive_labels:
            self.positive_synonyms.update(derive_label_aliases(label_text))
        for label_text in self.negative_labels:
            self.negative_synonyms.update(derive_label_aliases(label_text))

    def _format_messages(self, rule: str, samples: pd.DataFrame, idx: int) -> List[Dict[str, str]]:
        messages = self.prompt.inference({rule: None}, samples, idx)
        return messages

    def _format_multiple_messages(
        self,
        hypotheses: Sequence[str],
        samples: pd.DataFrame,
        idx: int,
    ) -> List[Dict[str, str]]:
        messages = self.prompt.multiple_hypotheses_inference(list(hypotheses), samples, idx)
        return append_answer_constraint(
            messages,
            allowed_answers(
                self.canonical_positive,
                self.canonical_negative,
                include_not_applicable=False,
            ),
        )

    def _format_multiple_with_label_messages(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        idx: int,
        predicted_label: str,
    ) -> List[Dict[str, str]]:
        messages = self.prompt.multiple_hypotheses_inference_with_linear_regression_and_label(
            list(weighted_hypotheses),
            bias,
            predicted_label,
            samples,
            idx,
        )
        return append_answer_constraint(
            messages,
            allowed_answers(
                self.canonical_positive,
                self.canonical_negative,
                include_not_applicable=False,
            ),
        )

    def score_rule(
        self,
        rule: str,
        samples: pd.DataFrame,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        total = len(samples)
        if total == 0:
            return []

        def _evaluate(row_idx: int) -> int:
            messages = self._format_messages(rule, samples, row_idx)
            return self._call_and_parse(
                messages,
                context={"rule": rule, "sample_idx": row_idx},
            )

        indices = list(range(total))
        results: List[int] = []

        if self._provider == "remote" and self._use_async_remote:
            return self._score_rule_remote(rule, samples, indices, progress_callback)

        if self.max_concurrent == 1:
            for idx in indices:
                results.append(_evaluate(idx))
                if progress_callback is not None:
                    progress_callback(1)
            return results

        # 滑动窗口模式：完成一个处理一个，避免被慢请求阻塞整批
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            results = [0] * len(indices)
            future_to_pos = {
                executor.submit(_evaluate, idx): pos
                for pos, idx in enumerate(indices)
            }
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                try:
                    results[pos] = future.result()
                except Exception:
                    results[pos] = 0
                if progress_callback is not None:
                    progress_callback(1)
        return results

    def _call_and_parse(self, messages: List[Dict[str, str]], context: Optional[Dict[str, object]] = None) -> int:
        """Call the model and parse the result; retry on parsing failures."""
        record_attempt(self.output_dir, "evaluation")
        prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
        last_response: str = ""
        last_error: str = ""
        for attempt in range(1, self.max_parse_retries + 1):
            try:
                response = self._call_model(messages)
                last_response = response or ""
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                if attempt >= self.max_parse_retries:
                    LOGGER.warning(
                        "Model call failed after %d attempts; returning not applicable. Error: %s",
                        attempt,
                        exc,
                    )
                    record_failure(
                        output_dir=self.output_dir,
                        stage="evaluation",
                        model=self.model,
                        attempts=attempt,
                        error=last_error,
                        prompt=prompt_text,
                        response=last_response,
                        round_idx=None,
                        rule=(context or {}).get("rule"),
                        sample_idx=(context or {}).get("sample_idx"),
                    )
                    return 0
                LOGGER.warning(
                    "Model call failed (attempt %d/%d); retrying once more. Error: %s",
                    attempt,
                    self.max_parse_retries,
                    exc,
                )
                continue
            try:
                return parse_prediction_response(
                    response,
                    self.canonical_positive,
                    self.canonical_negative,
                    self.positive_synonyms,
                    self.negative_synonyms,
                    self.neutral_synonyms,
                )
            except ValueError as exc:
                last_error = str(exc)
                if attempt >= self.max_parse_retries:
                    LOGGER.warning(
                        "Parse failed after %d attempts; returning not applicable. Error: %s. Full response: %s",
                        attempt,
                        exc,
                        response,
                    )
                    record_failure(
                        output_dir=self.output_dir,
                        stage="evaluation",
                        model=self.model,
                        attempts=attempt,
                        error=last_error,
                        prompt=prompt_text,
                        response=response or last_response,
                        round_idx=None,
                        rule=(context or {}).get("rule"),
                        sample_idx=(context or {}).get("sample_idx"),
                    )
                    return 0
                LOGGER.warning(
                    "Parse failed (attempt %d/%d); retrying once more. Full response: %s",
                    attempt,
                    self.max_parse_retries,
                    response,
                )
        return 0

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        backoff = self.min_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                if self._provider == "ollama":
                    response = self.client.chat(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.num_predict,
                        },
                        think=False,
                    )
                    message = response.get("message") or {}
                    return message.get("content", "")

                kwargs = {
                    "timeout": self.request_timeout,
                }

                # 根据重试次数选择客户端：首次尝试用常规 client，重试时轮换 key
                if attempt == 1:
                    client = self._client_pool.get_sync_client() if self._client_pool else self.client
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
                    content = extract_response_text(response)
                else:
                    kwargs.update(
                        {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                        }
                    )
                    extra_body = {}
                    # DeepSeek 官方 API 对未支持的额外参数较为严格，直接跳过 extra_body
                    base_url = self._base_url
                    if not (base_url and str(base_url).startswith("https://api.deepseek.com")):
                        if self._thinking_mode is not None:
                            extra_body["enable_thinking"] = self._thinking_mode
                        if self._reasoning_effort:
                            extra_body["reasoning_effort"] = self._reasoning_effort
                        if self._chat_template_kwargs:
                            extra_body.setdefault("chat_template_kwargs", self._chat_template_kwargs)
                        if extra_body:
                            kwargs["extra_body"] = extra_body
                    response = client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                record_usage(self.output_dir, "evaluation", getattr(response, "usage", None))
                # 添加请求间延迟，避免突发并发导致 API 限流
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return content
            except APIError as exc:
                if self._maybe_toggle_thinking(exc):
                    continue
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "Model request failed after %d attempts [sync]: model=%s; messages=%s; error=%s",
                        attempt,
                        self.model,
                        messages,
                        exc,
                    )
                    return "Final answer: not applicable"
                LOGGER.debug(
                    "Model request failed (attempt %d/%d) [sync]: model=%s; error=%s; messages=%s; backing off %.1fs",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                    messages,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue
            except JSONDecodeError as exc:  # API 返回非 JSON 响应
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "JSON decode failed after %d attempts [sync]: model=%s; error=%s; "
                        "这通常表示 API 服务返回了错误页面而非 JSON，请检查 API 服务状态",
                        attempt,
                        self.model,
                        exc,
                    )
                    return "Final answer: not applicable"

                # 第一次失败用 debug，避免污染日志
                log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                log_func(
                    "JSON decode failed (attempt %d/%d) [sync]: model=%s; error=%s",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                )
                # JSON 错误通常是服务器暂时性问题，使用更长退避时间
                backoff = min(backoff * 3, self.max_backoff)
                time.sleep(backoff)
                continue
            except Exception as exc:  # pragma: no cover
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "Unexpected error after %d attempts [sync]: model=%s; messages=%s; error=%s",
                        attempt,
                        self.model,
                        messages,
                        exc,
                    )
                    return "Final answer: not applicable"

                # 第一次失败用 debug，避免污染日志
                log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                log_func(
                    "Unexpected error (attempt %d/%d) [sync]: model=%s; error=%s",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue

        LOGGER.error("Model request failed after retries")
        LOGGER.warning("Model request exhausted retries; falling back to not applicable")
        return "Final answer: not applicable"

    def _run_async(self, coro):
        if self._provider != "remote":
            return asyncio.run(coro)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _execute():
            async with AsyncOpenAI(**self._async_client_kwargs) as client:
                self.async_client = client
                try:
                    return await coro
                finally:
                    self.async_client = None

        try:
            result = loop.run_until_complete(_execute())
            loop.run_until_complete(loop.shutdown_asyncgens())
            return result
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _score_rule_remote(
        self,
        rule: str,
        samples: pd.DataFrame,
        indices: List[int],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        async def _gather() -> List[int]:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            results: List[int] = [0 for _ in indices]

            async def _evaluate(position: int, row_idx: int):
                messages = self._format_messages(rule, samples, row_idx)
                results[position] = await self._call_and_parse_async(
                    messages,
                    semaphore,
                    context={"rule": rule, "sample_idx": row_idx},
                )
                if progress_callback is not None:
                    progress_callback(1)

            # 滑动窗口模式：一次性创建所有任务，完成一个处理一个
            tasks = [
                asyncio.create_task(_evaluate(pos, row_idx))
                for pos, row_idx in enumerate(indices)
            ]
            for coro in asyncio.as_completed(tasks):
                await coro
            return results

        return self._run_async(_gather())

    async def _call_model_async(
        self,
        messages: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
    ) -> str:
        backoff = self.min_backoff
        for attempt in range(1, self.max_retries + 1):
            async with semaphore:  # 移到循环内，失败时释放 semaphore
                try:
                    kwargs = {
                        "timeout": self.request_timeout,
                    }

                    # 根据重试次数选择客户端：首次尝试用常规 client，重试时轮换 key
                    if attempt == 1:
                        client = self._client_pool.get_async_client() if self._client_pool else self.async_client
                    else:
                        client = (
                            self._client_pool.get_async_client_for_retry(attempt)
                            if self._client_pool
                            else self.async_client
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
                        response = await client.responses.create(**kwargs)
                        content = extract_response_text(response)
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
                        response = await client.chat.completions.create(**kwargs)
                        if not response.choices:
                            LOGGER.error("Remote evaluation returned no choices")
                            return "Final answer: not applicable"
                        content = response.choices[0].message.content or ""
                    record_usage(self.output_dir, "evaluation", getattr(response, "usage", None))
                    # 添加请求间延迟，避免突发并发导致 API 限流
                    if self.request_delay > 0:
                        await asyncio.sleep(self.request_delay)
                    return content
                except (RateLimitError, APITimeoutError, APIError) as exc:
                    if isinstance(exc, APIError) and self._maybe_toggle_thinking(exc):
                        continue

                    # 第一次失败用 debug，避免污染日志
                    log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                    log_func(
                        "Remote request failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt == self.max_retries:
                        break
                    # 释放 semaphore 后再 sleep，避免占用并发槽
                except JSONDecodeError as exc:  # API 返回非 JSON 响应
                    if attempt == self.max_retries:
                        LOGGER.error(
                            "JSON decode failed after %d attempts [async]: model=%s; error=%s; "
                            "这通常表示 API 服务返回了错误页面而非 JSON，请检查 API 服务状态",
                            attempt,
                            self.model,
                            exc,
                        )
                        return "Final answer: not applicable"

                    # 第一次失败用 debug，避免污染日志
                    log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                    log_func(
                        "JSON decode failed (attempt %d/%d) [async]: model=%s; error=%s",
                        attempt,
                        self.max_retries,
                        self.model,
                        exc,
                    )
                    # JSON 错误通常是服务器暂时性问题，使用更长退避时间（在外部 sleep）
                    if attempt < self.max_retries:
                        await asyncio.sleep(backoff * 3)
                        backoff = min(backoff * 3, self.max_backoff)
                        continue
                except Exception as exc:  # pylint: disable=broad-except
                    if attempt == self.max_retries:
                        LOGGER.error("Unexpected remote error after %d attempts: %s", attempt, exc)
                        return "Final answer: not applicable"

                    # 第一次失败用 debug，避免污染日志
                    log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                    log_func("Unexpected remote error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                    # 继续到外部的 sleep

            # 在 semaphore 外部 sleep，释放并发槽
            if attempt < self.max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

        LOGGER.error("Remote evaluation exhausted retries; returning fallback")
        LOGGER.warning("Async model request exhausted retries; falling back to not applicable")
        return "Final answer: not applicable"

    async def _call_and_parse_async(
        self,
        messages: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
        context: Optional[Dict[str, object]] = None,
    ) -> int:
        record_attempt(self.output_dir, "evaluation")
        prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
        last_response: str = ""
        last_error: str = ""
        for attempt in range(1, self.max_parse_retries + 1):
            response = await self._call_model_async(messages, semaphore)
            last_response = response or ""
            try:
                return parse_prediction_response(
                    response,
                    self.canonical_positive,
                    self.canonical_negative,
                    self.positive_synonyms,
                    self.negative_synonyms,
                    self.neutral_synonyms,
                )
            except ValueError as exc:
                last_error = str(exc)
                if attempt >= self.max_parse_retries:
                    LOGGER.warning(
                        "Async parse failed after %d attempts; returning not applicable. Error: %s. Full response: %s",
                        attempt,
                        exc,
                        response,
                    )
                    record_failure(
                        output_dir=self.output_dir,
                        stage="evaluation",
                        model=self.model,
                        attempts=attempt,
                        error=last_error,
                        prompt=prompt_text,
                        response=last_response,
                        round_idx=None,
                        rule=(context or {}).get("rule"),
                        sample_idx=(context or {}).get("sample_idx"),
                    )
                    return 0
                LOGGER.warning(
                    "Async parse failed (attempt %d/%d); retrying once more. Full response: %s",
                    attempt,
                    self.max_parse_retries,
                    response,
                )
        return 0

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

    def batch_score_serial(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        return ruleset_eval.batch_score_serial(self, rules, samples, labels)

    def batch_score_parallel(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
        max_rule_workers: Optional[int] = None,
    ) -> List[List[int]]:
        return ruleset_eval.batch_score_parallel(
            self,
            rules,
            samples,
            labels,
            max_rule_workers=max_rule_workers,
        )

    def batch_score(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        return ruleset_eval.batch_score(self, rules, samples, labels)

    def evaluate_rule_set(
        self,
        hypotheses: Sequence[str],
        samples: pd.DataFrame,
    ) -> List[int]:
        return ruleset_eval.evaluate_rule_set(self, hypotheses, samples)

    def _score_ruleset_remote(
        self,
        hypotheses: Sequence[str],
        samples: pd.DataFrame,
        indices: List[int],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        return ruleset_eval._score_ruleset_remote(
            self,
            hypotheses,
            samples,
            indices,
            progress_callback=progress_callback,
        )

    def evaluate_rule_set_with_label(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        predicted_labels: Sequence[str],
    ) -> List[int]:
        return ruleset_eval.evaluate_rule_set_with_label(
            self,
            weighted_hypotheses,
            bias,
            samples,
            predicted_labels,
        )

    def _score_ruleset_with_label_remote(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        indices: List[int],
        predicted_labels: Sequence[str],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        return ruleset_eval._score_ruleset_with_label_remote(
            self,
            weighted_hypotheses,
            bias,
            samples,
            indices,
            predicted_labels,
            progress_callback=progress_callback,
        )

    def evaluate_rule_set_with_linear_regression(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
    ) -> List[int]:
        return ruleset_eval.evaluate_rule_set_with_linear_regression(
            self,
            weighted_hypotheses,
            bias,
            samples,
        )

    def _score_ruleset_with_linear_regression_remote(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        indices: List[int],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        return ruleset_eval._score_ruleset_with_linear_regression_remote(
            self,
            weighted_hypotheses,
            bias,
            samples,
            indices,
            progress_callback=progress_callback,
        )
