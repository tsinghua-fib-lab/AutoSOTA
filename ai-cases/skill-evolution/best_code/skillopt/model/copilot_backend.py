"""GitHub Copilot CLI chat backend.

Drives the locally installed `copilot` CLI as a chat model so a full SkillOpt
run (optimizer *and* target) can execute with no separate provider API key.
Inference still uses the GitHub Copilot cloud service through the operator's
existing subscription and OS credential store.

The CLI is a single-shot agent, not a chat-completions endpoint, so `system`
and `user` are composed into one prompt. Output is captured as JSONL
(`--output-format json`) and the `assistant.message` content is concatenated;
the plain-text/`--silent` modes do not reliably stream to stdout on all
platforms.

The CLI does not report token usage, so usage counters are reported as zeros.
Cost/quota is governed by the Copilot subscription rather than per-token
billing, so this does not lose billable accounting.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from .backend_config import get_copilot_chat_config
from .common import CompatAssistantMessage, TokenTracker

tracker = TokenTracker()

_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def build_copilot_subprocess_env(home: str = "") -> dict[str, str]:
    """Build a child environment without inherited unattended-tool approval."""
    env = os.environ.copy()
    env.pop("COPILOT_ALLOW_ALL", None)
    if home:
        env["COPILOT_HOME"] = home
    return env


def _compose_prompt(system: str, user: str) -> str:
    system = (system or "").strip()
    user = (user or "").strip()
    if system and user:
        return f"{system}\n\n---\n\n{user}"
    return system or user


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):  # OpenAI multi-part content
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    part_type = part.get("type") if isinstance(part, dict) else type(part).__name__
                    raise NotImplementedError(
                        "copilot_chat supports only text message parts; "
                        f"received unsupported multipart type {part_type!r}"
                    )
                text_parts.append(str(part.get("text", "")))
            content = "\n".join(text_parts)
        elif content is not None and not isinstance(content, str):
            raise TypeError(
                "copilot_chat message content must be a string, a list of text parts, or None"
            )
        content = str(content or "").strip()
        if not content:
            continue
        role = str(message.get("role", "user")).strip().lower()
        parts.append(content if role == "user" else f"[{role}]\n{content}")
    return "\n\n---\n\n".join(parts)


def parse_copilot_jsonl(raw: str) -> str:
    """Concatenate assistant text from a Copilot JSONL event stream."""
    parts: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "assistant.message":
            data = obj.get("data")
            if not isinstance(data, dict):
                continue
            content = data.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return "\n".join(parts).strip()


def _invoke(prompt: str, *, model: str, timeout: float | None) -> str:
    config = get_copilot_chat_config()
    cmd = [
        str(config["path"]),
        "-p",
        prompt,
        "--output-format",
        "json",
        "--stream",
        "off",
        "--no-color",
        "--log-level",
        "none",
        "--disable-builtin-mcps",
        "--no-custom-instructions",
        # A chat backend must behave like a completions endpoint, not an agent.
        # An empty allowlist removes built-in read/shell/write/web tools.
        "--available-tools=",
    ]
    chosen = model or str(config.get("model") or "")
    if chosen:
        cmd.extend(["--model", chosen])

    home = str(config.get("home") or "")
    env = build_copilot_subprocess_env(home)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout if timeout else float(config["timeout"]),
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        detail = ((proc.stderr or proc.stdout) or "").strip()[:4000]
        raise RuntimeError(f"Copilot CLI failed with exit code {proc.returncode}: {detail}")
    return parse_copilot_jsonl(proc.stdout or "")


def _chat_impl(
    prompt: str,
    retries: int,
    stage: str,
    *,
    model: str = "",
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            text = _invoke(prompt, model=model, timeout=timeout)
            if not text:
                raise RuntimeError("Copilot CLI returned an empty response")
            tracker.record(stage, 0, 0)
            return text, dict(_ZERO_USAGE)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max(1, retries) - 1:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Copilot CLI chat failed after {max(1, retries)} retries: {last_err}")


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del max_completion_tokens, reasoning_effort
    config = get_copilot_chat_config()
    return _chat_impl(
        _compose_prompt(system, user),
        retries,
        stage,
        model=str(config.get("optimizer_model") or ""),
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del max_completion_tokens, reasoning_effort
    config = get_copilot_chat_config()
    return _chat_impl(
        _compose_prompt(system, user),
        retries,
        stage,
        model=str(config.get("target_model") or ""),
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[str | CompatAssistantMessage, dict[str, int]]:
    del max_completion_tokens, reasoning_effort
    _reject_unsupported_tools(tools, tool_choice)
    config = get_copilot_chat_config()
    text, usage = _chat_impl(
        _messages_to_prompt(messages),
        retries,
        stage,
        model=str(config.get("optimizer_model") or ""),
        timeout=timeout,
    )
    return (CompatAssistantMessage(content=text) if return_message else text), usage


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[str | CompatAssistantMessage, dict[str, int]]:
    del max_completion_tokens, reasoning_effort
    _reject_unsupported_tools(tools, tool_choice)
    config = get_copilot_chat_config()
    text, usage = _chat_impl(
        _messages_to_prompt(messages),
        retries,
        stage,
        model=str(config.get("target_model") or ""),
        timeout=timeout,
    )
    return (CompatAssistantMessage(content=text) if return_message else text), usage


def _reject_unsupported_tools(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> None:
    if tools or tool_choice is not None:
        raise NotImplementedError(
            "copilot_chat does not support caller-supplied tools or tool_choice; "
            "use a tool-capable chat backend for this environment"
        )


def get_token_summary() -> dict[str, dict[str, int]]:
    """Per-stage usage, like every other backend.

    The Copilot CLI reports no token counts, so the token fields stay zero, but
    the *call* counts are real and belong in the run's token snapshots.
    """
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()
