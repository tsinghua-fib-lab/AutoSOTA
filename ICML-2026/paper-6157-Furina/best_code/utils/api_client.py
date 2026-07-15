#!/usr/bin/env python3
"""
Shared LLM provider classes and agent defaults for utils scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    env_prefix: str
    default_base_url: str | None = None
    fallback_to_proxy: bool = True


@dataclass(frozen=True)
class AgentSettings:
    name: str
    provider: str
    model: str



import random as _random

def backoff_sleep(attempt: int, base: float = 1.0, max_wait: float = 60.0) -> float:
    """Exponential backoff with jitter. Returns actual wait time."""
    wait = min(base * (2 ** (attempt - 1)) + _random.random(), max_wait)
    import time
    time.sleep(wait)
    return wait

class BaseProvider:
    settings: ProviderSettings

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    def _env(self, suffix: str) -> str | None:
        return os.getenv(f"{self.settings.env_prefix}_{suffix}")

    def resolve_api_key(self) -> str:
        api_key = self._env("API_KEY")
        if api_key:
            return api_key

        if self.settings.fallback_to_proxy:
            proxy_key = os.getenv("PROXY_API_KEY")
            if proxy_key:
                return proxy_key

        raise ValueError(
            f"Missing {self.settings.env_prefix}_API_KEY for provider '{self.settings.name}'."
        )

    def resolve_base_url(self) -> str:
        base_url = self._env("BASE_URL")
        if base_url:
            return base_url

        if self.settings.fallback_to_proxy:
            proxy_base_url = os.getenv("PROXY_BASE_URL")
            if proxy_base_url:
                return proxy_base_url

        if self.settings.default_base_url:
            return self.settings.default_base_url

        raise ValueError(
            f"Missing {self.settings.env_prefix}_BASE_URL for provider '{self.settings.name}'."
        )

    def get_client(self) -> OpenAI:
        # Clear proxy env vars that interfere with API connections
        for _v in ("ALL_PROXY", "all_proxy",
                    "HTTP_PROXY", "HTTPS_PROXY",
                    "http_proxy", "https_proxy"):
            os.environ.pop(_v, None)
        return OpenAI(
            api_key=self.resolve_api_key(),
            base_url=self.resolve_base_url(),
        )


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="openai",
                env_prefix="OPENAI",
                default_base_url="https://api.openai.com/v1",
                fallback_to_proxy=True,
            )
        )


class DeepSeekProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="deepseek",
                env_prefix="DEEPSEEK",
                default_base_url="https://api.deepseek.com",
                fallback_to_proxy=True,
            )
        )


class GeminiProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="gemini",
                env_prefix="GEMINI",
                default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                fallback_to_proxy=True,
            )
        )


class GrokProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="grok",
                env_prefix="GROK",
                default_base_url="https://api.groq.com/openai/v1",
                fallback_to_proxy=True,
            )
        )


class ClaudeProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="claude",
                env_prefix="CLAUDE",
                default_base_url="https://api.anthropic.com",
                fallback_to_proxy=True,
            )
        )


class ProxyProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(
            ProviderSettings(
                name="proxy",
                env_prefix="PROXY",
                default_base_url=None,
                fallback_to_proxy=False,
            )
        )


PROVIDERS: dict[str, BaseProvider] = {
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "gemini": GeminiProvider(),
    "grok": GrokProvider(),
    "claude": ClaudeProvider(),
    "proxy": ProxyProvider(),
    "relay": ProxyProvider(),
}

MODEL_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", ),
    "deepseek": ("deepseek",),
    "gemini": ("gemini",),
    "grok": ("grok",),
    "claude": ("claude",),
}


AGENT_DEFAULTS: dict[str, AgentSettings] = {
    "task_plan": AgentSettings(
        name="task_plan",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
    "probe_reasoning": AgentSettings(
        name="probe_reasoning",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
    "probe_optimizer": AgentSettings(
        name="probe_optimizer",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
    "probe_generator": AgentSettings(
        name="probe_generator",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
    "probe_responder": AgentSettings(
        name="probe_responder",
        provider="proxy",
        model="meta-llama-3-8b",
    ),
    "redteam_synthesizer": AgentSettings(
        name="redteam_synthesizer",
        provider="deepseek",
        model="deepseek-v4-pro",  # upgraded from v4-flash for better reasoning quality
    ),
    "redteam_judge": AgentSettings(
        name="redteam_judge",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
}


def get_provider(provider_name: str) -> BaseProvider:
    normalized = provider_name.strip().lower()
    if normalized not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name}")
    return PROVIDERS[normalized]


def infer_provider_from_model(model_name: str) -> str | None:
    normalized = (model_name or "").strip().lower()
    for provider_name, prefixes in MODEL_PROVIDER_PREFIXES.items():
        if normalized.startswith(prefixes):
            return provider_name
    return None


def get_provider_name_for_model(model_name: str, fallback: str = "openai") -> str:
    return infer_provider_from_model(model_name) or fallback


def get_api_env_var_name_for_provider(provider_name: str) -> str:
    provider = get_provider(provider_name)
    return f"{provider.settings.env_prefix}_API_KEY"


def get_base_url_for_provider(provider_name: str) -> str:
    provider = get_provider(provider_name)
    configured_base_url = os.getenv(f"{provider.settings.env_prefix}_BASE_URL")
    if configured_base_url:
        return configured_base_url

    if provider.settings.fallback_to_proxy:
        proxy_base_url = os.getenv("PROXY_BASE_URL")
        if proxy_base_url:
            return proxy_base_url

    return provider.settings.default_base_url or ""


def get_agent_settings(agent_name: str, model_override: str | None = None) -> AgentSettings:
    if agent_name not in AGENT_DEFAULTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    default_settings = AGENT_DEFAULTS[agent_name]
    if not model_override:
        return default_settings

    provider = default_settings.provider.strip().lower()
    if provider == "auto":
        provider = infer_provider_from_model(model_override) or "openai"

    return AgentSettings(
        name=default_settings.name,
        provider=provider,
        model=model_override,
    )


def get_client_for_provider(provider_name: str) -> OpenAI:
    return get_provider(provider_name).get_client()


def get_client_for_agent(agent_name: str, model_override: str | None = None) -> OpenAI:
    settings = get_agent_settings(agent_name, model_override=model_override)
    return get_client_for_provider(settings.provider)


def get_model_for_agent(agent_name: str, model_override: str | None = None) -> str:
    return get_agent_settings(agent_name, model_override=model_override).model
