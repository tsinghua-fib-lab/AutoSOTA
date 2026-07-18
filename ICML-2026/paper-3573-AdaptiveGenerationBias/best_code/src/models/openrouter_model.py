# openrouter_model.py

from typing import List, Dict, Any, Optional
import os
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.configs import ModelConfig
from .model import APIModel


class OpenRouterModel(APIModel):
    """
    Thread-safe wrapper for OpenRouter's /chat/completions:
      - Per-thread requests.Session (no cross-thread sharing)
      - Connection pool sizing and retries for 429/5xx
      - Pass-through of arbitrary args (incl. `reasoning`)
      - Sensible defaults (temperature, max_tokens, timeout)
      - Optional ranking headers via env or args

    Env:
      OPENROUTER_API_KEY (required)
      OPENROUTER_REFERRER -> 'HTTP-Referer' header
      OPENROUTER_TITLE    -> 'X-Title' header
      OPENROUTER_BASE_URL -> default https://openrouter.ai/api/v1
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        # --- Endpoint & Auth ---
        self.base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")

        # Optional ranking headers (prefer env; fall back to args)
        self.http_referer: Optional[str] = os.environ.get(
            "OPENROUTER_REFERRER"
        ) or self.config.args.pop("http_referer", None)
        self.x_title: Optional[str] = os.environ.get("OPENROUTER_TITLE") or self.config.args.pop(
            "x_title", None
        )

        # Defaults similar to your Together wrapper
        self.config.args.setdefault("temperature", 0.0)
        self.config.args.setdefault("max_tokens", 600)

        # Transport tuning (pop so they don't end up in JSON)
        self._timeout: float = float(self.config.args.pop("_timeout", 60))
        self._pool_connections: int = int(self.config.args.pop("_pool_connections", 32))
        self._pool_maxsize: int = int(self.config.args.pop("_pool_maxsize", 128))
        self._retries_total: int = int(self.config.args.pop("_retries_total", 4))
        self._retries_backoff: float = float(self.config.args.pop("_retries_backoff", 0.5))
        # Default retry statuses: include 429 & common 5xx
        self._retries_statuses = self.config.args.pop(
            "_retries_statuses",
            [429, 500, 502, 503, 504],
        )

        # Base headers used for each per-thread session
        self._base_headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self._base_headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            self._base_headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            self._base_headers["X-Title"] = self.x_title

        # Thread-local storage for sessions
        self._tls = threading.local()

    # --- Session management (per-thread) ---

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(self._base_headers)

        # Configure retries and pool sizes
        retry = Retry(
            total=self._retries_total,
            backoff_factor=self._retries_backoff,
            status_forcelist=self._retries_statuses,
            allowed_methods=frozenset(["POST"]),  # we only POST here
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=self._pool_connections,
            pool_maxsize=self._pool_maxsize,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def _get_session(self) -> requests.Session:
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = self._build_session()
            self._tls.session = sess
        return sess

    # --- Prediction ---

    def _predict_call(self, input: List[Dict[str, Any]]) -> str:
        """
        Perform a single non-streaming chat.completions call and return the
        first candidate's text content as a string.
        Uses the openai library for reliable connection handling.
        """
        from openai import OpenAI

        if not self.api_key and "Authorization" not in self._base_headers:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Please export it or provide an Authorization header."
            )

        # Use a shared client instance for connection pooling
        if not hasattr(self, "_openai_client"):
            import httpx
            http_client = httpx.Client(
                timeout=self._timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
            self._openai_client = OpenAI(
                base_url=self.base_url.rstrip("/") + "/",
                api_key=self.api_key,
                timeout=self._timeout,
                max_retries=self._retries_total,
                http_client=http_client,
            )

        body_args = dict(self.config.args)
        for k in list(body_args.keys()):
            if k.startswith("_"):
                del body_args[k]

        if "max_output_tokens" in body_args:
            body_args["max_tokens"] = body_args.pop("max_output_tokens")

        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.name,
                messages=input,
                **body_args,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            raise RuntimeError(f"OpenRouter request failed: {e}") from e