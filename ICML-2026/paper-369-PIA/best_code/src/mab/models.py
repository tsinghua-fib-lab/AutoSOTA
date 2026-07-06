"""
LLM Client Wrapper - Unified API Interface
"""
import os
import time
import random
import traceback
from typing import Dict, Any, Optional

try:
    from openai import OpenAI, APIError, APIConnectionError
except ImportError:
    OpenAI = None
    APIError = Exception
    APIConnectionError = Exception

from ..core import get_model_config
from ..core.utils import setup_logger

logger = setup_logger("LLMClient")


class MockLLMClient:
    """Mock client for testing"""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        logger.info(f"Mock Client Initialized for {model_name}")

    def chat_completions_create(self, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Return random response"""
        return random.choice(["Option A", "Option B"])


class LLMClient:
    """
    Unified LLM client
    Supports multiple API sources and Mock mode
    """

    def __init__(self, source: str, model: str, mock: bool = False, max_retries: int = 2, retry_delay: int = 5):
        """
        Args:
            source: API source (siliconflow, ollama, dashscope, dmxapi)
            model: Model name
            mock: Whether to use mock mode
            max_retries: Maximum retry attempts
            retry_delay: Retry delay (seconds)
        """
        self.source = source
        self.model = model
        self.mock = mock
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if mock:
            self.client = MockLLMClient(model)
            self.is_mock = True
        else:
            if OpenAI is None:
                raise ImportError("openai library not installed, please install or use mock mode")

            config = get_model_config(source, model)
            self.base_url = config["BASE_URL"]
            self.api_key = config["API_KEY"]

            if self.api_key == "placeholder-key":
                logger.warning("Using placeholder API key, recommend configuring real key")

            # Add timeout to OpenAI client
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=60.0  # 60 second timeout
            )
            self.is_mock = False

    def call(self, messages: list, temperature: float = 0.0, max_tokens: int = 512, timeout: int = 360) -> str:
        """
        Call LLM API with automatic retry mechanism

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            timeout: Request timeout in seconds (passed to OpenAI client)

        Returns:
            Model response text
        """
        logger.debug(f"Calling LLM API: source={self.source}, model={self.model}, mock={self.is_mock}")
        logger.debug(f"Messages: {messages}")

        if self.is_mock:
            return self.client.chat_completions_create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Debug: Check response structure
                if not hasattr(response, 'choices'):
                    logger.error(f"Response has no 'choices' attribute")
                    logger.error(f"  Response type: {type(response)}")
                    logger.error(f"  Response dir: {dir(response)}")
                    raise ValueError("API response missing 'choices' attribute")

                if not response.choices:
                    logger.error(f"Empty choices array - content likely filtered by API safety")
                    logger.error(f"  Model: {self.model} (source: {self.source})")
                    logger.error(f"  This is likely a content policy violation")
                    logger.error(f"  Solution: Use mock mode or switch to a different provider")
                    raise ValueError("API returned empty choices - content filtered by safety policy")

                return response.choices[0].message.content.strip()

            except (APIError, APIConnectionError) as e:
                error_type = "API Error" if isinstance(e, APIError) else "Connection Error"
                logger.warning(f"{error_type} Attempt {attempt + 1}/{self.max_retries} failed. Retrying in {self.retry_delay}s... Error: {e}")
                time.sleep(self.retry_delay)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == 0:
                    logger.error(f"  Model: {self.model}")
                    logger.error(f"  Source: {self.source}")
                    logger.error(f"  Base URL: {getattr(self, 'base_url', 'N/A')}")
                    logger.error(f"  Messages (first 200 chars): {str(messages)[:200]}")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"LLM API call failed after {self.max_retries} retries")