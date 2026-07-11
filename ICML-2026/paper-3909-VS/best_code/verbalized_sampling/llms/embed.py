# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import signal
import time
from dataclasses import dataclass
from functools import wraps

import numpy as np
from openai import OpenAI


@dataclass
class EmbeddingResponse:
    """Container for embedding response."""

    embedding: list[float]
    cost: float


class LocalEmbeddingModel:
    """Local sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", timeout: int = 128):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.timeout = timeout
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str, max_retries: int = 3) -> EmbeddingResponse:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return EmbeddingResponse(embedding=embedding.tolist(), cost=0.0)


class TfidfEmbeddingModel:
    """Hash-based embedding model as lightweight fallback (no GPU needed)."""

    def __init__(self, model_name: str = "hash", timeout: int = 128):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model_name = model_name
        self.timeout = timeout
        self.vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")

    def get_embedding(self, text: str, max_retries: int = 3) -> EmbeddingResponse:
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return EmbeddingResponse(embedding=embedding.tolist(), cost=0.0)


class TimeoutError(Exception):
    """Custom timeout exception."""


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Embedding request timed out")


def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to function calls."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up the timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                raise e
            finally:
                # Restore the old signal handler
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


class OpenAIEmbeddingModel:
    """OpenAI embedding model implementation."""

    def __init__(self, model_name: str = "text-embedding-3-small", timeout: int = 128):
        """Initialize the embedding model.

        Args:
            model_name: The name of the OpenAI embedding model to use.
                      Defaults to "text-embedding-3-small".
            timeout: Timeout in seconds for embedding requests. Defaults to 128.
        """
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,  # Set OpenAI client timeout as well
        )

        # Cost per 1K tokens for different models
        self.cost_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001,  # $0.0001 per 1K tokens
        }

    @with_timeout(128)  # 128 seconds timeout
    def _get_embedding_with_timeout(self, text: str) -> EmbeddingResponse:
        """Internal method to get embedding with timeout protection."""
        response = self.client.embeddings.create(model=self.model_name, input=text)

        # Calculate cost based on token count
        token_count = response.usage.total_tokens
        cost = (token_count / 1000) * self.cost_per_1k_tokens.get(self.model_name, 0.00002)

        return EmbeddingResponse(embedding=response.data[0].embedding, cost=cost)

    def _get_embedding_direct(self, text: str) -> EmbeddingResponse:
        """Internal method to get embedding without timeout protection."""
        response = self.client.embeddings.create(model=self.model_name, input=text)

        # Calculate cost based on token count
        token_count = response.usage.total_tokens
        cost = (token_count / 1000) * self.cost_per_1k_tokens.get(self.model_name, 0.00002)

        return EmbeddingResponse(embedding=response.data[0].embedding, cost=cost)

    def get_embedding(self, text: str, max_retries: int = 3) -> EmbeddingResponse:
        """Get embedding for a text with retry logic."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return self._get_embedding_direct(text)

            except Exception as e:
                last_exception = e
                if attempt < max_retries and (
                    "timeout" in str(e).lower() or "timed out" in str(e).lower()
                ):
                    print(
                        f"Embedding request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying..."
                    )
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Embedding request failed: {e}")
                    raise e

        raise last_exception


def get_embedding_model(
    model_name: str = "text-embedding-3-small", timeout: int = 128
):
    """Get an embedding model instance.

    Args:
        model_name: The name of the embedding model to use.
                   Defaults to "text-embedding-3-small".
                   Use "local" prefix for sentence-transformers (e.g., "local:all-MiniLM-L6-v2").
        timeout: Timeout in seconds for embedding requests. Defaults to 128.

    Returns:
        An embedding model instance.
    """
    if model_name.startswith("local:"):
        local_model = model_name.split(":", 1)[1]
        if local_model == "hash":
            return TfidfEmbeddingModel(model_name=local_model, timeout=timeout)
        return LocalEmbeddingModel(model_name=local_model, timeout=timeout)
    return OpenAIEmbeddingModel(model_name=model_name, timeout=timeout)
