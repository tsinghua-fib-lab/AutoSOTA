from dataclasses import dataclass
import json
from typing import Dict, Any, Optional, List, Union
import base64
import os
import re
from pathlib import Path
import httpx
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_not_exception_type
from loguru import logger
import tiktoken

from common.cache.decorator import create_cache_decorator

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class NonRetryableError(Exception):
    """Exception for errors that should not be retried (e.g., content policy violations)"""
    pass


@dataclass
class GeneratorOutput:
    output: str
    prompt_tokens: int
    completion_tokens: int
    cost: float = 0.0
    raw_response: Optional[str] = None  # Complete API response JSON string


@dataclass
class EmbeddingOutput:
    embeddings: List[float]
    prompt_tokens: int


class DirectGenerator:
    """Direct LLM generator with caching and retry logic"""
    
    def __init__(self, model_name: str, base_url: str, api_key: str, 
                 temperature: float = 0.0, top_p: float = 1.0, timeout: int = 500, max_tokens: Optional[int] = None,
                 stream: bool = False,
                 cache_config: Optional[Dict[str, Any]] = None,
                 pricing_config: Optional[Dict[str, Any]] = None,
                 extra_body: Optional[Dict[str, Any]] = None,
                 config_name: Optional[str] = None,
                 reasoning_effort: Optional[Any] = None):
        self.model_name = model_name
        self.config_name = config_name or model_name  # Use config name for cache keys
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.stream = stream
        self.reasoning_effort = reasoning_effort
        self.pricing_config = pricing_config or {}
        self.extra_body = extra_body or {}
        self.cache_config = cache_config
        
        # Initialize client and cache decorator
        self._initialize_client()
        self._cache_decorator = create_cache_decorator(cache_config)
        
        # Apply cache decorator to _generate method if enabled
        if self._cache_decorator.config.enabled:
            self._generate = self._cache_decorator(self._generate, generator_instance=self)
    
    def _initialize_client(self):
        """Initialize or reinitialize the OpenAI client"""
        if "21020" in self.base_url:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(verify=False)
            )
        else:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
    
    def _log_retry(self, retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(
                f"Retrying DirectGenerator.generate due to error: {str(exception)}. "
                f"Attempt {retry_state.attempt_number}/10"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_not_exception_type(NonRetryableError),
        before_sleep=lambda retry_state: DirectGenerator._log_retry(None, retry_state)
    )
    def _generate(self, question: str) -> GeneratorOutput:
        try:
            # Build API parameters
            if self.max_tokens is None:
                api_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "timeout": self.timeout,
                    "stream": self.stream,
                }
            else:
                api_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "timeout": self.timeout,
                    "max_tokens": self.max_tokens,
                    "stream": self.stream,
                }
            
            # Add reasoning_effort if specified
            if self.reasoning_effort is not None:
                api_params["reasoning_effort"] = self.reasoning_effort
            if self.stream:
                api_params["stream_options"] = {"include_usage": True}
            # Add extra_body if specified
            if self.extra_body:
                api_params["extra_body"] = self.extra_body

            response = self.client.chat.completions.create(**api_params)
            if self.stream:
                response = self._build_stream_response(response)
                response["object"] = "chat.completion"
                response = ChatCompletion(**response)
            usage = response.usage
            choices = response.choices

            if not choices:
                raise ValueError("Generation failed: Empty response from LLM")

            content = choices[0].message.content
            if content is None:
                model_extra = getattr(choices[0].message, "model_extra", {}) or {}
                reasoning_content = model_extra.get("reasoning_content", "")
                if not reasoning_content:
                    raise ValueError("Generation failed: Empty response from LLM")
                content = "Generation failed: think too long"

            if len(content) == 0:
                raise ValueError("Generation failed: Empty response from LLM")
            # Calculate cost
            cost = self._calculate_cost(usage)

            # Cache raw response if configured
            raw_response = None
            if self._should_cache_raw_response():
                raw_response = response.model_dump_json()

            result = GeneratorOutput(
                output=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=cost,
                raw_response=raw_response
            )

            # Hook for cache storage (handled by decorator)
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error in DirectGenerator._generate: {error_str}, model: {self.model_name}")
            # Check for sensitive_words_detected error - don't retry
            if "sensitive_words_detected" in error_str.lower():
                raise NonRetryableError(f"[SENSITIVE_WORDS_DETECTED] {error_str}")
            raise
    def _build_stream_response(self, response):
        raw_content_parts = []
        for chunk in response:
            raw_content_parts.append(
                chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else None)
        return_response = self._merge_streaming_chunks(raw_content_parts)
        usage = return_response.get("usage")
        if not usage:
            return_response["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        elif isinstance(usage, dict):
            usage.setdefault("prompt_tokens", 0)
            usage.setdefault("completion_tokens", 0)
            usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
        return return_response

    def _merge_streaming_chunks(self, raw_content_parts):
        """Merge streaming JSON chunks into a ChatCompletion-compatible dict."""
        empty_response = {
            "id": "",
            "object": "chat.completion",
            "created": 0,
            "model": self.model_name,
            "choices": [
                {
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "refusal": None,
                        "function_call": None,
                        "tool_calls": None,
                        "annotations": None,
                        "audio": None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        if not raw_content_parts:
            return empty_response

        chunks = []
        for part in raw_content_parts:
            if part:
                try:
                    chunks.append(json.loads(part))
                except json.JSONDecodeError:
                    continue

        if not chunks:
            return empty_response

        result = {}
        first_chunk = chunks[0]

        for key, value in first_chunk.items():
            if key != "choices":
                result[key] = value

        if result.get("object") == "chat.completion.chunk":
            result["object"] = "chat.completion"

        result["choices"] = []

        accumulated_delta = {}
        content_parts = []
        reasoning_parts = []
        finish_reason = None
        native_finish_reason = None

        for chunk in chunks:
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                for key, value in delta.items():
                    if key == "content" and value:
                        content_parts.append(value)
                    elif key == "reasoning_content" and value:
                        reasoning_parts.append(value)
                    elif key == "reasoning" and value:
                        reasoning_parts.append(value)
                    elif value is not None and key not in ["reasoning_details"]:
                        if key not in accumulated_delta:
                            accumulated_delta[key] = value

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                if choice.get("native_finish_reason"):
                    native_finish_reason = choice["native_finish_reason"]

                for key in choice:
                    if key not in ["delta", "finish_reason", "native_finish_reason"] and key not in ["index",
                                                                                                     "logprobs"]:
                        if key not in accumulated_delta:
                            accumulated_delta[key] = choice[key]

            if "usage" in chunk and chunk["usage"]:
                result["usage"] = chunk["usage"]

            for key, value in chunk.items():
                if key not in ["id", "choices", "created", "model", "object"] and value is not None:
                    result[key] = value

        message = {}

        if content_parts:
            message["content"] = "".join(content_parts)
        else:
            message["content"] = None

        if reasoning_parts:
            merged_reasoning = "".join(reasoning_parts)
            has_reasoning_content = any(
                "reasoning_content" in chunk.get("choices", [{}])[0].get("delta", {})
                for chunk in chunks if "choices" in chunk and chunk["choices"]
            )
            if has_reasoning_content:
                message["reasoning_content"] = merged_reasoning
            else:
                message["reasoning"] = merged_reasoning

        for key, value in accumulated_delta.items():
            if key not in ["content", "reasoning_content", "reasoning", "reasoning_details"]:
                message[key] = value

        if "role" not in message:
            message["role"] = "assistant"
        if "refusal" not in message:
            message["refusal"] = None
        if "function_call" not in message:
            message["function_call"] = None
        if "tool_calls" not in message:
            message["tool_calls"] = None
        if "annotations" not in message:
            message["annotations"] = None
        if "audio" not in message:
            message["audio"] = None

        merged_choice = {
            "finish_reason": finish_reason,
            "index": 0,
            "logprobs": None,
            "message": message
        }

        if native_finish_reason is not None:
            merged_choice["native_finish_reason"] = native_finish_reason

        result["choices"] = [merged_choice]

        return result

    def generate(self, question: str) -> GeneratorOutput:
        """Generate response with retry logic and error handling"""
        try:
            if self.config_name == "qwen3-235b-a22b-no-thinking":
                question+='/no_think'
            return self._generate(question)
        except Exception as e:
            logger.error(
                f"DirectGenerator.generate failed after all retries: {str(e)}, "
                f"model: {self.model_name}"
            )
            return GeneratorOutput(
                output=f"Generation failed: {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0
            )

    def _should_cache_raw_response(self) -> bool:
        """Check if raw response should be cached based on config"""
        if not self.cache_config:
            return False
        conditions = self.cache_config.get('conditions', {})
        return conditions.get('cache_raw_response', False)

    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on usage object"""
        try:
            # First try to use cost from usage object if available
            if hasattr(usage, 'cost') and usage.cost is not None:
                return float(usage.cost)
            
            # Fallback to pricing configuration
            if self.pricing_config:
                prompt_price = self.pricing_config.get('prompt_price_per_million', 0.0)
                completion_price = self.pricing_config.get('completion_price_per_million', 0.0)
                
                if prompt_price > 0 or completion_price > 0:
                    prompt_cost = (usage.prompt_tokens / 1_000_000) * prompt_price
                    completion_cost = (usage.completion_tokens / 1_000_000) * completion_price
                    return prompt_cost + completion_cost
            
            return 0.0
            
        except Exception as e:
            logger.warning(
                f"Failed to calculate cost for model {self.model_name}: {str(e)}. "
                f"Usage: prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}"
            )
            return 0.0


class MultimodalGenerator(DirectGenerator):
    """Multimodal LLM generator that supports images along with text"""
    
    def __init__(self, model_name: str, base_url: str, api_key: str, 
                 temperature: float = 0.0, top_p: float = 1.0, timeout: int = 500,
                 cache_config: Optional[Dict[str, Any]] = None,
                 pricing_config: Optional[Dict[str, Any]] = None,
                 extra_body: Optional[Dict[str, Any]] = None,
                 config_name: Optional[str] = None,
                 reasoning_effort: Optional[Any] = None):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            cache_config=cache_config,
            pricing_config=pricing_config,
            extra_body=extra_body,
            config_name=config_name,
            reasoning_effort=reasoning_effort
        )
        
        # Apply cache decorator to _generate_multimodal method if enabled
        if self._cache_decorator.config.enabled:
            self._generate_multimodal = self._cache_decorator(self._generate_multimodal, generator_instance=self)
        
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            # Handle relative paths by resolving them from current working directory
            if not os.path.isabs(image_path):
                # Try to resolve relative to current working directory first
                full_path = os.path.join(os.getcwd(), image_path)
                if not os.path.exists(full_path):
                    # If not found, try to resolve relative to data directory
                    full_path = os.path.join(os.getcwd(), "data", image_path)
            else:
                # For absolute paths, use as-is
                full_path = image_path
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image file not found: {full_path}")
            
            with open(full_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type for image based on file extension"""
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(suffix, 'image/jpeg')
    
    def _prepare_image_content(self, images: List[str]) -> List[Dict[str, Any]]:
        """Prepare image content for OpenAI vision API"""
        image_contents = []
        
        for image_path in images:
            try:
                # Check if it's already base64 encoded
                if image_path.startswith('data:image/'):
                    # Already in data URI format
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": image_path}
                    })
                elif image_path.startswith('http://') or image_path.startswith('https://'):
                    # URL format
                    image_contents.append({
                        "type": "image_url", 
                        "image_url": {"url": image_path}
                    })
                else:
                    # File path - encode to base64
                    base64_image = self._encode_image_to_base64(image_path)
                    mime_type = self._get_image_mime_type(image_path)
                    data_uri = f"data:{mime_type};base64,{base64_image}"
                    
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                    
            except Exception as e:
                logger.error(f"Failed to prepare image {image_path}: {str(e)}")
                # Continue with other images even if one fails
                continue
                
        return image_contents
    
    def _create_multimodal_message(self, question: str, images: List[str]) -> Dict[str, Any]:
        """Create multimodal message, aligning text-image order with inline <image> tags."""
        if not images:
            return {"role": "user", "content": [{"type": "text", "text": question}]}
        
        image_contents = self._prepare_image_content(images)
        simple_pattern = re.compile(r'<image>', flags=re.IGNORECASE)
        
        simple_matches = list(simple_pattern.finditer(question))
        
        if simple_matches:
            if len(simple_matches) != len(image_contents):
                logger.warning(
                    f"Image tag count ({len(simple_matches)}) doesn't match "
                    f"image count ({len(image_contents)}) for prompt: {question[:50]}..."
                )
            
            content = []
            last_idx = 0
            
            for idx, match in enumerate(simple_matches):
                segment = question[last_idx:match.start()]
                if segment:
                    content.append({"type": "text", "text": segment})
                
                if idx < len(image_contents):
                    content.append(image_contents[idx])
                
                last_idx = match.end()
            
            tail = question[last_idx:]
            if tail:
                content.append({"type": "text", "text": tail})
            
            for idx in range(len(simple_matches), len(image_contents)):
                logger.warning(f"Extra image at index {idx} will be appended to the end")
                content.append(image_contents[idx])
            
            if not content:
                content.append({"type": "text", "text": ""})
            
            return {"role": "user", "content": content}
        
        content = [{"type": "text", "text": question}]
        content.extend(image_contents)
        return {"role": "user", "content": content}
    
    def _log_retry(self, retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(
                f"Retrying MultimodalGenerator.generate due to error: {str(exception)}. "
                f"Attempt {retry_state.attempt_number}/10"
            )
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=100),
        retry=retry_if_not_exception_type(NonRetryableError),
        before_sleep=lambda retry_state: MultimodalGenerator._log_retry(None, retry_state)
    )
    def _generate_multimodal(self, question: str, images: Optional[List[str]] = None) -> GeneratorOutput:
        """Generate response with multimodal input support"""
        try:
            # Create message with images if provided
            if images:
                message = self._create_multimodal_message(question, images)
                # logger.info(f"Generating multimodal response with {len(images)} images for model {self.model_name}")
            else:
                # Fall back to text-only
                message = {"role": "user", "content": question}
                # logger.info(f"Generating text-only response for model {self.model_name}")
            
            # Build API parameters
            api_params = {
                "model": self.model_name,
                "messages": [message],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "timeout": self.timeout,
            }
            
            # Add reasoning_effort if specified
            if self.reasoning_effort is not None:
                api_params["reasoning_effort"] = self.reasoning_effort
                
            # Add extra_body if specified
            if self.extra_body:
                api_params["extra_body"] = self.extra_body
            
            response = self.client.chat.completions.create(**api_params)

            usage = response.usage
            choices = response.choices

            if not choices or choices[0].message.content is None:
                raise ValueError("Empty response from multimodal LLM")

            # Calculate cost
            cost = self._calculate_cost(usage)

            # Cache raw response if configured
            raw_response = None
            if self._should_cache_raw_response():
                raw_response = response.model_dump_json()

            result = GeneratorOutput(
                output=choices[0].message.content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=cost,
                raw_response=raw_response
            )

            # Hook for cache storage (handled by decorator)
            return result

        except Exception as e:
            error_str = str(e)
            # Reinitialize the client after multimodal provider-side errors.
  
            logger.warning(f"Image limit reached for model {self.model_name}, reinitializing client")
            self._initialize_client()

            logger.error(f"Error in MultimodalGenerator._generate_multimodal: {error_str}, model: {self.model_name}")
            # Check for sensitive_words_detected error - don't retry
            if "sensitive_words_detected" in error_str.lower():
                raise NonRetryableError(f"[SENSITIVE_WORDS_DETECTED] {error_str}")
            raise
    
    def generate_multimodal(self, question: str, images: Optional[List[str]] = None) -> GeneratorOutput:
        """Generate multimodal response with retry logic and error handling"""
        try:
            return self._generate_multimodal(question, images)
        except Exception as e:
            logger.error(
                f"MultimodalGenerator.generate_multimodal failed after all retries: {str(e)}, "
                f"model: {self.model_name}"
            )
            return GeneratorOutput(
                output=f"Multimodal generation failed: {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0
            )
    
    def generate(self, question: str, images: Optional[List[str]] = None) -> GeneratorOutput:
        """Generate response - supports both text-only and multimodal"""
        if images:
            return self.generate_multimodal(question, images)
        else:
            # Use parent class method for text-only generation
            return super().generate(question)


class EmbeddingGenerator(DirectGenerator):
    """Embedding generator for converting text to vectors"""

    def __init__(self, model_name: str, base_url: str, api_key: str,
                 timeout: int = 500,
                 cache_config: Optional[Dict[str, Any]] = None,
                 config_name: Optional[str] = None,
                 max_context_length: Optional[int] = None,
                 model_path: Optional[str] = None,
                 **kwargs):
        # Initialize with minimal parameters needed for embedding
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,  # Not used for embeddings
            top_p=1.0,        # Not used for embeddings
            timeout=timeout,
            cache_config=cache_config,
            config_name=config_name,
            # Ignore other parameters not relevant to embeddings
        )

        self.max_context_length = max_context_length
        self.model_path = model_path
        self._tokenizer = None
        self._tokenizer_type = None  # 'transformers' or 'tiktoken'
        if self.max_context_length:
            self._initialize_tokenizer()

        # Apply cache decorator to _generate_embedding method if enabled
        if self._cache_decorator.config.enabled:
            self._generate_embedding = self._cache_decorator(self._generate_embedding, generator_instance=self)

    def _log_retry(self, retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(
                f"Retrying EmbeddingGenerator.generate_embedding due to error: {str(exception)}. "
                f"Attempt {retry_state.attempt_number}/10"
            )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=100),
        retry=retry_if_not_exception_type(NonRetryableError),
        before_sleep=lambda retry_state: EmbeddingGenerator._log_retry(None, retry_state)
    )
    def _generate_embedding(self, text: str) -> EmbeddingOutput:
        """Generate embedding for a single text"""
        try:
            if self.max_context_length:
                text = self._enforce_context_limit(text)

            # Use embeddings API instead of chat completions
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                timeout=self.timeout
            )

            usage = response.usage
            embedding_data = response.data[0]

            if not embedding_data.embedding:
                raise ValueError("Empty embedding response from model")

            result = EmbeddingOutput(
                embeddings=embedding_data.embedding,
                prompt_tokens=usage.prompt_tokens
            )

            return result

        except Exception as e:
            error_str = str(e)
            logger.error(f"Error in EmbeddingGenerator._generate_embedding: {error_str}, model: {self.model_name}")
            # Check for sensitive_words_detected error - don't retry
            if "sensitive_words_detected" in error_str.lower():
                raise NonRetryableError(f"[SENSITIVE_WORDS_DETECTED] {error_str}")
            raise

    def generate_embedding(self, text: str) -> EmbeddingOutput:
        """Generate embedding with retry logic and error handling"""
        try:
            return self._generate_embedding(text)
        except Exception as e:
            logger.error(
                f"EmbeddingGenerator.generate_embedding failed after all retries: {str(e)}, "
                f"model: {self.model_name}"
            )
            # Return empty embedding on failure
            return EmbeddingOutput(
                embeddings=[],
                prompt_tokens=0
            )

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------
    def _initialize_tokenizer(self) -> None:
        """Prepare tokenizer for token-length based truncation.
        
        If model_path is provided and not empty, attempts to load tokenizer from that path.
        Otherwise, falls back to tiktoken.
        """
        # Try loading from model_path if provided
        if self.model_path and self.model_path.strip():
            if TRANSFORMERS_AVAILABLE:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._tokenizer_type = 'transformers'
                logger.info(
                    f"Loaded tokenizer from {self.model_path} for {self.model_name} context enforcement"
                )
                return
            else:
                logger.warning(
                    f"transformers library not available. Cannot load tokenizer from {self.model_path}. "
                    f"Falling back to tiktoken for {self.model_name}"
                )
        else:
            # Fallback to tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)
                self._tokenizer_type = 'tiktoken'
            except Exception:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                self._tokenizer_type = 'tiktoken'
                logger.warning(
                    "Falling back to cl100k_base tokenizer for %s context enforcement",
                    self.model_name,
                )

    def _enforce_context_limit(self, text: str) -> str:
        if not self.max_context_length or self.max_context_length <= 0:
            return text

        if not self._tokenizer:
            self._initialize_tokenizer()

        try:
            if self._tokenizer_type == 'transformers':
                # transformers tokenizer returns a list directly
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
            else:
                # tiktoken encoding returns a list directly
                tokens = self._tokenizer.encode(text)
        except Exception as exc:
            logger.warning(
                "Tokenizer encode failed for %s: %s; skipping truncation",
                self.model_name,
                exc,
            )
            return text

        if len(tokens) <= self.max_context_length:
            return text

        truncated_tokens = tokens[-self.max_context_length:]
        try:
            if self._tokenizer_type == 'transformers':
                truncated_text = self._tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            else:
                truncated_text = self._tokenizer.decode(truncated_tokens)
        except Exception as exc:
            logger.warning(
                f"Tokenizer decode failed for {self.model_name}: {exc}; returning original text"
            )
            return text

        logger.info(
            f"Truncated embedding input for {self.model_name} from {len(tokens)} to {len(truncated_tokens)} tokens"
        )
        return truncated_text
