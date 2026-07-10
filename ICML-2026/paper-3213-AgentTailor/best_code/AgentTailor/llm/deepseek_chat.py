import aiohttp
import json
from typing import List, Union, Optional, Dict, Any
from tenacity import (
    retry, 
    wait_random_exponential, 
    stop_after_attempt,
    retry_if_exception_type,
    RetryCallState
)
from dotenv import load_dotenv
import os
import logging
import asyncio

from AgentTailor.llm.format import Message
from AgentTailor.llm.price import cost_count, cost_count_by_tokens
from AgentTailor.llm.llm import LLM
from AgentTailor.llm.llm_registry import LLMRegistry

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# DeepSeek API configuration
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Retry configuration
MAX_RETRIES = 5  # Number of retry attempts
MAX_WAIT_TIME = 60  # Maximum wait time (seconds)
MIN_WAIT_TIME = 1   # Minimum wait time (seconds)
REQUEST_TIMEOUT = 120  # Request timeout (seconds)


def log_retry_attempt(retry_state: RetryCallState):
    """Log a retry attempt."""
    attempt_number = retry_state.attempt_number
    exception = retry_state.outcome.exception()
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        f"DeepSeek API call failed (attempt {attempt_number}/{MAX_RETRIES + 1}): "
        f"{type(exception).__name__}: {str(exception)}. "
        f"Retrying in {wait_time:.2f} seconds..."
    )


@retry(
    wait=wait_random_exponential(multiplier=1, min=MIN_WAIT_TIME, max=MAX_WAIT_TIME),
    stop=stop_after_attempt(MAX_RETRIES + 1),  # In total, try MAX_RETRIES + 1 times
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, KeyError)),
    before_sleep=log_retry_attempt,
    reraise=True
)
async def adeepseek_chat(
    model: str,
    messages: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = False
):
    """
    Asynchronously call the DeepSeek API.

    Parameters:
    - model: Model name.
    - messages: List of messages.
    - max_tokens: Maximum number of tokens.
    - temperature: Sampling temperature.
    - stream: Whether to stream responses.
    """
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    # Build request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    if temperature is not None:
        payload["temperature"] = temperature
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=30),
            ) as response:
                # Check HTTP status code
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"DeepSeek API returned non-200 status code {response.status}: {error_text[:200]}"
                    logger.error(error_msg)
                    response.raise_for_status()
                
                if stream:
                    # Handle streaming responses
                    full_response = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                            except json.JSONDecodeError:
                                continue
                    if not full_response:
                        raise ValueError("Empty streaming response.")
                    # Streaming responses may not expose accurate token usage; approximate it
                    import tiktoken
                    try:
                        encoder = tiktoken.get_encoding("cl100k_base")
                        prompt_text = "\n".join([msg.get('content', '') if isinstance(msg, dict) else str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in messages])
                        cost_count(prompt_text, full_response, model)
                    except:
                        pass
                    return full_response
                else:
                    # Handle normal (non-streaming) responses
                    result = await response.json()
                    
                    # Validate response format
                    if 'choices' not in result or len(result['choices']) == 0:
                        error_msg = f"DeepSeek API response format is invalid: {json.dumps(result, ensure_ascii=False)[:200]}"
                        logger.error(error_msg)
                        raise KeyError("Missing or empty 'choices' field in response")
                    
                    if 'message' not in result['choices'][0]:
                        error_msg = "DeepSeek API response format is invalid: 'message' field is missing in choices[0]"
                        logger.error(error_msg)
                        raise KeyError("Missing 'message' field in choices[0]")
                    
                    response_content = result['choices'][0]['message']['content']
                    
                    # Extract usage information and count tokens
                    prompt_text = "\n".join([msg.get('content', '') if isinstance(msg, dict) else str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in messages])
                    
                    if 'usage' in result:
                        # If usage is present, rely on API-provided token counts (more accurate)
                        usage = result['usage']
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        
                        # If the API returns zero token counts, fall back to tiktoken estimation
                        if prompt_tokens == 0:
                            import tiktoken
                            try:
                                encoder = tiktoken.get_encoding("cl100k_base")
                                prompt_tokens = len(encoder.encode(prompt_text))
                                completion_tokens = len(encoder.encode(response_content))
                            except:
                                prompt_tokens = 0
                                completion_tokens = 0
                    else:
                        # If usage is missing, estimate tokens via tiktoken
                        import tiktoken
                        try:
                            encoder = tiktoken.get_encoding("cl100k_base")
                            prompt_tokens = len(encoder.encode(prompt_text))
                            completion_tokens = len(encoder.encode(response_content))
                        except:
                            prompt_tokens = 0
                            completion_tokens = 0
                    
                    # Count tokens/cost: prefer API usage, fallback to local estimation.
                    if (
                        "usage" in result
                        and isinstance(result["usage"], dict)
                        and "prompt_tokens" in result["usage"]
                        and "completion_tokens" in result["usage"]
                    ):
                        cost_count_by_tokens(prompt_tokens, completion_tokens, model)
                    else:
                        cost_count(prompt_text, response_content, model)
                    
                    return response_content
                    
    except asyncio.TimeoutError as e:
        error_msg = f"DeepSeek API request timed out (timeout: {REQUEST_TIMEOUT} seconds)"
        logger.error(error_msg)
        raise
    except aiohttp.ClientConnectorError as e:
        error_msg = f"DeepSeek API connection failed: {str(e)}"
        logger.error(error_msg)
        raise
    except aiohttp.ClientResponseError as e:
        error_msg = f"DeepSeek API HTTP error: {e.status} - {e.message}"
        logger.error(error_msg)
        raise
    except aiohttp.ClientError as e:
        error_msg = f"DeepSeek API client error: {str(e)}"
        logger.error(error_msg)
        raise
    except KeyError as e:
        error_msg = f"Failed to parse DeepSeek response: {str(e)}"
        logger.error(error_msg)
        raise
    except json.JSONDecodeError as e:
        error_msg = f"DeepSeek API JSON decode failed: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"DeepSeek API unknown error: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

def convert_messages_to_deepseek_format(messages: List[Message]) -> List[Dict]:
    """
    Convert a list of Message objects into the request format required by the DeepSeek API.
    """
    deepseek_messages = []
    for msg in messages:
        deepseek_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    return deepseek_messages

@LLMRegistry.register('DeepSeekChat')
class DeepSeekChat(LLM):

    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name
        # Set default values
        self.DEFAULT_MAX_TOKENS = 2048
        self.DEFAULT_TEMPERATURE = 0.7
        self.DEFUALT_NUM_COMPLETIONS = 1

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        stream: bool = False
    ) -> Union[List[str], str]:
        """
        Asynchronously generate text using the DeepSeek chat model.
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        # Convert messages to API format
        #deepseek_messages = convert_messages_to_deepseek_format(messages)
        deepseek_messages=messages
        # Call the API
        response = await adeepseek_chat(
            model=self.model_name,
            messages=deepseek_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        # Handle multiple generations
        if num_comps > 1:
            # Simplified handling; in practice you may need multiple API calls
            return [response] * num_comps
        else:
            return response
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Synchronously generate text (implemented by wrapping the async call).
        """
        # Since this is a synchronous method, use asyncio.run to execute the async call
        import asyncio
        return asyncio.run(self.agen(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps
        ))
