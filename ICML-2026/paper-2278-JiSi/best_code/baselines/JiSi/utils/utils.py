import argparse
import os
import json
import time
import requests
import openai
import copy
from openai import OpenAI
from loguru import logger
import random
from typing import Optional, List, Union, Dict, Any, Generator

"""For the global configs"""
DEBUG = int(os.environ.get("DEBUG", "0"))

# Shared JiSi system prompt for experts and aggregators.
JISI_SYSTEM_PROMPT = ""

# Global variables
extra_api_keys: Dict[str, str] = {}
MODELCONFIG: Dict[str, Any] = {}

class ApiConfig:
    """Configuration for OpenAI-compatible API calls."""
    def __init__(self, api_key: Union[str, List[str]], full_model_name: str, base_url: str, stream: bool = False, max_tokens: Optional[int] = None):
        self.api_key = api_key if isinstance(api_key, list) else [api_key]
        self.full_model_name = full_model_name
        self.base_url = base_url
        self.stream = stream
        self.max_tokens = max_tokens
        self.cnt = 0

    def get_api_key(self) -> str:
        if not self.api_key:
            return ""
        key = self.api_key[self.cnt % len(self.api_key)]
        self.cnt += 1
        return key

class ApiConfigPost:
    """Configuration for raw HTTP POST API calls."""
    def __init__(self, api_key: Union[str, List[str]], full_model_name: str, base_url: str, stream: bool = False, max_tokens: Optional[int] = None):
        self.api_key = api_key if isinstance(api_key, list) else [api_key] # Handle list or string
        self.full_model_name = full_model_name
        self.base_url = base_url
        self.stream = stream
        self.max_tokens = max_tokens
        self.cnt = 0
    
    def get_api_key(self) -> str:
        if not self.api_key:
            return ""
        key = self.api_key[self.cnt % len(self.api_key)]
        self.cnt += 1
        return key
        
    @property
    def api_key_value(self) -> str:
        """Property to access single key for backwards compatibility or single use"""
        return self.get_api_key()


def _resolve_single_api_key(value: Optional[str]) -> str:
    """Resolve an API key from a literal value or environment variable name."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    if value.startswith("env:"):
        return os.getenv(value[4:], "")
    env_value = os.getenv(value)
    if env_value:
        return env_value
    return value


def _resolve_api_key_value(value: Union[str, List[str], None]) -> Union[str, List[str]]:
    if isinstance(value, list):
        return [_resolve_single_api_key(item) for item in value]
    return _resolve_single_api_key(value)


def setup_model_config(config_path: Optional[str] = None) -> None:
    """
    Load API keys and model configurations from a JSON file.
    
    Args:
        config_path (Optional[str]): Path to the configuration JSON file.
    """
    global extra_api_keys, MODELCONFIG
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            extra_api_keys.update({
                key: _resolve_single_api_key(value)
                for key, value in data.get('extra_api_keys', {}).items()
            })
            model_configs = data.get('model_configs', {})
    else:
        logger.warning(f"API config file not found or invalid: {config_path}")
        model_configs = {}

    MODELCONFIG.clear()
    
    for model_key, config in model_configs.items():
        mode = config.get('mode', 'openai')
        api_key = config.get('api_key')
        api_keys = config.get('api_keys')
        api_key_env = config.get('api_key_env')
        api_key_name = config.get('api_key_name')

        if api_keys is not None:
            resolved_api_key = _resolve_api_key_value(api_keys)
        elif api_key is not None:
            resolved_api_key = _resolve_api_key_value(api_key)
        elif api_key_env is not None:
            resolved_api_key = _resolve_single_api_key(api_key_env)
        elif api_key_name is not None:
            resolved_api_key = extra_api_keys.get(api_key_name, "")
        else:
            resolved_api_key = ""

        if not resolved_api_key:
            logger.warning(f"Model {model_key} has no resolved API key")
        
        # Create Config Object
        max_tokens = config.get('max_tokens')
        if mode == 'post':
            MODELCONFIG[model_key] = ApiConfigPost(
                api_key=resolved_api_key,
                full_model_name=config.get('model_name'),
                base_url=config.get('base_url'),
                stream=config.get('stream', False),
                max_tokens=max_tokens
            )
        else: # default to openai
            MODELCONFIG[model_key] = ApiConfig(
                api_key=resolved_api_key,
                full_model_name=config.get('model_name'),
                base_url=config.get('base_url'),
                stream=config.get('stream', False),
                max_tokens=max_tokens
            )

def inject_references_to_messages(
        messages: List[Dict[str, Any]],
        references: List[str],
        agg_prompt: str = 'normal',
        ref_score: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Inject model references into the system prompt of messages.
    
    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.
        references (List[str]): List of reference strings from other models.
        agg_prompt (str): Aggregation strategy ('normal' or 'with_score').
        ref_score (Optional[List[float]]): List of confidence scores corresponding to references.
        
    Returns:
        List[Dict[str, Any]]: Updated messages with injected references.
    """
    messages = copy.deepcopy(messages)
    # test_data already contains JISI_SYSTEM_PROMPT; add only the aggregation task.
    if agg_prompt == 'normal':
        system = f"""You have been provided with a set of responses from {len(references)} open-source models to the latest user query. The response of the X-th model is enclosed within <Model_X_Response> and </Model_X_Response>.  Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

    Responses from models:"""

        for i, reference in enumerate(references):
            system += f"\n\n {i + 1}. <Model_{i + 1}_Response> \n {reference} \n </Model_{i + 1}_Response>"
    elif agg_prompt == 'with_score':
        if ref_score is None:
            raise Exception("You must provide a reference score.")
        system = f"""You have been provided with responses from {len(references)} open-source models to a user query. 
        Crucially, each response comes with a **Confidence Score (0.0 to 1.0)**. This score represents the estimated prior probability that the model's answer is correct.

        Your task is to synthesize these responses into a single, superior answer. You must adapt your aggregation strategy based on these scores:

        1. **High Confidence Consensus:** If all models have high scores (e.g., > 0.7), assume the information is largely correct. Synthesize the content to improve flow, clarity, and completeness.
        2. **Low Confidence Scenarios:** If all models have low scores, treat the information with extreme skepticism. They may be hallucinating. You must critically evaluate every claim, think step-by-step to verify logic, and refuse to include unverified information.
        3. **Mixed Confidence:** If scores vary significantly, prioritize the information from the high-confidence models. Contrast this with the low-confidence responses to spot errors in the latter. Only include details from low-confidence models if they are verifiable or add necessary context missing from the high-confidence responses.

        **Goal:** Produce a refined, accurate, and comprehensive final response. Do not mention the confidence scores or the internal model references in your final output.

        Responses from models:"""
        for i, reference in enumerate(references):
            current_score = ref_score[i]
            system += f"\n\n--- Model {i + 1} (Confidence: {current_score:.4f}) ---\n"
            system += f"<Model_{i + 1}_Response>\n{reference}\n</Model_{i + 1}_Response>"
        system += ("\n\nBased on the confidence scores provided above, synthesize the final answer. "
                   "Use the **High Confidence models** to build the foundational structure and core facts of your answer. "
                   "Do not simply discard the **Low Confidence models**. Instead, treat them with skepticism but scan them "
                   "for **unique insights, creative angles, or specific nuances** that the high-confidence models might "
                   "have missed. If a low-confidence model offers an inspiring point, **verify it logically** against the "
                   "high-confidence consensus. If it holds up, integrate it to enrich the final answer.")
    else:
        raise Exception(f"Unknown aggregation strategy: {agg_prompt}")
    
    # Check if system message exists
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": JISI_SYSTEM_PROMPT + "\n\n" + system}] + messages

    return messages


def _merge_streaming_chunks(raw_content_parts: List[str]) -> str:
    """
    Merge streaming chunks into a single JSON object.
    
    Args:
        raw_content_parts (List[str]): List of JSON strings containing each chunk.
        
    Returns:
        str: Merged JSON string.
    """
    if not raw_content_parts:
        return json.dumps({})
    
    chunks = []
    for part in raw_content_parts:
        if part:
            try:
                chunks.append(json.loads(part))
            except json.JSONDecodeError:
                continue
    
    if not chunks:
        return json.dumps({})
    
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
    
    for chunk in chunks:
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            
            for key, value in delta.items():
                if key == "content" and value:
                    content_parts.append(value)
                elif key in ["reasoning_content", "reasoning"] and value:
                    reasoning_parts.append(value)
                elif value is not None and key not in ["reasoning_details"]:
                    if key not in accumulated_delta:
                        accumulated_delta[key] = value
            
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    
        if "usage" in chunk and chunk["usage"]:
            result["usage"] = chunk["usage"]

    message = {}
    if content_parts:
        message["content"] = "".join(content_parts)
    else:
        message["content"] = None
    
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    
    for key, value in accumulated_delta.items():
        if key not in ["content", "reasoning_content", "reasoning", "reasoning_details"]:
            message[key] = value
    
    if "role" not in message:
        message["role"] = "assistant"
    
    merged_choice = {
        "finish_reason": finish_reason,
        "index": 0,
        "logprobs": None,
        "message": message
    }
    
    result["choices"] = [merged_choice]
    
    return json.dumps(result, ensure_ascii=False)


def generate_general(
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        streaming: bool = False,
        logprobs: Optional[bool] = None
) -> str:
    """
    General interface for synchronous generation without caching.
    
    Args:
        model (str): Model name.
        messages (List[Dict[str, Any]]): List of messages.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        streaming (bool): Whether to stream response.
        logprobs (Optional[bool]): Whether to return logprobs.
        
    Returns:
        str: Generated text response.
    """
    output = None
    print(f"Sending messages to `{model}`...")
    
    # Retry loop
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            model_config = MODELCONFIG.get(model)
            if not model_config:
                raise ValueError(f"Model {model} not found in configuration")
                
            if isinstance(model_config, list):
                model_config = random.choices(model_config, k=1)[0]

            effective_max_tokens = getattr(model_config, 'max_tokens', None) or max_tokens
                
            if isinstance(model_config, ApiConfig):
                # OpenAI Client
                client = OpenAI(api_key=model_config.get_api_key(),
                                base_url=model_config.base_url,
                                timeout=600)
                res = client.chat.completions.create(
                    model=model_config.full_model_name,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    stream=False
                )
                output = res.choices[0].message.content
                
            elif isinstance(model_config, ApiConfigPost):
                # POST Request
                api_key = model_config.get_api_key()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                data = {
                    "model": model_config.full_model_name,
                    "messages": messages,
                    'max_tokens': effective_max_tokens,
                    "temperature": temperature,
                    "streaming": streaming
                }
                response = requests.post(model_config.base_url, headers=headers, json=data, verify=False, timeout=600)
                output_json = response.json()
                if 'reasoning_content' in output_json['choices'][0]['message']:
                    output = ('<think>\n' + output_json['choices'][0]['message']['reasoning_content'] +
                              '\n</think>\n' + output_json['choices'][0]['message']['content'])
                else:
                    output = output_json['choices'][0]['message']['content']
            
            print(f"Get `{model}` Response !!!")
            break

        except Exception as e:
            logger.info(f"Detecting error when using model {model}: {e}")
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return ""
        
    return output.strip()


def generate_with_references(
        model: str,
        messages: List[Dict[str, Any]],
        references: List[str] = [],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        logprobs: Optional[bool] = None,
        agg_prompt: str = 'normal',
        ref_score: Optional[List[float]] = None
) -> str:
    """
    Generate response with injected references (synchronous, no cache).
    
    Args:
        model (str): Model name.
        messages (List[Dict[str, Any]]): List of messages.
        references (List[str]): List of reference strings.
        max_tokens (int): Maximum tokens.
        temperature (float): Temperature.
        logprobs (Optional[bool]): Return logprobs.
        agg_prompt (str): Aggregation prompt strategy.
        ref_score (Optional[List[float]]): Confidence scores for references.
        
    Returns:
        str: Generated text response.
    """
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references, agg_prompt=agg_prompt, ref_score=ref_score)
    return generate_general(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs
    )


def generate_general_with_cache(
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        streaming: bool = False, # Ignored parameter kept for signature compatibility if needed
        logprobs: Optional[bool] = None
):
    """
    General interface for API calls with caching support.
    
    Args:
        model (str): Model name.
        messages (List[Dict[str, Any]]): List of messages.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        streaming (bool): Whether to stream response (handled internally).
        logprobs (Optional[bool]): Whether to return logprobs.
        
    Returns:
        GeneratorOutput: Object containing output, usage stats, and raw response.
    """
    from generators.generator import GeneratorOutput

    output = None
    kwargs = {}
    if model == "deepseek-v3.2-thinking":
        kwargs = {"extra_body": {"thinking": {"type": "enabled"}}}

    print(f"Sending messages to `{model}`...")
    
    for sleep_time in [1, 8, 16, 32]:
        try:
            model_config = MODELCONFIG.get(model)
            if not model_config:
                # If model is not in config, we can't proceed. 
                # But check if it was intended to be passed directly? 
                # For now assume everything is in MODELCONFIG.
                raise ValueError(f"Model {model} not found in configuration")

            if isinstance(model_config, list):
                model_config = random.choices(model_config, k=1)[0]

            effective_max_tokens = getattr(model_config, 'max_tokens', None) or max_tokens

            # ---------------------------------------------------------
            # ApiConfig (OpenAI Client)
            # ---------------------------------------------------------
            if isinstance(model_config, ApiConfig):
                client = OpenAI(api_key=model_config.get_api_key(),
                                base_url=model_config.base_url,
                                timeout=1800)
                
                # Setup stream options
                if model_config.stream:
                    kwargs.update({"stream": True, "stream_options": {"include_usage": True}})
                else:
                    kwargs.update({"stream": False})
                
                if "gpt-5.2" in model:
                    kwargs.update({"reasoning_effort": "high"})

                res = client.chat.completions.create(
                    model=model_config.full_model_name,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                
                full_response = ""
                raw_response = None
                usage = argparse.Namespace(prompt_tokens=0, completion_tokens=0, cost=0.0)

                if model_config.stream:
                    content_parts = []
                    raw_content_parts = []
                    for chunk in res:
                        if chunk.choices:
                            content = chunk.choices[0].delta.content or ""
                            content_parts.append(content)
                        raw_content_parts.append(
                            chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else None)
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage = chunk.usage
                    
                    raw_response = _merge_streaming_chunks(raw_content_parts)
                    full_response = "".join(content_parts)
                else:
                    usage = res.usage
                    choices = res.choices
                    if not choices or choices[0].message.content is None:
                        raise ValueError("Empty response from LLM")
                    
                    raw_response = res.model_dump_json() if hasattr(res, 'model_dump_json') else None
                    full_response = choices[0].message.content

                result = GeneratorOutput(
                    output=full_response,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    cost=0.0,
                    raw_response=raw_response
                )
                print(f"Get `{model}` Response !!!")
                return result

            # ---------------------------------------------------------
            # ApiConfigPost (Requests)
            # ---------------------------------------------------------
            elif isinstance(model_config, ApiConfigPost):
                api_key = model_config.get_api_key()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                data = {
                    "model": model_config.full_model_name,
                    "messages": messages,
                    'max_tokens': effective_max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "streaming": model_config.stream
                }
                data.update(kwargs)
                
                response = requests.post(model_config.base_url, headers=headers, json=data, verify=False, timeout=1200)
                output_json = response.json()

                # Extract content
                output_text = ""
                if 'reasoning_content' in output_json['choices'][0]['message']:
                    output_text = ('<think>\n' + output_json['choices'][0]['message']['reasoning_content'] +
                              '\n</think>\n' + output_json['choices'][0]['message']['content'])
                else:
                    output_text = output_json['choices'][0]['message']['content']

                # Extract usage information
                usage_info = output_json.get('usage', {})
                prompt_tokens = usage_info.get('prompt_tokens', 0)
                completion_tokens = usage_info.get('completion_tokens', 0)

                result = GeneratorOutput(
                    output=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=0.0,
                    raw_response=json.dumps(output_json) if output_json else None
                )
                print(f"Get `{model}` Response !!!")
                return result
            
            else:
                raise NotImplementedError(f"Model Config type {type(model_config)} is not implemented")

        except Exception as e:
            logger.info(f"Detecting error when using model {model}: {e}")
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return GeneratorOutput(
        output=f"Generation failed: All retries exhausted for model {model}",
        prompt_tokens=0, completion_tokens=0, cost=0.0
    )


def generate_with_references_stream(
        model: str,
        messages: List[Dict[str, Any]],
        references: List[str] = [],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        logprobs: Optional[bool] = None,
        agg_prompt: str = 'normal',
        ref_score: Optional[List[float]] = None
) -> Generator[str, None, None]:
    """
    Stream aggregation-model output chunk by chunk without using cache.
    """
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references, agg_prompt=agg_prompt, ref_score=ref_score)

    model_config = MODELCONFIG.get(model)
    if not model_config:
        yield f"[Error: Model {model} not found]"
        return
    if isinstance(model_config, list):
        model_config = random.choices(model_config, k=1)[0]

    effective_max_tokens = getattr(model_config, 'max_tokens', None) or max_tokens

    if isinstance(model_config, ApiConfig):
        client = OpenAI(api_key=model_config.get_api_key(), base_url=model_config.base_url, timeout=1800)
        kwargs = {"stream": True, "stream_options": {"include_usage": True}}
        if "gpt-5.2" in model:
            kwargs["reasoning_effort"] = "high"
        if "deepseek-v3.2-thinking" in model:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        try:
            res = client.chat.completions.create(
                model=model_config.full_model_name,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            in_reasoning = False
            completion_tokens = 0
            for chunk in res:
                if chunk.usage and hasattr(chunk.usage, "completion_tokens"):
                    completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                content = getattr(delta, "content", None)
                if reasoning:
                    if not in_reasoning:
                        yield "<think>\n"
                        in_reasoning = True
                    yield reasoning
                if content:
                    if in_reasoning:
                        yield "\n</think>\n"
                        in_reasoning = False
                    yield content
            return {"completion_tokens": completion_tokens}
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"\n\n[Generation interrupted: {str(e)}]"
    else:
        result = generate_general_with_cache(
            model=model, messages=messages,
            max_tokens=effective_max_tokens, temperature=temperature, top_p=top_p,
            logprobs=logprobs
        )
        for i in range(0, len(result.output), 32):
            yield result.output[i:i + 32]
        return {"completion_tokens": result.completion_tokens}


def generate_with_references_with_cache(
        model: str,
        messages: List[Dict[str, Any]],
        references: List[str] = [],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        logprobs: Optional[bool] = None,
        agg_prompt: str = 'normal',
        ref_score: Optional[List[float]] = None
):
    """
    Generate response with injected references (cached version).
    
    Args:
        model (str): Model name.
        messages (List[Dict[str, Any]]): List of messages.
        references (List[str]): List of reference strings.
        max_tokens (int): Maximum tokens.
        temperature (float): Temperature.
        top_p (float): Nucleus sampling.
        logprobs (Optional[bool]): Return logprobs.
        agg_prompt (str): Aggregation prompt strategy.
        ref_score (Optional[List[float]]): Confidence scores for references.
        
    Returns:
        GeneratorOutput: Result object.
    """
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references, agg_prompt=agg_prompt, ref_score=ref_score)
    return generate_general_with_cache(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=logprobs
    )

