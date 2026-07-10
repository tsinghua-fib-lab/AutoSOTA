import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
# Fix: httpx (used by openai) does not support socks5h proxy scheme.
# Clear ALL_PROXY so the async client can connect directly.
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
from openai import OpenAI, AsyncOpenAI

from AgentTailor.llm.format import Message
from AgentTailor.llm.price import cost_count, cost_count_by_tokens
from AgentTailor.llm.llm import LLM
from AgentTailor.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
# Prefer standard OpenAI env vars; keep legacy BASE_URL/API_KEY as fallback.
MINE_BASE_URL = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("BASE_URL")
)
MINE_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_API")
    or os.getenv("API_KEY")
)


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    client = AsyncOpenAI(base_url = MINE_BASE_URL, api_key = MINE_API_KEY,)
    chat_completion = await client.chat.completions.create(messages = msg,model = model,)
    #print(type(chat_completion))
    print(chat_completion)
    response = chat_completion.choices[0].message.content
    #response=chat_completion["choices"][0]["message"]["content"]
    
    # Extract usage and count tokens
    usage = chat_completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    
    # If usage is missing, try to estimate from messages
    if prompt_tokens == 0:
        import tiktoken
        try:
            encoder = tiktoken.encoding_for_model(model)
            prompt_text = "\n".join([msg.get('content', '') if isinstance(msg, dict) else str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in msg])
            prompt_tokens = len(encoder.encode(prompt_text))
            completion_tokens = len(encoder.encode(response)) if response else 0
        except:
            prompt_tokens = 0
            completion_tokens = 0
    
    # Count tokens and cost: prefer API usage, fallback to local estimation.
    prompt_text = "\n".join([msg.get('content', '') if isinstance(msg, dict) else str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in msg])
    if usage is not None:
        cost_count_by_tokens(prompt_tokens, completion_tokens, model)
    else:
        cost_count(prompt_text, response, model)
    
    return response
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        if isinstance(messages, str):
            messages = [{'role':'user', 'content':messages}]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass