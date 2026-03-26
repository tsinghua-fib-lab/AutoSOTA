import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any
import random
from retrying import retry


MODEL_ALIAS_MAP = {
    'gpt-4o': 'deepseek/deepseek-chat',
    'gpt-4o-mini': 'deepseek/deepseek-chat',
}

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIConnectionError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIConnectionError))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens, base_url=None) -> None:
        openai.api_key = API_KEY
        self.model_name = MODEL_ALIAS_MAP.get(model_name, model_name)
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        if base_url:
            openai.api_base = base_url
        

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."},
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1,
                stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        finish_reason = response['choices'][0]['finish_reason']
        return generated_text, finish_reason
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def generate(self, input_string, temperature = 0.0):
        if True:  # support all chat models
            try:
                return self.chat_generate(input_string, temperature)
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                raise
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant, one of the greatest AI scientists, logicians and mathematicians. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        finish_reason = [x['choices'][0]['finish_reason'].strip() for x in predictions]
        return [x['choices'][0]['message']['content'].strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0):
        if True:  # support all chat models
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text
