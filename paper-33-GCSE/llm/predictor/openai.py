import json
import torch
from openai import OpenAI
from openai._types import NotGiven
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List

NOT_GIVEN = NotGiven()


class Predictor():

    def __init__(self,
                 organization=None,
                 api_key=None,
                 base_url=None,
                 **args
                 ):
        '''
        Predictor: OpenAI API预测器 (OpenAI API predictor)

        ### Args:

        `organization`: OpenAI组织名 (OpenAI organization name)

        `api_key`: OpenAI API密钥 (OpenAI API key)
        '''

        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(organization=self.organization, api_key=self.api_key, base_url=self.base_url)
    

    def predict(self, query: str = '', history: List = None, model: str = 'gpt-4o-mini', max_length=NOT_GIVEN, max_new_tokens=NOT_GIVEN, top_p: float = NOT_GIVEN, temperature=NOT_GIVEN):
        if history is None:
            history = []
        raw = history + [{"role": "user", "content": query}]
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=raw,
            max_tokens=max_length,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )

        message = completion.choices[0].message.content
        raw.append({"role": "assistant", "content": message})
        return message, raw

    def stream_chat(self, query: str = '', history: List = None, model: str = 'gpt-4o-mini', max_length=NOT_GIVEN, max_new_tokens=NOT_GIVEN, top_p: float = NOT_GIVEN, temperature=NOT_GIVEN):
        if history is None:
            history = []
        raw = history + [{"role": "user", "content": query}]

        completion = self.client.chat.completions.create(
            model=model,
            messages=raw,
            max_tokens=max_length,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )

        result = ''
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content is None:
                continue
            result += delta.content
            yield result, raw + [{"role": "assistant", "content": result}]
    
    def __call__(self, query: str = '', history: List = None, model: str = 'gpt-4o-mini', max_length=NOT_GIVEN, max_new_tokens=NOT_GIVEN, top_p: float = NOT_GIVEN, temperature=NOT_GIVEN):
        return self.predict(query, history, model, max_length, max_new_tokens, top_p, temperature)
