from typing import List, Dict

import os
from together import Together
from src.configs import ModelConfig

from .model import APIModel


class TogetherModel(APIModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 600

    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=input,
            **self.config.args,
        )

        return response.choices[0].message.content
