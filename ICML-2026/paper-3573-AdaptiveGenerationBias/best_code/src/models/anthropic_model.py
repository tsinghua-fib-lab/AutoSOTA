from typing import List, Dict
import os
import anthropic
from src.configs import ModelConfig

from .model import APIModel


class AnthropicModel(APIModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 600

    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        system = ""
        for message in input:
            if message["role"] == "system":
                system = message["content"]
                input.remove(message)
                break

        response = self.client.messages.create(
            model=self.config.name,
            messages=input,
            **self.config.args,
            system=system,
        )

        return response.content[0].text
