from typing import List, Dict

from src.configs import ModelConfig
from .model import APIModel

from openai import OpenAI


class OpenAIGPT(APIModel):
    def __init__(self, config: ModelConfig, base_url: str = None, api_key: str = None):
        super().__init__(config)
        self.config = config

        if base_url is not None:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = OpenAI()

        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 600

        if "max_output_tokens" in self.config.args.keys():
            del self.config.args["max_tokens"]

    def _predict_call(self, input: List[Dict[str, str]], **kwargs) -> str:
        # Override the default args with any additional kwargs provided

        format = None

        if "response_format" in kwargs:
            format = kwargs["response_format"]
            del kwargs["response_format"]

        if "response_format" in self.config.args.keys():
            format = self.config.args["response_format"]

        combined_arguments = {**self.config.args, **kwargs}

        if "max_output_tokens" in self.config.args.keys():
            if format is not None:
                combined_arguments["text"]["format"] = {
                    "type": "json_schema",
                    "name": format["json_schema"]["name"],
                    "schema": format["json_schema"]["schema"],
                    "strict": format.get("strict", True),
                }

            if "response_format" in combined_arguments:
                del combined_arguments["response_format"]

            response = self.client.responses.create(
                model=self.config.name, input=input, **combined_arguments
            )
            return response.output_text
        else:
            if format is not None:
                combined_arguments["response_format"] = format
            response = self.client.chat.completions.create(
                model=self.config.name, messages=input, **combined_arguments
            )
            return response.choices[0].message.content
