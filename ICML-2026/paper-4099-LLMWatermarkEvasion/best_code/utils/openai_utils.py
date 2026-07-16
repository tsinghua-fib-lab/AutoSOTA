# Copyright 2024 THU-BPM MarkLLM.
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

# =============================================
# openai_utils.py
# Description: Utility functions for OpenAI API
# =============================================
import os
import time
import openai
from exceptions.exceptions import OpenAIModelConfigurationError


class OpenAIAPI:
    """API class for OpenAI API."""
    def __init__(self, model, temperature, system_content):
        """
            Initialize OpenAI API with model, temperature, and system content.

            Parameters:
                model (str): Model name for OpenAI API.
                temperature (float): Temperature value for OpenAI API.
                system_content (str): System content for OpenAI API.
        """

        self.model = model
        self.temperature = temperature
        self.system_content = system_content
        self.client = openai.OpenAI()
        

        # List of supported models
        supported_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-5']

        # Check if the provided model is within the supported models
        if self.model not in supported_models:
            raise OpenAIModelConfigurationError(f"Unsupported model '{self.model}'. Supported models are {supported_models}.")

    def get_result_from_gpt4(self, query):
        """get result from GPT-4 model."""
        response = self.client.chat.completions.create(
            model="gpt-4-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result_from_gpt3_5(self, query):
        """get result from GPT-3.5 model."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response

    def get_result_from_gpt5(self, query):
        """get result from GPT-5 model."""
        response = self.client.chat.completions.create(
            model="gpt-5",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response

    def get_result(self, query):
        """get result from OpenAI API."""
        while True:
            try:
                if self.model == 'gpt-3.5-turbo':
                    result = self.get_result_from_gpt3_5(query)
                elif self.model == 'gpt-4':
                    result = self.get_result_from_gpt4(query)
                elif self.model == 'gpt-5':
                    result = self.get_result_from_gpt5(query)
                break
            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
            time.sleep(10)
        return result.choices[0].message.content