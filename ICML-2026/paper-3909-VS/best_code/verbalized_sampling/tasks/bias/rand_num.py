# Copyright 2025 CHATS-Lab. All Rights Reserved.
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


from ..base import BaseTask


class RandomNumberTask(BaseTask):
    """Task for generating random numbers."""

    def __init__(self, **kwargs):
        """
        Initialize the StateNameTask.

        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "rand_num",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate a random number between 0 and 100.",
        }

    @property
    def task_type(self) -> str:
        return "rand_num"

    # def __init__(self, format: str = "direct"):
    #     self.format = format

    # def get_prompt(self, num_samples: int = 1) -> str:
    #     """Get the prompt for the task."""
    #     return "Generate a random number between 0 and 100."

    # def parse_response(self, response: str) -> Any:
    #     """Parse the model's response."""
    #     import json

    #     if self.format in ["structure", "vs_standard"]:
    #         try:
    #             if isinstance(response, str):
    #                 # Remove markdown code block wrappers if present
    #                 json_block = "```json"
    #                 code_block = "```"

    #                 # Find the actual JSON content
    #                 if json_block in response:
    #                     content = response[response.find(json_block) + len(json_block):]
    #                     if code_block in content:
    #                         content = content[:content.find(code_block)]
    #                 elif code_block in response:
    #                     content = response[response.find(code_block) + len(code_block):]
    #                     if code_block in content:
    #                         content = content[:content.rfind(code_block)]
    #                 else:
    #                     content = response

    #                 # Clean up the content
    #                 content = content.strip()

    #                 # Try to find the first { and last } to get just the JSON object
    #                 start = content.find('{')
    #                 end = content.rfind('}') + 1
    #                 if start != -1 and end != 0:
    #                     content = content[start:end]

    #                 # Remove trailing commas before closing brackets/braces
    #                 content = content.replace(',\n    ]', '\n    ]')
    #                 content = content.replace(',\n    }', '\n    }')
    #                 content = content.replace(',\n]', '\n]')
    #                 content = content.replace(',\n}', '\n}')

    #                 return json.loads(content)
    #             return response
    #         except json.JSONDecodeError:
    #             return None
    #     elif self.format == "seq":
    #         try:
    #             if isinstance(response, str):
    #                 # First try to parse as JSON array
    #                 start = response.find('[')
    #                 end = response.rfind(']') + 1
    #                 if start != -1 and end != 0:
    #                     try:
    #                         return json.loads(response[start:end])
    #                     except json.JSONDecodeError:
    #                         pass
    #         except ValueError:
    #             pass
    #         return None
    #     elif self.format == "direct":
    #         if isinstance(response, str):
    #             try:
    #                 if ":" in response:
    #                     return response.split(":")[1].strip()
    #                 else:
    #                     return response.strip()
    #             except ValueError:
    #                 return None
    #         return None
