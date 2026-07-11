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


class GSM8KTask(BaseTask):
    """Task for generating synthetic data to the GSM8K dataset."""

    def __init__(self, **kwargs):
        """
        Initialize the GSM8KTask.

        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "gsm8k",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Generate synthetic data to the GSM8K dataset",
        }

    @property
    def task_type(self) -> str:
        return "gsm8k"
