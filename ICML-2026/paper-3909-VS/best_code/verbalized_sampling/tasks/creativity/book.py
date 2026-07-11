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


class BookTask(BaseTask):
    """Task for generating book/novel continuations from prompts."""

    def __init__(self, **kwargs):
        """
        Initialize the BookTask.

        Args:
            num_prompts: Number of prompts to randomly sample from the dataset
            random_seed: Random seed for reproducible sampling
        """
        super().__init__(**kwargs)
        self.metadata = {
            "task_type": "book",
            "total_prompts": 0,
            "num_prompts": self.num_prompts,
            "random_seed": self.random_seed,
            "description": "Novel/book continuation task with prompts from literary works",
        }

    @property
    def task_type(self) -> str:
        return "book"

    # def get_metadata(self) -> dict:
    #     """Get task metadata."""
    #     return {
    #         "task_type": "book",
    #         "total_prompts": len(self._prompts) if self._prompts else 0,
    #         "num_prompts": self.num_prompts,
    #         "random_seed": self.random_seed,
    #         "description": "Novel/book continuation task with prompts from literary works"
    #     }
