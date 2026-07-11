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

from typing import Any, Dict, List, Optional

import tiktoken

from .base import BaseEvaluator, EvalResult, calculate_stats


class LengthEvaluator(BaseEvaluator):
    """Simple evaluator for computing token length of text responses using OpenAI tokenizer."""

    instance_plot_metrics = [("token_length", "violin")]
    aggregate_plot_metrics = ["avg_token_length"]
    key_plot_metrics = [
        ("avg_token_length", "Length (Token)"),
    ]

    def __init__(self, num_workers: int = 128):
        super().__init__("length", num_workers)
        # Get the tokenizer for the specified model
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def compute_instance_metric(self, prompt: str, response: Dict) -> Dict[str, float]:
        """Compute token length for a single response."""
        response_text = response.get("text", "")
        if isinstance(response_text, dict):
            response_text = str(response_text)

        return {
            "response": response_text,
            "prompt": prompt,
            "token_length": float(len(self.tokenizer.encode(response_text))),
        }

    def aggregate_metrics(self, instance_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate token lengths across all instances."""
        if not instance_metrics:
            return {}

        token_lengths = [metric["token_length"] for metric in instance_metrics]

        # Calculate basic stats
        length_stats = calculate_stats(token_lengths)

        return {
            # Basic metrics (backward compatible)
            "avg_token_length": length_stats["mean"],
            "min_token_length": length_stats["min"],
            "max_token_length": length_stats["max"],
            "total_tokens": sum(token_lengths),
            "num_responses": len(instance_metrics),
            # New error statistics
            "std_token_length": length_stats["std"],
            "med_token_length": length_stats["median"],
        }

    def evaluate(
        self, prompts: List[str], responses: List[Dict], metadata: Optional[Dict[str, Any]] = None
    ) -> EvalResult:
        """Evaluate responses for token length."""
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "evaluation_type": "token_length",
                "tokenizer_model": self.tokenizer.name,
                "num_responses": len(responses),
            }
        )

        return super().evaluate(prompts, responses, metadata)
