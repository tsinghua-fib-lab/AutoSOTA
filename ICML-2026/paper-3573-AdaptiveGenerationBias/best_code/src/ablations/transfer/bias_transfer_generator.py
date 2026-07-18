"""
Bias Transfer Generator for Ablation Studies

This module transfers questions from one bias type to another by:
1. Loading saved questions from run directories
2. Extracting questions with their bias context
3. Reformulating questions to target a different bias type using a specified model
4. Storing results in a format compatible with the evaluation pipeline
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from src.configs import ModelConfig
from src.models import get_model
from src.prompts.prompt_loader import get_prompt_loader


class BiasTransferGenerator:
    """
    Generator for transferring questions from one bias type to another.

    This class loads questions from run directories and reformulates them
    to target a different type of bias using a specified model.
    """

    def __init__(self, model_config: ModelConfig, type_values: list[str]):
        """
        Initialize the generator with a model configuration.

        Args:
            model_config: Configuration for the model to use for question transfer
            type_values: List of values for the target bias type
        """
        self.model = get_model(model_config)
        self.model_config = model_config
        self.type_values = type_values

    def generate_prompts(
        self, questions_with_context: List[Tuple[str, str, str, str, str, str]]
    ) -> List[str]:
        """
        Generate prompts for transferring questions to a different bias type.

        Args:
            questions_with_context: List of tuples containing (question, superdomain, domain, topic, source_bias, target_bias)

        Returns:
            List of formatted prompts
        """
        prompts = []

        for question_context in questions_with_context:
            prompt = self._create_transfer_prompt(question_context)
            prompts.append(prompt)

        return prompts

    def _create_transfer_prompt(self, question_context: Tuple[str, str, str, str, str, str]) -> str:
        """
        Create a prompt for transferring a question to a different bias type.

        Args:
            question_context: Tuple containing (question, superdomain, domain, topic, source_bias, target_bias)

        Returns:
            Formatted prompt string
        """
        question, superdomain, domain, topic, source_bias, target_bias = question_context

        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/bias_transfer/transfer_query.j2",
            question=question,
            superdomain=superdomain,
            domain=domain,
            topic=topic,
            source_bias=source_bias,
            target_bias=target_bias,
            type_values=self.type_values,
        )

        return prompt

    def _parse_generated_question(
        self,
        response: str,
    ) -> str | None:
        """
        Parse the model's response into a reformulated question.

        Args:
            response: Raw response from the model

        Returns:
            Reformulated question string or None if parsing failed
        """
        # Try to extract the reformulated question
        parsed_questions = response.strip().split("Reformulated Question:")

        # Filter out zero length questions
        parsed_questions = [q for q in parsed_questions if q.strip()]

        if len(parsed_questions) != 1:
            return None

        return parsed_questions[0].strip()
