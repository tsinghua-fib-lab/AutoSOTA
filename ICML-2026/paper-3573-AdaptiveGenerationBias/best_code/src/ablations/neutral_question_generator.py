"""
Neutral Question Generator for Ablation Studies

This module generates neutral questions from existing bias pipeline runs by:
1. Loading saved questions from run directories
2. Extracting superdomains, domains, and topics
3. Generating neutral questions about these topics using a specified model
4. Prefixing questions with "I am a person."
5. Storing results in a format compatible with the evaluation pipeline
"""

import json
import os
import random
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from src.configs import ModelConfig
from src.models import get_model
from src.prompts.prompt_loader import get_prompt_loader


class NeutralQuestionGenerator:
    """
    Generator for creating neutral questions from existing bias pipeline runs.

    This class loads questions from run directories, extracts their topics,
    and generates neutral versions using a specified model.
    """

    def __init__(self, model_config: ModelConfig, type_values: List[str]):
        """
        Initialize the generator with a model configuration.

        Args:
            model_config: Configuration for the model to use for question generation
            type_values: List of type values to use for generating neutral questions
        """
        self.model = get_model(model_config)
        self.model_config = model_config
        self.type_values = type_values

    def generate_prompts(
        self, topic_summaries: List[Tuple[str, str, str]], examples: List[Dict[str, str]]
    ) -> List[str]:
        prompts = []

        for summary in topic_summaries:
            selected_samples = random.sample(examples, min(3, len(examples)))
            prompt = self._create_generation_prompt(summary, selected_samples)
            prompts.append(prompt)

        return prompts

    def generate_prompts_from_topics(
        self,
        topics_data: List[Dict],
        examples: List[Dict[str, str]],
        num_questions_per_topic: int = 1,
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Generate prompts from topic mapping data loaded from files like topics.json.

        Args:
            topics_data: List of topic dictionaries with superdomain, domain, and topics
            examples: Example questions for context
            num_questions_per_topic: Number of questions to generate per topic

        Returns:
            List of formatted prompt strings
        """
        prompts = []
        topic_infos = []

        for domain_data in topics_data:
            superdomain = domain_data["superdomain"]
            domain = domain_data["domain"]

            for topic_info in domain_data["topics"]:
                topic = topic_info["topic"]
                description = topic_info.get("description", "")

                for _ in range(num_questions_per_topic):
                    selected_samples = random.sample(examples, min(3, len(examples)))
                    prompt = self._create_generation_prompt_from_topic_mapping(
                        superdomain, domain, topic, description, selected_samples
                    )
                    prompts.append(prompt)
                    topic_infos.append((superdomain, domain, topic))

        return prompts, topic_infos

    def generate_prompts_creative(
        self, num_questions: int, examples: List[Dict[str, str]]
    ) -> List[str]:
        """
        Generate prompts for completely new questions with creative superdomain/domain/topic combinations.

        Args:
            num_questions: Number of questions to generate
            examples: Example questions for context

        Returns:
            List of formatted prompt strings
        """
        prompts = []

        for _ in range(num_questions):
            selected_samples = random.sample(examples, min(3, len(examples)))
            prompt = self._create_creative_generation_prompt(selected_samples)
            prompts.append(prompt)

        return prompts

    def _create_generation_prompt(
        self, summary: Tuple[str, str, str], examples: List[Dict[str, str]]
    ) -> str:
        """
        Create a prompt for generating neutral questions about a topic.

        Args:
            summary: Topic summary to generate questions for

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/neutral_generation/neutral_query.j2",
            superdomain=summary[0],
            domain=summary[1],
            topic=summary[2],
            examples=examples,
        )

        return prompt

    def _create_generation_prompt_from_topic_mapping(
        self,
        superdomain: str,
        domain: str,
        topic: str,
        description: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Create a prompt for generating neutral questions from topic mapping data.

        Args:
            superdomain: The superdomain category
            domain: The domain category
            topic: The specific topic
            description: Description of the topic
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/neutral_generation/neutral_query_from_topics.j2",
            superdomain=superdomain,
            domain=domain,
            topic=topic,
            description=description,
            examples=examples,
        )

        return prompt

    def _create_creative_generation_prompt(self, examples: List[Dict[str, str]]) -> str:
        """
        Create a prompt for generating completely new questions with creative topics.

        Args:
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/neutral_generation/neutral_query_creative.j2",
            examples=examples,
        )

        return prompt

    def _parse_generated_question(
        self,
        response: str,
    ) -> str | None:
        """
        Parse the model's response into question objects.

        Args:
            response: Raw response from the model
            summary: Topic summary for context

        Returns:
            List of question dictionaries
        """
        # Try to parse as JSON first
        parsed_questions = response.strip().split("New Question:")

        # Filter out zero length questions
        parsed_questions = [q for q in parsed_questions if q.strip()]

        if len(parsed_questions) != 1:
            return None

        question = parsed_questions[0].strip()

        type_str = "/".join(self.type_values)

        question = "I am a {{" + type_str + "}}" + f" person. {question}"

        return question
