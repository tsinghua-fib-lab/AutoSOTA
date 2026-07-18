"""
Baseline Question Generator for Ablation Studies

This module generates baseline biased questions that are designed to reveal implicit bias by:
1. Loading saved questions from run directories OR loading topic mappings from files
2. Extracting or using superdomain, domain, and topic combinations
3. Generating bias-revealing questions using templates similar to refinement generation
4. Creating questions with proper attribute templating (e.g., {{male/female}})
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


class BaselineQuestionGenerator:
    """
    Generator for creating baseline biased questions from existing bias pipeline runs
    or from topic mapping files.

    This class generates questions designed to reveal implicit bias through natural,
    realistic scenarios that prompt genuine responses.
    """

    def __init__(self, model_config: ModelConfig, type_values: List[str], attribute: str):
        """
        Initialize the generator with a model configuration.

        Args:
            model_config: Configuration for the model to use for question generation
            type_values: List of type values to use for generating questions (e.g., ["male", "female"])
            attribute: The bias attribute being tested (e.g., "gender", "race")
        """
        self.model = get_model(model_config)
        self.model_config = model_config
        self.type_values = type_values
        self.attribute = attribute

    def _load_response_format(self, generation_format: str) -> Dict:
        """
        Load the appropriate JSON schema based on the generation format.

        Args:
            generation_format: The generation format ("from_question", "from_topic", "from_domain", "from_superdomain")

        Returns:
            Dictionary containing the response format configuration
        """
        # Map generation formats to schema files
        schema_mapping = {
            "from_question": "from_question.json",
            "from_topic": "from_topic.json",
            "from_domain": "from domain.json",  # Note: space in filename
            "from_superdomain": "from_superdomain.json",
        }

        if generation_format not in schema_mapping:
            raise ValueError(f"Unknown generation_format: {generation_format}")

        schema_file = schema_mapping[generation_format]
        schema_path = f"src/prompts/schemas/ablations/baseline/{schema_file}"

        try:
            with open(schema_path, "r") as f:
                schema_data = json.load(f)
            return schema_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in schema file: {schema_path}")

    def _get_model_with_format(self, generation_format: str):
        """
        Get a model instance with the appropriate response format for the generation format.

        Args:
            generation_format: The generation format to use

        Returns:
            Model instance configured with the appropriate response format
        """
        # Create a copy of the model config
        model_config = ModelConfig(**self.model_config.dict())

        # Load and set the response format
        response_format = self._load_response_format(generation_format)

        # Update the model args with the response format
        if "args" not in model_config.dict():
            model_config.args = {}
        model_config.args["response_format"] = response_format

        # Return a new model instance with the updated config
        return get_model(model_config)

    def generate_prompts(
        self, topic_summaries: List[Tuple[str, str, str]], examples: List[Dict[str, str]]
    ) -> List[str]:
        """Generate prompts from existing run topic summaries"""
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
        generation_format: str = "from_question",
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Generate prompts from topic mapping data loaded from files like topics.json.

        Args:
            topics_data: List of topic dictionaries with superdomain, domain, and topics
            examples: Example questions for context
            num_questions_per_topic: Number of questions to generate per topic
            generation_format: Format to use ("from_question", "from_topic", "from_domain", "from_superdomain")

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

                for _ in range(num_questions_per_topic):
                    selected_samples = random.sample(examples, min(3, len(examples)))

                    if generation_format == "from_question":
                        prompt = self._create_generation_prompt_from_question(
                            superdomain, domain, topic, selected_samples
                        )
                        topic_infos.append((superdomain, domain, topic))
                    elif generation_format == "from_topic":
                        prompt = self._create_generation_prompt_from_topic(
                            superdomain, domain, selected_samples
                        )
                        topic_infos.append((superdomain, domain))
                    elif generation_format == "from_domain":
                        prompt = self._create_generation_prompt_from_domain(
                            superdomain, selected_samples
                        )
                        topic_infos.append((superdomain,))
                    elif generation_format == "from_superdomain":
                        prompt = self._create_generation_prompt_from_superdomain(selected_samples)
                    else:
                        raise ValueError(f"Unknown generation_format: {generation_format}")

                    prompts.append(prompt)

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
        Create a prompt for generating baseline questions about a topic from existing runs.

        Args:
            summary: Topic summary to generate questions for (superdomain, domain, topic)
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/baseline/baseline_query.j2",
            superdomain=summary[0],
            domain_key=summary[1],
            topic=summary[2],
            attribute=self.attribute,
            type_values=self.type_values,
            examples=examples,
        )

        return prompt

    def _create_generation_prompt_from_question(
        self,
        superdomain: str,
        domain: str,
        topic: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Create a prompt for generating baseline questions from specific topic/question data.

        Args:
            superdomain: The superdomain category
            domain: The domain category
            topic: The specific topic
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/baseline/baseline_query_from_question.j2",
            superdomain=superdomain,
            domain_key=domain,
            topic=topic,
            attribute=self.attribute,
            type_values=self.type_values,
            examples=examples,
        )

        return prompt

    def _create_generation_prompt_from_topic(
        self,
        superdomain: str,
        domain: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Create a prompt for generating baseline questions from topic level.

        Args:
            superdomain: The superdomain category
            domain: The domain category
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/baseline/baseline_query_from_topic.j2",
            superdomain=superdomain,
            domain=domain,
            attribute=self.attribute,
            type_values=self.type_values,
            examples=examples,
        )

        return prompt

    def _create_generation_prompt_from_domain(
        self,
        superdomain: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Create a prompt for generating baseline questions from domain level.

        Args:
            superdomain: The superdomain category
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/baseline/baseline_query_from_domain.j2",
            superdomain=superdomain,
            attribute=self.attribute,
            type_values=self.type_values,
            examples=examples,
        )

        return prompt

    def _create_generation_prompt_from_superdomain(
        self,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Create a prompt for generating baseline questions from superdomain level.

        Args:
            examples: Example questions for context

        Returns:
            Formatted prompt string
        """
        prompt_loader = get_prompt_loader()

        prompt = prompt_loader.get_template(
            "ablations/baseline/baseline_query_from_superdomain.j2",
            attribute=self.attribute,
            type_values=self.type_values,
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
            "ablations/baseline/baseline_query_from_superdomain.j2",
            attribute=self.attribute,
            type_values=self.type_values,
            examples=examples,
        )

        return prompt

    def _parse_generated_question(
        self,
        response: str,
    ) -> Dict[str, str] | None:
        """
        Parse the model's response into question objects with metadata.

        Args:
            response: Raw response from the model

        Returns:
            Parsed question dict with metadata or None if parsing failed
        """
        # Try to parse as JSON first
        try:
            parsed = json.loads(response)
            if "question" in parsed:
                return parsed
        except:
            pass

        # Fallback: look for "New Question:" format
        parsed_questions = response.strip().split("New Question:")

        # Filter out zero length questions
        parsed_questions = [q for q in parsed_questions if q.strip()]

        if len(parsed_questions) != 1:
            return None

        question = parsed_questions[0].strip()

        # Remove any trailing periods or quotes
        question = question.strip(".\"'")

        return {"question": question}
