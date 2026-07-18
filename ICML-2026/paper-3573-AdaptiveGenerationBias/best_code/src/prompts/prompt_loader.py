"""
Core prompt loading functionality with support for static files and Jinja2 templates.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from src.bias_pipeline.questionaires.questionaire import Question
from functools import lru_cache
import logging

try:
    from jinja2 import Environment, FileSystemLoader, Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available. Template functionality will be limited.")


class PromptLoader:
    """
    Centralized prompt loading system with caching and template support.

    Features:
    - Lazy loading of prompts
    - LRU cache for performance
    - Support for both static files and Jinja2 templates
    - Environment-based prompt variations
    - Validation and error handling
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the prompt loader.

        Args:
            base_path: Base directory for prompts. Defaults to src/prompts/
        """
        if base_path is None:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            self.base_path = current_dir
        else:
            self.base_path = Path(base_path)

        self.templates_path = self.base_path / "templates"
        self.static_path = self.base_path / "static"

        # Initialize Jinja2 environment if available
        if JINJA2_AVAILABLE and self.templates_path.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_path)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja_env = None

        self._cache = {}

    @lru_cache(maxsize=128)
    def get_static_prompt(self, prompt_path: str) -> str:
        """
        Load a static prompt from file.

        Args:
            prompt_path: Relative path to the prompt file from static/ directory

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            IOError: If there's an error reading the file
        """
        full_path = self.static_path / prompt_path

        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except IOError as e:
            raise IOError(f"Error reading prompt file {full_path}: {e}")

    def get_template(self, template_path: str, **kwargs) -> str:
        """
        Load and render a Jinja2 template.

        Args:
            template_path: Relative path to the template file from templates/ directory
            **kwargs: Variables to pass to the template

        Returns:
            The rendered template as a string

        Raises:
            RuntimeError: If Jinja2 is not available
            FileNotFoundError: If the template file doesn't exist
            Exception: If there's an error rendering the template
        """
        if not JINJA2_AVAILABLE or self.jinja_env is None:
            raise RuntimeError("Jinja2 templates not available. Install jinja2 package.")

        try:
            template = self.jinja_env.get_template(template_path)
            return template.render(**kwargs).strip()
        except Exception as e:
            raise Exception(f"Error rendering template {template_path}: {e}")

    def get_json_data(self, json_path: str) -> Dict[str, Any]:
        """
        Load JSON data from file.

        Args:
            json_path: Relative path to the JSON file from static/ directory or project root

        Returns:
            The parsed JSON data

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        # First try relative to static directory
        full_path = self.static_path / json_path

        # If not found in static, try relative to project root
        if not full_path.exists():
            # Get project root (go up from src/prompts to project root)
            project_root = self.base_path.parent.parent
            full_path = project_root / json_path

        if not full_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {json_path} (tried static/ and project root)"
            )

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {full_path}: {e}")

    def prompt_exists(self, prompt_path: str, is_template: bool = False) -> bool:
        """
        Check if a prompt file exists.

        Args:
            prompt_path: Relative path to the prompt file
            is_template: Whether to check in templates/ or static/ directory

        Returns:
            True if the file exists, False otherwise
        """
        if is_template:
            full_path = self.templates_path / prompt_path
        else:
            full_path = self.static_path / prompt_path

        return full_path.exists()

    def list_prompts(self, subdirectory: str = "", is_template: bool = False) -> list[str]:
        """
        List available prompt files in a directory.

        Args:
            subdirectory: Subdirectory to search in
            is_template: Whether to search in templates/ or static/ directory

        Returns:
            List of available prompt file paths
        """
        if is_template:
            search_path = self.templates_path / subdirectory
        else:
            search_path = self.static_path / subdirectory

        if not search_path.exists():
            return []

        prompts = []
        for file_path in search_path.rglob("*"):
            if file_path.is_file():
                # Get relative path from the search directory
                rel_path = file_path.relative_to(search_path)
                prompts.append(str(rel_path))

        return sorted(prompts)

    def clear_cache(self):
        """Clear the prompt cache."""
        self.get_static_prompt.cache_clear()
        self._cache.clear()

    def get_cache_info(self):
        """Get cache statistics."""
        return self.get_static_prompt.cache_info()

    def create_refine_prompts(
        self,
        questions: List[Question],
        attribute: str = None,
        question_config=None,
        good_question_samples=None,
    ):
        """
        Create refinement prompts for a list of questions.

        Args:
            questions: List of Question objects to create refinement prompts for
            attribute: The bias attribute being tested (e.g., 'gender', 'race')
            question_config: QuestionConfig object containing type_values and type_examples

        Returns:
            List of tuples containing (system_prompt, query, question, prompt_type) for each question
        """
        prompts = []

        # Extract type information from question config
        type_values = getattr(question_config, "type_values", []) if question_config else []
        type_examples = getattr(question_config, "type_examples", []) if question_config else []

        for question in questions:
            try:
                system_prompt = self.get_template(
                    "refinement/refine/refine_system.j2",
                    attribute=attribute or "bias",
                    type_values=type_values,
                    type_examples=type_examples,
                )

                query = self.get_template(
                    "refinement/refine/refine_query.j2",
                    pos_examples=[],  # These would be filled by the caller if needed
                    neg_examples=[],  # These would be filled by the caller if needed
                    question=question,  # Pass the question object directly
                    gen_pos_examples=good_question_samples or [],
                    reference_setting=(question.superdomain, question.domain, question.topic),
                    attribute=attribute or "bias",
                    type_values=type_values,
                    type_examples=type_examples,
                )

                prompts.append((system_prompt, query, question, "refine"))

            except Exception as e:
                print(f"Error creating refine prompt for question: {e}")
                continue

        return prompts

    def create_replace_prompts(
        self,
        questions,
        attribute: str = None,
        similar_questions=None,
        question_config=None,
        good_question_samples=None,
    ):
        """
        Create replacement prompts for a list of questions.

        Args:
            questions: List of Question objects to create replacement prompts for
            attribute: The bias attribute being tested
            similar_questions: Optional list of similar questions for context
            question_config: QuestionConfig object containing type_values and type_examples

        Returns:
            List of tuples containing (system_prompt, query, question, prompt_type) for each question
        """
        prompts = []

        # Extract type information from question config
        type_values = getattr(question_config, "type_values", []) if question_config else []
        type_examples = getattr(question_config, "type_examples", []) if question_config else []

        for question in questions:
            try:
                system_prompt = self.get_template(
                    "refinement/replace/replace_system.j2",
                    attribute=attribute or "bias",
                    type_values=type_values,
                    type_examples=type_examples,
                )

                query = self.get_template(
                    "refinement/replace/replace_query.j2",
                    attribute=attribute or "bias",
                    original_question=question,
                    similar_questions=similar_questions or [],
                    gen_pos_examples=good_question_samples or [],
                )

                prompts.append((system_prompt, query, question, "replace"))

            except Exception as e:
                print(f"Error creating replace prompt for question: {e}")
                continue

        return prompts

    def create_new_topic_prompts(
        self,
        domain_keys: tuple[str, str],
        attribute: str = None,
        existing_topics=None,
        high_performing_examples=None,
        good_question_samples=None,
        question_config=None,
    ):
        """
        Create new topic generation prompts for a list of domains.

        Args:
            domain_keys: List of domain keys (format: "superdomain::domain")
            attribute: The bias attribute being tested
            existing_topics: Dict mapping domain keys to lists of existing topics
            high_performing_examples: Dict mapping domain keys to high-performing questions
            question_config: QuestionConfig object containing type_values, type_examples, and examples_path

        Returns:
            List of tuples containing (system_prompt, query, domain_key, prompt_type) for each domain
        """
        prompts = []

        # Extract type information from question config
        type_values = getattr(question_config, "type_values", []) if question_config else []
        type_examples = getattr(question_config, "type_examples", []) if question_config else []

        for domain_key in domain_keys:
            try:
                superdomain, domain_name = domain_key

                system_prompt = self.get_template(
                    "refinement/generation/generate_new_topic_system.j2",
                    attribute=attribute or "bias",
                    type_values=type_values,
                    type_examples=type_examples,
                )

                query = self.get_template(
                    "refinement/generation/generate_new_topic_query.j2",
                    attribute=attribute or "bias",
                    domain_key=domain_key,
                    superdomain=superdomain,
                    domain_name=domain_name,
                    type_examples=type_examples,
                    type_values=type_values,
                    existing_topics=existing_topics.get(domain_key, []) if existing_topics else [],
                    high_performing_examples=high_performing_examples
                    if high_performing_examples
                    else [],
                    gen_pos_examples=good_question_samples,
                    domain_context="",
                )

                prompts.append((system_prompt, query, domain_key, "new_topic"))

            except Exception as e:
                print(f"Error creating new topic prompt for domain {domain_key}: {e}")
                continue

        return prompts


# Global instance for easy access
_default_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get the default prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader
