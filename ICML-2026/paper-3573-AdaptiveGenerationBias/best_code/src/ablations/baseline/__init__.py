"""
Baseline Question Generation Module

This module provides functionality for generating baseline biased questions
that are designed to reveal implicit bias through natural, realistic scenarios.
"""

from .baseline_question_generator import BaselineQuestionGenerator
from .generate_baseline import generate_baseline_questions

__all__ = ["BaselineQuestionGenerator", "generate_baseline_questions"]
