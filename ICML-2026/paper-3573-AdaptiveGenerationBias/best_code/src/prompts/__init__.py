"""
Prompt management system for better-bias project.

This module provides centralized prompt loading and management functionality
to improve code readability by separating business logic from prompt text.
"""

from .prompt_loader import PromptLoader

__all__ = ["PromptLoader"]
