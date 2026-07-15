"""
Taboo Baseline Evaluation

This module implements a "taboo"-style intrinsic LLM capability evaluation.
Instead of using SelfIE whitebox interpretability, we ask the LLM to describe
topics without mentioning their names (like the game Taboo), then use embedding
retrieval to measure how well those descriptions identify the topics.

This serves as a skyline/baseline for comparing against SelfIE-based approaches:
- If the LLM can't describe a topic well enough to retrieve it, it probably
  doesn't know the topic well (useful for filtering)
- The difference between taboo eval and SelfIE eval shows the value-add of
  the interpretability approach
"""

from .qwen_inference import AsyncQwenInference
from .taboo_generator import TabooDescriptionGenerator

__all__ = ["AsyncQwenInference", "TabooDescriptionGenerator"]
