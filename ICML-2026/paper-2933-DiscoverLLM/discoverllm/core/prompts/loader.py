"""
Loader for the YAML prompt templates in this directory.

Every YAML in ``core/prompts/`` is expected to have a top-level
``system_prompt`` key whose value is the actual template string. Callers do:

    from discoverllm.core.prompts import load_prompt
    system_prompt = load_prompt("create_criteria.yaml")
"""

import os

import yaml

_PROMPTS_DIR = os.path.dirname(__file__)


def load_prompt(prompt_name: str) -> str:
    """Load a prompt YAML by filename and return its ``system_prompt`` value."""
    with open(os.path.join(_PROMPTS_DIR, prompt_name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]
