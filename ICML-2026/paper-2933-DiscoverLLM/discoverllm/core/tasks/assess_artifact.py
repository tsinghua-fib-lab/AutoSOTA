from typing import Any, Dict, List

from discoverllm.core.generate import generate_and_process_json
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import format_chat_history


def format_intents(criteria_objs: List[Dict[str, Any]]) -> str:
    intents_str = ""
    for i, c in enumerate(criteria_objs):
        intents_str += f"- Intent: {i + 1}: {c['criterion']}\n"
    return intents_str.strip()


def assess_artifact(
    chat_history: List[Dict[str, str]],
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5.1-2025-11-13",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Assess the artifact based on the chat history and criteria.

    Args:
        chat_history: The chat history between the user and the AI assistant
        criteria_objs: The list of criteria objects that the user is interested in
        MODEL_NAME: Name of the model to use
        TEMPERATURE: Sampling temperature
        MAX_TOKENS: Maximum tokens to generate

    Returns:
        Assessment of the artifact
    """
    system_prompt = load_prompt("assess_artifact.yaml")

    formatted_prompt = (
        f"<chat_history>\n\n"
        f"{format_chat_history(chat_history)}\n</chat_history>\n\n"
        f"<intents>\n\n"
        f"{format_intents(criteria_objs)}\n</intents>\n\n"
    )

    parsed_json, raw_output = generate_and_process_json(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=formatted_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose
    )

    return parsed_json

