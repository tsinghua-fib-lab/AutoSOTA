from typing import Any, Dict, List

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import format_chat_history


def slice_chat_history(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Slice the chat history up to the given index.
    """
    return format_chat_history(chat_history[:1] + "\n\n[... ABRIDGED ...]\n\n" + format_chat_history(chat_history[-3:]))


def format_criteria_and_laterals(criteria_objs: List[Dict[str, Any]], evaluation_type: str) -> str:
    """
    Format the criteria and lateral alternatives into a string.
    """
    criteria_str = ""
    for i, c in enumerate(criteria_objs):
        for j, a in enumerate(c['abstractions']):
            if a['awareness'] == "Ignore":
                continue
            if evaluation_type == "probing" and "Fully" in a['awareness']:
                continue
            if evaluation_type == "satisfaction" and "Fully" in a['satisfaction']:
                continue
            if len(a['lateral_alternatives']) == 0:
                continue
            criteria_str += f"- criterion_id: {i + 1}\n"
            criteria_str += f"  unsatisfied_criterion: {a['version']}\n"
            criteria_str += "  lateral_alternatives:\n"
            for k, alt in enumerate(a['lateral_alternatives']):
                criteria_str += f"    - alternative_id: {k + 1}\n"
                criteria_str += f"      alternative: {alt['alternative']}\n"
            break
    return criteria_str


def evaluate_message_on_lateral(
    chat_history: List[Dict[str, str]],
    criteria_objs: List[Dict[str, Any]],
    evaluation_type: str,
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Evaluate the satisfaction/relevance of the AI assistant's last message based on the lateral alternatives of the criteria.

    Args:
        chat_history: The chat history between the user and the AI assistant
        criteria_objs: The list of criteria objects that the user is interested in
        evaluation_type: The type of evaluation to perform (probing or satisfaction)
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        verbose: Whether to print verbose output
        max_retries: Maximum number of retries if the evaluation fails
    Returns:
        List of evaluations, evaluation type
    """
    system_prompt = load_prompt("evaluate_message_on_lateral.yaml")

    assistant_message = chat_history[-1]['content']

    # Find if it contains "**ACTION = DIVERGE**" or "**ACTION = CONVERGE**". If so, remove it
    # Or other similar tags like "ACTION = DIVERGE" or "ACTION = CONVERGE".
    if "**ACTION = DIVERGE**" in assistant_message:
        assistant_message = assistant_message.replace("**ACTION = DIVERGE**", "")
    elif "**ACTION = CONVERGE**" in assistant_message:
        assistant_message = assistant_message.replace("**ACTION = CONVERGE**", "")
    elif "ACTION = DIVERGE" in assistant_message:
        assistant_message = assistant_message.replace("ACTION = DIVERGE", "")
    elif "ACTION = CONVERGE" in assistant_message:
        assistant_message = assistant_message.replace("ACTION = CONVERGE", "")

    assistant_message = assistant_message.strip()

    criteria_and_laterals = format_criteria_and_laterals(criteria_objs, evaluation_type)
    # If there are no eligible criteria (e.g., all abstractions are fully satisfied/aware,
    # ignored, or have no lateral alternatives), skip the lateral evaluation entirely.
    # This avoids prompting the model with an empty "Criteria and Lateral Alternatives" section,
    # which can cause brittle YAML outputs and unnecessarily fail the whole run.
    if not criteria_and_laterals.strip():
        return []

    formatted_prompt = (
        f"# Initial User Request\n\n"
        f"{chat_history[0]['content']}\n\n"
        f"# Assistant's Last Message\n\n"
        f"{assistant_message}\n\n"
        f"# Evaluation Type\n\n"
        f"**{evaluation_type}**\n\n"
        f"# Criteria and Lateral Alternatives\n\n"
        f"{criteria_and_laterals}\n\n"
    )

    parsed_yaml, raw_output = generate_and_process_yaml(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=formatted_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose
    )

    return parsed_yaml['evaluations']

