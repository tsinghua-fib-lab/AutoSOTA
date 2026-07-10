from typing import Any, Dict, List

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt


def create_abstraction_laterals(
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Create lateral alternatives for each abstraction level of each criterion.

    Args:
        criteria_objs: The criteria objects to create lateral alternatives for
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of criteria objects with lateral alternatives
    """
    system_prompt = load_prompt("create_abstraction_laterals.yaml")
    criteria_str = ""
    for i, c_obj in enumerate(criteria_objs):
        criteria_str += f"- criterion_id: {i+1}\n"
        criteria_str += f"  criterion: \"{c_obj['criterion']}\"\n"
        criteria_str += "  abstractions:\n"
        for j, a in enumerate(c_obj['abstractions']):
            criteria_str += f"    - abstraction_id: {j}\n"
            criteria_str += f"      abstraction: \"{a['version']}\"\n"

    formatted_prompt = (
        f"# Criteria\n\n"
        f"{criteria_str}\n\n"
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

    return parsed_yaml['results']

