from typing import List

import yaml

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt


def abstract_criteria(
    criteria: List[str],
    artifact_type: str,
    artifact_topic: str,
    num_abstractions: List[int],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Abstract the criteria into a more general form.

    Args:
        criteria: The criteria to abstract
        artifact_type: The type of artifact that the criteria are assessing
        artifact_topic: The broad topic that the artifacts focus on
        num_abstractions: The number of times to abstract each criterion
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of abstracted criteria objects
    """
    system_prompt = load_prompt("abstract_criteria.yaml")
    criteria_yaml_obj = [
        {
            "criterion_id": i,
            "criterion": c["description"],
            "checklist": c["checklist"],
            "number_of_abstractions": num_abstractions[i]
        } for i, c in enumerate(criteria)
    ]
    criteria_str = "```yaml\ncriteria:\n" + yaml.dump(criteria_yaml_obj, sort_keys=False, indent=2) + "```"
    formatted_prompt = (
        f"# Criteria\n\n"
        f"{criteria_str}\n\n"
        f"# Artifact Details\n\n"
        f"**Type:** {artifact_type}\n\n"
        f"**Topic:** {artifact_topic}\n\n"
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

