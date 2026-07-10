from typing import Any, Dict, List, Tuple

import yaml

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import filter_hierarchy


def format_criteria(criteria: List[str]) -> str:
    criteria_yaml_obj = []
    for i, c in enumerate(criteria):
        criteria_yaml_obj.append({
            "criterion_id": i + 1,
            "criterion": c
        })
    return yaml.dump(criteria_yaml_obj, sort_keys=False, indent=2)

def simulate_initial_request(
    artifact_type: str,
    artifact_topic: str,
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Simulate a user's initial request to the AI assistant based on the criteria.

    Args:
        artifact_type: The type of artifact that the user is trying to create
        artifact_topic: The broad topic that the artifacts focus on
        criteria_objs: The list of criteria objects that the user is interested in
        MODEL_NAME: Name of the model to use
        TEMPERATURE: Sampling temperature
        MAX_TOKENS: Maximum tokens to generate

    Returns:
        User's request
    """
    system_prompt = load_prompt("simulate_initial_request.yaml")

    def _is_leaf_node(node: Dict[str, Any]) -> bool:
        return "children" not in node or len(node["children"]) == 0

    # Get the most abstract and most specific checklist items
    most_abstract_versions_str = [ item['text'] for c in criteria_objs for item in c['hierarchy']]
    most_specific_versions_str = [
        item['text'] for c in criteria_objs for item in filter_hierarchy(c['hierarchy'], _is_leaf_node)
    ]
    formatted_prompt = (
        f"# Artifact Details\n\n"
        f"**Type:** {artifact_type}\n\n"
        f"**Topic:** {artifact_topic}\n\n"
        f"# Criteria\n\n"
        f"{format_criteria(most_abstract_versions_str)}\n\n"
        f"# Latent Requirements\n\n"
        f"{format_criteria(most_specific_versions_str)}\n\n"
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

    selected_indices = [ c['criterion_id'] - 1 for c in parsed_yaml['selected_criteria'] ]
    if 'redundant_criteria' in parsed_yaml:
        redundant_indices = [ c['criterion_id'] - 1 for c in parsed_yaml['redundant_criteria'] ]
    else:
        redundant_indices = []
    current_idx = 0
    for i in range(len(criteria_objs)):
        for j in range(len(criteria_objs[i]["hierarchy"])):
            if current_idx not in selected_indices and current_idx not in redundant_indices:
                current_idx += 1
                continue
            criteria_objs[i]["hierarchy"][j]["aware"] = 1
            criteria_objs[i]["hierarchy"][j]["update_thresholds"] = []
            current_idx += 1

    return parsed_yaml['initial_request'], criteria_objs
