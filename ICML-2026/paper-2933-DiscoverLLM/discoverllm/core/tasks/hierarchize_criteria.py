import random
from typing import Any, Dict, List

import yaml

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import update_hierarchy


def hierarchize_criteria(
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Hierarchize the criteria into a hierarchy of checklists.

    Args:
        criteria_objs: The criteria objects to hierarchize
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Criteria hierarchy
    """
    def _initialize_hierarchy_node(node: Dict[str, Any]) -> Dict[str, Any]:
        node["id"] = str(node["id"])
        node["aware"] = 0
        node["satisfied"] = 0
        node["reason"] = ""
        node["is_update"] = False
        node["update_thresholds"] = [random.random() for _ in range(2)]
        return node

    system_prompt = load_prompt("hierarchize_criteria.yaml")
    for c_obj in criteria_objs:
        criterion_yaml_obj = {
            "num_abstractions": len(c_obj["abstractions"]),
            "abstractions": [
                {
                    "level": j + 1,
                    "checklist": [item["item"] for item in a["checklist"]]
                } for j, a in enumerate(c_obj["abstractions"])
            ]
        }
        criterion_str = "```yaml\ncriterion:\n" + yaml.dump(criterion_yaml_obj, sort_keys=False, indent=2) + "```"
        formatted_prompt = (
            f"# Criterion\n\n"
            f"{criterion_str}\n\n"
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

        criterion_hierarchy = parsed_yaml['hierarchy']
        c_obj["hierarchy"] = update_hierarchy(
            criterion_hierarchy,
            _initialize_hierarchy_node
        )

    return criteria_objs

