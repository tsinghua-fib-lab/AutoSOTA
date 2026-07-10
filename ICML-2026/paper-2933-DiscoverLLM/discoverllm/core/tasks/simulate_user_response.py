from typing import Any, Dict, List

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import filter_hierarchy, format_chat_history, is_skip_node


def items_to_str(items: List[Dict[str, Any]], with_reason: bool = False, with_update: bool = False) -> str:
    output_str = ""
    for item in items:
        output_str += f"- item: {item['text']}"
        if with_update and item.get('update') is not None:
            output_str += "\n  update: "
            if item['update']['type'] == "awareness":
                output_str += "unaware -> aware"
            elif item['update']['type'] == "satisfaction":
                output_str += "satisfied -> dissatisfied" if item['update']['status'] == 0 else "dissatisfied -> satisfied"
        if with_reason and len(item['reason'].strip()) > 0:
            output_str += f"\n  reason: {item['reason'].strip()}"
        output_str += "\n"
    return output_str

def get_goal_status_as_str(criteria_objs: List[Dict[str, Any]]) -> str:
    achieved = []
    pursuing_clear = []
    pursuing_fuzzy = []
    for c in criteria_objs:
        hierarchy = c["hierarchy"]
        # Prettified filtering for clarity and expressive logic
        curr_achieved = filter_hierarchy(
            hierarchy,
            lambda node: (
                node["satisfied"] == 1 and (
                    len(node["children"]) == 0
                    or not all(child["satisfied"] == 1 for child in node["children"])
                )
            )
        )
        curr_pursuing_clear = filter_hierarchy(
            hierarchy,
            lambda node: node["aware"] == 1 and node["satisfied"] != 1 and not is_skip_node(node)
        )
        curr_pursuing_fuzzy = filter_hierarchy(
            hierarchy,
            lambda node: node["aware"] == 0.5 and node["satisfied"] != 1 and not is_skip_node(node)
        )
        achieved.extend(curr_achieved)
        pursuing_clear.extend(curr_pursuing_clear)
        pursuing_fuzzy.extend(curr_pursuing_fuzzy)

    if len(achieved) > 0:
        achieved_str = items_to_str(filter(lambda node: node.get("update") is not None, achieved), with_reason=True, with_update=True)
        achieved_str += items_to_str(filter(lambda node: node.get("update") is None, achieved), with_reason=True)
        achieved_str = achieved_str.strip()
    else:
        achieved_str = "NO ITEMS"

    if len(pursuing_clear) > 0 or len(pursuing_fuzzy) > 0:
        pursuing_list = pursuing_clear if len(pursuing_clear) > 0 else pursuing_fuzzy
        pursuing_str = items_to_str(filter(lambda node: node.get("update") is not None, pursuing_list), with_reason=True, with_update=True)
        pursuing_str += items_to_str(filter(lambda node: node.get("update") is None, pursuing_list), with_reason=True)
        pursuing_str = pursuing_str.strip()

        return (
            f"<achieved>\n"
            f"{achieved_str}\n"
            f"</achieved>\n"
            f"<pursuing_{'clear' if len(pursuing_clear) > 0 else 'fuzzy'}>\n"
            f"{pursuing_str}\n"
            f"</pursuing_{'clear' if len(pursuing_clear) > 0 else 'fuzzy'}>\n"
        )
    else:
        latent_goal_str = ""
        for criteria_obj in criteria_objs:
            latent_goal_str += "\n".join([f"- {node['text']}" for node in get_latent_goal_nodes(criteria_obj["hierarchy"])])
            latent_goal_str += "\n"
        latent_goal_str = latent_goal_str.strip()

        return (
            f"<achieved>\n"
            f"{achieved_str}\n"
            f"</achieved>\n"
            f"<latent_goal>\n"
            f"{latent_goal_str}\n"
            f"</latent_goal>\n"
        )

def get_latent_goal_nodes(node_list: List[Dict[str, Any]], is_highest_found: bool = False) -> str:
    # Find all latent nodes in the criteria hierarchy
    # These includes the "highest level nodes" that have aware=0 and the "lowest level nodes" that also have aware=0
    latent_nodes = []
    for node in node_list:
        curr_found = False
        if not is_highest_found and node["aware"] == 0:
            latent_nodes.append(node)
            curr_found = True
        elif len(node["children"]) == 0 and node["aware"] == 0:
            latent_nodes.append(node)

        if len(node["children"]) > 0:
            latent_nodes.extend(get_latent_goal_nodes(node["children"], curr_found))
    return latent_nodes

def simulate_user_response(
    chat_history: List[Dict[str, str]],
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Simulate a user's response to the AI assistant based on the chat history and criteria.

    Args:
        chat_history: The chat history between the user and the AI assistant
        criteria_objs: The list of criteria objects that the user is interested in
        MODEL_NAME: Name of the model to use
        TEMPERATURE: Sampling temperature
        MAX_TOKENS: Maximum tokens to generate

    Returns:
        User's response
    """
    system_prompt = load_prompt("simulate_response.yaml")

    goal_status_str = get_goal_status_as_str(criteria_objs)

    formatted_prompt = (
        f"<chat_history>\n"
        f"{format_chat_history(chat_history)}\n"
        f"</chat_history>\n"
        f"<goal_status>\n"
        f"{goal_status_str}\n"
        f"</goal_status>\n"
        f"**CRITICAL: Remember that you are role-playing as the USER!**"
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

    try:
        return parsed_yaml['user_message']
    except Exception as e:
        raise Exception(f"Error simulating user response: {e}\n\nRAW OUTPUT:\n{raw_output}")

