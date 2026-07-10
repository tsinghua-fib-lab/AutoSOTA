from typing import Any, Dict, List

from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt
from discoverllm.utils import format_chat_history, format_criterion_hierarchy


def validate_yaml(yaml_obj: Dict[str, Any], criteria_objs: List[Dict[str, Any]]) -> None:
    """
    Validate the YAML object to ensure it is correct.
    """
    if "evaluations" not in yaml_obj and "evaluation_type" not in yaml_obj:
        raise ValueError("Invalid evaluation YAML object does not include 'evaluations' and 'evaluation_type'")

    if len(yaml_obj['evaluations']) != len(criteria_objs):
        raise ValueError("Number of evaluations does not match the number of criteria")

    for criterion_idx, evaluation in enumerate(yaml_obj['evaluations']):
        if "stepwise_evaluation" not in evaluation:
            raise ValueError(f"Evaluation for criterion {criterion_idx} does not include 'stepwise_evaluation'")

        if len(evaluation['stepwise_evaluation']) > len(criteria_objs[criterion_idx]['abstractions']):
            raise ValueError(f"Number of stepwise evaluations for criterion {criterion_idx} exceeds the number of abstraction versions")


def slice_chat_history(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Slice the chat history up to the given index.
    """
    return format_chat_history(chat_history[:1] + "\n\n[... ABRIDGED ...]\n\n" + format_chat_history(chat_history[-3:]))


def evaluate_message(
    chat_history: List[Dict[str, str]],
    criteria_objs: List[Dict[str, Any]],
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3
) -> str:
    """
    Evaluate the satisfaction/relevance of the AI assistant's last message based on the type of message.

    Args:
        chat_history: The chat history between the user and the AI assistant
        criteria_objs: The list of criteria objects that the user is interested in
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        verbose: Whether to print verbose output
        max_retries: Maximum number of retries if the evaluation fails
    Returns:
        List of evaluations, evaluation type
    """
    system_prompt = load_prompt("evaluate_message.yaml")

    assistant_message = chat_history[-1]['content']
    assistant_message = assistant_message.strip()

    chat_history_str = format_chat_history(chat_history)
    if len(chat_history) > 8:
        chat_history_str = format_chat_history(chat_history[:1])
        chat_history_str += "\n\n[... ABRIDGED ...]\n\n"
        chat_history_str += format_chat_history(chat_history[-7:])

    all_evaluations = []
    for i, criterion_obj in enumerate(criteria_objs):
        formatted_prompt = (
            f"<chat_history>\n"
            f"{chat_history_str}\n"
            f"</chat_history>\n\n"
            f"<criterion>\n\n"
            f"{format_criterion_hierarchy(criterion_obj)}\n"
            f"</criterion>\n"
        )

        parsed_yaml, raw_output = generate_and_process_yaml(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=formatted_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            verbose=verbose
            # validation_function=lambda yaml_obj: validate_yaml(yaml_obj, criteria_objs)
        )

        evaluations = parsed_yaml['evaluations']
        for ev in evaluations:
            ev['node_id'] = str(ev['node_id'])

        all_evaluations.append({
            "criterion_id": i,
            "evaluations": evaluations
        })

    try:
        return all_evaluations, parsed_yaml['evaluation_type']
    except Exception as e:
        raise Exception(f"Error evaluating message: {e}\n\nRAW OUTPUT:\n{raw_output}")
