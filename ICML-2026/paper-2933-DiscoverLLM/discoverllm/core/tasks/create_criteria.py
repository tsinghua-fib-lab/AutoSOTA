from discoverllm.core.generate import generate_and_process_yaml
from discoverllm.core.prompts import load_prompt


def create_criteria(
    artifact: str,
    artifact_type: str,
    max_num_criteria: int = 2,
    model_name: str = "gpt-5-2025-08-07",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    verbose: bool = False,
    max_retries: int = 3,
) -> tuple[list[str], str]:
    """
    Create a list of criteria that must be satisfied in order to create the given artifact,
    along with a short phrase describing the main topic of the artifact.

    Args:
        artifact: The original artifact
        artifact_type: The type of artifact
        max_num_criteria: Maximum number of criteria to generate
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        A tuple `(criteria, topic_phrase)` where:
            - criteria: List of criteria (List[str])
            - topic_phrase: Short phrase describing the main topic of the artifact
    """
    system_prompt = load_prompt("create_criteria.yaml")
    formatted_prompt = (
        f"# Maximum Number of Criteria\n\n"
        f"{max_num_criteria}\n\n"
        f"# Artifact\n\n"
        f"{artifact}\n\n"
        f"# Artifact Type\n\n"
        f"{artifact_type}\n\n"
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

    if parsed_yaml is None:
        raise ValueError("Failed to parse YAML output from criteria generation")

    criteria = [
        {
            "description": parsed_yaml.get("description", []),
            "checklist": parsed_yaml.get("checklist", [])
        }
    ]
    artifact_topic = parsed_yaml.get("artifact_topic", "")
    return criteria, artifact_topic

