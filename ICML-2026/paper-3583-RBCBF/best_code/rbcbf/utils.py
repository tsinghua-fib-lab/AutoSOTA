"""Utility for serializing prompt + response prefix for safety scorers."""


def serialize_prompt_and_prefix(
    prompt: str,
    prefix: str,
    include_prompt_as_context: bool = False,
) -> str:
    """Return a single string suitable for tokenization by a safety scorer.

    Args:
        prompt: The user's input prompt.
        prefix: The model's partial response prefix.
        include_prompt_as_context: If True, prepend the prompt so the scorer
            can condition on user intent (e.g. for a multi-rule scorer that
            jointly conditions on user request and response).
            If False, return only the prefix (used when scoring output-only).

    Returns:
        Serialized string ready for the encoder's tokenizer.
    """
    if include_prompt_as_context and prompt:
        return f"Human: {prompt}\n\nAssistant: {prefix}"
    return prefix
