from transformers import PreTrainedTokenizerBase


def count_text_tokens(tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: bool = True) -> int:
    """Count tokens in text using the provided tokenizer."""
    encoded = tokenizer(
        text,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    return len(encoded["input_ids"])


def count_text_characters(text: str) -> int:
    """Count characters in text."""
    return len(text)
