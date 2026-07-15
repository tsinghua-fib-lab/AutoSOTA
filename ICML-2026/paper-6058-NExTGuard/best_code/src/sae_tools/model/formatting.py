from typing import List, Tuple, Optional

def format_with_tokenizer(
    tokenizer, 
    prompt: str,
    response: Optional[str] = None
) -> Tuple[str, List[int], Tuple[int, int]]:
    """
    Format the input with the prompt template, and calculate the token indices of the prompt start position
    """

    if response is None:
        prefix = "User: "
        text = prefix + prompt
    else:
        prefix = f"User: {prompt}\nAssistant: "
        text = prefix + response
    
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    valid_start = len(prefix_tokens)
    
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    valid_end = len(text_tokens)
    
    return text, text_tokens, (valid_start, valid_end)