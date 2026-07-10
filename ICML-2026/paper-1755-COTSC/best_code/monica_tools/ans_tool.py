"""Answer extraction utilities for MONICA."""
import re


def extract_answer(text: str) -> str:
    """Extract the answer from model output text.
    
    Looks for \\boxed{...} patterns and fallback patterns like
    "the answer is X", "I choose X", etc.
    
    Args:
        text: model output text
    
    Returns:
        extracted answer string (may be empty)
    """
    if not text:
        return ""
    
    # Primary pattern: \boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try with nested braces
    boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fallback patterns (in order of reliability)
    fallback_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|:)\s+([A-E])",
        r"(?:I\s+)?(?:choose|select)\s+(?:option\s+)?([A-E])",
        r"(?:therefore|hence|thus),?\s+(?:the\s+)?(?:answer\s+)?(?:is\s+)?([A-E])",
        r"(?:I\s+)?conclude\s+(?:that\s+)?(?:the\s+)?(?:answer\s+)?(?:is\s+)?([A-E])",
        r"(?:I\s+)?believe\s+(?:the\s+)?(?:answer\s+)?(?:is\s+)?([A-E])",
        r"(?:I\s+)?think\s+(?:the\s+)?(?:answer\s+)?(?:is\s+)?([A-E])",
        r"correct\s+(?:answer\s+)?(?:is\s+)?([A-E])",
    ]
    
    # Search in the last portion of text (response, not thinking)
    sentences = re.split(r"[.!?\n]+", text)
    search_text = " ".join(sentences[-10:])  # Last ~10 sentences
    
    for pattern in fallback_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            if answer in "ABCDE":
                return answer
    
    # Try on the full text if nothing found in the last sentences
    for pattern in fallback_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            if answer in "ABCDE":
                return answer
    
    return ""
