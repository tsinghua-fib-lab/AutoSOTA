"""
Response Parser Utility

Handles different LLM response formats, especially Gemini's list-based content.
"""

from typing import Any, Union


def extract_text_content(content: Any) -> str:
    """
    Extract text from various LLM response content formats.
    
    Handles:
    - str: Return as-is
    - None: Return empty string
    - list: Extract text from content blocks (Gemini format)
    - dict: Extract 'text' field
    - objects with .text attribute
    
    Args:
        content: The response.content from LLM
        
    Returns:
        Extracted text as string
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Gemini returns: [{'type': 'text', 'text': '...', 'extras': {...}}]
        # Empty list (tool call only) should return empty string
        if not content:
            return ""
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(str(item["content"]))
            elif isinstance(item, str):
                text_parts.append(item)
            elif hasattr(item, 'text'):
                text_parts.append(item.text)
        return " ".join(text_parts) if text_parts else ""
    
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        return str(content)
    
    if hasattr(content, 'text'):
        return content.text
    
    # Fallback: convert to string
    return str(content)


