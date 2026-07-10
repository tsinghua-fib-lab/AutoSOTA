import json
import os
import random
import re
import string
import time
import warnings
from typing import Any, Callable, Dict, List

import yaml

# --------------------------------------------------------------------------- #
# Token counting (Llama-3.1-8B-Instruct tokenizer, lazy-loaded)               #
# --------------------------------------------------------------------------- #
_tokenizer = None


def get_tokenizer():
    """Get or lazily initialise the Llama-3.1-8B-Instruct tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception as e:
            warnings.warn(f"Failed to load Llama tokenizer: {e}. Token counts will be unavailable.")
            _tokenizer = False  # Mark as failed to avoid retrying
    return _tokenizer if _tokenizer is not False else None


def count_tokens_in_conversation(conversation_history: List[Dict[str, Any]]) -> int:
    """
    Count total tokens in a conversation, using Llama-3.1-8B-Instruct's
    chat template. Returns 0 if the tokenizer can't be loaded.
    """
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return 0
    try:
        messages = [
            msg for msg in conversation_history
            if isinstance(msg, dict) and "role" in msg and "content" in msg
        ]
        if not messages:
            return 0
        # On transformers>=5, apply_chat_template(tokenize=True) returns a
        # BatchEncoding rather than a flat list of ids; take input_ids when
        # present so len() actually counts tokens.
        encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        if hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        return len(encoded)
    except Exception as e:
        warnings.warn(f"Failed to count tokens: {e}. Returning 0.")
        return 0


def safe_load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def safe_write_json(path: str, data: Any) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def process_yaml(text: str) -> str:
    """
    Extracts YAML content from a markdown code block (```yaml ... ```), if it exists.
    Then, processes the YAML content by removing any comments and ensuring proper YAML formatting.
    Raises ValueError if the pattern is not found.
    """
    try:
        if "```yaml" in text:
            yaml_str = text.split("```yaml")[1].replace("```", "").strip()
        elif "```" in text:
            yaml_str = text.replace("```", "").strip()
        else:
            yaml_str = text

        if not yaml_str or not yaml_str.strip():
            raise ValueError(f"YAML string is empty or whitespace-only after extraction.\n\nOriginal text:\n{text}\n================================================")

        result = yaml.safe_load(yaml_str)
        if result is None:
            raise ValueError(f"yaml.safe_load() returned None (likely empty/whitespace YAML).\n\nExtracted YAML string:\n{repr(yaml_str)}\n\nOriginal text:\n{text}\n================================================")
        return result
    except ValueError:
        # Re-raise ValueError as-is (it already has the message)
        raise
    except Exception as e:
        raise ValueError(f"Could not extract and process YAML: {str(e)}\n\nOriginal text:\n{text}\n================================================")


def process_json(text: str) -> Any:
    """
    Extracts JSON content from a markdown code block (```json ... ```), if it exists,
    or from a generic code block (``` ... ```). Falls back to parsing the raw text.

    Raises ValueError with helpful context if parsing fails.
    """
    try:
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].replace("```", "").strip()
        elif "```" in json_str:
            json_str = json_str.replace("```", "").strip()

        if not json_str or not json_str.strip():
            raise ValueError(
                f"JSON string is empty or whitespace-only after extraction.\n\nOriginal text:\n{text}\n================================================"
            )

        return json.loads(json_str)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Could not extract and process JSON: {str(e)}\n\nOriginal text:\n{text}\n================================================"
        )


def retry_with_yaml_processing(
    func: Callable,
    max_retries: int = 3,
    verbose: bool = False
) -> Any:
    """
    Retry a function that may fail during YAML processing.

    Args:
        func: A callable that returns a value. If it raises ValueError (YAML parsing error),
              it will be retried.
        max_retries: Maximum number of retry attempts (default: 3)
        verbose: Whether to print retry messages

    Returns:
        The result of calling func

    Raises:
        ValueError: If all retries are exhausted
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except ValueError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                if verbose:
                    print(f"YAML processing failed, retrying in {wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                if verbose:
                    print(f"YAML processing failed after {max_retries} attempts")
                raise last_error
    raise last_error


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    chat_history_str = ""
    for message in chat_history:
        role = message['role']
        chat_history_str += f"<{role}_message>\n{message['content']}\n</{role}_message>\n\n"
    return chat_history_str.strip()

def is_distinct_response(response: str, all_responses: List[str]) -> bool:
    """
    Determine whether a response is meaningfully distinct from prior responses.

    Responses that only differ by spacing, casing, or light punctuation tweaks
    are considered the same.
    """

    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    normalized_response = normalize(response)
    for prev in all_responses:
        if normalize(prev) == normalized_response:
            return False
    return True


def is_skip_node(node: Dict[str, Any]) -> bool:
    """Check if a node should be skipped."""
    # Skip if satisfied
    if node.get("satisfied", 0) == 1:
        return True

    # Skip if aware and all children are aware
    if node.get("aware", 0) == 1:
        children = node.get("children", [])
        if len(children) > 0 and all(child.get("aware", 0) == 1 for child in children):
            return True

    return False

def format_criterion_hierarchy(criterion_obj: Dict[str, Any]) -> str:
    """
    Format criterion hierarchy as YAML, filtering out nodes that are satisfied
    or fully aware (where the node and all its children are aware).

    Filters out nodes where:
    - node["satisfied"] == 1 OR
    - (node["aware"] == 1 AND all(child_node["aware"] == 1 for child_node in node["children"]))
    """
    def _filter_hierarchy_node(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively filter a hierarchy node and its children.

        Returns a list of nodes. If the current node is filtered, returns its
        filtered children (promoting them). If not filtered, returns a list
        containing just this node with its filtered children.
        """
        # Recursively filter children first
        filtered_children_list = []
        for child in node.get("children", []):
            filtered_children_list.extend(_filter_hierarchy_node(child))

        # Check if this node should be filtered (using original children)
        if is_skip_node(node):
            # If node should be filtered, return its filtered children (promote them)
            return filtered_children_list

        # Node is not filtered, so include it with its filtered children
        filtered_node = {
            "node_id": node.get("id", ""),
            "text": node.get("text", "")
        }
        filtered_node["children"] = filtered_children_list

        return [filtered_node]

    # Filter each root node in the hierarchy
    filtered_hierarchy = []
    for root_node in criterion_obj["hierarchy"]:
        filtered_hierarchy.extend(_filter_hierarchy_node(root_node))

    criterion_str = yaml.dump(filtered_hierarchy, sort_keys=False, indent=2)
    return criterion_str

def update_hierarchy(hierarchy: Dict[str, Any], update_func: Callable) -> Dict[str, Any]:
    """
    Recursively apply a function to a hierarchy.
    """
    for item in hierarchy:
        item = update_func(item)
        if "children" in item:
            item["children"] = update_hierarchy(item["children"], update_func)
    return hierarchy

def filter_hierarchy(hierarchy: Dict[str, Any], filter_func: Callable, parent_node: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Recursively get items from a hierarchy that satisfy a filter function.
    """
    items = []
    for item in hierarchy:
        try:
            result = filter_func(item, parent_node)
        except TypeError:
            result = filter_func(item)
        if result:
            items.append(item)
        if "children" in item:
            items.extend(filter_hierarchy(item["children"], filter_func, item))
    return items
