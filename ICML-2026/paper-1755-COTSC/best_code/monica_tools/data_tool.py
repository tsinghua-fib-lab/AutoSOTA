"""Dataset loading and prompt formatting for MONICA."""
import json
from pathlib import Path


_TEMPLATES = {
    "deepseek_llama8b": (
        "<｜begin▁of▁sentence｜>{system_prompt}",
        "<｜User｜>{question}<｜Assistant｜>",
    ),
    "deepseek_qwen8b": (
        "<｜begin▁of▁sentence｜>{system_prompt}",
        "<｜User｜>{question}<｜Assistant｜>",
    ),
    "qwen3_1b": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        None,
    ),
    "qwen3_4b": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
        None,
    ),
}


def format_lrm_family(model_tag: str, system_prompt: str, question: str, extra: str = "") -> str:
    """Format a prompt for a specific model family.
    
    Args:
        model_tag: model identifier (e.g., qwen3_1b, deepseek_llama8b)
        system_prompt: system instruction text
        question: the user question/prompt
        extra: additional text to append
    
    Returns:
        formatted prompt string
    """
    family = model_tag.lower()
    
    if family in _TEMPLATES:
        template_pair = _TEMPLATES[family]
        if template_pair[1] is not None:
            # Two-part template (e.g., DeepSeek)
            formatted = template_pair[0].format(system_prompt=system_prompt)
            formatted += template_pair[1].format(question=question)
        else:
            # Single template (e.g., Qwen)
            formatted = template_pair[0].format(system_prompt=system_prompt, question=question)
    elif "deepseek" in family:
        formatted = f"<｜begin▁of▁sentence｜>{system_prompt}"
        formatted += f"<｜User｜>{question}<｜Assistant｜>"
    elif "qwen" in family:
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{question}<|im_end|>\n"
        formatted += f"<|im_start|>assistant\n"
    else:
        # Default chat format
        formatted = f"System: {system_prompt}\n\nUser: {question}\n\nAssistant:"
    
    if extra:
        formatted += extra
    
    return formatted


def resolve_cued_file(dataset_dir: Path, dataset_name: str, cue_type: str) -> Path | None:
    """Find the dataset file for a given dataset and cue type.
    
    Args:
        dataset_dir: directory containing dataset files
        dataset_name: name of the dataset (e.g., aime_2024_multichoice)
        cue_type: type of cue (e.g., metadata_leakage)
    
    Returns:
        Path to the dataset file, or None if not found
    """
    dataset_dir = Path(dataset_dir)
    
    # Map cue types to file naming patterns
    cue_to_suffix = {
        "metadata": "_metadata.json",
        "metadata_leakage": "_metadata_leakage.json",
        "user_suggestion": "_user_suggestion.json",
        "validation_function": "_validation_function.json",
        "unauthorized_access": "_unauthorized_access.json",
        "wrong_few_shot": "_wrong_few_shot.json",
        "biasing_few_shot": "_biasing_few_shot.json",
        "prefilled_wrong_answer": "_prefilled_wrong_answer.json",
        "tick_mark": "_tick_mark.json",
        "all": ".json",
    }
    
    # Try JSON first, then JSONL
    extensions = [".json", ".jsonl"]
    
    for ext in extensions:
        # Try cue-type-specific file
        if cue_type in cue_to_suffix:
            pattern = f"{dataset_name}{cue_to_suffix[cue_type]}"
            candidate = dataset_dir / pattern
            if candidate.exists():
                return candidate
        
        # Try without extension variation
        pattern = f"{dataset_name}_{cue_type}{ext}"
        candidate = dataset_dir / pattern
        if candidate.exists():
            return candidate
    
    # Try with just dataset name (for no-cue data)
    for ext in extensions:
        candidate = dataset_dir / f"{dataset_name}{ext}"
        if candidate.exists():
            return candidate
    
    return None


def load_dataset_json(path: Path) -> list[dict]:
    """Load a dataset file in JSON or JSONL format.
    
    Args:
        path: path to the dataset file
    
    Returns:
        list of data items
    """
    path = Path(path)
    
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    
    if not raw:
        return []
    
    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    
    # Try JSONL
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return lines
