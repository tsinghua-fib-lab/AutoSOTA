"""
Model-related utility functions
For handling model paths, names, etc.
"""

from pathlib import Path
from typing import Optional


def get_model_name_from_path(model_path: str) -> str:
    """
    Extract the model name from a model path

    Args:
        model_path: Model path (can be an absolute or relative path)

    Returns:
        Model name (e.g., "Llama-2-13b-hf", "Qwen14B")

    Examples:
        >>> get_model_name_from_path("/path/to/models/Llama-2-13b-hf")
        'Llama-2-13b-hf'
        >>> get_model_name_from_path("models/Qwen14B")
        'Qwen14B'
    """
    path = Path(model_path)

    # If the path is a directory, use the directory name directly
    if path.is_dir():
        return path.name

    # If the path is a file, use the parent directory name
    if path.exists():
        return path.parent.name

    # Try to extract from the path string
    # Handle "models/Llama-2-13b-hf" or "/path/to/models/Llama-2-13b-hf"
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "models":
        return parts[-1]

    # If the path contains "models/", extract the part after it
    path_str = str(path)
    if "models/" in path_str:
        model_name = path_str.split("models/")[-1].split("/")[0]
        return model_name

    # By default, return the last component of the path
    return path.name


def get_output_dir_for_model(base_output_dir: str, model_path: str, subdir: Optional[str] = None) -> Path:
    """
    Generate an output directory based on the model path

    Args:
        base_output_dir: Base output directory (e.g., "outputs")
        model_path: Model path
        subdir: Subdirectory (e.g., "ppd_pipeline")

    Returns:
        Full output directory path

    Examples:
        >>> get_output_dir_for_model("outputs", "models/Llama-2-13b-hf", "ppd_pipeline")
        Path('outputs/Llama-2-13b-hf/ppd_pipeline')
        >>> get_output_dir_for_model("outputs", "models/Qwen14B", "pruning_strength")
        Path('outputs/Qwen14B/pruning_strength')
    """
    model_name = get_model_name_from_path(model_path)
    base_dir = Path(base_output_dir)
    
    if subdir:
        output_dir = base_dir / model_name / subdir
    else:
        output_dir = base_dir / model_name
    
    return output_dir

