import json
import os
import yaml
import re
from datasets import load_dataset
from typing import List, Dict

NAME_REPO_MAPPING = {
    "GSM8K": "openai/gsm8k",
    "ASDiv-Aug": "xuyige/ASDiv-Aug",
    "MultiArith": "ChilleD/MultiArith",
    "SVAMP": "ChilleD/SVAMP",
}

# Local dataset paths for offline/cache use
LOCAL_DATASET_PATHS = {
    "GSM8K": "/autosota_cache/tmp/gsm8k_download",
    "ASDiv-Aug": None,
    "MultiArith": None,
    "SVAMP": None,
}


def extract_numeric_answer(answer_str: str) -> str:
    """Extract numeric answer from answer string.
    
    Handles formats like:
    - "#### 18" (GSM8K format)
    - "<<8+2=10>>\n####10" (ASDiv-Aug format)
    - "10" (plain number)
    """
    if "####" in answer_str:
        # Extract the number after ####
        answer = answer_str.split("####")[-1].strip()
    else:
        answer = answer_str.strip()
    
    # Remove commas from numbers like "1,000"
    answer = answer.replace(",", "")
    
    # Extract just the numeric value (handles negative numbers and decimals)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        answer = match.group()
    
    return answer


def extract_numeric_from_option(option_str: str) -> str:
    """Extract numeric value from AQUA option string.
    
    For AQUA dataset options like 'A)36', 'B)$65', 'C)0.37', etc.
    Extract only the numeric part.
    
    Args:
        option_str: Option string (e.g., 'A)36', 'B)$65')
    
    Returns:
        Numeric string if found, empty string otherwise
    """
    # Remove the option letter prefix (A), B), etc.)
    if ')' in option_str:
        option_str = option_str.split(')', 1)[-1].strip()
    
    # Remove common currency symbols and units
    option_str = option_str.replace('$', '').replace('Rs.', '').replace('Rs', '')
    option_str = option_str.replace(',', '')  # Remove thousands separators
    
    # Extract numeric value (handles negative numbers, decimals, fractions)
    match = re.search(r'-?\d+\.?\d*', option_str)
    if match:
        return match.group()
    
    return ""


def load_test_dataset(dataset_name: str) -> List[Dict[str, str]]:
    """Load test dataset and return uniform format.
    
    Args:
        dataset_name: Name of the dataset (GSM8K, ASDiv-Aug, MultiArith, SVAMP)
    
    Returns:
        List of dicts with 'question' and 'answer' fields, where answer is a single number.
    """
    if dataset_name not in NAME_REPO_MAPPING:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(NAME_REPO_MAPPING.keys())}")
    
    # Try local path first, fall back to HF repo
    local_path = LOCAL_DATASET_PATHS.get(dataset_name)
    if local_path and os.path.isdir(local_path):
        print(f"Loading {dataset_name} from local path: {local_path}")
        repo_name = local_path

    if dataset_name == "GSM8K":
        ds = load_dataset(repo_name, "main", split="test")
        data = [
            {
                "question": item["question"],
                "answer": extract_numeric_answer(item["answer"])
            }
            for item in ds
        ]

    elif dataset_name == "ASDiv-Aug":
        ds = load_dataset(repo_name, split="test")
        data = [
            {
                "question": item["question"],
                "answer": extract_numeric_answer(item["answer"])
            }
            for item in ds
        ]

    elif dataset_name == "MultiArith":
        ds = load_dataset(repo_name, split="test")
        data = [
            {
                "question": item["question"].strip(),
                "answer": item["final_ans"].strip()
            }
            for item in ds
        ]

    elif dataset_name == "SVAMP":
        ds = load_dataset(repo_name, split="test")
        data = [
            {
                "question": item["question_concat"],
                "answer": item["Answer"].strip()
            }
            for item in ds
        ]

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")

    return data
 