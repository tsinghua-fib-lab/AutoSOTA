import logging
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set the seed for the random number generator.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def numpy_serializer(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, float):
        return float(obj)
    # Handle torch tensors
    if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.cpu().item()
    # Handle torch scalar tensors
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass
    # Handle other numeric types
    if hasattr(obj, '__float__'):
        try:
            return float(obj)
        except (ValueError, TypeError):
            pass
    if hasattr(obj, '__int__'):
        try:
            return int(obj)
        except (ValueError, TypeError):
            pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def get_responses_above_threshold(baseline_data: dict, toxicity_threshold: float, mode: str = 'mean') -> dict:
    """
    Retrieve prompts whose average toxicity exceeds the given threshold.
    """
    key = 'avg_toxicity' if mode == 'mean' else 'max_toxicity'
    subset_data = [
        entry for entry in baseline_data 
        if entry[key] >= toxicity_threshold
    ]
    
    return subset_data


def print_toxicity_statistics(results: list,
                              dataset_name: str,
                              model_name: str,
                              bins: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    Print toxicity statistics in a visually appealing table format.
    Shows distributions for both mean (average) and max toxicity modes.
    
    Args:
        results: List of result dictionaries containing toxicity data
        dataset_name: Name of the dataset
        model_name: Name of the model
        bins: List of bins for the toxicity distribution
    """
    logger.info("="*100)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Total prompts: {len(results)}")
    logger.info("="*100)
    
    # Calculate max toxicity for each result
    results_with_max = []
    for entry in results:
        entry_copy = entry.copy()
        if isinstance(entry['toxicity_scores'], list):
            entry_copy['avg_toxicity'] = max(entry['toxicity_scores'])
        else:
            entry_copy['avg_toxicity'] = entry['toxicity_scores']
        results_with_max.append(entry_copy)
    
    # Print side-by-side comparison
    logger.info("\n  Toxicity Distribution:")
    logger.info("  " + "-"*100)
    logger.info(f"  {'Threshold':<12} {'Mean Mode':<40} {'Max Mode':<40}")
    logger.info(f"  {'':<12} {'Count':<10} {'Percentage':<28} {'Count':<10} {'Percentage':<28}")
    logger.info("  " + "-"*100)
    
    for th in bins:
        # Mean mode
        count_mean = len(get_responses_above_threshold(results, th))
        percentage_mean = count_mean / len(results)
        bar_mean = '█' * int(percentage_mean * 15)
        
        # Max mode
        count_max = len(get_responses_above_threshold(results_with_max, th))
        percentage_max = count_max / len(results)
        bar_max = '█' * int(percentage_max * 15)
        
        # Format with proper alignment - mean mode section (40 chars total)
        mean_section = f"{count_mean:<10} {percentage_mean:>6.2%}  {bar_mean:<15}"
        # Format with proper alignment - max mode section (40 chars total)  
        max_section = f"{count_max:<10} {percentage_max:>6.2%}  {bar_max:<15}"
        
        logger.info(f"  {f'>= {th:.2f}':<12} {mean_section:<40} {max_section:<40}")
    
    logger.info("  " + "-"*100)
    
    # Print summary statistics
    mean_toxicities = [entry['avg_toxicity'] for entry in results]
    max_toxicities = [entry['avg_toxicity'] for entry in results_with_max]
    
    logger.info("\n  Summary Statistics:")
    logger.info("  " + "-"*100)
    logger.info(f"  {'Metric':<20} {'Mean Mode':<40} {'Max Mode':<40}")
    logger.info("  " + "-"*100)
    
    # Calculate statistics
    avg_mean = sum(mean_toxicities) / len(mean_toxicities)
    avg_max = sum(max_toxicities) / len(max_toxicities)
    median_mean = sorted(mean_toxicities)[len(mean_toxicities) // 2]
    median_max = sorted(max_toxicities)[len(max_toxicities) // 2]
    min_mean = min(mean_toxicities)
    min_max = min(max_toxicities)
    max_mean = max(mean_toxicities)
    max_max = max(max_toxicities)
    
    # Format with proper alignment - each section gets exactly 40 characters
    logger.info(f"  {'Average:':<20} {avg_mean:<40.4f} {avg_max:<40.4f}")
    logger.info(f"  {'Median:':<20} {median_mean:<40.4f} {median_max:<40.4f}")
    logger.info(f"  {'Min:':<20} {min_mean:<40.4f} {min_max:<40.4f}")
    logger.info(f"  {'Max:':<20} {max_mean:<40.4f} {max_max:<40.4f}")
    logger.info("  " + "-"*100 + "\n")


def find_or_create_config_dir(base_dir: str, config_params: dict):
    """
    Find an existing config directory that matches the given parameters,
    or create a new one if no match is found.
    
    Args:
        base_dir: Base directory for all configs
        config_params: Dictionary of configuration parameters
    
    Returns:
        Tuple containing a boolean indicating if a matching config was found
        and the path to the config directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Find all existing config directories
    existing_configs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("config_"):
                try:
                    config_num = int(item.split("_")[1])
                    existing_configs.append((config_num, item_path))
                except (IndexError, ValueError):
                    continue
    
    # Check if any existing config matches
    for config_num, config_path in existing_configs:
        config_file = os.path.join(config_path, "config.yaml")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                existing_params = yaml.safe_load(f)
            
            # Compare parameters
            if existing_params == config_params:
                return True, config_path
    
    # No match found, create new config directory
    if existing_configs:
        max_config_num = max(num for num, _ in existing_configs)
        new_config_num = max_config_num + 1
    else:
        new_config_num = 0
    
    new_config_dir = os.path.join(base_dir, f"config_{new_config_num}")
    os.makedirs(new_config_dir, exist_ok=True)
    
    # Save config.yaml
    config_file = os.path.join(new_config_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config_params, f, default_flow_style=False)
    
    return False, new_config_dir
