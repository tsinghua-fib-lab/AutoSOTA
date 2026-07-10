"""
Common utilities for data adaptors.

Provides shared functionality for dataset splitting and transformation.
"""

from typing import List, Tuple, Dict
from collections import defaultdict
import random
from loguru import logger

from baselines.schema import BaselineRecord


def split_by_dataset(
    records: List[BaselineRecord],
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
    """
    Split records into train and test sets by dataset.

    Each dataset is split independently, ensuring both train and test contain
    samples from each dataset. Records are sorted by index before splitting.

    Args:
        records: List of baseline records to split
        train_ratio: Proportion of data to use for training (0.0-1.0)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_records, test_records)

    Example:
        >>> train, test = split_by_dataset(records, train_ratio=0.8, random_seed=42)
        >>> # Each dataset will have ~80% in train, ~20% in test
    """
    validate_train_ratio(train_ratio)

    # Group records by dataset
    dataset_groups = defaultdict(list)
    for record in records:
        dataset_groups[record.dataset_id].append(record)

    logger.info(f"Splitting {len(records)} records across {len(dataset_groups)} datasets")
    logger.info(f"Train ratio: {train_ratio}, Random seed: {random_seed}")

    train_records = []
    test_records = []

    # Set random seed
    random.seed(random_seed)

    # Split each dataset independently
    for dataset_id, dataset_records in dataset_groups.items():
        # Sort by record_index to ensure consistent ordering
        dataset_records.sort(key=lambda r: r.record_index)

        # Calculate split point
        n_train = int(len(dataset_records) * train_ratio)

        # Shuffle indices for random split
        indices = list(range(len(dataset_records)))
        random.shuffle(indices)

        # Split based on shuffled indices
        train_indices = set(indices[:n_train])

        dataset_train = [dataset_records[i] for i in range(len(dataset_records)) if i in train_indices]
        dataset_test = [dataset_records[i] for i in range(len(dataset_records)) if i not in train_indices]

        logger.debug(f"Dataset {dataset_id}: {len(dataset_train)} train, {len(dataset_test)} test")

        train_records.extend(dataset_train)
        test_records.extend(dataset_test)

    logger.info(f"Split complete: {len(train_records)} train, {len(test_records)} test")

    return train_records, test_records


def group_by_prompt(records: List[BaselineRecord]) -> Dict[str, List[BaselineRecord]]:
    """
    Group records by prompt (question).

    Args:
        records: List of baseline records

    Returns:
        Dictionary mapping prompt to list of records with that prompt
    """
    prompt_groups = defaultdict(list)
    for record in records:
        prompt_groups[record.prompt].append(record)
    return dict(prompt_groups)


def get_unique_models(records: List[BaselineRecord]) -> List[str]:
    """
    Get sorted list of unique model names from records.

    Args:
        records: List of baseline records

    Returns:
        Sorted list of unique model names
    """
    models = sorted(set(record.model_name for record in records))
    return models


def get_unique_prompts(records: List[BaselineRecord]) -> List[str]:
    """
    Get sorted list of unique prompts from records.

    Args:
        records: List of baseline records

    Returns:
        Sorted list of unique prompts
    """
    prompts = sorted(set(record.prompt for record in records))
    return prompts


def validate_train_ratio(train_ratio: float):
    """
    Validate train_ratio parameter.

    Args:
        train_ratio: Proportion of data to use for training

    Raises:
        ValueError: If train_ratio is not between 0 and 1
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")


def fill_missing_models_scores(
    scores_dict: Dict[str, float],
    all_models: List[str],
    fill_value: float = 0.0
) -> Dict[str, int]:
    """
    Fill missing model scores in a scores dictionary.

    This function modifies scores_dict in-place and tracks which models
    were filled.

    Args:
        scores_dict: Dictionary mapping model_name to score (modified in-place)
        all_models: List of all expected model names
        fill_value: Value to use for missing models (default: 0.0)

    Returns:
        Dictionary mapping model_name to count of filled occurrences
    """
    filled_count = {}
    for model in all_models:
        if model not in scores_dict:
            scores_dict[model] = fill_value
            filled_count[model] = 1
    return filled_count


def log_filled_statistics(
    filled_counter: Dict[str, int],
    prefix: str = "",
    top_n: int = 5
):
    """
    Log statistics about filled missing model scores.

    Args:
        filled_counter: Dictionary mapping model_name to fill count
        prefix: Optional prefix for log messages
        top_n: Number of top models to show (default: 5)
    """
    if not filled_counter:
        return

    total_filled = sum(filled_counter.values())
    logger.info(f"{prefix}Filled {total_filled} missing model entries")

    # Sort by count descending and show top N
    sorted_items = sorted(filled_counter.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_items[:top_n]:
        logger.info(f"  - {model}: {count} occurrences")
