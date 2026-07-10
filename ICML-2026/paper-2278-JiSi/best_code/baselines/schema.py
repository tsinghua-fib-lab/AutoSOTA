"""
Benchmark-result schema definitions for JiSi.

This module defines the unified schema for collected benchmark records,
ensuring consistent data structure across datasets and models.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from datetime import datetime


@dataclass
class BaselineRecord:
    """
    Unified baseline record format for benchmark results.

    This schema represents a single evaluation record from any benchmark,
    capturing the essential information needed for baseline comparisons.

    Attributes:
        dataset_id: Dataset identifier (e.g., 'aime', 'gpqa')
        split: Dataset split (e.g., 'test', 'train', 'valid')
        model_name: Model identifier (e.g., 'gpt-4', 'claude-3')
        record_index: Zero-based index of this record within the dataset

        origin_query: Original question/problem from the dataset
        prompt: Formatted prompt sent to the model (may include instructions)

        prediction: Extracted answer from model output
        raw_output: Complete model response before extraction

        ground_truth: Correct answer from the dataset
        score: Evaluation score (typically 0.0 or 1.0 for correctness)

        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        cost: API cost for this single record (in USD)
    """

    # Identification
    dataset_id: str
    split: str
    model_name: str
    record_index: int

    # Input
    origin_query: str
    prompt: str

    # Output
    prediction: str
    raw_output: Any

    # Evaluation
    ground_truth: str
    score: float

    # Resources
    prompt_tokens: int
    completion_tokens: int
    cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    def to_dict_compact(self, include_raw_output: bool = False,
                        include_prompt: bool = True,
                        included_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert record to dictionary with optional field selection.

        Args:
            include_raw_output: Whether to include the raw_output field (can be large)
                               Ignored if included_columns is specified
            include_prompt: Whether to include the full prompt field
                           Ignored if included_columns is specified
            included_columns: Explicit list of column names to include
                             If specified, only these columns will be returned

        Returns:
            Dictionary with selected fields
        """
        data = self.to_dict()

        # New behavior: explicit column selection (white-list approach)
        if included_columns is not None:
            return {key: data[key] for key in included_columns if key in data}

        # Legacy behavior: exclusion-based approach (for backward compatibility)
        if not include_raw_output:
            data.pop('raw_output', None)

        if not include_prompt:
            data.pop('prompt', None)

        return data


@dataclass
class AggregatedStats:
    """
    Aggregated statistics for a dataset/model combination.

    This represents summary statistics computed from multiple BaselineRecords.
    """

    dataset_id: str
    split: str
    model_name: str

    # Performance metrics
    avg_score: float
    total_records: int
    correct_records: int

    # Resource metrics
    total_cost: float
    avg_cost_per_record: float
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_prompt_tokens: float
    avg_completion_tokens: float

    # Optional metadata
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def accuracy(self) -> float:
        """Alias for avg_score (commonly used term)."""
        return self.avg_score

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.total_prompt_tokens + self.total_completion_tokens


@dataclass
class BaselineMetadata:
    """
    Metadata for a baseline dataset cache.

    Tracks information about cached baseline data for validation and versioning.
    """

    version: str
    created_at: str
    total_records: int
    total_datasets: int
    total_models: int
    dataset_splits: Dict[str, list]  # dataset_id -> list of splits
    models: list  # list of model names

    # File information
    cache_format: str  # 'parquet', 'jsonl', etc.
    cache_size_bytes: Optional[int] = None

    # Configuration hash for invalidation
    config_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineMetadata':
        """Create from dictionary."""
        return cls(**data)
