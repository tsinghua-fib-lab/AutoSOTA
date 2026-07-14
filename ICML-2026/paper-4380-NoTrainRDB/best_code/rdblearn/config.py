from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field
from fastdfs import DFSConfig


class TemporalDiffConfig(BaseModel):
    """Configuration for temporal difference feature generation."""
    enabled: bool = True
    # Columns to explicitly exclude from transformation
    exclude_columns: List[str] = Field(default_factory=list)


class RDBLearnConfig(BaseModel):
    """
    Configuration for RDBLearnEstimator.
    """
    # DFS Configuration (passed to fastdfs)
    # If None, defaults to: {"max_depth": 2, "agg_primitives": ["max", "min", "mean", "count", "mode", "std"], "engine": "dfs2sql"}
    dfs: Optional[DFSConfig] = None
    
    # Preprocessing Configuration (passed to AutoGluon feature generator)
    # If None, defaults to: {"enable_datetime_features": True, "enable_raw_text_features": False, ...}
    ag_config: Optional[Dict[str, Any]] = None
    
    # Sampling Configuration
    max_train_samples: int = 10000
    stratified_sampling: bool = False  # Ignored for RDBLearnRegressor
    balanced_sampling: bool = False  # Keep all minority examples, fill rest with majority
    
    # Target History Augmentation
    enable_target_augmentation: bool = False
    
    # Prediction Configuration
    predict_batch_size: int = 5000

    # Temporal Difference Configuration (post-DFS transformation)
    temporal_diff: Optional[TemporalDiffConfig] = TemporalDiffConfig()

    # Label-encode schema category columns (and object/category task columns) to float before DFS.
    encode_categorical_as_float: bool = False

    class Config:
        arbitrary_types_allowed = True
