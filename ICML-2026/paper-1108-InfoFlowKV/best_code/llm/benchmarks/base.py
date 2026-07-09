"""Base class for datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDataset(ABC):
    """Base class for all datasets."""

    def __init__(self):
        self.data = None

    @abstractmethod
    def load(self, num_samples: Optional[int] = None, **kwargs):
        """Load the dataset."""
        pass

    @abstractmethod
    def build_prompt(self, sample: Dict) -> str:
        """Build prompt for a sample."""
        pass

    @abstractmethod
    def get_answer(self, sample: Dict) -> str:
        """Get the ground truth answer for a sample."""
        pass

    @abstractmethod
    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check if prediction is correct."""
        pass

    def __len__(self):
        return len(self.data) if self.data else 0

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(name: str, **kwargs) -> BaseDataset:
    """
    Get dataset by name.

    Args:
        name: Dataset name, e.g., 'needle', '2wikimqa', 'hotpotqa', 'musique'
        **kwargs: Additional dataset-specific arguments

    Returns:
        Loaded dataset instance
    """
    if name == "needle":
        from .needle import NeedleDataset
        return NeedleDataset(**kwargs)
    elif name in ["2wikimqa", "hotpotqa", "musique", "narrativeqa", "qasper", "multifieldqa_en"]:
        from .longbench import LongBenchDataset
        return LongBenchDataset(name=name, **kwargs)
    elif name == "longbenchv2":
        from .longbenchv2 import LongBenchV2Dataset
        return LongBenchV2Dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
