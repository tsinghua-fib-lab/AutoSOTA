"""Base class for datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDataset(ABC):
    """Base class for all datasets."""

    @abstractmethod
    def load(self, split: str = "val"):
        """Load the dataset."""
        pass

    @abstractmethod
    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages in VLM format for a sample."""
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
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def select(self, indices):
        """Select a subset of the dataset."""
        self.data = self.data.select(indices)
        return self


def get_dataset(name: str, dataset_dir: str = None, output_dir: str = None) -> BaseDataset:
    """
    Get dataset by name.

    Args:
        name: Dataset name, e.g., 'blink_counting', 'realworldqa', 'mmbench'
        dataset_dir: Directory to cache/load datasets from
        output_dir: Directory to save results to

    Returns:
        Loaded dataset instance
    """
    from .blink import BlinkDataset
    from .realworldqa import RealWorldQADataset
    from .mmbench import MMBenchDataset
    from .chartqa import ChartQADataset
    from .docvqa import DocVQADataset
    from .ocrbench import OCRBenchDataset
    from .mathvista import MathVistaDataset

    # BLINK benchmark (multiple subsets)
    if name.startswith("blink_"):
        raw = name.split("_", 1)[1]
        # Normalize to known HF config names
        canonical = "".join(ch.lower() for ch in raw if ch.isalnum())
        mapping = {
            "artstyle": "Art_Style",
            "counting": "Counting",
            "forensicdetection": "Forensic_Detection",
            "functionalcorrespondence": "Functional_Correspondence",
            "iqtest": "IQ_Test",
            "jigsaw": "Jigsaw",
            "multiviewreasoning": "Multi-view_Reasoning",
            "objectlocalization": "Object_Localization",
            "relativedepth": "Relative_Depth",
            "relativereflectance": "Relative_Reflectance",
            "semanticcorrespondence": "Semantic_Correspondence",
            "spatialrelation": "Spatial_Relation",
            "visualcorrespondence": "Visual_Correspondence",
            "visualsimilarity": "Visual_Similarity",
        }
        if canonical not in mapping:
            raise ValueError(
                f"Unknown BLINK subset '{raw}'. Available: {list(mapping.values())}"
            )
        subset = mapping[canonical]
        return BlinkDataset(subset=subset, dataset_dir=dataset_dir, output_dir=output_dir).load()

    # RealWorldQA
    if name == "realworldqa":
        return RealWorldQADataset(dataset_dir=dataset_dir, output_dir=output_dir).load()

    # MMBench (with optional language subset)
    if name == "mmbench" or name.startswith("mmbench_"):
        subset = "en"
        if name.startswith("mmbench_"):
            subset = name.split("_", 1)[1]
        return MMBenchDataset(subset=subset, dataset_dir=dataset_dir, output_dir=output_dir).load()

    # ChartQA
    if name == "chartqa":
        return ChartQADataset(dataset_dir=dataset_dir, output_dir=output_dir).load()

    # DocVQA
    if name == "docvqa":
        return DocVQADataset(dataset_dir=dataset_dir, output_dir=output_dir).load()

    # OCRBench
    if name == "ocrbench":
        return OCRBenchDataset(dataset_dir=dataset_dir, output_dir=output_dir).load()

    # MathVista
    if name == "mathvista":
        return MathVistaDataset(dataset_dir=dataset_dir, output_dir=output_dir).load()

    raise ValueError(f"Unknown dataset: {name}. Available: blink_*, realworldqa, mmbench, chartqa, docvqa, ocrbench, mathvista")
