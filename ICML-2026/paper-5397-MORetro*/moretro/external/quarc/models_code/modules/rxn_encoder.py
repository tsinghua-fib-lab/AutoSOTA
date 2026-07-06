import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal


class ReactionClassEncoder:
    """Encodes reaction classes to indices and vice versa."""

    def __init__(
        self,
        class_path: str | Path,
        level: Literal["category", "subcategory", "full"] = "full",
    ):
        """Initialize the reaction class encoder.

        Args:
            class_path: Path to CSV file containing reaction class information.
            level: Hierarchical level to encode at. Defaults to "full".
        """
        self.level = level
        class_path = Path(class_path)
        if not class_path.exists():
            raise FileNotFoundError(f"Reaction class file not found at {class_path}")

        self._load_from_csv(class_path)

    def _load_from_csv(self, class_path: str):
        df = pd.read_csv(class_path).sort_values(by="Code")

        if self.level == "category":
            class_column = "Category"
        elif self.level == "subcategory":
            class_column = "Sub-category"
        else:
            class_column = "Code"

        self.classes_ = df[class_column]
        self.class_to_index = {cls_code: idx for idx, cls_code in enumerate(self.classes_)}

    def __len__(self) -> int:
        return len(self.classes_)

    def __call__(self, rxn_class: str) -> int:
        return self.encode(rxn_class)

    def encode(self, rxn_class: str) -> int:
        return self.class_to_index.get(rxn_class, 0)  # default to Unrecognized (0)

    def decode(self, index: int) -> str:
        if 0 <= index < len(self.classes_):
            return self.classes_[index]
        return "Unrecognized"

    def to_onehot(self, rxn_class: str) -> np.ndarray:
        index = self.encode(rxn_class)
        vector = np.zeros(len(self), dtype=bool)
        vector[index] = True
        return vector
