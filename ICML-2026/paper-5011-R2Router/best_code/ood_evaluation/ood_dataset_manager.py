"""
OOD Dataset Manager for category-based out-of-distribution evaluation.
"""
import pickle
import numpy as np
from typing import Tuple, List


class OODDatasetManager:
    """Manages OOD train/test splits based on category leave-one-out strategy."""

    def __init__(self, embeddings_path: str, ood_splits_path: str, test_category: str):
        """
        Initialize the OOD dataset manager.

        Args:
            embeddings_path: Path to the full embeddings pickle file
            ood_splits_path: Path to the OOD splits pickle file
            test_category: Category to use as OOD test set
        """
        # Load embeddings
        with open(embeddings_path, "rb") as f:
            emb_data = pickle.load(f)
            if isinstance(emb_data, dict):
                self.embeddings = emb_data["embeddings"]
            else:
                self.embeddings = emb_data

        # Load OOD splits
        with open(ood_splits_path, "rb") as f:
            ood_splits = pickle.load(f)

        if test_category not in ood_splits:
            available = list(ood_splits.keys())
            raise ValueError(
                f"Test category '{test_category}' not found. "
                f"Available categories: {available}"
            )

        self.test_category = test_category
        split = ood_splits[test_category]

        self.train_idx = split['train_idx']
        self.test_idx = split['test_idx']

        print(f"OODDatasetManager initialized:")
        print(f"  Test category (OOD): {test_category}")
        print(f"  Total samples: {len(self.embeddings)}")
        print(f"  Train samples: {len(self.train_idx)} ({100*len(self.train_idx)/len(self.embeddings):.1f}%)")
        print(f"  Test samples (OOD): {len(self.test_idx)} ({100*len(self.test_idx)/len(self.embeddings):.1f}%)")

    def get_embeddings(self) -> np.ndarray:
        """Return the full embeddings array."""
        return self.embeddings

    def get_train_embeddings(self) -> np.ndarray:
        """Return training embeddings."""
        return self.embeddings[self.train_idx]

    def get_test_embeddings(self) -> np.ndarray:
        """Return test embeddings (OOD)."""
        return self.embeddings[self.test_idx]

    def get_split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return train and test indices."""
        return self.train_idx, self.test_idx

    def get_train_idx(self) -> np.ndarray:
        """Return training indices."""
        return self.train_idx

    def get_test_idx(self) -> np.ndarray:
        """Return test indices (OOD)."""
        return self.test_idx

    def get_test_category(self) -> str:
        """Return the OOD test category name."""
        return self.test_category


def get_all_categories(ood_splits_path: str) -> List[str]:
    """
    Get list of all available categories for OOD evaluation.

    Args:
        ood_splits_path: Path to the OOD splits pickle file

    Returns:
        List of category names
    """
    with open(ood_splits_path, "rb") as f:
        ood_splits = pickle.load(f)
    return sorted(list(ood_splits.keys()))
