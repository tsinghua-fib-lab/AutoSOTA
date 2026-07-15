""" Annotator pool and confusion matrix definitions."""

import numpy as np
from typing import Dict, List, Optional


# Implemented step 3: Set pre-defined confusion matrices for annotator types
CONFUSION_MATRICES = {
    "normal": np.array([
        [0.85, 0.14, 0.01],  # z=0 (incorrect)
        [0.10, 0.80, 0.10],  # z=1 (partial)
        [0.01, 0.14, 0.85],  # z=2 (correct)
    ]),
    "strict": np.array([
        [0.92, 0.08, 0.00],  # z=0
        [0.35, 0.60, 0.05],  # z=1
        [0.10, 0.55, 0.35],  # z=2
    ]),
    "lenient": np.array([
        [0.35, 0.55, 0.10],  # z=0
        [0.05, 0.60, 0.35],  # z=1
        [0.00, 0.08, 0.92],  # z=2
    ]),
    "adversarial": np.array([
        [1/3, 1/3, 1/3],  # z=0
        [1/3, 1/3, 1/3],  # z=1
        [1/3, 1/3, 1/3],  # z=2
    ]),
}


class AnnotatorPool:
    """
    Pool of annotators with different types and confusion matrices.
    """
    
    def __init__(
        self,
        n_annotators: int,
        distribution: Dict[str, int],
        rng: np.random.Generator = None
    ):
        """
        Initialize annotator pool.
        
        Args:
            n_annotators: Total number of annotators
            distribution: Dict mapping annotator type to count
            rng: Random number generator
        """
        self.n_annotators = n_annotators
        self.distribution = distribution
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Validate distribution
        total = sum(distribution.values())
        if total != n_annotators:
            raise ValueError(
                f"Distribution sum ({total}) doesn't match n_annotators ({n_annotators})"
            )
        
        # Assign types to annotators
        self.annotator_types: List[str] = []
        for ann_type, count in distribution.items():
            self.annotator_types.extend([ann_type] * count)
        
        # Shuffle to randomize assignment
        self.rng.shuffle(self.annotator_types)
        
        # Store confusion matrices for each annotator
        self.confusion_matrices = np.zeros((n_annotators, 3, 3))
        for r, ann_type in enumerate(self.annotator_types):
            self.confusion_matrices[r] = CONFUSION_MATRICES[ann_type].copy()
    
    def get_confusion_matrix(self, annotator_id: int) -> np.ndarray:
        """Get confusion matrix for a specific annotator."""
        return self.confusion_matrices[annotator_id]
    
    def get_adjusted_confusion_matrix(
        self,
        annotator_id: int,
        ambiguity: float
    ) -> np.ndarray:
        """
        Implemented step 4: get confusion matrix adjusted for item ambiguity.
        
        π^(d)_r = (1 - d) * π_r + d * U
        
        where U is uniform (1/3 for each cell).
        
        Args:
            annotator_id: Annotator index
            ambiguity: Item ambiguity value d in [0, 1]
        
        Returns:
            Adjusted confusion matrix
        """
        base_matrix = self.confusion_matrices[annotator_id]
        uniform = np.ones((3, 3)) / 3
        
        adjusted = (1 - ambiguity) * base_matrix + ambiguity * uniform
        return adjusted
    
    def sample_annotators(self, m: int) -> np.ndarray:
        """
        Implemented step 4: sample m annotators without replacement.
        
        Args:
            m: Number of annotators to sample
        
        Returns:
            Array of annotator indices
        """
        return self.rng.choice(self.n_annotators, size=m, replace=False)
    
    def get_type_counts(self) -> Dict[str, int]:
        """ Helper: Get count of each annotator type."""
        from collections import Counter
        return dict(Counter(self.annotator_types))
    
    def summary(self) -> str:
        """Helper: Return summary string of annotator pool."""
        counts = self.get_type_counts()
        lines = [f"AnnotatorPool(n={self.n_annotators})"]
        for ann_type, count in sorted(counts.items()):
            lines.append(f"  {ann_type}: {count}")
        return "\n".join(lines)


def create_annotator_pool(
    n_annotators: int = 30,
    distribution: Optional[Dict[str, int]] = None,
    adversarial_fraction: Optional[float] = None,
    rng: np.random.Generator = None
) -> AnnotatorPool:
    """
    Factory function to create annotator pool.
    
    Args:
        n_annotators: Total annotators
        distribution: Explicit distribution dict
        adversarial_fraction: If set, compute distribution from this
        rng: Random number generator
    
    Returns:
        AnnotatorPool instance
    """
    if distribution is None:
        if adversarial_fraction is not None:
            # Compute distribution from adversarial fraction
            n_adv = int(round(adversarial_fraction * n_annotators))
            remaining = n_annotators - n_adv
            
            # Default ratios for non-adversarial (18:6:4 = 9:3:2)
            n_normal = int(round(remaining * 9 / 14))
            n_strict = int(round(remaining * 3 / 14))
            n_lenient = remaining - n_normal - n_strict
            
            distribution = {
                "normal": n_normal,
                "strict": n_strict,
                "lenient": n_lenient,
                "adversarial": n_adv,
            }
        else:
            # Default distribution
            distribution = {
                "normal": 18,
                "strict": 6,
                "lenient": 4,
                "adversarial": 2,
            }
    
    return AnnotatorPool(n_annotators, distribution, rng)