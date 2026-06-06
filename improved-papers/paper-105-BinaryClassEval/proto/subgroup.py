"""Domain models for subgroup analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np


@dataclass(frozen=True)
class SubgroupResults:
    """Represents a subgroup with prediction results and metadata.

    Attributes
    ----------
    name : str
        Unique identifier for the subgroup.
    y_true : np.ndarray
        Binary ground-truth labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    prevalence : float, default=None
        Prevalence (proportion of positive class) in the subgroup.
        Computed from y_true if None.
    display_label : str, default=None
        Label to display in plots and legends. Falls back to name if None.
    auc_roc : Optional[float], default=None
        AUC-ROC value if computed.
    """

    name: str
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    prevalence: float = None  # Computed from y_true if None provided
    display_label: str = None  # Falls back to name if None
    auc_roc: Optional[float] = None  # AUC-ROC value if computed

    def __post_init__(self):
        """Validate inputs and set defaults.

        Ensures arrays are numpy arrays, validates array lengths match,
        computes prevalence if not provided, and sets display_label if not provided.
        """
        # Convert inputs to numpy arrays if needed
        if not isinstance(self.y_true, np.ndarray):
            object.__setattr__(self, 'y_true', np.array(self.y_true))

        if not isinstance(self.y_pred_proba, np.ndarray):
            object.__setattr__(self, 'y_pred_proba', np.array(self.y_pred_proba))

        # Validate arrays
        if len(self.y_true) != len(self.y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

        # Calculate prevalence if not provided
        if self.prevalence is None:
            object.__setattr__(self, 'prevalence', np.mean(self.y_true))

        # Set display_label if not provided
        if self.display_label is None:
            object.__setattr__(self, 'display_label', self.name)

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the subgroup.

        Returns
        -------
        int
            Number of samples in this subgroup.
        """
        return len(self.y_true)

    @property
    def log_odds(self) -> float:
        """Return the log-odds of the subgroup prevalence.

        Returns
        -------
        float
            Log-odds transformation of prevalence: log(p / (1-p)).
        """
        return np.log(self.prevalence / (1 - self.prevalence))