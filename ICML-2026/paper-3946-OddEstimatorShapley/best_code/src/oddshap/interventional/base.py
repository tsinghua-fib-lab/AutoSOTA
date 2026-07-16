import numpy as np
from dataclasses import dataclass
from shapiq.explainer.tree.base import TreeModel


@dataclass
class CoeffsTreeModel(TreeModel):
    coeffs: np.ndarray | None = None
