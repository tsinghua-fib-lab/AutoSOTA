# causalbench_export_model.py
# Custom CausalBench inference function that EXPORTS the processed dataset
# and returns a trivial edge list (empty).

from __future__ import annotations

import os
import json
from typing import List, Tuple, Optional

import numpy as np


def infer_graph(
    expression_matrix: np.ndarray,
    interventions: List[str],
    gene_names: List[str],
    training_regime,
    seed: int = 0,
    export_path: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    This function signature matches the "custom" model interface used by causalbench_run.

    It writes:
      - expression_matrix (N x D) float32
      - interventions (N) strings (or empty strings)
      - gene_names (D) strings
      - regime + seed

    Then returns [] edges (we’re not trying to score a method here, just export data).
    """
    if export_path is None:
        export_path = os.environ.get("CAUSALBENCH_EXPORT_PATH", "causalbench_export.npz")

    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

    X = np.asarray(expression_matrix, dtype=np.float32)
    genes = np.asarray(list(gene_names), dtype=object)
    intervs = np.asarray(list(interventions), dtype=object)

    np.savez_compressed(
        export_path,
        X=X,
        gene_names=genes,
        interventions=intervs,
        seed=int(seed),
        training_regime=str(training_regime),
    )

    meta = {
        "export_path": export_path,
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "seed": int(seed),
        "training_regime": str(training_regime),
    }
    with open(os.path.splitext(export_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export_model] wrote: {export_path} (+ .json)")
    return []

