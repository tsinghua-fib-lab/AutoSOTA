"""Small artificial archives for protocol tests; no clinical samples."""

import numpy as np


def training_archive_fixture():
    rng = np.random.default_rng(10)
    result = {}
    for p, n in (("train", 12), ("val", 9)):
        result[f"{p}_sample_ids"] = np.asarray([f"{p}_{i}" for i in range(n)])
        result[f"{p}_case_ids"] = np.asarray(
            [f"{p}_case{i//2}" for i in range(n)], dtype="U64"
        )
        result[f"{p}_labels"] = np.arange(n) % 3
        result[f"{p}_features"] = rng.normal(size=(n, 7, 2688)).astype(np.float32)
        result[f"{p}_released_logits"] = rng.normal(size=(n, 7, 3)).astype(np.float32)
    return result
