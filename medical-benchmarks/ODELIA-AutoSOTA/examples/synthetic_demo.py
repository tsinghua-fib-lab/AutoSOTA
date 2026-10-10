"""Run the CLI end to end on synthetic inputs; not clinical performance."""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    root = p.parse_args().output
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(3107)
    arrays = {}
    for prefix, n in (("train", 12), ("val", 9)):
        arrays[f"{prefix}_sample_ids"] = np.asarray([f"{prefix}_{i}" for i in range(n)])
        arrays[f"{prefix}_case_ids"] = np.asarray(
            [f"{prefix}_case_{i//2}" for i in range(n)]
        )
        arrays[f"{prefix}_labels"] = np.arange(n) % 3
        arrays[f"{prefix}_features"] = rng.normal(size=(n, 7, 2688)).astype(np.float32)
        arrays[f"{prefix}_released_logits"] = rng.normal(size=(n, 7, 3)).astype(
            np.float32
        )
    np.savez(root / "training.npz", **arrays)
    ids = np.asarray([f"test_{i}" for i in range(9)])
    np.savez(
        root / "test_inputs.npz",
        sample_ids=ids,
        features=rng.normal(size=(9, 7, 2688)).astype(np.float32),
        released_logits=rng.normal(size=(9, 7, 3)).astype(np.float32),
    )
    np.savez(root / "test_labels.npz", sample_ids=ids, labels=np.arange(9) % 3)

    def run(*args):
        subprocess.run(
            [sys.executable, "-m", "odelia_autosota", *map(str, args)], check=True
        )

    run(
        "train",
        "--packed-cache",
        root / "training.npz",
        "--output-dir",
        root / "heads",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch-size",
        "12",
    )
    run(
        "baseline",
        "--inputs",
        root / "test_inputs.npz",
        "--output-dir",
        root / "baseline",
    )
    run(
        "infer",
        "--inputs",
        root / "test_inputs.npz",
        "--run-dir",
        root / "heads",
        "--output-dir",
        root / "adapted",
        "--device",
        "cpu",
    )
    for name in ("baseline", "adapted"):
        run(
            "evaluate",
            "--prediction-archive",
            root / name / "predictions.npz",
            "--labels",
            root / "test_labels.npz",
            "--output",
            root / f"{name}_metrics.json",
        )
    print("Synthetic demonstration complete. These numbers are not clinical results.")


if __name__ == "__main__":
    main()
