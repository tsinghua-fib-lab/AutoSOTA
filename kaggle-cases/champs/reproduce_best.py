#!/usr/bin/env python3
"""autosota's exact replay of Kaggle submission 57036892, without test labels."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", type=Path, required=True, help="External winning prediction CSV")
    parser.add_argument("--atomic", type=Path, required=True, help="External runner-up prediction CSV")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/blend_champion_050_atomic_050.csv")
    parser.add_argument("--verification-output", type=Path,
                        default=ROOT / "outputs/replay_verification.json")
    args = parser.parse_args()
    started = time.perf_counter()
    best = json.loads((ROOT / "artifacts/best_submission.json").read_text())
    evidence = json.loads((ROOT / "artifacts/kaggle_scoring/latest.json").read_text())
    expected_rows = best["rows"]
    frames, input_hashes, protected = [], {}, []
    for filename in ("champion_reproduced.csv", "runnerup_single_reproduced.csv"):
        path = (args.winner if filename == "champion_reproduced.csv" else args.atomic).resolve()
        protected.append(path)
        record = next(r for r in evidence["submissions"] if r["fileName"] == filename)
        actual = sha256(path)
        if actual != record["local_file_sha256"]:
            raise ValueError("Component SHA256 mismatch: " + filename)
        frame = pd.read_csv(path)
        if list(frame.columns) != ["id", "scalar_coupling_constant"]:
            raise ValueError("Unexpected columns: " + filename)
        if len(frame) != expected_rows or not frame["id"].is_unique:
            raise ValueError("Wrong row count or duplicate IDs: " + filename)
        if not np.isfinite(frame["scalar_coupling_constant"].to_numpy()).all():
            raise ValueError("Nonfinite predictions: " + filename)
        frames.append(frame)
        input_hashes[filename] = actual
    winner, atomic = frames
    if not np.array_equal(winner["id"].to_numpy(), atomic["id"].to_numpy()):
        raise ValueError("Component IDs or ordering differ")
    output = winner.copy()
    alpha = 0.5
    output["scalar_coupling_constant"] = (
        (1 - alpha) * winner["scalar_coupling_constant"].to_numpy()
        + alpha * atomic["scalar_coupling_constant"].to_numpy()
    )
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target in [p.resolve() for p in protected]:
        raise ValueError("Output cannot overwrite an input component")
    partial = target.with_name(target.name + ".partial")
    output.to_csv(partial, index=False, float_format="%.10g")
    actual = sha256(partial)
    if actual != best["sha256"]:
        partial.unlink()
        raise ValueError("Output differs from the file scored by Kaggle; use pinned replay dependencies")
    partial.replace(target)
    result = {
        "experiment_lead": "autosota", "verification": "PASS",
        "reproduction_scope": "exact blend replay from externally supplied component predictions",
        "rows": len(output), "unique_ids": int(output["id"].nunique()),
        "input_sha256": input_hashes, "output_file": target.name,
        "output_sha256": actual, "winner_weight": 0.5, "atomic_weight": 0.5,
        "matches_scored_submission": True, "kaggle_submission_ref": best["kaggle_submission_ref"],
        "recorded_public_score": best["kaggle_public_score"],
        "recorded_private_score": best["kaggle_private_score"],
        "scores_are_historical_api_evidence_not_new_scoring": True,
        "test_targets_read": False, "gpu_required": False, "network_required": False,
        "numpy": np.__version__, "pandas": pd.__version__,
        "replay_seconds": round(time.perf_counter() - started, 3),
    }
    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
