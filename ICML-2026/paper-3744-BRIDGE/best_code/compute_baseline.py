import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PARAMS_DIR = SCRIPT_DIR / "params"


def load_pyirt_data(input_path: str) -> tuple[dict, set, set]:
    """Load JSONL data and return responses dict, subject_ids, and item_ids."""
    responses = defaultdict(dict)  # responses[subject_id][item_id] = 0/1
    subject_ids = set()
    item_ids = set()

    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            subject_id = record["subject_id"]
            subject_ids.add(subject_id)
            for item_id, response in record["responses"].items():
                responses[subject_id][item_id] = response
                item_ids.add(item_id)

    return responses, subject_ids, item_ids


def build_response_matrix(
    responses: dict,
    subject_ids: list,
    item_ids: list,
) -> pd.DataFrame:
    """Build a response matrix (items x subjects) with NaN for missing responses."""
    matrix = pd.DataFrame(
        index=item_ids,
        columns=subject_ids,
        dtype=float,
    )

    for subject_id in subject_ids:
        for item_id, response in responses[subject_id].items():
            matrix.loc[item_id, subject_id] = response

    return matrix


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute logit(p) = log(p / (1-p)) with clipping to avoid inf."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def compute_baseline_estimates(response_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute baseline difficulty and ability estimates.

    Returns:
        item_params: DataFrame with columns [success_rate, baseline_difficulty, baseline_difficulty_logit]
        subject_params: DataFrame with columns [subject_id, success_rate, baseline_ability_logit]
    """
    # Task difficulty: row-wise average (across subjects)
    # Only count non-NaN entries
    item_success_rate = response_matrix.mean(axis=1, skipna=True)
    item_attempt_count = response_matrix.notna().sum(axis=1)

    # Transform to difficulty scale (higher = harder)
    # baseline_difficulty = 1 - success_rate (intuitive: harder tasks have lower success)
    # baseline_difficulty_logit = -logit(success_rate) (comparable to IRT b parameter)
    item_params = pd.DataFrame({
        "success_rate": item_success_rate,
        "attempt_count": item_attempt_count,
        "baseline_difficulty": 1 - item_success_rate,
        "baseline_difficulty_logit": -logit(item_success_rate.values),
    }, index=response_matrix.index)

    # Model ability: column-wise average (across items)
    subject_success_rate = response_matrix.mean(axis=0, skipna=True)
    subject_attempt_count = response_matrix.notna().sum(axis=0)

    # Transform to ability scale (higher = better)
    # baseline_ability_logit = logit(success_rate) (comparable to IRT ability parameter)
    subject_params = pd.DataFrame({
        "subject_id": subject_success_rate.index,
        "success_rate": subject_success_rate.values,
        "attempt_count": subject_attempt_count.values,
        "baseline_ability_logit": logit(subject_success_rate.values),
    })

    return item_params, subject_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute baseline difficulty/ability estimates using averaging"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL file (py-irt format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: params/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup paths
    input_path = Path(args.input_path).expanduser()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PARAMS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    item_output = output_dir / f"{stem}_baseline.csv"
    subject_output = output_dir / f"{stem}_baseline_abilities.csv"

    if args.verbose:
        print(f"Loading data from: {input_path}")

    # Load data
    responses, subject_ids, item_ids = load_pyirt_data(input_path)
    subject_ids = sorted(subject_ids)
    item_ids = sorted(item_ids)

    if args.verbose:
        print(f"  Subjects (models): {len(subject_ids)}")
        print(f"  Items (tasks): {len(item_ids)}")

    # Build response matrix
    response_matrix = build_response_matrix(responses, subject_ids, item_ids)

    # Check sparsity
    total_cells = response_matrix.size
    filled_cells = response_matrix.notna().sum().sum()
    sparsity = 1 - (filled_cells / total_cells)
    if args.verbose:
        print(f"  Response matrix sparsity: {sparsity:.1%}")

    # Compute baseline estimates
    item_params, subject_params = compute_baseline_estimates(response_matrix)

    # Save results
    item_params.to_csv(item_output)
    subject_params.to_csv(subject_output, index=False)

    if args.verbose:
        print(f"\nSaved item parameters to: {item_output}")
        print(f"Saved subject parameters to: {subject_output}")

        # Print summary statistics
        print("\n--- Item (Task) Statistics ---")
        print(f"  Success rate: mean={item_params['success_rate'].mean():.3f}, "
              f"std={item_params['success_rate'].std():.3f}")
        print(f"  Baseline difficulty (logit): mean={item_params['baseline_difficulty_logit'].mean():.3f}, "
              f"std={item_params['baseline_difficulty_logit'].std():.3f}")

        print("\n--- Subject (Model) Statistics ---")
        print(f"  Success rate: mean={subject_params['success_rate'].mean():.3f}, "
              f"std={subject_params['success_rate'].std():.3f}")
        print(f"  Baseline ability (logit): mean={subject_params['baseline_ability_logit'].mean():.3f}, "
              f"std={subject_params['baseline_ability_logit'].std():.3f}")


if __name__ == "__main__":
    main()
