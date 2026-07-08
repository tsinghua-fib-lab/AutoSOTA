"""
FILE: run_face_experiments.py
INPUT: Local CroppedYalePNG directory, split choices, methods, and solver budgets.
OUTPUT: CSV summaries under results/face_results by default.
POS: Command-line reproduction entry point for source-grounded face experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dro_subspace.baselines import kmedoids_clustering, multifactor_clustering
from dro_subspace.cord import cluster_list_to_membership, cord_clustering
from dro_subspace.dro import compute_dro_coefficients
from dro_subspace.experiments import DEFAULT_MAX_OUTER_ITERATIONS, DEFAULT_N_SIMULATIONS
from dro_subspace.images import (
    DEFAULT_FACE_HEIGHT,
    DEFAULT_FACE_WIDTH,
    face_design_for_indices,
    indices_for_subjects,
    load_cropped_yale_faces,
    random_subject_combination,
    standard_subject_splits,
)
from dro_subspace.metrics import clustering_score, spectral_clustering
from dro_subspace.nodewise import compute_lasso_coefficients, compute_sqrt_lasso_coefficients

DEFAULT_DATA_DIR = Path("data") / "external" / "CroppedYalePNG"
DEFAULT_OUTPUT_DIR = Path("results") / "face_results"
DEFAULT_METHODS = "dro,cord"
DEFAULT_SPLITS = "all"
DEFAULT_RANDOM_SEED_START = 2024
DEFAULT_RANDOM_TRIALS = 20
DEFAULT_SUBJECT_COUNT = 10
DEFAULT_DRO_ALPHA = 0.05
METHOD_CHOICES = {"dro", "lasso", "sqrt_lasso", "cord", "mfc", "kmedoids"}
MFC_PREFIX = "mfc_"


def parse_methods(methods: str) -> list[str]:
    selected = [method.strip().lower() for method in methods.split(",") if method.strip()]
    unknown = sorted(method for method in selected if method not in METHOD_CHOICES and _parse_mfc_method(method) is None)
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}.")
    return selected


def _parse_mfc_method(method: str) -> tuple[int, int] | None:
    if method == "mfc":
        return (1, 1)
    if not method.startswith(MFC_PREFIX):
        return None
    factor_spec = method.removeprefix(MFC_PREFIX)
    if "x" not in factor_spec:
        return None
    global_text, local_text = factor_spec.split("x", maxsplit=1)
    if not global_text.isdigit() or not local_text.isdigit():
        return None
    n_factors_global = int(global_text)
    n_factors_local = int(local_text)
    if n_factors_global < 0 or n_factors_local <= 0:
        return None
    return n_factors_global, n_factors_local


def _existing_records(path: Path) -> list[dict[str, int | float | str]]:
    if not path.exists():
        return []
    return [dict(row) for row in pd.read_csv(path).to_dict("records")]


def _save_records(records: list[dict[str, int | float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def split_specs(labels: np.ndarray, args: argparse.Namespace) -> list[tuple[str, int, np.ndarray]]:
    specs: list[tuple[str, int, np.ndarray]] = []
    if args.splits in {"standard", "all"}:
        for split_idx, subjects in enumerate(standard_subject_splits(labels), start=1):
            specs.append(("standard", split_idx, subjects))
    if args.splits in {"random", "all"}:
        labels_unique = sorted(int(label) for label in set(labels))
        for seed in range(args.random_seed_start, args.random_seed_start + args.random_trials):
            subjects = random_subject_combination(labels_unique, args.subject_count, seed)
            specs.append(("random", seed, subjects))
    return specs


def run_method(
    method: str,
    samples: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_simulations: int,
    max_outer_iterations: int,
    seed: int,
) -> tuple[float, float]:
    if method == "dro":
        coefs = compute_dro_coefficients(
            samples,
            alpha=DEFAULT_DRO_ALPHA,
            n_simulations=n_simulations,
            seed=seed,
            max_outer_iterations=max_outer_iterations,
        )
        predicted = spectral_clustering(coefs, n_clusters, samples.shape[0])
    elif method == "lasso":
        coefs = compute_lasso_coefficients(samples)
        predicted = spectral_clustering(coefs, n_clusters, samples.shape[0])
    elif method == "sqrt_lasso":
        coefs = compute_sqrt_lasso_coefficients(
            samples,
            alpha=DEFAULT_DRO_ALPHA,
            n_sim=n_simulations,
            seed=seed,
            sep=True,
            n_clusters=n_clusters,
        )
        predicted = spectral_clustering(coefs, n_clusters, samples.shape[0])
    elif method == "cord":
        corr = np.corrcoef(samples.T)
        clusters = cord_clustering(
            pd.DataFrame(corr),
            n_clusters=n_clusters,
            min_or_max="min",
            distinguish_direction=False,
        )
        predicted = cluster_list_to_membership(clusters)
    elif mfc_spec := _parse_mfc_method(method):
        n_factors_global, n_factors_local = mfc_spec
        predicted = multifactor_clustering(
            samples,
            n_clusters,
            n_factors_global=n_factors_global,
            n_factors_local=n_factors_local,
        )
    elif method == "kmedoids":
        predicted = kmedoids_clustering(samples, n_clusters)
    else:
        raise ValueError(f"Unsupported method: {method}.")
    score = clustering_score(labels, predicted)
    return score.accuracy, score.adjusted_mutual_information


def run_face_suite(args: argparse.Namespace) -> pd.DataFrame:
    methods = parse_methods(args.methods)
    images, labels = load_cropped_yale_faces(args.data_dir, new_width=args.width, new_height=args.height)
    output_path = args.output_dir / f"face_experiments_{args.height}x{args.width}.csv"
    records = _existing_records(output_path)
    done = {(str(row["split_type"]), int(row["split_id"]), str(row["method"])) for row in records}

    for split_type, split_id, subjects in tqdm(split_specs(labels, args), desc="Face splits"):
        indices = indices_for_subjects(labels, subjects)
        samples, selected_labels = face_design_for_indices(images, labels, indices)
        n_clusters = len(set(int(label) for label in selected_labels))
        subject_text = " ".join(str(int(subject)) for subject in subjects)
        for method in methods:
            key = (split_type, split_id, method)
            if key in done:
                continue
            accuracy, ami = run_method(
                method,
                samples,
                selected_labels,
                n_clusters,
                args.n_simulations,
                args.max_outer_iterations,
                seed=split_id,
            )
            records.append(
                {
                    "split_type": split_type,
                    "split_id": split_id,
                    "subjects": subject_text,
                    "method": method,
                    "n_images": samples.shape[1],
                    "n_clusters": n_clusters,
                    "accuracy": accuracy,
                    "ami": ami,
                }
            )
            _save_records(records, output_path)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-grounded Extended Yale B face clustering experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", choices=["standard", "random", "all"], default=DEFAULT_SPLITS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--width", type=int, default=DEFAULT_FACE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FACE_HEIGHT)
    parser.add_argument("--subject-count", type=int, default=DEFAULT_SUBJECT_COUNT)
    parser.add_argument("--random-seed-start", type=int, default=DEFAULT_RANDOM_SEED_START)
    parser.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    parser.add_argument("--n-simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument("--max-outer-iterations", type=int, default=DEFAULT_MAX_OUTER_ITERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    frame = run_face_suite(parse_args())
    logging.info("Face experiment rows: %s", len(frame))


if __name__ == "__main__":
    main()
