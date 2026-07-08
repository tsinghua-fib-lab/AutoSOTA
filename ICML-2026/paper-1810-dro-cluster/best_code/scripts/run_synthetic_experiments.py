"""
FILE: run_synthetic_experiments.py
INPUT: CLI arguments selecting scenarios, methods, seeds, and solver budgets.
OUTPUT: CSV summaries under results/synthetic_results by default.
POS: Command-line reproduction entry point for source-grounded synthetic experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from dro_subspace.baselines import kmedoids_clustering, multifactor_clustering
from dro_subspace.cord import cluster_list_to_membership, cord_clustering
from dro_subspace.dro import compute_dro_coefficients
from dro_subspace.experiments import (
    DEFAULT_MAX_OUTER_ITERATIONS,
    DEFAULT_N_EXPERIMENTS,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_SEED_START,
    ExperimentConfig,
    generate_trial,
)
from dro_subspace.metrics import clustering_score, spectral_clustering
from dro_subspace.nodewise import compute_lasso_coefficients, compute_sqrt_lasso_coefficients

DEFAULT_OUTPUT_DIR = Path("results") / "synthetic_results"
DEFAULT_METHODS = "dro,cord,kmeans"
DEFAULT_SCENARIO = "main"
DEFAULT_DRO_ALPHA = 0.05
COMMON_FACTOR_VALUES = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))
NOISE_VALUES = tuple(float(value) for value in np.arange(0.1, 2.1, 0.1))
COMMON_FACTOR_NOISE_RANGE = (1.0, 1.0)
NO_GLOBAL_FACTOR_RANGE = (0.0, 0.0)
ADDITIONAL_NOISE_RANGE = (0.1, 0.1)
METHOD_CHOICES = {"dro", "lasso", "sqrt_lasso", "cord", "mfc", "kmedoids", "kmeans"}
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


def scenario_configs(scenario: str) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    if scenario == "main":
        return [("main", (0.0, 0.5), (0.0, 0.5))]
    if scenario == "common-factor":
        return [(f"common_factor_{rho:.1f}", (rho, rho), COMMON_FACTOR_NOISE_RANGE) for rho in COMMON_FACTOR_VALUES]
    if scenario == "noise":
        return [(f"noise_{noise:.1f}", NO_GLOBAL_FACTOR_RANGE, (noise, noise)) for noise in NOISE_VALUES]
    if scenario == "additional":
        return [("additional_no_global_factor", NO_GLOBAL_FACTOR_RANGE, ADDITIONAL_NOISE_RANGE)]
    if scenario == "all":
        configs = scenario_configs("main")
        configs.extend(scenario_configs("common-factor"))
        configs.extend(scenario_configs("noise"))
        configs.extend(scenario_configs("additional"))
        return configs
    raise ValueError("scenario must be one of: main, common-factor, noise, additional, all.")


def _existing_records(path: Path) -> list[dict[str, int | float | str]]:
    if not path.exists():
        return []
    return [dict(row) for row in pd.read_csv(path).to_dict("records")]


def _save_records(records: list[dict[str, int | float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def run_method(
    method: str,
    samples: np.ndarray,
    labels: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[float, float]:
    if method == "dro":
        coefs = compute_dro_coefficients(
            samples,
            alpha=DEFAULT_DRO_ALPHA,
            n_simulations=config.n_simulations,
            seed=seed,
            max_outer_iterations=config.max_outer_iterations,
        )
        predicted = spectral_clustering(coefs, config.n_clusters, config.n_samples)
    elif method == "lasso":
        coefs = compute_lasso_coefficients(samples)
        predicted = spectral_clustering(coefs, config.n_clusters, config.n_samples)
    elif method == "sqrt_lasso":
        coefs = compute_sqrt_lasso_coefficients(
            samples,
            alpha=DEFAULT_DRO_ALPHA,
            n_sim=config.n_simulations,
            seed=seed,
            sep=True,
            n_clusters=config.n_clusters,
        )
        predicted = spectral_clustering(coefs, config.n_clusters, config.n_samples)
    elif method == "cord":
        corr = np.corrcoef(samples.T)
        clusters = cord_clustering(
            pd.DataFrame(corr),
            n_clusters=config.n_clusters,
            min_or_max="min",
            distinguish_direction=False,
        )
        predicted = cluster_list_to_membership(clusters)
    elif mfc_spec := _parse_mfc_method(method):
        n_factors_global, n_factors_local = mfc_spec
        predicted = multifactor_clustering(
            samples,
            config.n_clusters,
            n_factors_global=n_factors_global,
            n_factors_local=n_factors_local,
        )
    elif method == "kmedoids":
        predicted = kmedoids_clustering(samples, config.n_clusters)
    elif method == "kmeans":
        predicted = KMeans(n_clusters=config.n_clusters, random_state=0, n_init=10).fit(samples.T).labels_.astype(int)
    else:
        raise ValueError(f"Unsupported method: {method}.")
    score = clustering_score(labels, predicted)
    return score.accuracy, score.adjusted_mutual_information


def run_synthetic_suite(args: argparse.Namespace) -> pd.DataFrame:
    methods = parse_methods(args.methods)
    output_path = args.output_dir / "synthetic_experiments.csv"
    records = _existing_records(output_path)
    done = {(str(row["scenario"]), int(row["seed"]), str(row["method"])) for row in records}

    seeds = list(range(args.seed_start, args.seed_start + args.n_experiments))
    for scenario_name, rho_range, noise_range in scenario_configs(args.scenario):
        config = ExperimentConfig(
            n_samples=args.n_samples,
            n_variables=args.n_variables,
            n_clusters=args.n_clusters,
            rho_range=rho_range,
            noise_range=noise_range,
            n_simulations=args.n_simulations,
            max_outer_iterations=args.max_outer_iterations,
        )
        for seed in tqdm(seeds, desc=f"{scenario_name} seeds"):
            samples, labels = generate_trial(config, seed)
            for method in methods:
                key = (scenario_name, seed, method)
                if key in done:
                    continue
                accuracy, ami = run_method(method, samples, labels, config, seed)
                records.append(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method,
                        "rho_low": rho_range[0],
                        "rho_high": rho_range[1],
                        "noise_low": noise_range[0],
                        "noise_high": noise_range[1],
                        "accuracy": accuracy,
                        "ami": ami,
                    }
                )
                _save_records(records, output_path)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-grounded synthetic DRO subspace clustering experiments.")
    parser.add_argument(
        "--scenario",
        choices=["main", "common-factor", "noise", "additional", "all"],
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--n-experiments", type=int, default=DEFAULT_N_EXPERIMENTS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--n-variables", type=int, default=500)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--n-simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument("--max-outer-iterations", type=int, default=DEFAULT_MAX_OUTER_ITERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    frame = run_synthetic_suite(parse_args())
    logging.info("Synthetic experiment rows: %s", len(frame))


if __name__ == "__main__":
    main()
