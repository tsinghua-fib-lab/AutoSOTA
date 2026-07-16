"""
Script to explore spectral properties of the benchmarks using spectralexplain.
Computes sparse Walsh-Hadamard Transform (WHT) for a large budget.
"""

import argparse
import os
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import spectralexplain as se

from benchmark_exhaustive_approx import BenchmarkFactory


def json_safe(value):
    """Convert numpy scalar values before writing JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value

def predict_from_fourier(fourier_transform, X):
    """
    Predict outcomes from a sparse Fourier transform dictionary.
    fourier_transform: dict of tuple(bits) -> coefficient
    X: (N, n) array of binary coalitions
    """
    preds = np.zeros(X.shape[0])
    for S, coef in fourier_transform.items():
        S_arr = np.array(S)
        # Assuming standard WHT definition where x=0 -> +1 and x=1 -> -1
        # Thus the basis function is (-1)^{S \cdot x}
        signs = (-1) ** (X @ S_arr)
        preds += coef * signs
    return preds

def fwht(a):
    """In-place Fast Walsh-Hadamard Transform."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a / len(a)

def compute_exact_fourier_transform(value_function, n_players):
    """Compute exact WHT by enumerating all 2^n coalitions."""
    N = 1 << n_players
    
    # Generate all coalitions in standard binary order
    X = np.zeros((N, n_players), dtype=bool)
    for i in range(n_players):
        X[:, i] = (np.arange(N) & (1 << i)) > 0
        
    # Evaluate all coalitions at once
    y = value_function(X)
    
    # Perform Fast Walsh-Hadamard Transform
    coefs = fwht(y.astype(float))
    
    # Map back to tuple(bits) -> coefficient
    ft = {}
    for i in range(N):
        # Create an indicator vector of length n_players
        S = tuple(1 if (i & (1 << j)) else 0 for j in range(n_players))
        # Save non-zero coefficients
        if abs(coefs[i]) > 1e-12:
            ft[S] = coefs[i]
            
    return ft


def has_full_exhaustive_values(game_instance):
    """Return True when the game has every coalition value stored."""
    n_values_stored = getattr(game_instance, "n_values_stored", 0)
    return n_values_stored >= 2 ** game_instance.n_players


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def run_exact_wht(value_function, n_players):
    print(f"    Running exhaustive Fast WHT (enumerating all {1 << n_players} coalitions)...")
    start_time = time.time()
    ft = compute_exact_fourier_transform(value_function, n_players)
    runtime = time.time() - start_time
    print(f"    Finished in {runtime:.2f} seconds.")
    print("    R^2 is 1.0000 (Exact WHT)")
    return ft, runtime, 1.0, 1 << n_players, "exact_wht"


def run_spectralexplain(
    value_function,
    n_players,
    features,
    *,
    budget,
    algorithm,
    max_order,
    test_samples,
    random_state,
):
    print(f"    Running spectralexplain Explainer with budget {budget}...")
    start_time = time.time()
    explainer = se.Explainer(
        value_function,
        features,
        sample_budget=budget,
        max_order=max_order,
        algorithm=algorithm,
    )
    runtime = time.time() - start_time
    ft = explainer.fourier_transform
    print(f"    Finished in {runtime:.2f} seconds.")

    rng = np.random.default_rng(random_state)
    X_test = rng.integers(0, 2, size=(test_samples, n_players), dtype=np.int8)
    y_true = value_function(X_test.astype(bool))
    y_pred = predict_from_fourier(ft, X_test)
    r2 = compute_r2(y_true, y_pred)
    print(f"    R^2 on {test_samples} test samples: {r2:.4f}")
    return ft, runtime, r2, budget, algorithm

def main():
    parser = argparse.ArgumentParser(description="Explore spectral properties using sparse WHT.")
    parser.add_argument(
        "--budget", 
        type=int, 
        default=100000, 
        help="Sample budget for the sparse WHT computation."
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="approximations/spectral_properties", 
        help="Directory to save the spectral explain outputs."
    )
    parser.add_argument(
        "--method",
        choices=["auto", "exact", "spectralexplain"],
        default="auto",
        help="Use exact WHT, spectralexplain, or exact only when full exhaustive values exist.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="spex",
        help="spectralexplain algorithm to use when not running exact WHT.",
    )
    parser.add_argument(
        "--max_order",
        type=int,
        default=10,
        help="Maximum interaction order passed to spectralexplain.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=1000,
        help="Number of random coalitions used for spectralexplain R^2 validation.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=40,
        help="Random seed for spectralexplain validation samples.",
    )
    parser.add_argument(
        "--max_instances_per_game",
        type=int,
        default=None,
        help="Maximum number of instances to run for each game type.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    configs = ["shapiq-benchmark/benchmarks/configuration_interventional_sv.json",
               "shapiq-benchmark/benchmarks/configuration_exhaustive_sv.json"]
    for c in configs:
        # Load benchmarks
        print(f"Loading benchmarks from {c}...")
        benchmarks = BenchmarkFactory.load_benchmarks_from_json(config_path=c)

        for game_identifier, benchmark_info in benchmarks.items():
            print(f"Processing game type: {game_identifier}")
            games = benchmark_info["games"]

            if args.max_instances_per_game is None:
                game_iter = games
            else:
                game_iter = islice(games, args.max_instances_per_game)

            for id_explain, game_instance in enumerate(game_iter):
                print(f"  Game index: {id_explain}, n_players: {game_instance.n_players}")
                
                # Setup value function wrapper for spectralexplain
                def value_function(coalitions: np.ndarray) -> np.ndarray:
                    # Vectorised call to game_instance. Important: Ensure we pass bool/int, not floats
                    # because the shapiq baseline imputer crashes if we try to invert a float array (~array).
                    return game_instance(coalitions.astype(bool))

                d = game_instance.n_players
                features = list(range(d))
                use_exact = args.method == "exact" or (
                    args.method == "auto" and has_full_exhaustive_values(game_instance)
                )
                if args.method == "auto" and use_exact:
                    print("    Found full exhaustive values; using exact WHT.")
                elif args.method == "auto":
                    n_values_stored = getattr(game_instance, "n_values_stored", 0)
                    print(
                        "    Full exhaustive values not found "
                        f"({n_values_stored}/{1 << d} stored); using spectralexplain."
                    )
                
                try:
                    if use_exact:
                        ft, runtime, r2, budget_used, method_used = run_exact_wht(value_function, d)
                    else:
                        ft, runtime, r2, budget_used, method_used = run_spectralexplain(
                            value_function,
                            d,
                            features,
                            budget=args.budget,
                            algorithm=args.algorithm,
                            max_order=args.max_order,
                            test_samples=args.test_samples,
                            random_state=args.random_state + id_explain,
                        )

                    # Save results
                    results = {
                        "game_identifier": game_identifier,
                        "id_explain": id_explain,
                        "n_players": d,
                        "method": method_used,
                        "budget_used": budget_used,
                        "runtime_seconds": runtime,
                        "r2_test": r2,
                        "fourier_transform": {
                            str(k): json_safe(v) for k, v in ft.items()
                        } if ft else {}
                    }
                    
                    out_path = Path(args.out_dir) / f"{game_identifier}_{id_explain}_spex_{budget_used}.json"
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=4)
                        
                except Exception as e:
                    print(f"    Failed to compute spectral properties for {game_identifier} [{id_explain}]: {e}")

if __name__ == "__main__":
    main()
