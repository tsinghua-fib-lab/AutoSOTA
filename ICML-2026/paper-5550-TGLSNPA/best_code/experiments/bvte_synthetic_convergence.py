"""
Generate per-seed TE estimates across multiple sample sizes for the percentile plot.

For each T in a list, we:
  1) sample a 2D ARTG process (optionally reuse cached samples),
  2) split into train/test (T_train = T, T_test = T//4),
  3) estimate all pairwise transfer entropies on (train, test),
  4) save the TE matrices per seed.

Outputs:
- data/synthetic_bvte/te_estimates.json

Note: "ground truth" is estimated with T=100000.
"""
__date__ = "May - October 2025"


import argparse
import json
import os
from typing import Iterable, List

import jax
import jax.numpy as jnp

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fit_artg import (
    estimate_transfer_entropies,
    estimate_all_pairwise_transfer_entropies,
)
from src.simulate_ar import (
    ARTGPriorParams,
    sample_artg_params,
    sample_artg,
)


# ----------------------------- utilities --------------------------------------

def _seed_list(multiplier: int, num_seeds: int) -> List[int]:
    """Replicate original seed pattern: seeds = multiplier * [1..num_seeds]."""
    return [int(multiplier * i) for i in range(1, num_seeds + 1)]


# --------------------------- core sweep logic ---------------------------------

def run_te_percentile_sweep(
    *,
    L: int = 10,
    D: int = 2,
    use_prior: bool = False,
    num_seeds: int = 40,
    seed_multiplier: int = 17,
    param_seed: int = 1,
    sample_sizes: Iterable[int] = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
) -> None:
    """
    Run the TE estimation sweep and write per-seed TE matrices to disk.

    Parameters
    ----------
    L : int
        History length used by ARTG/TE estimators.
    D : int
        Dimensionality (kept 2 here).
    use_prior : bool
        Whether to pass an ARTG prior to the estimator.
    num_seeds : int
        Number of trials per T.
    seed_multiplier : int
        Seeds are constructed as multiplier * i for i in 1..num_seeds.
    param_seed : int
        PRNG seed for sampling ARTG parameters (kept constant across runs).
    sample_sizes : iterable of int
        Training lengths T to evaluate. Test length is T//4.
    """
    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_bvte",
        "te_estimates.json",
    )
    print("Output JSONs:", out_json)
    out_dir = os.path.split(out_json)[0]
    os.makedirs(out_dir, exist_ok=True)
    
    param_key = jax.random.PRNGKey(param_seed)
    seeds = _seed_list(seed_multiplier, num_seeds)

    # Sample parameters & data
    prior = ARTGPriorParams(d=D, allow_12=False)
    params = sample_artg_params(param_key, prior, L)

    # est_te[T][seed][direction] = val, where direction in ["1->2", "2->1"]
    est_te: Dict[str, Dict[str, Dict[str, float]]] = {}

    for T in sample_sizes:
        est_te[f"T={T}"] = {}
       
        T_test = T // 4
        T_total = T + T_test

        for seed in seeds:
            print("Seed:", seed)
            key = jax.random.PRNGKey(seed)
            key1, key2 = jax.random.split(key, 2)

            # Sample system.
            samples = sample_artg(key1, T_total, params)

            # Train/test split
            train, test = samples[:, :T], samples[:, T:]  # shapes (D, T), (D, T_test)

            train_prior = ARTGPriorParams() if use_prior else None

            # Estimate TE.
            te = estimate_all_pairwise_transfer_entropies(
                key2, train, test, L=L, prior=train_prior
            )
            print("Estimated pairwise TE:\n", te)
            
            est_te[f"T={T}"][f"seed={seed}"] = {"1->2": float(te[0,1]), "2->1": float(te[1,0])}

            with open(out_json, "w") as f:
                json.dump(est_te, f, indent=4)


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers, e.g., '64,128,256'."""
    return [int(x) for x in s.split(",") if x.strip()]

def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TE percentile data across sample sizes (per-seed TE matrices)."
    )
    parser.add_argument("--L", type=int, default=10, help="History length.")
    parser.add_argument("--D", type=int, default=2, help="Dimensionality (kept 2 in paper runs).")
    parser.add_argument("--use-prior", action="store_true", help="Pass an ARTG prior to the estimator.")
    parser.add_argument("--num-seeds", type=int, default=40, help="Trials per T.")
    parser.add_argument("--seed-multiplier", type=int, default=17, help="Seeds are multiplier * i.")
    parser.add_argument("--param-seed", type=int, default=1, help="PRNG seed for parameter sampling.")
    parser.add_argument(
        "--samples",
        type=_parse_int_list,
        default="64,128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated list of T values.",
    )

    args = parser.parse_args()
    run_te_percentile_sweep(
        L=args.L,
        D=args.D,
        use_prior=args.use_prior,
        num_seeds=args.num_seeds,
        seed_multiplier=args.seed_multiplier,
        param_seed=args.param_seed,
        sample_sizes=args.samples,
    )



if __name__ == "__main__":
    _run_cli()
