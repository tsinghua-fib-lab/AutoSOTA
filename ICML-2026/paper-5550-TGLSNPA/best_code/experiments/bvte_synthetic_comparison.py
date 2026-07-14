"""
Generate accuracy-vs-samples data for the bivariate causal discovery plot.

For each sample size T and many random seeds, we:
  1) sample a 2D ARTG process (with fixed causal intensity),
  2) compute several bivariate TE/cause-direction estimators on sine/cosine
     transforms of each variable,
  3) count the percentage of runs that identify the correct direction.

Outputs
-------
- data/synthetic_bvte/accuracies_*.json
"""
__date__ = 'July - October 2025'

import argparse
import json
import os
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

# External estimators
import infomeasure as im
from statsmodels.tsa.stattools import grangercausalitytests
from idtxl.estimators_jidt import JidtKraskovTE

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate_ar import (
    ARTGPriorParams,
    sample_artg_params,
    sample_artg,
)

from src.fit_artg import estimate_all_pairwise_transfer_entropies



def run_bvte_exp(
    *,
    sample_sizes: List[int] = (35, 50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200, 215, 230, 245, 260, 275, 290),
    num_seeds: int = 150,
    seed_multiplier: int = 17,
    strength: float = 0.1,
) -> None:
    """
    Run the TE comparison sweep and write JSON summaries.

    Parameters
    ----------
    sample_sizes : list[int]
        Values of T to evaluate (training length).
    num_seeds : int
        Number of random trials per T.
    seed_multiplier : int
        Seeds are constructed as `seed_multiplier * i`, i=1..num_seeds.
    strength : float
        Causal intensity used when sampling ARTG parameters.
    """
    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_bvte",
        f"accuracies_{strength}_{num_seeds}.json",
    )
    print("Output JSON:", out_json)
    os.makedirs(os.path.split(out_json)[0], exist_ok=True)

    # Infomeasure configuration
    approaches = {
        "kernel": {"bandwidth": 0.5, "kernel": "box"},
        "metric": {"k": 4, "minkowski_p": 2},
    }

    # Fixed model/settings
    L = 10
    D = 2
    use_prior = False

    # Build seeds deterministically.
    seeds = [seed_multiplier * i for i in range(1, num_seeds + 1)]

    results_json: Dict[int, Dict[str, float]] = {}

    for T in sample_sizes:        
        # Counters
        correct_torus_sm  = 0
        correct_torus_mle = 0
        correct_idtxl     = 0
        correct_kernel    = 0
        correct_metric    = 0
        correct_granger   = 0

        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            key1, key2, key3, key4 = jax.random.split(key, 4)

            # Sample a 2D ARTG process
            prior = ARTGPriorParams(
                d=D,
                allow_12=False,
                variance_initial_other=10,
                variance_initial_self=10,
                variance_slope=0.9,
            )
            params = sample_artg_params(key1, prior, L, strength)
            samples = sample_artg(key2, T, params)      # shape (D, T)

            # Use sin/cos channels as in the original
            x = np.array([np.cos(samples[0]), np.sin(samples[0])])  # (2, T)
            y = np.array([np.cos(samples[1]), np.sin(samples[1])])  # (2, T)

            # ----------------- Infomeasure baselines -----------------
            te_x_to_y = im.transfer_entropy(x[0], y[0], approach="kernel",
                                            src_hist_len=L, dest_hist_len=L, **approaches["kernel"])
            te_y_to_x = im.transfer_entropy(y[0], x[0], approach="kernel",
                                            src_hist_len=L, dest_hist_len=L, **approaches["kernel"])
            if te_y_to_x > 0 and (te_y_to_x - te_x_to_y) > 0:
                correct_kernel += 1

            te_x_to_y = im.transfer_entropy(x[0], y[0], approach="metric",
                                            src_hist_len=L, dest_hist_len=L, **approaches["metric"])
            te_y_to_x = im.transfer_entropy(y[0], x[0], approach="metric",
                                            src_hist_len=L, dest_hist_len=L, **approaches["metric"])
            if te_y_to_x > 0 and (te_y_to_x - te_x_to_y) > 0:
                correct_metric += 1

            # ----------------- IDTxl Kraskov TE ----------------------
            jidt_settings = {"history_target": L, "history": L}
            est = JidtKraskovTE(jidt_settings)
            te12 = est.estimate(x[0], y[0])
            te21 = est.estimate(y[0], x[0])
            if te21 > 0 and (te21 - te12) > 0:
                correct_idtxl += 1

            # ----------------- ARTG baselines  -----------------------
            OBJECTIVE = "score_matching"
            FIT_METHOD = "adam"
            N_ITER = 500
            train_prior = ARTGPriorParams() if use_prior else None
            te_mat = estimate_all_pairwise_transfer_entropies(
                key3, samples, samples, L=L, prior=train_prior,
                objective=OBJECTIVE, fit_method=FIT_METHOD,
                batch_size=min(64, T//2), n_iter=N_ITER,
            )
            if te_mat[1, 0] > 0 and (te_mat[1, 0] - te_mat[0, 1]) > 0:
                correct_torus_sm += 1
            
            OBJECTIVE = "mle"
            FIT_METHOD = "bfgs"
            te_mat = estimate_all_pairwise_transfer_entropies(
                key4, samples, samples, L=L, prior=train_prior,
                objective=OBJECTIVE, fit_method=FIT_METHOD,
                batch_size=min(64, T//2), n_iter=N_ITER
            )
            if te_mat[1, 0] > 0 and (te_mat[1, 0] - te_mat[0, 1]) > 0:
                correct_torus_mle += 1

            # ----------------- Granger -------------------------------
            # statsmodels expects columns [x, y]. Using F-test at lag L.
            data_xy = np.stack([x[0], y[0]]).T
            data_yx = np.stack([y[0], x[0]]).T
            g21 = grangercausalitytests(data_xy, L, verbose=False)[L][0]["params_ftest"][0]
            g12 = grangercausalitytests(data_yx, L, verbose=False)[L][0]["params_ftest"][0]
            if g21 > 0 and (g21 - g12) > 0:
                correct_granger += 1

        # Percent correct, guarding divisions like the original
        pct = lambda corr: (corr / max(num_seeds, 1)) * 100.0
        results_json[T] = {
            "correct_direction_torus":          pct(correct_torus_sm),
            "correct_direction_torus_mle":      pct(correct_torus_mle),
            "correct_direction_idt_c":          pct(correct_idtxl),
            "correct_direction_kernel_c":       pct(correct_kernel),
            "correct_direction_metric_c":       pct(correct_metric),
            "correct_direction_granger":        pct(correct_granger),
        }

        # Save incrementally.
        with open(out_json, "w") as f:
            json.dump(results_json, f, indent=4)


def _parse_int_list(s: str) -> List[int]:
    """Parse '35,50,65' into [35,50,65]."""
    return [int(x) for x in s.split(",") if x.strip()]


def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bivariate TE comparison (accuracy vs samples)."
    )
    parser.add_argument(
        "--samples",
        type=_parse_int_list,
        default="35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260,275,290",
        help="Comma-separated list of T values (e.g., '64,96,128').",
    )
    parser.add_argument("--num-seeds", type=int, default=150, help="Trials per T.")
    parser.add_argument("--seed-multiplier", type=int, default=17, help="Seed multiplier (seed = m*i).")
    parser.add_argument("--strength", type=float, default=0.1, help="Causal intensity.")
    args = parser.parse_args()

    run_bvte_exp(
        sample_sizes=args.samples if isinstance(args.samples, list) else _parse_int_list(args.samples),
        num_seeds=args.num_seeds,
        seed_multiplier=args.seed_multiplier,
        strength=args.strength,
    )


if __name__ == "__main__":
    _run_cli()
