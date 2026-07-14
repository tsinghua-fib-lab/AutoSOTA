"""
Autoregressive Torus Graph example

"""
__date__ = "May - July 2025"

import argparse
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


def _run_cli():
    parser = argparse.ArgumentParser(
        description="Autoregressive Torus Graph example: estimate pairwise TE on synthetic data."
    )
    parser.add_argument("--L", type=int, default=10, help="History length.")
    parser.add_argument("--T", type=int, default=800, help="Train length.")
    parser.add_argument("--T-test", type=int, default=800, dest="T_test",
                        help="Test length.")
    parser.add_argument("--D", type=int, default=2, help="Dimensionality.")
    parser.add_argument("--use-prior", action="store_true",
                        help="Use ARTG prior during training.")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument(
        "--objective",
        choices=["mle", "score_matching"],
        default="mle",
        help="Training objective.",
    )
    parser.add_argument(
        "--fit-method",
        choices=["bfgs", "adam"],
        default="bfgs",
        help="Optimizer / fitting method.",
    )
    parser.add_argument(
        "--entropy-method",
        choices=["log_prob", "vm_entropy"],
        default="log_prob",
        help="Entropy estimator used inside TE.",
    )
    parser.add_argument("--n-iter", type=int, default=500,
                        dest="n_iter", help="Number of optimization iterations.")

    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    key1, key2, key3 = jax.random.split(key, 3)

    prior = ARTGPriorParams(d=args.D, allow_12=False)
    params = sample_artg_params(key1, prior, args.L)
    samples = sample_artg(key2, args.T + args.T_test, params)
    samples, samples_test = samples[:, :args.T], samples[:, args.T:]  # (D,T), (D,T_test)

    train_prior = ARTGPriorParams() if args.use_prior else None

    te = estimate_all_pairwise_transfer_entropies(
        key3,
        samples,
        samples_test,
        L=args.L,
        prior=train_prior,
        objective=args.objective,
        fit_method=args.fit_method,
        entropy_method=args.entropy_method,
        n_iter=args.n_iter,
    )
    print("Estimated pairwise TE:")
    print(te)

    if args.D == 2:
        te_gt = estimate_transfer_entropies(
            key3,
            samples,
            samples_test,
            bivariate_params=params,
            L=args.L,
            prior=train_prior,
            objective=args.objective,
            fit_method=args.fit_method,
            entropy_method=args.entropy_method,
            n_iter=args.n_iter,
        )
        print("Ground Truth:", te_gt)


if __name__ == "__main__":
    _run_cli()