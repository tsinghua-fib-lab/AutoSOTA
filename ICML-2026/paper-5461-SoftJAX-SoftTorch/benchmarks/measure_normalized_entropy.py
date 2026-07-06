"""Measure normalized entropy of soft permutation matrices across methods, modes, and sizes.

For each configuration, computes argsort to get the soft permutation matrix P,
then measures the mean normalized row entropy: mean(-sum(P * log(P))) / log(n).

Range: 0 (hard/one-hot) to 1 (uniform).

Usage:
    uv run python benchmarks/measure_normalized_entropy.py
    uv run python benchmarks/measure_normalized_entropy.py --methods softsort,neuralsort --modes smooth,c0
    uv run python benchmarks/measure_normalized_entropy.py --sizes 32,128,512 --softness 0.1,1.0,10.0
"""

import argparse
import itertools

import jax
import jax.numpy as jnp

import softjax


def normalized_entropy(P: jax.Array) -> float:
    """Mean normalized row entropy H(P)/log(n). Range: 0 (one-hot) to 1 (uniform)."""
    n = P.shape[-1]
    P_safe = jnp.clip(P, 1e-30, 1.0)
    row_entropy = -jnp.sum(P * jnp.log(P_safe), axis=-1)
    return float(jnp.mean(row_entropy) / jnp.log(n))


def measure(
    methods: list[str],
    modes: list[str],
    sizes: list[int],
    softness_values: list[float],
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Run measurements across all configurations."""
    key = jax.random.PRNGKey(seed)
    results = []

    for n, softness in itertools.product(sizes, softness_values):
        # Generate random inputs once per (n, softness)
        x = jax.random.normal(key, (n_samples, n))

        for method, mode in itertools.product(methods, modes):
            try:
                P = softjax.argsort(
                    x,
                    axis=-1,
                    softness=jnp.array(softness),
                    mode=mode,
                    method=method,
                )  # (n_samples, n, n)
                ne = normalized_entropy(P)
            except (NotImplementedError, ValueError):
                ne = float("nan")

            row = {
                "method": method,
                "mode": mode,
                "n": n,
                "softness": softness,
                "normalized_entropy": ne,
            }
            results.append(row)
            status = f"{ne:.4f}" if not jnp.isnan(ne) else "N/A"
            print(
                f"  method={method:<16s} mode={mode:<7s} n={n:<5d} "
                f"softness={softness:<8.2f} H_norm={status}"
            )

    return results


def print_table(results: list[dict], softness_values: list[float]) -> None:
    """Print results as a readable table, one per softness value."""
    methods = sorted({r["method"] for r in results})
    modes = sorted({r["mode"] for r in results})
    sizes = sorted({r["n"] for r in results})

    for softness in softness_values:
        print(f"\n{'=' * 80}")
        print(f"softness = {softness}")
        print(f"{'=' * 80}")

        # Header
        col_headers = [f"{m}/{mo}" for m, mo in itertools.product(methods, modes)]
        header = f"{'n':>6}" + "".join(f"{h:>18}" for h in col_headers)
        print(header)
        print("-" * len(header))

        for n in sizes:
            row_str = f"{n:>6}"
            for method in methods:
                for mode in modes:
                    match = [
                        r
                        for r in results
                        if r["method"] == method
                        and r["mode"] == mode
                        and r["n"] == n
                        and r["softness"] == softness
                    ]
                    if match and not jnp.isnan(match[0]["normalized_entropy"]):
                        row_str += f"{match[0]['normalized_entropy']:>18.4f}"
                    else:
                        row_str += f"{'N/A':>18}"
            print(row_str)


def print_compact_table(results: list[dict], softness_values: list[float]) -> None:
    """Print a compact table: rows = (n, mode), columns = methods, one table per softness."""
    methods = sorted({r["method"] for r in results})
    modes = sorted({r["mode"] for r in results})
    sizes = sorted({r["n"] for r in results})

    for softness in softness_values:
        print(f"\nsoftness = {softness}")
        header = f"{'n':>6} {'mode':>7}" + "".join(f"{m:>16}" for m in methods)
        print(header)
        print("-" * len(header))

        for n in sizes:
            for mode in modes:
                row_str = f"{n:>6} {mode:>7}"
                for method in methods:
                    match = [
                        r
                        for r in results
                        if r["method"] == method
                        and r["mode"] == mode
                        and r["n"] == n
                        and r["softness"] == softness
                    ]
                    if match and not jnp.isnan(match[0]["normalized_entropy"]):
                        row_str += f"{match[0]['normalized_entropy']:>16.4f}"
                    else:
                        row_str += f"{'N/A':>16}"
                print(row_str)
            if n != sizes[-1]:
                print()


def main():
    parser = argparse.ArgumentParser(description="Measure normalized entropy of soft permutation matrices")
    parser.add_argument(
        "--methods",
        type=str,
        default="softsort,neuralsort,sorting_network,ot",
        help="Comma-separated methods (default: softsort,neuralsort,sorting_network,ot)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="smooth,c0,c1,c2",
        help="Comma-separated modes (default: smooth,c0,c1,c2)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="16,32,64,128,256",
        help="Comma-separated problem sizes (default: 16,32,64,128,256)",
    )
    parser.add_argument(
        "--softness",
        type=str,
        default="0.1",
        help="Comma-separated softness values (default: 0.1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random vectors to average over (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-x64",
        action="store_true",
        default=False,
        help="Disable float64 (default: x64 is enabled)",
    )
    args = parser.parse_args()

    if not args.no_x64:
        jax.config.update("jax_enable_x64", True)

    methods = [m.strip() for m in args.methods.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    softness_values = [float(s.strip()) for s in args.softness.split(",")]

    print(f"Methods: {methods}")
    print(f"Modes: {modes}")
    print(f"Sizes: {sizes}")
    print(f"Softness: {softness_values}")
    print(f"Samples: {args.n_samples}, Seed: {args.seed}")
    print()

    results = measure(methods, modes, sizes, softness_values, args.n_samples, args.seed)

    print_compact_table(results, softness_values)


if __name__ == "__main__":
    main()
