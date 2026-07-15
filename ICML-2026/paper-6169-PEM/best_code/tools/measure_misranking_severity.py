#!/usr/bin/env python3
"""
Measure "misranking severity" by comparing ranks under two noisy draws on the same
candidate set.

Outputs per-problem and overall summaries of:
- rank_disagreement: mean |rank1-rank2| / lambda
- topmu_overlap: |Topμ1 ∩ Topμ2| / μ

This is useful for calibrating synthetic noise wrappers against bbob-noisy.
"""

import argparse
import csv
import os
from collections import defaultdict

import cocoex
import numpy as np

from _project import BASE_DIR, repo_relpath

try:
    import cma  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cma = None


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def format_filter(dims: list[int], funcs: list[int], instances: list[int]) -> str:
    dims_s = ",".join(str(d) for d in dims)
    funcs_s = ",".join(str(f) for f in funcs)
    inst_s = ",".join(str(i) for i in instances)
    return f"dimensions:{dims_s} function_indices:{funcs_s} instance_indices:{inst_s}"


def rank_disagreement(f_a: np.ndarray, f_b: np.ndarray) -> float:
    lam = int(f_a.size)
    if lam <= 1:
        return 0.0
    order_a = np.argsort(f_a)
    order_b = np.argsort(f_b)
    ranks_a = np.empty(lam, dtype=int)
    ranks_b = np.empty(lam, dtype=int)
    ranks_a[order_a] = np.arange(lam)
    ranks_b[order_b] = np.arange(lam)
    return float(np.mean(np.abs(ranks_a - ranks_b)) / float(lam))


def topmu_overlap(f_a: np.ndarray, f_b: np.ndarray, mu: int) -> float:
    lam = int(f_a.size)
    mu = int(max(1, min(mu, lam)))
    top_a = set(np.argsort(f_a)[:mu].tolist())
    top_b = set(np.argsort(f_b)[:mu].tolist())
    return float(len(top_a.intersection(top_b)) / float(mu))


def kendall_pairwise_disagreement(f_a: np.ndarray, f_b: np.ndarray) -> float:
    """
    Fraction of discordant pairs between two induced rankings (Kendall tau distance / C(lam,2)).

    NOTE: This assumes (or approximates) no ties; under ties, the metric treats tied pairs as
    neither concordant nor discordant (which slightly weakens the theoretical sandwich).
    """

    lam = int(f_a.size)
    if lam <= 1:
        return 0.0
    order_a = np.argsort(f_a)
    order_b = np.argsort(f_b)
    ranks_a = np.empty(lam, dtype=int)
    ranks_b = np.empty(lam, dtype=int)
    ranks_a[order_a] = np.arange(lam)
    ranks_b[order_b] = np.arange(lam)

    discordant = 0
    total = lam * (lam - 1) // 2
    for i in range(lam):
        for j in range(i + 1, lam):
            da = int(ranks_a[i]) - int(ranks_a[j])
            db = int(ranks_b[i]) - int(ranks_b[j])
            if da == 0 or db == 0:
                continue
            if da * db < 0:
                discordant += 1
    return float(discordant) / float(total)


def sample_points(
    rng: np.random.RandomState,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    lam: int,
    sampling: str,
    sigma_x: float,
) -> np.ndarray:
    lam = int(lam)
    dim = int(lower.size)
    if sampling == "gaussian":
        width = np.maximum(upper - lower, 1e-12)
        scale = float(max(0.0, sigma_x)) * width
        xs = center[None, :] + rng.randn(lam, dim) * scale[None, :]
        return np.clip(xs, lower, upper)
    return rng.uniform(lower, upper, size=(lam, dim))


def apply_noise(
    rng: np.random.RandomState,
    f_true: np.ndarray,
    *,
    noise_model: str,
    noise_sigma: float,
) -> np.ndarray:
    s = float(noise_sigma)
    if s <= 0.0:
        return f_true.astype(float, copy=True)
    z = rng.randn(f_true.size).astype(float, copy=False)
    if noise_model == "additive":
        return f_true + s * z
    if noise_model == "additive_rel":
        return f_true + s * (1.0 + np.abs(f_true)) * z
    if noise_model == "additive_rel_t":
        # Student-t heavy-tailed relative noise (df=3) rescaled to unit variance.
        df = 3.0
        t = rng.standard_t(df, size=f_true.size).astype(float, copy=False)
        t = t / np.sqrt(df / max(1e-12, (df - 2.0)))
        return f_true + s * (1.0 + np.abs(f_true)) * t
    if noise_model == "lognormal_mult":
        factor = np.exp(s * z - 0.5 * (s * s))
        return f_true * factor
    raise ValueError(f"Unknown noise_model: {noise_model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=True,
        choices=["bbob-noisy", "bbob-largescale"],
        help="COCO suite to evaluate.",
    )
    parser.add_argument("--dims", default="40")
    parser.add_argument("--functions", default="1-10")
    parser.add_argument("--instances", default="1")
    parser.add_argument("--lambda", dest="lam", type=int, default=32, help="Candidate set size.")
    parser.add_argument("--mu", type=int, default=8, help="Top-μ size for overlap.")
    parser.add_argument("--num-sets", type=int, default=10, help="Number of candidate sets per problem.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "gaussian", "es"],
        help="How to sample candidate sets (uniform over bounds or gaussian around initial_solution).",
    )
    parser.add_argument(
        "--sigma-x",
        type=float,
        default=0.1,
        help="Gaussian sampling std as a fraction of (upper-lower) per coordinate (used when --sampling gaussian).",
    )
    parser.add_argument(
        "--noise-model",
        default="lognormal_mult",
        choices=["additive", "additive_rel", "additive_rel_t", "lognormal_mult"],
        help="Only used for bbob-largescale (synthetic wrapper).",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.2,
        help="Only used for bbob-largescale (synthetic wrapper).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path (default: Results/misranking_<suite>.csv).",
    )
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    funcs = parse_int_list(args.functions)
    inst = parse_int_list(args.instances)
    suite_filter = format_filter(dims, funcs, inst)

    suite = cocoex.Suite(str(args.suite), "", suite_filter)
    rng = np.random.RandomState(int(args.seed) & 0xFFFFFFFF)

    out_rows = []

    for problem in suite:
        dim = int(problem.dimension)
        lower = np.asarray(getattr(problem, "lower_bounds", -5.0 * np.ones(dim)), dtype=float)
        upper = np.asarray(getattr(problem, "upper_bounds", 5.0 * np.ones(dim)), dtype=float)
        center = np.asarray(getattr(problem, "initial_solution", np.zeros(dim)), dtype=float)

        sampling = str(args.sampling)
        if sampling == "es":
            if cma is None:
                raise SystemExit("cma is required for --sampling es")

            lam = int(args.lam)
            width = np.maximum(upper - lower, 1e-12)
            sigma0 = 0.3 * float(np.min(width))
            seed_es = int(rng.randint(0, 2**32 - 1))
            opts = {
                "bounds": [lower, upper],
                "seed": seed_es,
                "verbose": -9,
                "verb_log": 0,
                "verb_time": 0,
                "CMA_diagonal": True,
                "popsize": lam,
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": int(1e9),
                "tolxstagnation": False,
                "tolflatfitness": int(1e9),
            }
            es = cma.CMAEvolutionStrategy(np.clip(center, lower, upper), sigma0, opts)

            # Treat `--num-sets` as number of generations / candidate sets.
            for gen_id in range(int(args.num_sets)):
                xs = np.asarray(es.ask(), dtype=float)
                if xs.ndim != 2 or xs.shape[1] != dim:
                    xs = np.asarray(xs, dtype=float).reshape((lam, dim))
                xs = np.clip(xs, lower, upper)

                if args.suite == "bbob-largescale":
                    f_true = np.array([float(problem(x)) for x in xs], dtype=float)
                    es.tell(xs.tolist(), f_true.tolist())
                    rng1 = np.random.RandomState(rng.randint(0, 2**32 - 1))
                    rng2 = np.random.RandomState(rng.randint(0, 2**32 - 1))
                    f1 = apply_noise(
                        rng1,
                        f_true,
                        noise_model=args.noise_model,
                        noise_sigma=float(args.noise_sigma),
                    )
                    f2 = apply_noise(
                        rng2,
                        f_true,
                        noise_model=args.noise_model,
                        noise_sigma=float(args.noise_sigma),
                    )
                else:
                    f1 = np.array([float(problem(x)) for x in xs], dtype=float)
                    es.tell(xs.tolist(), f1.tolist())
                    f2 = np.array([float(problem(x)) for x in xs], dtype=float)

                rd = rank_disagreement(f1, f2)
                ov = topmu_overlap(f1, f2, int(args.mu))
                kd = kendall_pairwise_disagreement(f1, f2)
                out_rows.append(
                    {
                        "suite": str(args.suite),
                        "sampling": "es",
                        "noise_model": str(args.noise_model) if args.suite == "bbob-largescale" else "native",
                        "noise_sigma": float(args.noise_sigma) if args.suite == "bbob-largescale" else float("nan"),
                        "function": int(problem.id_function),
                        "dimension": dim,
                        "instance": int(problem.id_instance),
                        "set_id": int(gen_id),
                        "lambda": int(args.lam),
                        "mu": int(args.mu),
                        "rank_disagreement": float(rd),
                        "topmu_overlap": float(ov),
                        "topmu_disagreement": float(1.0 - ov),
                        "kendall_pairwise_disagreement": float(kd),
                    }
                )
            continue

        for set_id in range(int(args.num_sets)):

            xs = sample_points(
                rng,
                lower=lower,
                upper=upper,
                center=center,
                lam=int(args.lam),
                sampling=sampling,
                sigma_x=float(args.sigma_x),
            )

            if args.suite == "bbob-largescale":
                f_true = np.array([float(problem(x)) for x in xs], dtype=float)
                rng1 = np.random.RandomState(rng.randint(0, 2**32 - 1))
                rng2 = np.random.RandomState(rng.randint(0, 2**32 - 1))
                f1 = apply_noise(rng1, f_true, noise_model=args.noise_model, noise_sigma=float(args.noise_sigma))
                f2 = apply_noise(rng2, f_true, noise_model=args.noise_model, noise_sigma=float(args.noise_sigma))
            else:
                f1 = np.array([float(problem(x)) for x in xs], dtype=float)
                f2 = np.array([float(problem(x)) for x in xs], dtype=float)

            rd = rank_disagreement(f1, f2)
            ov = topmu_overlap(f1, f2, int(args.mu))
            kd = kendall_pairwise_disagreement(f1, f2)
            out_rows.append(
                {
                    "suite": str(args.suite),
                    "sampling": sampling,
                    "noise_model": str(args.noise_model) if args.suite == "bbob-largescale" else "native",
                    "noise_sigma": float(args.noise_sigma) if args.suite == "bbob-largescale" else float("nan"),
                    "function": int(problem.id_function),
                    "dimension": dim,
                    "instance": int(problem.id_instance),
                    "set_id": int(set_id),
                    "lambda": int(args.lam),
                    "mu": int(args.mu),
                    "rank_disagreement": float(rd),
                    "topmu_overlap": float(ov),
                    "topmu_disagreement": float(1.0 - ov),
                    "kendall_pairwise_disagreement": float(kd),
                }
            )

    out_path = str(args.output_csv).strip()
    if not out_path:
        out_dir = os.path.join(BASE_DIR, "Results")
        os.makedirs(out_dir, exist_ok=True)
        tag = str(args.suite).replace("-", "_")
        if args.suite == "bbob-largescale":
            tag += f"_{args.noise_model}_sigma{str(args.noise_sigma).replace('.', 'p')}"
        out_path = os.path.join(out_dir, f"misranking_{tag}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)

    # Print quick overall summary.
    rds = [float(r["rank_disagreement"]) for r in out_rows]
    ovs = [float(r["topmu_overlap"]) for r in out_rows]
    kds = [float(r.get("kendall_pairwise_disagreement", float("nan"))) for r in out_rows]
    print("Wrote:", repo_relpath(out_path))
    if rds:
        print("rank_disagreement mean:", float(np.mean(rds)), "median:", float(np.median(rds)))
        print("topmu_overlap mean:", float(np.mean(ovs)), "median:", float(np.median(ovs)))
        if all(np.isfinite(k) for k in kds):
            print(
                "kendall_pairwise_disagreement mean:",
                float(np.mean(kds)),
                "median:",
                float(np.median(kds)),
            )


if __name__ == "__main__":
    main()
