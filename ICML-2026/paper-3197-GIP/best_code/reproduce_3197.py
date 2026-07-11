#!/usr/bin/env python3
"""Reproduction script for paper 3197: Gaussian CBO on Targets A, B, C, D.

Usage:
  python3 reproduce_3197.py                     # Target A (default)
  python3 reproduce_3197.py --target A|B|C|D     # Specific target
  python3 reproduce_3197.py --target all          # All targets (summary)

Targets (from paper Table 1):
  A: 2-component bimodal GMM, d=2
  B: 4-component GMM, d=2 (quadrants)
  C: 9-component GMM, d=2 (3x3 grid)
  D: random GMM, d=10

Parameters: dt=0.05, lambda=1, sigma=5, alpha=10000, N=20 particles
T=10, 100 runs (fewer for d=10)
"""
import os, sys, time, json, argparse
import numpy as np

sys.path.insert(0, "/repo")

from gmm_target import GMM2D, GMM4, GMMNd, random_gmm_nd
from sigma_points import expect_scalar, unscented_points
from cbo import LBWCBO, CBOConfig, GaussianParticle
from geometry import _sym


# ── Target definitions ─────────────────────────────────────────────────
def build_target_a():
    """Target A: 2-component bimodal GMM, d=2 (paper Table 1)."""
    return GMM2D(
        w1=0.5,
        m1=(-2.2, 0.0), S1=(1.0, 0.2, 0.2, 0.6),
        m2=( 2.2, 0.0), S2=(1.0,-0.2,-0.2, 0.6),
    )


def build_target_b():
    """Target B: 4-component GMM, d=2 (quadrants, paper Table 1)."""
    return GMM4(
        w1=0.25, w2=0.25, w3=0.25,
        m1=(-2.0, -2.0), S1=(1.0, 0.0, 0.0, 1.0),
        m2=( 2.0, -2.0), S2=(1.0, 0.0, 0.0, 1.0),
        m3=(-2.0,  2.0), S3=(1.0, 0.0, 0.0, 1.0),
        m4=( 2.0,  2.0), S4=(1.0, 0.0, 0.0, 1.0),
    )


def build_target_c():
    """Target C: 9-component GMM, d=2 (3x3 grid, paper Table 1)."""
    # 3x3 grid centered at origin, spacing ~2.5
    positions = np.array([
        [-2.5, -2.5], [0.0, -2.5], [2.5, -2.5],
        [-2.5,  0.0], [0.0,  0.0], [2.5,  0.0],
        [-2.5,  2.5], [0.0,  2.5], [2.5,  2.5],
    ])
    K = 9
    d = 2
    w = np.ones(K) / K
    m = positions.astype(float)
    S = np.stack([0.5 * np.eye(d) for _ in range(K)], axis=0)
    return GMMNd(w=w, m=m, S=S)


def build_target_d(seed=42):
    """Target D: random K-component GMM, d=10 (paper Table 1)."""
    rng = np.random.default_rng(seed)
    d = 10
    K = 6  # reasonable number of components for d=10
    return random_gmm_nd(d=d, K=K, rng=rng, mean_radius=3.0)


TARGETS = {
    'A': build_target_a,
    'B': build_target_b,
    'C': build_target_c,
    'D': build_target_d,
}


# ── Single-target evaluation ───────────────────────────────────────────
def _init_particles_standard(rng, d, n_particles):
    """Standard initialization: all particles near a single random mean."""
    m0 = rng.uniform(-5.0, 5.0, size=d)
    particles = []
    for _ in range(n_particles):
        m_i = m0 + 0.1 * rng.normal(size=d)
        T_i = 0.1 * _sym(rng.normal(size=(d, d)))
        particles.append(GaussianParticle(m_i, T_i))
    return particles


def _init_particles_stratified(rng, d, n_particles):
    """Stratified initialization: particle means spread via Latin hypercube."""
    from scipy.stats import qmc
    # Generate n_particles initial means spread across [-5,5]^d
    sampler = qmc.LatinHypercube(d=d, seed=rng.integers(0, 2**31))
    sample = sampler.random(n=n_particles)
    means = sample * 10.0 - 5.0  # map [0,1]^d -> [-5,5]^d

    particles = []
    for i in range(n_particles):
        m_i = means[i] + 0.1 * rng.normal(size=d)
        T_i = 0.1 * _sym(rng.normal(size=(d, d)))
        particles.append(GaussianParticle(m_i.copy(), T_i))
    return particles


def evaluate_target(target_name, gmm, n_runs=100, n_particles=20,
                    t_final=10.0, dt=0.05, sigma=5.0, alpha=1e4,
                    lmbda=1.0, seed_base=0, verbose=True,
                    cbo_config_overrides=None, init_mode="standard",
                    sigma_decay=0.0, alpha_growth=0.0,
                    update_base=0, diagonal=False):
    """Evaluate CBO on one target, return results dict."""
    steps = int(np.floor(t_final / dt))
    d = gmm.d if hasattr(gmm, 'd') else 2

    def KL(m, S):
        return gmm.KL_q_to_p(m, S, expect_scalar)

    KL_runs = np.full((n_runs, steps + 1), np.nan)
    t_start = time.time()

    init_fn = _init_particles_stratified if init_mode == "stratified" else _init_particles_standard

    for run_idx in range(n_runs):
        rng = np.random.default_rng(seed_base + run_idx)
        particles = init_fn(rng, d, n_particles)

        n_update = 2000 if update_base <= 0 else update_base
        cbo_cfg = CBOConfig(
            alpha=alpha, sigma=sigma, lmbda=lmbda, dt=dt,
            seed=seed_base + run_idx,
            n_update_base=n_update,
            sigma_decay=sigma_decay,
            alpha_growth=alpha_growth,
            diagonal=diagonal,
        )
        if cbo_config_overrides:
            for k, v in cbo_config_overrides.items():
                setattr(cbo_cfg, k, v)

        cbo = LBWCBO(lambda m, S: KL(m, S), cbo_cfg)
        traj_cbo, _ = cbo.run(particles, steps=steps)

        for t_idx, (m_bar, S_bar) in enumerate(traj_cbo):
            KL_runs[run_idx, t_idx] = KL(m_bar, S_bar)

        if verbose and (run_idx + 1) % 20 == 0:
            median_sofar = np.nanmedian(KL_runs[:run_idx+1, -1])
            elapsed = time.time() - t_start
            print(f"  [{target_name}] Run {run_idx+1:3d}/{n_runs}  median KL={median_sofar:.4f}  elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    median_final = float(np.nanmedian(KL_runs[:, -1]))
    q25_final = float(np.nanquantile(KL_runs[:, -1], 0.25))
    q75_final = float(np.nanquantile(KL_runs[:, -1], 0.75))
    mean_final = float(np.nanmean(KL_runs[:, -1]))

    nan_count = int(np.sum(np.isnan(KL_runs[:, -1])))

    return {
        "target": f"Target_{target_name}",
        "dim": d,
        "n_runs": n_runs,
        "n_particles": n_particles,
        "sigma": sigma,
        "alpha": alpha,
        "median_KL": median_final,
        "q25_KL": q25_final,
        "q75_KL": q75_final,
        "mean_KL": mean_final,
        "nan_runs": nan_count,
        "elapsed_seconds": elapsed_total,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper 3197 CBO Evaluation")
    parser.add_argument("--target", type=str, default="A",
                       choices=["A", "B", "C", "D", "all"],
                       help="Target distribution (default: A)")
    parser.add_argument("--runs", type=int, default=None,
                       help="Override number of runs")
    parser.add_argument("--sigma", type=float, default=5.0,
                       help="Noise scale sigma")
    parser.add_argument("--alpha", type=float, default=10000.0,
                       help="Inverse temperature alpha")
    parser.add_argument("--particles", type=int, default=20,
                       help="Number of particles N")
    parser.add_argument("--lmbda", type=float, default=1.0,
                       help="Drift strength lambda")
    parser.add_argument("--t-final", type=float, default=10.0,
                       help="Final time T")
    parser.add_argument("--dt", type=float, default=0.05,
                       help="Step size dt")
    parser.add_argument("--init", type=str, default="standard",
                       choices=["standard", "stratified"],
                       help="Particle initialization mode (default: standard)")
    parser.add_argument("--sigma-decay", type=float, default=0.0,
                       help="Sigma annealing decay rate (0=disabled)")
    parser.add_argument("--alpha-growth", type=float, default=0.0,
                       help="Alpha annealing growth rate (0=disabled)")
    parser.add_argument("--update-base", type=int, default=0,
                       help="Reference base update frequency (0=disabled, e.g. 20)")
    parser.add_argument("--diagonal", action="store_true",
                       help="Restrict covariance to diagonal")
    parser.add_argument("--seed", type=int, default=0,
                       help="Base random seed")
    args = parser.parse_args()

    targets_to_run = list(TARGETS.keys()) if args.target == "all" else [args.target]

    # Per-target run counts
    default_runs = {"A": 100, "B": 100, "C": 100, "D": 50}

    all_results = []

    for tname in targets_to_run:
        n_runs = args.runs if args.runs is not None else default_runs[tname]
        gmm = TARGETS[tname]()

        if tname == 'D' and n_runs > 50:
            n_runs = 50  # cap d=10 runs for time

        print(f"\n{'='*70}")
        print(f"Target {tname}: d={gmm.d if hasattr(gmm, 'd') else 2}")
        print(f"  N_runs={n_runs}, N_particles={args.particles}")
        print(f"  sigma={args.sigma}, alpha={args.alpha}, lambda={args.lmbda}")
        print(f"  T={args.t_final}, dt={args.dt}")
        print(f"{'='*70}")

        result = evaluate_target(
            target_name=tname,
            gmm=gmm,
            n_runs=n_runs,
            n_particles=args.particles,
            t_final=args.t_final,
            dt=args.dt,
            sigma=args.sigma,
            alpha=args.alpha,
            lmbda=args.lmbda,
            seed_base=args.seed,
            init_mode=args.init,
            sigma_decay=args.sigma_decay,
            alpha_growth=args.alpha_growth,
            update_base=args.update_base,
            diagonal=args.diagonal,
        )
        all_results.append(result)

        print(f"\n  Target {tname} Results:")
        print(f"    median_KL = {result['median_KL']:.6f}")
        print(f"    IQR       = [{result['q25_KL']:.6f}, {result['q75_KL']:.6f}]")
        print(f"    mean_KL   = {result['mean_KL']:.6f}")
        print(f"    NaN runs  = {result['nan_runs']}")
        print(f"    Time      = {result['elapsed_seconds']:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    if len(all_results) > 1:
        medians = [r['median_KL'] for r in all_results]
        geo_mean = float(np.exp(np.mean(np.log([max(m, 1e-10) for m in medians]))))
        print(f"\n{'='*70}")
        print(f"MULTI-TARGET SUMMARY")
        print(f"{'='*70}")
        for r in all_results:
            print(f"  {r['target']}: median_KL={r['median_KL']:.6f}  (d={r['dim']}, runs={r['n_runs']})")
        print(f"  Geometric mean: {geo_mean:.6f}")
        print(f"  Target A median_KL: {all_results[0]['median_KL']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────
    results_path = "/repo/reproduce_3197_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Metric output for parsing ─────────────────────────────────────
    primary = all_results[0]['median_KL']  # Target A is primary
    print(f"\nMETRIC:median_KL={primary:.6f}")

    # Also print per-target metrics
    for r in all_results:
        tlabel = r['target'].replace('Target_', '')
        print(f"METRIC:median_KL_target{tlabel}={r['median_KL']:.6f}")


if __name__ == "__main__":
    main()
