import time
import numpy as np

# Import from local solvers
from iamp_sk_solver import *
from mip_crim import *


def make_sk_matrix(n, seed=0):
    """GOE(n): J_ij ~ N(0,1), J_ii = 0, J symmetric."""
    rng = np.random.default_rng(seed)
    # ensure J_ij ~ N(0,1/n) with zero diagonal and symmetry
    W   = rng.standard_normal((n, n)) 
    J = (W + W.T) / 2
    np.fill_diagonal(J, 0)  # zero out diagonal for SK model
    J = np.round(J, decimals=5)  # lsb ~ 1e-5, so gamma_0 = 1e-5
    return J


def make_sk_goe_matrix(n, seed=0):
    """GOE(n): J_ij ~ N(0,1/n), J_ii ~ N(0,2/n), J symmetric."""
    rng = np.random.default_rng(seed)
    # ensure J_ij ~ N(0,1/n) with zero diagonal and symmetry
    W   = rng.standard_normal((n, n)) / np.sqrt(n)
    J   = np.triu(W, 1)
    J   = J + J.T
    # ensure J_ii ~ N(0,2/n) to match the original GOE definition (and Parisi PDE scaling)
    J = J + np.diag(2 * rng.standard_normal((n)) / np.sqrt(n))
    J = np.round(J, decimals=5)  # lsb ~ 1e-5, so gamma_0 = 1e-5
    return J


def run_mip_crim(J, seed=0):
    """Run MiP-CRIM with best tuned parameters for SK model."""
    n = J.shape[0]
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)

    # Best tuned parameters for the SK Model
    params = dict(
        T=10, K=200, 
        alpha=0.000014996, beta=0.001, lambda_=0.0707, 
        # gamma_0 = lsb = 1e-5 satisfies: 3*beta(lambda)^2 < alpha < beta(lambda)^2 + gamma_0
        step=1.00, beta1=0.09, beta2=0.999, eps=1e-8, 
        sigma_noise=1e-3
    )

    J_mat = J.copy()  # ensure we don't modify the original J
    np.fill_diagonal(J_mat, 0)  # zero out diagonal for MiP-CRIM

    t0 = time.perf_counter()
    sigma = MiP_CRIM(J_mat, x0, rng=rng, **params)
    elapsed = time.perf_counter() - t0

    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)
    return dict(energy=energy, sync=sync, time=elapsed)

def run_iamp(J, beta=6.0, delta=0.02, n_restarts=15, pde_cache=None, seed=0):
    """Run IAMP (Montanari 2019) with caching for Parisi PDE components."""
    if pde_cache is None:
        pde_cache = {}

    t0 = time.perf_counter()

    if beta not in pde_cache:
        q_star = estimate_q_star(beta)
        pde = ParisiPDE(beta, q_star)
        pde_cache[beta] = (q_star, pde)

    q_star, pde = pde_cache[beta]

    res = iamp_solve(J, beta=beta, delta=delta, n_restarts=n_restarts,
                     q_star=q_star, pde=pde, seed=seed)
    
    elapsed = time.perf_counter() - t0

    sigma = res["sigma"]
    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)

    return dict(energy=energy, sync=sync, time=elapsed, pde_cache=pde_cache)

def main():
    spins = [100, 200, 500, 1000]
    n_trials = 100

    print("=" * 80)
    print(f"  Benchmarking: SK Model (IAMP vs MiP-CRIM) across {n_trials} trials")
    print("=" * 80)

    final_results = []

    for matrix_type, make_matrix in [("GOE", make_sk_goe_matrix), ("Standard", make_sk_matrix)]:
        print(f"\n{'='*80}")
        print(f"  Matrix Type: {matrix_type}")
        print(f"{'='*80}")
        
        for n in spins:
            print(f"\nEvaluating SK ({matrix_type}) Model with n={n} spins...")
            iamp_energies, iamp_syncs, iamp_times = [], [], []
            mip_energies, mip_syncs, mip_times = [], [], []
            
            pde_cache = {}

            for trial in range(n_trials):
                print(f"  Running trial {trial+1}/{n_trials}...", end="\r", flush=True)
                seed = trial * 137 + 42
                J = make_matrix(n, seed=seed)

                # IAMP
                ri = run_iamp(J, pde_cache=pde_cache, seed=seed)
                iamp_energies.append(ri["energy"])
                iamp_syncs.append(ri["sync"])
                iamp_times.append(ri["time"])

                # MiP-CRIM
                rm = run_mip_crim(J, seed=seed)
                mip_energies.append(rm["energy"])
                mip_syncs.append(rm["sync"])
                mip_times.append(rm["time"])

            print(" " * 40, end="\r") # clear the line
            
            # Calculate Means and Standard Deviations
            ie_mean = np.mean(iamp_energies)
            me_mean = np.mean(mip_energies)

            is_mean = np.mean(iamp_syncs)
            ms_mean = np.mean(mip_syncs)

            it_mean = np.mean(iamp_times)
            mt_mean = np.mean(mip_times)

            # Accumulate for final table (Note: for energy = -0.5*s^T*J*s, lower is better)
            ie_best = np.min(iamp_energies)
            me_best = np.min(mip_energies)
            
            is_best = np.max(iamp_syncs)
            ms_best = np.max(mip_syncs)
            
            final_results.append({
                "spin": n, "type": matrix_type, "method": "IAMP",
                "e_avg": ie_mean, "e_best": ie_best,
                "s_avg": is_mean, "s_best": is_best,
                "t_avg": it_mean
            })
            final_results.append({
                "spin": n, "type": matrix_type, "method": "MiP-CRIM",
                "e_avg": me_mean, "e_best": me_best,
                "s_avg": ms_mean, "s_best": ms_best,
                "t_avg": mt_mean
            })

    print("\n" + "=" * 105)
    print("  FINAL SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Type':<10} | {'Spins':<7} | {'Method':<10} | {'Avg Energy':<12} | {'Best Energy':<12} | {'Avg Sync':<10} | {'Best Sync':<10} | {'Avg Time (s)':<12} |")
    print("-" * 105)
    for res in final_results:
        print(f"{res['type']:<10} | {res['spin']:<7} | {res['method']:<10} | {res['e_avg']:>12.2f} | {res['e_best']:>12.2f} | {res['s_avg']:>10.3f} | {res['s_best']:>10.3f} | {res['t_avg']:>12.3f} |")
        if res['method'] == "MiP-CRIM":
            print("-" * 105)

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
