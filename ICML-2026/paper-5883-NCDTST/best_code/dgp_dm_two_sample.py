# dgp_dm_two_sample.py
# ============================================================
# Dirichlet–Multinomial (DM) two-sample DGP for compositional data
#   counts -> normalize to simplex compositions
#
# Key design goals:
#   - Simple, correct, reproducible DGP with zeros arising naturally
#   - Deterministic/fixed library size N per scenario (recommended)
#   - Optional one-time calibration of N to target a pooled zero fraction (approx.)
#   - Plot utilities to validate DGP quality
#
# Dependencies: numpy, matplotlib
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0) Utilities: validation and helpers
# ============================================================
def _to_1d_float_array(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like.")
    return x

def validate_simplex(mu: np.ndarray, name: str, tol: float = 1e-12) -> np.ndarray:
    """
    Enforce: mu strictly positive and sums to 1 (normalized if needed).
    """
    mu = _to_1d_float_array(mu, name)
    if np.any(~np.isfinite(mu)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(mu <= 0):
        raise ValueError(f"{name} must be strictly positive for a Dirichlet mean (no zeros).")
    s = float(mu.sum())
    if s <= 0:
        raise ValueError(f"{name} sum must be positive.")
    mu = mu / s
    # numerical check
    if not (abs(mu.sum() - 1.0) <= 1e-10):
        raise ValueError(f"{name} normalization failed unexpectedly.")
    return mu

def avg_zero_fraction(counts: np.ndarray) -> float:
    """
    Fraction of zero entries in a count matrix.
    """
    counts = np.asarray(counts)
    return float(np.mean(counts == 0))

def per_sample_zero_fraction(counts: np.ndarray) -> np.ndarray:
    """
    Row-wise fraction of zeros (vector length n_samples).
    """
    counts = np.asarray(counts)
    return np.mean(counts == 0, axis=1)

def normalize_counts_to_compositions(counts: np.ndarray) -> np.ndarray:
    """
    Convert counts to compositions (row sums = 1).
    If N>=1, multinomial rows sum to N so this is safe.
    """
    counts = np.asarray(counts, dtype=float)
    rs = counts.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, 1.0)  # safeguard
    return counts / rs


# ============================================================
# 1) DM sampler
# ============================================================
def dirichlet_multinomial_counts(rng: np.random.Generator,
                                 alpha: np.ndarray,
                                 N: int,
                                 size: int) -> np.ndarray:
    """
    DM sampling:
      pi ~ Dirichlet(alpha)
      x  ~ Multinomial(N, pi)
    Returns counts: (size, d)
    """
    alpha = _to_1d_float_array(alpha, "alpha")
    if np.any(alpha <= 0):
        raise ValueError("alpha must be strictly positive.")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be an integer >= 1.")
    if not (isinstance(size, (int, np.integer)) and size >= 1):
        raise ValueError("size must be an integer >= 1.")

    pis = rng.dirichlet(alpha, size=size)  # (size, d)
    # Multinomial per row
    return np.vstack([rng.multinomial(int(N), p) for p in pis])


# ============================================================
# 2) Scenario and DGP core
# ============================================================
@dataclass(frozen=True)
class DMScenario:
    d: int
    m: int
    n: int
    N: int                  # fixed library size (recommended)
    mu0: np.ndarray         # simplex mean for group 0
    mu1: np.ndarray         # simplex mean for group 1
    kappa_base: float = 25.0
    eta: float = 1.0        # dispersion multiplier for group 1 (alpha1 = kappa_base*eta*mu1)

    def __post_init__(self):
        if not (isinstance(self.d, int) and self.d >= 2):
            raise ValueError("d must be an integer >= 2.")
        if not (isinstance(self.m, int) and self.m >= 2):
            raise ValueError("m must be an integer >= 2.")
        if not (isinstance(self.n, int) and self.n >= 2):
            raise ValueError("n must be an integer >= 2.")
        if not (isinstance(self.N, int) and self.N >= 1):
            raise ValueError("N must be an integer >= 1.")
        if not (np.isfinite(self.kappa_base) and self.kappa_base > 0):
            raise ValueError("kappa_base must be > 0.")
        if not (np.isfinite(self.eta) and self.eta > 0):
            raise ValueError("eta must be > 0.")
        # Validate mu vectors
        mu0 = validate_simplex(self.mu0, "mu0")
        mu1 = validate_simplex(self.mu1, "mu1")
        if mu0.size != self.d or mu1.size != self.d:
            raise ValueError("mu0 and mu1 must be length d.")

def generate_dm_two_sample(sc: DMScenario, seed: int = 0) -> dict:
    """
    Generate one replicate under DMScenario.
    Returns:
      counts0, counts1, comps0, comps1, alpha0, alpha1, pz_empirical, etc.
    """
    rng = np.random.default_rng(seed)
    mu0 = validate_simplex(sc.mu0, "mu0")
    mu1 = validate_simplex(sc.mu1, "mu1")

    alpha0 = float(sc.kappa_base) * mu0
    alpha1 = float(sc.kappa_base) * float(sc.eta) * mu1

    counts0 = dirichlet_multinomial_counts(rng, alpha0, sc.N, sc.m)
    counts1 = dirichlet_multinomial_counts(rng, alpha1, sc.N, sc.n)

    comps0 = normalize_counts_to_compositions(counts0)
    comps1 = normalize_counts_to_compositions(counts1)

    pooled_counts = np.vstack([counts0, counts1])
    pooled_comps = np.vstack([comps0, comps1])

    # Basic sanity
    max_dev0 = float(np.max(np.abs(comps0.sum(axis=1) - 1.0)))
    max_dev1 = float(np.max(np.abs(comps1.sum(axis=1) - 1.0)))

    return {
        "scenario": sc,
        "counts0": counts0,
        "counts1": counts1,
        "comps0": comps0,
        "comps1": comps1,
        "pooled_counts": pooled_counts,
        "pooled_comps": pooled_comps,
        "mu0": mu0,
        "mu1": mu1,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "N": sc.N,
        "pz_empirical": avg_zero_fraction(pooled_counts),
        "max_row_sum_dev_grp0": max_dev0,
        "max_row_sum_dev_grp1": max_dev1,
    }


# ============================================================
# 3) Optional: one-time calibration of N to target pooled zero fraction
#    (Use ONLY for scenario setup; keep N fixed inside simulation loops.)
# ============================================================
def calibrate_N_for_target_pz(
    d: int,
    m: int,
    n: int,
    mu0: np.ndarray,
    mu1: np.ndarray,
    kappa_base: float,
    eta: float,
    target_pz: float,
    seed: int = 123,
    N_lo: int = 5,
    N_hi: int = 5000,
    max_iter: int = 25,
    reps: int = 5,
) -> int:
    """
    Binary-search a library size N such that pooled zero fraction in DM counts
    is approximately target_pz. This is only approximate and depends on (d,m,n,mu,kappa,eta).

    NOTE:
      - This is convenient for matching sparsity across scenarios.
      - For size/power studies, calibrate once per setting and then FIX N.
    """
    target_pz = float(np.clip(target_pz, 0.0, 0.999))
    if target_pz <= 0.0:
        return int(N_hi)
    if target_pz >= 0.999:
        return int(N_lo)

    mu0 = validate_simplex(mu0, "mu0")
    mu1 = validate_simplex(mu1, "mu1")

    alpha0 = float(kappa_base) * mu0
    alpha1 = float(kappa_base) * float(eta) * mu1

    rng = np.random.default_rng(seed)
    lo, hi = int(N_lo), int(N_hi)

    best_N = lo
    best_err = float("inf")

    for _ in range(max_iter):
        mid = (lo + hi) // 2

        pzs = []
        # IMPORTANT: use independent randomness per rep to reduce noise
        for r in range(reps):
            rng_r = np.random.default_rng(seed + 10_000 * (_ + 1) + r)
            c0 = dirichlet_multinomial_counts(rng_r, alpha0, mid, m)
            c1 = dirichlet_multinomial_counts(rng_r, alpha1, mid, n)
            pzs.append(avg_zero_fraction(np.vstack([c0, c1])))

        est = float(np.mean(pzs))
        err = abs(est - target_pz)
        if err < best_err:
            best_err = err
            best_N = mid

        # monotonic trend: as N increases, zeros tend to decrease
        if est > target_pz:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break

    return int(best_N)


# ============================================================
# 4) Plot-based DGP quality checks
# ============================================================
def plot_dgp_quality(out: dict, title_prefix: str = "DGP Quality Check") -> None:
    """
    Produce diagnostic plots for one replicate.
    """
    sc: DMScenario = out["scenario"]
    counts0, counts1 = out["counts0"], out["counts1"]
    comps0, comps1 = out["comps0"], out["comps1"]
    mu0, mu1 = out["mu0"], out["mu1"]

    # --- Summary text block ---
    txt = (
        f"d={sc.d}, m={sc.m}, n={sc.n}, N={sc.N}\n"
        f"kappa_base={sc.kappa_base:.2f}, eta={sc.eta:.2f}\n"
        f"pz_empirical={out['pz_empirical']:.3f}\n"
        f"max row-sum dev: grp0={out['max_row_sum_dev_grp0']:.2e}, grp1={out['max_row_sum_dev_grp1']:.2e}"
    )

    # 1) Sanity: row sums should be 1
    s0 = comps0.sum(axis=1) - 1.0
    s1 = comps1.sum(axis=1) - 1.0
    plt.figure()
    plt.hist(s0, bins=30, alpha=0.6, label="Group 0 (sum-1)")
    plt.hist(s1, bins=30, alpha=0.6, label="Group 1 (sum-1)")
    plt.title(f"{title_prefix}: Row-sum sanity (compositions)")
    plt.xlabel("row sum - 1")
    plt.ylabel("count")
    plt.legend(loc="best", frameon=True)
    plt.show()

    # 2) True means mu0 vs mu1
    x = np.arange(sc.d)
    width = 0.42
    plt.figure()
    plt.bar(x - width/2, mu0, width=width, label="mu0 (true mean)")
    plt.bar(x + width/2, mu1, width=width, label="mu1 (true mean)")
    plt.title(f"{title_prefix}: True simplex means")
    plt.xlabel("Component index")
    plt.ylabel("Proportion")
    plt.legend(loc="best", frameon=True)
    plt.show()

    # 3) Empirical means vs true means
    emp0 = comps0.mean(axis=0)
    emp1 = comps1.mean(axis=0)
    width = 0.22
    plt.figure()
    plt.bar(x - 1.5*width, mu0,  width=width, label="mu0 (true)")
    plt.bar(x - 0.5*width, mu1, width=width, label="mu1 (true)")
    plt.bar(x + 0.5*width, emp0, width=width, label="Emp mean grp0")
    plt.bar(x + 1.5*width, emp1, width=width, label="Emp mean grp1")
    plt.title(f"{title_prefix}: True vs empirical mean compositions")
    plt.xlabel("Component index")
    plt.ylabel("Proportion")
    plt.legend(loc="best", frameon=True)
    plt.show()

    # 4) Zero fraction per sample (counts)
    zf0 = per_sample_zero_fraction(counts0)
    zf1 = per_sample_zero_fraction(counts1)
    plt.figure()
    plt.boxplot([zf0, zf1], tick_labels=["Group 0", "Group 1"])
    # show text summary
    plt.plot([], [], " ", label=txt)
    plt.legend(loc="best", frameon=True, title="DGP parameters")
    plt.title(f"{title_prefix}: Per-sample zero fraction in DM counts")
    plt.ylabel("Zero fraction (counts)")
    plt.show()

    # 5) Dispersion proxy: component-wise variance of compositions
    var0 = comps0.var(axis=0, ddof=1)
    var1 = comps1.var(axis=0, ddof=1)
    plt.figure()
    plt.plot(x, var0, marker="o", linestyle="-", label="Var grp0")
    plt.plot(x, var1, marker="o", linestyle="--", label="Var grp1")
    plt.title(f"{title_prefix}: Component-wise variance (dispersion proxy)")
    plt.xlabel("Component index")
    plt.ylabel("Variance of proportions")
    plt.legend(loc="best", frameon=True)
    plt.show()

    # 6) Optional: CLR-distance separation quick look (no claims; just a diagnostic)
    #    Use pseudocount to avoid log(0); this is for visualization only.
    eps = 1e-8
    Z = np.vstack([comps0, comps1])
    Zp = Z + eps
    Zp = Zp / np.maximum(Zp.sum(axis=1, keepdims=True), 1e-300)
    logZ = np.log(Zp)
    clr = logZ - logZ.mean(axis=1, keepdims=True)

    # distance to own-group mean in CLR space
    clr0 = clr[:sc.m]
    clr1 = clr[sc.m:]
    m0 = clr0.mean(axis=0, keepdims=True)
    m1 = clr1.mean(axis=0, keepdims=True)
    d0 = np.sqrt(np.sum((clr0 - m0)**2, axis=1))
    d1 = np.sqrt(np.sum((clr1 - m1)**2, axis=1))

    plt.figure()
    plt.boxplot([d0, d1], tick_labels=["Group 0", "Group 1"])
    plt.title(f"{title_prefix}: CLR distance-to-own-mean (diagnostic)")
    plt.ylabel("Euclidean distance in CLR space")
    plt.show()


# ============================================================
# 5) Example usage (run this file directly)
# ============================================================
if __name__ == "__main__":
    # --- Choose a base mu ---
    d = 12
    base = np.linspace(1, d, d)
    mu = base / base.sum()

    # --- Scenario A: Same mean, different dispersion (eta != 1) ---
    # eta < 1 => group1 more dispersed; eta > 1 => group1 less dispersed
    m, n = 120, 120
    kappa_base = 25.0
    eta = 0.5

    # Option 1 (recommended): choose N directly
    N = 200

    # Option 2 (optional): calibrate N once to hit a target pooled zero fraction
    # target_pz = 0.55
    # N = calibrate_N_for_target_pz(d, m, n, mu, mu, kappa_base, eta, target_pz, seed=777)

    sc = DMScenario(d=d, m=m, n=n, N=N, mu0=mu, mu1=mu, kappa_base=kappa_base, eta=eta)
    out = generate_dm_two_sample(sc, seed=7)

    print("==== DGP Sanity ====")
    print(f"sum(mu0)={out['mu0'].sum():.12f}, sum(mu1)={out['mu1'].sum():.12f}")
    print(f"max row-sum dev grp0={out['max_row_sum_dev_grp0']:.3e}")
    print(f"max row-sum dev grp1={out['max_row_sum_dev_grp1']:.3e}")
    print(f"N={out['N']}, pooled zero fraction (emp)={out['pz_empirical']:.3f}")

    plot_dgp_quality(out, title_prefix="Scenario A (same mean, dispersion change)")

    # --- Scenario B: Mean shift + optional dispersion change ---
    # Create a controlled mean shift by moving mass from some components to others.
    mu1 = mu.copy()
    # shift mass: move delta from last 3 components to first 3 (keeps positivity if delta small)
    delta = 0.06
    idx_plus = np.array([0, 1, 2])
    idx_minus = np.array([d-3, d-2, d-1])
    mu1[idx_plus] += delta / len(idx_plus)
    mu1[idx_minus] -= delta / len(idx_minus)
    mu1 = validate_simplex(mu1, "mu1_shifted")

    eta_b = 1.0  # keep dispersion same here; change if desired
    sc_b = DMScenario(d=d, m=m, n=n, N=N, mu0=mu, mu1=mu1, kappa_base=kappa_base, eta=eta_b)
    out_b = generate_dm_two_sample(sc_b, seed=9)
    plot_dgp_quality(out_b, title_prefix="Scenario B (mean shift)")