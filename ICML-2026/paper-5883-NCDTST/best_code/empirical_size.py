import numpy as np
import pandas as pd

# ============================================================
# UPDATED: Use the DGP from dgp_dm_two_sample.py
# ============================================================
from dgp_dm_two_sample import DMScenario, generate_dm_two_sample

# ============================================================
# Compositional two-sample simulation engine (cached & PSD-safe)
# Baselines included:
#   (1) CLR + Energy distance  (CLR-ED)
#   (2) Gaussian MMD on CLR (CLR-MMD)
#   (3) PERMANOVA with Aitchison distance (PERMANOVA-AIT)
#   (4) Hellinger geometry MMD (HEL-MMD)  [Gaussian on sqrt(x)]
#   (5) Probability Product Kernel MMD (PPK-MMD, PSD simplex kernel)
#   (6) Max test in CLR space, calibrated by permutation (CLL-Tmax)
#
# Key design points:
#   - DM counts -> normalize to simplex generates zeros naturally.
#   - FIXED library size N (deterministic per scenario).
#   - Permutation p-values reuse CACHED pooled matrices per replicate.
# ============================================================

METH_SEED = {
    "CLR-ED": 101,
    "CLR-MMD": 102,
    "HEL-MMD": 103,
    "PPK-MMD": 104,
    "PERMANOVA-AIT": 105,
    "CLL-Tmax": 106,
}


# ============================================================
# 0) Simplex hygiene and transforms
# ============================================================
def add_pseudocount_and_renorm(X, eps=1e-8):
    """
    CLR requires strictly positive components.
    Add eps and renormalize to simplex.
    """
    Xp = X + eps
    Xp = Xp / np.maximum(Xp.sum(axis=1, keepdims=True), 1e-300)
    return Xp


def clr_transform(X):
    """
    clr(x) = log(x) - mean_j log(x_j)
    """
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


# ============================================================
# 1) Pairwise distances and kernels (cached objects)
# ============================================================
def pairwise_l2_dist(Z):
    """
    Full pairwise Euclidean distance matrix.
    """
    G = Z @ Z.T
    sq = np.sum(Z * Z, axis=1, keepdims=True)
    D2 = np.maximum(sq + sq.T - 2.0 * G, 0.0)
    return np.sqrt(D2)


def median_heuristic_sigma_from_dist(D, max_points=500, seed=0):
    """
    sigma = median(D_ij, i<j, D_ij>0), using optional subsample.
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    Ds = D[np.ix_(idx, idx)]
    vals = Ds[np.triu_indices(Ds.shape[0], k=1)]
    vals = vals[vals > 0]
    med = float(np.median(vals)) if vals.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return med


def gaussian_kernel_from_dist(D, sigma):
    sigma = max(float(sigma), 1e-12)
    return np.exp(-(D**2) / (2.0 * sigma * sigma))


def probability_product_kernel(Z, beta=0.5):
    """
    PSD kernel on simplex:
      k(x,y) = sum_j x_j^beta y_j^beta
    """
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must be in (0,1].")
    Phi = np.power(np.maximum(Z, 0.0), beta)
    return Phi @ Phi.T


# ============================================================
# 2) Cached test statistics from pooled matrices
# ============================================================
def mean_offdiag(Dsub):
    n = Dsub.shape[0]
    if n <= 1:
        return 0.0
    return (Dsub.sum() - np.trace(Dsub)) / (n * (n - 1))


def energy_from_dist(D, idxX, idxY):
    """
    Energy distance statistic from pooled distance matrix:
      E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    Within-group uses off-diagonal mean distances (correct).
    """
    DXy = D[np.ix_(idxX, idxY)].mean()
    DXx = mean_offdiag(D[np.ix_(idxX, idxX)])
    DYy = mean_offdiag(D[np.ix_(idxY, idxY)])
    return float(2.0 * DXy - DXx - DYy)


def mmd2_unbiased_from_kernel(K, idxX, idxY):
    """
    Unbiased MMD^2 from pooled kernel matrix.
    """
    idxX = np.asarray(idxX)
    idxY = np.asarray(idxY)
    m = idxX.size
    n = idxY.size
    if m < 2 or n < 2:
        return 0.0

    Kxx = K[np.ix_(idxX, idxX)].copy()
    Kyy = K[np.ix_(idxY, idxY)].copy()
    Kxy = K[np.ix_(idxX, idxY)]

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def gower_centered_G_from_dist(D):
    """
    Gower-centered matrix for PERMANOVA from distances D.
    """
    N = D.shape[0]
    A = -0.5 * (D**2)
    H = np.eye(N) - np.ones((N, N)) / N
    return H @ A @ H


def permanova_F_from_G(G, idxX, idxY):
    """
    Two-group PERMANOVA pseudo-F from cached G.
    """
    N = G.shape[0]
    SST = float(np.trace(G))
    SSW = float(np.trace(G[np.ix_(idxX, idxX)]) + np.trace(G[np.ix_(idxY, idxY)]))
    SSB = float(SST - SSW)
    dfB = 1
    dfW = N - 2
    return float((SSB / dfB) / max(SSW / dfW, 1e-300))


def Tmax_clr(Z_clr, idxX, idxY):
    """
    Max standardized mean difference in CLR coordinates.
    Calibrated by permutation => valid under i.i.d. null.
    """
    X = Z_clr[idxX]
    Y = Z_clr[idxY]
    m, n = X.shape[0], Y.shape[0]

    delta = X.mean(axis=0) - Y.mean(axis=0)
    varX = X.var(axis=0, ddof=1)
    varY = Y.var(axis=0, ddof=1)
    se2 = varX / max(m, 1) + varY / max(n, 1)
    T = delta / np.sqrt(np.maximum(se2, 1e-300))
    return float(np.max(np.abs(T)))


# ============================================================
# 3) Permutation p-values using cached objects
# ============================================================
def perm_pvalue(obs_stat, stat_on_split, Ntot, m, B=500, seed=0):
    """
    p = (1 + #{b: T_b >= T_obs}) / (B+1)
    """
    rng = np.random.default_rng(seed)
    base = np.arange(Ntot)

    ge = 0
    for _ in range(B):
        perm = rng.permutation(base)
        idxX = perm[:m]
        idxY = perm[m:]
        tb = stat_on_split(idxX, idxY)
        ge += (tb >= obs_stat)

    return (1 + ge) / (B + 1)


# ============================================================
# 4) One replicate: build pooled cached matrices ONCE
# ============================================================
def one_replicate_pvalues(
    d, m, n, Nlib, means, eta,
    eps_clr=1e-8,
    kappa_base=25.0,
    beta_ppk=0.5,
    B_perm=500,
    seed=0
):
    """
    One replicate:
      (i) generate DM compositions (with zeros) using the external DGP,
      (ii) compute pooled cached matrices,
      (iii) compute observed stats,
      (iv) compute permutation p-values reusing caches.
    """
    mu0 = np.asarray(means[0], float)
    mu1 = np.asarray(means[1], float)
    if mu0.size != d or mu1.size != d:
        raise ValueError("means[0], means[1] must be length d.")
    if np.any(mu0 <= 0) or np.any(mu1 <= 0):
        raise ValueError("Dirichlet means must be strictly positive.")

    # ---- UPDATED: generate via DMScenario + generate_dm_two_sample ----
    sc = DMScenario(
        d=int(d), m=int(m), n=int(n), N=int(Nlib),
        mu0=mu0, mu1=mu1,
        kappa_base=float(kappa_base),
        eta=float(eta),
    )
    out = generate_dm_two_sample(sc, seed=int(seed))

    X = out["comps0"]
    Y = out["comps1"]
    Z = np.vstack([X, Y])
    Ntot = m + n

    # ---- CLR (Aitchison geometry) with pseudocount ----
    Zp = add_pseudocount_and_renorm(Z, eps=eps_clr)
    Z_clr = clr_transform(Zp)

    # ---- Distances ----
    D_clr = pairwise_l2_dist(Z_clr)          # Aitchison distance
    D_hel = pairwise_l2_dist(np.sqrt(Z))     # Hellinger geometry distance

    # ---- Kernels ----
    sig_clr = median_heuristic_sigma_from_dist(D_clr, seed=seed + 1)
    K_clr = gaussian_kernel_from_dist(D_clr, sig_clr)

    sig_hel = median_heuristic_sigma_from_dist(D_hel, seed=seed + 2)
    K_hel = gaussian_kernel_from_dist(D_hel, sig_hel)

    K_ppk = probability_product_kernel(Z, beta=beta_ppk)

    # PERMANOVA cache
    G = gower_centered_G_from_dist(D_clr)

    idxX0 = np.arange(m)
    idxY0 = np.arange(m, Ntot)

    obs = {
        "CLR-ED": energy_from_dist(D_clr, idxX0, idxY0),
        "CLR-MMD": mmd2_unbiased_from_kernel(K_clr, idxX0, idxY0),
        "HEL-MMD": mmd2_unbiased_from_kernel(K_hel, idxX0, idxY0),
        "PPK-MMD": mmd2_unbiased_from_kernel(K_ppk, idxX0, idxY0),
        "PERMANOVA-AIT": permanova_F_from_G(G, idxX0, idxY0),
        "CLL-Tmax": Tmax_clr(Z_clr, idxX0, idxY0),
    }

    stat_perm = {
        "CLR-ED": lambda ix, iy: energy_from_dist(D_clr, ix, iy),
        "CLR-MMD": lambda ix, iy: mmd2_unbiased_from_kernel(K_clr, ix, iy),
        "HEL-MMD": lambda ix, iy: mmd2_unbiased_from_kernel(K_hel, ix, iy),
        "PPK-MMD": lambda ix, iy: mmd2_unbiased_from_kernel(K_ppk, ix, iy),
        "PERMANOVA-AIT": lambda ix, iy: permanova_F_from_G(G, ix, iy),
        "CLL-Tmax": lambda ix, iy: Tmax_clr(Z_clr, ix, iy),
    }

    pvals = {}
    for meth in obs.keys():
        pvals[meth] = perm_pvalue(
            obs_stat=obs[meth],
            stat_on_split=stat_perm[meth],
            Ntot=Ntot,
            m=m,
            B=B_perm,
            seed=seed + 10_000 + METH_SEED[meth],
        )

    meta = {
        "Nlib": int(out["N"]),
        "pz_emp": float(out["pz_empirical"]),  # UPDATED key
        "sig_clr": float(sig_clr),
        "sig_hel": float(sig_hel),
        "beta_ppk": float(beta_ppk),
        "eps_clr": float(eps_clr),
        "eta": float(eta),
        "kappa_base": float(kappa_base),
    }
    return pvals, meta


# ============================================================
# 5) Grid runner: empirical size (H0) table
# ============================================================
def run_size_grid(
    d_list=(10, 30, 100),
    mn_list=((50, 50), (50, 100), (200, 200), (200, 400)),
    Nlib_by_pz={0.00: 2000, 0.05: 200, 0.20: 50},
    pz_list=(0.00, 0.05, 0.20),
    alphas=(0.01, 0.05, 0.10),
    R=200,
    B_perm=500,
    eps_clr=1e-8,
    kappa_base=25.0,
    eta=1.0,
    beta_ppk=0.5,
    seed0=20260106
):
    methods = ["CLR-ED", "CLR-MMD", "HEL-MMD", "PPK-MMD", "PERMANOVA-AIT", "CLL-Tmax"]

    def make_mu(d):
        v = np.linspace(1, d, d)
        return v / v.sum()

    rows = []
    job = 0

    for d in d_list:
        mu = make_mu(d)
        means = (mu, mu)  # H0

        for (m, n) in mn_list:
            for pz in pz_list:
                job += 1
                if pz not in Nlib_by_pz:
                    raise ValueError(f"pz={pz} not in Nlib_by_pz.")
                Nlib = int(Nlib_by_pz[pz])

                print("\n========================================================")
                print(f"[JOB {job}] SIZE RUN (H0: identical distributions)")
                print(f"  d={d}, (m,n)=({m},{n}), Nlib={Nlib}, nominal pz label={pz:.2f}, eta={eta}")
                print("========================================================")

                rej_counts = {meth: np.zeros(len(alphas), dtype=int) for meth in methods}
                pz_emp_acc = []

                for r in range(R):
                    seed = seed0 + 1_000_000 * job + r
                    pvals, meta = one_replicate_pvalues(
                        d=d, m=m, n=n,
                        Nlib=Nlib,
                        means=means,
                        eta=eta,
                        eps_clr=eps_clr,
                        kappa_base=kappa_base,
                        beta_ppk=beta_ppk,
                        B_perm=B_perm,
                        seed=seed
                    )

                    pz_emp_acc.append(meta["pz_emp"])

                    for meth in methods:
                        for k, a in enumerate(alphas):
                            rej_counts[meth][k] += int(pvals[meth] <= a)

                for meth in methods:
                    vec = rej_counts[meth] / float(R)
                    rows.append({
                        "d": d, "m": m, "n": n,
                        "pz_label": float(pz),
                        "Nlib": Nlib,
                        "pz_emp_mean": float(np.mean(pz_emp_acc)),
                        "method": meth,
                        "rej@1%": float(vec[0]),
                        "rej@5%": float(vec[1]),
                        "rej@10%": float(vec[2]),
                        "R": int(R),
                        "B_perm": int(B_perm),
                        "eta": float(eta),
                        "eps_clr": float(eps_clr),
                        "kappa_base": float(kappa_base),
                        "beta_ppk": float(beta_ppk),
                    })

    df = pd.DataFrame(rows).sort_values(["d", "m", "n", "pz_label", "method"]).reset_index(drop=True)
    return df


# ============================================================
# 6) Main
# ============================================================
if __name__ == "__main__":
    RUN_SINGLE_GRID = True
    RUN_100_RUNS = True

    # --- single grid run ---
    if RUN_SINGLE_GRID:
        df = run_size_grid(
            d_list=(10, 30, 100),
            mn_list=((50, 50), (50, 100), (200, 200), (200, 400)),
            Nlib_by_pz={0.00: 2000, 0.05: 200, 0.20: 50},
            pz_list=(0.00, 0.05, 0.20),
            alphas=(0.01, 0.05, 0.10),
            R=200,
            B_perm=500,
            eps_clr=1e-8,
            kappa_base=25.0,
            eta=1.0,
            beta_ppk=0.5,
            seed0=20260106
        )

        print("\n===== Empirical SIZE table (H0: identical distributions) =====")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 400)
        print(df.to_string(index=False))

        out_csv = "size_table_psd_cached.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

    # --- 100 independent grid runs + mean/sd summary ---
    if RUN_100_RUNS:
        base_seed = 20260106
        n_runs = 100

        all_dfs = []
        for r in range(n_runs):
            df_r = run_size_grid(
                d_list=(10, 30, 100),
                mn_list=((50, 50), (50, 100), (200, 200), (200, 400)),
                Nlib_by_pz={0.00: 2000, 0.05: 200, 0.20: 50, 0.50: 20},
                pz_list=(0.00, 0.05, 0.20, 0.50),
                alphas=(0.01, 0.05, 0.10),
                R=200,
                B_perm=500,
                eps_clr=1e-8,
                kappa_base=25.0,
                eta=1.0,
                beta_ppk=0.5,
                seed0=base_seed + r
            )
            df_r["run_id"] = r
            df_r["seed0"] = base_seed + r
            all_dfs.append(df_r)

        df_all = pd.concat(all_dfs, ignore_index=True)

        key_cols = [
            "d","m","n","pz_label","Nlib","method",
            "R","B_perm","eta","eps_clr","kappa_base","beta_ppk"
        ]
        key_cols = [c for c in key_cols if c in df_all.columns]

        metric_cols = ["rej@1%","rej@5%","rej@10%","pz_emp_mean"]
        metric_cols = [c for c in metric_cols if c in df_all.columns]

        agg_dict = {c: ["mean", "std"] for c in metric_cols}

        summary = (
            df_all
            .groupby(key_cols, as_index=False)
            .agg(agg_dict)
        )

        summary.columns = [
            "_".join([x for x in col if x]) if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        preferred = []
        for c in ["rej@1%","rej@5%","rej@10%","pz_emp_mean"]:
            if f"{c}_mean" in summary.columns: preferred.append(f"{c}_mean")
            if f"{c}_std"  in summary.columns: preferred.append(f"{c}_std")
        other_cols = [c for c in summary.columns if c not in preferred]
        summary = summary[other_cols + preferred]

        print("\n===== Empirical SIZE summary over 100 runs: mean (sd) =====")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 400)
        print(summary.to_string(index=False))

        df_all.to_csv("size_table_psd_cached_raw_100runs.csv", index=False)
        summary.to_csv("size_table_psd_cached_summary_100runs.csv", index=False)
        print("\nSaved: size_table_psd_cached_raw_100runs.csv")
        print("Saved: size_table_psd_cached_summary_100runs.csv")