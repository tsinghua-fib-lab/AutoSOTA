import numpy as np
import pandas as pd

# ============================================================
# IMPORTANT: Reuse the core replicate engine from the UPDATED size file.
# This guarantees power and size use identical test implementations.
# ============================================================
from empirical_size import one_replicate_pvalues

# ----------------------------
# Mean shift constructor (simplex-safe)
# ----------------------------
def make_mean_shifted(mu0, delta=0.0, pattern="sparse2"):
    """
    Construct mu1 from mu0 via multiplicative log-contrast tilt:
        mu1 ∝ mu0 * exp(delta * v), with sum(v)=0.

    Args:
      mu0     : base mean on simplex (length d, positive, sums to 1)
      delta   : effect size (>=0). delta=0 -> mu1=mu0.
      pattern : direction template for v.
                - "sparse2": +1 on comp 0, -1 on comp 1
                - "block":  +1 on first k, -1 on last k
                - "linear": increasing/decreasing contrast

    Returns:
      mu1 on simplex.
    """
    mu0 = np.asarray(mu0, float)
    d = mu0.size
    if np.any(mu0 <= 0):
        raise ValueError("mu0 must be strictly positive.")
    mu0 = mu0 / mu0.sum()

    if float(delta) == 0.0:
        return mu0.copy()

    if pattern == "sparse2":
        v = np.zeros(d)
        i_pos = 0
        i_neg = 1 if d > 1 else 0
        v[i_pos] = 1.0
        v[i_neg] = -1.0
    elif pattern == "block":
        k = max(1, d // 5)  # 20% block
        v = np.zeros(d)
        v[:k] = 1.0
        v[-k:] = -1.0
    elif pattern == "linear":
        v = np.linspace(-1, 1, d)
    else:
        raise ValueError("Unknown pattern. Use 'sparse2', 'block', or 'linear'.")

    v = v - v.mean()  # enforce sum(v)=0
    mu1 = mu0 * np.exp(float(delta) * v)

    # numerical guards
    mu1 = np.maximum(mu1, 1e-300)
    mu1 = mu1 / mu1.sum()
    return mu1


# ----------------------------
# Power grid runner
# ----------------------------
def run_power_grid(
    d_list=(10, 30, 100),
    mn_list=((50, 50), (50, 100), (200, 200), (200, 400)),
    Nlib_by_pz={0.00: 2000, 0.05: 200, 0.20: 50, 0.50: 20},
    pz_list=(0.00, 0.05, 0.20, 0.50),
    alphas=(0.01, 0.05, 0.10),
    R=200,
    B_perm=500,
    eps_clr=1e-8,
    kappa_base=25.0,
    beta_ppk=0.5,
    seed0=20260106,
    # --- effects ---
    mean_shift_list=(0.0, 0.2, 0.4, 0.6),
    eta_list=(1.0,),                 # dispersion effect for group 1
    shift_pattern="sparse2",
):
    """
    Empirical POWER under alternatives:
      - mean shift in group 1 via mu1 = tilt(mu0; delta)
      - dispersion change in group 1 via eta (alpha1 = kappa_base*eta*mu1)

    Output: one row per (scenario, method, effect config), with rejection rates.
    """
    methods = ["CLR-ED", "CLR-MMD", "HEL-MMD", "PPK-MMD", "PERMANOVA-AIT", "CLL-Tmax"]

    def make_mu(d):
        v = np.linspace(1, d, d)
        return v / v.sum()

    rows = []
    job = 0

    for d in d_list:
        mu0 = make_mu(d)

        for (m, n) in mn_list:
            for pz in pz_list:
                if pz not in Nlib_by_pz:
                    raise ValueError(f"pz={pz} not in Nlib_by_pz.")
                Nlib = int(Nlib_by_pz[pz])

                for delta in mean_shift_list:
                    mu1 = make_mean_shifted(mu0, delta=float(delta), pattern=shift_pattern)

                    for eta in eta_list:
                        job += 1

                        print("\n========================================================")
                        print(f"[JOB {job}] POWER RUN (H1)")
                        print(f"  d={d}, (m,n)=({m},{n}), Nlib={Nlib}, pz label={pz:.2f}")
                        print(f"  mean_shift(delta)={float(delta):.3f}, dispersion(eta)={float(eta):.3f}, pattern={shift_pattern}")
                        print("========================================================")

                        means = (mu0, mu1)

                        rej_counts = {meth: np.zeros(len(alphas), dtype=int) for meth in methods}
                        pz_emp_acc = []

                        for r in range(R):
                            seed = seed0 + 1_000_000 * job + r

                            # Reuse exactly the same replicate engine as size:
                            pvals, meta = one_replicate_pvalues(
                                d=d, m=m, n=n,
                                Nlib=Nlib,
                                means=means,
                                eta=float(eta),
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
                                "d": int(d), "m": int(m), "n": int(n),
                                "pz_label": float(pz),
                                "Nlib": int(Nlib),
                                "mean_shift": float(delta),
                                "eta": float(eta),
                                "shift_pattern": str(shift_pattern),
                                "pz_emp_mean": float(np.mean(pz_emp_acc)),
                                "method": meth,
                                "rej@1%": float(vec[0]),
                                "rej@5%": float(vec[1]),
                                "rej@10%": float(vec[2]),
                                "R": int(R),
                                "B_perm": int(B_perm),
                                "eps_clr": float(eps_clr),
                                "kappa_base": float(kappa_base),
                                "beta_ppk": float(beta_ppk),
                            })

    df = pd.DataFrame(rows).sort_values(
        ["d","m","n","pz_label","mean_shift","eta","method"]
    ).reset_index(drop=True)
    return df


# ----------------------------
# Example main: run power grid + save
# ----------------------------
if __name__ == "__main__":
    df_power = run_power_grid(
        d_list=(10, 30, 100),
        mn_list=((50, 50), (50, 100), (200, 200), (200, 400)),
        Nlib_by_pz={0.00: 2000, 0.05: 200, 0.20: 50},
        pz_list=(0.00, 0.05, 0.20),
        alphas=(0.01, 0.05, 0.10),
        R=200,
        B_perm=500,
        eps_clr=1e-8,
        kappa_base=25.0,
        beta_ppk=0.5,
        seed0=123456,
        mean_shift_list=(0.0, 0.2, 0.4, 0.6),
        eta_list=(1.0, 0.7, 0.4, 1.5),
        shift_pattern="sparse2",
    )

    print("\n===== Empirical POWER table (H1) =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(df_power.to_string(index=False))

    out_csv = "power_table_psd_cached.csv"
    df_power.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")