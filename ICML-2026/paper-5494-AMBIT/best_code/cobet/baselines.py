"""
baselines.py
------------
Baseline independence tests: HSIC and dCov (Distance Correlation).

Both use the same Clayton-copula + transform DGP as CoBET / wa-dCoBET
and produce results in the same dict / DataFrame format for easy comparison.

Dependencies
------------
    pip install hyppo

Public API
----------
simulate_hsic              – power / Type-I sweep via HSIC (hyppo)
simulate_dcorr             – power / Type-I sweep via dCor (hyppo)
run_multi_n_hsic           – multi-n sweep + Excel export for HSIC
run_multi_n_dcorr          – multi-n sweep + Excel export for dCor
print_results              – pretty-print a list of result dicts
"""

import numpy as np
import pandas as pd

from .utils import generate_XY


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _normalise_b_list(b_list):
    if not isinstance(b_list, (list, tuple, np.ndarray)):
        return [b_list]
    return list(b_list)


def _save_excel(all_results_by_n, excel_path, method_name="results"):
    """Write one sheet per n to an Excel file; fall back to CSVs."""
    for eng in ("openpyxl", "xlsxwriter"):
        try:
            with pd.ExcelWriter(excel_path, engine=eng) as writer:
                for n, df in all_results_by_n.items():
                    df.to_excel(writer, sheet_name=f"n={n}", index=False)
            print(f"Saved → {excel_path}  (engine={eng})")
            return
        except Exception:
            continue

    # CSV fallback
    for n, df in all_results_by_n.items():
        path = excel_path.replace(".xlsx", f"_n{n}.csv")
        df.to_csv(path, index=False)
        print(f"Excel unavailable; saved CSV → {path}")


def print_results(results, method=""):
    """Print a list of result dicts in a readable table."""
    by_key = {}
    for r in results:
        key = (r["n"], r["transform"])
        by_key.setdefault(key, []).append(r)
    for (n, t), rows in sorted(by_key.items()):
        D = rows[0].get("D", rows[0].get("d", "?"))
        print(f"\n=== {method} | n={n}, transform={t}, D={D} ===")
        for r in rows:
            if r["metric"] == "typeI":
                print(f"  Type I (b=0):     {r['value']:.3f}")
            else:
                print(f"  Power  (b={r['b']}): {r['value']:.3f}")


# ─────────────────────────────────────────────
# HSIC
# ─────────────────────────────────────────────

def simulate_hsic(
    n, theta, D, b_config,
    n_simulations=500, alpha=0.05, seed=123,
    report_typeI=True,
):
    """
    Monte Carlo power / Type-I sweep using HSIC (Gaussian kernel, median heuristic).

    Parameters
    ----------
    n             : int   – sample size
    theta         : float – Clayton copula parameter
    D             : int   – dimension
    b_config      : dict  – {transform_key: [b_values]}
    n_simulations : int
    alpha         : float
    seed          : int
    report_typeI  : bool  – include b=0 (Type I) rows

    Returns
    -------
    list of dicts with keys: n, transform, D, b, metric ('typeI'|'power'), value
    """
    try:
        from hyppo.independence import Hsic
    except ImportError as exc:
        raise ImportError(
            "hyppo is required for HSIC. Install with: pip install hyppo"
        ) from exc

    rng  = np.random.default_rng(seed)
    hsic = Hsic()
    results = []

    for transform_key, b_list in b_config.items():
        b_list = _normalise_b_list(b_list)

        # Type I (b = 0)
        if report_typeI:
            rej = 0
            for _ in range(n_simulations):
                X, Y = generate_XY(n, theta, D, transform_key, b=0.0,
                                   rng=np.random.RandomState(int(rng.integers(0, 2**31))))
                _, pval = hsic.test(X, Y, random_state=int(rng.integers(0, 2**31 - 1)))
                rej += pval < alpha
            results.append({
                "n": n, "transform": transform_key, "D": D,
                "b": 0.0, "metric": "typeI", "value": float(rej / n_simulations),
            })

        # Power (b > 0)
        for b in b_list:
            rej = 0
            for _ in range(n_simulations):
                X, Y = generate_XY(n, theta, D, transform_key, b=b,
                                   rng=np.random.RandomState(int(rng.integers(0, 2**31))))
                _, pval = hsic.test(X, Y, random_state=int(rng.integers(0, 2**31 - 1)))
                rej += pval < alpha
            b_out = float(b) if np.isscalar(b) else tuple(np.asarray(b).tolist())
            results.append({
                "n": n, "transform": transform_key, "D": D,
                "b": b_out, "metric": "power", "value": float(rej / n_simulations),
            })

    return results


def run_multi_n_hsic(
    n_list, theta, D, b_config_by_n,
    n_simulations=500, alpha=0.05, seed=123,
    excel_path="hsic_results.xlsx",
):
    """
    Run simulate_hsic for multiple sample sizes and export to Excel.

    Parameters
    ----------
    n_list        : list of int
    theta         : float
    D             : int
    b_config_by_n : dict  – {n: {transform: [b_values]}}
    n_simulations : int
    alpha         : float
    seed          : int
    excel_path    : str

    Returns
    -------
    dict: {n: pd.DataFrame}
    """
    all_results = {}
    for n in n_list:
        rows = simulate_hsic(
            n=n, theta=theta, D=D, b_config=b_config_by_n[n],
            n_simulations=n_simulations, alpha=alpha, seed=seed,
            report_typeI=True,
        )
        df = pd.DataFrame(rows)
        all_results[n] = df
        print_results(rows, method="HSIC")

    _save_excel(all_results, excel_path, method_name="HSIC")
    return all_results


# ─────────────────────────────────────────────
# dCov / dCor
# ─────────────────────────────────────────────

def simulate_dcorr(
    n, theta, D, b_config,
    n_simulations=500, alpha=0.05, seed=123,
    report_typeI=True, reps=1000, workers=1,
):
    """
    Monte Carlo power / Type-I sweep using distance correlation (dCor).

    Parameters
    ----------
    n             : int
    theta         : float
    D             : int
    b_config      : dict  – {transform_key: [b_values]}
    n_simulations : int
    alpha         : float
    seed          : int
    report_typeI  : bool
    reps          : int   – permutation replications for p-value
    workers       : int   – parallel workers (1 = serial)

    Returns
    -------
    list of dicts with keys: n, transform, D, b, metric, value
    """
    try:
        from hyppo.independence import Dcorr
    except ImportError as exc:
        raise ImportError(
            "hyppo is required for dCov/dCor. Install with: pip install hyppo"
        ) from exc

    rng   = np.random.default_rng(seed)
    dcorr = Dcorr()
    results = []

    for transform_key, b_list in b_config.items():
        b_list = _normalise_b_list(b_list)

        # Type I (b = 0)
        if report_typeI:
            rej = 0
            for _ in range(n_simulations):
                X, Y = generate_XY(n, theta, D, transform_key, b=0.0,
                                   rng=np.random.RandomState(int(rng.integers(0, 2**31))))
                _, pval = dcorr.test(X, Y, reps=reps, workers=workers)
                rej += pval < alpha
            results.append({
                "n": n, "transform": transform_key, "D": D,
                "b": 0.0, "metric": "typeI", "value": float(rej / n_simulations),
            })

        # Power (b > 0)
        for b in b_list:
            rej = 0
            for _ in range(n_simulations):
                X, Y = generate_XY(n, theta, D, transform_key, b=b,
                                   rng=np.random.RandomState(int(rng.integers(0, 2**31))))
                _, pval = dcorr.test(X, Y, reps=reps, workers=workers)
                rej += pval < alpha
            b_out = float(b) if np.isscalar(b) else tuple(np.asarray(b).tolist())
            results.append({
                "n": n, "transform": transform_key, "D": D,
                "b": b_out, "metric": "power", "value": float(rej / n_simulations),
            })

    return results


def run_multi_n_dcorr(
    n_list, theta, D, b_config_by_n,
    n_simulations=500, alpha=0.05, seed=123,
    report_typeI=True, reps=1000, workers=1,
    excel_path="dcorr_results.xlsx",
):
    """
    Run simulate_dcorr for multiple sample sizes and export to Excel.

    Parameters
    ----------
    n_list        : list of int
    theta         : float
    D             : int
    b_config_by_n : dict  – {n: {transform: [b_values]}}
    n_simulations : int
    alpha         : float
    seed          : int
    report_typeI  : bool
    reps          : int
    workers       : int
    excel_path    : str

    Returns
    -------
    dict: {n: pd.DataFrame}
    """
    all_results = {}
    for n in n_list:
        rows = simulate_dcorr(
            n=n, theta=theta, D=D, b_config=b_config_by_n[n],
            n_simulations=n_simulations, alpha=alpha, seed=seed,
            report_typeI=report_typeI, reps=reps, workers=workers,
        )
        df = pd.DataFrame(rows)
        all_results[n] = df
        print_results(rows, method="dCor")

    _save_excel(all_results, excel_path, method_name="dCor")
    return all_results


# ─────────────────────────────────────────────
# Quick-start example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    theta, D, alpha, seed = 2, 10, 0.05, 123
    n_simulations = 200   # increase for production runs

    n_list = [250, 500, 1000]
    b_config_by_n = {
        250:  {"trigU": [0.05, 0.08, 0.1],  "logquad": [0.1, 0.3, 0.5]},
        500:  {"trigU": [0.03, 0.05, 0.10], "logquad": [0.1, 0.2, 0.3]},
        1000: {"trigU": [0.01, 0.03, 0.05], "logquad": [0.07, 0.15, 0.2]},
    }

    run_multi_n_hsic(
        n_list=n_list, theta=theta, D=D,
        b_config_by_n=b_config_by_n,
        n_simulations=n_simulations, alpha=alpha, seed=seed,
        excel_path="hsic_results.xlsx",
    )

    run_multi_n_dcorr(
        n_list=n_list, theta=theta, D=D,
        b_config_by_n=b_config_by_n,
        n_simulations=n_simulations, alpha=alpha, seed=seed,
        reps=499,   # fewer permutations for speed
        excel_path="dcorr_results.xlsx",
    )
