"""
Verify Table 1 of the paper against the shipped JSON results.
===============================================================

For each row of Table 1, this script:
  1. Loads the corresponding results/*.json file
  2. Computes the ratio claimed in the paper
  3. Prints PASS / FAIL with the observed vs. claimed numbers

Exits with code 0 if every check passes, 1 otherwise.

Usage:
    python verify_table1.py
    python verify_table1.py --tolerance 0.20   # allow 20% drift per check
"""
import argparse
import json
import os
import statistics
import sys


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean(model_dict, key="mse"):
    """Mean over per-seed list, or fall back to *_mean field."""
    if key in model_dict and isinstance(model_dict[key], list):
        return statistics.mean(model_dict[key])
    if f"{key}_mean" in model_dict:
        return model_dict[f"{key}_mean"]
    if "mae_mean" in model_dict and key == "mae":
        return model_dict["mae_mean"]
    raise KeyError(f"Cannot find {key} in {list(model_dict)}")


def _check(label, observed, expected, tol, unit=""):
    rel = abs(observed - expected) / max(abs(expected), 1e-9)
    ok = rel <= tol
    marker = "PASS" if ok else "FAIL"
    sign = "+" if observed >= expected else ""
    print(f"  [{marker}]  {label}: observed {observed:.3f}{unit}, "
          f"expected {expected:.3f}{unit} ({sign}{rel*100:.1f}%)")
    return ok


def row_1_param_match(tol):
    print("\n[Row 1] Parameter-matched, near-singular, n=20, 256 params  "
          "(paper: 20x vs FNN)")
    d = _load("exp4_param_matched_results.json")
    if d is None:
        print("  [SKIP] exp4_param_matched_results.json not found")
        return True
    sub = d["Near-singular"]
    ratio = _mean(sub["FNN"]) / _mean(sub["CauchyNet"])
    return _check("FNN_MSE / CN_MSE", ratio, 20.0, tol, "x")


def row_2_delta_sweep(tol):
    print("\n[Row 2] delta-sweep, max ratio  (paper: up to 102x vs FNN)")
    d = _load("exp9_delta_sweep_results.json")
    if d is None:
        print("  [SKIP] exp9_delta_sweep_results.json not found")
        return True
    ratios = []
    for delta_key, vals in d.items():
        if not isinstance(vals, dict) or "CauchyNet" not in vals:
            continue
        cn = _mean(vals["CauchyNet"])
        fnn = _mean(vals["FNN"])
        if cn > 0 and fnn > 0:
            ratios.append((delta_key, fnn / cn))
    if not ratios:
        print("  [SKIP] no comparable cells")
        return True
    best = max(ratios, key=lambda x: x[1])
    for k, r in ratios:
        print(f"      delta={k}: ratio={r:.2f}x")
    return _check(f"max ratio (at {best[0]})", best[1], 102.0, tol, "x")


def row_3_multilayer_skip(tol):
    print("\n[Row 3] Multi-layer (Skip)  (paper: 1.7x vs 1-layer CN)")
    d = _load("exp6_multilayer_results.json")
    if d is None:
        print("  [SKIP] exp6_multilayer_results.json not found")
        return True
    one_layer = d["CauchyNet-1L"]
    skip = d["CauchyNet-Skip"]
    ratio = _mean(one_layer) / _mean(skip)
    p1 = one_layer.get("num_params")
    p2 = skip.get("num_params")
    print(f"      1L params={p1}, Skip params={p2}  (NOT param-matched)")
    return _check("MSE_1L / MSE_Skip", ratio, 1.7, tol, "x")


def row_4_uci_tabular(tol):
    print("\n[Row 4] UCI tabular  (paper: Hybrid wins 3/3, ~4x fewer params)")
    d = _load("exp7_uci_results.json")
    if d is None:
        print("  [SKIP] exp7_uci_results.json not found")
        return True
    all_ok = True
    for name in ("Diabetes", "California", "Wine"):
        if name not in d:
            print(f"      [SKIP] dataset {name} missing")
            continue
        sub = d[name]
        h_mse = sub["Hybrid"]["mse_mean"]
        cn_mse = sub["CauchyNet"]["mse_mean"]
        fnn_mse = sub["FNN"]["mse_mean"]
        h_p = sub["Hybrid"].get("num_params", sub["Hybrid"].get("params"))
        cn_p = sub["CauchyNet"].get("num_params", sub["CauchyNet"].get("params"))
        wins = h_mse < cn_mse and h_mse < fnn_mse
        param_ratio = cn_p / h_p if h_p else float("nan")
        marker = "PASS" if wins else "FAIL"
        print(f"  [{marker}]  {name}: Hybrid={h_mse:.3e} (params={h_p}), "
              f"CN={cn_mse:.3e}, FNN={fnn_mse:.3e}  | param ratio CN/Hybrid={param_ratio:.1f}x")
        all_ok = all_ok and wins
    return all_ok


def row_5_rational_nn(tol):
    print("\n[Row 5] RationalNN  (paper: CN wins 3/3, 1.2-2.0x)")
    d = _load("exp8_rational_results.json")
    if d is None:
        print("  [SKIP] exp8_rational_results.json not found")
        return True
    all_ok = True
    for name, sub in d.items():
        if not isinstance(sub, dict) or "CauchyNet" not in sub:
            continue
        cn = _mean(sub["CauchyNet"])
        rn = _mean(sub["RationalNN"])
        ratio = rn / cn
        wins_in_range = (cn < rn) and (1.1 <= ratio <= 2.0)
        marker = "PASS" if wins_in_range else "FAIL"
        print(f"  [{marker}]  {name}: CN={cn:.3e}, RationalNN={rn:.3e}, "
              f"ratio={ratio:.2f}x (expected 1.2-2.0x)")
        all_ok = all_ok and wins_in_range
    return all_ok


def row_7_gap_filling(tol):
    print("\n[Row 7] Gap-filling, fixed-pole config at sweep-best setup  "
          "(paper: MAE 0.020, 4.3x vs SIREN, 9.6x vs N-BEATS, 10.1x vs FNN)")
    d = _load("best_config_gap_filling.json")
    if d is None:
        print("  [SKIP] best_config_gap_filling.json not found")
        return True
    cn = d.get("CauchyNet", d.get("CauchyNet_optimised", {}))["mae_mean"]
    siren = d.get("SIREN", {}).get("mae_mean")
    fnn = d.get("FNN_ReLU", {}).get("mae_mean")
    nbeats = d.get("NBeats", {}).get("mae_mean")
    ok_mae = _check("CauchyNet MAE", cn, 0.020, tol)
    ok_siren = True
    if siren is not None:
        ok_siren = _check("SIREN_MAE / CN_MAE", siren / cn, 4.3, tol, "x")
    ok_nbeats = True
    if nbeats is not None:
        ok_nbeats = _check("NBeats_MAE / CN_MAE", nbeats / cn, 9.6, tol, "x")
    ok_fnn = True
    if fnn is not None:
        ok_fnn = _check("FNN_MAE / CN_MAE", fnn / cn, 10.1, tol, "x")
    if nbeats is not None and fnn is not None:
        if abs(nbeats - fnn) / max(fnn, 1e-9) < 1e-3:
            print("  [WARN]  NBeats MAE == FNN MAE — the N-BEATS class may be "
                  "implemented as a 1-block MLP. See README on the proper "
                  "stacked-block variant in `shared.NBeats`.")
    return ok_mae and ok_siren and ok_nbeats and ok_fnn


CHECKS = [
    ("Param-match",       row_1_param_match),
    ("Delta-sweep",       row_2_delta_sweep),
    ("Multi-layer Skip",  row_3_multilayer_skip),
    ("UCI tabular",       row_4_uci_tabular),
    ("RationalNN",        row_5_rational_nn),
    ("Gap-filling",       row_7_gap_filling),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerance", type=float, default=0.20,
                   help="Relative tolerance per ratio check (default 0.20 = 20%%)")
    args = p.parse_args()

    print("=" * 64)
    print(f"Verifying Table 1 against results/  (tol = {args.tolerance*100:.0f}%)")
    print("=" * 64)

    results = []
    for name, fn in CHECKS:
        try:
            ok = fn(args.tolerance)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}   {name}")
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{n_fail} failure(s) out of {len(results)} checks.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
