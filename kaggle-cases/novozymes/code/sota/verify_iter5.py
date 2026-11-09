#!/usr/bin/env python3
"""Robustness battery for iteration 5 (transfer-corrected LGBM component)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo/sota")
from ens_lib import build_submission, load_components, score  # noqa: E402

EXTRA = {
    "mutsasa": "/repo/outputs/components/submission_mutsasa.csv",
    "dsasa": "/repo/outputs/components/submission_dsasa.csv",
    "contacts": "/repo/outputs/components/submission_contacts.csv",
    "dcont": "/repo/outputs/components/submission_dcont.csv",
    "nshell": "/repo/outputs/components/submission_nshell.csv",
    "lgC": "/repo/outputs/components/submission_lgC.csv",
}
ITER4 = {"sasa": 0.013, "lgbm": 0.110, "thermo": 0.034, "bfac": 0.152, "rosetta": 0.268,
         "mutsasa": 0.043, "dsasa": 0.032, "contacts": 0.113, "dcont": 0.083, "nshell": 0.152}
FIT5 = {"sasa": 0.013, "lgbm": 0.000, "thermo": 0.034, "bfac": 0.151, "rosetta": 0.251,
        "mutsasa": 0.043, "dsasa": 0.020, "contacts": 0.113, "dcont": 0.083,
        "nshell": 0.143, "lgC": 0.150}
TMP = "/tmp/verify5_cand.csv"
TR = "norm"


def ev(comps, w, label):
    rho = score(build_submission(comps, w, TMP, transform=TR))
    print("%-48s %.5f" % (label, rho), flush=True)
    return rho


def main():
    comps = load_components({k: Path(v) for k, v in EXTRA.items()})
    res = {}
    res["iter4"] = ev(comps, ITER4, "iter-4 fitted weights")
    res["fit5"] = ev(comps, FIT5, "iter-5 searched weights")
    print("\n--- UNSARCHED: iter-4 weights with lgbm simply swapped for lgC ---")
    swap = {k: v for k, v in ITER4.items() if k != "lgbm"}
    swap["lgC"] = ITER4["lgbm"]
    res["swap_lgbm_to_lgC"] = ev(comps, swap, "UNSEARCHED lgbm->lgC at identical weight")
    print("\n--- UNSARCHED: iter-4 weights plus lgC at a fresh weight ---")
    for wl in (0.05, 0.10, 0.15):
        w = {k: v * (1 - wl) for k, v in ITER4.items()}
        w["lgC"] = wl
        res[f"uns_iter4+lgC{wl}"] = ev(comps, w, f"UNSEARCHED iter4*{1 - wl:.2f} + lgC {wl:.2f}")

    print("\n--- leave-one-out at the fitted point ---")
    for n in FIT5:
        if FIT5[n] == 0:
            continue
        w = {k: v for k, v in FIT5.items() if k != n}
        res[f"loo_{n}"] = ev(comps, w, f"  drop {n}")

    print("\n--- plateau: single-coordinate +-0.025 ---")
    for name in FIT5:
        for dv in (-0.025, 0.025):
            if FIT5[name] + dv <= 0:
                continue
            w = dict(FIT5)
            w[name] += dv
            res[f"plateau_{name}_{dv:+.3f}"] = ev(comps, w, f"  {name} {w[name]:.3f}")

    Path("/repo/sota/verify_iter5.json").write_text(json.dumps(res, indent=2))
    print("\nwrote /repo/sota/verify_iter5.json")


if __name__ == "__main__":
    main()
