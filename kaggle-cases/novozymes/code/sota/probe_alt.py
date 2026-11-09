#!/usr/bin/env python3
"""Cheap probes of orthogonal hypotheses:
  * are the raw input columns (position, seq_id) predictive on their own?
  * does a different combination rule beat the linear rank average?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/repo/sota")
from ens_lib import build_submission, load_components, score  # noqa: E402

EXTRA = {
    "mutsasa": "/repo/outputs/components/submission_mutsasa.csv",
    "dsasa": "/repo/outputs/components/submission_dsasa.csv",
    "contacts": "/repo/outputs/components/submission_contacts.csv",
    "dcont": "/repo/outputs/components/submission_dcont.csv",
    "nshell": "/repo/outputs/components/submission_nshell.csv",
}
FIT = {"sasa": 0.0, "lgbm": 0.099, "thermo": 0.035, "bfac": 0.165, "rosetta": 0.258,
       "mutsasa": 0.031, "dsasa": 0.033, "contacts": 0.146, "dcont": 0.085,
       "nshell": 0.148}

PAPER = Path("/paper_data")
tmp = Path("/tmp/probe_alt.csv")


def write(vals, name):
    sample = pd.read_csv(PAPER / "sample_submission.csv")
    pd.DataFrame({"seq_id": sample.seq_id.to_numpy(), "tm": np.asarray(vals, float)}).to_csv(tmp, index=False)
    rho = score(tmp)
    print("%-40s %.5f" % (name, rho), flush=True)
    return rho


def main():
    test = pd.read_csv(PAPER / "test.csv")
    ids = pd.read_csv(PAPER / "sample_submission.csv").seq_id.to_numpy()
    # mutation position, recomputed exactly as the pipeline does
    WT = ("VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK"
          "GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSK"
          "PQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK")
    pos = {}
    for r in test.itertuples(index=False):
        p = 0
        for p, (a, b) in enumerate(zip(r.protein_sequence, WT)):
            if a != b:
                break
        pos[r.seq_id] = p
    df = pd.DataFrame({"seq_id": ids})
    df["position"] = df.seq_id.map(pos)

    print("--- raw input columns as single components ---")
    for col in ("position", "seq_id"):
        v = df[col].to_numpy(float)
        write(v, f"{col} (+)")
        write(-v, f"{col} (-)")

    comps = load_components({k: Path(v) for k, v in EXTRA.items()})
    rho_lin = score(build_submission(comps, FIT, tmp))
    print("\n--- combination rules at the fitted 10-component weights ---")
    print("%-40s %.5f" % ("linear rank average (current)", rho_lin), flush=True)

    n = len(ids)
    R = {k: rankdata(comps[k].to_numpy()) for k in FIT}
    wsum = sum(FIT.values())

    # logarithmic opinion pool: sum_k w_k * log(1 - (r_k - 0.5)/n)
    for eps in (0.0, 0.02, 0.05):
        p = {k: np.clip((n + 1 - R[k]) / n, 1e-9, 1.0) * (1 - eps) + eps for k in FIT}
        s = sum(FIT[k] * np.log(p[k]) for k in FIT)
        write(s, f"log opinion pool (eps={eps})")

    # power transform of the normalised rank before averaging
    for pw in (0.5, 2.0, 3.0):
        s = sum(FIT[k] * np.sign(R[k] - (n + 1) / 2) * np.abs(R[k] - (n + 1) / 2) ** pw
                for k in FIT)
        write(s, f"power p={pw} linear pool")

    # normal scores (van der Waerden) instead of raw ranks
    from scipy.stats import norm
    z = {k: norm.ppf((R[k] - 0.5) / n) for k in FIT}
    s = sum(FIT[k] * z[k] for k in FIT)
    write(s, "van der Waerden (normal scores) pool")
    print(f"(weights sum {wsum:.3f})")


if __name__ == "__main__":
    main()
