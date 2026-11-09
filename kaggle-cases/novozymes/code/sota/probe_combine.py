#!/usr/bin/env python3
"""Iteration-5 probes: refinements of the pooling step that need no new component.

  A. leave-one-out positional shrinkage: pull each prediction toward the mean of the
     *other* substitutions at the same residue position (a James-Stein style
     denoiser exploiting the ~11.6 substitutions per position in this DMS).
  B. a global exponent on the pooled normal scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, "/repo/sota")
from ens_lib import transformed_ranks, load_components, score  # noqa: E402

EXTRA = {
    "mutsasa": "/repo/outputs/components/submission_mutsasa.csv",
    "dsasa": "/repo/outputs/components/submission_dsasa.csv",
    "contacts": "/repo/outputs/components/submission_contacts.csv",
    "dcont": "/repo/outputs/components/submission_dcont.csv",
    "nshell": "/repo/outputs/components/submission_nshell.csv",
}
FIT = {"sasa": 0.013, "lgbm": 0.110, "thermo": 0.034, "bfac": 0.152, "rosetta": 0.268,
       "mutsasa": 0.043, "dsasa": 0.032, "contacts": 0.113, "dcont": 0.083, "nshell": 0.152}
PAPER = Path("/paper_data")
WT = ("VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK"
      "GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSK"
      "PQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK")


def write(vals, label):
    sample = pd.read_csv(PAPER / "sample_submission.csv")
    p = Path("/tmp/probe_combine.csv")
    pd.DataFrame({"seq_id": sample.seq_id.to_numpy(), "tm": np.asarray(vals, float)}).to_csv(p, index=False)
    rho = score(p)
    print("%-42s %.5f" % (label, rho), flush=True)
    return rho


def main():
    ids = pd.read_csv(PAPER / "sample_submission.csv").seq_id.to_numpy()
    test = pd.read_csv(PAPER / "test.csv").set_index("seq_id")
    pos = {}
    for sid, seq in test.protein_sequence.items():
        p = 0
        for p, (a, b) in enumerate(zip(seq, WT)):
            if a != b:
                break
        pos[sid] = p
    posv = np.array([pos[s] for s in ids])

    comps = load_components({k: Path(v) for k, v in EXTRA.items()})
    wsum = sum(FIT.values())
    Z = {k: transformed_ranks(comps[k].to_numpy(), "norm") for k in FIT}
    base = sum(FIT[k] * Z[k] for k in FIT) / wsum
    rho0 = write(base, "current: normal-score pool (baseline)")
    print(f"  (ids={len(ids)}, positions={len(set(posv.tolist()))})")

    # ---- A. leave-one-out positional shrinkage ------------------------------
    groups = {}
    for i, p in enumerate(posv):
        groups.setdefault(int(p), []).append(i)
    loo = np.array(base, dtype=float)
    for p, idxs in groups.items():
        if len(idxs) < 4:
            continue
        s = base[idxs].sum()
        for i in idxs:
            loo[i] = base[i]  # placeholder; filled below
    # compute the LOO mean only where the group is big enough
    for p, idxs in groups.items():
        if len(idxs) < 4:
            continue
        s = base[idxs].sum()
        for i in idxs:
            loo[i] = (s - base[i]) / (len(idxs) - 1)

    for lam in (0.05, 0.10, 0.20, 0.30, 0.40):
        v = (1 - lam) * base + lam * loo
        write(v, f"A. positional LOO shrinkage lam={lam:.2f}")

    # ---- B. global exponent on the pooled scores ----------------------------
    for q in (0.8, 1.2, 1.5, 2.0):
        v = np.sign(base) * np.abs(base) ** q
        write(v, f"B. pooled-score exponent q={q}")

    # ---- C. rank-average of the *ensemble* with its own position mean -------
    # (no LOO restriction; includes the row itself -> stronger smoothing)
    pm = np.array([np.mean(base[groups[int(p)]]) for p in posv])
    for lam in (0.10, 0.20):
        write((1 - lam) * base + lam * pm, f"C. positional mean shrinkage lam={lam:.2f}")


if __name__ == "__main__":
    main()
