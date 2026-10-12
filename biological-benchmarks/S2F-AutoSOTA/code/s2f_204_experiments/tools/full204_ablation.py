#!/usr/bin/env python3
"""IDEA-702: full-204 component ablation table from the champion dump.

Rows (all at champion backbone ESM-1v-ens5, offline exact recombination):
  1. S2F alone            (w=0, c=0, no PSSM)
  2. + ESM-1v masked      (w=0.50, c=0, no PSSM)   [ESM-1v published recipe]
  3. + ESM-1v WT          (w=0.20, c=0.30, no PSSM)
  4. + PSSM flat          (w=0.20, c=0.30, e=0.25)
  5. + PSSM Neff-adaptive (w=0.20, c=0.30, e=sigmoid)  == champion
Also prints per-assay dev-40 and sel-80 intersections for cross-checking.
"""
import os
import sys

import numpy as np

S3F = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F"
sys.path.insert(0, S3F)
from tools.simplex_search import load_assays, spearman_np  # noqa: E402

CACHE = os.path.join(S3F, "dataset", "DMS_pssm_cache")


def row(assays, w, c, e_mode):
    sps = []
    for name, d in assays.items():
        fused = (1 - c) * ((1 - w) * d["s2f"] + w * d["masked"]) + c * d["wt"]
        cov = (d["cov"] >= 1 - 1e-9).astype(np.float64)
        if e_mode == "none":
            e = 0.0
        elif e_mode == "flat":
            e = 0.25
        else:
            z = np.load(os.path.join(CACHE, name + ".npz"))
            neff = float(z["neff"])
            e = float(np.clip(
                1.0 / (1.0 + np.exp(-8.0 * (np.log10(neff) - 1.47))), 0, 1))
        final = (1 - e * cov) * fused + e * cov * d["pssm"]
        t = d["target"]
        m = np.isfinite(t)
        if not m.all():
            final, t = final[m], t[m]
        sps.append(spearman_np(final, t))
    return float(np.mean(sps))


if __name__ == "__main__":
    a = load_assays(sys.argv[1])
    print("== %s (%d assays) ==" % (sys.argv[1], len(a)))
    print("  row1 S2F alone          : %.4f" % row(a, 0, 0, "none"))
    print("  row2 +masked (0.5,0)    : %.4f" % row(a, 0.5, 0, "none"))
    print("  row2 +masked (0.2,0)    : %.4f" % row(a, 0.2, 0, "none"))
    print("  row3 +WT (0.20,0.30)    : %.4f" % row(a, 0.20, 0.30, "none"))
    print("  row4 +PSSM flat e=.25   : %.4f" % row(a, 0.20, 0.30, "flat"))
    print("  row5 +PSSM adaptive     : %.4f" % row(a, 0.20, 0.30, "adaptive"))
