#!/usr/bin/env python3
"""IDEA-604: per-position conservation gating of the evolutionary term.

Recomputes the per-position PSSM log-odds contributions from the cached
frequencies (dataset/DMS_pssm_cache) + the assay mutation CSVs, so any
per-position gate can be tested offline on the sel-80 component dumps without
re-running the model.

Gate: g_i = f(H_i) with H_i the normalized column entropy of the weighted
frequencies (0 = fully conserved, 1 = uniform). Gated row score:
    sum_i g_i * (logf(mt,i) - logf(wt,i))
which replaces the plain PSSM row score in the (adaptive-e) blend.

Anchor: the ungated recomputed row sums must reproduce the dumped pssm arrays
exactly (guards against row misalignment).

Usage: python tools/pssm_gate_fit.py <dump_dir>
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy.stats import rankdata

DUMPS = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F/results/component_dumps/sel80_iter2"
CACHE = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F/dataset/DMS_pssm_cache"
SUBST = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F/dataset/DMS_ProteinGym_substitutions"


def spearman(p, t):
    p, t = rankdata(p), rankdata(t)
    cov = (p * t).mean() - p.mean() * t.mean()
    return float(cov / (p.std() * t.std() + 1e-10))


def e_603(neff):
    return float(np.clip(1.0 / (1.0 + np.exp(-8.0 * (np.log10(neff) - 1.47))),
                         0.0, 1.0))


def per_position_rows(assay_id, mutations_csv):
    """Per-row per-site PSSM log-odds for one assay, from cached logf.

    Rows are ordered EXACTLY like evaluate.py::load_dataset: sorted by the
    tuple (0-indexed mutation sites, mutant strings) — the dumped arrays are
    in that sorted-site order, not raw CSV order.
    Returns (row_deltas: list of per-row [(col_idx, delta)] lists,
             n_sites list, positions, logf).
    """
    z = np.load(os.path.join(CACHE, assay_id + ".npz"))
    positions = z["positions"]
    logf = z["logf"]
    pos2row = {int(p): i for i, p in enumerate(positions)}
    AA2IDX = {aa: i for i, aa in
              enumerate(sorted("ACDEFGHIKLMNPQRSTVWY"))}
    raw = []
    with open(mutations_csv) as f:
        reader = csv.reader(f)
        fields = next(reader)
        for values in reader:
            muts = None
            for field, value in zip(fields, values):
                if field == "mutant":
                    muts = value.split(":")
            if muts is None:
                continue
            sites = tuple(int(m[1:-1]) - 1 for m in muts)
            raw.append((sites, muts))
    raw = sorted(raw)  # load_dataset order: sorted by (sites, muts)
    rows, n_sites = [], []
    for sites, muts in raw:
        row = []
        for m in muts:
            p = int(m[1:-1])
            k = pos2row.get(p)
            if k is None:
                continue
            mt, wt = m[-1], m[0]
            if mt in AA2IDX and wt in AA2IDX:
                row.append((k, logf[k, AA2IDX[mt]] - logf[k, AA2IDX[wt]]))
        rows.append(row)
        n_sites.append(len(muts))
    return rows, n_sites, positions, logf


def entropy_gates(logf):
    """Normalized column entropy (0 conserved .. 1 uniform) per position."""
    f = np.exp(logf)
    H = -(f * logf).sum(axis=-1) / np.log(20.0)
    return np.clip(H, 0.0, 1.0)


def gated_row(row, gates, gfn):
    return float(np.sum(gfn(gates[k]) * d for k, d in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", default=DUMPS, nargs="?")
    args = ap.parse_args()

    # sel-80 assay ids and their mutation CSVs (from DMS_substitutions.csv)
    meta = {}
    with open(os.path.join(SUBST, "DMS_substitutions.csv")) as f:
        for r in csv.DictReader(f):
            meta[r["DMS_id"]] = r

    assays = []
    max_anchor_err = 0.0
    for f in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        aid = os.path.basename(f)[:-4]
        z = np.load(f)
        a = {k: z[k].astype(np.float64) for k in z.files}
        rows, n_sites, positions, logf = per_position_rows(
            aid, os.path.join(SUBST, meta[aid]["DMS_filename"]))
        if len(rows) != len(a["pssm"]):
            print("ROW COUNT MISMATCH %s: %d vs %d" %
                  (aid, len(rows), len(a["pssm"])))
            continue
        # anchor: ungated recomputed sums must equal dumped pssm
        sums = np.array([float(np.sum(d for _, d in r)) for r in rows])
        cov_check = np.array([len(r) / n for r, n in zip(rows, n_sites)])
        err = np.abs(sums - a["pssm"]).max()
        err2 = np.abs(cov_check - a["cov"]).max()
        max_anchor_err = max(max_anchor_err, err, err2)
        gates = entropy_gates(logf)
        m = np.isfinite(a["target"])
        assays.append(dict(
            aid=aid, fused=0.70 * a["s2f"] + 0.30 * a["wt"],
            cov=(a["cov"] >= 1 - 1e-9).astype(np.float64),
            target=a["target"][m], rows=rows, gates=gates,
            e=e_603(float(np.load(os.path.join(CACHE, aid + ".npz"))["neff"])),
            pssm=a["pssm"][m]))
    print("[gate] %d assays loaded; max anchor deviation vs dump: %.2e"
          % (len(assays), max_anchor_err))

    def mean_score(gfn):
        out = []
        for a in assays:
            g = np.array([gated_row(r, a["gates"], gfn) for r in a["rows"]])
            final = ((1 - a["e"] * a["cov"]) * a["fused"]
                     + a["e"] * a["cov"] * g)
            m = np.isfinite(a["target"])
            out.append(spearman(final[m], a["target"][m]))
        return float(np.mean(out))

    families = {
        "ungated (603 baseline)": lambda h: 1.0,
        "g=(1-H)^0.5": lambda h: (1 - h) ** 0.5,
        "g=(1-H)^1": lambda h: 1 - h,
        "g=(1-H)^2": lambda h: (1 - h) ** 2,
        "g=(1-H)^4": lambda h: (1 - h) ** 4,
        "hard g=1[H<0.30]": lambda h: (h < 0.30).astype(float),
        "hard g=1[H<0.50]": lambda h: (h < 0.50).astype(float),
        "hard g=1[H<0.70]": lambda h: (h < 0.70).astype(float),
        "soft clip g=clip((1-H)/0.3,0,1)": lambda h: np.clip((1 - h) / 0.3, 0, 1),
        "soft clip g=clip((1-H)/0.5,0,1)": lambda h: np.clip((1 - h) / 0.5, 0, 1),
    }
    base = None
    for name, gfn in families.items():
        s = mean_score(gfn)
        if base is None:
            base = s
        print("  %-30s %.4f (%+.4f)" % (name, s, s - base))
    print("[gate] adoption threshold: +0.002 over ungated")


if __name__ == "__main__":
    main()
