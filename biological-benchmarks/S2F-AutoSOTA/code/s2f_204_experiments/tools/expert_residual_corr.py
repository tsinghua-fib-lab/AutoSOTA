"""Measure per-expert skill and inter-expert RESIDUAL correlation.

The fusion is a log-linear pool, so what governs the optimal weights is not
each expert's skill alone but how correlated their errors are. Nothing in the
project ever measured that - F7's "errors are not decorrelated enough" was
inferred from the ensemble landing at the member mean. This measures it.
"""
import glob, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr

EXPERTS = ["s2f", "masked", "wt", "pssm"]

def run(dump):
    files = sorted(glob.glob(os.path.join(dump, "*.npz")))
    skill = {e: [] for e in EXPERTS}
    pairs, n_cov = {}, 0
    for f in files:
        z = np.load(f)
        t = z["target"].astype(np.float64)
        m = np.isfinite(t)
        if m.sum() < 20:
            continue
        t = t[m]
        cov = z["cov"].astype(np.float64)[m]
        # PSSM only defined where the MSA covers the site.
        use_pssm = cov.min() >= 1 - 1e-9
        n_cov += use_pssm
        tr = rankdata(t)
        res = {}
        for e in EXPERTS:
            if e == "pssm" and not use_pssm:
                continue
            v = z[e].astype(np.float64)[m]
            skill[e].append(spearmanr(v, t).correlation)
            # residual in rank space = what the expert gets wrong
            r = rankdata(v) - tr
            res[e] = (r - r.mean()) / (r.std() + 1e-12)
        for a in res:
            for b in res:
                if a < b:
                    pairs.setdefault((a, b), []).append(float(np.mean(res[a] * res[b])))
    print("== %s (%d assays, PSSM-covered %d) ==" % (os.path.basename(dump), len(skill["s2f"]), n_cov))
    print("  -- expert skill (mean per-assay Spearman) --")
    for e in EXPERTS:
        if skill[e]:
            print("     %-8s %.4f  (n=%d)" % (e, np.mean(skill[e]), len(skill[e])))
    print("  -- residual correlation (0 = complementary, 1 = redundant) --")
    for k in sorted(pairs, key=lambda k: -np.mean(pairs[k])):
        print("     %-8s vs %-8s %+.3f" % (k[0], k[1], np.mean(pairs[k])))
    print()

for d in sys.argv[1:]:
    run(d)
