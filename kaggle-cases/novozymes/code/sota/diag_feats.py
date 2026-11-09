#!/usr/bin/env python3
"""Label-free diagnostic: how do the new structural features relate to the five
existing (already-scored) components and to their fitted consensus?

Nothing here touches test_labels.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sf = pd.read_csv("/repo/outputs/struct_features.csv")
c = "/repo/outputs/components/"
comps = {
    "sasa": pd.read_csv(c + "submission_WT_sasa_alpha.csv").tm.values,
    "lgbm": pd.read_csv(c + "submission_lgbm.csv").tm.values,
    "bfac": pd.read_csv(c + "submission_diff_bfactors_alpha.csv").tm.values,
    "thermo": pd.read_csv("/paper_data/components/thermonet2_submission.csv").tm.values,
    "rosetta": pd.read_csv("/paper_data/components/rosetta_scores.csv").tm.values,
}
w = {"sasa": 0.148, "lgbm": 0.189, "thermo": 0.184, "bfac": 0.155, "rosetta": 0.324}
cons = sum(w[k] * rankdata(v) for k, v in comps.items()) / sum(w.values())

skip = {"seq_id", "position", "AA1", "AA2", "struct_res_at_site", "pdb_found"}
cand = [x for x in sf.columns if x not in skip]

hdr = "%-22s " % "feature" + " ".join("%8s" % k for k in comps) + "  consensus"
print(hdr)
rows = []
for col in cand:
    v = sf[col].values.astype(float)
    m = ~np.isnan(v)
    r = [spearmanr(v[m], comps[k][m]).correlation for k in comps]
    rc = spearmanr(v[m], np.asarray(cons)[m]).correlation
    rows.append((col, rc, r))
for col, rc, r in sorted(rows, key=lambda x: -abs(x[1])):
    print("%-22s " % col + " ".join("%8.3f" % x for x in r) + "  %8.3f" % rc)

print("\nmax |rank corr| among candidate structural features:")
X = sf[cand].apply(lambda s: rankdata(s.fillna(s.median())))
cc = X.corr()
np.fill_diagonal(cc.values, 0)
order = np.dstack(np.unravel_index(np.argsort(-np.abs(cc.values), axis=None), cc.shape))[0]
seen = set()
shown = 0
for i, j in order:
    a, b = sorted([cand[i], cand[j]])
    if (a, b) in seen:
        continue
    seen.add((a, b))
    print("  %-20s %-20s %+.3f" % (a, b, cc.values[i, j]))
    shown += 1
    if shown >= 12:
        break

print("\nAA2 composition counts (top 12):")
print(sf.AA2.value_counts().head(12).to_dict())
print("distinct AA1:", sf.AA1.nunique(), "distinct positions:", sf.position.nunique())
