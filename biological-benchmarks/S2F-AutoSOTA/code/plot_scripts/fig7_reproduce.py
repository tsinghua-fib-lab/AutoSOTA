#!/usr/bin/env python3
"""Standalone reproduction of fig7_structure.png + the per-site results table.

Self-contained on purpose: everything fig7 needs (champion fusion, row->mutation
mapping, per-site statistic, colour scale) is in this one file so the colours and
the statistic can be checked without reading the rest of the report code.

Case study: NCAP_I34A1_Doud_2015 - influenza A nucleoprotein, all 498 residues
deep-mutationally scanned.

Outputs
  fig7_structure.png                        the figure
  NCAP_I34A1_Doud_2015_per_site.csv         the per-site table
  NCAP_I34A1.pdb                            copy of the structure actually used

Run:  python fig7_reproduce.py
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.stats import spearmanr

# ----------------------------------------------------------------- config
S3F = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F"
OUT = os.path.dirname(os.path.abspath(__file__))

CASE = "NCAP_I34A1_Doud_2015"
PDB_ID = "NCAP_I34A1"
CHAIN = "A"

DUMP = os.path.join(S3F, "results", "component_dumps", "full204_ens5", CASE + ".npz")
PCACHE = os.path.join(S3F, "dataset", "DMS_pssm_cache", CASE + ".npz")
DMS_CSV = os.path.join(S3F, "dataset", "DMS_ProteinGym_substitutions", CASE + ".csv")
PDB = os.path.join(S3F, "dataset", "ProteinGym_AF2_structures", PDB_ID + ".pdb")

# champion fusion weights (config/evaluate/s2f.yaml)
W_MASKED, C_WT = 0.20, 0.30
# evolutionary gate  e(Neff) = sigmoid(a*(log10 Neff - b))
E_A, E_B = 8.0, 1.47

MIN_MUTANTS = 8          # sites with fewer assayed mutants are dropped
PCTL_SATURATE = 92       # colour scale saturation percentile

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
BLUE, ORANGE = "#2E5EAA", "#E8833A"


# ----------------------------------------------------------------- data
def champion_score(d):
    """Exact final scoring stack, in log-probability space.

    fused = (1-c)*((1-w)*S2F + w*ESM1v_masked) + c*ESM1v_wt
    score = (1 - e*cov)*fused + e*cov*PSSM_log_odds

    `cov` is 1 where the MSA covers the mutated site. Uncovered rows fall back
    to the two-expert score rather than being zeroed, which would corrupt the
    ranking.
    """
    w, c = W_MASKED, C_WT
    fused = (1 - c) * ((1 - w) * d["s2f"] + w * d["masked"]) + c * d["wt"]
    cov = (d["cov"] >= 1 - 1e-9).astype(np.float64)
    neff = float(np.load(PCACHE)["neff"])
    e = float(np.clip(1 / (1 + np.exp(-E_A * (np.log10(neff) - E_B))), 0, 1))
    return (1 - e * cov) * fused + e * cov * d["pssm"], neff, e


def load_rows():
    """Join the per-row score dump back onto (site, wt_aa, mt_aa).

    The dump arrays are in the order load_dataset produced, which sorts
    mutations by (site, mutant). That is verified against DMS_score here rather
    than assumed - a row-order mismatch would silently scramble every per-site
    number downstream.
    """
    z = np.load(DUMP)
    d = {k: z[k].astype(np.float64) for k in z.files}

    df = pd.read_csv(DMS_CSV)
    df = df[~df["mutant"].str.contains(":")].copy()      # single mutants only
    df["site"] = df["mutant"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["site", "mutant"], kind="stable").reset_index(drop=True)

    assert len(df) == len(d["target"]), (
        "row count mismatch: csv %d vs dump %d" % (len(df), len(d["target"])))
    resid = np.abs(df["DMS_score"].values - d["target"]).max()
    assert resid < 1e-4, "row order mismatch (max |DMS_score diff| = %.3g)" % resid

    df["wt_aa"] = df["mutant"].str[0]
    df["mt_aa"] = df["mutant"].str[-1]
    df["s2f"] = d["s2f"]
    ours, neff, e = champion_score(d)
    df["ours"] = ours
    print("  row/mutation mapping verified (max |diff| = %.2e)" % resid)
    print("  assay Neff = %.2f  ->  evolutionary gate e = %.4f" % (neff, e))
    return df


def read_pdb_ca(path, chain):
    """CA coordinates and residue identity, keyed by PDB residue number."""
    ca, aa, plddt = {}, {}, {}
    with open(path) as fh:
        for ln in fh:
            if not ln.startswith("ATOM") or ln[21] != chain:
                continue
            if ln[12:16].strip() != "CA":
                continue
            r = int(ln[22:26])
            ca[r] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            aa[r] = AA3TO1.get(ln[17:20].strip(), "X")
            plddt[r] = float(ln[60:66])
    return ca, aa, plddt


# ----------------------------------------------------------------- table
def per_site_table(df, pdb_aa, pdb_plddt):
    recs = []
    for site, g in df.groupby("site"):
        n = len(g)
        wt = g["wt_aa"].iloc[0]
        rho_s2f = spearmanr(g["s2f"], g["DMS_score"]).correlation
        rho_ours = spearmanr(g["ours"], g["DMS_score"]).correlation
        recs.append({
            "site": int(site),
            "wt_aa_dms": wt,
            "wt_aa_pdb": pdb_aa.get(int(site), ""),
            "wt_match": pdb_aa.get(int(site), "") == wt,
            "plddt": pdb_plddt.get(int(site), np.nan),
            "n_mutants": n,
            "rho_s2f": rho_s2f,
            "rho_ours": rho_ours,
            "delta": (rho_ours - rho_s2f
                      if np.isfinite(rho_ours) and np.isfinite(rho_s2f) else np.nan),
            "used_in_figure": bool(n >= MIN_MUTANTS
                                   and np.isfinite(rho_ours) and np.isfinite(rho_s2f)),
        })
    return pd.DataFrame(recs).sort_values("site").reset_index(drop=True)


# ----------------------------------------------------------------- figure
def make_figure(tab, ca):
    used = tab[tab["used_in_figure"]]
    per = dict(zip(used["site"], used["delta"]))
    res = sorted(ca)
    xyz = np.array([ca[r] for r in res])

    v = float(np.percentile(np.abs(list(per.values())), PCTL_SATURATE))
    norm = plt.Normalize(-v, v)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.12)

    ax = fig.add_subplot(gs[0], projection="3d")
    segs, cols = [], []
    for i in range(len(res) - 1):
        segs.append([xyz[i], xyz[i + 1]])
        vv = [x for x in (per.get(res[i]), per.get(res[i + 1])) if x is not None]
        cols.append(cmap(norm(np.mean(vv))) if vv else (0.85, 0.85, 0.85, 1.0))
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=2.6))
    mn, mx = xyz.min(0), xyz.max(0)
    ctr, rad = (mn + mx) / 2, (mx - mn).max() / 2 * 0.62
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
        setter(c - rad, c + rad)
    ax.set_axis_off()
    ax.set_title("%s \u2014 AlphaFold backbone\ncoloured by per-site change in "
                 "rank correlation" % CASE, fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=-0.02, shrink=0.62,
                 orientation="horizontal",
                 label="\u0394 per-site \u03c1 (ours \u2212 S2F)")

    ax2 = fig.add_subplot(gs[1])
    ks = list(used["site"]); vs = list(used["delta"])
    ax2.bar(ks, vs, color=[ORANGE if x > 0 else BLUE for x in vs], width=1.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("residue position")
    ax2.set_ylabel("\u0394 per-site Spearman (ours \u2212 S2F)")
    ax2.set_title("Improvement along the sequence: %d/%d assayed sites improve"
                  % (sum(1 for x in vs if x > 0), len(vs)), fontsize=9)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    out = os.path.join(OUT, "fig7_structure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  colour scale saturated at +/- %.4f (%dth pctl of |delta|)"
          % (v, PCTL_SATURATE))
    return out


if __name__ == "__main__":
    print("== %s ==" % CASE)
    df = load_rows()
    ca, pdb_aa, pdb_plddt = read_pdb_ca(PDB, CHAIN)
    print("  PDB chain %s: %d CA atoms, residues %d..%d"
          % (CHAIN, len(ca), min(ca), max(ca)))

    tab = per_site_table(df, pdb_aa, pdb_plddt)
    nm = (~tab["wt_match"]).sum()
    print("  WT residue identity check: %d/%d sites agree between DMS and PDB%s"
          % (tab["wt_match"].sum(), len(tab), "" if nm == 0 else "  <-- MISMATCH"))

    csv_out = os.path.join(OUT, CASE + "_per_site.csv")
    tab.to_csv(csv_out, index=False, float_format="%.6f")
    print("  wrote %s  (%d sites)" % (os.path.basename(csv_out), len(tab)))

    pdb_out = os.path.join(OUT, PDB_ID + ".pdb")
    if os.path.abspath(pdb_out) != os.path.abspath(PDB):
        shutil.copy2(PDB, pdb_out)
        print("  copied %s" % os.path.basename(pdb_out))

    print("  wrote %s" % os.path.basename(make_figure(tab, ca)))
    u = tab[tab["used_in_figure"]]
    print("\n  summary: %d sites in figure, %d improved (%.1f%%), "
          "mean delta %+.4f, median %+.4f"
          % (len(u), (u["delta"] > 0).sum(), 100 * (u["delta"] > 0).mean(),
             u["delta"].mean(), u["delta"].median()))
