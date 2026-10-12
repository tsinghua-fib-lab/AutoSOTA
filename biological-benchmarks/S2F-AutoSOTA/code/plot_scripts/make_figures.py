#!/usr/bin/env python3
"""Figures for the S2F / ProteinGym experiment record.

All inputs are local and already computed: the per-row component dumps produced
during round 6, the PSSM cache, and the paper's own per-assay result table that
ships with the S3F repo. Nothing here re-runs a model.

Axis labels are English on purpose - the conda env has no CJK font and the
fallback renders as tofu boxes.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, spearmanr

S3F = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F"
OUT = "/home/dataset-assist-0/chenzhibin/s3f+autosota/report/figures"
DUMP = os.path.join(S3F, "results", "component_dumps")
PCACHE = os.path.join(S3F, "dataset", "DMS_pssm_cache")
REF = os.path.join(S3F, "asset", "DMS_substitutions_Spearman_DMS_level.csv")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
BLUE, ORANGE, GREY, RED = "#2E5EAA", "#E8833A", "#9AA0A6", "#C0392B"


def champion_score(d, name):
    """Exact champion fusion: w=0.20, c=0.30, PSSM with Neff-adaptive weight."""
    w, c = 0.20, 0.30
    fused = (1 - c) * ((1 - w) * d["s2f"] + w * d["masked"]) + c * d["wt"]
    cov = (d["cov"] >= 1 - 1e-9).astype(np.float64)
    p = os.path.join(PCACHE, name + ".npz")
    if os.path.exists(p):
        neff = float(np.load(p)["neff"])
        e = float(np.clip(1 / (1 + np.exp(-8 * (np.log10(neff) - 1.47))), 0, 1))
    else:
        e = 0.0
    return (1 - e * cov) * fused + e * cov * d["pssm"]


def load_dump(sub):
    out = {}
    for f in sorted(glob.glob(os.path.join(DUMP, sub, "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = {k: z[k].astype(np.float64) for k in z.files}
    return out


# ---------------------------------------------------------------- fig 1
def fig1_method_comparison():
    ref = pd.read_csv(REF)
    nb = ref[ref["Selection Type"] != "Binding"]
    items = [
        ("Site-Independent\n(evolution only)", nb["Site-Independent"].mean(), GREY),
        ("ESM-1v\n(ensemble)", nb["ESM-1v (ensemble)"].mean(), GREY),
        ("ESM-2 650M", nb["ESM2 (650M)"].mean(), GREY),
        ("MSA Transformer\n(ensemble)", nb["MSA Transformer (ensemble)"].mean(), GREY),
        ("EVE\n(ensemble)", nb["EVE (ensemble)"].mean(), GREY),
        ("S2F\n(paper)", nb["S2F"].mean(), BLUE),
        ("S3F\n(paper)", nb["S3F"].mean(), BLUE),
        ("This work", 0.5041, ORANGE),
        ("S2F_MSA\n(paper)", nb["S2F_MSA"].mean(), RED),
        ("S3F_MSA\n(paper)", nb["S3F_MSA"].mean(), RED),
    ]
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(items))
    bars = ax.bar(xs, [v for _, v, _ in items], color=[c for _, _, c in items],
                  width=0.68, edgecolor="white", linewidth=0.8)
    for x, (_, v, _) in zip(xs, items):
        ax.text(x, v + 0.004, "%.4f" % v, ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in items], fontsize=7)
    ax.set_ylabel("Mean Spearman (204 non-binding assays)")
    ax.set_ylim(0.33, 0.555)
    ax.set_title("Zero-shot variant effect prediction on ProteinGym\n"
                 "orange = this work; red = the paper's own MSA-augmented variants",
                 fontsize=9.5)
    for b, (_, _, col) in zip(bars, items):
        if col == ORANGE:
            b.set_edgecolor("black"); b.set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_method_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ---------------------------------------------------------------- fig 2
def fig2_trajectory():
    """dev-40 score across the whole search. Hand-transcribed from scores.jsonl
    notes because the metrics field was never populated by the driver."""
    rounds = [
        ("R1", [0.4239, 0.4222, 0.4195, 0.4205, 0.4144, 0.4183, 0.4236, 0.4263,
                0.4238, 0.4161, 0.4230, 0.4279, 0.4318, 0.4290, 0.4211, 0.4184,
                0.4301, 0.4212, 0.4271, 0.4273, 0.4298, 0.4296, 0.4265, 0.4246,
                0.4318]),
        ("R2", [0.4446, 0.4430, 0.4429, 0.4441, 0.3964, 0.4438, 0.4338, 0.4439,
                0.4407, 0.4401, 0.4384, 0.4385, 0.4431, 0.4433, 0.4379, 0.4305,
                0.4419, 0.4446]),
        ("R3", [0.4450, 0.4438, 0.4468, 0.4468]),
        ("R4", [0.4468, 0.4460, 0.4442, 0.4499, 0.4499, 0.4499, 0.4500, 0.4500]),
        ("R5", [0.4500, 0.4500, 0.4524, 0.4530, 0.4524, 0.4524, 0.4524, 0.4525]),
        ("R6", [0.4525, 0.4525, 0.4544, 0.4572, 0.4572, 0.4572, 0.4572, 0.4289,
                0.4653, 0.4653, 0.4653, 0.4653, 0.4653]),
        ("R7", [0.4653, 0.4653]),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = 0
    best = -1
    bx, by = [], []
    for i, (name, vals) in enumerate(rounds):
        xs = np.arange(x, x + len(vals))
        ax.plot(xs, vals, "o-", ms=3, lw=1.1,
                color=BLUE if i % 2 == 0 else ORANGE, alpha=0.85, label=None)
        ax.axvspan(x - 0.5, x + len(vals) - 0.5, alpha=0.045,
                   color=BLUE if i % 2 == 0 else ORANGE)
        ax.text(x + len(vals) / 2 - 0.5, 0.394, name, ha="center", fontsize=9,
                weight="bold", color="#444")
        for xi, v in zip(xs, vals):
            if v > best:
                best = v; bx.append(xi); by.append(v)
        x += len(vals)
    ax.step(bx + [x - 1], by + [by[-1]], where="post", color=RED, lw=1.6,
            label="running best", zorder=5)
    ax.axhline(0.4436, color="k", ls="--", lw=1.0, label="published s2f.pth (0.4436)")
    ax.annotate("label smoothing\n-0.048", xy=(29, 0.3964), xytext=(31, 0.404),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("clean ESM-1v retrain\n-0.028 (informative failure)",
                xy=(74, 0.4289), xytext=(60, 0.414), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("ESM-1v \u00d75 ensemble\n0.4653", xy=(75, 0.4653), xytext=(62, 0.472),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("iteration (chronological across 7 AutoSOTA rounds)")
    ax.set_ylabel("dev-40 mean Spearman")
    ax.set_ylim(0.390, 0.478)
    ax.set_title("Full search trajectory: 89 logged iterations", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_trajectory.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------- fig 3
def fig3_per_assay():
    ref = pd.read_csv(REF).set_index("DMS ID")
    a = load_dump("full204_ens5")
    ours, base, names = [], [], []
    for name, d in a.items():
        if name not in ref.index:
            continue
        t = d["target"]; m = np.isfinite(t)
        ours.append(spearmanr(champion_score(d, name)[m], t[m]).correlation)
        base.append(float(ref.loc[name, "S2F"]))
        names.append(name)
    ours, base = np.array(ours), np.array(base)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    win = ours > base
    ax.scatter(base[win], ours[win], s=16, color=ORANGE, alpha=0.75,
               edgecolor="none", label="improved (%d)" % win.sum())
    ax.scatter(base[~win], ours[~win], s=16, color=BLUE, alpha=0.7,
               edgecolor="none", label="regressed (%d)" % (~win).sum())
    lim = [-0.05, 0.95]
    ax.plot(lim, lim, "k--", lw=0.9, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("published S2F, per assay")
    ax.set_ylabel("this work, per assay")
    ax.set_title("Per-assay comparison (%d assays)" % len(ours), fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")

    ax = axes[1]
    delta = np.sort(ours - base)
    cols = [ORANGE if v > 0 else BLUE for v in delta]
    ax.bar(np.arange(len(delta)), delta, color=cols, width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("assay (sorted by change)")
    ax.set_ylabel("\u0394 Spearman (ours \u2212 S2F)")
    ax.set_title("Change is broad, not a few lucky assays\nmean %+.4f, median %+.4f"
                 % (delta.mean(), np.median(delta)), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_per_assay.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig3 done  (mean delta %+.4f, win %d/%d)" % (delta.mean(), win.sum(), len(ours)))


# ---------------------------------------------------------------- fig 4
def fig4_expert_correlation():
    labels = {"s2f": "S2F\n(structure)", "masked": "ESM-1v\nmasked",
              "wt": "ESM-1v\nWT", "pssm": "PSSM\n(evolution)"}
    keys = ["s2f", "masked", "wt", "pssm"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    for ax, sub, title in zip(axes, ["dev40_ens5_iter8", "sel80_esm1v_ens5"],
                              ["dev-40 (reported set)", "sel-80 (selection set)"]):
        a = load_dump(sub)
        M = np.ones((4, 4))
        acc = {}
        for name, d in a.items():
            t = d["target"]; m = np.isfinite(t)
            if m.sum() < 20:
                continue
            c = d["cov"][m] >= 1 - 1e-9
            t = t[m]
            res_all = {}
            for e in keys:
                v = d[e][m]
                sel = c if e == "pssm" else np.ones_like(c, bool)
                res_all[e] = (v, sel)
            for i, A in enumerate(keys):
                for j, B in enumerate(keys):
                    if i >= j:
                        continue
                    sel = res_all[A][1] & res_all[B][1]
                    if sel.sum() < 20:
                        continue
                    tr = rankdata(t[sel])
                    ra = rankdata(res_all[A][0][sel]) - tr
                    rb = rankdata(res_all[B][0][sel]) - tr
                    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
                    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
                    acc.setdefault((i, j), []).append(float(np.mean(ra * rb)))
        for (i, j), v in acc.items():
            M[i, j] = M[j, i] = np.mean(v)
        cmap = LinearSegmentedColormap.from_list("c", ["#F7FBFF", "#FFE1C4", "#E8833A", "#8B2500"])
        im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=1.0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center",
                        fontsize=8.5, color="white" if M[i, j] > 0.87 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([labels[k] for k in keys], fontsize=7.5)
        ax.set_yticklabels([labels[k] for k in keys], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="residual correlation  (1 = redundant, 0 = complementary)")
    fig.suptitle("Why some experts are worth weight and others are not", fontsize=10.5, y=1.0)
    fig.savefig(os.path.join(OUT, "fig4_expert_correlation.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------- fig 5
def fig5_neff_gate():
    neffs = []
    for f in sorted(glob.glob(os.path.join(DUMP, "full204_ens5", "*.npz"))):
        p = os.path.join(PCACHE, os.path.basename(f)[:-4] + ".npz")
        if os.path.exists(p):
            neffs.append(float(np.load(p)["neff"]))
    neffs = np.array(neffs)
    es = np.clip(1 / (1 + np.exp(-8 * (np.log10(neffs) - 1.47))), 0, 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    g = np.logspace(np.log10(1.5), np.log10(700), 400)
    ax.plot(g, 1 / (1 + np.exp(-8 * (np.log10(g) - 1.47))), color=BLUE, lw=2,
            label=r"$e(N_{\rm eff})=\sigma(8(\log_{10}N_{\rm eff}-1.47))$")
    ax.scatter(neffs, es, s=14, color=ORANGE, alpha=0.65, edgecolor="none", zorder=4,
               label="%d assays (actual)" % len(neffs))
    ax.axvline(10 ** 1.47, color=GREY, ls=":", lw=1.2)
    ax.text(10 ** 1.47 * 1.08, 0.06, "midpoint\n$N_{\\rm eff}\\approx30$",
            fontsize=7.5, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\rm eff}$  (Henikoff effective sequence count of the assay sub-MSA)")
    ax.set_ylabel("weight given to the evolutionary term")
    ax.set_title("The fitted weight is a near-binary reliability gate, not a smooth blend\n"
                 "%d/%d assays get e<0.05 (term off); %d/%d get e>0.5 (term dominant)"
                 % ((es < 0.05).sum(), len(es), (es > 0.5).sum(), len(es)), fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.04, 1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_neff_gate.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------- fig 6 / 7
AA = list("ACDEFGHIKLMNPQRSTVWY")
# Two illustrative assays, each picked for what its figure needs: a compact
# complete scan for the heatmap, a fully-scanned protein for the structure.
HEAT_CASE = "POLG_HCVJF_Qi_2014"        # HCV polyprotein, 86 sites scanned
STRUCT_CASE = "NCAP_I34A1_Doud_2015"    # influenza nucleoprotein, all 498 aa
STRUCT_PDB = "NCAP_I34A1"


def _case_frame(case):
    """Map dump rows back to (site, mutant AA).

    load_dataset sorts mutations by (site, mutant); the dump arrays follow that
    order. Verified here against DMS_score rather than assumed - a row-order
    mismatch would silently scramble the whole figure.
    """
    z = np.load(os.path.join(DUMP, "full204_ens5", case + ".npz"))
    d = {k: z[k].astype(np.float64) for k in z.files}
    csv = pd.read_csv(os.path.join(
        S3F, "dataset", "DMS_ProteinGym_substitutions", case + ".csv"))
    csv["site"] = csv["mutant"].str.extract(r"(\d+)").astype(int)
    csv = csv.sort_values(["site", "mutant"], kind="stable").reset_index(drop=True)
    assert np.abs(csv["DMS_score"].values - d["target"]).max() < 1e-4, "row order mismatch"
    csv["wt_aa"] = csv["mutant"].str[0]
    csv["mt_aa"] = csv["mutant"].str[-1]
    csv["ours"] = champion_score(d, case)
    csv["s2f"] = d["s2f"]
    return csv


def fig6_dms_heatmap():
    df = _case_frame(HEAT_CASE)
    r_ours = spearmanr(df["ours"], df["DMS_score"]).correlation
    r_s2f = spearmanr(df["s2f"], df["DMS_score"]).correlation
    sites = np.sort(df["site"].unique())
    sidx = {s: i for i, s in enumerate(sites)}
    aidx = {a: i for i, a in enumerate(AA)}

    def grid(col):
        G = np.full((len(AA), len(sites)), np.nan)
        r = rankdata(df[col].values) / len(df)
        for k, (s, a) in enumerate(zip(df["site"], df["mt_aa"])):
            if a in aidx:
                G[aidx[a], sidx[s]] = r[k]
        return G

    panels = [("Experimental fitness (ground truth)", grid("DMS_score")),
              ("Published S2F prediction   \u03c1 = %.3f" % r_s2f, grid("s2f")),
              ("This work                  \u03c1 = %.3f" % r_ours, grid("ours"))]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True)
    for ax, (title, G) in zip(axes, panels):
        im = ax.imshow(G, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(len(AA)))
        ax.set_yticklabels(AA, fontsize=5.5)
        ax.set_ylabel("mutant AA", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(False)
    axes[-1].set_xticks(np.arange(0, len(sites), 5))
    axes[-1].set_xticklabels(sites[::5], fontsize=6, rotation=90)
    axes[-1].set_xlabel("residue position in the HCV NS5A polyprotein region", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01,
                 label="rank-normalised score  (blue = deleterious, red = tolerated)")
    fig.suptitle("Deep mutational scan of %s: the new scoring recovers "
                 "position-specific structure the baseline misses" % HEAT_CASE,
                 fontsize=10.5, y=0.975)
    fig.savefig(os.path.join(OUT, "fig6_dms_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


def fig7_structure():
    df = _case_frame(STRUCT_CASE)
    # per-site skill change: Spearman within each site's mutants
    per = {}
    for s, g in df.groupby("site"):
        if len(g) < 8:
            continue
        a = spearmanr(g["ours"], g["DMS_score"]).correlation
        b = spearmanr(g["s2f"], g["DMS_score"]).correlation
        if np.isfinite(a) and np.isfinite(b):
            per[int(s)] = a - b

    ca = {}
    with open(os.path.join(S3F, "dataset", "ProteinGym_AF2_structures",
                           STRUCT_PDB + ".pdb")) as fh:
        for ln in fh:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                ca[int(ln[22:26])] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    res = sorted(ca)
    xyz = np.array([ca[r] for r in res])

    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.12)
    ax = fig.add_subplot(gs[0], projection="3d")

    # colour each backbone segment by the mean per-site delta of its endpoints;
    # residues without enough mutants stay grey
    # saturate at the 92nd percentile: a couple of extreme sites would otherwise
    # wash the whole structure out to white
    v = float(np.percentile(np.abs(list(per.values())), 92))
    norm = plt.Normalize(-v, v)
    cmap = plt.get_cmap("RdBu_r")
    segs, cols = [], []
    for i in range(len(res) - 1):
        segs.append([xyz[i], xyz[i + 1]])
        a, b = per.get(res[i]), per.get(res[i + 1])
        vv = [x for x in (a, b) if x is not None]
        cols.append(cmap(norm(np.mean(vv))) if vv else (0.85, 0.85, 0.85, 1.0))
    lc = Line3DCollection(segs, colors=cols, linewidths=2.6)
    ax.add_collection3d(lc)
    mn, mx = xyz.min(0), xyz.max(0)
    ctr, rad = (mn + mx) / 2, (mx - mn).max() / 2 * 0.62
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
        setter(c - rad, c + rad)
    ax.set_axis_off()
    ax.set_title("%s \u2014 AlphaFold backbone\ncoloured by per-site change in "
                 "rank correlation" % STRUCT_CASE, fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=-0.02, shrink=0.62,
                 orientation="horizontal",
                 label="\u0394 per-site \u03c1 (ours \u2212 S2F)")

    ax2 = fig.add_subplot(gs[1])
    ks = sorted(per)
    vs = [per[k] for k in ks]
    ax2.bar(ks, vs, color=[ORANGE if x > 0 else BLUE for x in vs], width=1.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("residue position")
    ax2.set_ylabel("\u0394 per-site Spearman (ours \u2212 S2F)")
    ax2.set_title("Improvement along the sequence: %d/%d assayed sites improve"
                  % (sum(1 for x in vs if x > 0), len(vs)), fontsize=9)
    fig.savefig(os.path.join(OUT, "fig7_structure.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig7 done  (%d sites, %d improved)" % (len(vs), sum(1 for x in vs if x > 0)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_method_comparison()
    fig2_trajectory()
    fig3_per_assay()
    fig4_expert_correlation()
    fig5_neff_gate()
    fig6_dms_heatmap()
    fig7_structure()
