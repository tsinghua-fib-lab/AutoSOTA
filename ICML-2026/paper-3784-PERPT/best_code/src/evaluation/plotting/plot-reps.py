#!/usr/bin/env python3
import os, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D

# ------------------------------- config --------------------------------
MODELS = [
    ("Gamba NTP-only",   "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_seq_only_step_56000/reps_gamba_all_roi.npz", "Gamba", False, 66_492_392, 2048),
    ("Gamba NTP-Only Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_seq_only_step_random_init/reps_gamba_all_roi.npz", "Gamba", False, 66_492_392, 2048),

    ("Gamba CEP-only",   "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_cons_only_step_44000/reps_gamba_all_roi.npz", "Gamba", True, 66_492_392, 2048),
    ("Gamba CEP-Only Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_cons_only_step_random_init/reps_gamba_all_roi.npz", "Gamba", True, 66_492_392, 2048),

    ("Gamba NTP+CEP",    "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_dual_step_44000/reps_gamba_all_roi.npz", "Gamba", True, 66_493_418, 2048),
    ("Gamba NTP+CEP Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/gamba_dual_step_random_init/reps_gamba_all_roi.npz", "Gamba", True, 66_493_418, 2048),

    ("Bi-Gamba MLM-only", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_seq_only_step_44000/reps_caduceus_all_roi.npz", "Bi-Gamba", False, 3_864_832, 2048),
    ("Bi-Gamba MLM-Only Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_seq_only_step_random_init/reps_caduceus_all_roi.npz", "Bi-Gamba", False, 3_864_832, 2048),

    ("Bi-Gamba MEM-only", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_cons_only_step_44000/reps_caduceus_all_roi.npz", "Bi-Gamba", True, 3_864_832, 2048),
    ("Bi-Gamba MEM-Only Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_cons_only_step_random_init/reps_caduceus_all_roi.npz", "Bi-Gamba", True, 3_864_832, 2048),

    ("Bi-Gamba MLM+MEM", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_dual_step_44000/reps_caduceus_all_roi.npz", "Bi-Gamba", True, 3_869_442, 2048),
    ("Bi-Gamba MLM+MEM Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus_dual_step_random_init/reps_caduceus_all_roi.npz", "Bi-Gamba", True, 3_869_442, 2048),

    ("NT multi-species", "/home/mica/NucleotideTransformer/global_representations/nt-ms/reps_nt-ms_all_roi.npz", "Other", False, 498_345_436, 1000),
    ("NT multi-species Random-Init", "/home/mica/NucleotideTransformer/global_representations/nt-ms-random-init/reps_nt-ms-random-init_all_roi.npz", "Other", False, 498_345_436, 1000),

    ("NT human-ref", "/home/mica/NucleotideTransformer/global_representations/nt-human/reps_nt-human_all_roi.npz", "Other", False, 480_438_241, 1000),
    ("NT human-ref Random-Init", "/home/mica/NucleotideTransformer/global_representations/nt-human-random-init/reps_nt-human-random-init_all_roi.npz", "Other", False, 480_438_241, 1000),

    ("PhyloGPN", "/home/mica/NucleotideTransformer/global_representations/phyloGPN/reps_phyloGPN_all_roi.npz", "Other", False, 83_185_924, 481),
    ("PhyloGPN Random-Init", "/home/mica/NucleotideTransformer/global_representations/phyloGPN-random-init/reps_phyloGPN-random-init_all_roi.npz", "Other", False, 83_185_924, 481),

    ("K-mer (k=6)", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/baseline/kmer6/reps_kmer6_all_roi.npz", "Other", False, 0, 2048),
    ("PhyloP (6D)", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/baseline/phylop/reps_phylop_all_roi.npz", "Other", True, 0, 2048),

    ("HyenaDNA", "/home/mica/NucleotideTransformer/global_representations/hyenaDNA/reps_hyenaDNA_all_roi.npz", "Other", False, 6_551_040, 160_000),
    ("HyenaDNA Random-Init", "/home/mica/NucleotideTransformer/global_representations/hyenaDNA-random-init/reps_hyenaDNA-random-init_all_roi.npz", "Other", False, 6_551_040, 160_000),

    ("Caduceus", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus-theirs/reps_caduceus-theirs_all_roi.npz", "Other", False, 7_725_312, 131_000),
    ("Caduceus Random-Init", "/home/mica/gamba/data_processing/data/240-mammalian/global_representations/caduceus-theirs-random-init/reps_caduceus-theirs-random-init_all_roi.npz", "Other", False, 7_725_312, 131_000),
]

# colors
BLUE, PURPLE, ORANGE, GREY = "#4287f5", "#6F2DA8", "#FF8C32", "#B0B0B0"
SEQ_PLUS_PHY = {"Bi-Gamba MLM+MEM", "Gamba NTP+CEP"}
PHY_ONLY     = {"Bi-Gamba MEM-only", "Gamba CEP-only"}
SEQ_ONLY     = {"Bi-Gamba MLM-only", "Gamba NTP-only"}
BASELINES = {"K-mer (k=6)": "k-mer (k=6) baseline", "PhyloP (6D)": "phyloP (6d) baseline"}

# markers and sizes
GAMBA_MARK, BIGAMBA_MARK, OTHER_MARK = 's', 'o', '^'   # force triangles for all "Other"
MS, CAP, ERR = 6, 2, "black"

# ------------------------------- utils ----------------------------------
def strip_rand(s): return re.sub(r"\s*random-init\s*$", "", s, flags=re.IGNORECASE).strip()
def base_key(s):  return re.sub(r"\s+", " ", strip_rand(s)).lower()
def color_for(name):
    b = strip_rand(name)
    if b in SEQ_PLUS_PHY: return BLUE
    if b in PHY_ONLY:     return PURPLE
    if b in SEQ_ONLY:     return ORANGE
    return GREY

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    if {"embeddings","labels"}.issubset(z.files): return z["embeddings"], z["labels"].astype(str)
    raise KeyError(f"bad npz keys in {path}: {sorted(z.files)}")

def loo_preds(X, y):
    X = np.asarray(X); y = np.asarray(y)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    return y, y[idx[:,1]]

def per_class_recalls(y_true, y_pred):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n = cm.sum(axis=1).astype(float); k = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(k, np.where(n==0, 1, n))
    return classes, p, n

def ba_ci_from_pc(p, n):
    K = len(p)
    var = np.nansum(p*(1-p)/np.where(n==0, np.inf, n)) / (K**2)
    ba  = float(np.nanmean(p))*100; se = np.sqrt(var)*100
    return ba, ba-1.96*se, ba+1.96*se

def build_offsets(models, span=2):
    order = sorted(models); seq=[0]; k=1
    while len(seq)<len(order): seq += [k,-k]; k+=1
    return {m:max(-span,min(span,s)) for m,s in zip(order, seq[:len(order)])}

def jitter(x, model, width, OFF): return x*(10**(width*OFF.get(model,0)))

# simple label de-overlap for "Other" points: stagger small y offsets within local x-bins
def label_offsets(df, xcol, ycol, bin_width_log=0.15, step_pts=6):
    if df.empty: return {}
    logx = np.log10(df[xcol].to_numpy())
    y = df[ycol].to_numpy()
    bins = np.floor(logx/bin_width_log).astype(int)
    order = np.argsort(y)
    offsets = {}
    per_bin_count = {}
    for i in order:
        b = bins[i]
        k = per_bin_count.get(b, 0)
        # cycle offsets: 0, +1, -1, +2, -2, ...
        seq = [0]
        t=1
        while len(seq) <= k:
            seq += [t, -t]; t += 1
        offsets[df.index[i]] = seq[k]*step_pts
        per_bin_count[b] = k+1
    return offsets

# -------------------------- load & aggregate ----------------------------
rows, missing = [], []
pc_store = {}

for label, path, fam, uses_phy, params, ctx in MODELS:
    if not os.path.exists(path):
        missing.append((label, path)); continue
    try:
        X, y = load_npz(path)
        yt, yp = loo_preds(X, y)
        cls, p, n = per_class_recalls(yt, yp)
        ba, lo, hi = ba_ci_from_pc(p, n)
        rows.append(dict(Model=label, Base=strip_rand(label), BaseKey=base_key(label),
                         Family=fam, UsesPhyloP=uses_phy, Params=params,
                         Params_plot=(params if params>0 else 1), Context=ctx,
                         BA=ba, BA_lo=lo, BA_hi=hi))
        pc_store[(base_key(label), "rand" if "random-init" in label.lower() else "main")] = dict(
            classes=cls, p=p, n=n, fam=fam, uses=uses_phy, params=params, ctx=ctx, model=label)
    except Exception as e:
        print(f"[skip] {label}: {e}")

if not rows: raise SystemExit("no npz loaded successfully.")

df_all = pd.DataFrame(rows)
OFF = build_offsets(df_all["Model"].unique(), span=2)
df_all["Params_j"]  = [jitter(x, m, 0.03, OFF) for x,m in zip(df_all["Params_plot"], df_all["Model"])]
df_all["Context_j"] = [jitter(x, m, 0.02, OFF) for x,m in zip(df_all["Context"],     df_all["Model"])]

mask_rand = df_all["Model"].str.contains(r"random-init", case=False, na=False)
dfA = df_all[~mask_rand].copy()

is_baseline = dfA["Base"].isin(BASELINES.keys())
df_base = dfA[is_baseline].copy()
df_pts  = dfA[~is_baseline].copy()
if df_pts.empty: df_pts = dfA.copy()

# delta vs random-init
paired = []
for key in sorted({k for k,_ in pc_store.keys()}):
    if (key,"main") in pc_store and (key,"rand") in pc_store:
        a, b = pc_store[(key,"main")], pc_store[(key,"rand")]
        classes = np.union1d(a["classes"], b["classes"])
        def to_vec(rec):
            idx = {c:i for i,c in enumerate(rec["classes"])}
            p = np.array([rec["p"][idx[c]] if c in idx else np.nan for c in classes], float)
            n = np.array([rec["n"][idx[c]] if c in idx else 0.0   for c in classes], float)
            return p, n
        p1, n1 = to_vec(a); p0, n0 = to_vec(b)
        delta = (np.nanmean(p1) - np.nanmean(p0))*100
        var = (np.nansum(p1*(1-p1)/np.where(n1==0,np.inf,n1)) +
               np.nansum(p0*(1-p0)/np.where(n0==0,np.inf,n0))) / (len(classes)**2)
        se = np.sqrt(var)*100
        main_row = dfA[dfA["BaseKey"]==key].iloc[0]
        paired.append(dict(Model=main_row["Model"], Base=main_row["Base"], BaseKey=key,
                           Family=main_row["Family"], UsesPhyloP=main_row["UsesPhyloP"],
                           Params_plot=main_row["Params_plot"], Context=main_row["Context"],
                           Params_j=main_row["Params_j"], Context_j=main_row["Context_j"],
                           Delta=delta, Delta_lo=delta-1.96*se, Delta_hi=delta+1.96*se))
m = pd.DataFrame(paired)


#for the SEM of the BAs, do the pooled variance 

#for balanced accuracy deltas, subtract between all the models recalls and their random init and then do the average 
#fit a line to the pretrained- random init and get the slope as improvement and errorn as the error from the fit
#fit intercept to be 0

# ------------------------------- plotting --------------------------------
plt.rcParams.update({"font.size":12})

def draw_panel_with_baselines(ax, pts_tbl, base_tbl, xcol, xlabel, title, annotate_points=True, annotate_baselines=True):
    # points
    # force markers: gamba squares, bi-gamba circles, all other triangles
    for _, r in pts_tbl.iterrows():
        if r["Family"] == "Gamba":     mkr = GAMBA_MARK
        elif r["Family"] == "Bi-Gamba": mkr = BIGAMBA_MARK
        else:                           mkr = OTHER_MARK
        col = color_for(r["Model"])
        ax.errorbar(r[xcol], r["BA"],
                    yerr=[[r["BA"]-r["BA_lo"]],[r["BA_hi"]-r["BA"]]],
                    fmt=mkr, markersize=MS, mfc=col, mec="black", mew=0.9,
                    ecolor=ERR, elinewidth=1.0, capsize=CAP, alpha=0.95)

    ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("balanced accuracy (%)")
    ax.set_title(title)

    # limits
    ymin = max(0, np.floor(np.nanmin(pts_tbl["BA"].to_numpy()) - 5))
    ymax = min(100, np.ceil(np.nanmax(pts_tbl["BA"].to_numpy()) + 5))
    ax.set_ylim(ymin, ymax)
    xs = pts_tbl[xcol].to_numpy()
    ax.set_xlim(xs.min()*0.8, xs.max()*1.25)

    # baseline lines with labels just under the line
    for _, r in base_tbl.iterrows():
        y = r["BA"]
        ax.axhline(y, ls="--", lw=1.2, color="0.35", zorder=0)
        if annotate_baselines:
            ax.annotate(BASELINES.get(r["Base"], r["Base"]),
                        xy=(xs.max()*1.22, y), xytext=(0, -3), textcoords="offset points",
                        ha="right", va="top", fontsize=9, color="0.35",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # annotate only "Other" models with decluttered offsets
    if annotate_points:
        other = pts_tbl[pts_tbl["Family"]=="Other"]
        offsets = label_offsets(other, xcol, "BA", bin_width_log=0.15, step_pts=6)
        for idx, r in other.iterrows():
            dy = offsets.get(idx, 0)
            ax.annotate(strip_rand(r["Model"]), (r[xcol], r["BA"]),
                        xytext=(4, 2+dy), textcoords="offset points", fontsize=9)

def draw_delta(ax, table, xcol, xlabel, title, annotate_points=True):
    if table.empty:
        ax.set_visible(False); return
    for _, r in table.iterrows():
        if r["Family"] == "Gamba":     mkr = GAMBA_MARK
        elif r["Family"] == "Bi-Gamba": mkr = BIGAMBA_MARK
        else:                           mkr = OTHER_MARK
        col = color_for(r["Model"])
        ax.errorbar(r[xcol], r["Delta"],
                    yerr=[[r["Delta"]-r["Delta_lo"]],[r["Delta_hi"]-r["Delta"]]],
                    fmt=mkr, markersize=MS, mfc=col, mec="black", mew=0.9,
                    ecolor=ERR, elinewidth=1.0, capsize=CAP, alpha=0.95)
    ax.axhline(0, color="0.6", lw=1, zorder=0)
    ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Δ balanced accuracy (pp)")
    ax.set_title(title)

    if annotate_points:
        other = table[table["Family"]=="Other"]
        offsets = label_offsets(other, xcol, "Delta", bin_width_log=0.15, step_pts=6)
        for idx, r in other.iterrows():
            dy = offsets.get(idx, 0)
            ax.annotate(strip_rand(r["Model"]), (r[xcol], r["Delta"]),
                        xytext=(4, 2+dy), textcoords="offset points", fontsize=9)

# ---------- figure with labels ----------
fig1, axes1 = plt.subplots(1, 2, figsize=(14.5, 6))
fig1.subplots_adjust(right=0.80)
draw_panel_with_baselines(axes1[0], df_pts, df_base, "Params_j", "parameters (log)",
                          "panel a: accuracy vs parameter count",
                          annotate_points=True, annotate_baselines=True)
draw_delta(axes1[1], m, "Params_j", "parameters (log)", "panel c: Δ vs random-init",
           annotate_points=True)

shape_handles = [
    Line2D([0],[0], marker='s', ls='None', mfc='white', mec='black', mew=1, label='gamba (square)'),
    Line2D([0],[0], marker='o', ls='None', mfc='white', mec='black', mew=1, label='bi-gamba (circle)'),
    Line2D([0],[0], marker='^', ls='None', mfc='white', mec='black', mew=1, label='other models'),
]
color_handles = [
    Line2D([0],[0], marker='o', ls='None', mfc=BLUE,   mec='black', mew=0.9, label='seq + phyloP'),
    Line2D([0],[0], marker='o', ls='None', mfc=PURPLE, mec='black', mew=0.9, label='phyloP-only'),
    Line2D([0],[0], marker='o', ls='None', mfc=ORANGE, mec='black', mew=0.9, label='seq-only'),
    Line2D([0],[0], ls='--', color='0.35', label='baseline line'),
]
fig1.legend(handles=shape_handles + color_handles, loc="center left",
            bbox_to_anchor=(0.80, 0.5), title="symbols and colors", frameon=False)

# ---------- figure without any labels ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(14.5, 6))
fig2.subplots_adjust(right=0.80)
draw_panel_with_baselines(axes2[0], df_pts, df_base, "Params_j", "parameters (log)",
                          "panel a: accuracy vs parameter count",
                          annotate_points=False, annotate_baselines=False)
draw_delta(axes2[1], m, "Params_j", "parameters (log)", "panel c: Δ vs random-init",
           annotate_points=False)
fig2.legend(handles=shape_handles + color_handles, loc="center left",
            bbox_to_anchor=(0.80, 0.5), title="symbols and colors", frameon=False)

# save
outdir = "/home/mica/gamba/data_processing/data/240-mammalian/figures"
os.makedirs(outdir, exist_ok=True)
fig1.savefig(f"{outdir}/ba_vs_params_context.png", dpi=300, bbox_inches="tight")
fig1.savefig(f"{outdir}/ba_vs_params_context.svg", format="svg", bbox_inches="tight")
fig2.savefig(f"{outdir}/ba_vs_params_context_nolabels.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{outdir}/ba_vs_params_context_nolabels.svg", format="svg", bbox_inches="tight")
print("[info] wrote labeled and nolabels figures to:", outdir)
