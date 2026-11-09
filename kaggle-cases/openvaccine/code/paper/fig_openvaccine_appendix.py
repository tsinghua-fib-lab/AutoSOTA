"""Appendix figure for the OpenVaccine case: search trajectory, model complementarity,
pseudo-labelling rounds and masked-node pretraining ablation.

  python paper/fig_openvaccine_appendix.py   -> paper/fig_openvaccine_appendix.{pdf,png}

Scores are taken from runs/*/results.json and scripts/blend.py outputs; the complementarity
curves are recomputed from out-of-fold predictions."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from blend import cv  # noqa: E402

KERAS, TFM, MIX = "#E08A4B", "#4C78A8", "#2A9D8F"
GREY, WINNER, WORSE = "#9AA0A6", "#5F6368", "#C0392B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titlepad": 6,
})


def oof(run):
    prep = np.load(os.path.join(ROOT, "runs", run, "prep.npz"))
    heads = {}
    for f in glob.glob(os.path.join(ROOT, "runs", run, "models", "*_preds.npz")):
        h = os.path.basename(f).rsplit("_f", 1)[0]
        d = np.load(f)
        heads.setdefault(h, np.zeros(prep["y"].shape, np.float32))[d["oof_idx"]] = d["oof"]
    return np.mean(list(heads.values()), 0), prep


def title(ax, letter, text):
    ax.set_title(r"$\mathbf{%s}$   %s" % (letter, text) if letter else text)


# (label, family, public, in final lineage; None = reproduced baseline)
SEARCH = [
    ("s1", KERAS, 0.23636, True), ("s2", KERAS, 0.23321, None), ("s3", KERAS, 0.23332, False),
    ("e1", KERAS, 0.23702, False), ("t1", TFM, 0.24881, False), ("t2", TFM, 0.24628, False),
    ("t3", TFM, 0.24128, True), ("t4", TFM, 0.23555, True), ("t5", TFM, 0.23566, True),
    ("s4", KERAS, 0.23126, True), ("R1", MIX, 0.23005, True), ("t6", TFM, 0.23505, False),
    ("t7", TFM, 0.23579, False), ("s5", KERAS, 0.22959, True), ("R2", MIX, 0.22922, False),
    ("t8", TFM, 0.23532, False), ("s6", KERAS, 0.22938, False), ("R3", MIX, 0.22941, False),
    ("t9", TFM, 0.23967, True), ("t10", TFM, 0.23369, True), ("s5b", KERAS, 0.22998, False),
    ("Final", MIX, 0.22859, True),
]
LABEL_BELOW = {"t1", "t2", "t3", "t9", "R2", "R3"}


def draw_search(ax):
    x = np.arange(len(SEARCH))
    for xs in (2.5, 10.5, 17.5, 20.5):
        ax.axvline(xs, color="#E3E5E8", lw=0.6, zorder=0)
    for xc, name in [(1, "Repro."), (6.5, "Round 1"), (14, "Rounds 2–3"), (19, "Pretraining")]:
        ax.text(xc, 0.2497, name, ha="center", va="bottom", fontsize=6, color="#5f6368")
    best = np.minimum.accumulate([p if c == MIX or l in ("s1", "s2") else np.inf for l, c, p, _ in SEARCH])
    ax.step(x, best, where="post", color=MIX, lw=0.9, alpha=0.55, zorder=2)
    ax.axhline(0.22815, color=WINNER, ls="--", lw=0.7, zorder=1)
    ax.text(-0.6, 0.22815 + 0.0002, "Historical winner: 0.22815", ha="left", va="bottom", fontsize=6, color=WINNER)
    for i, (lab, col, pub, keep) in enumerate(SEARCH):
        if lab == "Final":
            ax.scatter(i, pub, marker="D", s=30, color=MIX, edgecolor="k", lw=0.5, zorder=4)
        elif keep is None:
            ax.scatter(i, pub, marker="s", s=20, color="white", edgecolor="k", lw=0.8, zorder=3)
        elif keep:
            ax.scatter(i, pub, s=20, color=col, edgecolor="none", zorder=3)
        else:
            ax.scatter(i, pub, s=20, facecolor="white", edgecolor=GREY, lw=0.8, zorder=3)
        below = lab in LABEL_BELOW
        ax.text(i, pub + (-0.0005 if below else 0.0005), lab, ha="center", va="top" if below else "bottom",
                fontsize=5.8, color="k" if keep else "#6b6b6b")
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(SEARCH) - 0.3)
    ax.set_ylim(0.2272, 0.2512)
    ax.set_ylabel("Public MCRMSE")
    ax.set_xlabel("Experiments in chronological order")
    title(ax, "a", "Search trajectory (public score used for selection)")
    handles = [
        Line2D([], [], marker="o", ls="", color=KERAS, label="Keras (reference family)"),
        Line2D([], [], marker="o", ls="", color=TFM, label="Transformer (new family)"),
        Line2D([], [], marker="o", ls="", color=MIX, label="Cross-family blend"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="Not retained"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor="k", label="Reproduced baseline"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.48, 0.93), frameon=False, ncol=2,
              handletextpad=0.2, columnspacing=1.0, labelspacing=0.4)


def draw_complementarity(ax):
    s1, prep = oof("s1_nopl")
    w = np.round(np.arange(0, 1.0001, 0.05), 2)
    for run, col, ls, lab in [("t3_sweep", TFM, "-", "Keras + Transformer"),
                              ("t9_pre", MIX, "--", "Keras + pretrained Transformer")]:
        t, _ = oof(run)
        c = np.array([cv((1 - a) * s1 + a * t, prep) for a in w])
        ax.plot(w, c, ls, color=col, lw=1.2, label=lab)
        i = int(np.argmin(c))
        ax.scatter(w[i], c[i], color=col, s=12, zorder=3)
    base = cv(s1, prep)
    c03 = cv(0.7 * s1 + 0.3 * oof("t3_sweep")[0], prep)
    ax.scatter(0.3, c03, marker="D", s=20, color=TFM, edgecolor="k", lw=0.5, zorder=4)
    ax.annotate("teacher weight 0.3\nCV %.4f" % c03, (0.3, c03), xytext=(0.47, 0.2178), fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.annotate("Keras only\nCV %.4f" % base, (0, base), xytext=(0.1, 0.2212), fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.set_xlabel("Transformer weight in out-of-fold blend")
    ax.set_ylabel("Cross-validation MCRMSE")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.2154, 0.2276)
    ax.legend(frameon=False, loc="upper left")
    title(ax, "b", "Model complementarity")


ROUNDS = {
    "pub": {"same": [0.23636, 0.23321, 0.23332], "cross": [0.23395, 0.23005, 0.22922, 0.22941]},
    "priv": {"same": [0.34767, 0.34382, 0.34493], "cross": [0.34486, 0.34038, 0.33945, 0.33949]},
}


def draw_round(ax, key, letter, text):
    same, cross = ROUNDS[key]["same"], ROUNDS[key]["cross"]
    ax.plot([0, 1, 2], same, "o-", color=KERAS, lw=1.2, ms=3.5, label="Same-family teacher")
    ax.plot([0, 1, 2], cross[:3], "o-", color=MIX, lw=1.2, ms=3.5, label="Cross-family teacher")
    ax.plot([2, 3], cross[2:], "--", color=GREY, lw=1.0)
    ax.scatter([3], [cross[3]], facecolor="white", edgecolor=GREY, s=16, zorder=3, label="Stopped (public worse)")
    ax.scatter([2], [same[2]], marker="x", color=WORSE, s=20, zorder=4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["None", "1", "2", "3"])
    ax.set_xlim(-0.3, 3.3)
    ax.set_xlabel("Pseudo-labelling round")
    ax.set_ylabel("%s MCRMSE" % ("Public" if key == "pub" else "Private"))
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    title(ax, letter, text)


def draw_pretraining(ax):
    rows = [
        ("Single-model CV, no PL", 0.23366, 0.23681),
        ("Ensemble CV, no PL", 0.22559, 0.22475),
        ("Public, no PL (t3→t9)", 0.24128, 0.23967),
        ("Private, no PL (t3→t9)", 0.34725, 0.34561),
        ("Public, PL round 2 (t6→t10)", 0.23505, 0.23369),
        ("Private, PL round 2 (t6→t10)", 0.34192, 0.34145),
    ]
    y = np.arange(len(rows))[::-1]
    rel = [100 * (b - a) / a for _, a, b in rows]
    ax.barh(y, rel, color=[WORSE if r > 0 else MIX for r in rel], height=0.62)
    for yi, r in zip(y, rel):
        ax.text(max(r, 0) + 0.06, yi, "%+.2f%%" % r, va="center", ha="left", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1.1, 1.9)
    ax.set_xlabel("Relative change in MCRMSE (%)")
    ax.tick_params(axis="y", length=0)
    title(ax, "d", "Masked-node pretraining")


def main():
    fig = plt.figure(figsize=(7.2, 4.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.04, hspace=0.06, wspace=0.03)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.95])
    top = gs[0].subgridspec(1, 2, width_ratios=[1.7, 1])
    bottom = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    ax_c1, ax_c2, ax_d = (fig.add_subplot(bottom[i]) for i in range(3))

    draw_search(ax_a)
    draw_complementarity(ax_b)
    draw_round(ax_c1, "pub", "c", "Pseudo-labelling: public")
    draw_round(ax_c2, "priv", "", "Pseudo-labelling: private")
    ax_c2.axhline(0.34198, color=WINNER, ls="--", lw=0.7)
    ax_c2.text(3.25, 0.34198 + 0.0003, "Historical winner: 0.34198", ha="right", va="bottom",
               fontsize=6, color=WINNER)
    ax_c1.set_ylim(0.2286, 0.2393)
    ax_c1.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.03), handlelength=1.6)
    draw_pretraining(ax_d)
    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)

    out = os.path.join(ROOT, "paper", "fig_openvaccine_appendix")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=220)
    print(out)


if __name__ == "__main__":
    main()
