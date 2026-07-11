import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Parse results.txt ────────────────────────────────────────────────────────
rows = []
with open("results/results.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("seq") or line.startswith("---"):
            continue
        parts = line.split()
        seq_len = int(parts[0])
        method = parts[1]
        r = None if parts[2] == "-" else int(parts[2])
        num_bins = None if parts[3] == "-" else int(parts[3])
        p20 = float(parts[4])
        median = float(parts[5])
        p80 = float(parts[6])
        error = None if parts[7] == "-" else float(parts[7])
        rows.append(dict(seq_len=seq_len, method=method, r=r,
                         num_bins=num_bins, p20=p20, median=median,
                         p80=p80, error=error))

df = pd.DataFrame(rows)
flash = df[df.method == "flash"].copy()
wildcat = df[df.method == "wildcat"].copy()
wildcat = wildcat.copy()
wildcat["ratio"] = wildcat["r"] // wildcat["num_bins"]

ratio_values = sorted(wildcat["r"].unique())
bins_values = sorted(wildcat["num_bins"].unique())

ratio_colors = {rv: c for rv, c in zip(ratio_values, plt.cm.tab10(np.linspace(0, 0.9, len(ratio_values))))}
bins_markers = {bv: m for bv, m in zip(bins_values, ["*", "s", "x"])}

# ── Scatterplot — median speed (log x) vs max_abs_error (y) ─────────────
fig, ax = plt.subplots(figsize=(7, 5))

plotted_bins = set()
plotted_ratios = set()
for _, row in wildcat.iterrows():
    nb, ratio = row["num_bins"], row["r"]
    color = ratio_colors[ratio]
    marker = bins_markers[nb]
    ax.scatter(row["median"], row["error"], color=color, marker=marker,
                alpha=0.8, s=60)
    ax.annotate(f"{int(row.seq_len)}", (row["median"], row["error"]),
                 textcoords="offset points", xytext=(4, 2), fontsize=6)
    plotted_bins.add(nb)
    plotted_ratios.add(ratio)

# Legend: marker = num_bins
for nb in sorted(plotted_bins):
    ax.scatter([], [], color="grey", marker=bins_markers[nb], label=f"B={nb}")
# Legend: colour = r
for rv in sorted(plotted_ratios):
    ax.scatter([], [], color=ratio_colors[rv], marker="o", label=f"r={rv}")

ax.set_xscale("log")

ax.set_xlabel("Median runtime (ms)")
ax.set_ylabel(r"Approximation error $\| \mathbf{O} - \mathbf{\hat{O}}\|_{\text{max}}$")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("plot_ablations.pdf")
print("Saved plot_ablations.pdf")

plt.show()
