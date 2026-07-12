#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""README phase-machine diagram: inventory view.

Shows how the # of decoding (still-active) requests evolves over time:
starts at N, decays during Phase 1 (decode) as completions arrive,
hits N−k̂* when the scheduler switches to Phase 2 (refill), then jumps
back up to N as k̂* new requests are prefilled. Repeats indefinitely.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Parameters (illustrative values; not from any specific run)
N = 100  # batch capacity
k_hat = 40  # phase-switching threshold
p_0 = 0.025  # per-iter completion probability (so τ* ≈ ln(1-θ)/ln(1-p₀))
prefill_duration = 8  # iters for Phase 2 (refill)
num_cycles = 3  # number of decode/refill cycles to draw

# Build the trajectory
ts, ys = [0.0], [float(N)]  # (time, # decoding requests)
phases = []  # list of (t_start, t_end, "decode"|"refill")
t = 0.0

for cycle in range(num_cycles):
    # ---- Phase 1: decode ----
    # Fluid trajectory n(τ) = N(1-p_0)^τ, switch when n reaches N-k_hat
    t_start = t
    n_cur = N
    while n_cur > N - k_hat:
        t += 1
        n_cur = N * ((1 - p_0) ** (t - t_start))
        ts.append(t)
        ys.append(max(n_cur, N - k_hat))
        if n_cur <= N - k_hat:
            break
    phases.append((t_start, t, "decode"))

    # ---- Phase 2: refill (linear ramp) ----
    t_start = t
    for i in range(prefill_duration):
        t += 1
        # Linear refill from N-k_hat back to N
        ramp = (N - k_hat) + k_hat * (i + 1) / prefill_duration
        ts.append(t)
        ys.append(ramp)
    phases.append((t_start, t, "refill"))

fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=150)

# Phase shading
for t0, t1, kind in phases:
    if kind == "decode":
        ax.axvspan(t0, t1, alpha=0.08, color="#2E8B57", zorder=0)
    else:
        ax.axvspan(t0, t1, alpha=0.10, color="#E8A33D", zorder=0)

# Reference lines for N and N-k̂*
ax.axhline(N, color="#666", linewidth=1.0, linestyle="--", zorder=1)
ax.axhline(N - k_hat, color="#666", linewidth=1.0, linestyle="--", zorder=1)

# Trajectory
ax.plot(ts, ys, color="#1f4e79", linewidth=2.4, zorder=3)

xmax = ts[-1]

# Side arrow + label showing the drop is k̂*
# Position arrow in the first decode region
mid_decode_x = (phases[0][0] + phases[0][1]) / 2
ax.annotate(
    "",
    xy=(mid_decode_x, N - k_hat + 0.5),
    xytext=(mid_decode_x, N - 0.5),
    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4),
    zorder=4,
)
ax.text(
    mid_decode_x + 1,
    N - k_hat / 2,
    r"$\hat{k}^*$ completions",
    fontsize=11.5,
    ha="left",
    va="center",
    color="#222",
    bbox=dict(
        boxstyle="round,pad=0.25", facecolor="white", edgecolor="#888", alpha=0.95
    ),
    zorder=5,
)

# Phase labels (above plot, in each region of cycle 1)
ax.text(
    (phases[0][0] + phases[0][1]) / 2,
    N + 7,
    "Phase 1\nDecode",
    ha="center",
    fontsize=10.5,
    color="#2E8B57",
    fontweight="bold",
)
ax.text(
    (phases[1][0] + phases[1][1]) / 2,
    N + 7,
    "Phase 2\nRefill",
    ha="center",
    fontsize=10.5,
    color="#B07424",
    fontweight="bold",
)

# Smaller labels for subsequent cycles (just "Decode" / "Refill")
for i, (t0, t1, kind) in enumerate(phases[2:], start=2):
    label = "Decode" if kind == "decode" else "Refill"
    color = "#2E8B57" if kind == "decode" else "#B07424"
    ax.text(
        (t0 + t1) / 2,
        N + 4,
        label,
        ha="center",
        fontsize=9.5,
        color=color,
        style="italic",
    )

# Axes setup
ax.set_xlim(-xmax * 0.05, xmax * 1.02)
ax.set_ylim(N - k_hat - 12, N + 17)
ax.set_xticks([])  # time axis is illustrative; drop tick numbers
# Y-ticks at the two reference levels (renders OUTSIDE plot area)
ax.set_yticks([N - k_hat, N])
ax.set_yticklabels([r"$N - \hat{k}^*$", r"$N$"], fontsize=14, fontweight="bold")
ax.set_xlabel("Time (iterations)", fontsize=11.5)
ax.set_ylabel("# decoding requests in batch", fontsize=11.5)

# Title
ax.set_title(
    "EB inventory dynamics: batch oscillates between "
    r"$N$ and $N - \hat{k}^*$",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

# Legend (phase shading)
decode_patch = mpatches.Patch(
    color="#2E8B57", alpha=0.25, label="Phase 1: Decode  (advance all active requests)"
)
refill_patch = mpatches.Patch(
    color="#E8A33D",
    alpha=0.30,
    label=r"Phase 2: Refill  (prefill $\hat{k}^*$ new requests)",
)
ax.legend(
    handles=[decode_patch, refill_patch],
    loc="lower right",
    fontsize=10.5,
    framealpha=0.95,
    edgecolor="#888",
)

plt.tight_layout()
out_path = "/data/yuzhou/projects/aproj/vllm-sched/eb-vllm/assets/eb_phase_machine.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
