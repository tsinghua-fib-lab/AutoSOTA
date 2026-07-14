"""
Plot the spindle PLV and TG-HMM figure.

1) We look at PLV differences near and far from spindles, which show dense differences.
2) Then we fit a TG-HMM and find a state associated with spindles.
3) This state shows sparse and interpretable differences from other states.
"""
__date__ = "September 2025"

import jax.numpy as jnp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import stats_to_colors


channels = [
    "MD_Thal_01",
    "MD_Thal_02",
    "Cg_Cx_L_01",
    "Cg_Cx_R_01",
    "IL_Cx_L_01",
    "PrL_Cx_L_01",
    "PrL_Cx_R_01",
    "S1_Cx_01",
    "dHipp_01",
    "vHipp_01"
]
F = 6
K = 7

tick_vals = [F * (i+0.5) for i in [0.5, 2.5, 4, 5.5, 7, 8.5]]
reduced_channels = ["Thal", "Cg Cx", "IL Cx", "PrL Cx", "S1 Cx", "Hipp"]

if __name__ == '__main__':
    d = jnp.load(os.path.join(ROOT, "data", "spindle_data", "spindles_refined_sorted.npz"))
    print(list(d.keys()))
    lfps = d["windows_raw"][:,:,d["ref_channel"]] # [N,T]
    del d
    N, T = lfps.shape
    print(lfps.shape)
    ts = np.linspace(-0.75, 0.75, T)

    fig, axarr = plt.subplots(ncols=4, figsize=(10,2.8))
    axarr = axarr.flatten()
    plt.sca(axarr[0])
    plt.title("Aligned Spindles")
    vmax = np.quantile(np.abs(lfps),0.99)
    im = plt.imshow(
        lfps,
        aspect=1,
        origin="lower",
        extent=[-750,750,1,N],
        cmap="PRGn",
        vmin=-vmax,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ticks=[-vmax, 0, vmax])
    cbar.ax.set_yticklabels(["min", "0", "max"])
    plt.ylabel("Spindle # (sorted)")
    plt.xlabel("Time (ms)")

    # Plot ΔPLV
    plt.sca(axarr[1])
    full_data = jnp.load("spindle_cwt.npy") # (N,T,C,F)
    print("Original data shape:", full_data.shape)
    ts = np.linspace(-0.75, 0.75, full_data.shape[1])[::2]
    full_data = full_data[:,::2]
    N, T, C, F = full_data.shape
    full_data = full_data.reshape(N, T, -1).astype(jnp.float32)
    print("Modified data shape:", full_data.shape)
    mask = (np.abs(ts) < 0.2).astype(jnp.bool)
    phases_1 = full_data[:,mask].reshape(-1,C*F) # (N1, CF)
    print(phases_1.shape)
    phases_2 = full_data[:,~mask].reshape(-1,C*F) # (N2, CF)
    print(phases_2.shape)
    plv_1 = jnp.abs(jnp.mean(jnp.exp(1j * (phases_1[:,None] - phases_1[:,:,None])), axis=0))
    plv_2 = jnp.abs(jnp.mean(jnp.exp(1j * (phases_2[:,None] - phases_2[:,:,None])), axis=0))
    diff = plv_1 - plv_2
    vmax = np.max(np.abs(diff))
    plt.title("$\Delta$ PLV")
    plt.imshow(diff, vmin=-vmax, vmax=vmax, cmap='bwr')
    plt.colorbar()
    ax = plt.gca()
    plt.yticks(tick_vals, reduced_channels)

    # Plot TG-HMM Timecourse
    plt.sca(axarr[2])
    zs = jnp.load(os.path.join(ROOT, "data", "spindle_data", f"hmm_spindle_all_z_seq_{K}.npz"))["all_seq"]
    unique_states = np.unique(zs)
    print(
        "unique_states", unique_states,
    )

    # Get colormap
    colors = ["gray" for i in unique_states]
    colors[3] = "mediumpurple"

    # Average across N (over trials, at each time)
    occupancy_T = np.array([
        np.mean(zs == s, axis=0) for s in unique_states
    ])  # shape: (num_states, T)
    # plt.axvline(x=0, c='k', alpha=0.6, lw=0.5, ls='--')
    for i, s in enumerate(unique_states):
        if i == 4:
            plt.plot(1e3 * ts, occupancy_T[i], color=colors[i], label="Other States")
        elif i == 3:
            plt.plot(1e3 * ts, occupancy_T[i], color=colors[i], label="Spindle State")
        plt.plot(1e3 * ts, occupancy_T[i], color=colors[i])
    plt.legend(loc="best")
    plt.title("TG-HMM State Occupancy")
    plt.xlabel("Time (ms)")
    plt.ylabel("Average State Occupancy")

    # Plot Δ phi
    plt.sca(axarr[3])
    d = joblib.load(os.path.join(ROOT, "data", "spindle_data", f"hmm_spindle_info_{K}.joblib"))
    avg_occ = np.mean(occupancy_T, axis=1)
    idx = np.argmax(avg_occ)
    print("idx:", idx)
    avg_occ /= np.sum([val for i, val in enumerate(avg_occ) if i != idx])

    phis = d["phis"]
    phi_spindle = phis[idx]
    phi_other = sum(w * phi for i, (w, phi) in enumerate(zip(avg_occ, phis)) if i != idx)

    phi_diff = phi_spindle - phi_other
    r_max = np.quantile(np.abs(phi_diff[...,0] + 1j * phi_diff[..., 1]), 1.0)
    print("r_max:", r_max)
    rgb = stats_to_colors(phi_diff, r_max=r_max, mode="expanded_complex")
    plt.imshow(rgb)
    plt.xticks([])
    plt.yticks([])
    plt.title("$\Delta \ \phi$")

    plt.tight_layout()
    plt.savefig("hmm_spindle_plot.pdf")
    plt.close("all")

    # Mean spindle waveform per channel
    d2 = jnp.load(os.path.join(ROOT, "data", "spindle_data", "spindles_refined_sorted.npz"))
    all_windows = np.array(d2["windows_raw"])  # [N, T, C]
    ts_waveform = np.linspace(-0.75, 0.75, all_windows.shape[1])
    mean_waveforms = all_windows.mean(axis=0)  # [T, C]
    n_channels = mean_waveforms.shape[1]
    channel_labels = channels[:n_channels]

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    colors = plt.cm.tab20(np.linspace(0, 1, n_channels))

    for c in range(n_channels):
        ax2.plot(
            1e3 * ts_waveform,
            mean_waveforms[:, c],
            label=channel_labels[c],
            lw=0.8,
            alpha=0.9,
            color=colors[c],
        )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Mean LFP (AU)")
    ax2.set_title("Mean Spindle Waveform by Channel")
    for dir in ["top", "right"]:
        ax2.spines[dir].set_visible(False)
    ax2.legend(loc="upper left", fontsize=7, ncols=2, frameon=False)
    ax2.grid(alpha=0.15, linewidth=0.5)
    fig2.tight_layout()
    fig2.savefig("spindle_mean_waveforms.pdf")
    plt.close("all")

    # TG-HMM state occupancy for K = 3..8, all on one subplot
    K_values = list(range(3, 9))
    spindle_colors = plt.cm.tab20(np.linspace(0, 1, len(K_values)))
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    legend_handles = []
    for ki, k in enumerate(K_values):
        zs_k = jnp.load(os.path.join(ROOT, "data", "spindle_data", f"hmm_spindle_all_z_seq_{k}.npz"))["all_seq"]
        ts_k = np.linspace(-0.75, 0.75, zs_k.shape[1])
        unique_states_k = np.unique(zs_k)
        occupancy_k = np.array([np.mean(zs_k == s, axis=0) for s in unique_states_k])
        spindle_idx = int(np.argmax(occupancy_k.mean(axis=1)))
        for i in range(len(unique_states_k)):
            if i == spindle_idx:
                ax3.plot(1e3 * ts_k, occupancy_k[i], color=spindle_colors[ki], lw=1.2, alpha=0.9)
            else:
                ax3.plot(1e3 * ts_k, occupancy_k[i], color="gray", lw=0.6, alpha=0.4)
        legend_handles.append(plt.Line2D([0], [0], color=spindle_colors[ki], lw=1.5, label=f"K={k} spindle"))
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Average State Occupancy")
    ax3.set_title("TG-HMM Spindle State Occupancy")
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    ax3.grid(alpha=0.15, linewidth=0.5)
    ax3.legend(handles=legend_handles, fontsize=7, frameon=False, loc="center", ncols=2)
    fig3.tight_layout()
    fig3.savefig("hmm_state_occupancy_all_K.pdf")
    fig3.savefig("hmm_state_occupancy_all_K.png")
    plt.close("all")