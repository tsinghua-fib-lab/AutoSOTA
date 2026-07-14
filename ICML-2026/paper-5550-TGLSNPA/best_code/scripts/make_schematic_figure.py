"""
Make wavelet schematic figures

"""
__date__ = "July - October 2025"


import numpy as np
from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from scipy.signal import morlet2, cwt

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import stats_to_colors
from src.stats import get_stats, solve_tg_exact
from src.von_mises import nu
    


if __name__ == '__main__':
    data_fn = os.path.join(ROOT, "data", "lfp_data", "torus_data.npz")
    save_fmt = ".pdf"

    start_idx = 1000
    fs = 250
    duration = 2

    # load shape (T, C)
    mouse_data = jnp.load(data_fn)["lfps"].astype(jnp.float32)
    # [0,28,31,37,49] = [Amy, MdThal, Nac, PRL, VTa]
    mouse_data = mouse_data[start_idx:start_idx + fs * duration, [0,28,31,37,49]]
    mouse_data = 0.25 * np.asarray(mouse_data)
    T = mouse_data.shape[0]
    t = np.arange(T) / fs

    # --- 1) Raw signal ---
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(t, mouse_data[:, 0], color='k', lw=1.5)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("figure1_raw_signal"+save_fmt)
    

    # --- 2) Extracted phase (unit‐norm / phasor) ---
    freqs = [4, 5, 6]  # Hz
    w = 5.0  # Morlet central frequency parameter
    # scale → width relation: f = w / (2π·scale)  ⇒ scale = w / (2π·f)
    scales = w * fs / (2 * np.pi * np.array(freqs))
    cwt_mat = cwt(mouse_data[:, 0], lambda M, s: morlet2(M, s, w), scales) # [F,T]
    phasors = cwt_mat / (np.abs(cwt_mat) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, f in enumerate(freqs):
        cs = stats_to_colors(phasors[i], mode='complex')
        for j in range(len(cs)):
            ax.plot(t[j:j+2], 3 * i + phasors[i,j:j+2].real, c=cs[j])
        ax.text(-0.2, 3*i, f"$x_{i+1}$")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("figure2_cwt_phase"+save_fmt)
    
    # --- 3) Color wheel plot ---
    fig, ax = plt.subplots(figsize=(2, 2))
    n = 512
    d = jnp.linspace(-1,1,n)
    d = d[None] + 1j * d[:,None]
    d = d.at[jnp.abs(d) > 1.0].set(jnp.nan)
    big_arr = stats_to_colors(d, mode="complex")
    plt.imshow(big_arr, extent=[-1,1,-1,1], origin="lower")
    plt.text(1.1,0,"0", ha="center", va="center")
    plt.text(0,1.1,"π/2", ha="center", va="center")
    plt.text(-1,0,"π", ha="center", va="center")
    plt.text(0,-1.1,"3π/2", ha="center", va="center")
    plt.axis("off")
    fig.tight_layout()
    fig.savefig("figure3_color_wheel"+save_fmt)
    plt.close("all")

    # --- 4) Stats plot ---
    fig, ax = plt.subplots(figsize=(2.2,2.2))
    phases = jnp.angle(cwt_mat).T # [T,F]
    stats = vmap(get_stats)(phases)
    stats = jnp.mean(stats, axis=0)
    big_arr = stats_to_colors(stats, mode="expanded_complex")
    ax.imshow(big_arr)
    for i in range(len(freqs)):
        ax.text(i, i, f"$x_{i+1}$", ha="center", va="center", color="k")
        for j in range(i+1, len(freqs)):
            ax.text(i, j, f"$x_{j+1} - x_{i+1}$", ha="center", va="center", color="k")
        for j in range(i):
            ax.text(i, j, f"$x_{i+1} + x_{j+1}$", ha="center", va="center", color="k")
    ax.set_title(r"TG Stats, $S$")
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    fig.savefig("figure4_phase_stats"+save_fmt)
    plt.close("all")

    # --- 5) Parameter plot ---
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    phi = solve_tg_exact(phases, reg=0.1)
    phi = phi[..., 0] + phi[..., 1]
    phi = jnp.exp(1j * jnp.angle(phi)) * nu(jnp.abs(phi))
    big_arr = stats_to_colors(phi, mode="complex")
    ax.imshow(big_arr)
    for i in range(len(freqs)):
        ax.text(i, i, f"$x_{i+1}$", ha="center", va="center", color="k")
        for j in range(i+1, len(freqs)):
            ax.text(i, j, f"$x_{j+1} - x_{i+1}$", ha="center", va="center", color="k")
        for j in range(i):
            ax.text(i, j, f"$x_{i+1} + x_{j+1}$", ha="center", va="center", color="k")
    ax.set_title(r"TG Parameters, $\nu(\phi)$")
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    fig.savefig("figure5_tg_params"+save_fmt)
    plt.close("all")

