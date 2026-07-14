"""
Plot the LFP SSM figure

"""
__date__ = "September 2025"

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import stats_to_colors
from src.von_mises import nu

C, F = 62, 30
out_dir = ROOT / "data" / "mouse_tg"

if __name__ == '__main__':
    phi0 = np.load(out_dir / "sleep_0_sgd_phi.npy", allow_pickle=True).item()["phi"]
    phi1 = np.load(out_dir / "sleep_1_sgd_phi.npy", allow_pickle=True).item()["phi"]
    stats0 = np.load(out_dir / "sleep_0_sgd_stats.npz")["stats"]
    stats1 = np.load(out_dir / "sleep_1_sgd_stats.npz")["stats"]

    phi0 = phi0[...,0] + 1j * phi0[...,1] # (CF, CF)
    phi1 = phi1[...,0] + 1j * phi1[...,1] # (CF, CF)
    stats0 = stats0[...,0] + 1j * stats0[...,1] # (CF, CF)
    stats1 = stats1[...,0] + 1j * stats1[...,1] # (CF, CF)

    # Nonlinearity
    phi0 = jnp.exp(1j * jnp.angle(phi0)) * nu(jnp.abs(phi0))

    # Save nu(phi).
    rgb = stats_to_colors(phi0, mode='complex')
    rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
    img = Image.fromarray(np.array(rgb))
    img.save(out_dir / "nu_phi_wake.png")

    # Stats and phi details.
    # Detail 1:
    stats_detail = stats0[F:3*F,F:3*F]
    phi_detail = phi0[F:3*F,F:3*F]
    # Detail 2:
    stats_detail = stats0[-6*F:-4*F,-12*F:-10*F]
    phi_detail = phi0[-6*F:-4*F,-12*F:-10*F]

    stats_rgb = stats_to_colors(stats_detail, mode='complex')
    phi_rgb = stats_to_colors(phi_detail, mode='complex')

    plt.imshow(stats_rgb, extent=[1,110,110,1])
    plt.xticks([2, 20, 40, 56, 76, 96], ['1', '20', '40', '1', '20', '40'])
    plt.yticks([2, 20, 40, 56, 76, 96], ['1', '20', '40', '1', '20', '40'])
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("Freq. (Hz)")
    plt.title("$S_{Wake}$")
    plt.savefig(out_dir / "stats_phi_detail.pdf")
    plt.close("all")

    # Plot S and phi angles.
    thresh_phi = jnp.quantile(jnp.abs(phi0), 0.9)
    thresh_stats = jnp.quantile(jnp.abs(stats0), 0.9)
    print("thresh_phi:", thresh_phi)
    print("thresh_stats:", thresh_stats)

    idx0 = jnp.argwhere(jnp.abs(phi0).flatten() > thresh_phi).flatten()
    idx1 = jnp.argwhere(jnp.abs(stats0).flatten() > thresh_stats).flatten()
    histogram_kwargs = dict(bins=81, range=(-jnp.pi, jnp.pi), density=True)
    
    symm_angles_0 = jnp.angle(stats0).flatten()[idx1]
    symm_angles_0 = jnp.concatenate([symm_angles_0, -symm_angles_0], 0)
    val0, bin_edges = np.histogram(symm_angles_0, **histogram_kwargs)
    symm_angles_1 = jnp.angle(phi0).flatten()[idx0]
    symm_angles_1 = jnp.concatenate([symm_angles_1, -symm_angles_1], 0)
    val1, bin_edges = np.histogram(symm_angles_1, **histogram_kwargs)

    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.subplots(figsize=(4,3))
    plt.plot(x, np.log(val0), lw=1.5, label="Wake $S$", c="k", ls='--')
    plt.plot(x, np.log(val1), lw=1.5, label="Wake $\phi$", c="k")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Log Density")

    plt.xticks([-jnp.pi, 0, np.pi], ['$-\pi$', '0', '$\pi$'])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_dir / "stats_phi_log_density.pdf")
    plt.close("all")

    # Connection strength plot.
    thresh_phi = jnp.quantile(jnp.abs(phi0), 0.9)
    phi0 = jnp.transpose(phi0.reshape(C,F,C,F), (0,2,1,3))
    phi1 = jnp.transpose(phi1.reshape(C,F,C,F), (0,2,1,3))
    idx = jnp.arange(F)
    phi0 = phi0[:,:,idx,idx]
    phi1 = phi1[:,:,idx,idx]
    norm_diff = jnp.abs(phi0) - jnp.abs(phi1)

    plt.subplots(figsize=(4,3))
    freq = jnp.linspace(1, 55, F)
    for i in range(1,C):
        for j in range(0,i):
            if jnp.max(jnp.abs(phi0[i,j])) > thresh_phi:
                plt.plot(freq, norm_diff[i,j], alpha=0.2, c='k', lw=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("$|\phi_{Wake}| - |\phi_{NREM}|$")

    for direc in ["top", "right"]:
        plt.gca().spines[direc].set_visible(False)

    plt.gca().grid()
    plt.tight_layout()
    plt.savefig(out_dir / "phi_connection_strength.pdf")
    
    