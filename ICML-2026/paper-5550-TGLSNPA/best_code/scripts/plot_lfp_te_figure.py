"""
Make the plot for the MVTE LFP plots.

"""
__date__ = "September 2025"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGIONS = [
    'Amy', 'Cg_Cx', 'DL_OFC', 'DL_Str', 'DM_Str', 'IL_Cx', 'LH', 'M1',
    'MD_Thal', 'NAc', 'PrL', 'S1', 'SNR', 'V1', 'VTA', 'aIns', 'dHipp', 'vHipp',
]


def _channel_region(ch):
    for r in REGIONS:
        if ch == r or ch.startswith(r + '_'):
            return r
    return ch


def censor_te_within_regions(te, channels):
    C = len(channels)
    for region in REGIONS:
        idx = get_region_idx(region, channels)
        if len(idx):
            te[:, idx[0]:idx[-1]+1, idx[0]:idx[-1]+1] = np.nan
    return te


def get_region_idx(region, channels):
    return np.array([i for i in range(len(channels)) if _channel_region(channels[i]) == region])


if __name__ == '__main__':
    data_fn = ROOT / "data" / "lfp_data" / "torus_data.npz"
    channels = np.load(data_fn)["channels"]
    channels = np.array(channels).astype(str)
    print(channels)
    C = len(channels)

    te_dir = ROOT / "data" / "mouse_mvte"

    # Frequencies (Hz) corresponding to TE arrays
    freqs = np.linspace(1, 55, 30)  # shape (F,)

    # Load TE (F, C, C). sleep_0 = wake, sleep_1 = NREM
    te_wake = np.load(te_dir / "sleep_0_TE.npz")["te"].clip(0, None)
    te_nrem = np.load(te_dir / "sleep_1_TE.npz")["te"].clip(0, None)

    te_wake = censor_te_within_regions(te_wake, channels)
    te_nrem = censor_te_within_regions(te_nrem, channels)

    idx = np.argmin(np.abs(freqs - 18.0))
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(10,8))

    for i in range(2):
        for j in range(2):
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])

    axarr[0,0].set_ylabel("Wake TE")
    axarr[1,0].set_ylabel("Wake - NREM TE")

    vmax = np.nanquantile(te_wake[idx], 0.99)
    print("te_wake[idx] vmax:", vmax)
    vmax = 0.0056 # hard-coding
    axarr[0,0].imshow(te_wake[idx], vmin=0, vmax=vmax, cmap='binary')
    axarr[0,0].set_title("Wake, 18 Hz")

    diff = te_wake[idx] - te_nrem[idx]
    vmax = np.nanquantile(np.abs(diff), 0.99)
    print("np.abs(diff) vmax:", vmax)
    print(vmax)
    vmax = 0.002 # hard-coding
    axarr[1,0].imshow(diff, vmin=-vmax, vmax=vmax, cmap='bwr')
    axarr[1,0].set_title("Wake - NREM, 18 Hz")

    idx = np.argmin(np.abs(freqs - 45.0))
    vmax=np.nanquantile(te_wake[idx], 0.99)
    print("te_wake[idx] vmax:", vmax)
    print(vmax)
    vmax = 0.0056
    im01 = axarr[0,1].imshow(te_wake[idx], vmin=0, vmax=vmax, cmap='binary')
    axarr[0,1].set_title("Wake, 45 Hz")
    div01 = make_axes_locatable(axarr[0,1])
    cax01 = div01.append_axes('right', size='5%', pad=0.05)
    cbar01 = plt.colorbar(im01, cax=cax01)
    fmt01 = ScalarFormatter(useMathText=True)
    fmt01.set_powerlimits((0, 0))
    cbar01.formatter = fmt01
    cbar01.update_ticks()

    diff = te_wake[idx] - te_nrem[idx]
    vmax = np.nanquantile(np.abs(diff), 0.99)
    print("np.abs(diff) vmax:", vmax)
    print(vmax)
    vmax = 0.002
    im11 = axarr[1,1].imshow(diff, vmin=-vmax, vmax=vmax, cmap='bwr')
    axarr[1,1].set_title("Wake - NREM, 45 Hz")
    div11 = make_axes_locatable(axarr[1,1])
    cax11 = div11.append_axes('right', size='5%', pad=0.05)
    cbar11 = plt.colorbar(im11, cax=cax11)
    fmt11 = ScalarFormatter(useMathText=True)
    fmt11.set_powerlimits((0, 0))
    cbar11.formatter = fmt11
    cbar11.update_ticks()

    # Plot asymmetry score.
    prl_idx = get_region_idx("PrL", channels)
    str_idx = get_region_idx("DM_Str", channels)
    thal_idx = get_region_idx("MD_Thal", channels)
    cg_idx = get_region_idx("Cg_Cx", channels)
    vta_idx = get_region_idx("VTA", channels)
    snr_idx = get_region_idx("SNR", channels)
    amy_idx = get_region_idx("Amy", channels)
    il_idx = get_region_idx("IL_Cx", channels)


    # Plot Wake - NREM TE.
    diff = te_wake - te_nrem
    plt.sca(axarr[0,2])
    d = C

    # plt.imshow(diff[-1], interpolation='nearest')
    flag1, flag2, flag3 = True, True, True
    for i in range(d):
        for j in range(d):
            if np.isnan(diff[0,i,j]):
                continue
            if i in prl_idx and j in str_idx:
                if flag1:
                    flag1 = False
                    plt.plot(freqs, diff[:,i,j], c='goldenrod', lw=0.6, alpha=0.6, zorder=3, label=r"PFC $\rightarrow$ Striatum")
                else:
                    plt.plot(freqs, diff[:,i,j], c='goldenrod', lw=0.6, alpha=0.6, zorder=3)
            elif i in il_idx and j in cg_idx:
                if flag2:
                    flag2 = False
                    plt.plot(freqs, diff[:,i,j], c='mediumpurple', lw=0.6, alpha=0.6, zorder=3, label=r"IL $\rightarrow$ Cg, PFC")
                else:
                    plt.plot(freqs, diff[:,i,j], c='mediumpurple', lw=0.6, alpha=0.6, zorder=3)
            elif i in il_idx and j in prl_idx:
                plt.plot(freqs, diff[:,i,j], c='mediumpurple', lw=0.6, alpha=0.6, zorder=3)
            elif i in vta_idx and j in snr_idx:
                if flag3:
                    flag3 = False
                    plt.plot(freqs, diff[:,i,j], c='mediumseagreen', lw=0.6, alpha=0.6, zorder=3, label=r"VTA $\rightarrow$ SNr")
                else:
                    plt.plot(freqs, diff[:,i,j], c='mediumseagreen', lw=0.6, alpha=0.6, zorder=3)
            plt.plot(freqs, diff[:,i,j], c='k', lw=0.5, alpha=0.4)
    # plt.legend(loc='best')
    plt.xlabel("Frequency (Hz)")
    plt.title("Wake - NREM TE")


    plt.sca(axarr[1,2])
    plt.axhline(y=0.0, c='k', ls='--', alpha=0.7)
    te_wake_t = np.transpose(te_wake, (0,2,1))
    asymm = (te_wake - te_wake_t) / (te_wake + te_wake_t + 1e-3)

    lines_1, lines_2, lines_3 = [], [], []
    for i in range(d):
        for j in range(d):
            if np.isnan(diff[0,i,j]):
                continue

            if i in prl_idx and j in str_idx:
                # plt.plot(freqs, asymm[:,i,j], c='goldenrod', lw=0.6, alpha=0.6, zorder=3)
                lines_1.append(asymm[:,i,j])
            elif i in il_idx and j in cg_idx:
                # plt.plot(freqs, asymm[:,i,j], c='mediumpurple', lw=0.6, alpha=0.6, zorder=3)
                lines_2.append(asymm[:,i,j])
            elif i in il_idx and j in prl_idx:
                # plt.plot(freqs, asymm[:,i,j], c='mediumpurple', lw=0.6, alpha=0.6, zorder=3)
                lines_2.append(asymm[:,i,j])
            elif i in vta_idx and j in snr_idx:
                # plt.plot(freqs, asymm[:,i,j], c='mediumseagreen', lw=0.6, alpha=0.6, zorder=3)
                lines_3.append(asymm[:,i,j])
            # tmp = np.abs(diff[:,i,j]) / (0.001 + np.maximum(te_wake[:,i,j], te_nrem[:,i,j]))
            # plt.plot(freqs, tmp, c='k', lw=0.5, alpha=0.4)
        
    lines_1 = np.array(lines_1)
    lines_2 = np.array(lines_2)
    lines_3 = np.array(lines_3)

    plt.fill_between(
        freqs,
        np.min(lines_2, axis=0),
        np.max(lines_2, axis=0),
        fc='mediumpurple',
        alpha=0.5,
        ec='mediumpurple',
        label=r"IL $\rightarrow$ Cg, PFC",
    )
    plt.fill_between(
        freqs,
        np.min(lines_3, axis=0),
        np.max(lines_3, axis=0),
        fc='mediumseagreen',
        alpha=0.5,
        ec='mediumseagreen',
        label=r"VTA $\rightarrow$ SNr",
    )
    plt.fill_between(
        freqs,
        np.min(lines_1, axis=0),
        np.max(lines_1, axis=0),
        fc='goldenrod',
        alpha=0.5,
        ec='goldenrod',
        label=r"PFC $\rightarrow$ Striatum",
    )
    
    plt.legend(loc='best')
    plt.title("Edge Asymmetry")

    plt.tight_layout()
    plt.savefig("plot_lfp_te_figure.pdf")
    plt.savefig("plot_lfp_te_figure.png")


###