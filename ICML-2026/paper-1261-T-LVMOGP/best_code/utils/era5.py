import h5py
from typing import Optional
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def era5_plot(
        data, timepoint: int, mask: Optional[np.ndarray] = None,
        supplementary_info_path="./data/raw_data/ERA5/plot_supplementary_info.h5",
        exp_result_folder: Optional[str] = None, vmin: float = -3, vmax: float = 2. # TODO: fix color-value mapping
):
    """
    ERA5 UK region
    data: np.ndarray, of shape (num_inputs, num_outputs) = (num_times, num_spots), i.e. (30, 3395)
    mask: np.ndarray, same shape as data, True for observed (to plot), False for missing (leave as np.nan)
    """
    with h5py.File(supplementary_info_path, "r") as f:
        nan_mask = f["nan_mask"][:]
        t2m_uk_lat = f["t2m_uk_lat"][:]
        t2m_uk_lon = f["t2m_uk_lon"][:]

    if mask is not None:
        data = data.copy()
        data[~mask] = np.nan

    field = np.full(nan_mask.shape, np.nan)
    field[~nan_mask] = data[timepoint, :]

    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pcm = ax.pcolormesh(
        t2m_uk_lon, t2m_uk_lat, field, cmap = "coolwarm", transform=ccrs.PlateCarree(), shading="auto",
    )  # for cmap: RdYlBu_r, coolwarm, bwr, seismic, viridis, Blues, Reds
    contours = ax.contour(
        t2m_uk_lon, t2m_uk_lat, field, levels=10, colors='black', linewidths=0.5, alpha=0.3
    )
    ax.coastlines()

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, label="K")
    ax.set_title("ERA5 UK 2m Temperature")
    plt.tight_layout()
    if exp_result_folder is not None:
        plt.savefig(f"{exp_result_folder}/era5_uk.pdf", bbox_inches="tight")