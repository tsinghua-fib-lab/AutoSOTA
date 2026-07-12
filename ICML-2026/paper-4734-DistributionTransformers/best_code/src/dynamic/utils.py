
import torch
from torch import Tensor
from torch.distributions import Distribution

import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Union


def plot_filtered_series(filter_distributions: Union[list[Distribution], Distribution],
                         series: Optional[Tensor] = None,
                         bounds: Optional[tuple[float, float]] = None,
                         n_vertical: int = 1000,
                         axis_label: str = "Horizontal Displacement (km)",
                         cmaps: Union[str, list[str]] = "BuPu",
                         legend_labels: Union[str, list[str]] = "Filtered Series",
                         *args,
                         **kwargs) -> plt.Figure:
    plt.style.use(['seaborn-v0_8-paper'])

    if bounds is None:
        assert series is not None, "series_bounds cannot be inferred if true_series is not provided"
        bounds = (series.max().item() + 1., series.min().item() - 1.)

    if isinstance(filter_distributions, Distribution):
        filter_distributions = [filter_distributions]
    if isinstance(cmaps, str):
        cmaps = [cmaps]
    if isinstance(legend_labels, str):
        legend_labels = [legend_labels]

    x = torch.arange(series.shape[0], dtype=torch.float32) * 10
    y = torch.linspace(bounds[0], bounds[1], n_vertical)
    X, Y = torch.meshgrid(x, y)
    Z = [dist.log_prob(Y.T.unsqueeze(-1)).exp() for dist in filter_distributions]
    fig, ax = plt.subplots()
    ax.set_xlim((x[0], x[-1]))
    ax.set_ylim(bounds)
    for i, (Zi, cmap, legend_label) in enumerate(zip(Z, cmaps, legend_labels)):
        norm = colors.Normalize(vmin=Zi.min(), vmax=Zi.max())
        alpha_map = Zi / Zi.max(dim=-1, keepdim=True)[0]
        alpha_map = alpha_map.nan_to_num(0)
        plt.imshow(Zi, aspect="auto", origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                   cmap=cmap, norm=norm, alpha=alpha_map)
        plt.colorbar(label=legend_label)
        #cbar.solids.set(alpha=1)
    plt.plot(x, series, "k")
    plt.xlabel("Time (s)")
    plt.ylabel(axis_label)
    plt.legend(["Actual Trajectory"])
    plt.show()
    return fig
