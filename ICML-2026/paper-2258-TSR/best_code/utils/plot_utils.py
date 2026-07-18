import torch
import matplotlib.pyplot as plt


def plot_sample_density(samples, ax=None, num_points=100, bandwidth=None,
                        cmap='viridis', num_levels=50, margin=3.0, 
                        xy_lims=None, vlims=[None, None], **kwargs):
    """
    Estimate and plot the 2D density of given samples via Gaussian KDE.

    Args:
        samples: torch.Tensor or array-like of shape (M, 2). The 2D points.
        ax:      matplotlib Axes or None. If provided, plot into this axis; 
                 otherwise, a new figure+axis is created.
        num_points: int. Resolution of grid along each axis.
        bandwidth: float or None. KDE bandwidth h. If None, use a rule-of-thumb:
            h = mean(std) * M^(-1/6) for 2D data.
        cmap:    colormap for filled contours.
        num_levels: int. Number of contour levels for filled contour.
        margin:  float. Multiplier for extending the plot range beyond sample std.

    Returns:
        ax: the matplotlib Axes containing the plot.
    """
    # Convert samples to torch.Tensor on CPU (or GPU if desired)
    if not torch.is_tensor(samples):
        pts = torch.tensor(samples, dtype=torch.float32)
    else:
        pts = samples.detach().cpu().float()

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("samples must have shape (M, 2)")
    M = pts.shape[0]
    if M == 0:
        raise ValueError("samples must contain at least one point")

    # Compute bounding box: mean ± margin * std
    mean = torch.mean(pts, dim=0)     # (2,)
    std  = torch.std(pts, dim=0)      # (2,)
    # If std is zero in any dimension (all samples identical), set small positive
    std = torch.where(std < 1e-6, torch.tensor(1e-6), std)

    if xy_lims is None:
        x_min = (mean[0] - margin * std[0]).item()
        x_max = (mean[0] + margin * std[0]).item()
        y_min = (mean[1] - margin * std[1]).item()
        y_max = (mean[1] + margin * std[1]).item()
    else:
        x_min, x_max, y_min, y_max = xy_lims

    # Determine device for computation: use CPU or GPU if pts on GPU
    device = pts.device

    # Bandwidth selection if None: rule-of-thumb for 2D
    if bandwidth is None:
        # Silverman-like: h ~ std * M^{-1/(d+4)}, here d=2 → exponent = -1/6
        # Use average of per-dim std:
        avg_std = float(torch.mean(std).item())
        h = avg_std * (M ** (-1/6))
        # Avoid extremely small h
        if h <= 0:
            h = 1e-3
    else:
        h = float(bandwidth)
    h_tensor = torch.tensor(h, device=device, dtype=torch.float32)

    # Create grid
    x = torch.linspace(x_min, x_max, num_points, device=device)
    y = torch.linspace(y_min, y_max, num_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')  # (num_points, num_points)
    grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (G,2)
    G = grid_points.shape[0]

    # Compute pairwise squared distances between grid points and samples
    # torch.cdist gives Euclidean distances; square them
    # Move pts to same device
    pts_dev = pts.to(device)
    # cdist may use a lot of memory if G*M large; ensure M, num_points moderate
    dists = torch.cdist(grid_points, pts_dev)  # (G, M)
    sq_dists = dists.pow(2)                    # (G, M)

    # Compute KDE: Gaussian kernel
    # K(x) = (1/(2π h^2)) * exp(-||x - xi||^2 / (2 h^2))
    # density at grid point j: (1/M) * sum_{i=1}^M K(x_j, x_i)
    # Compute kernel values:
    exp_term = torch.exp(-0.5 * sq_dists / (h_tensor ** 2))  # (G, M)
    coef = 1.0 / (2 * torch.pi * (h_tensor ** 2))            # scalar
    # Sum over samples:
    density_vals = coef * exp_term.mean(dim=1)               # (G,)
    # Reshape to grid
    density_grid = density_vals.reshape(num_points, num_points).cpu().detach().numpy()

    # Prepare plotting
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Setup axis
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Filled contour
    cf = ax.contourf(x_np, y_np, density_grid, levels=num_levels, vmin=vlims[0], vmax=vlims[1], **kwargs)
    # Optionally overlay contour lines for reference
    cs = ax.contour(x_np, y_np, density_grid, levels=int(num_levels/5), linewidths=0.5, alpha=0.5, vmin=vlims[0], vmax=vlims[1], **kwargs)
    if created_fig:
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('Estimated density')

    ax.set_title(f'Sample KDE density (M={M}, h={h:.3f})')

    if save_path := None:
        pass  # no save here; user can save externally
    # If user wants to save, they can do: fig.savefig(...) externally when ax is None.

    return ax

def plot_sample_points(samples, ax=None, s=5, alpha=0.6, margin=3.0, xy_lims=None, color='blue', **kwargs):
    """
    Plot 2D sample points with optional axis setup and limits.

    Args:
        samples: torch.Tensor or array-like of shape (M, 2). The 2D points.
        ax:      matplotlib Axes or None. If provided, plot into this axis;
                 otherwise, a new figure+axis is created.
        s:       Marker size.
        alpha:   Marker transparency.
        margin:  float. Multiplier for extending the plot range beyond sample std.
        xy_lims: Optional (x_min, x_max, y_min, y_max) tuple.
        color:   Marker color.

    Returns:
        ax: the matplotlib Axes containing the plot.
    """
    import torch
    import matplotlib.pyplot as plt

    if not torch.is_tensor(samples):
        pts = torch.tensor(samples, dtype=torch.float32)
    else:
        pts = samples.detach().cpu().float()

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("samples must have shape (M, 2)")
    M = pts.shape[0]
    if M == 0:
        raise ValueError("samples must contain at least one point")

    # Compute bounding box: mean ± margin * std
    mean = torch.mean(pts, dim=0)     # (2,)
    std  = torch.std(pts, dim=0)      # (2,)
    std = torch.where(std < 1e-6, torch.tensor(1e-6), std)

    if xy_lims is None:
        x_min = (mean[0] - margin * std[0]).item()
        x_max = (mean[0] + margin * std[0]).item()
        y_min = (mean[1] - margin * std[1]).item()
        y_max = (mean[1] + margin * std[1]).item()
    else:
        x_min, x_max, y_min, y_max = xy_lims

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=alpha, color=color)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Sample points (M={M})')

    if created_fig:
        plt.tight_layout()

    return ax