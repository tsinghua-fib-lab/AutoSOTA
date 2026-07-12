import numpy as np
import torch
import gpytorch
# (Removed scipy.optimize.minimize)


def optimize_t_for_x_batch_torch(
    model,
    x_batch_torch: torch.Tensor,
    objective: str,
    t_grid_size: int = 101,
    beta: float = 1.96,
    x_chunk_size: int = 256,
    pred_batch_size: int = 4096,
):
    """
    Parallel search for optimal t on [0, 1] for a batch of x (N, dim_x).
    Full GPU computation and memory efficient.

    Parameters
    ----------
    model : GPModel
        Trained GP model wrapper.
    x_batch_torch : torch.Tensor
        Input tensor of shape (N, dim_x), must be on the correct device/dtype.
    objective : str
        Objective function, supports:
            - 'mean'      : Maximize mean
            - 'variance'  : Maximize variance
            - 'ucb'       : Maximize UCB = mean + beta * std
            - 'lcb'       : Maximize LCB = mean - beta * std
            - 'slope_ucb' : Maximize |d mean / dt| * std
    t_grid_size : int, default 101
        Number of discrete points in the t grid.
    beta : float, default 1.96
        Confidence parameter for UCB/LCB.
    x_chunk_size : int, default 256
        Chunk size for x dimension to avoid OOM.
    pred_batch_size : int, default 4096
        Internal batch size for GP model predictions.

    Returns
    -------
    best_t_batch : torch.Tensor (N,)
        Optimal t for each x.
    best_val_batch : torch.Tensor (N,)
        Corresponding optimal objective value.
    """
    N_eval, dim_x = x_batch_torch.shape

    if N_eval == 0:
        empty = torch.empty(0, device=x_batch_torch.device, dtype=x_batch_torch.dtype)
        return empty, empty

    device = model.device
    dtype = model.dtype

    if x_batch_torch.device != device or x_batch_torch.dtype != dtype:
        x_batch_torch = x_batch_torch.to(device=device, dtype=dtype)

    # [t_grid_size, 1]
    t_grid = torch.linspace(0.0, 1.0, t_grid_size, device=device, dtype=dtype).view(-1, 1)
    if t_grid_size > 1:
        dt = (t_grid[1] - t_grid[0]).item()
    else:
        dt = 1.0  # Fallback for degenerate case

    best_t_chunks = []
    best_val_chunks = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_chunk in torch.split(x_batch_torch, x_chunk_size, dim=0):
            n_chunk = x_chunk.shape[0]
            if n_chunk == 0:
                continue

            # Cartesian product for current sub-batch (t, x)
            # x_rep: [n_chunk * t_grid_size, dim_x]
            x_rep = x_chunk.repeat_interleave(t_grid_size, dim=0)
            # t_rep: [n_chunk * t_grid_size, 1]
            t_rep = t_grid.expand(n_chunk, t_grid_size, 1).reshape(-1, 1)
            # super_batch: [n_chunk * t_grid_size, dim_x + 1]
            super_batch = torch.cat([x_rep, t_rep], dim=1)

            mean_list = []
            var_list = []
            for sb in torch.split(super_batch, pred_batch_size, dim=0):
                # pred = model.likelihood(model(sb))
                pred = model(sb)
                mean_list.append(pred.mean)
                var_list.append(pred.variance)

            if not mean_list:
                raise RuntimeError("optimize_t_for_x_batch_torch: prediction chunks are empty.")

            mean_flat = torch.cat(mean_list, dim=0)  # [n_chunk * t_grid_size]
            var_flat = torch.cat(var_list, dim=0)    # [n_chunk * t_grid_size]

            # Reshape to [n_chunk, t_grid_size]
            mean_grid = mean_flat.view(n_chunk, t_grid_size)
            var_grid = var_flat.view(n_chunk, t_grid_size)
            std_grid = var_grid.clamp_min(1e-9).sqrt()

            # Construct values_grid: [n_chunk, t_grid_size]
            if objective == 'mean':
                values_grid = mean_grid

            elif objective == 'variance':
                values_grid = var_grid

            elif objective == 'ucb':
                values_grid = mean_grid + beta * std_grid

            elif objective == 'lcb':
                values_grid = mean_grid - beta * std_grid

            elif objective == 'slope_ucb':
                # Slope * Uncertainty:
                # 1) Finite difference for d mean / dt
                # 2) Absolute value |d mean / dt|
                # 3) Multiply: |dmean/dt| * std

                values_grid = torch.zeros_like(mean_grid)

                if t_grid_size == 1:
                    pass

                elif t_grid_size == 2:
                    dmu_dt = (mean_grid[:, 1] - mean_grid[:, 0]) / dt
                    slope_mag = dmu_dt.abs()
                    std_avg = 0.5 * (std_grid[:, 0] + std_grid[:, 1])
                    acq = slope_mag * std_avg

                    values_grid[:, 0] = acq
                    values_grid[:, 1] = acq

                else:
                    # Central difference for internal points
                    dmu_dt_mid = (mean_grid[:, 2:] - mean_grid[:, :-2]) / (2.0 * dt)
                    std_mid = std_grid[:, 1:-1]

                    slope_mag_mid = dmu_dt_mid.abs()
                    acq_mid = slope_mag_mid * std_mid

                    values_grid[:, 1:-1] = acq_mid

                    # Fill boundaries with adjacent internal values
                    values_grid[:, 0] = acq_mid[:, 0]
                    values_grid[:, -1] = acq_mid[:, -1]

            else:
                raise ValueError(
                    "Objective must be 'mean', 'variance', 'ucb', 'lcb', or 'slope_ucb'"
                )

            # Maximize over t grid
            best_val, best_idx = torch.max(values_grid, dim=1)
            best_t = t_grid.flatten()[best_idx]

            best_t_chunks.append(best_t)
            best_val_chunks.append(best_val)

    best_t_batch = torch.cat(best_t_chunks, dim=0)
    best_val_batch = torch.cat(best_val_chunks, dim=0)

    return best_t_batch, best_val_batch