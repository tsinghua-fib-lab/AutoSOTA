import numpy as np
import torch

def CRPS_ensemble(pred_samples: torch.Tensor,
                  y_true: torch.Tensor,
                  reduction: str = "mean"):
    """
    Sample-based CRPS (no distributional assumption).

    pred_samples: [B, N, L, 1]  (N samples/trajectories)
    y_true:       [B, L, 1]

    returns:
      - if reduction="none": [B, L]
      - if reduction="mean": scalar
      - if reduction="sum":  scalar
    """
    # squeeze last dim -> pred: [B, N, L], y: [B, L]
    pred = pred_samples.squeeze(-1)
    y = y_true.squeeze(-1)

    # term1 = E|X - y| over samples  -> [B, L]
    term1 = (pred - y.unsqueeze(1)).abs().mean(dim=1)

    # term2 = 0.5 * E|X - X'| over pairs of samples -> [B, L]
    # pairwise diffs: [B, N, N, L]
    pairwise = (pred.unsqueeze(2) - pred.unsqueeze(1)).abs()
    term2 = 0.5 * pairwise.mean(dim=(1, 2))

    crps_bt = term1 - term2  # [B, L]

    if reduction == "none":
        return crps_bt
    elif reduction == "mean":
        return crps_bt.mean()
    elif reduction == "sum":
        return crps_bt.sum()
    else:
        raise ValueError(f"Unknown reduction={reduction}")

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
