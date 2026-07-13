import torch

from basicts.metrics.torch_metrics.coverage import (
    MaskedCoverage,
    MaskedPIWidth,
    MaskedPIMedianWidth,
)
from basicts.metrics.torch_metrics.winkler import MaskedWinklerScore
from basicts.utils.data_utils import find_close


def evaluate_quantile_intervals(
    q_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor | None,
    alphas,
    target_qs,
    loader_name: str = "eval",
):
    """Evaluate interval metrics using the torchmetrics in basicts.

    Args:
        q_hat: quantile predictions shaped (Q, B, H, N) or (Q, B, T, N).
        y_true: targets shaped (B, H, N).
        mask: boolean mask shaped (B, H, N) or None.
        alphas: list of miscoverage levels.
        target_qs: list of quantiles corresponding to q_hat's Q dimension.
        loader_name: prefix for metric keys.
    """
    if q_hat.dim() != 4:
        raise ValueError(f"q_hat must be 4-D (Q,B,H,N). Got {tuple(q_hat.shape)}")
    if y_true.dim() != 3:
        raise ValueError(f"y_true must be 3-D (B,H,N). Got {tuple(y_true.shape)}")

    if mask is None:
        mask = torch.ones_like(y_true, dtype=torch.bool)
    else:
        mask = mask.bool()

    results = {}
    for target_alpha in alphas:
        idx_low = find_close(target_alpha / 2, target_qs)
        idx_high = find_close(1 - target_alpha / 2, target_qs)
        interval = torch.stack([q_hat[idx_low], q_hat[idx_high]], dim=0)

        coverage = MaskedCoverage()
        pi_width = MaskedPIWidth()
        pi_width_median = MaskedPIMedianWidth()
        winkler = MaskedWinklerScore(alpha=target_alpha)

        coverage.update(interval, y_true, mask)
        pi_width.update(interval, y_true, mask)
        pi_width_median.update(interval, y_true, mask)
        winkler.update(interval, y_true, mask)

        results[f"{loader_name}_coverage_at_{(1 - target_alpha) * 100}"] = coverage.compute()
        results[f"{loader_name}_pi_width_at_{(1 - target_alpha) * 100}"] = pi_width.compute()
        results[f"{loader_name}_pi_width_median_at_{(1 - target_alpha) * 100}"] = pi_width_median.compute()
        results[f"{loader_name}_winkler_at_{(1 - target_alpha) * 100}"] = winkler.compute()

    return results
