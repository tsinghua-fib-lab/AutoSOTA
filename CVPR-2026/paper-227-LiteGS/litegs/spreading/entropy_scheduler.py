"""Entropy weight scheduler for managing entropy loss weight during training."""


def _entropy_weight_strategy_default(lambda_entropy: float, period_info: tuple) -> float:
    """
    Default entropy weight scheduling strategy.

    Args:
        lambda_entropy: Base entropy loss weight coefficient
        period_info: Training period information

    Returns:
        Entropy weight for the current epoch
    """
    period_number, epoch_in_period, period_length, progress_ratio, is_final_period, epoch, is_final_reset = period_info

    entropy_weight = 0.0
    if not is_final_period and 2 <= period_number <= 3:
        if 1 <= epoch_in_period <= 7 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy / 2.0
    if not is_final_period and 4 <= period_number:
        if 1 <= epoch_in_period <= 7 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy
    if is_final_period:
        if 1 <= epoch_in_period <= 15 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy * 2.0

    return entropy_weight


def _entropy_weight_strategy_with_reset(lambda_entropy: float, period_info: tuple) -> float:
    """
    Entropy weight scheduling strategy for use with scale reset.

    Args:
        lambda_entropy: Base entropy loss weight coefficient
        period_info: Training period information

    Returns:
        Entropy weight for the current epoch
    """
    period_number, epoch_in_period, period_length, progress_ratio, is_final_period, epoch, is_final_reset = period_info

    entropy_weight = 0.0
    if not is_final_period and 2 <= period_number <= 3:
        if period_number % 2 == 1 and 1 <= epoch_in_period <= 7 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy / 2.0
    if not is_final_period and 4 <= period_number:
        if period_number % 2 == 1 and 1 <= epoch_in_period <= 7 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy
    if is_final_period:
        if 3 <= epoch_in_period <= 15 and epoch_in_period % 2 == 1:
            entropy_weight = lambda_entropy * 2.0

    return entropy_weight


def get_entropy_weight(lambda_entropy: float, period_info: tuple, scale_reset_factor: float) -> float:
    """
    Calculate entropy weight for the current training epoch.

    Args:
        lambda_entropy: Base entropy loss weight coefficient
        period_info: Training period information
        scale_reset_factor: Scale reset factor value

    Returns:
        Entropy weight for the current epoch
    """
    if lambda_entropy <= 0.0:
        return 0.0

    # Choose strategy based on scale reset factor
    if scale_reset_factor > 0.0:
        return _entropy_weight_strategy_with_reset(lambda_entropy, period_info)
    else:
        return _entropy_weight_strategy_default(lambda_entropy, period_info)
