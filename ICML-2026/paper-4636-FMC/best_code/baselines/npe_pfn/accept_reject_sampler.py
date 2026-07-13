"""Accept-reject sampling utilities for NPE-PFN."""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm


@torch.no_grad()
def accept_reject_sample(
    proposal: Callable,
    accept_reject_fn: Callable,
    num_samples: int,
    show_progress_bars: bool = False,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: Optional[Dict] = None,
    max_iter_rejection: int | None = None,
) -> Tuple[Tensor, float]:
    """Returns samples from a proposal according to an acceptance criterion.

    Args:
        proposal: Function that generates proposal samples
        accept_reject_fn: Function that evaluates which samples are accepted
        num_samples: Desired number of samples
        show_progress_bars: Whether to show a progressbar during sampling
        max_sampling_batch_size: Maximum batch size for sampling
        proposal_sampling_kwargs: Arguments passed to proposal function

    Returns:
        Tuple of (accepted_samples, acceptance_rate)
    """
    if proposal_sampling_kwargs is None:
        proposal_sampling_kwargs = {}

    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    accepted = []
    accepted_log_probs = []

    num_remaining = num_samples
    num_sampled_total = 0
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    i = 0

    while num_remaining > 0:
        i += 1
        # Sample and reject
        candidates, log_probs = proposal(
            sampling_batch_size, **proposal_sampling_kwargs
        )
        are_accepted = accept_reject_fn(candidates)

        # Store accepted samples
        accepted.append(candidates[are_accepted])
        if log_probs is not None:
            accepted_log_probs.append(log_probs[are_accepted])

        # Update counters
        num_accepted = are_accepted.sum().item()
        num_sampled_total += sampling_batch_size
        num_remaining -= num_accepted
        pbar.update(num_accepted)

        # Adjust batch size based on acceptance rate
        acceptance_rate = sum(len(x) for x in accepted) / num_sampled_total
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(acceptance_rate, 1e-12)), 100),
        )

        if max_iter_rejection is not None and i > max_iter_rejection:
            accepted.append(candidates)
            accepted_log_probs.append(log_probs)
            break

    pbar.close()

    # Concatenate and trim to exact number of samples
    samples = torch.cat(accepted, dim=0)[:num_samples]
    log_probs = (
        torch.cat(accepted_log_probs, dim=0)[:num_samples]
        if accepted_log_probs  # if list is not empty
        else None
    )

    final_acceptance_rate = len(samples) / num_sampled_total

    return samples, log_probs, final_acceptance_rate
