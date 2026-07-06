# -*- coding: utf-8 -*-
"""Training metrics: spike rate counting and energy estimation."""

import torch


def count_spikes_epoch(
    model,
    loader,
    device,
    timesteps: int,
    neuromorphic: bool = False,
    dtype=torch.float32,
):
    """
    Compute average spike rate over a full dataset epoch.

    Reads model.last_spike_rate (set by forward pass) for each batch and
    averages across all batches.

    Args:
        model: SNN model with a `last_spike_rate` attribute.
        loader: DataLoader to iterate over.
        device: Compute device.
        timesteps: Number of time steps (used to estimate total spike count).
        neuromorphic: Unused; kept for API compatibility.
        dtype: Tensor dtype.

    Returns:
        Tuple of (mean_spike_rate, estimated_total_spikes).

    Example:
        >>> rate, total = count_spikes_epoch(model, test_loader, device, timesteps=10)
    """
    model.train(False)
    spike_rates = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device=device, dtype=dtype)
            _ = model(x)
            if model.last_spike_rate is not None:
                rate_val = model.last_spike_rate
                spike_rates.append(rate_val.item() if hasattr(rate_val, "item") else float(rate_val))

    spike_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0.0
    total_spikes = spike_rate * len(loader.dataset) * timesteps
    return spike_rate, total_spikes


def compute_energy_proxy(spike_rate: float, num_neurons: int = 0, timesteps: int = 1) -> float:
    """
    Compute a normalized energy proxy from the spike rate.

    Energy ~ #spikes. Normalized to a baseline rate of 0.5
    (random spiking), so 1.0x = same energy as a random network.

    Args:
        spike_rate: Average spike rate per neuron per timestep.
        num_neurons: Unused; kept for API compatibility.
        timesteps: Unused; kept for API compatibility.

    Returns:
        Energy proxy as a multiple of the 0.5-rate baseline.
    """
    baseline_rate = 0.5
    return spike_rate / baseline_rate if baseline_rate > 0 else spike_rate
