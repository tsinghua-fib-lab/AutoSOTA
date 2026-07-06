# -*- coding: utf-8 -*-
"""
Main training loop for UltraLIF experiments.

Functions:
    train_model: Train an SNN model with optional spike tracking and sparsity penalty.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import count_spikes_epoch


def train_model(
    model,
    train_loader,
    test_loader,
    epochs: int,
    lr: float,
    device,
    verbose: bool = True,
    use_tpu: bool = False,
    save_path=None,
    track_spikes: bool = False,
    neuromorphic: bool = False,
    timesteps: int = 30,
    sparsity_lambda: float = 0.0,
    dtype=torch.float32,
):
    """
    Train an SNN model with Adam + cosine annealing and optional sparsity penalty.

    Loss = CrossEntropy + sparsity_lambda * spike_rate

    Tracks learnable parameters (eps, k/DSpike temperature, tau, spike_scale)
    after every epoch and saves the best checkpoint if save_path is provided.

    Args:
        model: SNN model (must expose `last_spike_rate` attribute).
        train_loader: Training DataLoader.
        test_loader: Evaluation DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        device: Compute device.
        verbose: Show tqdm progress bars.
        use_tpu: Sync after each step with torch_xla.
        save_path: Path to save the best checkpoint (.pt). None = no save.
        track_spikes: Compute spike rate every 10 epochs.
        neuromorphic: Whether data is from a neuromorphic dataset.
        timesteps: Time steps (for spike rate estimation).
        sparsity_lambda: Penalty strength on spike rate (0 = disabled).
        dtype: Tensor dtype.

    Returns:
        Tuple of (best_accuracy, epoch_history, final_info).

        epoch_history: List of dicts with keys epoch, acc, loss, and
            optionally eps, k, tau, spike_rate.
        final_info: Dict with full per-epoch histories for eps, k, tau,
            spike_scale, and spike_rate.
    """
    if use_tpu:
        import torch_xla.core.xla_model as xm

    model = model.to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    best_state = None
    history = []
    eps_history, k_history, tau_history, spike_scale_history, spike_rate_history = [], [], [], [], []

    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch", disable=not verbose)

    for ep in epoch_pbar:
        model.train()
        train_loss = 0.0
        n_batches = 0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False, disable=not verbose)
        for x, y in batch_pbar:
            x, y = x.to(device=device, dtype=dtype), y.to(device)
            opt.zero_grad()
            out = model(x)
            ce_loss = crit(out, y)
            loss = (
                ce_loss + sparsity_lambda * model.last_spike_rate
                if sparsity_lambda > 0 and model.last_spike_rate is not None
                else ce_loss
            )
            loss.backward()
            opt.step()
            if use_tpu:
                xm.mark_step()
            train_loss += loss.item()
            n_batches += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = train_loss / n_batches

        # Collect learnable params from first neuron(s)
        if hasattr(model, "neuron"):
            neurons = [model.neuron]
        elif hasattr(model, "neuron1"):
            neurons = [model.neuron1, model.neuron2]
        else:
            neurons = []

        eps_vals, k_vals, tau_vals, sc_vals = [], [], [], []
        for n in neurons:
            if hasattr(n, "eps"):
                eps_vals.append(n.eps.item())
            if hasattr(n, "k"):
                k_vals.append(n.k.item())
            if hasattr(n, "tau") and isinstance(n.tau, torch.Tensor):
                tau_vals.append(n.tau.item())
            if hasattr(n, "spike_scale"):
                sc_vals.append(n.spike_scale.item())

        def _avg(lst):
            return sum(lst) / len(lst) if lst else None

        if eps_vals:
            eps_history.append(_avg(eps_vals))
        if k_vals:
            k_history.append(_avg(k_vals))
        if tau_vals:
            tau_history.append(_avg(tau_vals))
        if sc_vals:
            spike_scale_history.append(_avg(sc_vals))

        # Evaluate
        model.train(False)
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device=device, dtype=dtype), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total

        spike_rate = None
        if track_spikes and (ep + 1) % 10 == 0:
            spike_rate, _ = count_spikes_epoch(model, test_loader, device, timesteps, neuromorphic, dtype)
            spike_rate_history.append({"epoch": ep + 1, "rate": spike_rate})

        epoch_record = {"epoch": ep + 1, "acc": acc, "loss": avg_loss}
        if eps_history:
            epoch_record["eps"] = eps_history[-1]
        if k_history:
            epoch_record["k"] = k_history[-1]
        if tau_history:
            epoch_record["tau"] = tau_history[-1]
        if spike_rate is not None:
            epoch_record["spike_rate"] = spike_rate
        history.append(epoch_record)

        if acc > best:
            best = acc
            if save_path:
                ckpt_spike_rate = spike_rate
                if track_spikes and ckpt_spike_rate is None:
                    ckpt_spike_rate, _ = count_spikes_epoch(
                        model, test_loader, device, timesteps, neuromorphic, dtype
                    )
                best_state = {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "best_acc": acc,
                    "epoch": ep + 1,
                    "spike_rate": ckpt_spike_rate,
                    "energy_proxy": ckpt_spike_rate * timesteps if ckpt_spike_rate else None,
                    "eps": eps_history[-1] if eps_history else None,
                    "tau": tau_history[-1] if tau_history else None,
                }

        postfix = {"acc": f"{acc:.2%}", "best": f"{best:.2%}", "loss": f"{avg_loss:.4f}"}
        if eps_history:
            postfix["eps"] = f"{eps_history[-1]:.2f}"
        if k_history:
            postfix["k"] = f"{k_history[-1]:.2f}"
        if spike_scale_history:
            postfix["sc"] = f"{spike_scale_history[-1]:.2f}"
        epoch_pbar.set_postfix(postfix)

    if save_path and best_state:
        torch.save(best_state, save_path)
        print(f"  Saved: {save_path}")

    final_info = {
        "eps_history": eps_history,
        "k_history": k_history,
        "tau_history": tau_history,
        "spike_scale_history": spike_scale_history,
        "spike_rate_history": spike_rate_history,
    }

    return best, history, final_info
