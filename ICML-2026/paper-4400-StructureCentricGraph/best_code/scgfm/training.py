from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import trange

from scgfm.data.graph_features import precompute_graph_statistics
from scgfm.models.geometric_bases import GeometricBasesModel
from scgfm.utils import ensure_dir, write_json


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool, accumulation_steps: int):
    model.train()
    total_loss = 0.0
    log_agg = {"loss_gw": 0.0, "loss_rec": 0.0, "loss_div": 0.0}
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        if use_amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, logs = model(batch)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss, logs = model(batch)
            loss = loss / accumulation_steps
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        for key in log_agg:
            log_agg[key] += logs.get(key, 0.0)

    n_batches = max(1, len(loader))
    for key in log_agg:
        log_agg[key] /= n_batches
    return total_loss / n_batches, log_agg


def pretrain_bases(graphs, config: dict, device: torch.device, output_dir: str | Path):
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    feature_dim = int(model_cfg.get("feature_dim", 50))
    graphs = precompute_graph_statistics(graphs, feature_dim=feature_dim)

    loader = DataLoader(
        graphs,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = GeometricBasesModel(
        K=int(model_cfg.get("K", 16)),
        M=int(model_cfg.get("M", 32)),
        feature_dim=feature_dim,
        tau=float(model_cfg.get("tau", 0.1)),
        lambda_gw=float(model_cfg.get("lambda_gw", 1.0)),
        lambda_recon=float(model_cfg.get("lambda_recon", 1.0)),
        lambda_div=float(model_cfg.get("lambda_div", 0.02)),
        div_margin=float(model_cfg.get("div_margin", 8.0)),
        num_projections=int(model_cfg.get("num_projections", 50)),
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 0.01)))
    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    epochs = int(train_cfg.get("epochs", 60))
    log_interval = int(train_cfg.get("log_interval", 20))

    tau_start = float(model_cfg.get("tau_start", 1.0))
    tau_end = float(model_cfg.get("tau", 0.3))
    tau_anneal_epochs = int(model_cfg.get("tau_anneal_epochs", 40))

    history = []
    for epoch in trange(1, epochs + 1, desc="Pretrain"):
        # Tau annealing: cosine schedule from tau_start to tau_end
        if epoch <= tau_anneal_epochs:
            progress = epoch / tau_anneal_epochs
            current_tau = tau_end + (tau_start - tau_end) * (1 + np.cos(np.pi * progress)) / 2
        else:
            current_tau = tau_end
        model.set_tau(current_tau)

        avg_loss, logs = train_one_epoch(model, loader, optimizer, scaler, device, use_amp, accumulation_steps)
        row = {"epoch": epoch, "loss": avg_loss, "tau": current_tau, **logs}
        history.append(row)
        if epoch == 1 or epoch % log_interval == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} "
                f"gw={logs['loss_gw']:.4f} rec={logs['loss_rec']:.4f} div={logs['loss_div']:.4f} tau={current_tau:.3f}"
            )

    out_dir = ensure_dir(output_dir)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_cfg,
        "train_config": train_cfg,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    torch.save(model.get_normalized_bases().detach().cpu(), out_dir / "learned_bases.pt")
    write_json(out_dir / "metrics.json", {"history": history, "epochs": epochs})
    return model

