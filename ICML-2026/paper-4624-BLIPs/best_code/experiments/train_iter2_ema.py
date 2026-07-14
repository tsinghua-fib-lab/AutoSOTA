#!/usr/bin/env python3
"""
Iter-2: Training with EMA (Exponential Moving Average) of model weights.
EMA smooths weight trajectories, often improving generalization for Bayesian NNs.
Uses same KL annealing + cosine LR as iter-1.
"""
import sys, os
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/experiments')

import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from blip.model import MessagePassingNetwork
from blip.posterior import PosteriorModel
from blip.bayes import BayesianModelWrapper, KL

from utils import (
    directory_handler, set_seed_precision, choose_activation,
    save_checkpoint, train, validation, compute_kl_scale,
)
from nbody.dataset import NBodyGeometricDataset

warnings.filterwarnings("ignore")


class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch=None):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear_out(x)
        return x


class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Replace model weights with EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    set_seed_precision(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    directory_handler(["./logs", "./ckpt", "./data"])

    model_id = "GNN"
    dataset_name = "nbody_small"

    train_dataset = NBodyGeometricDataset(
        partition="train", dataset_name=dataset_name,
        max_samples=3000, model_id=model_id,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, generator=g,
    )
    val_dataset = NBodyGeometricDataset(
        partition="val", dataset_name=dataset_name, model_id=model_id
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, generator=g,
    )

    node_dim, edge_dim = 6, 1
    regression_head = RegressionHead(64, 64, 3)

    model = MessagePassingNetwork(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64,
        n_message_layers=2, n_update_layers=2,
        num_layers=4, activation=choose_activation("silu"),
        regression_head=regression_head,
    )

    track_data = next(iter(val_loader))
    model = BayesianModelWrapper(model, kl=KL(0.5, 0.01))
    model.warm_up(
        num_nodes=track_data.num_nodes, num_edges=track_data.num_edges,
        batch=track_data, regex_pattern=[r"^blocks\."],
    )
    posterior = PosteriorModel(
        node_dim=node_dim, edge_dim=edge_dim,
        num_alphas_edge=model.num_alphas_edge, num_alphas_node=model.num_alphas_node,
        hidden_dim=64, n_node_layers=2, n_edge_layers=2,
        activation=choose_activation("silu"),
    )
    model.add_posterior_net(posterior)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_epochs = 500
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_epochs - warmup_epochs,
        eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(
        log_dir=f"logs/nbody/{args.seed}/BayesGNN_nbody_small_logs_iter2_ema/"
    )

    ckpt_file = f"ckpt/nbody/{args.seed}/BayesGNN_nbody_ckpt.pth"
    kl_warmup_epochs = 2000
    use_bayesian = hasattr(model, "posterior")

    ema = EMAModel(model, decay=args.ema_decay)
    best_val_loss = 1e20

    print(f"Training starts with EMA (decay={args.ema_decay})")
    for epoch in range(args.max_epochs):
        kl_scale = 1.0
        if use_bayesian and epoch < kl_warmup_epochs:
            kl_scale = compute_kl_scale(epoch, kl_warmup_epochs, args.max_epochs)

        train_loss = train(
            model, train_loader, optimizer, scheduler, criterion,
            args.device, use_bayesian, "regression", False, kl_scale=kl_scale,
        )

        # Update EMA after each optimizer step
        ema.update()

        val_loss = validation(model, val_loader, criterion, args.device, "regression")

        writer.add_scalar("Train/loss", train_loss[0], epoch)
        writer.add_scalar("Train/kl", train_loss[1], epoch)
        writer.add_scalar("Val/loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save EMA weights
            ema.apply_shadow()
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_file)
            ema.restore()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}: train_loss={train_loss[0]:.6f}, kl={train_loss[1]:.6f}, val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

    print(f"Training complete. Best val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
