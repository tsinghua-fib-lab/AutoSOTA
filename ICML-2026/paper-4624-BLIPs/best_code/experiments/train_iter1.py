#!/usr/bin/env python3
"""
Training script for BLIP BayesGNN on N-body with improvements:
- 10000 training epochs (paper specification)
- Cosine annealing LR schedule with warmup
- KL annealing (warmup beta from 0 to full over 2000 epochs)
- Gradient clipping
"""
import sys, os
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/experiments')

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from blip.model import MessagePassingNetwork
from blip.posterior import PosteriorModel
from blip.bayes import BayesianModelWrapper, KL

from utils import (
    directory_handler,
    set_seed_precision,
    run,
    choose_activation,
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--prior_probability', type=float, default=0.5)
    parser.add_argument('--kl_warmup_epochs', type=int, default=2000)
    parser.add_argument('--lr_warmup_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_id', type=str, default='BayesGNN')
    args = parser.parse_args()

    set_seed_precision(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    directory_handler(["./logs", "./ckpt", "./data"])

    model_id = "GNN"
    dataset_name = "nbody_small"
    max_training_samples = 3000

    train_dataset = NBodyGeometricDataset(
        partition="train", dataset_name=dataset_name,
        max_samples=max_training_samples, model_id=model_id,
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

    # Build model
    node_dim, edge_dim = 6, 1
    regression_head = RegressionHead(args.hidden_dim, args.hidden_dim, 3)

    model = MessagePassingNetwork(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim,
        n_message_layers=2, n_update_layers=2,
        num_layers=args.num_layers, activation=choose_activation("silu"),
        regression_head=regression_head,
    )

    # Bayesian wrapper
    track_data = next(iter(val_loader))
    model = BayesianModelWrapper(
        model, kl=KL(args.prior_probability, args.beta),
    )
    model.warm_up(
        num_nodes=track_data.num_nodes, num_edges=track_data.num_edges,
        batch=track_data, regex_pattern=[r"^blocks\."],
    )
    posterior = PosteriorModel(
        node_dim=node_dim, edge_dim=edge_dim,
        num_alphas_edge=model.num_alphas_edge, num_alphas_node=model.num_alphas_node,
        hidden_dim=args.hidden_dim, n_node_layers=2, n_edge_layers=2,
        activation=choose_activation("silu"),
    )
    model.add_posterior_net(posterior)

    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    # Optimizer with cosine annealing + warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)

    # Cosine annealing: after warmup, cosine decay from lr to lr*0.01
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.lr_warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_epochs - args.lr_warmup_epochs,
        eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.lr_warmup_epochs]
    )

    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(
        log_dir=f"logs/nbody/{args.seed}/{args.model_id}_nbody_small_logs_iter1/"
    )

    ckpt_file = f"ckpt/nbody/{args.seed}/{args.model_id}_nbody_ckpt.pth"

    run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=args.device,
        writer=writer,
        ckpt_file=ckpt_file,
        run_type="train",
        max_epochs=args.max_epochs,
        kl_warmup_epochs=args.kl_warmup_epochs,
        clip_grad=True,
    )

    print(f"Training complete. Checkpoint saved to {ckpt_file}")


if __name__ == "__main__":
    main()
