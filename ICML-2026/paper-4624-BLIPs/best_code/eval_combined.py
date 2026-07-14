#!/usr/bin/env python3
"""
Combined evaluation script for BLIP N-body experiment.
Runs test evaluation and computes MSE, NLL, CRPS.
Usage: python3 eval_combined.py --model_id BayesGNN --seed 0 [--mc_steps 100]
"""
import os, sys, argparse
import torch
import numpy as np
from scipy.stats import norm

# Add repo paths
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/experiments')

from blip.model import MessagePassingNetwork
from blip.posterior import PosteriorModel
from blip.bayes import BayesianModelWrapper, KL
from nbody.dataset import NBodyGeometricDataset
from nbody.args import parse_args as nbody_parse_args
from utils import set_seed_precision, choose_activation, load_checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


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


def build_model(args, val_loader):
    """Build the BLIP model matching nbody_main.py"""
    if args.model_id in ["GNN", "DropGNN", "BayesGNN"]:
        model_id = "GNN"
        node_dim, edge_dim = 6, 1
        Module = MessagePassingNetwork
        regression_head = RegressionHead(args.hidden_dim, args.hidden_dim_regression_head, 3)
    else:
        model_id = "EGNN"
        node_dim, edge_dim = 1, 2
        Module = None  # Would need EGNN import
        regression_head = None

    kwargs = {}
    if args.model_id == "GNN" or args.model_id == "EGNN":
        args.mc_steps = 0
    if args.model_id in ["DropGNN", "DropEGNN"]:
        kwargs["dropout"] = args.dropout
    else:
        kwargs["dropout"] = None

    model = Module(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim,
        n_message_layers=args.n_message_layers, n_update_layers=args.n_update_layers,
        num_layers=args.num_layers, activation=choose_activation(args.activation),
        regression_head=regression_head, **kwargs,
    )

    if args.model_id in ["BayesGNN", "BayesEGNN"]:
        track_data = next(iter(val_loader))
        model = BayesianModelWrapper(model, kl=KL(args.prior_probability, args.beta))
        model.warm_up(
            num_nodes=track_data.num_nodes, num_edges=track_data.num_edges,
            batch=track_data, regex_pattern=[r"^blocks\."],
        )
        posterior = PosteriorModel(
            node_dim=node_dim, edge_dim=edge_dim,
            num_alphas_edge=model.num_alphas_edge, num_alphas_node=model.num_alphas_node,
            hidden_dim=args.hidden_dim_posterior, n_node_layers=args.n_node_layers,
            n_edge_layers=args.n_edge_layers, activation=choose_activation(args.activation_posterior),
        )
        model.add_posterior_net(posterior)

    return model.to(args.device)


def run_test(model, test_loader, device, mc_steps):
    """Run MC test evaluation"""
    stats = {"true_positions": [], "mean_positions": [], "var_positions": []}
    for data in test_loader:
        data = data.to(device)
        if mc_steps > 0:
            total_output = []
            for _ in range(mc_steps):
                out = model(batch=data)
                total_output.append(out.cpu().detach())
            total_output = torch.stack(total_output, dim=0)
            mean = total_output.mean(dim=0)
            var = total_output.var(dim=0)
        else:
            model.eval()
            mean = model(batch=data)
            var = None
        stats["true_positions"].append(data.y.cpu())
        stats["mean_positions"].append(mean.cpu())
        if var is not None:
            stats["var_positions"].append(var.cpu())

    stats["true_positions"] = torch.cat(stats["true_positions"], dim=0)
    stats["mean_positions"] = torch.cat(stats["mean_positions"], dim=0)
    stats["var_positions"] = torch.cat(stats["var_positions"], dim=0) if stats["var_positions"] else None
    return stats


def compute_metrics(stats):
    """Compute MSE, NLL, CRPS"""
    true_pos = stats['true_positions']
    mean_pos = stats['mean_positions']
    var_pos = stats.get('var_positions')

    true_flat = true_pos.reshape(-1).float()
    mean_flat = mean_pos.reshape(-1).float()
    squared_errors = (true_flat - mean_flat) ** 2
    mse = squared_errors.mean().item()

    results = {'MSE': mse, 'MSE_x10^{-1}': mse * 10}

    if var_pos is not None:
        var_flat = var_pos.reshape(-1).float()
        var_flat = torch.clamp(var_flat, min=1e-12)
        log_var = torch.log(2 * np.pi * var_flat)
        precision_weighted_error = squared_errors / var_flat
        nll = 0.5 * (log_var + precision_weighted_error).mean().item()
        results['NLL'] = nll

        std_flat = torch.sqrt(var_flat)
        z_np = ((true_flat - mean_flat) / std_flat).numpy()
        sigma_np = std_flat.numpy()
        phi = norm.pdf(z_np)
        Phi = norm.cdf(z_np)
        crps = (sigma_np * (z_np * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))).mean()
        results['CRPS'] = float(crps)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='BayesGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mc_steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--all_seeds', action='store_true', help='Evaluate all 4 seeds')
    args_cli = parser.parse_args()

    seeds = [0, 1, 2, 3] if args_cli.all_seeds else [args_cli.seed]
    all_results = []

    # Default args matching paper hyperparameters
    class DefaultArgs:
        model_id = 'BayesGNN'
        hidden_dim = 64
        hidden_dim_regression_head = 64
        hidden_dim_posterior = 64
        n_message_layers = 2
        n_update_layers = 2
        num_layers = 4
        activation = 'silu'
        activation_posterior = 'silu'
        n_node_layers = 2
        n_edge_layers = 2
        dropout = 0.5
        beta = 0.01
        prior_probability = 0.5
        dataset = 'nbody_small'
        max_training_samples = 3000
        batch_size = 100
        device = args_cli.device
        mc_steps = args_cli.mc_steps

    args = DefaultArgs()
    args.model_id = args_cli.model_id
    args.mc_steps = args_cli.mc_steps

    g = torch.Generator()

    for seed in seeds:
        args.seed = seed
        set_seed_precision(seed)
        g.manual_seed(seed)

        model_id = "GNN" if args.model_id in ["GNN", "DropGNN", "BayesGNN"] else "EGNN"

        # Load data (need val_loader for warmup)
        val_dataset = NBodyGeometricDataset(partition="val", dataset_name=args.dataset, model_id=model_id)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, generator=g)

        test_dataset = NBodyGeometricDataset(partition="test", dataset_name=args.dataset, model_id=model_id)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, generator=g)

        # Build model
        model = build_model(args, val_loader)

        # Load checkpoint
        ckpt_file = f"ckpt/nbody/{seed}/{args.model_id}_nbody_ckpt.pth"
        if not os.path.exists(ckpt_file):
            print(f"SKIP seed {seed}: checkpoint not found at {ckpt_file}")
            continue

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        load_checkpoint(ckpt_file, model, args.device, optimizer, scheduler)

        # Run test
        stats = run_test(model, test_loader, args.device, args.mc_steps)
        metrics = compute_metrics(stats)
        all_results.append(metrics)

        nll_str = f"{metrics.get('NLL', 'N/A'):.4f}" if metrics.get('NLL') is not None else "N/A"
        crps_str = f"{metrics.get('CRPS', 'N/A'):.6f}" if metrics.get('CRPS') is not None else "N/A"
        print(f"Seed {seed}: MSE={metrics['MSE']:.6f}, MSE×10⁻¹={metrics['MSE_x10^{-1}']:.4f}, NLL={nll_str}, CRPS={crps_str}")

    if len(all_results) > 0:
        mse_vals = [r['MSE_x10^{-1}'] for r in all_results]
        print(f"\nSummary: MSE×10⁻¹ = {np.mean(mse_vals):.4f} ± {np.std(mse_vals, ddof=1) if len(mse_vals) > 1 else 0:.4f}")
        print(f"Paper target: MSE×10⁻¹ = 0.092 ± 0.004")
    else:
        print("No results!")
        sys.exit(1)


if __name__ == "__main__":
    main()
