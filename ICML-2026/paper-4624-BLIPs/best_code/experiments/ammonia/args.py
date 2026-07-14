import argparse
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN on NBody")

    # Dataset and logging
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default=None,
        help="Checkpoint to load for inference",
    )
    parser.add_argument(
        "--mc_steps",
        type=int,
        default=100,
        help="Number of monte carlo steps to compute statistics",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run type for the sesssion",
    )

    # Posterior network configuration
    parser.add_argument(
        "--beta",
        type=float,
        default=10,
        help="Weight of the KL optimization, usually 1/batch_size",
    )
    parser.add_argument(
        "--prior_probability",
        type=float,
        default=0.5,
        help="Prior probability for the KL optimization",
    )
    parser.add_argument(
        "--hidden_dim_posterior",
        type=int,
        default=64,
        help="Hidden dimension for the posterior network",
    )
    parser.add_argument(
        "--activation_posterior",
        type=str,
        default="silu",
        choices=["silu", "relu", "tanh"],
        help="Activation function index for posterior",
    )
    parser.add_argument(
        "--n_node_layers",
        type=int,
        default=2,
        help="Number of layers in node posterior network",
    )

    parser.add_argument(
        "--n_edge_layers",
        type=int,
        default=2,
        help="Number of layers in edge posterior network",
    )

    # Training configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Experiment seed",
    )
    return parser.parse_args()
