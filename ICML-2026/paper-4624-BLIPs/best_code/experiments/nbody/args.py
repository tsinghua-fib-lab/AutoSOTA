import argparse
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN on NBody")

    # Dataset and logging
    parser.add_argument(
        "--max_training_samples",
        type=int,
        default=3000,
        help="maximum amount of training samples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nbody_small",
        help="nbody_small, nbody_small_out_dist",
    )
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
    parser.add_argument(
        "--model_id",
        type=str,
        default="BayesEGNN",
        choices=["GNN", "BayesGNN", "DropGNN", "EGNN", "BayesEGNN", "DropEGNN"],
        help="Model name or identification for saving weights",
    )

    # Model configuration
    parser.add_argument(
        "--deep_ensemble",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use deep ensemble of the specific model during inference.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for MPNN layers",
    )
    parser.add_argument(
        "--n_message_layers",
        type=int,
        default=2,
        help="Number of message-passing layers in the MPNN",
    )
    parser.add_argument(
        "--n_update_layers",
        type=int,
        default=2,
        help="Number of node update layers in the MPNN",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of message passing layers in the MPNN",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "relu", "tanh"],
        help="Activation function index for MPNN",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Adding dropout to the model",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Weight of the KL optimization, usually 1/batch_size",
    )
    parser.add_argument(
        "--prior_probability",
        type=float,
        default=0.5,
        help="Prior probability for the KL optimization",
    )

    # Classification Head configuration
    parser.add_argument(
        "--hidden_dim_regression_head",
        type=int,
        default=64,
        help="Hidden dimension for the classification head",
    )

    # Posterior network configuration
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
        "--max_epochs",
        type=int,
        default=10000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
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
