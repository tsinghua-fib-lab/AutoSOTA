"""Shared CLI arguments and default hyperparameters for PSAHS."""
from __future__ import annotations

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    # Dataset / method
    parser.add_argument("-d", "--dataset", type=str, default="Blog")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="DANN_rw",
        help="Use a name containing 'rw' to enable progressive target adjustment.",
    )
    parser.add_argument("--src_name", type=str, default="Blog1")
    parser.add_argument("--tgt_name", type=str, default="Blog2")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds for repeated runs (paper: 5 runs).",
    )

    # Model
    parser.add_argument("--conv_type", type=str, default="dir-gcn-weighted")
    parser.add_argument("--jk", type=str, default="cat")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learn_alpha", action="store_true", default=False)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--conv_dim", type=int, default=128)
    parser.add_argument("--cls_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--dc_layers", type=int, default=2)
    parser.add_argument("--class_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--backbone", type=str, default="GCN")
    parser.add_argument("--resnet", action="store_true", default=False)
    parser.add_argument("--rw_lmda", type=float, default=0.5)

    # MLP auxiliary classifier (Step 1)
    parser.add_argument("--mlp_conv_dim", type=int, default=128)
    parser.add_argument("--mlp_cls_dim", type=int, default=64)
    parser.add_argument("--mlp_class_layers", type=int, default=2)

    # Optimization
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--opt_scheduler", type=str, default="step")
    parser.add_argument("--opt_decay_step", type=int, default=50)
    parser.add_argument("--opt_decay_rate", type=float, default=0.8)
    parser.add_argument("--valid_data", type=str, default="tgt", choices=["src", "tgt"])

    # Progressive structure adjustment
    parser.add_argument("--h_threshold", type=float, default=0.6)
    parser.add_argument("--rw_freq", type=int, default=20)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--alphatimes", type=float, default=1.5)
    parser.add_argument("--alphamin", type=float, default=1.0)

    # Noncircle synthetic benchmark
    parser.add_argument("--num_nodes", type=int, default=4000)
    parser.add_argument("--noncircle_src", type=str, default="")
    parser.add_argument("--noncircle_tgt", type=str, default="")

    parser.add_argument("--gpu_ratio", type=float, default=None)
    return parser


def apply_psahs_defaults(args):
    """Apply tuned defaults used in the paper implementation."""
    args.reweight = "rw" in (args.method or "")
    if args.reweight:
        args.rw_freq = 20
        args.start_epoch = 0
    args.lr = 0.003
    args.opt_decay_rate = 0.8
    args.opt_decay_step = 50
    args.hidden_dim = 128
    args.conv_dim = 128
    args.cls_dim = 64
    args.alphamin = 1.0
    args.alphatimes = 1.5
    args.epochs = 200
    args.h_threshold = 0.6
    args.K = 2
    if (args.dataset or "").strip().lower() == "noncircle":
        args.dataset = "Noncircle"

        args.opt_decay_step = 100
    return args


def apply_mlp_defaults(args):
    """Defaults for auxiliary MLP pretraining."""
    args.reweight = "rw" in (args.method or "")
    args.lr = 0.007
    args.opt_decay_rate = 0.8
    args.opt_decay_step = 50
    args.hidden_dim = 128
    args.mlp_conv_dim = 128
    args.mlp_cls_dim = 64
    args.alphamin = 0.2
    args.alphatimes = 1.0
    args.epochs = 300
    args.K = 3
    if (args.dataset or "").strip().lower() == "noncircle":
        args.dataset = "Noncircle"
    return args
