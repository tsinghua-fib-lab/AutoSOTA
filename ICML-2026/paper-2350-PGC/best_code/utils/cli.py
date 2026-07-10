import argparse
import os
from typing import List, Tuple

import torch

# Choices for ``--dino_variant``.  Centralised so train and test cannot drift.
DINO_CHOICES = [
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "dinov3-vitb16-pretrain-lvd1689m",
]

# Choices for ``--optim``.
OPTIM_CHOICES = ["adam", "sgd"]


def build_train_parser() -> argparse.ArgumentParser:
    """Construct the training CLI parser.

    Every option used anywhere in the training pipeline is registered here
    with a sensible default; downstream modules access them directly via
    attribute lookup.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PGC (Peak-Guided Calibration) training CLI",
    )

    # ----- Identity / I/O -------------------------------------------------
    parser.add_argument(
        "--name",
        type=str,
        default="pgc_train",
        help="Experiment name (also used as log filename stem).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory under which logs and checkpoints are written.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="3",
        help=(
            "CUDA devices, e.g. '0' or '0,1'; '-1' for CPU. "
            "RTX 4090 #3 is the test box."
        ),
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of DataLoader worker processes for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    # ----- DINO backbone & LoRA -------------------------------------------
    parser.add_argument(
        "--dino_variant",
        type=str,
        default="dinov2-large",
        choices=DINO_CHOICES,
        help="DINO backbone variant.",
    )
    parser.add_argument(
        "--dino_pretrained_root",
        type=str,
        default=None,
        help=(
            "Root directory containing local DINO checkpoints. "
            "When set, the loader expects subfolders named after the variant. "
            "Required for training; entry-point validates explicitly."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank for adapter layers.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="LoRA scaling factor (alpha / rank determines effective scale).",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability inside LoRA adapter layers.",
    )

    # ----- Training data ---------------------------------------------------
    parser.add_argument(
        "--real_image_dir",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more root directories of real training images. "
            "Example: --real_image_dir /path/A /path/B. All directories are "
            "scanned recursively and the resulting samples are merged + "
            "shuffled before training."
        ),
    )
    parser.add_argument(
        "--fake_image_dir",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more root directories of fake / generated training "
            "images. Example: --fake_image_dir /path/A /path/B. All "
            "directories are scanned recursively and the resulting samples "
            "are merged + shuffled before training."
        ),
    )
    parser.add_argument(
        "--cropSize",
        type=int,
        default=224,
        help="Pad-then-crop input resolution (square).",
    )

    # ----- Mid-training evaluation ----------------------------------------
    parser.add_argument(
        "--test_root",
        type=str,
        default=None,
        help=(
            "Optional root directory of the evaluation dataset "
            "(UniversalFakeDetect layout: ``<root>/<subset>/{0_real,1_fake}``). "
            "If supplied the trainer evaluates on it periodically."
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size used during mid-training evaluation.",
    )
    parser.add_argument(
        "--eval_num_threads",
        type=int,
        default=8,
        help="DataLoader workers for mid-training evaluation.",
    )

    # ----- TTA (Test-Time Augmentation) -----------------------------------
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable test-time augmentation during evaluation.",
    )
    parser.add_argument(
        "--tta_mode",
        type=str,
        default="flip",
        choices=["flip", "flip_multiscale"],
        help="TTA mode: 'flip' for horizontal flip only, "
             "'flip_multiscale' for flip + multi-scale crops.",
    )

    # ----- Optimization ----------------------------------------------------
    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=OPTIM_CHOICES,
        help="Optimizer family.",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label-smoothing epsilon for binary cross-entropy.",
    )

    # ----- PGCM (Peak-Guided Calibration Module) hyper-parameters ---------
    parser.add_argument(
        "--tau_rgb",
        type=float,
        default=0.5,
        help="Temperature for the RGB-stream peak aggregation in PGCM.",
    )
    parser.add_argument(
        "--tau_res",
        type=float,
        default=0.5,
        help="Temperature for the residual-stream peak aggregation in PGCM.",
    )

    # ----- Stability / reproducibility ------------------------------------
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping (0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy and PyTorch.",
    )

    # ----- Bookkeeping -----------------------------------------------------
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=200,
        help=(
            "Run mid-training evaluation every N optimization steps. "
            "A checkpoint named "
            "``step{N}_acc{int}_ap{int}.pth`` is also saved right after "
            "each such evaluation (in addition to the best-AP ``best.pth``)."
        ),
    )

    # Internal flag used by ``finalize_opt`` to know which parser produced
    # the namespace.  Not exposed as a CLI option (no ``add_argument``).
    parser.set_defaults(isTrain=True)

    return parser


def build_test_parser() -> argparse.ArgumentParser:
    """Construct the inference / evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PGC (Peak-Guided Calibration) test CLI",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        required=True,
        help="Root directory of the evaluation dataset.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory under which the test log file is written.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pgc_test",
        help="Experiment name used to derive the test log filename.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes for evaluation.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="3",
        help=(
            "CUDA devices, e.g. '0' or '0,1'; '-1' for CPU. "
            "RTX 4090 #3 is the test box."
        ),
    )

    parser.add_argument(
        "--cropSize",
        type=int,
        default=224,
        help="Pad-then-crop input resolution (must match training).",
    )
    parser.add_argument(
        "--dino_variant",
        type=str,
        default="dinov2-large",
        choices=DINO_CHOICES,
        help="DINO backbone variant (must match training).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (must match training).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="LoRA alpha (must match training).",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (must match training).",
    )
    parser.add_argument(
        "--dino_pretrained_root",
        type=str,
        default=None,
        help="Root directory containing local DINO checkpoints.",
    )
    parser.add_argument(
        "--tau_rgb",
        type=float,
        default=0.5,
        help="PGCM RGB-stream temperature (must match training).",
    )
    parser.add_argument(
        "--tau_res",
        type=float,
        default=0.5,
        help="PGCM residual-stream temperature (must match training).",
    )

    # ----- TTA (Test-Time Augmentation) -----------------------------------
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable test-time augmentation during evaluation.",
    )
    parser.add_argument(
        "--tta_mode",
        type=str,
        default="flip",
        choices=["flip", "flip_multiscale"],
        help="TTA mode: 'flip' for horizontal flip only, "
             "'flip_multiscale' for flip + multi-scale crops.",
    )

    parser.set_defaults(isTrain=False)

    return parser


def parse_devices(devices_str: str) -> Tuple[List[int], torch.device]:

    str_ids = [s for s in devices_str.split(",") if s.strip() != ""]
    parsed_ids = [int(x) for x in str_ids]
    valid_ids = [i for i in parsed_ids if i >= 0]

    if torch.cuda.is_available() and len(valid_ids) > 0:
        return valid_ids, torch.device(f"cuda:{valid_ids[0]}")
    return [], torch.device("cpu")


def cli_message_for_opt(
    parser: argparse.ArgumentParser, opt: argparse.Namespace
) -> str:

    message = "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        if k.startswith("_"):
            continue
        default = parser.get_default(k)
        comment = f"\t[default: {default}]" if v != default else ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    return message


def finalize_opt(opt: argparse.Namespace) -> str:

    parser = build_train_parser() if opt.isTrain else build_test_parser()

    message = cli_message_for_opt(parser, opt)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    with open(os.path.join(expr_dir, "opt.txt"), "wt") as opt_file:
        opt_file.write(message + "\n")

    devices_list, device = parse_devices(opt.devices)
    opt.devices = devices_list
    opt.device = device
    if torch.cuda.is_available() and len(devices_list) > 0:
        torch.cuda.set_device(devices_list[0])

    return message
