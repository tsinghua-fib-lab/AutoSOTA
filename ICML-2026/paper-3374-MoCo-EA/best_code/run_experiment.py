"""
Dataset-aware entry point for MoCo-EA experiments.

The experiment modules keep their original defaults for reproducibility. This
wrapper only centralizes CLI routing and common path/device options.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent

EXPERIMENT_MODULES = {
    ("connectivity", "cifar10"): ROOT / "experiments" / "connectivity.py",
    ("connectivity", "imagenet"): ROOT / "experiments" / "connectivity.py",
    ("evolutionary", "cifar10"): ROOT / "experiments" / "evolutionary.py",
    ("evolutionary", "imagenet"): ROOT / "experiments" / "evolutionary.py",
    ("transferability", "cifar10"): ROOT / "experiments" / "transferability.py",
    ("transferability", "imagenet"): ROOT / "experiments" / "transferability.py",
}

DEFAULT_DATA_ROOTS = {
    "cifar10": "./data",
    "imagenet": "./dataset/imagenet/val",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MoCo-EA experiments.")
    parser.add_argument(
        "--experiment",
        choices=sorted({name for name, _ in EXPERIMENT_MODULES}),
        required=True,
    )
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default="resnet18_cifar10_best.pth",
                        help="CIFAR-10 checkpoint path. Ignored for ImageNet experiments.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_module(path, experiment, dataset):
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(
        f"mocoea_{dataset}_{experiment.replace('-', '_')}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    key = (args.experiment, args.dataset)
    if key not in EXPERIMENT_MODULES:
        valid = ", ".join(f"{exp}/{dataset}" for exp, dataset in sorted(EXPERIMENT_MODULES))
        raise SystemExit(f"Unsupported combination: {args.experiment}/{args.dataset}. Valid: {valid}")

    module = load_module(EXPERIMENT_MODULES[key], args.experiment, args.dataset)
    module_args = SimpleNamespace(
        dataset=args.dataset,
        data_root=args.data_root or DEFAULT_DATA_ROOTS[args.dataset],
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )

    if hasattr(module, "cli_main"):
        module.cli_main(module_args)
    else:
        module.main(module_args)


if __name__ == "__main__":
    main()
