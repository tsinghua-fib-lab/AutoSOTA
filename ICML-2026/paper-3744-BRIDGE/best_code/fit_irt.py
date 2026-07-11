import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer

from two_param_logistic import TwoParamLogistic

SCRIPT_DIR = Path(__file__).parent.resolve()
PARAMS_DIR = SCRIPT_DIR / "params"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a 2PL IRT model and export parameters."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL file (py-irt format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Make reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

    # Process input and output paths
    input_path = Path(args.input_path).expanduser()
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PARAMS_DIR / f"{input_path.stem}.csv"
    abilities_out_csv = PARAMS_DIR / f"{input_path.stem}_abilities.csv"

    if args.verbose:
        print(f"Input: {input_path}")
        print(f"Output item params: {out_csv}")
        print(f"Output abilities: {abilities_out_csv}")

    # Load input data
    data = Dataset.from_jsonlines(str(input_path))
    config = IrtConfig(
        model_type=TwoParamLogistic,
        priors="hierarchical",
    )

    if args.verbose:
        print(f"Training IRT model for {args.epochs} epochs on {args.device}...")

    # Fit IRT model
    trainer = IrtModelTrainer(
        config=config,
        data_path=None,
        dataset=data,
    )
    trainer.train(epochs=args.epochs, device=args.device)

    # Extract IRT parameters
    discriminations = [np.exp(i) for i in trainer.best_params["disc"]]  # Since parameter in log space
    difficulties = list(trainer.best_params["diff"])
    item_ids = trainer.best_params["item_ids"].values()
    irt_model = pd.DataFrame({"a": discriminations, "b": difficulties}, index=item_ids)
    irt_model.to_csv(out_csv)

    # Map learned abilities back to each subject
    subject_ids = list(trainer.best_params["subject_ids"].values())
    abilities = list(trainer.best_params["ability"])
    abilities_df = pd.DataFrame({"subject_id": subject_ids, "ability": abilities})
    abilities_df.to_csv(abilities_out_csv, index=False)

    if args.verbose:
        print(f"Saved item parameters ({len(item_ids)} items) to {out_csv}")
        print(f"Saved subject abilities ({len(subject_ids)} subjects) to {abilities_out_csv}")


if __name__ == "__main__":
    main()
