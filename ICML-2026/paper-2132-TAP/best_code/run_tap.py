"""Command-line entry point for TAP."""

import argparse
from pathlib import Path

import pandas as pd
from tabcamel.data.dataset import TabularDataset

from config import get_default_config
from generators import TabDiffGenerator, train_tabdiff
from tap import TAPTrainer
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAP and write synthetic tabular data.")
    parser.add_argument("--dataset", type=str, default="custom", help="TabCamel dataset name or run label.")
    parser.add_argument("--data_path", type=str, default=None, help="Optional CSV file.")
    parser.add_argument("--target_col", type=str, default=None, help="Target column name.")
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
    )
    parser.add_argument("--n_real", type=str, default="50", help="Number or fraction of real rows to use.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--final_samples", type=int, default=500)
    parser.add_argument("--gen_steps", type=int, default=8000)
    parser.add_argument("--skip_gen", action="store_true", help="Load an existing TabDiff model from the run folder.")
    parser.add_argument("--output_dir", type=str, default="runs")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_path and args.target_col is None:
        raise ValueError("--target_col is required when --data_path is used.")

    set_seed(args.seed)
    data_df, target_col = load_data(args)
    n_real = resolve_n_real(args.n_real, len(data_df))
    train_df = sample_real_rows(data_df, target_col, n_real, args.task_type, args.seed)

    run_dir = Path(args.output_dir) / f"{safe_name(args.dataset)}_n{n_real}"
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_gen:
        generator = TabDiffGenerator(
            model_dir=str(model_dir),
            data_dir=str(model_dir / "data"),
            device=args.device,
        )
    else:
        generator = train_tabdiff(
            train_data=train_df,
            target_col=target_col,
            save_path=str(model_dir),
            steps=args.gen_steps,
            device=args.device,
            seed=args.seed,
            task_type=args.task_type,
        )

    config = get_default_config()
    config.data.task_type = args.task_type
    config.data.target_column = target_col
    config.generator.device = args.device
    config.train.seed = args.seed
    config.train.checkpoint_dir = str(run_dir / "checkpoints")
    config.train.log_dir = str(run_dir / "logs")

    trainer = TAPTrainer(config=config, task_type=args.task_type)
    synthetic = trainer.train_policy(
        train_data=train_df,
        generator=generator,
        target_col=target_col,
        num_steps=args.num_steps,
        final_samples=args.final_samples,
    )

    out_path = run_dir / "synthetic_data.csv"
    synthetic.to_csv(out_path, index=False)
    print(f"Wrote {len(synthetic)} rows to {out_path}")


def load_data(args):
    if args.data_path:
        return pd.read_csv(args.data_path), args.target_col

    dataset = TabularDataset(dataset_name=args.dataset, task_type=args.task_type)
    return dataset.data_df.copy(), dataset.target_col


def resolve_n_real(value: str, total: int) -> int:
    n = float(value)
    if n <= 1.0:
        n = int(total * n)
    return max(1, min(int(n), total))


def sample_real_rows(df: pd.DataFrame, target_col: str, n_rows: int, task_type: str, seed: int):
    if n_rows >= len(df):
        return df.reset_index(drop=True)

    if task_type == "classification":
        try:
            return (
                df.groupby(target_col, group_keys=False)
                .sample(frac=n_rows / len(df), random_state=seed)
                .sample(n=n_rows, replace=False, random_state=seed)
                .reset_index(drop=True)
            )
        except ValueError:
            pass

    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


if __name__ == "__main__":
    main()
