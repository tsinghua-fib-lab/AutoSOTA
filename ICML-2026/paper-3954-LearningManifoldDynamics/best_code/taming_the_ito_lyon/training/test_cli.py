import argparse
import os

from taming_the_ito_lyon.config import load_toml_config
from taming_the_ito_lyon.training.experiment import run_test, run_test_from_run_dir


def _parse_seeds(raw: list[str] | None) -> list[int] | None:
    if raw is None:
        return None
    seeds: list[int] = []
    for token in raw:
        for chunk in token.split(","):
            chunk_stripped = chunk.strip()
            if chunk_stripped:
                seeds.append(int(chunk_stripped))
    return seeds if len(seeds) > 0 else None


def _resolve_run_dir_paths(run_dir: str) -> tuple[str, str]:
    config_path = os.path.join(run_dir, "config.toml")
    checkpoint_path = os.path.join(run_dir, "best.eqx")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return config_path, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the test epoch for a saved run directory"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the saved run directory containing config.toml and best.eqx",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=str,
        default=None,
        help="Seeds for replicate testing (e.g. --seeds 1 2 3 or --seeds 1,2,3).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Start seed for sequential replicates (used with --seed-count)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Number of sequential seeds to run (used with --seed-start)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional label for model fan plots (defaults to 'Preds').",
    )
    args = parser.parse_args()

    if args.model_name:
        os.environ["ROUGH_VOL_FAN_PREDS_LABEL"] = str(args.model_name)

    seeds = _parse_seeds(args.seeds)
    seed_start = args.seed_start
    seed_count = args.seed_count
    if seeds is not None and (seed_start is not None or seed_count is not None):
        raise ValueError("Use either --seeds or --seed-start/--seed-count, not both.")

    if seeds is None and (seed_start is None and seed_count is None):
        run_test_from_run_dir(args.run_dir)
        return

    if seed_start is not None or seed_count is not None:
        if seed_start is None or seed_count is None:
            raise ValueError("--seed-start and --seed-count must be used together.")
        if seed_count <= 0:
            raise ValueError("--seed-count must be > 0.")
        seeds = [int(seed_start) + i for i in range(int(seed_count))]

    assert seeds is not None
    config_path, checkpoint_path = _resolve_run_dir_paths(args.run_dir)
    metrics_path = os.path.join(args.run_dir, "test_metrics.json")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    base_config = load_toml_config(config_path)
    for seed in seeds:
        config = base_config.model_copy(deep=True)
        config.experiment_config.seed = int(seed)
        run_test(
            config=config,
            checkpoint_path=checkpoint_path,
            run_dir=args.run_dir,
            metrics_name="test_metrics.json",
            metrics_seed=int(seed),
        )


if __name__ == "__main__":
    main()
