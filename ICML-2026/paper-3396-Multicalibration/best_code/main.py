# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for running experiments from config files."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from argparse_dataclass import ArgumentParser

from wmcal.configs import CONFIG_REGISTRY, ConfigEntry
from wmcal.experiments import ExperimentConfig, run_experiment


def _load_config_file(config_path: str) -> ConfigEntry:
    """Load a Python config file and return its experiments and workers."""
    import importlib.util

    path = Path(config_path)
    spec = importlib.util.spec_from_file_location("config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    experiments = getattr(mod, "experiments", [])
    workers = getattr(mod, "WORKERS", 1)
    return ConfigEntry(experiments=experiments, workers=workers)


def _resolve_config(cfg: str) -> ConfigEntry:
    """Resolve config from short key (e.g. sweep/top_k) or file path."""
    # Try registry lookup first
    if cfg in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[cfg]

    # Fall back to file path
    path = Path(cfg)
    if path.suffix == ".py":
        return _load_config_file(cfg)

    raise ValueError(
        f"Config '{cfg}' not found in registry and is not a .py file. "
        f"Available keys: {list(CONFIG_REGISTRY.keys())}"
    )


def _run_single(config: ExperimentConfig, redo: bool) -> tuple[str, bool]:
    """Top-level wrapper for running a single experiment (must be picklable)."""
    success = run_experiment(config, redo=redo)
    return config.id, success


@dataclass
class Args:
    cfg: str = field(metadata={"help": "Config key (e.g., sweep/top_k) or path to .py file"})
    redo: bool = field(default=False, metadata={"help": "Rerun even if already completed"})


def main(args: Args):
    entry = _resolve_config(args.cfg)
    experiments = entry.experiments
    workers = entry.workers

    if workers <= 1:
        for config in experiments:
            run_experiment(config, redo=args.redo)
        return

    # Parallel execution
    total = len(experiments)
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(_run_single, config, args.redo): config.id
            for config in experiments
        }
        for future in as_completed(future_to_id):
            eid = future_to_id[future]
            completed += 1
            try:
                _, success = future.result()
                if not success:
                    failed += 1
                    print(f"[{completed}/{total}] failed/skipped: {eid}")
                else:
                    print(f"[{completed}/{total}] done: {eid}")
            except Exception as exc:
                failed += 1
                print(f"[{completed}/{total}] CRASHED: {eid}: {exc}")

    print(f"\nFinished {total} experiments: {total - failed} succeeded, {failed} failed")


if __name__ == "__main__":
    parser = ArgumentParser(Args)
    args = parser.parse_args()
    main(args)
