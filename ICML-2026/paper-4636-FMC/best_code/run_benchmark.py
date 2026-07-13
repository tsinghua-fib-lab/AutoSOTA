#!/usr/bin/env python
"""Benchmark experiment CLI for NPE/FMPE comparison.

This script provides a unified interface for running benchmark experiments:
- Phase 1: Foundation training (simulation models + baselines)
- Phase 2: Variant training (fm_post_transform and other methods)
- Phase 3: Evaluation (compute metrics on test set)
- Phase 4: Aggregation (generate comparison tables)

Usage:
    # Train foundation (Phase 1)
    python run_benchmark.py command=train_foundation +experiment=comparison_benchmark

    # Train a variant (Phase 2)
    python run_benchmark.py command=train_variant \\
        +experiment=comparison_benchmark \\
        variant_name=fm_pt_fmpe_lr1e3

    # Train variant with HPO
    python run_benchmark.py command=train_variant \\
        +experiment=comparison_benchmark \\
        variant_name=fm_pt_hpo \\
        hpo=true hpo_trials=50

    # Evaluate (Phase 3)
    python run_benchmark.py command=evaluate +experiment=comparison_benchmark

    # Aggregate results (Phase 4)
    python run_benchmark.py command=aggregate +experiment=comparison_benchmark

    # Full pipeline
    python run_benchmark.py command=run_all +experiment=comparison_benchmark
"""

import sys
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


def get_tasks_from_config(cfg: DictConfig) -> List[str]:
    """Extract task list from config."""
    # Prefer explicit tasks list (from experiment config)
    if "tasks" in cfg and cfg.tasks:
        return list(cfg.tasks)
    # Fall back to single task from task config
    elif "task" in cfg and cfg.task:
        task_name = cfg.task.get("name") if hasattr(cfg.task, "get") else None
        if task_name:
            return [task_name]
    return []


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for benchmark CLI."""
    # Get command from config
    command = cfg.get("command", None)

    if command is None:
        print("Usage: python run_benchmark.py command=<command> [options]")
        print("\nCommands:")
        print("  train_foundation  Train simulation models and baselines (Phase 1)")
        print("  train_variant     Train a method variant (Phase 2)")
        print("  evaluate          Evaluate methods on test set (Phase 3)")
        print("  evaluate_all      Evaluate all methods and variants")
        print("  aggregate         Aggregate results into tables (Phase 4)")
        print("  run_all           Run full pipeline")
        print("\nExample:")
        print("  python run_benchmark.py command=train_foundation +experiment=comparison_benchmark")
        return

    # Dispatch to appropriate handler
    if command == "train_foundation":
        train_foundation_cmd(cfg)
    elif command == "train_variant":
        train_variant_cmd(cfg)
    elif command == "evaluate":
        evaluate_cmd(cfg)
    elif command == "evaluate_all":
        evaluate_all_cmd(cfg)
    elif command == "aggregate":
        aggregate_cmd(cfg)
    elif command == "run_all":
        run_all_cmd(cfg)
    else:
        print(f"Unknown command: {command}")
        print("Use --help for available commands")


def train_foundation_cmd(cfg: DictConfig) -> None:
    """Train foundation models (Phase 1)."""
    from benchmark import train_foundation

    tasks = get_tasks_from_config(cfg)
    if not tasks:
        print("No tasks specified. Use 'tasks=[task1,task2]' or set in experiment config.")
        return

    print(f"\n{'='*60}")
    print("PHASE 1: FOUNDATION TRAINING")
    print(f"{'='*60}")
    print(f"Tasks: {tasks}")
    print(f"Experiment: {cfg.get('experiment', {}).get('name', 'unknown')}")

    # Get options
    skip_existing = cfg.get("skip_existing", True)
    parallel = cfg.get("parallel", {}).get("enabled", False)

    results = train_foundation(cfg, tasks, skip_existing, parallel)

    print(f"\n{'='*60}")
    print("Foundation training complete!")
    print(f"{'='*60}")
    for task, task_results in results.items():
        print(f"\n{task}:")
        for model, status in task_results.get("simulation_models", {}).items():
            print(f"  {model}: {status.get('status', 'unknown')}")


def train_variant_cmd(cfg: DictConfig) -> None:
    """Train a variant (Phase 2)."""
    from benchmark import train_variant, run_variant_hpo

    variant_name = cfg.get("variant_name")
    if not variant_name:
        print("Error: variant_name is required")
        print("Usage: python run_benchmark.py train_variant variant_name=my_variant ...")
        return

    tasks = get_tasks_from_config(cfg)
    if not tasks:
        print("No tasks specified. Use 'tasks=[task1,task2]' or set in experiment config.")
        return

    # Check for HPO mode
    hpo_enabled = cfg.get("hpo", {}).get("enabled", False) or cfg.get("variant", {}).get("hpo", {}).get("enabled", False)

    print(f"\n{'='*60}")
    print(f"PHASE 2: VARIANT TRAINING - {variant_name}")
    print(f"{'='*60}")
    print(f"Tasks: {tasks}")
    print(f"HPO: {'enabled' if hpo_enabled else 'disabled'}")

    if hpo_enabled:
        # HPO mode
        n_trials = cfg.get("hpo_trials", cfg.get("variant", {}).get("hpo", {}).get("n_trials", 50))
        hpo_seed = cfg.get("hpo_seed", 0)
        hpo_ncal = cfg.get("hpo_ncal", 200)
        storage = cfg.get("hpo_storage", None)

        results = run_variant_hpo(
            cfg,
            variant_name,
            tasks,
            n_trials=n_trials,
            seed=hpo_seed,
            ncal=hpo_ncal,
            storage=storage,
        )

        print(f"\n{'='*60}")
        print("HPO complete!")
        print(f"{'='*60}")
        for task, task_results in results.items():
            print(f"\n{task}:")
            print(f"  Best params: {task_results.get('best_params', {})}")
            print(f"  Best value: {task_results.get('best_value', 'N/A'):.4f}")
    else:
        # Regular training
        skip_existing = cfg.get("skip_existing", True)
        parallel = cfg.get("parallel", {}).get("enabled", False)

        # Single task/seed/ncal mode (for cluster jobs)
        if cfg.get("task") and cfg.get("seed") is not None and cfg.get("ncal"):
            from benchmark import train_variant_single
            result = train_variant_single(
                cfg,
                variant_name,
                cfg.task if isinstance(cfg.task, str) else cfg.task.name,
                cfg.seed,
                cfg.ncal,
                skip_existing,
                parallel,
            )
            print(f"Result: {result}")
        else:
            # Full training across all tasks/seeds/ncals
            results = train_variant(cfg, variant_name, tasks, skip_existing=skip_existing, parallel=parallel)

            print(f"\n{'='*60}")
            print("Variant training complete!")
            print(f"{'='*60}")


def evaluate_cmd(cfg: DictConfig) -> None:
    """Evaluate a specific variant or baseline (Phase 3)."""
    from benchmark import evaluate_variant, evaluate_baselines

    tasks = get_tasks_from_config(cfg)
    if not tasks:
        print("No tasks specified.")
        return

    variant_name = cfg.get("variant_name") or cfg.get("variant")
    # Use evaluation.skip_existing for metrics (distinct from top-level skip_existing for training)
    skip_existing = cfg.get("evaluation", {}).get("skip_existing", True)

    print(f"\n{'='*60}")
    print("PHASE 3: EVALUATION")
    print(f"{'='*60}")
    print(f"Tasks: {tasks}")

    if variant_name:
        print(f"Variant: {variant_name}")
        results = evaluate_variant(cfg, variant_name, tasks, skip_existing)
    else:
        print("Evaluating baselines...")
        results = evaluate_baselines(cfg, tasks, skip_existing)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


def evaluate_all_cmd(cfg: DictConfig) -> None:
    """Evaluate all methods and variants (Phase 3)."""
    from benchmark import evaluate_all

    tasks = get_tasks_from_config(cfg)
    if not tasks:
        print("No tasks specified.")
        return

    # Use evaluation.skip_existing for metrics (distinct from top-level skip_existing for training)
    skip_existing = cfg.get("evaluation", {}).get("skip_existing", True)

    print(f"\n{'='*60}")
    print("PHASE 3: FULL EVALUATION")
    print(f"{'='*60}")
    print(f"Tasks: {tasks}")

    results = evaluate_all(cfg, tasks, skip_existing)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")

    # Print summary
    for category, cat_results in results.items():
        if cat_results:
            print(f"\n{category}:")
            for name in cat_results.keys():
                print(f"  - {name}")


def aggregate_cmd(cfg: DictConfig) -> None:
    """Aggregate results into tables (Phase 4)."""
    from benchmark import aggregate_results, generate_tables, generate_report

    tasks = get_tasks_from_config(cfg)
    if not tasks:
        print("No tasks specified.")
        return

    print(f"\n{'='*60}")
    print("PHASE 4: AGGREGATION")
    print(f"{'='*60}")
    print(f"Tasks: {tasks}")

    # Aggregate results
    results = aggregate_results(cfg, tasks)

    # Generate tables
    generate_tables(cfg, tasks)

    # Generate report
    report = generate_report(cfg, tasks)

    print(f"\n{'='*60}")
    print("Aggregation complete!")
    print(f"{'='*60}")
    print("\nSummary:")
    print(report[:500] + "..." if len(report) > 500 else report)


def run_all_cmd(cfg: DictConfig) -> None:
    """Run full pipeline."""
    print(f"\n{'#'*60}")
    print("RUNNING FULL BENCHMARK PIPELINE")
    print(f"{'#'*60}")

    # Phase 1
    print("\n\n" + "="*60)
    print("STARTING PHASE 1: FOUNDATION TRAINING")
    print("="*60)
    train_foundation_cmd(cfg)

    # Phase 3: Evaluate (baselines and simulation models)
    print("\n\n" + "="*60)
    print("STARTING PHASE 3: EVALUATION (baselines)")
    print("="*60)
    evaluate_all_cmd(cfg)

    # Phase 4
    print("\n\n" + "="*60)
    print("STARTING PHASE 4: AGGREGATION")
    print("="*60)
    aggregate_cmd(cfg)

    print(f"\n{'#'*60}")
    print("BENCHMARK PIPELINE COMPLETE!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
