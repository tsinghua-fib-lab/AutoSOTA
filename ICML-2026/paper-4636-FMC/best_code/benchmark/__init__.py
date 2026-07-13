"""Benchmark experiment system for NPE/FMPE comparison.

This module provides a modular experiment system that:
- Trains simulation models (NPE, FMPE) and baselines once (Phase 1)
- Allows iterative optimization of fm_post_transform variants (Phase 2)
- Supports incremental addition of new methods/seeds
- Enables parallel execution with proper file locking
- Provides flexible evaluation and result aggregation (Phase 3 & 4)

Directory structure:
    results/{experiment}/
        shared/                              # Phase 1: Train once
            data/{task}/
                data_split.json
                calibration_pool.pt
                test_data.pt
                simulations.pt

            simulation/{task}/
                npe/model.pkl, config.json, training_log.json
                fmpe/model.pkl, config.json, training_log.json

            baselines/{task}/seed_{seed}/
                dpe/ncal_{ncal}/model.pkl, metrics.json
                mf_npe/ncal_{ncal}/model.pkl, metrics.json

        variants/{variant_name}/             # Phase 2: Iterate methods
            config.yaml
            calibration/{task}/seed_{seed}/
                {method}/ncal_{ncal}/
                    model.pkl
                    training_log.json
            evaluation/{task}/seed_{seed}/
                {method}/ncal_{ncal}/
                    samples.pt
                    metrics.json

        aggregated/                          # Phase 3 & 4: Unified results
            {task}/
                all_methods_metrics.json
                comparison_table.csv

Usage:
    # Phase 1: Train foundation
    python run_benchmark.py train_foundation +experiment=comparison_benchmark

    # Phase 2: Train variant
    python run_benchmark.py train_variant +experiment=comparison_benchmark \\
        variant_name=fm_pt_fmpe_lr1e3

    # Phase 3: Evaluate
    python run_benchmark.py evaluate +experiment=comparison_benchmark

    # Phase 4: Aggregate
    python run_benchmark.py aggregate +experiment=comparison_benchmark
"""

from benchmark.foundation import (
    FoundationConfig,
    FoundationTrainer,
    train_foundation,
    generate_and_save_data,
    train_simulation_model,
    train_baselines,
)
from benchmark.variants import (
    VariantConfig,
    VariantTrainer,
    train_variant,
    train_variant_single,
    run_variant_hpo,
)
from benchmark.evaluation import (
    EvaluationConfig,
    BenchmarkEvaluator,
    evaluate_variant,
    evaluate_baselines,
    evaluate_all,
)
from benchmark.aggregation import (
    AggregationConfig,
    BenchmarkAggregator,
    aggregate_results,
    generate_tables,
    generate_report,
)

__all__ = [
    # Foundation (Phase 1)
    "FoundationConfig",
    "FoundationTrainer",
    "train_foundation",
    "generate_and_save_data",
    "train_simulation_model",
    "train_baselines",
    # Variants (Phase 2)
    "VariantConfig",
    "VariantTrainer",
    "train_variant",
    "train_variant_single",
    "run_variant_hpo",
    # Evaluation (Phase 3)
    "EvaluationConfig",
    "BenchmarkEvaluator",
    "evaluate_variant",
    "evaluate_baselines",
    "evaluate_all",
    # Aggregation (Phase 4)
    "AggregationConfig",
    "BenchmarkAggregator",
    "aggregate_results",
    "generate_tables",
    "generate_report",
]
