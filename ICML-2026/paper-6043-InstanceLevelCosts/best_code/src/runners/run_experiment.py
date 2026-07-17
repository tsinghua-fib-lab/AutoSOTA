"""
Unified CLI runner for cost-sensitive learning experiments.

Usage:
    # P1: Basic experiment
    python -m src.runners.run_experiment --dataset jigsaw --model tfidf \\
        --method classification --weighting none --seed 42

    # P2: Sample size scaling
    python -m src.runners.run_experiment --dataset jigsaw --model tfidf \\
        --method classification --weighting none --seed 42 --sample_size 10000

    # P3: Sampling strategy
    python -m src.runners.run_experiment --dataset jigsaw --model tfidf \\
        --method classification --weighting none --seed 42 --strategy P_up
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union

import joblib
import numpy as np
import pandas as pd

from data import load_dataset, list_datasets, DatasetSplit
from models import get_model, list_models, get_model_feature_types
from tasks.classify import run_classification, ClassifyResult
from tasks.delta_regress import run_regression, RegressResult
from core.seed import set_seed
from core import logging as wandb_log


# Available sampling strategies for P3
SAMPLING_STRATEGIES = ['U', 'P_up', 'Tdown50', 'Tdown30', 'Tdown70']


def apply_sampling_strategy(
    train_split: DatasetSplit,
    strategy: str,
    seed: int,
) -> DatasetSplit:
    """
    Apply a sampling strategy to the training split.

    Args:
        train_split: Original training split
        strategy: One of 'U', 'P_up', 'Tdown50', 'Tdown30'
        seed: Random seed for reproducibility

    Returns:
        New DatasetSplit with sampled data
    """
    rng = np.random.default_rng(seed)

    X = train_split.X
    y = train_split.y
    delta = train_split.delta
    abs_delta = train_split.abs_delta

    n = len(y)
    indices = np.arange(n)

    if strategy == 'U':
        # Uniform - no change
        sampled_idx = indices

    elif strategy == 'P_up':
        # Probabilistic upsampling proportional to |Δ| within each class
        sampled_idx = []
        for cls in [0, 1]:
            cls_mask = (y == cls)
            cls_indices = indices[cls_mask]
            cls_weights = abs_delta[cls_mask].astype(np.float64)

            # Normalize weights to probabilities
            if cls_weights.sum() > 0:
                probs = cls_weights / cls_weights.sum()
            else:
                probs = None  # uniform

            # Sample with replacement, same size as original class
            sampled = rng.choice(cls_indices, size=len(cls_indices), replace=True, p=probs)
            sampled_idx.extend(sampled)

        sampled_idx = np.array(sampled_idx)
        rng.shuffle(sampled_idx)

    elif strategy.startswith('Tdown'):
        # Top-k by |Δ| within each class
        keep_frac = int(strategy.replace('Tdown', '')) / 100.0

        sampled_idx = []
        for cls in [0, 1]:
            cls_mask = (y == cls)
            cls_indices = indices[cls_mask]
            cls_abs_delta = abs_delta[cls_mask]

            # Find threshold for top keep_frac
            threshold = np.quantile(cls_abs_delta, 1 - keep_frac)

            # Keep samples with |Δ| >= threshold
            keep_mask = cls_abs_delta >= threshold
            sampled_idx.extend(cls_indices[keep_mask])

        sampled_idx = np.array(sampled_idx)
        rng.shuffle(sampled_idx)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build new split
    if isinstance(X, list):
        new_X = [X[i] for i in sampled_idx]
    elif isinstance(X, np.ndarray):
        new_X = X[sampled_idx]
    elif isinstance(X, pd.DataFrame):
        new_X = X.iloc[sampled_idx].reset_index(drop=True)
    else:
        raise TypeError(f"Unsupported X type: {type(X)}")

    # Update indices if present (indices into original full dataset, not local indices)
    new_indices = None
    if train_split.indices is not None:
        new_indices = train_split.indices[sampled_idx]

    return DatasetSplit(
        X=new_X,
        y=y[sampled_idx],
        delta=delta[sampled_idx],
        abs_delta=abs_delta[sampled_idx],
        ids=train_split.ids[sampled_idx] if train_split.ids is not None else None,
        indices=new_indices,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cost-sensitive learning experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list_datasets(),
        help='Dataset to use',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list_models(),
        help='Model to use',
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['classification', 'regression'],
        help='Task type',
    )
    parser.add_argument(
        '--weighting',
        type=str,
        required=True,
        choices=['none', 'absdelta', 'alpha_balanced'],
        help='Weighting strategy',
    )

    # Optional arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Subsample size (None = use all data)',
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Validation set fraction',
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Test set fraction',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging',
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='cost-sensitive-learning',
        help='Wandb project name',
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Override wandb run name',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Override default data path for dataset',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        choices=SAMPLING_STRATEGIES,
        help='P3 sampling strategy (None = no sampling, use full train set)',
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations."""
    # Mutually exclusive: sample_size (P2) vs strategy (P3)
    if args.sample_size and args.strategy:
        raise ValueError("Cannot use both --sample_size (P2) and --strategy (P3)")

    # Check model-dataset compatibility
    model_types = get_model_feature_types(args.model)
    dataset_type_map = {
        'jigsaw': 'text',
        'turkey': 'image',
        'nhanes': 'tabular',
        'inaturalist': 'image',
        'synthetic': 'tabular',
    }
    dataset_type = dataset_type_map.get(args.dataset, 'unknown')

    if dataset_type not in model_types:
        raise ValueError(
            f"Model '{args.model}' (expects {model_types}) is not compatible "
            f"with dataset '{args.dataset}' (provides {dataset_type})"
        )


def get_output_path(args: argparse.Namespace) -> Path:
    """Generate output CSV path."""
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename based on experiment type
    # P1: {model}_{method}_{weighting}_s{seed}.csv
    # P2: {model}_{method}_{weighting}_s{seed}_n{sample_size}.csv
    # P3: {model}_{method}_{weighting}_s{seed}_{strategy}.csv
    if args.sample_size:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}_n{args.sample_size}.csv"
    elif args.strategy:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}_{args.strategy}.csv"
    else:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}.csv"
    return output_dir / filename


def get_model_path(args: argparse.Namespace) -> Path:
    """Generate model save path (.joblib)."""
    model_dir = Path('cache/models') / args.dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    # Same naming convention as results, but with .joblib extension
    if args.sample_size:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}_n{args.sample_size}.joblib"
    elif args.strategy:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}_{args.strategy}.joblib"
    else:
        filename = f"{args.model}_{args.method}_{args.weighting}_s{args.seed}.joblib"
    return model_dir / filename


def result_to_dataframe(
    result,
    args: argparse.Namespace,
    original_train_size: Optional[int] = None,
    sampled_train_size: Optional[int] = None,
) -> pd.DataFrame:
    """Convert result to DataFrame with metadata."""
    # Get base result dict
    row = result.to_dict()

    # Add experiment metadata
    row['dataset'] = args.dataset
    row['sample_size'] = args.sample_size
    row['strategy'] = args.strategy
    row['val_size'] = args.val_size
    row['test_size'] = args.test_size
    row['timestamp'] = datetime.now().isoformat()

    # P3: track original vs sampled train size
    if original_train_size is not None:
        row['original_train_size'] = original_train_size
    if sampled_train_size is not None:
        row['sampled_train_size'] = sampled_train_size

    return pd.DataFrame([row])


def run_experiment(args: argparse.Namespace) -> None:
    """Run the experiment."""
    # Set global seed
    set_seed(args.seed)

    # Print experiment info
    print(f"=" * 60)
    print(f"Experiment: {args.dataset} / {args.model} / {args.method}")
    print(f"Weighting: {args.weighting}, Seed: {args.seed}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    if args.strategy:
        print(f"Sampling strategy: {args.strategy}")
    print(f"=" * 60)

    # Initialize wandb if requested
    if args.wandb:
        # Build run name based on experiment type (or use override)
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        elif args.sample_size:
            run_name = f"{args.dataset}-{args.model}-{args.method}-{args.weighting}-s{args.seed}-n{args.sample_size}"
        elif args.strategy:
            run_name = f"p3-{args.dataset}-{args.model}-{args.strategy}-s{args.seed}"
        else:
            run_name = f"{args.dataset}-{args.model}-{args.method}-{args.weighting}-s{args.seed}"

        tags = [args.dataset, args.model, args.method, args.weighting]
        if args.strategy:
            tags.extend(['P3', args.strategy])

        wandb_log.init_run(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=tags,
        )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    load_kwargs = {}
    if args.data_path:
        load_kwargs['path'] = args.data_path

    dataset = load_dataset(
        name=args.dataset,
        sample_size=args.sample_size if not args.strategy else None,  # P3 uses full data, then samples
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        **load_kwargs,
    )
    original_train_size = len(dataset.train)
    print(f"  Train: {original_train_size}, Val: {len(dataset.val) if dataset.val else 0}, Test: {len(dataset.test)}")

    # P3: Apply sampling strategy to training set
    train_split = dataset.train
    if args.strategy:
        print(f"\nApplying sampling strategy: {args.strategy}...")
        train_split = apply_sampling_strategy(dataset.train, args.strategy, args.seed)
        print(f"  Sampled train size: {len(train_split)} ({len(train_split)/original_train_size*100:.1f}% of original)")

    # Create model
    print(f"\nCreating model: {args.model} ({args.method})...")
    task = args.method  # 'classification' or 'regression'

    # Pass feature names for tabular models
    model_kwargs = {}
    if args.model == 'histgbm' and 'feature_names' in dataset.metadata:
        # NHANES has numeric features: age, gender, race, bmi
        # Gender and race are categorical (coded as floats)
        feature_names = dataset.metadata['feature_names']
        num_features = ['RIDAGEYR', 'BMXBMI']  # Age, BMI
        cat_features = ['RIAGENDR', 'RIDRETH3']  # Gender, Race
        model_kwargs['num_features'] = [f for f in num_features if f in feature_names]
        model_kwargs['cat_features'] = [f for f in cat_features if f in feature_names]

    # Enable disk caching for text embedding models (roberta)
    if args.model == 'roberta':
        model_kwargs['cache_name'] = args.dataset

    model = get_model(args.model, task=task, **model_kwargs)
    print(f"  {model}")

    # Run experiment
    # Note: train_split = dataset.train when no strategy is set (backwards compatible)
    print(f"\nRunning {args.method} with {args.weighting} weighting...")

    # Extract indices for pre-computed embedding lookup (if available)
    train_indices = train_split.indices
    val_indices = dataset.val.indices if dataset.val else None
    test_indices = dataset.test.indices

    if args.method == 'classification':
        result = run_classification(
            model=model,
            X_train=train_split.X,
            y_train=train_split.y,
            X_test=dataset.test.X,
            y_test=dataset.test.y,
            delta_train=train_split.delta,
            delta_test=dataset.test.delta,
            X_val=dataset.val.X if dataset.val else None,
            y_val=dataset.val.y if dataset.val else None,
            delta_val=dataset.val.delta if dataset.val else None,
            weighting=args.weighting,
            seed=args.seed,
            model_name=args.model,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
    else:  # regression
        result = run_regression(
            model=model,
            X_train=train_split.X,
            y_train=train_split.y,
            X_test=dataset.test.X,
            y_test=dataset.test.y,
            delta_train=train_split.delta,
            delta_test=dataset.test.delta,
            X_val=dataset.val.X if dataset.val else None,
            y_val=dataset.val.y if dataset.val else None,
            delta_val=dataset.val.delta if dataset.val else None,
            weighting=args.weighting,
            seed=args.seed,
            model_name=args.model,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    # Print results
    print(f"\nResults:")
    print(f"  Train metrics:")
    for k, v in result.train_metrics.items():
        print(f"    {k}: {v:.4f}")

    if result.val_metrics:
        print(f"  Val metrics:")
        for k, v in result.val_metrics.items():
            print(f"    {k}: {v:.4f}")

    print(f"  Test metrics:")
    for k, v in result.test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save trained model
    model_path = get_model_path(args)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Log to wandb
    if args.wandb:
        # Log test metrics as summary
        wandb_log.log_summary({f"test_{k}": v for k, v in result.test_metrics.items()})

        # Log all metrics
        all_metrics = {}
        for k, v in result.train_metrics.items():
            all_metrics[f'train_{k}'] = v
        if result.val_metrics:
            for k, v in result.val_metrics.items():
                all_metrics[f'val_{k}'] = v
        for k, v in result.test_metrics.items():
            all_metrics[f'test_{k}'] = v
        wandb_log.log_metrics(all_metrics)

    # Save results
    output_path = get_output_path(args)
    df = result_to_dataframe(
        result, args,
        original_train_size=original_train_size,
        sampled_train_size=len(train_split),
    )
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Finish wandb run
    if args.wandb:
        wandb_log.finish_run()

    print("\nDone!")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        validate_args(args)
        run_experiment(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.wandb:
            wandb_log.finish_run(exit_code=1)
        raise


if __name__ == '__main__':
    main()
