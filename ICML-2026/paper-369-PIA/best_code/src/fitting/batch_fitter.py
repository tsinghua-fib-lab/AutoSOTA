"""
Batch Fitting Tool - Process Multiple Scenario Groups (Optimized)
"""
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm  # <-- added

from .cognitive_fitter import CognitiveFitter, FitResult
from .strategies import get_strategy_for_group


def batch_fit_cognitive_model(
    df: pd.DataFrame,
    extended_mode: bool = False,  # Deprecated parameter, kept for compatibility
    baseline_first: bool = True,
    config: dict = None
) -> Dict[str, FitResult]:
    """
    Batch fit cognitive parameters for all scenario groups.

    Design principles:
    - extended_mode is deprecated and ignored.
    - Baseline parameters are used ONLY for initialization.
    - H, NLL, and BIC are all preserved via FitResult.
    - Fitting is performed per-group, never pooled.
    """

    fitter = CognitiveFitter(df, config=config)
    results: Dict[str, FitResult] = {}

    # -------------------------------------------------
    # Precompute group list and strategies (single pass)
    # -------------------------------------------------
    groups = df['group'].unique().tolist()
    group_strategies = {
        g: get_strategy_for_group(g) for g in groups
    }

    # -------------------------------------------------
    # 1. Fit baseline group first (optional)
    # -------------------------------------------------
    init_params: Optional[dict] = None

    if baseline_first:
        baseline_groups = [
            g for g, s in group_strategies.items() if s == 'BASELINE'
        ]

        if baseline_groups:
            baseline_group = baseline_groups[0]

            base_res = fitter.fit_scenario(
                group_name=baseline_group,
                strategy='BASELINE',
                baseline_params=None
            )

            if base_res is not None:
                results[baseline_group] = base_res
                init_params = base_res.params
            else:
                # baseline fitting failed → explicitly disable initialization
                init_params = None

    # -------------------------------------------------
    # 2. Fit remaining groups with progress bar
    # -------------------------------------------------
    remaining_groups = [g for g in groups if g not in results]

    for group in tqdm(remaining_groups, desc="Batch Fitting Groups"):
        strategy = group_strategies[group]

        res = fitter.fit_scenario(
            group_name=group,
            strategy=strategy,
            baseline_params=init_params
        )

        if res is not None:
            results[group] = res
        # else: fitting failure is intentionally skipped

    return results
