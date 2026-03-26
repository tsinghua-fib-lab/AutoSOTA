#!/usr/bin/env python
"""Sample script demonstrating the new API on the eICU dataset.

This script loads Apache scores and mortality data from the eICU dataset,
and evaluates prediction performance on different demographic subgroups
(ethnicities or gender) using the helper class :data:`data.eicu.EICU`.
It then visualizes the resulting net-benefit curves with the new :func:`plot_net_benefit_curves`.
"""
from __future__ import annotations
import os
from typing import List, Optional

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special

from etl.eicu import EICU
from proto.subgroup import SubgroupResults  # Updated import
from core.curves import plot_net_benefit_curves  # Updated import
from sklearn.metrics import roc_auc_score
from stats.ece import expected_calibration_error  # Updated import

# Constants
MIN_SUBGROUP_SIZE = 50
DEFAULT_COST_RATIO = 0.5
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Desktop")


def get_subgroup_stats(
        model: EICU,
        subgroup_field: str,
        subgroups: Optional[List[str]] = None,
        min_size: int = MIN_SUBGROUP_SIZE,
        compute_auc: bool = False,
        compute_ece: bool = False
) -> List[SubgroupResults]:
    """Evaluate the model on different demographic subgroups.

    Parameters
    ----------
    model : EICU
        The EICU model instance containing patient data.
    subgroup_field : str
        Field to split subgroups by (e.g., 'ethnicity', 'gender').
    subgroups : Optional[List[str]], default=None
        Optional list of specific subgroup values to evaluate.
        If None, all unique values in subgroup_field are used.
    min_size : int, default=MIN_SUBGROUP_SIZE
        Minimum number of samples required for a subgroup to be included.
    compute_auc : bool, default=False
        Whether to compute AUC-ROC for each subgroup.
    compute_ece : bool, default=False
        Whether to compute Expected Calibration Error for each subgroup.

    Returns
    -------
    List[SubgroupResults]
        List of SubgroupResults objects for each valid subgroup.
    """
    # Get all unique values for the subgroup field if not specified
    if subgroups is None:
        subgroups = model.data[subgroup_field].dropna().unique()

    # Initialize results list
    results = []

    # Calculate metrics for each subgroup
    for subgroup in subgroups:
        # Get data for this subgroup
        subgroup_mask = model.data[subgroup_field] == subgroup
        subgroup_data = model.data[subgroup_mask].copy()

        # Skip subgroups with too few members
        if len(subgroup_data) < min_size:
            print(f"Skipping {subgroup} with only {len(subgroup_data)} samples (< {min_size})")
            continue

        # Convert predictions to numeric and handle NaN values
        subgroup_data['pred_numeric'] = pd.to_numeric(
            subgroup_data['predicted_hospital_mortality'],
            errors='coerce'
        )

        # Keep only rows with valid predictions
        valid_mask = ~subgroup_data['pred_numeric'].isna()
        valid_data = subgroup_data[valid_mask]

        # Skip if we don't have enough valid data
        if len(valid_data) < min_size:
            print(f"Skipping {subgroup} with only {len(valid_data)} valid samples (< {min_size})")
            continue

        # Get the valid predictions and corresponding labels
        predictions = valid_data['pred_numeric'].values
        valid_labels = valid_data['mortality_binary'].values

        # Calculate prevalence for this subgroup
        prevalence = np.mean(valid_labels)
        
        # Calculate AUC-ROC if requested
        auc_roc = None
        if compute_auc:
            auc_roc = roc_auc_score(valid_labels, predictions)
            print("AUC-ROC for subgroup", subgroup, ":", auc_roc)
    
        ece = None
        if compute_ece:
            ece = expected_calibration_error(valid_labels, predictions[:, np.newaxis])
            print("ECE for subgroup", subgroup, ":", ece)

        if True:
            brier = np.mean((predictions - valid_labels)**2)
            cross_entropy = -np.mean(valid_labels * np.log(predictions + 1e-15) +
                                    (1 - valid_labels) * np.log(1 - predictions + 1e-15))
            print(f"Subgroup: {subgroup}, Brier : {brier:.5f}, Cross-Entropy: {cross_entropy:.5f}")

        # Create SubgroupResults object
        sg_result = SubgroupResults(
            name=subgroup,
            #display_label=f"{subgroup} (n={len(valid_data)})",
            display_label=f"{subgroup}",
            y_true=valid_labels,
            y_pred_proba=predictions,
            prevalence=prevalence,
            auc_roc=auc_roc  # Include the AUC-ROC value in the constructor
        )

        # Add to results list
        results.append(sg_result)

    return results


def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all CLI options.
    """
    parser = argparse.ArgumentParser(description='Analyze eICU data with visualization options')

    # Data options
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with limited data')
    parser.add_argument('--subgroup-field', type=str, default='ethnicity',
                        help='Field to use for subgroup analysis (default: ethnicity)')
    parser.add_argument('--subgroups', type=str, nargs='+',
                        default=['Caucasian', 'African American'],
                        help='Specific subgroups to analyze (default: all with sufficient samples)')
    parser.add_argument('--min-size', type=int, default=MIN_SUBGROUP_SIZE,
                        help=f'Minimum subgroup size to include in analysis '
                        f'(default: {MIN_SUBGROUP_SIZE})')

    # Visualization options
    parser.add_argument('--ci', action='store_true', help='Show confidence intervals in the plot')
    parser.add_argument('--diamonds', action='store_true', help='Show diamond markers in the plot')
    parser.add_argument('--averages', action='store_true',
                        help='Show average decomposition in the plot')
    parser.add_argument('--nomain', action='store_true',
                        help='Hide main curves')
    parser.add_argument('--calibration', action='store_true',
                        help='Show calibration lines in the plot')
    parser.add_argument('--auc', action='store_true',
                        help='Show AUC-ROC horizontal lines for each subgroup')
    parser.add_argument('--ece', action='store_true',
                        help='Show ECE horizontal lines for each subgroup')
    parser.add_argument('--cost-ratio', type=float, default=DEFAULT_COST_RATIO,
                        help=f'Cost ratio for net benefit calculation '
                        f'(default: {DEFAULT_COST_RATIO})')

    # Output options
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--title', type=str, default='EICU',
                        help='Custom title for the plot')
    parser.add_argument('--maxlogodds', type=float, default=0.99)
    parser.add_argument('--full-width-average', action='store_true',
                        help='Show average lines over full width')
    parser.add_argument('--minaccuracy', type=float, default=0.85)
    parser.add_argument('--style-cycle-offset', type=int, default=None)

    return parser


def main() -> None:
    """Main function to run the eICU analysis and visualization.

    Loads eICU data, computes subgroup statistics, and generates
    net benefit curve plots based on command-line arguments.
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Create EICU model instance
    model = EICU(demo=args.demo)
    print(f"Loaded eICU data for {len(model.data)} patients")

    # Get subgroup results
    subgroup_results = get_subgroup_stats(
        model,
        subgroup_field=args.subgroup_field,
        subgroups=args.subgroups,
        min_size=args.min_size,
        compute_auc=args.auc,  # Pass the AUC flag to compute AUC-ROC
        compute_ece=args.ece
    )

    # Check if we have any valid subgroups
    if len(subgroup_results) == 0:
        print("No subgroups with sufficient samples found. Exiting.")
        return

    # Create figure and axes
    _, ax = plt.subplots(figsize=(4, 4))

    # Plot using the new API directly
    ax = plot_net_benefit_curves(
        subgroups=subgroup_results,
        cost_ratio=args.cost_ratio,
        ax=ax,
        compute_ci=args.ci,
        n_bootstrap=100 if args.ci else 0,
        compute_calibrated=args.calibration,
        show_diamonds=args.diamonds,
        show_averages=args.averages,
        hide_main=args.nomain,
        # Set diamond_shift_amount to scipy.special.logit(0.16) if diamonds are shown
        diamond_shift_amount=scipy.special.logit(0.16) if args.diamonds else None,
        max_logodds=np.log(args.maxlogodds / (1 - args.maxlogodds)),
        full_width_average=args.full_width_average,
        min_accuracy=args.minaccuracy,
        style_cycle_offset=args.style_cycle_offset
    )

    # Set title
    title = args.title or f"eICU In-Hospital Mortality Accuracy by {args.subgroup_field.title()}"
    ax.set_title(title)
    #ax.get_legend().remove()
    plt.ylabel("Accuracy")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create descriptive filename
    filename_parts = [
        f"c-{args.cost_ratio}"
    ]

    # Add flags to filename
    if args.demo:
        filename_parts.append("demo")
    if args.subgroup_field:
        filename_parts.append(args.subgroup_field)
    if args.ci:
        filename_parts.append("ci")
    if args.diamonds:
        filename_parts.append("diamonds")
    if args.averages:
        filename_parts.append("averages")
    if args.calibration:
        filename_parts.append("calibration")
    if args.auc:
        filename_parts.append("auc")

    # Save plot
    output_path = os.path.join(args.output_dir, f"{'-'.join(filename_parts)}-new.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
