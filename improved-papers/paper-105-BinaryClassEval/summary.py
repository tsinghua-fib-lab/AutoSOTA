#!/usr/bin/env python
"""
Evaluates model performance across demographic subgroups in the eICU dataset.
Calculates AUC-ROC with bootstrap confidence intervals and accuracy using
prevalence-adjusted thresholds.
"""
from __future__ import annotations

import argparse
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from etl.eicu import EICU
from reproducibility.seed_manager import get_random_generator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay
import os

# Constants
MIN_SAMPLES = 50  # Minimum number of samples required for analysis
DEFAULT_BOOTSTRAPS = 1000
DEFAULT_CI = 0.95


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    ci: float = DEFAULT_CI,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence intervals for AUC-ROC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probabilities for the positive class.
    n_bootstraps : int, default=DEFAULT_BOOTSTRAPS
        Number of bootstrap samples to draw.
    ci : float, default=DEFAULT_CI
        Confidence interval level (e.g., 0.95 for 95% CI).
    random_seed : Optional[int], default=None
        Seed for random number generation. If None, uses default seed from config.

    Returns
    -------
    Tuple[float, float, float]
        Tuple of (AUC-ROC, lower CI bound, upper CI bound).
    """
    # Get the config to access the seed if not provided
    from reproducibility.config import get_config
    config = get_config()
    if random_seed is None:
        random_seed = config.seed
        
    auc_roc = roc_auc_score(y_true, y_pred)

    bootstrap_aucs = []
    rng = get_random_generator(seed=random_seed, component_name="bootstrap_auc_ci")
    
    # Log this analysis for reproducibility
    config.log_analysis(
        "bootstrap_auc_ci", 
        {
            "seed": random_seed, 
            "n_bootstraps": n_bootstraps,
            "ci": ci
        }
    )

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip bootstrap samples with only one class
        bootstrap_aucs.append(roc_auc_score(y_true[indices], y_pred[indices]))

    # Get confidence interval bounds
    alpha = (1 - ci) / 2
    lower_bound = max(0.0, float(np.percentile(bootstrap_aucs, alpha * 100)))
    upper_bound = min(1.0, float(np.percentile(bootstrap_aucs, (1 - alpha) * 100)))

    return float(auc_roc), lower_bound, upper_bound


def calculate_accuracy_at_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prevalence: float
) -> Tuple[float, float]:
    """Calculate accuracy using 1 - prevalence as the threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    prevalence : float
        Positive class prevalence (proportion).

    Returns
    -------
    Tuple[float, float]
        Tuple of (accuracy at the specified threshold, threshold value).
    """
    threshold = 1 - prevalence
    y_pred = (y_pred_proba >= threshold).astype(int)
    return accuracy_score(y_true, y_pred), threshold


def get_valid_subgroup_data(
    model: EICU,
    subgroup_field: str,
    subgroup_value: str
) -> Optional[Tuple[np.ndarray, np.ndarray, int, float]]:
    """Extract valid prediction data for a specific subgroup.

    Parameters
    ----------
    model : EICU
        The EICU model instance containing patient data.
    subgroup_field : str
        Field to filter by (e.g., 'ethnicity', 'gender').
    subgroup_value : str
        Value of the subgroup field to select.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, int, float]]
        Tuple of (predictions, labels, count, prevalence) or None if insufficient data.
    """
    # Filter data for this subgroup
    subgroup_mask = model.data[subgroup_field] == subgroup_value
    subgroup_data = model.data[subgroup_mask].copy()

    # Check if subgroup has enough samples
    if len(subgroup_data) < MIN_SAMPLES:
        return None

    # Convert predictions to numeric and handle NaN values
    subgroup_data['pred_numeric'] = pd.to_numeric(
        subgroup_data['predicted_hospital_mortality'],
        errors='coerce'
    )

    # Keep only rows with valid predictions
    valid_mask = ~subgroup_data['pred_numeric'].isna()
    valid_data = subgroup_data[valid_mask]

    # Check if there are enough valid predictions
    if len(valid_data) < MIN_SAMPLES:
        return None

    # Get the valid predictions and corresponding labels
    predictions = valid_data['pred_numeric'].values
    labels = valid_data['mortality_binary'].values
    count = len(valid_data)
    prevalence = np.mean(labels)

    return predictions, labels, count, prevalence


def subgroup_metrics(
    model: EICU,
    subgroup_field: str = 'ethnicity',
    n_bootstraps: int = DEFAULT_BOOTSTRAPS
) -> Dict[str, Dict[str, float]]:
    """Evaluate the model on different demographic subgroups.

    Parameters
    ----------
    model : EICU
        The EICU model instance containing patient data.
    subgroup_field : str, default='ethnicity'
        Field to split subgroups by (e.g., 'ethnicity', 'gender').
    n_bootstraps : int, default=DEFAULT_BOOTSTRAPS
        Number of bootstrap samples for confidence intervals.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping subgroup values to dictionaries of metrics.
    """
    # Get all unique values for the subgroup field
    subgroups = model.data[subgroup_field].dropna().unique()
    results = {}

    # Calculate metrics for each subgroup
    for subgroup in subgroups:
        subgroup_data = get_valid_subgroup_data(model, subgroup_field, subgroup)
        if subgroup_data is None:
            continue

        predictions, labels, count, prevalence = subgroup_data

        # Calculate AUC-ROC with bootstrap confidence intervals
        auc_roc, auc_lower, auc_upper = bootstrap_auc_ci(
            labels, predictions, n_bootstraps=n_bootstraps
        )

        # Calculate accuracy using 1 - prevalence as threshold
        accuracy, threshold = calculate_accuracy_at_threshold(
            labels, predictions, prevalence
        )

        # Store results
        results[subgroup] = {
            'prevalence': prevalence,
            'count': count,
            'auc_roc': auc_roc,
            'auc_lower': auc_lower,
            'auc_upper': auc_upper,
            'threshold': threshold,
            'accuracy': accuracy
        }

    return results


def print_results(
    subgroup_results: Dict[str, Dict[str, Any]],
    subgroup_field: str,
    target_groups: list = ['African American', 'Caucasian']
) -> None:
    """Print formatted results for specified target subgroups.

    Parameters
    ----------
    subgroup_results : Dict[str, Dict[str, Any]]
        Dictionary of results by subgroup.
    subgroup_field : str
        Field used for subgroup analysis (e.g., 'ethnicity').
    target_groups : list, default=['African American', 'Caucasian']
        List of target groups to include in the results.
    """
    # Print AUC-ROC for target subgroups with confidence intervals
    print(f"\nAUC-ROC by {subgroup_field} with 95% confidence intervals:")
    for subgroup, data in subgroup_results.items():
        if subgroup in target_groups:
            print(f"{subgroup}: {data['auc_roc']:.4f} "
                f"(95% CI: {data['auc_lower']:.4f}-{data['auc_upper']:.4f}, "
                f"n={data['count']})")

    # Print accuracy for target subgroups with threshold = 1 - prevalence
    print(f"\nAccuracy by {subgroup_field} using threshold = 1 - prevalence:")
    for subgroup, data in subgroup_results.items():
        if subgroup in target_groups:
            print(f"{subgroup}: {data['accuracy']:.4f} "
                f"(threshold: {data['threshold']:.4f}, "
                f"prevalence: {data['prevalence']:.4f}, "
                f"n={data['count']})")


def plot_roc_curves(
    model: EICU,
    subgroup_results: Dict[str, Dict[str, Any]],
    subgroup_field: str = 'ethnicity',
    save_path: Optional[str] = None,
    target_groups: list = ['African American', 'Caucasian']
) -> None:
    """Plot ROC curves for specified target subgroups with confidence intervals.

    Parameters
    ----------
    model : EICU
        The EICU model instance containing patient data.
    subgroup_results : Dict[str, Dict[str, Any]]
        Dictionary of results by subgroup.
    subgroup_field : str, default='ethnicity'
        Field used for subgroup analysis (e.g., 'ethnicity', 'gender').
    save_path : Optional[str], default=None
        Path to save the plot. If None, the plot is displayed.
    target_groups : list, default=['African American', 'Caucasian']
        List of target groups to include in the plot.
    """
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a colormap for different subgroups
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Plot the diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, lw=1, label='Chance')
    
    # Loop through each subgroup and plot their ROC curve
    for i, (subgroup, data) in enumerate(subgroup_results.items()):
        # Skip if not in target groups
        if subgroup not in target_groups:
            continue
            
        # Get the raw data for this subgroup
        subgroup_data = get_valid_subgroup_data(model, subgroup_field, subgroup)
        if subgroup_data is None:
            continue
        
        predictions, labels, count, prevalence = subgroup_data
        
        # Plot the ROC curve for this subgroup
        fpr, tpr, _ = roc_curve(labels, predictions)
        display = RocCurveDisplay(
            fpr=fpr, 
            tpr=tpr, 
            roc_auc=data['auc_roc'],
            estimator_name=f"{subgroup} (n={count})"
        )
        display.plot(ax=ax, color=colors[i % len(colors)])
        
        # Add confidence interval to the legend label
        # Find the line corresponding to this subgroup
        for line in ax.get_lines():
            if hasattr(line, 'get_label'):
                label = str(line.get_label())  # Convert to string to safely use 'in' operator
                if subgroup in label:
                    line.set_label(f"{subgroup} (AUC: {data['auc_roc']:.3f}, 95% CI: {data['auc_lower']:.3f}-{data['auc_upper']:.3f})")
    
    # Customize the plot
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if target_groups == ['African American', 'Caucasian']:
        ax.set_title('ROC Curves Comparing African American and Caucasian Patients')
    else:
        ax.set_title(f'ROC Curves by {subgroup_field.title()}')
    
    # Adjust the legend
    ax.legend(loc='lower right', fontsize='small')
    
    # Set the x and y limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add grid for better readability
    ax.grid(alpha=0.3)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    """Run subgroup analysis on eICU data.

    Loads eICU data, computes AUC-ROC and accuracy metrics across subgroups,
    and generates ROC curve plots with confidence intervals.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze eICU data performance across demographic subgroups'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with limited data')
    parser.add_argument('--bootstraps', type=int, default=DEFAULT_BOOTSTRAPS,
                        help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--subgroup-field', type=str, default='ethnicity',
                        help='Demographic field to analyze (e.g., ethnicity, gender)')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save the ROC plot (if not provided, plot is displayed)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting ROC curves')
    parser.add_argument('--target-groups', nargs='+', 
                        default=['African American', 'Caucasian'],
                        help='Target groups to include in the analysis (default: African American and Caucasian)')
    args = parser.parse_args()

    # Load model and data
    model = EICU(demo=args.demo)
    print(f"Loaded eICU data for {len(model.data)} patients")
    print(f"Overall mortality rate: {model.train_prevalence*100:.2f}%")

    # Get subgroup analysis data
    subgroup_results = subgroup_metrics(
        model,
        subgroup_field=args.subgroup_field,
        n_bootstraps=args.bootstraps
    )

    # Use the target groups from command line
    target_groups = args.target_groups
    
    # Print results for target groups only
    print_results(subgroup_results, args.subgroup_field, target_groups=target_groups)
    
    # Plot ROC curves unless explicitly disabled
    if not args.no_plot:
        plot_roc_curves(
            model, 
            subgroup_results, 
            subgroup_field=args.subgroup_field,
            save_path=args.save_plot,
            target_groups=target_groups
        )


if __name__ == "__main__":
    main()
