"""
Diagnostic utilities for analyzing multi-task regression models on macaque reaching data.

This module provides functions to diagnose model performance by comparing predictions
to targets and computing various statistics.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def diagnose_macaque_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[torch.device, str] = 'cpu',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostics on a trained macaque reaching model.
    
    This function collects predictions and targets from the validation set,
    computes various statistics, and compares model performance to a baseline
    mean predictor. It's designed for multi-task models that predict both
    position (pos_xy) and velocity (vel_xy).
    
    Args:
        model: Trained PyTorch model with forward() returning dict with 'preds_tasks'
        dataloader: DataLoader for validation/test set
        device: Device to run inference on ('cpu', 'cuda', or torch.device)
        verbose: If True, print detailed diagnostic information
    
    Returns:
        Dictionary containing all diagnostic statistics with keys:
            - 'pos_targets': Statistics dict for position targets
            - 'vel_targets': Statistics dict for velocity targets
            - 'pos_preds': Statistics dict for position predictions
            - 'vel_preds': Statistics dict for velocity predictions
            - 'r2': R² scores dict
            - 'correlation': Pearson correlation dict
            - 'mse_comparison': MSE comparison dict (model vs baseline)
            - 'raw_data': Dict with tensors (pos_targets, vel_targets, pos_preds, vel_preds)
    
    Example:
        >>> diagnostics = diagnose_macaque_model(model, val_loader, device='cuda')
        >>> print(f"Velocity R²: {diagnostics['r2']['vel_mean']:.4f}")
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # ============================================================================
    # Step 1: Collect predictions and targets
    # ============================================================================
    pos_list, vel_list, pos_preds_list, vel_preds_list = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            
            # Check if multi-task or single-task
            if 'preds_tasks' in outputs:
                # Multi-task: collect both pos and vel
                if hasattr(batch, 'pos_xy'):
                    pos_list.append(batch.pos_xy)
                if hasattr(batch, 'vel_xy'):
                    vel_list.append(batch.vel_xy)
                if 'pos' in outputs['preds_tasks']:
                    pos_preds_list.append(outputs['preds_tasks']['pos'])
                if 'vel' in outputs['preds_tasks']:
                    vel_preds_list.append(outputs['preds_tasks']['vel'])
            else:
                # Single-task: determine which task from model's target_name
                target_names = getattr(model, 'target_names')
                target_key = target_names[0] if (target_names is not None and len(target_names) == 1) \
                    else 'pos_xy'
                if 'vel' in target_key.lower():
                    vel_list.append(batch.vel_xy)
                    vel_preds_list.append(outputs['preds'])
                elif 'pos' in target_key.lower():
                    pos_list.append(batch.pos_xy)
                    pos_preds_list.append(outputs['preds'])
                else:
                    raise ValueError(
                        f"Cannot determine task from target_key: {target_key}"
                    )
    
    # Concatenate all batches (handle empty lists for single-task)
    pos_targets = torch.cat(pos_list, dim=0).cpu() if pos_list else None
    vel_targets = torch.cat(vel_list, dim=0).cpu() if vel_list else None
    pos_preds = torch.cat(pos_preds_list, dim=0).cpu() if pos_preds_list else None
    vel_preds = torch.cat(vel_preds_list, dim=0).cpu() if vel_preds_list else None
    
    # ============================================================================
    # Step 2: Compute statistics
    # ============================================================================
    
    def compute_stats(tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute mean, std, var, min, max for a tensor."""
        return {
            'mean': tensor.mean(dim=0).numpy(),
            'std': tensor.std(dim=0).numpy(),
            'var': tensor.var(dim=0).numpy(),
            'min': tensor.min(dim=0).values.numpy(),
            'max': tensor.max(dim=0).values.numpy(),
        }
    
    def compute_r2(
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute R² = 1 - SS_res / SS_tot per dimension and mean."""
        ss_res = ((preds - targets) ** 2).sum(dim=0)
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1 - (ss_res / ss_tot)
        r2_np = r2.numpy()
        return {
            'per_dim': r2_np,
            'mean': float(r2_np.mean()),
        }
    
    def compute_correlation(
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute Pearson correlation per dimension and mean."""
        corr = np.zeros(preds.shape[1])
        for i in range(preds.shape[1]):
            corr[i] = np.corrcoef(preds[:, i].numpy(), targets[:, i].numpy())[0, 1]
        return {
            'per_dim': corr,
            'mean': float(corr.mean()),
        }
    
    def compute_mse_comparison(
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Compare model MSE to baseline (mean predictor) MSE."""
        baseline_mse = ((targets - targets.mean(dim=0)) ** 2).mean(dim=0).numpy()
        model_mse = ((preds - targets) ** 2).mean(dim=0).numpy()
        improvement = baseline_mse - model_mse
        return {
            'baseline': baseline_mse,
            'model': model_mse,
            'improvement': improvement,
            'relative_improvement': improvement / (baseline_mse + 1e-10),
        }
    
    # Compute all diagnostics (only for tasks with data)
    results = {
        'pos_targets': compute_stats(pos_targets) if pos_targets is not None else None,
        'vel_targets': compute_stats(vel_targets) if vel_targets is not None else None,
        'pos_preds': compute_stats(pos_preds) if pos_preds is not None else None,
        'vel_preds': compute_stats(vel_preds) if vel_preds is not None else None,
        'r2': {},
        'correlation': {},
        'mse_comparison': {},
        'raw_data': {
            'pos_targets': pos_targets,
            'vel_targets': vel_targets,
            'pos_preds': pos_preds,
            'vel_preds': vel_preds,
        }
    }
    
    # Compute per-task metrics only if data exists
    if pos_preds is not None and pos_targets is not None:
        results['r2']['pos'] = compute_r2(pos_preds, pos_targets)['per_dim']
        results['r2']['pos_mean'] = compute_r2(pos_preds, pos_targets)['mean']
        results['correlation']['pos'] = compute_correlation(pos_preds, pos_targets)['per_dim']
        results['correlation']['pos_mean'] = compute_correlation(pos_preds, pos_targets)['mean']
        results['mse_comparison']['pos'] = compute_mse_comparison(pos_preds, pos_targets)
    
    if vel_preds is not None and vel_targets is not None:
        results['r2']['vel'] = compute_r2(vel_preds, vel_targets)['per_dim']
        results['r2']['vel_mean'] = compute_r2(vel_preds, vel_targets)['mean']
        results['correlation']['vel'] = compute_correlation(vel_preds, vel_targets)['per_dim']
        results['correlation']['vel_mean'] = compute_correlation(vel_preds, vel_targets)['mean']
        results['mse_comparison']['vel'] = compute_mse_comparison(vel_preds, vel_targets)
    
    # ============================================================================
    # Step 3: Print diagnostics (if verbose)
    # ============================================================================
    
    if verbose:
        # Determine which tasks have data
        active_tasks = []
        if pos_preds is not None:
            active_tasks.append('pos')
        if vel_preds is not None:
            active_tasks.append('vel')
        
        # print('=' * 80)
        print("DIAGNOSTIC 1: Target Variance Analysis")
        print("-" * 80)
        print("Target Statistics (should have meaningful variance):")
        for task in active_tasks:
            stats = results[f'{task}_targets']
            if stats is not None:
                print(f"  {task}_xy:")
                print(f"    mean: {stats['mean']}")
                print(f"    std:  {stats['std']}")
                print(f"    var:  {stats['var']}")
                print(f"    min:  {stats['min']}")
                print(f"    max:  {stats['max']}")
        
        # print("\n" + '=' * 80)
        print("\nDIAGNOSTIC 2: Prediction Statistics (check if nearly constant)")
        print("-" * 80)
        print("Prediction Statistics:")
        for task in active_tasks:
            stats = results[f'{task}_preds']
            if stats is not None:
                print(f"  {task}_preds:")
                print(f"    mean: {stats['mean']}")
                print(f"    std:  {stats['std']}")
                print(f"    var:  {stats['var']}")
                print(f"    min:  {stats['min']}")
                print(f"    max:  {stats['max']}")
        
        # print("\n" + '=' * 80)
        print("\nDIAGNOSTIC 3: Manual R² Calculation")
        print("-" * 80)
        print(f"R² Scores (per dimension):")
        for task in active_tasks:
            if f'{task}' in results['r2']:
                print(f"  {task}_xy: {results['r2'][task]}")
        for task in active_tasks:
            if f'{task}_mean' in results['r2']:
                print(f"  {task}_xy (mean): {results['r2'][f'{task}_mean']:.6f}")
        
        # print("\n" + '=' * 80)
        print("\nDIAGNOSTIC 4: Prediction-Target Correlation")
        print("-" * 80)
        print(f"Pearson Correlation (preds vs targets, per dimension):")
        for task in active_tasks:
            if task in results['correlation']:
                print(f"  {task}_xy: {results['correlation'][task]}")
        for task in active_tasks:
            if f'{task}_mean' in results['correlation']:
                print(f"  {task}_xy (mean): {results['correlation'][f'{task}_mean']:.6f}")
        
        # print("\n" + '=' * 80)
        print("\nDIAGNOSTIC 5: Baseline Comparison (Mean Predictor)")
        print("-" * 80)
        print(f"MSE Comparison (lower is better):")
        for task in active_tasks:
            if task in results['mse_comparison']:
                comp = results['mse_comparison'][task]
                print(f"  {task}_xy:")
                print(f"    Baseline (mean predictor): {comp['baseline']}")
                print(f"    Model:                     {comp['model']}")
                print(f"    Improvement:               {comp['improvement']}")
                # Handle both array and scalar cases for relative improvement
                rel_imp = comp['relative_improvement']
                if isinstance(rel_imp, np.ndarray) and rel_imp.size > 1:
                    rel_imp_mean = rel_imp.mean() * 100
                    print(f"    Relative improvement:      {rel_imp} (mean: {rel_imp_mean:.2f}%)")
                else:
                    rel_imp_scalar = float(rel_imp) if isinstance(rel_imp, np.ndarray) else rel_imp
                    print(f"    Relative improvement:      {rel_imp_scalar*100:.2f}%")
        
        # print("\n" + '=' * 80)
        print("\nSUMMARY")
        print("-" * 80)
        
        # Compute overall metrics only from active tasks
        r2_values = [results['r2'][f'{task}_mean'] for task in active_tasks if f'{task}_mean' in results['r2']]
        corr_values = [results['correlation'][f'{task}_mean'] for task in active_tasks if f'{task}_mean' in results['correlation']]
        
        if r2_values:
            overall_r2 = sum(r2_values) / len(r2_values)
            print(f"Overall R² (mean across {'/'.join(active_tasks)}): {overall_r2:.6f}")
        if corr_values:
            overall_corr = sum(corr_values) / len(corr_values)
            print(f"Overall Correlation: {overall_corr:.6f}")
        
        # Interpretation hints (use primary task for interpretation)
        print("\nInterpretation:")
        
        # Determine primary task for hints (prefer vel, fall back to pos)
        primary_task = 'vel' if 'vel' in active_tasks else ('pos' if 'pos' in active_tasks else None)
        
        if primary_task and f'{primary_task}_mean' in results['r2']:
            r2_val = results['r2'][f'{primary_task}_mean']
            if r2_val < 0.1:
                print(f"  ⚠️  Low R² (<0.1): Model is barely better than predicting the mean")
            elif r2_val < 0.5:
                print(f"  ⚠️  Moderate R² (0.1-0.5): Model has learned some patterns but could improve")
            else:
                print(f"  ✓  Good R² (>0.5): Model is capturing meaningful variance")
        
        if primary_task and results[f'{primary_task}_preds'] and results[f'{primary_task}_targets']:
            pred_std = results[f'{primary_task}_preds']['std'].mean()
            target_std = results[f'{primary_task}_targets']['std'].mean()
            pred_std_ratio = pred_std / target_std if target_std > 0 else 0
            if pred_std_ratio < 0.3:
                print("  ⚠️  Predictions have much lower variance than targets (ratio < 0.3)")
                print("     -> Model may be underfitting or not confident in its predictions")
            elif pred_std_ratio > 2.0:
                print("  ⚠️  Predictions have much higher variance than targets (ratio > 2.0)")
                print("     -> Model may be overfitting or unstable")
        
        # print('=' * 80)
    
    return results


def run_diagnostics(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Union[torch.device, str] = 'cpu',
    verbose: bool = True,
    r2_threshold: float = 0.1,
    mse_ratio_threshold: float = 1.3,
) -> Dict[str, Any]:
    """
    Diagnose overfitting by comparing model performance on train vs validation sets.
    
    This function runs the model on both training and validation sets, computes
    metrics for each, and compares them to detect signs of overfitting. Overfitting
    is indicated when training metrics are substantially better than validation metrics.
    
    Args:
        model: Trained PyTorch model with forward() returning dict with 'preds_tasks'
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        device: Device to run inference on ('cpu', 'cuda', or torch.device)
        verbose: If True, print detailed diagnostic information
        r2_threshold: Threshold for R² difference (train - val) to flag overfitting
        mse_ratio_threshold: Threshold for MSE ratio (val/train) to flag overfitting
    
    Returns:
        Dictionary containing:
            - 'train_results': Full diagnostics dict from train set
            - 'val_results': Full diagnostics dict from validation set
            - 'overfitting_metrics': Dict with comparison metrics
            - 'overfitting_detected': Bool indicating if overfitting was detected
    
    Example:
        >>> results = diagnose_overfitting(model, train_loader, val_loader, device='cuda')
        >>> if results['overfitting_detected']:
        ...     print("Warning: Overfitting detected!")
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # ============================================================================
    # Step 1: Run diagnostics on both datasets
    # ============================================================================
    
    if verbose:
        print("\n" + '=' * 80)
        print("RUNNING DIAGNOSTICS ON TRAINING SET")
        print('=' * 80)
    
    train_results = diagnose_macaque_model(
        model=model,
        dataloader=train_loader,
        device=device,
        verbose=verbose,
    )
    
    if verbose:
        print("\n" + '=' * 80)
        print("RUNNING DIAGNOSTICS ON VALIDATION SET")
        print('=' * 80)
    
    val_results = diagnose_macaque_model(
        model=model,
        dataloader=val_loader,
        device=device,
        verbose=verbose,
    )
    
    # ============================================================================
    # Step 2: Compare metrics
    # ============================================================================
    
    overfitting_metrics = {
        'r2_diff': {},
        'mse_ratio': {},
        'correlation_diff': {},
    }
    
    # Determine which tasks are present
    active_tasks = []
    if 'pos_mean' in train_results['r2'] and 'pos_mean' in val_results['r2']:
        active_tasks.append('pos')
    if 'vel_mean' in train_results['r2'] and 'vel_mean' in val_results['r2']:
        active_tasks.append('vel')
    
    # Compute comparison metrics for each task
    for task in active_tasks:
        # R² difference (train - val): positive means train is better
        train_r2 = train_results['r2'][f'{task}_mean']
        val_r2 = val_results['r2'][f'{task}_mean']
        overfitting_metrics['r2_diff'][task] = train_r2 - val_r2
        
        # MSE ratio (val / train): > 1 means val is worse
        train_mse = train_results['mse_comparison'][task]['model'].mean()
        val_mse = val_results['mse_comparison'][task]['model'].mean()
        overfitting_metrics['mse_ratio'][task] = val_mse / (train_mse + 1e-10)
        
        # Correlation difference (train - val)
        train_corr = train_results['correlation'][f'{task}_mean']
        val_corr = val_results['correlation'][f'{task}_mean']
        overfitting_metrics['correlation_diff'][task] = train_corr - val_corr
    
    # ============================================================================
    # Step 3: Detect overfitting
    # ============================================================================
    
    overfitting_flags = []
    
    for task in active_tasks:
        r2_diff = overfitting_metrics['r2_diff'][task]
        mse_ratio = overfitting_metrics['mse_ratio'][task]
        
        if r2_diff > r2_threshold:
            overfitting_flags.append(
                f"{task}: R² gap of {r2_diff:.4f} (train >> val)"
            )
        
        if mse_ratio > mse_ratio_threshold:
            overfitting_flags.append(
                f"{task}: MSE ratio of {mse_ratio:.4f} (val MSE >> train MSE)"
            )
    
    overfitting_detected = len(overfitting_flags) > 0
    
    # ============================================================================
    # Step 4: Print overfitting diagnostics
    # ============================================================================
    
    if verbose:
        print("\n" + '=' * 80)
        print("OVERFITTING ANALYSIS")
        print('=' * 80)
        print("Comparison of Train vs Validation Metrics:")
        print("-" * 80)
        
        for task in active_tasks:
            print(f"{task.upper()} Task:")
            print(f"  R² Score:")
            print(f"    Train:      {train_results['r2'][f'{task}_mean']:.6f}")
            print(f"    Validation: {val_results['r2'][f'{task}_mean']:.6f}")
            print(f"    Difference: {overfitting_metrics['r2_diff'][task]:+.6f} " +
                  f"{'⚠️' if overfitting_metrics['r2_diff'][task] > r2_threshold else '✓'}")
            
            print(f"  MSE:")
            train_mse = train_results['mse_comparison'][task]['model'].mean()
            val_mse = val_results['mse_comparison'][task]['model'].mean()
            print(f"    Train:      {train_mse:.6f}")
            print(f"    Validation: {val_mse:.6f}")
            print(f"    Ratio (val/train): {overfitting_metrics['mse_ratio'][task]:.4f} " +
                  f"{'⚠️' if overfitting_metrics['mse_ratio'][task] > mse_ratio_threshold else '✓'}")
            
            print(f"  Correlation:")
            print(f"    Train:      {train_results['correlation'][f'{task}_mean']:.6f}")
            print(f"    Validation: {val_results['correlation'][f'{task}_mean']:.6f}")
            print(f"    Difference: {overfitting_metrics['correlation_diff'][task]:+.6f}")
        
        # print("\n" + "-" * 80)
        print("\nOVERFITTING SUMMARY")
        print("-" * 80)
        
        if overfitting_detected:
            print("⚠️  OVERFITTING DETECTED")
            print("\nSigns of overfitting:")
            for flag in overfitting_flags:
                print(f"  - {flag}")
            
            print("\nRecommendations:")
            print("  1. Increase regularization (weight decay, dropout)")
            print("  2. Reduce model complexity")
            print("  3. Collect more training data")
            print("  4. Apply data augmentation")
            print("  5. Use early stopping based on validation metrics")
        else:
            print("✓  NO SIGNIFICANT OVERFITTING DETECTED")
            print(f"\nTrain and validation metrics are reasonably close:")
            print(f"  - R² differences < {r2_threshold}")
            print(f"  - MSE ratios < {mse_ratio_threshold}")
        
        print('=' * 80)
    
    return {
        'train_results': train_results,
        'val_results': val_results,
        'overfitting_metrics': overfitting_metrics,
        'overfitting_detected': overfitting_detected,
        'overfitting_flags': overfitting_flags,
    }