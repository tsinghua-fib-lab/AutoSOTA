#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

import torch
import numpy as np

from models.product import MultiBinaryProductModelWithOracle

def train_until_convergence(
    model, 
    optimizer, 
    inputs, 
    max_steps, 
    convergence_threshold, 
    stagnation_window=-1, 
    stagnation_threshold=1e-3,
    verbose=True,
    print_interval=100,
    record_history=False,
    record_weights=False,
):
    """
    Train model until p_diff < convergence_threshold or max_steps reached.
    
    Args:
        model: Model to train (must return y_oracle, y_model, p_epsilon, p_diff)
        optimizer: Optimizer instance
        inputs: Input data tensor [batch_size, n_inputs]
        max_steps: Maximum number of training steps
        convergence_threshold: Convergence threshold for mean p_diff
        stagnation_window: Window size for checking improvement rate (default: -1 = disabled)
                          If > 0, training stops early if mean improvement over window < stagnation_threshold
        stagnation_threshold: Minimum mean improvement per step required (default: 1e-3)
                             Rate is measured as: (p_diff[start] - p_diff[end]) / window_size
        verbose: Whether to print progress (default: True)
        print_interval: Print progress every N steps (default: 100)
        record_history: Whether to record full training history (default: False)
        record_weights: Whether to record weight during training (default: False)
    
    Returns:
        Dictionary containing:
        - steps: Number of steps taken (max_steps if didn't converge or stagnated)
        - final_p_diff: Final mean p_diff value
        - final_p_epsilon: Final mean p_epsilon value
        - final_loss: Final loss value
        - converged: Whether convergence was reached
        - stagnated: Whether stagnation was detected
        - oracle_weights: Oracle weights tensor [n_inputs, n_outputs] (if record_weights=True)
        - model_weights_init: Model weights tensor at init [n_inputs, n_outputs] (if record_weights=True)
        - history: Training history (if record_history=True), contains:
            - p_diff: List of mean p_diff values per step
            - p_epsilon: List of mean p_epsilon values per step
            - loss: List of loss values per step
            - model_weights: List of model weights at recorded steps (if record_weights=True)
    """
    p_diff_history = []
    p_epsilon_history = []
    loss_history = []
    if record_weights:
        weights_history = []
        with torch.no_grad():
            oracle_weights = model.product_oracle.product_weights.cpu().numpy()
            if 'oracle_weights' not in locals():
                oracle_weights = []
        with torch.no_grad():
            model_weights_init = model.product_model.product_weights.cpu().numpy()
            if 'model_weights_init' not in locals():
                model_weights_init = []
    
    converged = False
    stagnated = False
    
    for step in range(max_steps):
        optimizer.zero_grad()
        y_oracle, y_model, p_epsilon, p_diff = model(inputs)
        
        # Compute loss: sum over outputs (independent XOR nodes), mean over batch
        # (needed if the number of outputs is variable from exp to exp)
        loss = torch.sum((y_model - y_oracle) ** 2, dim=1).mean()
        loss.backward()
        optimizer.step()
        
        # Track metrics - ALWAYS compute these, needed for convergence check
        mean_p_diff = p_diff.mean().item()
        mean_p_epsilon = p_epsilon.mean().item()
        loss_value = loss.item()
        
        # Store in history
        if record_history:
            # If recording full history, keep all values
            p_diff_history.append(mean_p_diff)
            p_epsilon_history.append(mean_p_epsilon)
            loss_history.append(loss_value)
            if record_weights:
                with torch.no_grad():
                    model_weights = model.product_model.product_weights.cpu().numpy()
                    if 'model_weights' not in locals():
                        model_weights = []
                    weights_history.append(model_weights)

        else:
            # If not recording full history, only keep sliding window for stagnation detection
            p_diff_history.append(mean_p_diff)
            if len(p_diff_history) > max(stagnation_window, 1):
                p_diff_history.pop(0)
        
        # Check convergence
        if mean_p_diff < convergence_threshold:
            converged = True
            if verbose:
                print(f"\r  Converged at step {step+1}: p_diff={mean_p_diff:.4f} < {convergence_threshold:.4f}                    ")
            return {
                'steps': step + 1,
                'final_p_diff': mean_p_diff,
                'final_p_epsilon': mean_p_epsilon,
                'final_loss': loss_value,
                'converged': True,
                'stagnated': False,
                'oracle_weights': oracle_weights if record_weights else None,
                'model_weights_init': model_weights_init if record_weights else None,
                'history': {
                    'p_diff': p_diff_history,
                    'p_epsilon': p_epsilon_history,
                    'loss': loss_history,
                    'model_weights': weights_history if record_weights else None
                } if record_history else None
            }
        
        # Check stagnation (if enabled)
        if stagnation_window > 0 and len(p_diff_history) >= stagnation_window:
            # Calculate mean improvement over the window
            window_start = p_diff_history[-stagnation_window]
            window_end = p_diff_history[-1]
            mean_improvement = (window_start - window_end) / stagnation_window
            
            # If improvement is too small (or negative = getting worse), stop early
            if mean_improvement < stagnation_threshold or np.isnan(mean_improvement):
                stagnated = True
                if verbose:
                    print(f"\r  Stagnation detected at step {step+1}: improvement={mean_improvement:.2e} < {stagnation_threshold:.2e}                    ")
                return {
                    'steps': -1,
                    'final_p_diff': mean_p_diff,
                    'final_p_epsilon': mean_p_epsilon,
                    'final_loss': loss_value,
                    'converged': False,
                    'stagnated': True,
                    'oracle_weights': oracle_weights if record_weights else None,
                    'model_weights_init': model_weights_init if record_weights else None,
                    'history': {
                        'p_diff': p_diff_history,
                        'p_epsilon': p_epsilon_history,
                        'loss': loss_history,
                        'model_weights': weights_history if record_weights else None
                    } if record_history else None
                }
        
        # Print progress
        if verbose and (step + 1) % print_interval == 0:
            if stagnation_window > 0 and len(p_diff_history) >= stagnation_window:
                # Show current improvement rate
                recent_improvement = (p_diff_history[-stagnation_window] - p_diff_history[-1]) / stagnation_window
                print(f"\r  Step {step+1}/{max_steps}, p_diff={mean_p_diff:.4f}, improvement_rate={recent_improvement:.2e}", end="")
            else:
                print(f"\r  Step {step+1}/{max_steps}, p_diff={mean_p_diff:.4f}", end="")
    
    # Did not converge
    if verbose:
        print(f"\r  Did not converge in {max_steps} steps (p_diff={mean_p_diff:.4f})                    ")
    
    return {
        'steps': -1,
        'final_p_diff': mean_p_diff,
        'final_p_epsilon': mean_p_epsilon,
        'final_loss': loss_value,
        'converged': False,
        'stagnated': False,
        'oracle_weights': oracle_weights if record_weights else None,
        'model_weights_init': model_weights_init if record_weights else None,
        'history': {
            'p_diff': p_diff_history,
            'p_epsilon': p_epsilon_history,
            'loss': loss_history,
            'model_weights': weights_history if record_weights else None
        } if record_history else None
    }


def create_and_train_model(
    n_inputs,
    n_outputs,
    p_e,
    learning_rate=0.1,
    batch_size=100,
    max_steps=1000,
    convergence_threshold=0.01,
    stagnation_window=-1,
    stagnation_threshold=1e-6,
    device='mps',
    seed=42,
    verbose=True,
    print_interval=100,
    record_history=False,
    record_weights=False,
    p_w = 0.5,
    use_gaussian_init=True,
    gaussian_mean= 0.5,
    gaussian_std= 0.25,
    match_oracle_weights = False,
):
    """
    Create, initialize, and train a model with the specified parameters.
    
    Args:
        n_inputs: Number of input features
        n_outputs: Number of parallel XOR units
        oracle_weights: Oracle weights tensor [n_inputs, n_outputs]
        inputs: Input data tensor [batch_size, n_inputs]
        learning_rate: Learning rate for optimizer
        max_steps: Maximum number of training steps
        convergence_threshold: Convergence threshold for mean p_diff
        p_e: Error probability for BSC channel (default: 0.0)
        stagnation_window: Window size for stagnation detection (default: -1 = disabled)
        stagnation_threshold: Minimum improvement per step (default: 1e-3)
        device: Device to use ('cpu', 'cuda', 'mps') (default: 'cpu')
        seed: Random seed for reproducibility (default: 42)
        verbose: Whether to print progress (default: True)
        print_interval: Print progress every N steps (default: 100)
        record_history: Whether to record full training history (default: False)
        record_weights: Whether to record weights during training (default: False)
        p_w: proportion of Oracle weights equal to 1 (default: 0.5),
        use_gaussian_init: Whether to use Gaussian initialization for model weights (default: True)
        gaussian_mean: Mean of Gaussian initialization (default: 0.5)
        gaussian_std: Standard deviation of Gaussian initialization (default: 0.25)
        match_oracle_weights: Whether to initialize model weights to match oracle weights with dispersion following gaussian init. If gaussian mean and std equal 0, then exact match (default: False)
    
    Returns:
        Dictionary containing:
        - training_results: Results from train_until_convergence
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create model
    default_model_kwargs = {
        'use_gaussian_init': use_gaussian_init,
        'gaussian_mean': gaussian_mean,
        'gaussian_std': gaussian_std
    }
    model = MultiBinaryProductModelWithOracle(
        n_outputs=n_outputs,
        p_e=p_e,
        **default_model_kwargs
    ).to(device)
    oracle_weights = (torch.rand((n_inputs, n_outputs), dtype=torch.float32, device=device) < p_w).float()
    inputs = torch.zeros(batch_size, n_inputs, dtype=torch.float32, device=device)

    if match_oracle_weights:
        # Initialize model weights to match oracle weights with dispersion following gaussian init
        # When oracle weight equal 0 we add positive noise, when equal 1 we add negative noise 
        # If gaussian mean = 0.5 and std equal 0.25 we have the default init (ie all weights around 0.5)
        # If gaussian mean and std equal 0, then exact match of oracle weights
        model_weights = oracle_weights.clone()
        if use_gaussian_init:
            noise = torch.normal(mean=gaussian_mean, std=gaussian_std, size=model_weights.shape, device=device)
            noise = (1 - 2 * oracle_weights) * torch.abs(noise)  # Positive noise for oracle 0, negative for oracle 1
            model_weights += noise
        model.set_model_parameters(model_weights)
    
    # Initialize model with inputs
    _ = model(inputs)
    model.set_oracle_parameters(oracle_weights)
    
    # Print memory usage (if MPS)
    if verbose and torch.backends.mps.is_available():
        mem_mb = torch.mps.current_allocated_memory() / 1024**2
        print(f"  MPS Memory: {mem_mb:.1f} MB")
    
    # Create optimizer   
    default_optimizer_kwargs = {
            'momentum': 0.0,
            'weight_decay': 0.0,
            'dampening': 0.0,
            'nesterov': False
        } 
    optimizer = torch.optim.SGD(
        model.product_model.parameters(),
        lr=learning_rate,
        **default_optimizer_kwargs
    )
    
    # Train model
    training_results = train_until_convergence(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        max_steps=max_steps,
        convergence_threshold=convergence_threshold,
        stagnation_window=stagnation_window,
        stagnation_threshold=stagnation_threshold,
        verbose=verbose,
        print_interval=print_interval,
        record_history=record_history,
        record_weights=record_weights,
    )
    
    # Clean up memory
    del model, optimizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'training_results': training_results,
    }

