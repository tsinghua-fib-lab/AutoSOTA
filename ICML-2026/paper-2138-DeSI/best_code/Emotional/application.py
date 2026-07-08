import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from nn_loc_composition import train_compositional_model

# Load the data
data = pd.read_csv('../data_emotion.csv')

# Extract response variables (b2_1 to b2_4)
response_cols = ['b2_1', 'b2_2', 'b2_3', 'b2_4']
responses = data[response_cols].values

# Normalize responses to sum to 1 (compositional constraint)
# The values appear to be percentages, so divide by 100
responses = responses / 100.0
# Ensure they sum to exactly 1 (handle any rounding issues)
responses = responses / responses.sum(axis=1, keepdims=True)

# Take square root of responses
y = np.sqrt(responses)

# Extract predictor variables (all other columns)
predictor_cols = [col for col in data.columns if col not in response_cols]
predictors = data[predictor_cols].values

# Display information about the data
print(f"Shape of y (square-root responses): {y.shape}")
print(f"Shape of predictors: {predictors.shape}")
print(f"\nResponse columns: {response_cols}")
print(f"Predictor columns: {predictor_cols}")

# Verify that sum of squares equals 1 (since original responses sum to 1)
y_squared_sums = (y ** 2).sum(axis=1)
print(f"\nSum of squares of y (should be close to 1): min={y_squared_sums.min():.6f}, max={y_squared_sums.max():.6f}, mean={y_squared_sums.mean():.6f}")

# Monte Carlo simulation parameters
n_monte_carlo = 100
n_folds = 10
n_samples = y.shape[0]
import os
n_cores = os.cpu_count() - 1  # Use all cores minus 1

print(f"\n{'='*60}")
print(f"Starting Monte Carlo Simulation: {n_monte_carlo} iterations")
print(f"Each iteration performs {n_folds}-fold Cross-Validation")
print(f"Using {n_cores} parallel cores (all cores - 1)")
print(f"{'='*60}")

def run_monte_carlo_iteration(mc_iter, predictors, y, n_samples, n_folds):
    """
    Run a single Monte Carlo iteration with 10-fold CV.
    
    Parameters
    ----------
    mc_iter : int
        Monte Carlo iteration number (used as seed)
    predictors : np.ndarray
        Predictor variables
    y : np.ndarray
        Response variables (square-root transformed)
    n_samples : int
        Number of samples
    n_folds : int
        Number of folds for CV
    
    Returns
    -------
    dict
        Dictionary with iteration, seed, and mean_mspe
    """
    seed = mc_iter  # Use iteration number as seed
    
    # Set random seed for this iteration
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device (use CPU for parallel processing to avoid GPU conflicts)
    device = torch.device('cpu')
    
    # Manual KFold implementation - separate all folds first
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    
    # Create all folds before the loop
    folds = []
    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples
        folds.append(indices[test_start:test_end])
    
    mspe_folds = []
    theta_folds = []  # Store theta estimates for each fold
    
    # 10-fold cross-validation
    for fold_idx in range(n_folds):
        # Use fold i as test set
        test_idx = folds[fold_idx]
        
        # All other folds combined as training set
        train_val_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        
        # Split training set into train (80%) and validation (20%)
        n_train_val = len(train_val_idx)
        n_train = int(0.8 * n_train_val)
        train_val_perm = np.random.RandomState(seed=seed + fold_idx).permutation(train_val_idx)
        train_idx = train_val_perm[:n_train]
        val_idx = train_val_perm[n_train:]
        
        # Extract data splits
        X_train = predictors[train_idx]
        X_val = predictors[val_idx]
        X_test = predictors[test_idx]
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        # Convert to torch tensors (as lists of tensors for Y)
        Y_train = [torch.tensor(y_train[i], dtype=torch.float32, device=device) for i in range(len(y_train))]
        Y_val = [torch.tensor(y_val[i], dtype=torch.float32, device=device) for i in range(len(y_val))]
        Y_test = [torch.tensor(y_test[i], dtype=torch.float32, device=device) for i in range(len(y_test))]
        
        # Train the model
        try:
            theta_pred, mspe = train_compositional_model(
                X_train, Y_train, X_val, Y_val, X_test, Y_test,
                batch_size=64,
                lr=0.5,
                num_epochs=1000,
                patience=5,
                delta=1e-4,
                lambda_reg=5,
                hidden_dim=64,
                dropout_rate=0.2,
                device=device
            )
            # Convert theta_pred to numpy array if it's a tensor
            # theta_pred is shape (n_test, p) - one theta per test sample
            # Take the mean across test samples to get the single index direction for this fold
            if torch.is_tensor(theta_pred):
                theta_pred_np = theta_pred.detach().cpu().numpy()
            else:
                theta_pred_np = np.array(theta_pred)
            # Average across test samples to get single index direction
            theta_mean = np.mean(theta_pred_np, axis=0)
            mspe_folds.append(mspe)
            theta_folds.append(theta_mean)
        except Exception as e:
            print(f"  Error in iteration {mc_iter}, fold {fold_idx}: {e}")
            # Use a large MSPE value if training fails
            mspe_folds.append(10.0)
            # Use zeros as placeholder for theta if training fails
            theta_folds.append(np.zeros(predictors.shape[1]))
    
    # Calculate mean MSPE for this Monte Carlo iteration
    mean_mspe = np.mean(mspe_folds)
    
    print(f"  Completed iteration {mc_iter}/{n_monte_carlo} (seed: {seed}) - Mean MSPE: {mean_mspe:.6f}")
    
    return {
        'iteration': mc_iter,
        'seed': seed,
        'mean_mspe': mean_mspe,
        'theta_folds': theta_folds  # List of theta arrays for each fold
    }

# Run Monte Carlo iterations in parallel
print(f"\nRunning {n_monte_carlo} iterations in parallel using {n_cores} cores...")
monte_carlo_results = Parallel(n_jobs=n_cores, verbose=10)(
    delayed(run_monte_carlo_iteration)(mc_iter, predictors, y, n_samples, n_folds)
    for mc_iter in range(1, n_monte_carlo + 1)
)

# Extract theta_folds and save them to CSV, then create a summary DataFrame
summary_results = []
theta_results = []  # Store all theta estimates for CSV

for result in monte_carlo_results:
    mc_iter = result['iteration']
    seed = result['seed']
    mean_mspe = result['mean_mspe']
    theta_folds = result['theta_folds']
    
    # Save theta for each fold to the theta_results list
    for fold_idx, theta in enumerate(theta_folds, 1):
        theta_dict = {
            'iteration': mc_iter,
            'seed': seed,
            'fold': fold_idx
        }
        # Add each dimension of theta as a separate column
        for dim_idx, theta_val in enumerate(theta):
            theta_dict[f'theta_{dim_idx}'] = theta_val
        theta_results.append(theta_dict)
    
    # Create summary entry
    summary_results.append({
        'iteration': mc_iter,
        'seed': seed,
        'mean_mspe': mean_mspe
    })

# Save summary results to CSV
results_df = pd.DataFrame(summary_results)
output_file = 'monte_carlo_mspe_results.csv'
results_df.to_csv(output_file, index=False)

# Save theta estimates to CSV
theta_df = pd.DataFrame(theta_results)
theta_output_file = 'monte_carlo_theta_results.csv'
theta_df.to_csv(theta_output_file, index=False)

print(f"\n{'='*60}")
print(f"Monte Carlo Simulation Complete!")
print(f"{'='*60}")
print(f"\nResults saved to: {output_file}")
print(f"Theta estimates saved to: {theta_output_file}")
print(f"\nOverall Statistics:")
print(f"  Mean MSPE across all iterations: {results_df['mean_mspe'].mean():.6f}")
print(f"  Std MSPE across all iterations: {results_df['mean_mspe'].std():.6f}")
print(f"  Min MSPE: {results_df['mean_mspe'].min():.6f}")
print(f"  Max MSPE: {results_df['mean_mspe'].max():.6f}")

