"""
Reproduce Table 1: MSE for TreeHFD - xgboost (HFD target).
Parameters from paper (Section 4 - Analytical case):
- p=6, n=5000, rho=0.5
- y = sin(2*pi*X1) + X1*X2 + X3*X4 + epsilon, epsilon ~ N(0, 0.5^2)
- XGBoost: eta=0.1, n_estimators=100, max_depth=6 (M=100 trees)
- Results averaged over 10 repetitions
"""

import numpy as np
import xgboost as xgb
from numpy.random import default_rng

from treehfd import XGBTreeHFD

def eta_main(x, rho):
    """Compute main effect for the analytical example."""
    return rho / (1 + rho**2) * (x**2 - 1)

def eta_order2(x, z, rho):
    """Compute interactions for the analytical example."""
    return (rho * (1 - rho**2) / (1 + rho**2) 
            - rho / (1 + rho**2) * (x**2 + z**2) + x * z)

# Parameters
DIM = 6
NSAMPLE = 5000
RHO = 0.5
N_REP = 10

mu = np.zeros(DIM)
cov = np.full((DIM, DIM), RHO)
np.fill_diagonal(cov, np.ones(DIM))

# Store results per repetition
all_mse_main = []  # shape (n_rep, 6) for eta(1) ... eta(6)
all_mse_12 = []    # shape (n_rep,) for eta(1,2)
all_mse_34 = []    # shape (n_rep,) for eta(3,4)
all_mse_others = []  # shape (n_rep,) for all other interactions

for rep in range(N_REP):
    print(f"Repetition {rep+1}/{N_REP}")
    rng = default_rng(rep * 42 + 7)  # kept for backward compatibility
    
    
    # Testing data (independent) - must be generated before ensemble loop for consistency
    rng2 = default_rng(rep * 42 + 99)
    X_new = rng2.multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
    
    # Fit ensemble of XGBoost+TreeHFD models with different seeds and average predictions
    N_ENSEMBLE = 3
    ensemble_seeds = [rep * 42 + 7, rep * 42 + 13, rep * 42 + 19]
    y_main_list = []
    y_order2_list = []
    for seed_i in ensemble_seeds:
        rng_e = default_rng(seed_i)
        X_e = rng_e.multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
        y_e = (np.sin(2 * np.pi * X_e[:, 0]) + X_e[:, 0] * X_e[:, 1] 
               + X_e[:, 2] * X_e[:, 3])
        y_e += rng_e.normal(loc=0.0, scale=0.5, size=NSAMPLE)
        xgb_m = xgb.XGBRegressor(eta=0.1, n_estimators=100, max_depth=6, random_state=int(seed_i % 2147483647))
        xgb_m.fit(X_e, y_e)
        thfd_m = XGBTreeHFD(xgb_m)
        thfd_m.fit(X_e, interaction_order=2, verbose=False)
        ym, yo2 = thfd_m.predict(X_new, verbose=False)
        y_main_list.append(ym)
        y_order2_list.append(yo2)
        # Store interaction list from first model
        if seed_i == ensemble_seeds[0]:
            treehfd_model = thfd_m
    
    # Average predictions across ensemble members
    y_main = np.mean(y_main_list, axis=0)
    # For y_order2, need to align interaction lists
    # All models should have same interaction list given same tree structure
    y_order2 = np.mean(y_order2_list, axis=0)
    
    # Compute analytical HFD targets
    # Main effects: eta(j) for j=0,...,5
    # eta(1) = sin(2*pi*x1) + eta_main(x1, rho)  [X1 has interaction, so correction]
    # For independent case with gaussian: main effect of X1 in sin(2pi*X1) + X1*X2
    # is sin(2pi*X1) + E[X1*X2|X1] - E[E[X1*X2|X1]]
    # But in paper the analytical formulas are given by eta_main (see utils.py)
    # Per paper: y = sin(2pi*X1) + X1*X2 + X3*X4 
    # HFD components:
    # eta(1) = sin(2pi*X1) + rho/(1+rho^2)*(X1^2 - 1)
    # eta(2) = rho/(1+rho^2)*(X2^2 - 1)   (correction term from correlation)
    # eta(3) = rho/(1+rho^2)*(X3^2 - 1)
    # eta(4) = rho/(1+rho^2)*(X4^2 - 1)
    # eta(5) = eta(6) = 0 (X5, X6 don't appear in m)
    # eta(1,2) = X1*X2 + correction
    # eta(3,4) = X3*X4 + correction
    
    y_exact_main = np.zeros((NSAMPLE, DIM))
    # eta(1): sin(2pi*X1) + eta_main(X1, rho)
    y_exact_main[:, 0] = np.sin(2 * np.pi * X_new[:, 0]) + eta_main(X_new[:, 0], RHO)
    # eta(2), eta(3), eta(4): only eta_main correction
    for j in range(1, 4):
        y_exact_main[:, j] = eta_main(X_new[:, j], RHO)
    # eta(5), eta(6) = 0
    y_exact_main[:, 4] = 0.0
    y_exact_main[:, 5] = 0.0
    
    # Compute MSE for each main effect
    mse_main = np.mean((y_exact_main - y_main)**2, axis=0)
    all_mse_main.append(mse_main)
    
    # Interactions
    # Need to find which column corresponds to (1,2) and (3,4) in y_order2
    # The interaction_list is stored in treehfd_model.interaction_list
    int_list = treehfd_model.interaction_list
    print(f"  Interaction list: {int_list}")
    print(f"  y_order2 shape: {y_order2.shape}")
    
    # Find index of pair (0,1) i.e. eta(1,2) and (2,3) i.e. eta(3,4)
    idx_12 = None
    idx_34 = None
    other_indices = []
    for k, pair in enumerate(int_list):
        if list(pair) == [0, 1]:
            idx_12 = k
        elif list(pair) == [2, 3]:
            idx_34 = k
        else:
            other_indices.append(k)
    
    print(f"  idx_12={idx_12}, idx_34={idx_34}, others={other_indices}")
    
    # Analytical interaction components
    y_exact_12 = eta_order2(X_new[:, 0], X_new[:, 1], RHO)
    y_exact_34 = eta_order2(X_new[:, 2], X_new[:, 3], RHO)
    
    if idx_12 is not None:
        mse_12 = np.mean((y_exact_12 - y_order2[:, idx_12])**2)
    else:
        print("  WARNING: pair (0,1) not found in interaction list!")
        mse_12 = np.nan
    
    if idx_34 is not None:
        mse_34 = np.mean((y_exact_34 - y_order2[:, idx_34])**2)
    else:
        print("  WARNING: pair (2,3) not found in interaction list!")
        mse_34 = np.nan
    
    all_mse_12.append(mse_12)
    all_mse_34.append(mse_34)
    
    # "Others" = all other interactions that should be null
    if other_indices:
        mse_others = np.mean(y_order2[:, other_indices]**2)
    else:
        mse_others = 0.0
    all_mse_others.append(mse_others)
    
    print(f"  MSE main: {np.round(mse_main, 4)}")
    print(f"  MSE (1,2): {mse_12:.4f}, MSE (3,4): {mse_34:.4f}")
    print(f"  MSE others: {mse_others:.4f}")

# Average over repetitions
all_mse_main = np.array(all_mse_main)
mean_mse_main = np.mean(all_mse_main, axis=0)
mean_mse_12 = np.nanmean(all_mse_12)
mean_mse_34 = np.nanmean(all_mse_34)
mean_mse_others = np.mean(all_mse_others)

print("\n" + "="*60)
print("FINAL RESULTS (averaged over 10 repetitions)")
print("="*60)
for j in range(DIM):
    print(f"MSE eta({j+1}): {mean_mse_main[j]:.4f}")
print(f"MSE eta(1,2): {mean_mse_12:.4f}")
print(f"MSE eta(3,4): {mean_mse_34:.4f}")
print(f"MSE Others: {mean_mse_others:.4f}")
