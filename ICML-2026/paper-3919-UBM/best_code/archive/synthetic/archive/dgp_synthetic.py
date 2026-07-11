# -*- coding: utf-8 -*-

"""
Description: This script [brief description of what the script does]
Author: ZMH
Date: 10-07-2024
Version: 1.0

"""
from biases_synthetic import apply_bias_wrapper
from estimator_synthetic import rs_estimator_cate

import logging
import sys
import numpy as np
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to generate RCT data with different outcome surfaces
def generate_rct_data(X, outcome_surface='linear', beta_X=0.5, beta_A=0.7, sigma=1.0):
    """
    Generates RCT data with specified outcome surface.
    
    Parameters:
    - X: array-like, covariate
    - outcome_surface: str, 'linear' or 'cubic'
    - beta_X: float, coefficient for X
    - beta_A: float, coefficient for A
    - sigma: float, standard deviation of noise
    
    Returns:
    - DataFrame with columns ['X', 'A', 'Y']
    """
    # 2. Randomly Assign Intervention A
    A = np.random.binomial(1, 0.5, len(X))  # 50% probability for treatment

    # 3. Generate Outcome Y based on the specified outcome surface
    if outcome_surface == 'linear':
        Y = beta_X * X + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'cubic':
        Y = beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    else:
        raise ValueError("Unsupported outcome surface for RCT. Choose 'linear' or 'cubic'.")

    # Combine into DataFrame
    data_rct = pd.DataFrame({
        'X': X,
        'A': A,
        'Y': Y
    })

    return data_rct

# Function to generate Observational Study data with different outcome surfaces
def generate_observational_data(X, outcome_surface='linear', beta_X=0.5, beta_A=0.7, sigma=1.0):
    """
    Generates Observational Study data with specified outcome surface.
    
    Parameters:
    - X: array-like, covariate
    - outcome_surface: str, 'linear', 'cubic', 'jump', or 'negative'
    - beta_X: float, coefficient for X or X^3
    - beta_A: float, coefficient for A
    - sigma: float, standard deviation of noise
    
    Returns:
    - DataFrame with columns ['X', 'A', 'Y']
    """
    # 2. Assign Intervention A based on X (Propensity Score)
    # Define propensity score using logistic function
    # For example, higher X increases likelihood of receiving treatment
    linear_propensity = 2.5* X  # Modify as needed for different scenarios
    propensity = 1 / (1 + np.exp(-linear_propensity))
    
    # Assign A based on propensity score
    A = np.random.binomial(1, propensity, len(X))
    
    # 3. Generate Outcome Y based on the specified outcome surface
    if outcome_surface == 'linear':
        Y = beta_X * X + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'cubic':
        Y = beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'jump':
        Y = beta_X * (X ** 3) + 5 * ((X >= 0) & (X <= 1)).astype(int) + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'negative':
        Y = -beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    else:
        raise ValueError("Unsupported outcome surface for OBS. Choose 'linear', 'cubic', 'jump', or 'negative'.")

    # Combine into DataFrame
    data_obs = pd.DataFrame({
        'X': X,
        'A': A,
        'Y': Y
    })

    return data_obs

# main function
def gen_synthetic_data(data_params):
    """
    Generate synthetic data for testing and benchmarking.
    """
    logging.info("Generating synthetic RCT and OBS")

    # Set seed for reproducibility
    np.random.seed(42) 

    # Generate covariate X
    X = np.random.uniform(low=-2, high=2, size=data_params["n"])
    # Generate RCT data  
    data_rct = generate_rct_data(X, 
                    outcome_surface=data_params["rct_outcome_surface"], 
                    beta_X=data_params["beta_X"], 
                    beta_A=data_params["beta_A"], 
                    sigma=data_params["sigma"])
    logging.info("RCT generated successfully") 
    # Generate OBS data 
    data_obs = generate_observational_data(X, 
                    outcome_surface=data_params["obs_outcome_surface"], 
                    beta_X=data_params["beta_X"], 
                    beta_A=data_params["beta_A"], 
                    sigma=data_params["sigma"])
    logging.info("OBS generated successfully") 
    
    return data_rct, data_obs
    
if __name__ == "__main__":
    logging.info("Starting the script") 
    # Set data parameters
    data_params = { 
        "n": 10000,
        "rct_outcome_surface": "cubic",
        "obs_outcome_surface": "cubic",
        "beta_X": 0.5,
        "beta_A": 10.0,
        "sigma": 1.0,         
        "bias_dict": { 
            "type": "selection_bias",
            "params": { 
                "selection_dependence": "berksons",
                "beta_S_A": 10.,
                "beta_S_Y": 10.
            }
        }
    }
    # Generate data
    try:
        data_rct, data_obs = gen_synthetic_data(data_params)
        # apply bias 
        data_obs = apply_bias_wrapper(data_obs, data_params["bias_dict"])
        # Save data in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_rct.to_csv(os.path.join(current_dir, "data/data_rct_biased.csv"), index=False)
        data_obs.to_csv(os.path.join(current_dir, "data/data_obs_biased.csv"), index=False)
        logging.info("Data saved successfully in the current directory")
        # Calculate CATE over values of X 
        cate_obs = rs_estimator_cate(data_obs)
        # save cate_obs 
        pd.DataFrame(np.concatenate([data_obs["X"].values[:, np.newaxis], cate_obs[:, np.newaxis]], axis=1), 
                     columns=["X", "CATE"]).to_csv(os.path.join(current_dir, "data/cate_obs_biased.csv"), index=False)
        logging.info("CATE saved successfully in the current directory")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
