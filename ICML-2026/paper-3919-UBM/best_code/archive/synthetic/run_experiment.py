import argparse
import numpy as np
from bias_generation import BaseUnbiasedSetup, SelectionBiasType1or2, SelectionBiasType3
from estimator import CovarianceEstimator, PsiEstimator

def run_single_experiment(bias_setup, seed, n_rct, n_obs):
    # Set up the experiment
    setup = bias_setup(n_rct=n_rct, n_obs=n_obs, random_seed=seed)
    
    # Generate data
    df_rct, df_obs = setup.generate_data()
    
    # Generate treatment, outcome, and selection
    setup.generate_treatment_outcome_selection(df_rct, study="RCT")
    setup.generate_treatment_outcome_selection(df_obs, study="OBS")
    
    # Fit models and get merged dataset
    df_merged = setup.fit_models(df_rct, df_obs)
    
    # Calculate contrasts
    psi_estimator = PsiEstimator()
    psi_estimator.calc_contrasts(df_merged)
    
    df_merged.loc[df_merged["X1"] == 1, "w(X)"] = df_merged.query("X1 == 1")["psi"].mean()
    df_merged.loc[df_merged["X1"] == -1, "w(X)"] = df_merged.query("X1 == -1")["psi"].mean()
    df_obs = df_merged.query("R==0")
    
    # Compute covariances for different X1 values
    estimator = CovarianceEstimator()
    results = {}
    results["X1_all"] = estimator.compute_covariance_SE(df_obs, use_witness_function=True)
    for x1_val in [-1, 1]:
        df_x1 = df_obs.query(f"X1 == {x1_val}")
        results[f"X1_{x1_val}"] = estimator.compute_covariance_SE(df_x1, use_witness_function=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run bias experiments')
    parser.add_argument('--bias_type', type=str, choices=['unbiased', 'selection-type1or2', 'selection-type3'],
                      required=True, help='Type of bias setup to use')
    parser.add_argument('--n_runs', type=int, default=10,
                      help='Number of experimental runs')
    parser.add_argument('--n_rct', type=int, default=1000,
                      help='RCT sample size')
    parser.add_argument('--n_obs', type=int, default=10000,
                      help='OBS sample size')
    parser.add_argument('--seed', type=int, default=42,
                      help='Base random seed')
    args = parser.parse_args()
    
    # Select bias setup
    bias_setups = {
        'unbiased': BaseUnbiasedSetup,
        'selection-type1or2': SelectionBiasType1or2,
        'selection-type3': SelectionBiasType3
    }
    bias_setup = bias_setups[args.bias_type]
    
    # Run experiments
    all_results = []
    for i in range(args.n_runs):
        seed = args.seed + i
        results = run_single_experiment(bias_setup, seed, args.n_rct, args.n_obs)
        all_results.append(results)
        print(f"Completed run {i+1}/{args.n_runs}")
        
    # Print summary statistics
    print("\nSummary Statistics:")
    for x1_val in [-1, 1, "all"]:
        print(f"\nResults for X1 = {x1_val}:")
        for signal in ["outcome", "treatment", "selection"]:
            values = [r[f"X1_{x1_val}"][signal] for r in all_results]
            mean = np.mean(values)
            std = np.std(values)
            ci = 1.96 * std / np.sqrt(len(values))  # 95% confidence interval
            print(f"{signal.capitalize()} covariance: {mean:.4f} ± {ci:.4f}")

if __name__ == "__main__":
    main()