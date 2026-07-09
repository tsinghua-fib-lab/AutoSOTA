from copy import deepcopy
import numpy as np
# import pandas as pd
# from copy import deepcopy

# def run_multiple_tests(data_generator, model, conformal_predictor, warmup=50, n_seeds=250, seed=None):
#     """
#     Run multiple random experiments and compute average behavior.
    
#     Args:
#         data_generator: Callable that returns (X, Y) data tuple. Called fresh each trial.
#         model: A regression model instance.
#         conformal_predictor: A ConformalPredictor instance.
#         n_seeds: Number of independent trials to run.
#         warmup: Number of initial points to skip for evaluation.
#         seed: Random seed for reproducibility. If None, no seed is set.
        
#     Returns:
#         results: List of individual trial results
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     results = []
    
#     for _ in range(n_seeds):
#         # Generate fresh data, model, and predictor for each trial
#         data = data_generator()
                
#         # Reinitialize model and predictor in case they have memory effects
#         fresh_model = deepcopy(model)
#         fresh_conformal_predictor = deepcopy(conformal_predictor)
        
#         # Run single experiment
#         result = run_experiment(data, fresh_model, fresh_conformal_predictor, warmup=warmup)
#         results.append(result)
    
#     return results


# def get_statistics(results, warmup=50, window=100):
#     """
#     Compute mean statistics across multiple experiment results.
    
#     Args:
#         results: List of individual trial results (dictionaries).
#         warmup: Number of initial points to skip for evaluation.
#         window: Size of the moving window.
#     Returns:
#         Dictionary with:
#             - mean_* : Average of each metric across trials
#             - std_* : Standard deviation of each metric across trials
#             - cumulative_* : Cumulative statistics across trials
#             - local_* : Local statistics across trials
#     """
#     # Stack results for vectorized computation
#     all_predicted_radii = np.array([r["predicted_radii"] for r in results])
#     all_observed_radii = np.array([r["observed_radii"] for r in results])
#     all_coverages = np.array([r["coverages"] for r in results])
#     all_errors = np.array([r["errors"] for r in results])
    
#     mean_predicted_radii = np.mean(all_predicted_radii[:, warmup:], axis=0)
#     std_predicted_radii = np.std(all_predicted_radii[:, warmup:], axis=0)
#     mean_local_radii = pd.Series(mean_predicted_radii).rolling(window=window, min_periods=1).mean().to_numpy()
#     mean_local_radii_deviations = pd.Series(mean_predicted_radii).rolling(window=window, min_periods=1).std().to_numpy()
    
#     mean_observed_radii = np.mean(all_observed_radii[:, warmup:], axis=0)
#     std_observed_radii = np.std(all_observed_radii[:, warmup:], axis=0)
    
#     mean_coverages = np.mean(all_coverages[:, warmup:], axis=0)
#     std_coverages = np.std(all_coverages[:, warmup:], axis=0)
#     mean_cumulative_coverages = np.cumsum(mean_coverages) / np.arange(1, len(mean_coverages) + 1)
#     mean_local_coverages = pd.Series(mean_coverages).rolling(window=window, min_periods=1).mean().to_numpy()
    
#     mean_errors = np.mean(all_errors[:, warmup:], axis=0)
#     std_errors = np.std(all_errors[:, warmup:], axis=0)
    
#     return {
#         "mean_predicted_radii": mean_predicted_radii,
#         "std_predicted_radii": std_predicted_radii,
#         "mean_local_radii": mean_local_radii,
#         "mean_local_radii_deviations": mean_local_radii_deviations,
#         "mean_observed_radii": mean_observed_radii,
#         "std_observed_radii": std_observed_radii,
        
#         "mean_coverages": mean_coverages,
#         "std_coverages": std_coverages,
#         "mean_cumulative_coverages": mean_cumulative_coverages,
#         "mean_local_coverages": mean_local_coverages,
        
#         "mean_errors": mean_errors,
#         "std_errors": std_errors,
#     }
    
def run_conformal_inference(scores, conformal_predictor, T_burnin=100):
    """
    Run a single conformal inference experiment.
        
    Args:
        data: Tuple of (X, Y) data
        model: Regression model with fit and predict methods
        conformal_predictor: ConformalPredictor instance
        warmup: Number of initial points to skip for evaluation
    Returns:
        Inference result with all metrics
    """
    n_steps = len(scores)

    # Initialize storage for metrics
    predicted_scores = np.zeros(n_steps)
    coverages = np.zeros(n_steps)

    # Warmup initialization: set initial s to empirical quantile of warmup scores
    # to avoid cold-start overshoot, and pre-populate internal state for faster convergence
    if hasattr(conformal_predictor, 'warmup_init') and conformal_predictor.warmup_init:
        warmup_scores = scores[:T_burnin]
        warmup_quantile = np.quantile(warmup_scores, 1.0 - conformal_predictor.alpha)
        conformal_predictor.s = warmup_quantile
        # Pre-populate covered_count and t to warmup-consistent values
        # so x starts near its asymptotic value instead of 0.5
        warmup_coverage_rate = np.mean(warmup_scores <= warmup_quantile)
        conformal_predictor.covered_count = int(round(warmup_coverage_rate * T_burnin))
        conformal_predictor.t = T_burnin
        # Also initialize effective counts for decay mode
        if hasattr(conformal_predictor, 'eff_t'):
            conformal_predictor.eff_t = float(T_burnin)
            conformal_predictor.eff_covered = float(warmup_coverage_rate * T_burnin)

    # Simple online regression using all past data
    for t in range(n_steps):
        # Get current radius from predictor
        predicted_scores[t] = conformal_predictor.predict()

        if predicted_scores[t] >= scores[t]:
            coverages[t] = 1

        if t >= T_burnin:
            # Update predictor with the newly observed score
            conformal_predictor.update(scores[t])
        
    return {"predicted_scores": predicted_scores,
            "coverages": coverages}
    
def run_conformal_inference_on_ensemble(scores_ensemble, conformal_predictor, T_burnin):
    results = []
    for scores in scores_ensemble:
        fresh_conformal_predictor = deepcopy(conformal_predictor)

        result = run_conformal_inference(scores["scores"], fresh_conformal_predictor, T_burnin)
        results.append(result)
        
    return results
    
   