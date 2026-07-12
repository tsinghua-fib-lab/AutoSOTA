"""
knob_sensitivity.py

This script runs the experiment with different knob settings and plots the results.

The knobs are:
- aug_data_size_factor
- max_n_features_to_perturb
- max_perturb_percent
- aug_data_weight

The script runs the experiment with the default knob settings and then varies each knob one by one while keeping the other knobs at the default values.

The results are saved in the ./scripts/figures/ directory.
"""

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import matplotlib.pyplot as plt
from src.utils.config import Config
from src.utils.logger import Logger
from src.experiment import Experiment
import pandas as pd

config_dict = {
    "baseline": "mlp",
    "dataset_path": os.path.join(ROOT, "data", "HousePrice.csv"),
    "results_dir": os.path.join(ROOT, "scripts", "HousePrice"),
    "aug_data_size_factor": 1.25,
    "max_n_features_to_perturb": 2,
    "max_perturb_percent": 0.7,
    "min_perturb_percent": -0.7,
    "aug_data_weight": 0.75,
    "num_seeds": 5
}


def run(config):
    try:
        results_df = Experiment(config, Logger(log_to_console=False)).run()
    except Exception as e:
        # check if exception is about previous experiment
        if "Existing experiment results found at:" in str(e):
            # Extract the path from the error message
            path = str(e).split(": ")[1]
            # Load and return the results from the previous experiment
            return pd.read_csv(os.path.join(path, "results.csv"))
        else:
            raise e
    return results_df

def plot_results(results, name, model):
    plt.figure(figsize=(10, 6))
    plt.plot(results['variation'], results['mean_delta_mse'], '-o', linewidth=2)
    plt.fill_between(results['variation'], 
                     results['mean_delta_mse'] - results['std_delta_mse'],
                     results['mean_delta_mse'] + results['std_delta_mse'], 
                     alpha=0.2)
    plt.xlabel(name)
    plt.ylabel('MSE % Change')
    out_dir = os.path.join(ROOT, "scripts", "figures", model)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}_sensitivity.png"))


aug_data_size_factor_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
max_n_features_to_perturb_list = [1, 2, 3, 4, 5]
max_perturb_percent_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_perturb_percent_list = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
aug_data_weight_list = [0.2, 0.4, 0.6, 0.8, 1.0]



def run_for_mlp():
    # vary the aug_data_size_factor
    config_dict_copy = config_dict.copy()
    aug_data_size_factor_results = []
    for aug_data_size_factor in aug_data_size_factor_list:
        config_dict_copy["aug_data_size_factor"] = aug_data_size_factor
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"aug_data_size_factor: {aug_data_size_factor}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        aug_data_size_factor_results.append({
            "variation": aug_data_size_factor,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(aug_data_size_factor_results), "aug_data_size_factor", model="mlp")

    # vary the max_n_features_to_perturb
    config_dict_copy = config_dict.copy()
    max_n_features_to_perturb_results = []
    for max_n_features_to_perturb in max_n_features_to_perturb_list:
        config_dict_copy["max_n_features_to_perturb"] = max_n_features_to_perturb
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"max_n_features_to_perturb: {max_n_features_to_perturb}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        max_n_features_to_perturb_results.append({
            "variation": max_n_features_to_perturb,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(max_n_features_to_perturb_results), "max_n_features_to_perturb", model="mlp")

    # vary the max_perturb_percent
    config_dict_copy = config_dict.copy()
    max_perturb_percent_results = []
    for max_perturb_percent in max_perturb_percent_list:
        config_dict_copy["max_perturb_percent"] = max_perturb_percent
        config_dict_copy["min_perturb_percent"] = -max_perturb_percent
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"max_perturb_percent: {max_perturb_percent}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        max_perturb_percent_results.append({
            "variation": max_perturb_percent,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(max_perturb_percent_results), "min_max_perturb_percent", model="mlp")

    # vary the aug_data_weight
    config_dict_copy = config_dict.copy()
    aug_data_weight_results = []
    for aug_data_weight in aug_data_weight_list:
        config_dict_copy["aug_data_weight"] = aug_data_weight
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"aug_data_weight: {aug_data_weight}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        aug_data_weight_results.append({
            "variation": aug_data_weight,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(aug_data_weight_results), "aug_data_weight", model="mlp")



def run_for_xgb():
    config_dict["baseline"] = "xgboost"

    # vary the aug_data_size_factor
    config_dict_copy = config_dict.copy()
    aug_data_size_factor_results = []
    for aug_data_size_factor in aug_data_size_factor_list:
        config_dict_copy["aug_data_size_factor"] = aug_data_size_factor
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"aug_data_size_factor: {aug_data_size_factor}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        aug_data_size_factor_results.append({
            "variation": aug_data_size_factor,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(aug_data_size_factor_results), "aug_data_size_factor", model="xgb")

    # vary the max_n_features_to_perturb
    config_dict_copy = config_dict.copy()
    max_n_features_to_perturb_results = []
    for max_n_features_to_perturb in max_n_features_to_perturb_list:
        config_dict_copy["max_n_features_to_perturb"] = max_n_features_to_perturb
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"max_n_features_to_perturb: {max_n_features_to_perturb}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        max_n_features_to_perturb_results.append({
            "variation": max_n_features_to_perturb,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(max_n_features_to_perturb_results), "max_n_features_to_perturb", model="xgb")

    # vary the max_perturb_percent
    config_dict_copy = config_dict.copy()
    max_perturb_percent_results = []
    for max_perturb_percent in max_perturb_percent_list:
        config_dict_copy["max_perturb_percent"] = max_perturb_percent
        config_dict_copy["min_perturb_percent"] = -max_perturb_percent
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"max_perturb_percent: {max_perturb_percent}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        max_perturb_percent_results.append({
            "variation": max_perturb_percent,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(max_perturb_percent_results), "min_max_perturb_percent", model="xgb")

    # vary the aug_data_weight
    config_dict_copy = config_dict.copy()
    aug_data_weight_results = []
    for aug_data_weight in aug_data_weight_list:
        config_dict_copy["aug_data_weight"] = aug_data_weight
        config = Config(**config_dict_copy)
        results_df = run(config)
        # get the mean and std of the delta_mse metric
        delta_mse_rows = results_df[results_df['metric'] == 'delta_mse']
        mean_delta_mse = delta_mse_rows['mean'].values[0]
        std_delta_mse = delta_mse_rows['std'].values[0]
        print(f"aug_data_weight: {aug_data_weight}, mean_delta_mse: {mean_delta_mse:.4f}, std_delta_mse: {std_delta_mse:.4f}")
        aug_data_weight_results.append({
            "variation": aug_data_weight,
            "mean_delta_mse": mean_delta_mse,
            "std_delta_mse": std_delta_mse
        })

    plot_results(pd.DataFrame(aug_data_weight_results), "aug_data_weight", model="xgb")


if __name__ == "__main__":
    run_for_mlp()
    run_for_xgb()
