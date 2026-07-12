import torch
import numpy as np
import pandas as pd


def compute_mse(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    return np.mean((pred - target) ** 2)


def compute_rmse(pred, target):
    return np.sqrt(compute_mse(pred, target))


def compute_mae(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    return np.mean(np.abs(pred - target))


def load_baseline_results():
    baseline_results = {
        'PhysioNet': {
            'RNN': 2.92,
            'RNN-VAE': 5.93,
            'ODE-RNN': 2.23,
            'GRU-D': 3.33,
            'Latent ODE': 8.34,
            'LS4': 0.63,
            'iHT': 3.35
        },
        'USHCN': {
            'RNN': 4.32,
            'RNN-VAE': 7.56,
            'ODE-RNN': 2.47,
            'GRU-D': 3.40,
            'Latent ODE': 6.86,
            'LS4': 0.06,
            'iHT': 1.46
        }
    }
    
    return baseline_results


def format_results_table(results_dict, dataset_name):
    baseline_results = load_baseline_results()[dataset_name]
    
    all_results = {**baseline_results, **results_dict}
    
    df = pd.DataFrame(list(all_results.items()), columns=['Method', 'MSE (×10^-3)'])
    df = df.sort_values('MSE (×10^-3)')
    
    return df


def save_results(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def print_comparison_table(physionet_results, ushcn_results):
    print("\n" + "="*80)
    print("Interpolation Results Comparison")
    print("="*80)
    
    print("\nPhysioNet Dataset:")
    print("-"*80)
    physionet_table = format_results_table(physionet_results, 'PhysioNet')
    print(physionet_table.to_string(index=False))
    
    print("\n\nUSHCN Dataset:")
    print("-"*80)
    ushcn_table = format_results_table(ushcn_results, 'USHCN')
    print(ushcn_table.to_string(index=False))
    
    print("\n" + "="*80)
