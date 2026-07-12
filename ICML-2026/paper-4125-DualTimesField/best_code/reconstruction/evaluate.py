import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dualfield import DualTimesField
from .datasets import MultiDatasetLoader


DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'weather', 'illness', 'traffic']


def evaluate_reconstruction(model, test_loader, device):
    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    test_batches = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            output, ctf_out, dgf_out = model(x, t)
            test_mse += F.mse_loss(output, x).item()
            test_mae += F.l1_loss(output, x).item()
            test_batches += 1
            all_outputs.append(output.cpu().numpy())
            all_targets.append(x.cpu().numpy())
    
    test_mse /= max(test_batches, 1)
    test_mae /= max(test_batches, 1)
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return {
        'mse': test_mse,
        'mae': test_mae,
        'outputs': all_outputs,
        'targets': all_targets
    }


def visualize_reconstruction(outputs, targets, dataset_name, save_path):
    num_vars = min(4, outputs.shape[-1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} - Reconstruction', fontsize=14)
    
    sample_idx = 0
    for idx, ax in enumerate(axes.flat):
        if idx >= num_vars:
            ax.axis('off')
            continue
        
        target = targets[sample_idx, :, idx]
        output = outputs[sample_idx, :, idx]
        t = np.arange(len(target))
        
        ax.plot(t, target, 'b-', label='Ground Truth', linewidth=1.5)
        ax.plot(t, output, 'r--', label='Reconstruction', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Variable {idx}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dir = Path(__file__).parents[1] / 'data'
    save_dir = Path(__file__).parents[1] / 'outputs' / 'reconstruction'
    checkpoint_dir = save_dir / 'checkpoints'
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets if args.datasets else DATASETS
    results = []
    
    loader = MultiDatasetLoader(data_dir)
    
    for dataset_name in datasets:
        print(f'\n{"="*60}')
        print(f'Evaluating {dataset_name}')
        print(f'{"="*60}')
        
        checkpoint_path = checkpoint_dir / f'{dataset_name}_dualfield.pt'
        if not checkpoint_path.exists():
            print(f'  No checkpoint found for {dataset_name}')
            continue
        
        try:
            _, _, test_dataset = loader.get_dataset(
                dataset_name,
                seq_length=args.seq_length,
                stride=max(1, args.seq_length // 4),
                normalize=True
            )
        except Exception as e:
            print(f'  Failed to load {dataset_name}: {e}')
            continue
        
        num_variables = loader.get_num_variables(dataset_name)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = DualTimesField(
            num_variables=num_variables,
            seq_length=args.seq_length,
            num_frequencies=16,
            hidden_dim=64,
            num_layers=3,
            freq_cutoff=10.0,
            num_atoms=16,
            sigma_base=0.05,
            sparsity_lambda=0.001,
            smoothness_lambda=0.001
        ).to(device)
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'  Loaded checkpoint from {checkpoint_path}')
        
        result = evaluate_reconstruction(model, test_loader, device)
        active_atoms = model.dgf.get_active_atoms() if hasattr(model, 'dgf') else None

        results.append({
            'dataset': dataset_name,
            'mse': result['mse'],
            'mae': result['mae'],
            'active_atoms': active_atoms,
        })

        print(f'  MSE: {result["mse"]:.6f}, MAE: {result["mae"]:.6f}')
        if active_atoms is not None:
            print(f'  Active Atoms: {active_atoms}')
        
        viz_path = viz_dir / f'{dataset_name}_reconstruction.png'
        visualize_reconstruction(result['outputs'], result['targets'], dataset_name, viz_path)
        print(f'  Visualization saved to {viz_path}')
    
    if results:
        df = pd.DataFrame(results)
        print('\n' + '='*60)
        print('Summary')
        print('='*60)
        print(df.to_string(index=False))
        
        results_path = save_dir / 'evaluation_results.csv'
        df.to_csv(results_path, index=False)
        print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
