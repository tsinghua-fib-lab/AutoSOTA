import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dualfield import DualTimesField
from .datasets import MultiDatasetLoader


DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'weather', 'illness', 'traffic']


def train_single_dataset(
    dataset_name,
    data_dir,
    save_dir,
    seq_length=512,
    batch_size=32,
    epochs=500,
    lr=1e-3,
    num_frequencies=16,
    hidden_dim=64,
    num_atoms=16,
    sparsity_lambda=0.01,
    smoothness_lambda=0.001,
    device='cuda'
):
    loader = MultiDatasetLoader(data_dir)
    
    try:
        train_dataset, val_dataset, test_dataset = loader.get_dataset(
            dataset_name,
            seq_length=seq_length,
            stride=max(1, seq_length // 4),
            normalize=True
        )
    except Exception as e:
        print(f'Failed to load {dataset_name}: {e}')
        return None
    
    num_variables = loader.get_num_variables(dataset_name)
    
    actual_batch_size = min(batch_size, max(1, len(train_dataset)))
    if len(train_dataset) < 1:
        print(f'Not enough data for {dataset_name}')
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(actual_batch_size, len(val_dataset))), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=max(1, min(actual_batch_size, len(test_dataset))), shuffle=False)
    
    print(f'  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    model = DualTimesField(
        num_variables=num_variables,
        seq_length=seq_length,
        num_frequencies=num_frequencies,
        hidden_dim=hidden_dim,
        num_layers=3,
        freq_cutoff=10.0,
        num_atoms=num_atoms,
        sigma_base=0.05,
        sparsity_lambda=sparsity_lambda,
        smoothness_lambda=smoothness_lambda
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    init_batch = next(iter(train_loader))
    x_init, t_init = init_batch
    model.initialize_atoms(x_init.to(device), t_init.to(device))
    
    pbar = tqdm(range(epochs), desc=f'  {dataset_name}')
    for epoch in pbar:
        model.train()
        model.set_epoch(epoch)
        train_loss = 0.0
        num_batches = 0
        
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            losses = model.compute_loss(x, t)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += losses['reconstruction'].item()
            num_batches += 1
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(device), t.to(device)
                output, _, _ = model(x, t)
                val_loss += F.mse_loss(output, x).item()
                val_batches += 1
        
        if val_batches > 0:
            val_loss /= val_batches
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        pbar.set_postfix({'train': f'{train_loss/num_batches:.4f}', 'val': f'{val_loss:.4f}', 'best': f'{best_val_loss:.4f}'})
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_path = save_dir / 'checkpoints' / f'{dataset_name}_dualfield.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    test_batches = 0
    
    all_original = []
    all_recon = []
    all_ctf = []
    all_dgf = []
    
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            output, ctf_out, dgf_out = model(x, t)
            test_mse += F.mse_loss(output, x).item()
            test_mae += F.l1_loss(output, x).item()
            test_batches += 1
            all_original.append(x.cpu().numpy())
            all_recon.append(output.cpu().numpy())
            all_ctf.append(ctf_out.cpu().numpy())
            all_dgf.append(dgf_out.cpu().numpy())
    
    test_mse /= max(test_batches, 1)
    test_mae /= max(test_batches, 1)
    
    all_original = np.concatenate(all_original, axis=0)
    all_recon = np.concatenate(all_recon, axis=0)
    all_ctf = np.concatenate(all_ctf, axis=0)
    all_dgf = np.concatenate(all_dgf, axis=0)
    
    viz_dir = save_dir / 'visualizations' / dataset_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(5, len(all_original))
    num_vars = min(4, all_original.shape[2])
    
    fig, axes = plt.subplots(num_samples, num_vars, figsize=(4 * num_vars, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_vars == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        for j in range(num_vars):
            ax = axes[i, j]
            ax.plot(all_original[i, :, j], 'b-', alpha=0.7, linewidth=1, label='Original')
            ax.plot(all_recon[i, :, j], 'r--', alpha=0.7, linewidth=1, label='Reconstructed')
            if i == 0:
                ax.set_title(f'Variable {j+1}')
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}')
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(num_samples, num_vars, figsize=(4 * num_vars, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_vars == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        for j in range(num_vars):
            ax = axes[i, j]
            ax.plot(all_original[i, :, j], 'b-', alpha=0.5, linewidth=1, label='Original')
            ax.plot(all_ctf[i, :, j], 'g-', alpha=0.8, linewidth=1.5, label='CTF (Trend)')
            ax.plot(all_dgf[i, :, j], 'm-', alpha=0.8, linewidth=1.5, label='DGF (Detail)')
            if i == 0:
                ax.set_title(f'Variable {j+1}')
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}')
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'ctf_dgf_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  Visualizations saved to {viz_dir}')
    
    active_atoms = model.dgf.get_active_atoms() if hasattr(model, 'dgf') else None

    return {
        'dataset': dataset_name,
        'num_variables': num_variables,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'active_atoms': active_atoms,
        'model': model
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=336)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_atoms', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dir = Path(__file__).parents[1] / 'data'
    save_dir = Path(__file__).parents[1] / 'outputs' / 'reconstruction'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets if args.datasets else DATASETS
    results = []
    
    for dataset_name in datasets:
        print(f'\n{"="*60}')
        print(f'Training on {dataset_name}')
        print(f'{"="*60}')
        
        result = train_single_dataset(
            dataset_name, data_dir, save_dir, args.seq_length, args.batch_size,
            args.epochs, args.lr, 16, args.hidden_dim,
            args.num_atoms, 0.001, 0.001, device
        )
        
        if result is not None:
            results.append(result)
            print(f'\n  MSE: {result["test_mse"]:.6f}, MAE: {result["test_mae"]:.6f}')
            if result['active_atoms'] is not None:
                print(f'  Active Atoms: {result["active_atoms"]}')
    
    if results:
        df_data = [{k: v for k, v in r.items() if k != 'model'} for r in results]
        df = pd.DataFrame(df_data)
        print('\n' + '='*60)
        print('Summary')
        print('='*60)
        print(df.to_string(index=False))
        
        results_path = save_dir / 'reconstruction_results.csv'
        df.to_csv(results_path, index=False)
        print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
