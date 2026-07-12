import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.dualfield import DualTimesField
from .datasets import MultiDatasetLoader
from .baselines import SIREN, WIRE, NBeatsBaseline, TimesNetBaseline, PatchTSTBaseline, iTransformerBaseline, PCABaseline, FourierBaseline


def get_model(model_name, num_variables, seq_length, hidden_dim=64):
    if model_name == 'DualTimesField':
        return DualTimesField(
            num_variables=num_variables,
            seq_length=seq_length,
            num_frequencies=16,
            hidden_dim=hidden_dim,
            num_layers=3,
            freq_cutoff=8.0,
            num_atoms=16,
            sigma_base=0.05,
            sparsity_lambda=0.001,
            smoothness_lambda=0.001
        )
    elif model_name == 'SIREN':
        return SIREN(
            num_variables=num_variables,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=3
        )
    elif model_name == 'WIRE':
        return WIRE(
            num_variables=num_variables,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=3
        )
    elif model_name == 'N-BEATS':
        return NBeatsBaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_blocks=4,
            num_layers=3
        )
    elif model_name == 'TimesNet':
        return TimesNetBaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=2
        )
    elif model_name == 'PatchTST':
        n_heads = 8 if hidden_dim % 8 == 0 else (4 if hidden_dim % 4 == 0 else 1)
        return PatchTSTBaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            patch_len=16,
            stride=8,
            d_model=hidden_dim,
            n_heads=n_heads,
            num_layers=3,
            d_ff=hidden_dim * 4,
            dropout=0.1
        )
    elif model_name == 'iTransformer':
        n_heads = 8 if hidden_dim % 8 == 0 else (4 if hidden_dim % 4 == 0 else 1)
        return iTransformerBaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            d_model=hidden_dim,
            n_heads=n_heads,
            num_layers=3,
            d_ff=hidden_dim * 4,
            dropout=0.1,
            use_norm=True
        )
    elif model_name == 'PCA':
        return PCABaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            n_components=16
        )
    elif model_name == 'Fourier':
        return FourierBaseline(
            num_variables=num_variables,
            seq_length=seq_length,
            n_components=16
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, train_loader, val_loader, epochs, lr, device, model_name):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    if hasattr(model, 'initialize_atoms'):
        init_batch = next(iter(train_loader))
        x_init, t_init = init_batch
        model.initialize_atoms(x_init.to(device), t_init.to(device))
    
    pbar = tqdm(range(epochs), desc=f'  {model_name}', leave=False)
    for epoch in pbar:
        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        
        train_loss = 0.0
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            losses = model.compute_loss(x, t)
            loss = losses['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, t in val_loader:
                x, t = x.to(device), t.to(device)
                losses = model.compute_loss(x, t)
                val_loss += losses['reconstruction'].item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}'})
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            if hasattr(model, 'forward'):
                output = model(x, t)
                if isinstance(output, tuple):
                    output = output[0]
            
            mse = F.mse_loss(output, x, reduction='sum').item()
            mae = F.l1_loss(output, x, reduction='sum').item()
            
            total_mse += mse
            total_mae += mae
            num_samples += x.numel()
    
    return {
        'mse': total_mse / num_samples,
        'mae': total_mae / num_samples
    }


def run_comparison(
    datasets,
    models,
    data_dir='data',
    save_dir='outputs/reconstruction/comparison',
    seq_length=512,
    batch_size=32,
    epochs=200,
    lr=1e-3,
    hidden_dim=64,
    device='cuda'
):
    os.makedirs(save_dir, exist_ok=True)
    loader = MultiDatasetLoader(data_dir)
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)
        
        try:
            train_dataset, val_dataset, test_dataset = loader.get_dataset(
                dataset_name,
                seq_length=seq_length,
                stride=max(1, seq_length // 4),
                normalize=True
            )
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue
        
        sample_x, _ = train_dataset[0]
        actual_seq_length = sample_x.shape[0]
        num_variables = sample_x.shape[1]
        actual_batch_size = min(batch_size, len(train_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=max(1, min(actual_batch_size, len(val_dataset))), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=max(1, min(actual_batch_size, len(test_dataset))), shuffle=False)
        
        for model_name in models:
            print(f"\n  Training {model_name}...")
            
            try:
                model = get_model(model_name, num_variables, actual_seq_length, hidden_dim)
                model = train_model(model, train_loader, val_loader, epochs, lr, device, model_name)
                metrics = evaluate_model(model, test_loader, device)
                
                result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'params': sum(p.numel() for p in model.parameters()),
                }
                
                if hasattr(model, 'dgf') and hasattr(model.dgf, 'get_active_atoms'):
                    result['active_atoms'] = model.dgf.get_active_atoms()
                
                all_results.append(result)
                print(f"    MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
                
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(save_dir, 'comparison_results.csv'), index=False)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    pivot_mse = results_df.pivot(index='dataset', columns='model', values='mse')
    print("\nMSE by Dataset and Model:")
    print(pivot_mse.to_string())
    
    create_comparison_plots(results_df, save_dir)
    
    return results_df


def create_comparison_plots(results_df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = results_df['dataset'].unique()
    models = results_df['model'].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        mse_values = [model_data[model_data['dataset'] == d]['mse'].values[0] 
                      if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                      for d in datasets]
        axes[0].bar(x + i * width, mse_values, width, label=model)
    
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Comparison')
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        mae_values = [model_data[model_data['dataset'] == d]['mae'].values[0] 
                      if len(model_data[model_data['dataset'] == d]) > 0 else 0 
                      for d in datasets]
        axes[1].bar(x + i * width, mae_values, width, label=model)
    
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison')
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_metrics.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', 
                        default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'weather', 'illness', 'traffic'])
    parser.add_argument('--models', nargs='+',
                        default=['DualTimesField', 'SIREN', 'WIRE', 'N-BEATS', 'TimesNet', 'PatchTST', 'iTransformer', 'PCA', 'Fourier'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='outputs/reconstruction/comparison')
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_comparison(
        datasets=args.datasets,
        models=args.models,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device
    )
