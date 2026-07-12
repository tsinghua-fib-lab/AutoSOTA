# -*- coding: utf-8 -*-
"""
scChord Stage 1: ProteinVAE Training
Objective: Learn latent representation z_prot of proteins
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from data import load_data, load_data_cross_dataset, get_dataloader
from models import ProteinVAE


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(
    model: ProteinVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_raw_for_nb: bool = False
) -> dict:
    """Train for one epoch.
    
    Args:
        use_raw_for_nb: If True and model uses NB distribution, 
                        compute loss using raw counts.
    """
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    n_batches = 0
    
    for batch in dataloader:
        prot_norm = batch['prot_norm'].to(device)
        prot_raw = batch['prot_raw'].to(device)
        batch_id = batch['batch_id'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_hat, mu, logvar, z, pi_logit = model(prot_norm, batch_id)
        
        # Compute loss
        if model.dist_type in ['NB', 'ZINB'] and use_raw_for_nb:
            # NB/ZINB distribution uses raw counts for loss computation
            losses = model.loss(prot_norm, y_hat, mu, logvar, y_raw=prot_raw, pi_logit=pi_logit)
        else:
            losses = model.loss(prot_norm, y_hat, mu, logvar, pi_logit=pi_logit)
        loss = losses['loss_total']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['loss_total'].item()
        total_nll += losses['nll'].item()
        total_kl += losses['kl'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'nll': total_nll / n_batches,
        'kl': total_kl / n_batches
    }


@torch.no_grad()
def validate(
    model: ProteinVAE,
    dataloader: DataLoader,
    device: torch.device,
    use_raw_for_nb: bool = False
) -> dict:
    """Validation.
    
    Args:
        use_raw_for_nb: If True and model uses NB distribution,
                        compute loss using raw counts.
    """
    model.eval()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    n_batches = 0
    
    for batch in dataloader:
        prot_norm = batch['prot_norm'].to(device)
        prot_raw = batch['prot_raw'].to(device)
        batch_id = batch['batch_id'].to(device)
        
        y_hat, mu, logvar, z, pi_logit = model(prot_norm, batch_id)
        
        if model.dist_type in ['NB', 'ZINB'] and use_raw_for_nb:
            losses = model.loss(prot_norm, y_hat, mu, logvar, y_raw=prot_raw, pi_logit=pi_logit)
        else:
            losses = model.loss(prot_norm, y_hat, mu, logvar, pi_logit=pi_logit)
        
        total_loss += losses['loss_total'].item()
        total_nll += losses['nll'].item()
        total_kl += losses['kl'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'nll': total_nll / n_batches,
        'kl': total_kl / n_batches
    }


def main(args):
    # Set random seed
    set_seed(args.seed)

    # Determine split seed
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.test_data_path:
        # Cross-dataset mode: dataset1 as training set, dataset2 as test set
        print("\n" + "=" * 60)
        print("Using CROSS-DATASET mode")
        print(f"Train: {args.data_path}")
        print(f"Test:  {args.test_data_path}")
        print("=" * 60 + "\n")
        train_dataset, test_dataset, data_info = load_data_cross_dataset(
            train_path=args.data_path,
            test_path=args.test_data_path,
            n_top_genes=args.n_top_genes
        )
    else:
        # Single-dataset mode: random train/test split
        print("\n" + "=" * 60)
        print("Using SINGLE-DATASET mode (random split)")
        print(f"Data: {args.data_path}")
        print(f"Split seed: {split_seed}")
        print("=" * 60 + "\n")
        train_dataset, test_dataset, data_info = load_data(
            args.data_path,
            n_top_genes=args.n_top_genes,
            train_ratio=args.train_ratio,
            random_state=split_seed
        )
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Save data info
    torch.save(data_info, output_dir / 'data_info.pt')
    
    # Create model
    model = ProteinVAE(
        n_proteins=data_info['n_proteins'],
        dz=args.dz,
        hidden_dims=args.hidden_dims,
        batch_emb_dim=args.batch_emb_dim,
        n_batches=args.n_batches,
        beta_kl=args.beta_kl,
        learnable_dispersion=True,
        dist_type=args.dist_type
    ).to(device)
    
    print(f"\nProteinVAE (dist={args.dist_type}) parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with 50-epoch warmup
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    warmup_epochs = 50
    cosine_epochs = 150
    min_lr = args.lr * 0.01

    after_cosine_epochs = args.epochs - warmup_epochs - cosine_epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr),
            LinearLR(optimizer, start_factor=1.0, end_factor=min_lr, total_iters=after_cosine_epochs),
        ],
        milestones=[warmup_epochs, warmup_epochs + cosine_epochs]
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("Stage 1: Training ProteinVAE")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_raw_for_nb=args.use_raw_for_nb
        )
        
        # Validation
        val_metrics = validate(
            model, val_loader, device,
            use_raw_for_nb=args.use_raw_for_nb
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print log
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_metrics['loss']:.4f} (NLL: {train_metrics['nll']:.4f}, KL: {train_metrics['kl']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (NLL: {val_metrics['nll']:.4f}, KL: {val_metrics['kl']:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {
                    'n_proteins': data_info['n_proteins'],
                    'dz': args.dz,
                    'hidden_dims': args.hidden_dims,
                    'batch_emb_dim': args.batch_emb_dim,
                    'n_batches': args.n_batches,
                    'beta_kl': args.beta_kl,
                    'dist_type': args.dist_type,
                }
            }, output_dir / 'vae_best.pt')
        
        # Periodically save checkpoints
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'vae_epoch_{epoch:03d}.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, output_dir / 'vae_final.pt')
    
    print("\n" + "=" * 60)
    print(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Train ProteinVAE')
    
    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='./data/example.h5ad',
                        help='Path to training H5AD data file (or single dataset for random split)')
    parser.add_argument('--test_data_path', type=str, default=None,
                        help='Path to test H5AD data file (if provided, use cross-dataset mode)')
    parser.add_argument('--n_top_genes', type=int, default=1000,
                        help='Number of highly variable genes')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio (only used in single-dataset mode)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Random seed for train/test split (default: same as --seed)')
    
    # Model parameters
    parser.add_argument('--dz', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--batch_emb_dim', type=int, default=8,
                        help='Batch embedding dimension')
    parser.add_argument('--n_batches', type=int, default=2,
                        help='Number of batches')
    parser.add_argument('--beta_kl', type=float, default=1.0,
                        help='KL loss weight')
    parser.add_argument('--dist_type', type=str, default='Gaussian',
                        choices=['Gaussian', 'NB', 'ZINB'],
                        help='Distribution type: Gaussian, NB (Negative Binomial), or ZINB (Zero-Inflated NB)')
    parser.add_argument('--use_raw_for_nb', action='store_true',
                        help='Use raw protein counts for NB/ZINB loss (only effective when dist_type=NB or ZINB)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)

