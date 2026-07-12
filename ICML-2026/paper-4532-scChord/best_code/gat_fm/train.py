#!/usr/bin/env python
"""
Training script for GAT-FM model.

Usage:
    python train.py --config config.yaml
    python train.py --adata_path dataset/GSE100866_adata.h5ad --rna_embed_path dataset/rna_embedding/GSE100866.npy

Example:
    python train.py \
        --adata_path dataset/GSE100866_adata.h5ad \
        --rna_embed_path dataset/rna_embedding/GSE100866.npy \
        --ppi_path dataset/string_interactions_short_GSE100866.tsv \
        --output_dir outputs/gse100866_run \
        --epochs 100 \
        --batch_size 32 \
        --hidden_size 256 \
        --depth 6
"""

import argparse
from pathlib import Path
import torch
import yaml

from gat_fm import (
    GATFM,
    DITFM,
    GATFMWrapper,
    ProteinGraph,
    ProteinDataset,
    Trainer,
    TrainingConfig,
    create_dataloader,
    set_seed,
    get_device,
    model_summary,
    save_config,
)


def parse_args():
    # First, parse only the config argument to load defaults
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    temp_args, remaining = temp_parser.parse_known_args()
    
    # Load config if provided, or try default location
    config_dict = {}
    config_path = temp_args.config
    
    # If no config specified, try default locations
    if config_path is None:
        default_configs = ['config.yaml', 'config.yml']
        for default_config in default_configs:
            if Path(default_config).exists():
                config_path = default_config
                break
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
            print(f"Loaded config file: {config_path}")
        except FileNotFoundError:
            print(f"Warning: config file not found: {config_path}")
        except Exception as e:
            print(f"Warning: failed to load config file {config_path}: {e}")
    
    # Helper function to get default from config or fallback
    def get_default(key, fallback):
        return config_dict.get(key, fallback)
    
    # Now create the full parser with defaults from config
    parser = argparse.ArgumentParser(description='Train GAT-FM model')
    
    # Data paths
    parser.add_argument('--adata_path', type=str, default=get_default('adata_path', None),
                        help='Path to AnnData (.h5ad) file')
    parser.add_argument('--rna_embed_path', type=str, default=get_default('rna_embed_path', None),
                        help='Path to RNA embeddings (.npy) file')
    parser.add_argument('--ppi_path', type=str, default=get_default('ppi_path', None),
                        help='Path to PPI network (.tsv) file')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=get_default('output_dir', 'outputs'),
                        help='Output directory')
    parser.add_argument('--run_name', type=str, default=get_default('run_name', 'gat_fm_run'),
                        help='Run name for saving')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default=get_default('model_type', 'gat_fm'),
                        choices=['gat_fm', 'dit_fm'],
                        help='Model type: gat_fm (GAT-based) or dit_fm (DiT-based)')
    parser.add_argument('--hidden_size', type=int, default=get_default('hidden_size', 256),
                        help='Model hidden dimension')
    parser.add_argument('--depth', type=int, default=get_default('depth', 6),
                        help='Number of blocks (GAT or DiT)')
    parser.add_argument('--num_heads', type=int, default=get_default('num_heads', 4),
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=get_default('mlp_ratio', 4.0),
                        help='MLP hidden dimension ratio')
    parser.add_argument('--dropout', type=float, default=get_default('dropout', 0.1),
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=get_default('epochs', 100),
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=get_default('batch_size', 32),
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=get_default('learning_rate', 1e-4),
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=get_default('weight_decay', 0.01),
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=get_default('warmup_epochs', 5),
                        help='Warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=get_default('grad_clip', 1.0),
                        help='Gradient clipping norm')
    
    # Flow Matching
    parser.add_argument('--sigma', type=float, default=get_default('sigma', 0.0),
                        help='OT-CFM noise scale')
    
    # Sampling evaluation during training
    parser.add_argument('--sample_mask_ratio', type=float, default=get_default('sample_mask_ratio', 0.2),
                        help='Fraction of proteins to mask for sampling evaluation')
    parser.add_argument('--sample_num_steps', type=int, default=get_default('sample_num_steps', 50),
                        help='Number of ODE steps for sampling evaluation')
    parser.add_argument('--sample_solver', type=str, default=get_default('sample_solver', 'euler'),
                        choices=['euler', 'midpoint', 'heun', 'rk4'],
                        help='ODE solver for sampling evaluation')
    parser.add_argument('--sample_num_batches', type=int, default=get_default('sample_num_batches', 10),
                        help='Number of batches to sample during training evaluation')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=get_default('train_split', 0.8),
                        help='Training data fraction')
    parser.add_argument('--val_split', type=float, default=get_default('val_split', 0.1),
                        help='Validation data fraction')
    
    # Misc
    parser.add_argument('--seed', type=int, default=get_default('seed', 42),
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=get_default('num_workers', 4),
                        help='Data loader workers')
    parser.add_argument('--gpu', type=int, default=get_default('gpu', None),
                        help='GPU ID (None for auto)')
    parser.add_argument('--resume', type=str, default=get_default('resume', None),
                        help='Path to checkpoint to resume from')
    
    # Parse remaining arguments
    args = parser.parse_args(remaining)
    
    # Validate required arguments
    if not args.adata_path or not args.rna_embed_path:
        parser.error('--adata_path and --rna_embed_path are required (provide via --config or command line)')
    
    return args


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.adata_path}...")
    full_dataset = ProteinDataset(
        adata_path=args.adata_path,
        rna_embed_path=args.rna_embed_path,
        normalize=True,
        compute_stats=True,
    )
    
    print(f"Dataset: {len(full_dataset)} cells, {full_dataset.num_proteins} proteins")
    print(f"RNA embedding dim: {full_dataset.rna_embed_dim}")
    
    # Split dataset
    import numpy as np
    n = len(full_dataset)
    indices = np.random.RandomState(args.seed).permutation(n)
    
    n_train = int(n * args.train_split)
    n_val = int(n * args.val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_indices)} test")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    
    # Create protein graph
    protein_names = full_dataset.get_protein_names()
    protein_graph = ProteinGraph(
        protein_names=protein_names,
        ppi_path=args.ppi_path,
        score_threshold=0.4,
        add_self_loops_flag=True,
    )
    
    graph_stats = protein_graph.get_stats()
    print(f"\nProtein graph: {graph_stats}")
    
    # Get number of datasets
    num_datasets = len(np.unique(full_dataset.dataset_id))
    print(f"Number of datasets: {num_datasets}")
    
    # Create model based on model_type
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == 'gat_fm':
        model = GATFM(
            protein_dim=full_dataset.num_proteins,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_datasets=num_datasets,
            rna_embed_dim=full_dataset.rna_embed_dim,
            dropout=args.dropout,
        )
    elif args.model_type == 'dit_fm':
        model = DITFM(
            protein_dim=full_dataset.num_proteins,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_datasets=num_datasets,
            rna_embed_dim=full_dataset.rna_embed_dim,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
      
    # Wrap for training
    wrapper = GATFMWrapper(model, sigma=args.sigma)
    
    # Create training config
    config = TrainingConfig(
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        sigma=args.sigma,
        sample_mask_ratio=args.sample_mask_ratio,
        sample_num_steps=args.sample_num_steps,
        sample_solver=args.sample_solver,
        sample_num_batches=args.sample_num_batches,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    
    # Create trainer
    trainer = Trainer(
        model=wrapper,
        protein_graph=protein_graph,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Save config
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / 'config.yaml')
    
    # Save normalization stats
    full_dataset.norm_stats.save(str(output_dir / 'norm_stats.npz'))
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    train_history, val_history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
