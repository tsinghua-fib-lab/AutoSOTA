#!/usr/bin/env python
"""
Sampling/Inference script for GAT-FM model.

Usage:
    python sample.py --checkpoint outputs/gat_fm_run/best.ckpt --adata_path dataset/test.h5ad

Example:
    python sample.py \
        --checkpoint outputs/gse100866_run/best.ckpt \
        --adata_path dataset/GSE100866_adata.h5ad \
        --rna_embed_path dataset/rna_embedding/GSE100866.npy \
        --output_path predictions.npy \
        --num_steps 100 \
        --solver euler
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

from gat_fm import (
    GATFM,
    GATFMWrapper,
    ProteinGraph,
    ProteinDataset,
    NormalizationStats,
    FlowMatchingSampler,
    GuidedFlowMatchingSampler,
    create_sample_mask,
    create_dataloader,
    denormalize_protein_expression,
    compute_pcc,
    compute_rmse,
    compute_mae,
    compute_pcc_protein,
    compute_pcc_cell,
    compute_rmse_standardized,
    set_seed,
    get_device,
    load_config,
    TrainingConfig,
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
        default_configs = ['sample.yaml', 'config.yaml']
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
    parser = argparse.ArgumentParser(description='Sample from trained GAT-FM model')
    
    # Required
    parser.add_argument('--checkpoint', type=str, default=get_default('checkpoint', None),
                        help='Path to model checkpoint')
    parser.add_argument('--adata_path', type=str, default=get_default('adata_path', None),
                        help='Path to AnnData (.h5ad) file for inference')
    parser.add_argument('--rna_embed_path', type=str, default=get_default('rna_embed_path', None),
                        help='Path to RNA embeddings (.npy) file')
    
    # Optional paths
    parser.add_argument('--config_path', type=str, default=get_default('config_path', None),
                        help='Path to config file (auto-detect from checkpoint dir)')
    parser.add_argument('--norm_stats_path', type=str, default=get_default('norm_stats_path', None),
                        help='Path to normalization stats (auto-detect from checkpoint dir)')
    parser.add_argument('--ppi_path', type=str, default=get_default('ppi_path', None),
                        help='Path to PPI network file')
    parser.add_argument('--output_path', type=str, default=get_default('output_path', 'predictions.npy'),
                        help='Path to save predictions')
    
    # Sampling
    parser.add_argument('--num_steps', type=int, default=get_default('num_steps', 100),
                        help='Number of ODE integration steps')
    parser.add_argument('--solver', type=str, default=get_default('solver', 'euler'),
                        choices=['euler', 'midpoint', 'heun', 'rk4'],
                        help='ODE solver')
    
    # Guided sampling
    guided_default = get_default('guided', False)
    parser.add_argument('--guided', action='store_true', default=guided_default,
                        help='Use guided sampling with ground truth correction')
    parser.add_argument('--sample_mask_ratio', type=float, default=get_default('sample_mask_ratio', 0.2),
                        help='Fraction of proteins to mask for guided sampling prediction')
    
    # Misc
    parser.add_argument('--batch_size', type=int, default=get_default('batch_size', 64),
                        help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=get_default('seed', 42),
                        help='Random seed')
    gpu_default = get_default('gpu', None)
    parser.add_argument('--gpu', type=int, default=gpu_default if gpu_default is not None else None,
                        help='GPU ID (None for auto)')
    evaluate_default = get_default('evaluate', False)
    parser.add_argument('--evaluate', action='store_true', default=evaluate_default,
                        help='Compute evaluation metrics')
    save_normalized_default = get_default('save_normalized', False)
    parser.add_argument('--save_normalized', action='store_true', default=save_normalized_default,
                        help='Save normalized predictions (default: denormalized)')
    
    # Parse remaining arguments
    args = parser.parse_args(remaining)
    
    # Override boolean flags from config if they exist in config
    # (action='store_true' doesn't work well with config defaults, so we set them manually)
    if 'guided' in config_dict:
        # Only set if not explicitly provided in command line
        if '--guided' not in remaining:
            args.guided = bool(config_dict['guided'])
    if 'evaluate' in config_dict:
        if '--evaluate' not in remaining:
            args.evaluate = bool(config_dict['evaluate'])
    if 'save_normalized' in config_dict:
        if '--save_normalized' not in remaining:
            args.save_normalized = bool(config_dict['save_normalized'])
    
    # Validate required arguments
    if not args.checkpoint or not args.adata_path or not args.rna_embed_path:
        parser.error('--checkpoint, --adata_path and --rna_embed_path are required (provide via --config or command line)')
    
    return args


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict = None,
    device: torch.device = None,
):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Model configuration (optional, will try to load from checkpoint)
        device: Device to load model on
        
    Returns:
        model: Loaded GATFM model
        config: Configuration used
    """
    # PyTorch 2.6+ requires explicit allowlist for custom classes
    # Use safe_globals context manager to allow TrainingConfig
    with torch.serialization.safe_globals([TrainingConfig]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Try to get config from checkpoint or argument
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
            if hasattr(config, '__dict__'):
                config = vars(config)
        else:
            raise ValueError("Config not found in checkpoint. Please provide --config_path")
    
    # We need protein_dim from the checkpoint state dict
    # Look for it in the model weights
    state_dict = checkpoint['model_state_dict']
    
    # Find protein_dim from final layer
    for key in state_dict.keys():
        if 'final_layer.linear.bias' in key:
            protein_dim = state_dict[key].shape[0]
            break
    else:
        raise ValueError("Could not determine protein_dim from checkpoint")
    
    # Find num_datasets from dataset embedder
    for key in state_dict.keys():
        if 'dataset_embedder.embedding_table.weight' in key:
            num_datasets = state_dict[key].shape[0]
            break
    else:
        num_datasets = 1
    
    # Create model
    model = GATFM(
        protein_dim=protein_dim,
        hidden_size=config.get('hidden_size', 256),
        depth=config.get('depth', 6),
        num_heads=config.get('num_heads', 4),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        num_datasets=num_datasets,
        rna_embed_dim=512,  # Fixed
        dropout=config.get('dropout', 0.0),  # No dropout at inference
    )
    
    # Load state dict
    # Need to handle wrapper prefix
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            model_state_dict[key[9:]] = value  # Remove 'backbone.' prefix
        else:
            model_state_dict[key] = value
    
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def sample_predictions(
    model: GATFM,
    dataloader,
    protein_graph: ProteinGraph,
    num_steps: int = 100,
    solver: str = 'euler',
    device: torch.device = None,
    guided: bool = False,
    sample_mask_ratio: float = 0.2,
):
    """
    Generate predictions for a dataset.
    
    Args:
        model: Trained GATFM model
        dataloader: Data loader
        protein_graph: Protein graph
        num_steps: ODE integration steps
        solver: ODE solver
        device: Device
        guided: Whether to use guided sampling with ground truth correction
        sample_mask_ratio: Fraction of proteins to mask for guided sampling
        
    Returns:
        predictions: (N, P) array of predicted protein expression
        targets: (N, P) array of ground truth (if available)
        masks: (N, P) array of observation masks
        sample_masks: (N, P) array of sample masks (only for guided mode)
    """
    model.eval()
    
    if guided:
        sampler = GuidedFlowMatchingSampler(model, num_steps=num_steps, solver=solver)
    else:
        sampler = FlowMatchingSampler(model, num_steps=num_steps, solver=solver)
    
    all_predictions = []
    all_targets = []
    all_masks = []
    all_sample_masks = []
    
    for batch in tqdm(dataloader, desc='Sampling'):
        x1 = batch['protein_expr'].to(device)
        mask = batch['protein_mask'].to(device)
        rna_embed = batch['rna_embed'].to(device)
        dataset_id = batch['dataset_id'].to(device)
        
        edge_index = protein_graph.base_edge_index.to(device)
        
        if guided:
            # Create sample mask: randomly mask sample_mask_ratio of observed proteins
            sample_mask = create_sample_mask(
                protein_mask=mask,
                sample_mask_ratio=sample_mask_ratio,
            )
            
            # Generate predictions with guided sampling
            predictions = sampler.sample(
                x1_true=x1,
                sample_mask=sample_mask,
                edge_index=edge_index,
                cond_rna=rna_embed,
                cond_dataset=dataset_id,
                full_mask=mask,
                device=device,
            )
            
            all_sample_masks.append(sample_mask.cpu().numpy())
        else:
            # Generate predictions with standard sampling
            predictions = sampler.sample(
                shape=x1.shape,
                edge_index=edge_index,
                cond_rna=rna_embed,
                cond_dataset=dataset_id,
                mask=mask,
                device=device,
            )
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(x1.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    if guided:
        sample_masks = np.concatenate(all_sample_masks, axis=0)
        return predictions, targets, masks, sample_masks
    
    return predictions, targets, masks


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # Auto-detect paths
    checkpoint_dir = Path(args.checkpoint).parent
    
    if args.config_path is None:
        config_path = checkpoint_dir / 'config.yaml'
        if config_path.exists():
            args.config_path = str(config_path)
    
    if args.norm_stats_path is None:
        norm_stats_path = checkpoint_dir / 'norm_stats.npz'
        if norm_stats_path.exists():
            args.norm_stats_path = str(norm_stats_path)
    
    # Load config
    config = None
    if args.config_path:
        print(f"Loading config from {args.config_path}")
        config = load_config(args.config_path)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, config, device)
    
    # Load normalization stats
    norm_stats = None
    if args.norm_stats_path:
        print(f"Loading normalization stats from {args.norm_stats_path}")
        norm_stats = NormalizationStats.load(args.norm_stats_path)
    
    # Load dataset
    print(f"Loading dataset from {args.adata_path}")
    dataset = ProteinDataset(
        adata_path=args.adata_path,
        rna_embed_path=args.rna_embed_path,
        normalize=True,
        norm_stats=norm_stats,
    )
    
    print(f"Dataset: {len(dataset)} cells, {dataset.num_proteins} proteins")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # For inference
        drop_last=False,
    )
    
    # Create protein graph
    protein_names = dataset.get_protein_names()
    protein_graph = ProteinGraph(
        protein_names=protein_names,
        ppi_path=args.ppi_path,
        add_self_loops_flag=True,
    )
    protein_graph = protein_graph.to(device)
    
    # Generate predictions
    if args.guided:
        print(f"\nGuided sampling with {args.num_steps} steps using {args.solver} solver...")
        print(f"Sample mask ratio: {args.sample_mask_ratio}")
        predictions, targets, masks, sample_masks = sample_predictions(
            model=model,
            dataloader=dataloader,
            protein_graph=protein_graph,
            num_steps=args.num_steps,
            solver=args.solver,
            device=device,
            guided=True,
            sample_mask_ratio=args.sample_mask_ratio,
        )
        eval_mask = sample_masks  # Only evaluate on masked (predicted) proteins
    else:
        print(f"\nSampling with {args.num_steps} steps using {args.solver} solver...")
        predictions, targets, masks = sample_predictions(
            model=model,
            dataloader=dataloader,
            protein_graph=protein_graph,
            num_steps=args.num_steps,
            solver=args.solver,
            device=device,
            guided=False,
        )
        eval_mask = masks
    
    # Denormalize if requested
    if not args.save_normalized and norm_stats is not None:
        print("Denormalizing predictions...")
        predictions = denormalize_protein_expression(predictions, norm_stats)
        targets = denormalize_protein_expression(targets, norm_stats)
    
    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluation metrics:")
        
        if args.guided:
            # Use standardized metrics for guided sampling (following ComputePCC&CMD&RMSE.ipynb)
            pcc_protein, protein_pccs_list = compute_pcc_protein(predictions, targets, eval_mask, standardize=True)
            pcc_cell, cell_pccs_list = compute_pcc_cell(predictions, targets, eval_mask, standardize=True)
            rmse = compute_rmse_standardized(predictions, targets, eval_mask, standardize=True)
            
            print(f"  PCC (protein-wise): {pcc_protein:.4f}")
            print(f"  PCC (cell-wise):    {pcc_cell:.4f}")
            print(f"  RMSE (standardized): {rmse:.4f}")
        else:
            pcc = compute_pcc(predictions, targets, eval_mask)
            rmse = compute_rmse(predictions, targets, eval_mask)
            mae = compute_mae(predictions, targets, eval_mask)
            
            print(f"  PCC:  {pcc:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
        
        # Per-protein metrics
        print("\nPer-protein PCC (top 10):")
        protein_pccs = []
        for i in range(predictions.shape[1]):
            p, t = predictions[:, i], targets[:, i]
            m = eval_mask[:, i] > 0
            if m.sum() > 1:
                pcc_i = np.corrcoef(p[m], t[m])[0, 1]
                if not np.isnan(pcc_i):
                    protein_pccs.append((protein_names[i], pcc_i))
        
        protein_pccs.sort(key=lambda x: x[1], reverse=True)
        for name, pcc in protein_pccs[:10]:
            print(f"  {name}: {pcc:.4f}")
    
    # Save predictions
    print(f"\nSaving predictions to {args.output_path}")
    np.save(args.output_path, predictions)
    
    # Also save a results dict
    results_path = Path(args.output_path).with_suffix('.npz')
    if args.guided:
        np.savez(
            results_path,
            predictions=predictions,
            targets=targets,
            masks=masks,
            sample_masks=sample_masks,
            protein_names=protein_names,
        )
    else:
        np.savez(
            results_path,
            predictions=predictions,
            targets=targets,
            masks=masks,
            protein_names=protein_names,
        )
    print(f"Full results saved to {results_path}")


if __name__ == '__main__':
    main()
