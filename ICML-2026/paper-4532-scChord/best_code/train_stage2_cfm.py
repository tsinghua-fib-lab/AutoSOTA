# -*- coding: utf-8 -*-
"""
scChord Stage 2: CFM (Conditional Flow Matching) Training
Objective: Learn flow to generate protein latent variable z_prot conditioned on RNA

Example usage:
python train_stage2_cfm.py \
    --data_path "/path/to/dataset.h5ad" \
    --vae_path ./outputs/stage1/vae_best.pt \
    --output_dir ./outputs/stage2 \
    --device cuda:0 \
    --epochs 200 \
    --n_top_genes 1000 \
    --batch_size 512 \
    --lr 1e-4 \
    --dc 512 \
    --p_uncond 0.2 \
    --lambda_cons 0.1 \
    --n_steps 25 \
    --cfg_scale 2.0
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from torchdiffeq import odeint

from data import load_data, load_data_cross_dataset, get_dataloader
from models import ProteinVAE, RNAEncoder, FlowNet, apply_gene_mask
from metrics import evaluate_predictions
from visualization import save_evaluation_results


class ODEFunc(nn.Module):
    """
    ODE function wrapper for torchdiffeq.
    
    Wraps FlowNet to support CFG (Classifier-Free Guidance).
    """
    
    def __init__(
        self, 
        flow_net: FlowNet, 
        c: torch.Tensor, 
        batch_id: torch.Tensor,
        cfg_scale: float = 1.0,
        use_cfg: bool = True
    ):
        """
        Args:
            flow_net: Flow network
            c: Condition vector [B, dc]
            batch_id: Batch ID [B]
            cfg_scale: CFG weight
            use_cfg: Whether to use CFG
        """
        super().__init__()
        self.flow_net = flow_net
        self.c = c
        self.batch_id = batch_id
        self.cfg_scale = cfg_scale
        self.use_cfg = use_cfg
        
        B = c.shape[0]
        device = c.device
        self.cond_null = flow_net.get_cond_null(B, device)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        ODE right-hand side: dx/dt = v(x, t, c)
        
        Args:
            t: Time (scalar)
            x: Current state [B, dz]
            
        Returns:
            v: Vector field [B, dz]
        """
        B = x.shape[0]
        t_batch = torch.full((B,), t.item(), device=x.device)
        
        if self.use_cfg and self.cfg_scale != 1.0:
            # Conditional and unconditional vector fields
            v_cond = self.flow_net(x, t_batch, self.c, self.batch_id)
            v_uncond = self.flow_net(x, t_batch, self.cond_null, self.batch_id)
            # CFG
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            v = self.flow_net(x, t_batch, self.c, self.batch_id)
        
        return v


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(
    vae: ProteinVAE,
    rna_encoder: RNAEncoder,
    flow_net: FlowNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    p_uncond: float = 0.15,
    lambda_cons: float = 0.1,
    mask_ratio_range: tuple = (0.2, 0.5),
    epsilon_w: float = 0.0
) -> dict:
    """
    Train for one epoch.
    
    Args:
        vae: Frozen VAE
        rna_encoder: RNA encoder
        flow_net: Flow network
        dataloader: Data loader
        optimizer: Optimizer
        device: Device
        p_uncond: Probability of unconditional training (CFG)
        lambda_cons: Consistency loss weight
        mask_ratio_range: Gene mask ratio range
    """
    rna_encoder.train()
    flow_net.train()
    vae.eval()  # VAE always in eval mode
    
    total_loss = 0
    total_cfm = 0
    total_cons = 0
    n_batches = 0
    
    for batch in dataloader:
        rna_norm = batch['rna_norm'].to(device)
        prot_norm = batch['prot_norm'].to(device)
        batch_id = batch['batch_id'].to(device)
        B = rna_norm.shape[0]
        
        optimizer.zero_grad()
        
        # 1. Construct RNA masking
        rna_masked = apply_gene_mask(rna_norm, mask_ratio_range)
        
        # 2. Encode RNA
        c_full = rna_encoder(rna_norm, batch_id)      # [B, dc]
        c_mask = rna_encoder(rna_masked, batch_id)    # [B, dc]
        
        # 3. Consistency loss
        L_cons = ((c_full - c_mask) ** 2).mean()
        
        # 4. CFG dropout
        drop_mask = torch.rand(B, device=device) < p_uncond  # [B]
        cond_null = flow_net.get_cond_null(B, device)  # [B, dc]
        c_used = torch.where(drop_mask.unsqueeze(-1), cond_null, c_full)
        
        # 5. Get protein latent x1 (VAE encoder)
        with torch.no_grad():
            mu_z, logvar_z = vae.encode(prot_norm, batch_id)
            x1 = vae.reparameterize(mu_z, logvar_z)  # [B, dz]
        
        # 6. Sample noise x0
        x0 = torch.randn_like(x1)  # [B, dz]
        
        # 7. Sample time t
        t = torch.rand(B, device=device)  # [B]
        
        # 8. Linear interpolation path
        t_expand = t.unsqueeze(-1)  # [B, 1]
        x_t = (1 - t_expand) * x0 + t_expand * x1  # [B, dz]
        u_t = x1 - x0  # Target vector field
        
        # 9. Predict vector field
        v = flow_net(x_t, t, c_used, batch_id)  # [B, dz]
        
        # 10. CFM loss (with optional importance weighting)
        if epsilon_w > 0:
            with torch.no_grad():
                # Normalize distance by latent dimension for scale invariance
                dist = ((x0 - x1) ** 2).sum(dim=-1) / vae.dz  # [B], per-dim MSE
                weights = torch.exp(-dist / epsilon_w)  # [B], Gibbs kernel
                weights = weights / (weights.sum() + 1e-8) * B  # normalize to mean=1
            L_cfm = (((v - u_t) ** 2).mean(dim=-1) * weights).mean()
        else:
            L_cfm = ((v - u_t) ** 2).mean()
        
        # 11. Total loss
        loss = L_cfm + lambda_cons * L_cons
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(rna_encoder.parameters()) + list(flow_net.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        total_cfm += L_cfm.item()
        total_cons += L_cons.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'cfm': total_cfm / n_batches,
        'cons': total_cons / n_batches
    }


@torch.no_grad()
def validate(
    vae: ProteinVAE,
    rna_encoder: RNAEncoder,
    flow_net: FlowNet,
    dataloader: DataLoader,
    device: torch.device,
    lambda_cons: float = 0.1
) -> dict:
    """Validation (without mask and CFG dropout)."""
    rna_encoder.eval()
    flow_net.eval()
    vae.eval()
    
    total_loss = 0
    total_cfm = 0
    total_cons = 0
    n_batches = 0
    
    for batch in dataloader:
        rna_norm = batch['rna_norm'].to(device)
        prot_norm = batch['prot_norm'].to(device)
        batch_id = batch['batch_id'].to(device)
        B = rna_norm.shape[0]
        
        # Encode RNA
        c_full = rna_encoder(rna_norm, batch_id)
        
        # Get protein latent
        mu_z, logvar_z = vae.encode(prot_norm, batch_id)
        x1 = vae.reparameterize(mu_z, logvar_z)
        
        # Sample
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device)
        
        # Interpolation
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        u_t = x1 - x0
        
        # Predict
        v = flow_net(x_t, t, c_full, batch_id)
        
        # Loss
        L_cfm = ((v - u_t) ** 2).mean()
        L_cons = torch.tensor(0.0, device=device)  # No cons loss during validation
        loss = L_cfm
        
        total_loss += loss.item()
        total_cfm += L_cfm.item()
        total_cons += L_cons.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'cfm': total_cfm / n_batches,
        'cons': total_cons / n_batches
    }


@torch.no_grad()
def inference_and_evaluate(
    vae: ProteinVAE,
    rna_encoder: RNAEncoder,
    flow_net: FlowNet,
    dataloader: DataLoader,
    device: torch.device,
    n_steps: int = 50,
    cfg_scale: float = 2.0,
    ode_method: str = 'dopri5',
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> tuple:
    """
    Inference and evaluation using torchdiffeq ODE solver.
    
    Args:
        vae: VAE model
        rna_encoder: RNA encoder
        flow_net: Flow network
        dataloader: Data loader
        device: Device
        n_steps: Number of ODE integration steps (for fixed-step methods)
        cfg_scale: CFG weight
        ode_method: ODE solver method: 'dopri5', 'rk4', 'euler', 'midpoint', etc.
        rtol: Relative tolerance (for adaptive step-size methods)
        atol: Absolute tolerance (for adaptive step-size methods)
        
    Returns:
        predictions: Predicted protein expression [N, M]
        ground_truth: Ground truth protein expression [N, M]
    """
    rna_encoder.eval()
    flow_net.eval()
    vae.eval()
    
    all_preds = []
    all_truth = []
    
    # Integration time points
    t_span = torch.tensor([0.0, 1.0], device=device)
    
    for batch in tqdm(dataloader, desc='Inference'):
        rna_norm = batch['rna_norm'].to(device)
        prot_norm = batch['prot_norm'].to(device)
        batch_id = batch['batch_id'].to(device)
        B = rna_norm.shape[0]
        
        # Encode RNA
        c = rna_encoder(rna_norm, batch_id)  # [B, dc]
        
        # Sample initial noise
        x0 = torch.randn(B, vae.dz, device=device)  # [B, dz]
        
        # Create ODE function
        ode_func = ODEFunc(
            flow_net=flow_net,
            c=c,
            batch_id=batch_id,
            cfg_scale=cfg_scale,
            use_cfg=True
        )
        
        # Use torchdiffeq odeint for integration
        # Returns shape: [len(t_span), B, dz]
        x_trajectory = odeint(
            ode_func,
            x0,
            t_span,
            method=ode_method,
            rtol=rtol,
            atol=atol
        )
        
        # Take final state (t=1)
        x = x_trajectory[-1]  # [B, dz]

        # Decode
        y_hat = vae.decode(x, batch_id)  # [B, M]

        # For ZINB/NB: decoder outputs raw-count-space predictions.
        # Log-normalize to match the evaluation space (log-normalized) used by ground truth.
        if vae.dist_type in ['NB', 'ZINB']:
            y_hat = y_hat / (torch.sum(y_hat, dim=1, keepdim=True) + 1e-8) * 1e4
            y_hat = torch.log(y_hat + 1.)

        all_preds.append(y_hat.cpu().numpy())
        all_truth.append(prot_norm.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_truth, axis=0)
    
    return predictions, ground_truth


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
    
    # Load VAE
    vae_ckpt_path = Path(args.vae_path)
    print(f"\nLoading VAE from {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_config = vae_ckpt['config']
    
    vae = ProteinVAE(
        n_proteins=vae_config['n_proteins'],
        dz=vae_config['dz'],
        hidden_dims=vae_config['hidden_dims'],
        batch_emb_dim=vae_config['batch_emb_dim'],
        n_batches=vae_config['n_batches'],
        beta_kl=vae_config['beta_kl'],
        learnable_dispersion=True,
        dist_type=vae_config.get('dist_type', 'Gaussian')
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"VAE loaded. Epoch: {vae_ckpt['epoch']}, Val loss: {vae_ckpt['val_loss']:.4f}")
    
    # Create RNAEncoder and FlowNet
    rna_encoder = RNAEncoder(
        n_genes=data_info['n_genes'],
        dc=args.dc,
        hidden_dims=args.rna_hidden_dims,
        batch_emb_dim=args.batch_emb_dim,
        n_batches=args.n_batches,
        dropout=0.1
    ).to(device)
    
    flow_net = FlowNet(
        dz=vae_config['dz'],
        dc=args.dc,
        hidden_dim=args.flow_hidden_dim,
        n_blocks=args.flow_n_blocks,
        time_emb_dim=64,
        batch_emb_dim=args.batch_emb_dim,
        n_batches=args.n_batches,
        dropout=0.1
    ).to(device)
    
    print(f"\nRNAEncoder parameters: {sum(p.numel() for p in rna_encoder.parameters()):,}")
    print(f"FlowNet parameters: {sum(p.numel() for p in flow_net.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(rna_encoder.parameters()) + list(flow_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with 25-epoch warmup
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    warmup_epochs = 25
    main_epochs = args.epochs - warmup_epochs

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=args.lr * 0.01)
        ],
        milestones=[warmup_epochs]
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = None
    
    print("\n" + "=" * 60)
    print("Stage 2: Training CFM (RNAEncoder + FlowNet)")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_metrics = train_epoch(
            vae, rna_encoder, flow_net, train_loader, optimizer, device,
            p_uncond=args.p_uncond,
            lambda_cons=args.lambda_cons,
            mask_ratio_range=(0.2, 0.5),
            epsilon_w=args.epsilon_w
        )
        
        # Validation
        val_metrics = validate(
            vae, rna_encoder, flow_net, val_loader, device,
            lambda_cons=args.lambda_cons
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print log
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_metrics['loss']:.4f} (CFM: {train_metrics['cfm']:.4f}, Cons: {train_metrics['cons']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (CFM: {val_metrics['cfm']:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'rna_encoder_state_dict': rna_encoder.state_dict(),
                'flow_net_state_dict': flow_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {
                    'n_genes': data_info['n_genes'],
                    'dc': args.dc,
                    'rna_hidden_dims': args.rna_hidden_dims,
                    'flow_hidden_dim': args.flow_hidden_dim,
                    'flow_n_blocks': args.flow_n_blocks,
                    'batch_emb_dim': args.batch_emb_dim,
                    'n_batches': args.n_batches,
                    'dz': vae_config['dz'],
                }
            }, output_dir / 'flow_best.pt')
        
        # Periodically save checkpoints
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'rna_encoder_state_dict': rna_encoder.state_dict(),
                'flow_net_state_dict': flow_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'flow_epoch_{epoch:03d}.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'rna_encoder_state_dict': rna_encoder.state_dict(),
        'flow_net_state_dict': flow_net.state_dict(),
    }, output_dir / 'flow_final.pt')
    
    print("\n" + "=" * 60)
    print(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    # Load best model and evaluate
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    print("=" * 60)
    
    best_ckpt = torch.load(output_dir / 'flow_best.pt', map_location=device)
    rna_encoder.load_state_dict(best_ckpt['rna_encoder_state_dict'])
    flow_net.load_state_dict(best_ckpt['flow_net_state_dict'])
    
    predictions, ground_truth = inference_and_evaluate(
        vae, rna_encoder, flow_net, val_loader, device,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
        ode_method=args.ode_method,
        rtol=args.ode_rtol,
        atol=args.ode_atol
    )
    
    # Evaluate
    results = evaluate_predictions(
        predictions, ground_truth,
        protein_names=data_info['protein_names'],
        verbose=True
    )
    
    # Save prediction results
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'ground_truth.npy', ground_truth)
    
    # Save visualization results
    print("\n" + "=" * 60)
    print("Generating visualization plots...")
    print("=" * 60)
    
    save_evaluation_results(
        results=results,
        save_dir=output_dir / 'figures',
        protein_names=data_info['protein_names'],
        title_prefix="scChord"
    )
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Train CFM')
    
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

    # VAE parameters
    parser.add_argument('--vae_path', type=str, default='./outputs/stage1/vae_best.pt',
                        help='Path to pretrained VAE')
    
    # Model parameters
    parser.add_argument('--dc', type=int, default=512,
                        help='Condition vector dimension')
    parser.add_argument('--rna_hidden_dims', type=int, nargs='+', default=[1024, 512],
                        help='RNAEncoder hidden layer dimensions')
    parser.add_argument('--flow_hidden_dim', type=int, default=256,
                        help='FlowNet hidden dimension')
    parser.add_argument('--flow_n_blocks', type=int, default=4,
                        help='Number of AdaLN blocks in FlowNet')
    parser.add_argument('--batch_emb_dim', type=int, default=8,
                        help='Batch embedding dimension')
    parser.add_argument('--n_batches', type=int, default=2,
                        help='Number of batches')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--p_uncond', type=float, default=0.15,
                        help='Probability of unconditional training (CFG)')
    parser.add_argument('--epsilon_w', type=float, default=0.0,
                        help='Gibbs kernel epsilon for CFM importance weighting (0=disabled)')
    parser.add_argument('--lambda_cons', type=float, default=0.1,
                        help='Consistency loss weight')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Inference parameters
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of ODE integration steps (for fixed-step methods)')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='CFG scale for inference')
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['dopri5', 'dopri8', 'rk4', 'euler', 'midpoint', 'heun3', 'adaptive_heun'],
                        help='ODE solver method (dopri5 recommended for accuracy)')
    parser.add_argument('--ode_rtol', type=float, default=1e-5,
                        help='Relative tolerance for adaptive ODE solvers')
    parser.add_argument('--ode_atol', type=float, default=1e-5,
                        help='Absolute tolerance for adaptive ODE solvers')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs/stage2',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)

