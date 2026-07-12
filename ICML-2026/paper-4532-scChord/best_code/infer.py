# -*- coding: utf-8 -*-
"""
scChord Inference Script

This script loads trained models and generates protein expression from RNA.
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

from torchdiffeq import odeint

from data import log_normalize, SingleCellDataset, get_dataloader
from models import ProteinVAE, RNAEncoder, FlowNet
from metrics import evaluate_predictions


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
            batch_id: Batch identifiers [B]
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


class scChord:
    """scChord inference class for protein expression prediction."""
    
    def __init__(
        self,
        vae_path: str,
        flow_path: str,
        data_info_path: str,
        device: str = 'cuda:0'
    ):
        """
        Args:
            vae_path: Path to VAE model checkpoint
            flow_path: Path to Flow model checkpoint (contains RNAEncoder and FlowNet)
            data_info_path: Path to data info file
            device: Device to use for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load data info
        self.data_info = torch.load(data_info_path, map_location='cpu')
        self.gene_names = self.data_info['gene_names']
        self.protein_names = self.data_info['protein_names']
        self.batch_mapping = self.data_info.get('batch_mapping', None)
        
        # Load VAE
        vae_ckpt = torch.load(vae_path, map_location=self.device)
        vae_config = vae_ckpt['config']
        
        self.vae = ProteinVAE(
            n_proteins=vae_config['n_proteins'],
            dz=vae_config['dz'],
            hidden_dims=vae_config['hidden_dims'],
            batch_emb_dim=vae_config['batch_emb_dim'],
            n_batches=vae_config['n_batches'],
            beta_kl=vae_config['beta_kl'],
            learnable_dispersion=True,
            dist_type=vae_config.get('dist_type', 'Gaussian')
        ).to(self.device)
        self.vae.load_state_dict(vae_ckpt['model_state_dict'])
        self.vae.eval()
        
        # Load Flow
        flow_ckpt = torch.load(flow_path, map_location=self.device)
        flow_config = flow_ckpt['config']
        
        self.rna_encoder = RNAEncoder(
            n_genes=flow_config['n_genes'],
            dc=flow_config['dc'],
            hidden_dims=flow_config['rna_hidden_dims'],
            batch_emb_dim=flow_config['batch_emb_dim'],
            n_batches=flow_config['n_batches'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        self.rna_encoder.load_state_dict(flow_ckpt['rna_encoder_state_dict'])
        self.rna_encoder.eval()
        
        self.flow_net = FlowNet(
            dz=flow_config['dz'],
            dc=flow_config['dc'],
            hidden_dim=flow_config['flow_hidden_dim'],
            n_blocks=flow_config['flow_n_blocks'],
            time_emb_dim=64,
            batch_emb_dim=flow_config['batch_emb_dim'],
            n_batches=flow_config['n_batches'],
            dropout=0.0
        ).to(self.device)
        self.flow_net.load_state_dict(flow_ckpt['flow_net_state_dict'])
        self.flow_net.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Genes: {len(self.gene_names)}, Proteins: {len(self.protein_names)}")
    
    @torch.no_grad()
    def predict(
        self,
        rna_counts: np.ndarray,
        batch_ids: np.ndarray,
        n_steps: int = 50,
        cfg_scale: float = 2.0,
        batch_size: int = 256,
        ode_method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> np.ndarray:
        """
        Predict protein expression from RNA counts using torchdiffeq ODE solver.
        
        Args:
            rna_counts: RNA counts [N, G], must be HVG-filtered
            batch_ids: Batch identifiers [N]
            n_steps: Number of ODE integration steps (for fixed-step methods)
            cfg_scale: CFG weight
            batch_size: Inference batch size
            ode_method: ODE solver method: 'dopri5', 'rk4', 'euler', 'midpoint', etc.
            rtol: Relative tolerance (for adaptive step methods)
            atol: Absolute tolerance (for adaptive step methods)
            
        Returns:
            prot_pred: Predicted protein expression [N, M] (log normalized)
        """
        # Preprocess RNA
        rna_norm = log_normalize(rna_counts, target_sum=1e4).astype(np.float32)
        
        # Build DataLoader
        dataset = TensorDataset(
            torch.from_numpy(rna_norm),
            torch.from_numpy(batch_ids.astype(np.int64))
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Integration time points
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        
        all_preds = []
        
        for rna_batch, batch_id in tqdm(dataloader, desc='Predicting'):
            rna_batch = rna_batch.to(self.device)
            batch_id = batch_id.to(self.device)
            B = rna_batch.shape[0]
            
            # Encode RNA
            c = self.rna_encoder(rna_batch, batch_id)
            
            # Sample initial noise
            x0 = torch.randn(B, self.vae.dz, device=self.device)
            
            # Create ODE function
            ode_func = ODEFunc(
                flow_net=self.flow_net,
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
            y_hat = self.vae.decode(x, batch_id)
            all_preds.append(y_hat.cpu().numpy())
        
        return np.concatenate(all_preds, axis=0)
    
    def predict_from_adata(
        self,
        adata,
        batch_key: str = 'batch_id',
        n_steps: int = 50,
        cfg_scale: float = 2.0,
        batch_size: int = 256,
        ode_method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> pd.DataFrame:
        """
        Predict protein expression from AnnData object.
        
        Args:
            adata: Single-cell AnnData object
            batch_key: Batch column name in adata.obs
            n_steps: Number of ODE integration steps (for fixed-step methods)
            cfg_scale: CFG weight
            batch_size: Inference batch size
            ode_method: ODE solver method: 'dopri5', 'rk4', 'euler', 'midpoint', etc.
            rtol: Relative tolerance (for adaptive step methods)
            atol: Absolute tolerance (for adaptive step methods)
            
        Returns:
            pred_df: Predicted protein expression DataFrame
        """
        # Filter HVGs
        common_genes = [g for g in self.gene_names if g in adata.var_names]
        if len(common_genes) < len(self.gene_names):
            print(f"Warning: Only {len(common_genes)}/{len(self.gene_names)} genes found in adata")
        
        adata_hvg = adata[:, common_genes]
        
        # Extract RNA counts
        X = adata_hvg.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        
        # If gene count doesn't match, rearrange
        if len(common_genes) < len(self.gene_names):
            rna_counts = np.zeros((X.shape[0], len(self.gene_names)), dtype=np.float32)
            gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
            for i, g in enumerate(common_genes):
                rna_counts[:, gene_to_idx[g]] = X[:, i]
        else:
            # Rearrange to training order
            gene_order = [list(adata_hvg.var_names).index(g) for g in self.gene_names]
            rna_counts = X[:, gene_order].astype(np.float32)
        
        # Get batch_id, handle string types
        if batch_key in adata.obs.columns:
            batch_col = adata.obs[batch_key]
            
            if pd.api.types.is_numeric_dtype(batch_col):
                batch_ids = batch_col.to_numpy(dtype=np.int64, copy=False)
            else:
                # Convert non-numeric batch labels to categorical codes
                if pd.api.types.is_categorical_dtype(batch_col):
                    batch_col = batch_col.astype(object).where(batch_col.notna(), 'unknown').astype(str)
                else:
                    batch_col = batch_col.fillna('unknown').astype(str)
                
                # If batch_mapping was saved during training, use it for mapping
                if hasattr(self, 'batch_mapping') and self.batch_mapping is not None:
                    # Reverse mapping: batch_name -> batch_id
                    name_to_id = {v: k for k, v in self.batch_mapping.items()}
                    batch_ids = np.array([name_to_id.get(str(b), 0) for b in batch_col], dtype=np.int64)
                    print(f"Using saved batch_mapping: {self.batch_mapping}")
                else:
                    # Otherwise create new encoding
                    batch_cat = pd.Categorical(batch_col)
                    batch_ids = batch_cat.codes.astype(np.int64)
                    print(f"Created new batch encoding: {dict(enumerate(batch_cat.categories.astype(str)))}")
        else:
            batch_ids = np.zeros(adata.n_obs, dtype=np.int64)
        
        # Predict
        prot_pred = self.predict(
            rna_counts, batch_ids,
            n_steps=n_steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            ode_method=ode_method,
            rtol=rtol,
            atol=atol
        )
        
        # Build DataFrame
        pred_df = pd.DataFrame(
            prot_pred,
            index=adata.obs_names,
            columns=self.protein_names
        )
        
        return pred_df


def main(args):
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = scChord(
        vae_path=args.vae_path,
        flow_path=args.flow_path,
        data_info_path=args.data_info_path,
        device=device
    )
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    
    # Predict
    pred_df = model.predict_from_adata(
        adata,
        batch_key=args.batch_key,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        ode_method=args.ode_method,
        rtol=args.ode_rtol,
        atol=args.ode_atol
    )
    
    # If ground truth protein data is available, evaluate
    results = None
    if 'protein_expression' in adata.obsm:
        print("\nEvaluating predictions...")
        true_prot = adata.obsm['protein_expression']
        
        # Match protein names
        common_prots = [p for p in pred_df.columns if p in true_prot.columns]
        if len(common_prots) == 0:
            print("Warning: No common proteins found between predictions and ground truth!")
        else:
            pred_aligned = pred_df[common_prots].values
            
            # Ground truth also needs log normalization
            true_aligned = log_normalize(true_prot[common_prots].values)
            
            results = evaluate_predictions(
                pred_aligned, true_aligned,
                protein_names=common_prots,
                verbose=True
            )
            
            # Save evaluation results
            output_dir = Path(args.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics to file
            metrics_path = output_dir / 'inference_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("Inference Evaluation Metrics\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Data path: {args.data_path}\n")
                f.write(f"Number of cells: {adata.n_obs}\n")
                f.write(f"Number of proteins evaluated: {len(common_prots)}\n\n")
                f.write("-" * 60 + "\n")
                f.write("Overall Metrics:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean Correlation: {results['mean_corr']:.4f} ± {results['std_corr']:.4f}\n")
                f.write(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}\n")
                f.write(f"Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}\n\n")
            print(f"Metrics saved to {metrics_path}")
            
            # If visualization module is available, save plots
            try:
                from visualization import save_evaluation_results
                fig_path = output_dir / 'inference_evaluation.png'
                save_evaluation_results(
                    pred_aligned, true_aligned,
                    protein_names=common_prots,
                    save_path=str(fig_path)
                )
                print(f"Visualization saved to {fig_path}")
            except ImportError:
                print("visualization module not available, skipping plots")
    
    # Save predictions
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path)
    print(f"\nPredictions saved to {output_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scChord Inference')
    
    # Model paths
    parser.add_argument('--vae_path', type=str, default='./outputs/stage1/vae_best.pt',
                        help='Path to VAE model')
    parser.add_argument('--flow_path', type=str, default='./outputs/stage2/flow_best.pt',
                        help='Path to Flow model')
    parser.add_argument('--data_info_path', type=str, default='./outputs/stage1/data_info.pt',
                        help='Path to data info')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/example.h5ad',
                        help='Path to H5AD data file for inference')
    parser.add_argument('--batch_key', type=str, default='batch_id',
                        help='Batch key in adata.obs')
    
    # Inference arguments
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of ODE integration steps (for fixed-step methods)')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='CFG scale')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Inference batch size')
    parser.add_argument('--ode_method', type=str, default='dopri5',
                        choices=['dopri5', 'dopri8', 'rk4', 'euler', 'midpoint', 'heun3', 'adaptive_heun'],
                        help='ODE solver method (dopri5 recommended for accuracy)')
    parser.add_argument('--ode_rtol', type=float, default=1e-5,
                        help='Relative tolerance for adaptive ODE solvers')
    parser.add_argument('--ode_atol', type=float, default=1e-5,
                        help='Absolute tolerance for adaptive ODE solvers')
    
    # Output
    parser.add_argument('--output_path', type=str, default='./predictions.csv',
                        help='Output path for predictions')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    main(args)

