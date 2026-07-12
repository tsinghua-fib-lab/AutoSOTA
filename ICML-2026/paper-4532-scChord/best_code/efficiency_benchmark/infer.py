# -*- coding: utf-8 -*-
"""
scBridge-Flow inference script.
Loads trained models and generates protein expression from RNA.
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
from resource_monitor import ResourceMonitor, save_metrics_json


class ODEFunc(nn.Module):
    """
    ODE function wrapper for torchdiffeq.

    Wraps FlowNet to support classifier-free guidance (CFG).
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
        Parameters
        ----------
        flow_net : FlowNet
            Flow network.
        c : torch.Tensor
            Condition vector [B, dc].
        batch_id : torch.Tensor
            Batch IDs [B].
        cfg_scale : float
            CFG scale.
        use_cfg : bool
            Whether to use CFG.
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
        ODE right-hand side: dx/dt = v(x, t, c).

        Parameters
        ----------
        t : torch.Tensor
            Time (scalar).
        x : torch.Tensor
            Current state [B, dz].

        Returns
        ----------
        v : torch.Tensor
            Vector field [B, dz].
        """
        B = x.shape[0]
        t_batch = torch.full((B,), t.item(), device=x.device)
        
        if self.use_cfg and self.cfg_scale != 1.0:
            # Conditional and unconditional vector fields.
            v_cond = self.flow_net(x, t_batch, self.c, self.batch_id)
            v_uncond = self.flow_net(x, t_batch, self.cond_null, self.batch_id)
            # CFG combination.
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            v = self.flow_net(x, t_batch, self.c, self.batch_id)
        
        return v


class scBridgeFlow:
    """scBridge-Flow inference class."""
    
    def __init__(
        self,
        vae_path: str,
        flow_path: str,
        data_info_path: str,
        device: str = 'cuda:0'
    ):
        """
        Parameters
        ----------
        vae_path : str
            Path to VAE checkpoint.
        flow_path : str
            Path to Flow checkpoint (includes RNAEncoder and FlowNet).
        data_info_path : str
            Path to data metadata.
        device : str
            Device.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load data metadata.
        self.data_info = torch.load(data_info_path, map_location='cpu')
        self.gene_names = self.data_info['gene_names']
        self.protein_names = self.data_info['protein_names']
        self.batch_mapping = self.data_info.get('batch_mapping', None)
        
        # Load VAE.
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
        
        # Load Flow.
        flow_ckpt = torch.load(flow_path, map_location=self.device)
        flow_config = flow_ckpt['config']
        
        self.rna_encoder = RNAEncoder(
            n_genes=flow_config['n_genes'],
            dc=flow_config['dc'],
            hidden_dims=flow_config['rna_hidden_dims'],
            batch_emb_dim=flow_config['batch_emb_dim'],
            n_batches=flow_config['n_batches'],
            dropout=0.0  # Dropout is disabled during inference.
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
        Predict protein expression from RNA counts using torchdiffeq.

        Parameters
        ----------
        rna_counts : np.ndarray
            RNA counts [N, G], after HVG selection.
        batch_ids : np.ndarray
            Batch IDs [N].
        n_steps : int
            ODE integration steps (for fixed-step methods).
        cfg_scale : float
            CFG scale.
        batch_size : int
            Inference batch size.
        ode_method : str
            ODE solver method: 'dopri5', 'rk4', 'euler', 'midpoint', etc.
        rtol : float
            Relative tolerance for adaptive methods.
        atol : float
            Absolute tolerance for adaptive methods.

        Returns
        ----------
        prot_pred : np.ndarray
            Predicted protein expression [N, M] after log normalization.
        """
        # Preprocess RNA.
        rna_norm = log_normalize(rna_counts, target_sum=1e4).astype(np.float32)
        
        # Build DataLoader.
        dataset = TensorDataset(
            torch.from_numpy(rna_norm),
            torch.from_numpy(batch_ids.astype(np.int64))
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Integration time points.
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        
        all_preds = []
        
        for rna_batch, batch_id in tqdm(dataloader, desc='Predicting'):
            rna_batch = rna_batch.to(self.device)
            batch_id = batch_id.to(self.device)
            B = rna_batch.shape[0]
            
            # Encode RNA.
            c = self.rna_encoder(rna_batch, batch_id)
            
            # Sample initial noise.
            x0 = torch.randn(B, self.vae.dz, device=self.device)
            
            # Build ODE function.
            ode_func = ODEFunc(
                flow_net=self.flow_net,
                c=c,
                batch_id=batch_id,
                cfg_scale=cfg_scale,
                use_cfg=True
            )
            
            # Integrate with torchdiffeq odeint.
            # Returned shape: [len(t_span), B, dz]
            x_trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method=ode_method,
                rtol=rtol,
                atol=atol
            )
            
            # Final state at t=1.
            x = x_trajectory[-1]  # [B, dz]
            
            # Decode.
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
        Predict protein expression from an AnnData object.

        Parameters
        ----------
        adata : AnnData
            Single-cell data.
        batch_key : str
            Batch column name.
        n_steps : int
            ODE integration steps (for fixed-step methods).
        cfg_scale : float
            CFG scale.
        batch_size : int
            Inference batch size.
        ode_method : str
            ODE solver method: 'dopri5', 'rk4', 'euler', 'midpoint', etc.
        rtol : float
            Relative tolerance for adaptive methods.
        atol : float
            Absolute tolerance for adaptive methods.

        Returns
        ----------
        pred_df : pd.DataFrame
            Predicted protein expression DataFrame.
        """
        # Select HVGs.
        common_genes = [g for g in self.gene_names if g in adata.var_names]
        if len(common_genes) < len(self.gene_names):
            print(f"Warning: Only {len(common_genes)}/{len(self.gene_names)} genes found in adata")
        
        adata_hvg = adata[:, common_genes]
        
        # Extract RNA counts.
        X = adata_hvg.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        
        # Reorder if gene count does not match.
        if len(common_genes) < len(self.gene_names):
            rna_counts = np.zeros((X.shape[0], len(self.gene_names)), dtype=np.float32)
            gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
            for i, g in enumerate(common_genes):
                rna_counts[:, gene_to_idx[g]] = X[:, i]
        else:
            # Reorder to the training-time gene order.
            gene_order = [list(adata_hvg.var_names).index(g) for g in self.gene_names]
            rna_counts = X[:, gene_order].astype(np.float32)
        
        # Build batch IDs with support for string labels.
        if batch_key in adata.obs.columns:
            batch_col = adata.obs[batch_key]
            
            if pd.api.types.is_numeric_dtype(batch_col):
                batch_ids = batch_col.to_numpy(dtype=np.int64, copy=False)
            else:
                # Convert non-numeric batch labels to categorical IDs.
                if pd.api.types.is_categorical_dtype(batch_col):
                    batch_col = batch_col.astype(object).where(batch_col.notna(), 'unknown').astype(str)
                else:
                    batch_col = batch_col.fillna('unknown').astype(str)
                
                # Reuse saved batch mapping when available.
                if hasattr(self, 'batch_mapping') and self.batch_mapping is not None:
                    # Reverse mapping: batch_name -> batch_id.
                    name_to_id = {v: k for k, v in self.batch_mapping.items()}
                    batch_ids = np.array([name_to_id.get(str(b), 0) for b in batch_col], dtype=np.int64)
                    print(f"Using saved batch_mapping: {self.batch_mapping}")
                else:
                    # Otherwise create a new encoding.
                    batch_cat = pd.Categorical(batch_col)
                    batch_ids = batch_cat.codes.astype(np.int64)
                    print(f"Created new batch encoding: {dict(enumerate(batch_cat.categories.astype(str)))}")
        else:
            batch_ids = np.zeros(adata.n_obs, dtype=np.int64)
        
        # Predict.
        prot_pred = self.predict(
            rna_counts, batch_ids,
            n_steps=n_steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            ode_method=ode_method,
            rtol=rtol,
            atol=atol
        )
        
        # Build DataFrame.
        pred_df = pd.DataFrame(
            prot_pred,
            index=adata.obs_names,
            columns=self.protein_names
        )
        
        return pred_df


def main(args):
    # Set device.
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load model.
    model = scBridgeFlow(
        vae_path=args.vae_path,
        flow_path=args.flow_path,
        data_info_path=args.data_info_path,
        device=device
    )
    
    # Load data.
    print(f"\nLoading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)

    if args.subset_size is not None:
        if args.subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {args.subset_size}")
        if args.subset_size > adata.n_obs:
            raise ValueError(f"subset_size ({args.subset_size}) exceeds available cells ({adata.n_obs})")
        rng = np.random.default_rng(args.subset_seed)
        sampled = rng.choice(adata.n_obs, size=args.subset_size, replace=False)
        adata = adata[sampled].copy()
        print(f"Using subset_size={args.subset_size}, subset_seed={args.subset_seed}")

    monitor = ResourceMonitor(device=torch.device(device))
    monitor.start()
    
    # Predict.
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
    
    # Benchmark mode: record inference resource usage only.
    
    # Save predictions.
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path)
    print(f"\nPredictions saved to {output_path}")

    metrics_path = Path(args.resource_metrics_path) if args.resource_metrics_path else (output_path.parent / 'resource_metrics_infer.json')
    resource_metrics = monitor.stop()
    resource_metrics.update({
        'stage': 'inference',
        'dataset_name': Path(args.data_path).name,
        'subset_size': args.subset_size,
        'subset_seed': args.subset_seed,
        'n_steps': args.n_steps,
        'cfg_scale': args.cfg_scale,
        'ode_method': args.ode_method,
        'ode_rtol': args.ode_rtol,
        'ode_atol': args.ode_atol,
        'batch_size': args.batch_size,
        'n_cells': int(adata.n_obs),
    })
    save_metrics_json(resource_metrics, str(metrics_path))
    print(f"Inference resource metrics saved to {metrics_path}")
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scBridge-Flow Inference')
    
    # Model paths.
    parser.add_argument('--vae_path', type=str, default='./outputs/stage1/vae_best.pt',
                        help='Path to VAE model')
    parser.add_argument('--flow_path', type=str, default='./outputs/stage2/flow_best.pt',
                        help='Path to Flow model')
    parser.add_argument('--data_info_path', type=str, default='./outputs/stage1/data_info.pt',
                        help='Path to data info')
    
    # Data parameters.
    parser.add_argument('--data_path', type=str, default='./data/example.h5ad',
                        help='Path to H5AD data file for inference')
    parser.add_argument('--batch_key', type=str, default='batch_id',
                        help='Batch key in adata.obs')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of cells to sample for inference')
    parser.add_argument('--subset_seed', type=int, default=0,
                        help='Random seed for subset sampling')
    
    # Inference parameters.
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
    
    # Output.
    parser.add_argument('--output_path', type=str, default='./predictions.csv',
                        help='Output path for predictions')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--resource_metrics_path', type=str, default=None,
                        help='Path to save inference resource metrics JSON')
    
    args = parser.parse_args()
    main(args)

