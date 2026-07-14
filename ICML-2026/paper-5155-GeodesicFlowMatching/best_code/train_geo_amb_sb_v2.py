"""Train geo_amb_sb with LR scheduler for SOTA optimization iteration 6."""
import sys, os, time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/repo')

from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.cleanup_methods import FlowMatching
from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset
from utils.ot_pairs import angular_ot_pairs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Config
ssp_cfg = {
    'bundle_type': 'hexagonal', 'n_rotates': 13, 'n_scales': 13,
    'length_scale': 0.2, 'domain_dim': 2, 'domain_bounds': [[-1, 1], [-1, 1]],
}
enc_dim = resolve_encoded_dim(ssp_cfg)
ssp_cfg['encoded_dim'] = enc_dim
domain_bounds = np.asarray(ssp_cfg.get('domain_bounds', [[-1, 1], [-1, 1]]), dtype=np.float64)
ssp_space = build_ssp_space(ssp_cfg, domain_dim=2, domain_bounds=domain_bounds)
print(f'SSP space: dim={ssp_space.ssp_dim}')

data_root = Path('/autosota_cache/data')
dataset_spec = DatasetSpec(
    data_root=data_root, dataset_type='coordinate_ssps', encoded_dim=enc_dim,
    length_scale=float(ssp_cfg['length_scale']), train_samples=20000, test_samples=5000,
    sampling_method='sobol', train_subdir='train', test_subdir='test',
)
dataset_info = ensure_target_dataset(dataset_spec, ssp_cfg, ssp_space)
print(f'Dataset: group={dataset_info["dataset_group"]}')

# Build model
arch = ResidualMLP(ssp_space.ssp_dim, flow=True, time_embed_dim=128).to(device)
print(f'Model params: {sum(p.numel() for p in arch.parameters()):,}')

# Flow matching wrapper
fm = FlowMatching(model=arch, sampling='geo_amb_sb', device=device, sigma_min=0.1)

# Dataset
ds = SSPDataset(
    data_dir=str(dataset_info['train_dir']), ssp_dim=ssp_space.ssp_dim,
    target_type='coordinate', noise_type='uniform_hypersphere', signal_strength=0.0,
)
train_ds, val_ds = ds.split_dataset(0.1)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

# Optimizer and scheduler
optimizer = torch.optim.Adam(arch.parameters(), lr=1e-4, weight_decay=1e-4)
warmup_epochs = 5
total_epochs = 100
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

criterion = nn.CosineEmbeddingLoss()

# OT helper
def renorm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# Training loop
print(f'\n=== Training geo_amb_sb with LR scheduler ===')
best_val = float('inf')
for epoch in range(total_epochs):
    arch.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        z0_all = batch[0].squeeze(1).to(device)
        z1_all = batch[1].squeeze(1).to(device)
        
        # Sinkhorn OT pairing
        z0n, z1n = renorm(z0_all), renorm(z1_all)
        jdx = angular_ot_pairs(z0n, z1n, reg=0.05, squared=True, hard=False)
        z0, z1 = z0_all, z1_all[jdx]
        
        # FM targets
        z_t, t, u_true = fm.get_train_tuple(z0, z1)
        u_pred = arch(z_t, t)
        
        # Project to tangent for geodesic mode
        phi = renorm(z_t)
        u_pred = u_pred - (u_pred * phi).sum(dim=-1, keepdim=True) * phi
        
        loss = criterion(u_pred, u_true, torch.ones(u_pred.size(0), device=device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    arch.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            z0_all = batch[0].squeeze(1).to(device)
            z1_all = batch[1].squeeze(1).to(device)
            z0n, z1n = renorm(z0_all), renorm(z1_all)
            jdx = angular_ot_pairs(z0n, z1n, reg=0.05, squared=True, hard=False)
            z0, z1 = z0_all, z1_all[jdx]
            z_t, t, u_true = fm.get_train_tuple(z0, z1)
            u_pred = arch(z_t, t)
            phi = renorm(z_t)
            u_pred = u_pred - (u_pred * phi).sum(dim=-1, keepdim=True) * phi
            loss = criterion(u_pred, u_true, torch.ones(u_pred.size(0), device=device))
            val_loss += loss.item()
    
    scheduler.step()
    
    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}/{total_epochs}: train={avg_train:.4e}, val={avg_val:.4e}, lr={lr:.2e}')
    
    if avg_val < best_val:
        best_val = avg_val

# Save checkpoint
ckpt_dir = Path('/autosota_cache/checkpoints') / dataset_info['dataset_group']
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / 'drift_geo_amb_sb.pt'
torch.save(arch.state_dict(), ckpt_path)
print(f'\nCheckpoint saved to {ckpt_path}')
print(f'Best val loss: {best_val:.6f}')
