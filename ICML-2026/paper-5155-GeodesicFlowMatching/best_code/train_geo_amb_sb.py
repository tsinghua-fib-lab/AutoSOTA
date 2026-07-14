"""Train geo_amb_sb model for SOTA optimization iteration 4."""
import sys, os, time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/repo')

from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.run import FlowTrainer
from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset

# Config matching the reproduction setup
ssp_cfg = {
    'bundle_type': 'hexagonal',
    'n_rotates': 13,
    'n_scales': 13, 
    'length_scale': 0.2,
    'domain_dim': 2,
    'domain_bounds': [[-1, 1], [-1, 1]],
}

tr_cfg = {
    'data_root': '/autosota_cache/data',
    'checkpoint_dir': '/autosota_cache/checkpoints',
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'val_split': 0.1,
    'noise_type': 'uniform_hypersphere',
    'target_type': 'coordinate',
    'signal_strength': 0.0,
    'sigma_min': 0.1,
    'beta_min': 0.1,
    'beta_max': 20.0,
    'ot_method': 'sinkhorn',
    'ot_reg': 0.005,
    'ot_reg_sb': 0.05,
    'device': 'cuda',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Build SSP space
enc_dim = resolve_encoded_dim(ssp_cfg)
ssp_cfg['encoded_dim'] = enc_dim
domain_bounds = np.asarray(ssp_cfg.get('domain_bounds', [[-1, 1], [-1, 1]]), dtype=np.float64)
ssp_space = build_ssp_space(ssp_cfg, domain_dim=2, domain_bounds=domain_bounds)
print(f'SSP space: dim={ssp_space.ssp_dim}')

# Ensure dataset
data_root = Path(tr_cfg['data_root'])
dataset_spec = DatasetSpec(
    data_root=data_root,
    dataset_type='coordinate_ssps',
    encoded_dim=enc_dim,
    length_scale=float(ssp_cfg['length_scale']),
    train_samples=20000,
    test_samples=5000,
    sampling_method='sobol',
    train_subdir='train',
    test_subdir='test',
)
dataset_info = ensure_target_dataset(dataset_spec, ssp_cfg, ssp_space)
print(f'Dataset: group={dataset_info["dataset_group"]}, id={dataset_info["dataset_id"]}')

# Train geo_amb_sb
print('\n=== Training geo_amb_sb ===')
arch = ResidualMLP(ssp_space.ssp_dim, flow=True, time_embed_dim=128).to(device)

trainer = FlowTrainer(
    encoded_dim=ssp_space.ssp_dim,
    architecture=arch,
    data_dir=str(dataset_info['train_dir']),
    batch_size=tr_cfg['batch_size'],
    epochs=tr_cfg['epochs'],
    lr=tr_cfg['lr'],
    weight_decay=tr_cfg['weight_decay'],
    val_split=tr_cfg['val_split'],
    noise_type=tr_cfg['noise_type'],
    target_type=tr_cfg['target_type'],
    device=device,
    sampling_mode='geo_amb_sb',
    sigma_min=tr_cfg['sigma_min'],
    beta_min=tr_cfg['beta_min'],
    beta_max=tr_cfg['beta_max'],
    use_ot_train=True,
    ot_method='sinkhorn',
    ot_reg=tr_cfg['ot_reg_sb'],
    ot_cost='angular',
)

models, train_losses, val_losses = trainer.train()

# Save checkpoint
ckpt_dir = Path(tr_cfg['checkpoint_dir']) / dataset_info['dataset_group']
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / 'drift_geo_amb_sb.pt'
torch.save(models[0].state_dict(), ckpt_path)
print(f'\nCheckpoint saved to {ckpt_path}')

# Print final losses
print(f'Final train loss: {train_losses[-1]:.6f}')
print(f'Final val loss: {val_losses[-1]:.6f}')
