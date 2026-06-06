"""
Run the full FORGE pipeline:
1. Train FCM-VAE on each site
2. Export synthetic samples
3. Run FORGE continual learning
"""
import os
import sys
import json
import numpy as np
import torch

# Add repo paths
sys.path.insert(0, '/repo/fcmvae')
sys.path.insert(0, '/repo/forge')

from fcmvae.main import Config as FCMVAEConfig, main as train_fcmvae
from forge.forge import Cfg as FORGEConfig, main as run_forge

# Configuration
os.makedirs('/repo/data/synth', exist_ok=True)

# Step 1: Train FCM-VAE on each site and generate synthetic data
for site_num in [6, 14, 15, 16]:
    mat_path = f'/repo/data/site{site_num}.mat'
    npz_path = f'/repo/data/synth/site{site_num}.npz'
    checkpoint_path = f'/repo/data/fcmvae_site{site_num}.pt'

    if os.path.exists(npz_path):
        print(f'Site {site_num}: npz already exists, skipping FCM-VAE training')
        continue

    print(f'\n{"="*60}')
    print(f'Training FCM-VAE for Site {site_num}')
    print(f'{"="*60}')

    cfg = FCMVAEConfig()
    cfg.mat_path_graph = mat_path
    cfg.checkpoint_path = checkpoint_path
    cfg.export_npz_path = npz_path
    cfg.epochs = 100
    cfg.batch_size = 8
    cfg.lr = 1e-3
    cfg.beta_kl = 2.0
    cfg.seed = 42
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Train FCM-VAE
    model = train_fcmvae(cfg)
    print(f'Site {site_num}: FCM-VAE training complete, synthetic data at {npz_path}')

# Step 2: Run FORGE
print(f'\n{"="*60}')
print('Running FORGE continual learning')
print(f'{"="*60}')

forge_cfg = FORGEConfig()
forge_cfg.PT_PATH = '/repo/data/real_sites.pt'
forge_cfg.NPZ_HOSP_PATHS = [
    f'/repo/data/synth/site{sn}.npz' for sn in [6, 14, 15, 16]
]
forge_cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
forge_cfg.SEED = 42
forge_cfg.VERBOSE = True

# Paper-specific settings
forge_cfg.EPOCHS_PER_TASK = 200
forge_cfg.WARMUP_EPOCHS = 20
forge_cfg.PATIENCE_ON_ACC = 80
forge_cfg.BATCH_SIZE = 32
forge_cfg.LR = 1e-3
forge_cfg.WEIGHT_DECAY = 2e-4
forge_cfg.HIDDEN = 128
forge_cfg.EMBED = 128
forge_cfg.LAYERS = 4
forge_cfg.DROPOUT = 0.30
forge_cfg.ALPHA = 0.10
forge_cfg.BETA = 0.40
forge_cfg.GAMMA_G = 0.30
forge_cfg.GAMMA_R = 0.00
forge_cfg.ADJ_THRESHOLD = 0.4
forge_cfg.TOT_SYNTH_CAPACITY = 256
forge_cfg.REPLAY_AFTER_FIRST = True
forge_cfg.VAL_RATIO = 0.20

results = run_forge(forge_cfg)

print(f'\n{"="*60}')
print('FORGE RESULTS:')
print(json.dumps(results['summary'], indent=2))
print(f'{"="*60}')

# Compute FOR (Forgetting Rate)
metric_matrix = results['metric_matrix']
T = len(metric_matrix)
if T > 1:
    last_row = metric_matrix[-1]
    for_values = []
    for i in range(T - 1):
        max_prev = max(metric_matrix[j][i] for j in range(i, T) if i < len(metric_matrix[j]))
        final = metric_matrix[-1][i] if i < len(last_row) else 0
        for_values.append(max_prev - final)
    FOR = np.mean(for_values) if for_values else 0.0
else:
    FOR = 0.0

print(f'\nAAA = {results["summary"]["aaa"]:.4f}')
print(f'FOR = {FOR:.4f}')
print(f'Last row accuracy: {results["summary"]["last_row"]}')
