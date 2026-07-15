#!/usr/bin/env python3
"""Enhanced training pipeline: full budget + gradient clipping + LR warmup/cosine decay for SED."""
import os, sys, time, math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, LinearLR

sys.path.insert(0, '/repo')
from sed.data.scrna import SparseCellDataModule
from sed.models.vae.svae import SVAE
from sed.models.diffusion.diffusion import Diffusion

def rate(step, model_size, factor, warmup):
    if step == 0: step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def cycle_loader(loader):
    """Create an infinite iterator from a DataLoader."""
    while True:
        for batch in loader:
            yield batch

device = torch.device('cuda:0')
BATCH_SIZE = 256  # Larger batch for better gradient estimates (CODE-4)

# ---- Data ----
dm = SparseCellDataModule(
    train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
    batch_size=BATCH_SIZE, data_dimensions=1000, input_mode='scrna'
)
dm.setup()
train_iter = cycle_loader(dm.train_dataloader())
print(f'Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}')

# ---- Stage 1: SAVAE (100K steps) ----
print('=' * 60)
print('STAGE 1: SAVAE Training (100K steps)')
print('=' * 60)

svae = SVAE(
    data_dimensions=1000, num_layers=3, d_model=256, d_ff=1024,
    h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None
).to(device)
print(f'SAVAE params: {sum(p.numel() for p in svae.parameters())/1e6:.2f}M')

optimizer = Adam(svae.parameters(), lr=1.0, betas=(0.9, 0.99), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda s: rate(s, 256, factor=1, warmup=4000))

svae_steps = 100000  # Paper budget
os.makedirs('/repo/svae_output', exist_ok=True)
svae.train()
t0 = time.time()

for step in range(1, svae_steps + 1):
    batch = next(train_iter)
    in_pos, in_val = batch[0].to(device), batch[1].to(device)
    
    optimizer.zero_grad()
    loss_dict, _, _, _ = svae.step((in_pos, in_val))
    loss_dict['loss'].backward()
    torch.nn.utils.clip_grad_norm_(svae.parameters(), 1.0)  # CODE-3
    optimizer.step()
    scheduler.step()
    
    if step % 500 == 0:
        elapsed = time.time() - t0
        sps = step / elapsed
        eta = (svae_steps - step) / sps / 3600
        print(f'SAVAE Step {step}/{svae_steps} | Loss: {loss_dict["loss"].item():.4f} | '
              f'{sps:.1f} st/s | ETA: {eta:.1f}h', flush=True)
    
    if step % 10000 == 0:
        ckpt_path = f'/repo/svae_output/svae_step{step}.pth'
        torch.save(svae.state_dict(), ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

torch.save(svae.state_dict(), '/repo/svae_output/svae_100k.pth')
print(f'SAVAE done in {(time.time()-t0)/3600:.1f}h')

# ---- Stage 2: SED Training (500K steps) ----
print('=' * 60)
print('STAGE 2: SED Training (500K steps)')
print('=' * 60)

svae.eval()
svae.requires_grad_(False)

latent_dim = 256
diffusion = Diffusion(
    unet_config={'hidden_dim': [512, 512, 256, 128], 'dropout': 0.1, 'input_dim': latent_dim},
    image_size=latent_dim, timesteps=1000, use_ddim=False,
    noise_schedule='cosine'
).to(device)
print(f'UNet params: {sum(p.numel() for p in diffusion.unet_model.parameters())/1e6:.2f}M')

# CODE-2: LR warmup + cosine decay
BASE_LR = 1e-4
WARMUP_STEPS = 25000  # 5% of 500K
sed_steps = 500000

opt = Adam(diffusion.unet_model.parameters(), lr=BASE_LR, betas=(0.9, 0.99))
warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=WARMUP_STEPS)
cosine_scheduler = CosineAnnealingLR(opt, T_max=sed_steps - WARMUP_STEPS, eta_min=BASE_LR/10)
scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])

os.makedirs('/repo/sed_output', exist_ok=True)
t0 = time.time()

for step in range(1, sed_steps + 1):
    batch = next(train_iter)
    in_pos, in_val = batch[0].to(device), batch[1].to(device)
    
    with torch.no_grad():
        _, mu, _, _, _ = svae(in_pos, in_val)
    
    mu_norm = (mu * 2) - 1
    pred = diffusion(mu_norm)
    loss = nn.functional.mse_loss(pred, mu_norm)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.unet_model.parameters(), 1.0)  # CODE-3
    opt.step()
    scheduler.step()
    
    if step % 500 == 0:
        elapsed = time.time() - t0
        sps = step / elapsed
        eta = (sed_steps - step) / sps / 3600
        current_lr = opt.param_groups[0]['lr']
        print(f'SED Step {step}/{sed_steps} | Loss: {loss.item():.6f} | LR: {current_lr:.2e} | '
              f'{sps:.1f} st/s | ETA: {eta:.1f}h', flush=True)
    
    if step % 50000 == 0:
        ckpt_path = f'/repo/sed_output/sed_step{step}.pth'
        torch.save({
            'svae_state_dict': svae.state_dict(),
            'diffusion_state_dict': diffusion.state_dict(),
            'step': step,
        }, ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

torch.save({
    'svae_state_dict': svae.state_dict(),
    'diffusion_state_dict': diffusion.state_dict(),
    'step': sed_steps,
}, '/repo/sed_output/sed_500k.pth')
print(f'SED done in {(time.time()-t0)/3600:.1f}h')
print('Training complete!')
