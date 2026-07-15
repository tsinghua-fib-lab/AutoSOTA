#!/usr/bin/env python3
"""Continue SED training from existing checkpoint: 100K -> 200K additional steps."""
import os, sys, time, math
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, '/repo')
from sed.data.scrna import SparseCellDataModule
from sed.models.vae.svae import SVAE
from sed.models.diffusion.diffusion import Diffusion

def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

device = torch.device('cuda:0')

dm = SparseCellDataModule(
    train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
    batch_size=256, data_dimensions=1000, input_mode='scrna'
)
dm.setup()
train_iter = cycle_loader(dm.train_dataloader())
print(f'Train: {len(dm.train_dataset)} cells, batch=256')

# Load SVAE
print('Loading SVAE...')
svae = SVAE(data_dimensions=1000, num_layers=3, d_model=256, d_ff=1024,
            h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None).to(device)
svae.load_state_dict(torch.load('/repo/svae_output/svae_20k.pth', map_location=device))
svae.eval()
svae.requires_grad_(False)

# Load SED and continue training
print('Loading SED...')
diffusion = Diffusion(
    unet_config={'hidden_dim': [512, 512, 256, 128], 'dropout': 0.1},
    image_size=256, timesteps=1000, use_ddim=False, noise_schedule='cosine'
).to(device)
sed_state = torch.load('/repo/sed_output/sed_100k.pth', map_location=device)
diffusion.unet_model.load_state_dict(
    {k.replace('unet_model.', ''): v for k, v in sed_state['diffusion_state_dict'].items() 
     if k.startswith('unet_model.')}
)
print(f'  Loaded from /repo/sed_output/sed_100k.pth')

additional_steps = 200000
BASE_LR = 1e-4
opt = Adam(diffusion.unet_model.parameters(), lr=BASE_LR, betas=(0.9, 0.99))
os.makedirs('/repo/sed_output', exist_ok=True)

t0 = time.time()
for step in range(1, additional_steps + 1):
    batch = next(train_iter)
    in_pos, in_val = batch[0].to(device), batch[1].to(device)
    
    with torch.no_grad():
        _, mu, _, _, _ = svae(in_pos, in_val)
    
    mu_norm = (mu * 2) - 1
    pred = diffusion(mu_norm)
    loss = nn.functional.mse_loss(pred, mu_norm)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.unet_model.parameters(), 1.0)
    opt.step()
    
    total_step = 100000 + step
    if step % 500 == 0:
        elapsed = time.time() - t0
        sps = step / elapsed
        eta = (additional_steps - step) / sps / 3600
        print(f'SED Step {total_step}/300000 | Loss: {loss.item():.6f} | '
              f'{sps:.1f} st/s | ETA: {eta:.1f}h', flush=True)
    
    if step % 50000 == 0:
        ckpt_path = f'/repo/sed_output/sed_cont_{total_step}.pth'
        torch.save({
            'svae_state_dict': svae.state_dict(),
            'diffusion_state_dict': diffusion.state_dict(),
            'step': total_step,
        }, ckpt_path)
        print(f'  Checkpoint: {ckpt_path}')

final_ckpt = '/repo/sed_output/sed_300k.pth'
torch.save({
    'svae_state_dict': svae.state_dict(),
    'diffusion_state_dict': diffusion.state_dict(),
    'step': 300000,
}, final_ckpt)
print(f'SED continued training done in {(time.time()-t0)/3600:.1f}h')
print(f'Final checkpoint: {final_ckpt}')
