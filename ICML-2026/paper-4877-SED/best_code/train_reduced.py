import os, sys, time, torch
sys.path.insert(0, '/repo')
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sed.data.scrna import SparseCellDataModule
from sed.models.vae.svae import SVAE
from sed.models.diffusion.diffusion import Diffusion

def rate(step, model_size, factor, warmup):
    if step == 0: step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

device = torch.device('cuda:0')

dm = SparseCellDataModule(
    train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
    batch_size=128, data_dimensions=1000, input_mode='scrna'
)
dm.setup()
train_loader = dm.train_dataloader()
print(f'Train: {len(dm.train_dataset)} cells')

# SAVAE
print('\n=== SAVAE ===')
svae = SVAE(data_dimensions=1000, num_layers=3, d_model=256, d_ff=1024,
            h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None).to(device)
opt = Adam(svae.parameters(), lr=1.0, betas=(0.9, 0.99), eps=1e-9)
sched = LambdaLR(opt, lr_lambda=lambda s: rate(s, 256, factor=1, warmup=4000))
svae.train()
svae_steps = 20000
t0 = time.time()
for step in range(1, svae_steps + 1):
    batch = next(iter(train_loader))
    in_pos, in_val = batch[0].to(device), batch[1].to(device)
    opt.zero_grad()
    ld, _, _, _ = svae.step((in_pos, in_val))
    ld['loss'].backward()
    torch.nn.utils.clip_grad_norm_(svae.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 1000 == 0:
        print(f'SAVAE {step}/{svae_steps} loss={ld["loss"].item():.3f} ({step/(time.time()-t0):.1f}st/s)', flush=True)
print(f'SAVAE done: {(time.time()-t0)/60:.0f}min')
torch.save(svae.state_dict(), '/repo/svae_output/svae_20k.pth')

# SED
print('\n=== SED ===')
svae.eval()
svae.requires_grad_(False)
diffusion = Diffusion(
    unet_config={'hidden_dim': [512, 512, 256, 128], 'dropout': 0.1},
    image_size=256, timesteps=1000, use_ddim=False, noise_schedule='cosine'
).to(device)
opt_d = Adam(diffusion.unet_model.parameters(), lr=1e-4, betas=(0.9, 0.99))
sed_steps = 100000
t0 = time.time()
for step in range(1, sed_steps + 1):
    batch = next(iter(train_loader))
    in_pos, in_val = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        _, mu, _, _, _ = svae(in_pos, in_val)
    mu_norm = (mu * 2) - 1
    pred = diffusion(mu_norm)
    loss = torch.nn.functional.mse_loss(pred, mu_norm)
    opt_d.zero_grad(); loss.backward(); opt_d.step()
    if step % 5000 == 0:
        print(f'SED {step}/{sed_steps} loss={loss.item():.6f} ({step/(time.time()-t0):.1f}st/s)', flush=True)
print(f'SED done: {(time.time()-t0)/60:.0f}min')
torch.save({'svae_state_dict': svae.state_dict(), 'diffusion_state_dict': diffusion.state_dict()}, '/repo/sed_output/sed_100k.pth')
print('\nTraining complete!')
