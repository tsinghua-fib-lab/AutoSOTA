import os, sys, time, math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import lightning as L

sys.path.insert(0, '/repo')
from sed.data.scrna import SparseCellDataModule
from sed.models.vae.svae import SVAE

def rate(step, model_size, factor, warmup):
    if step == 0: step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def main():
    device = torch.device('cuda:0')
    
    # Data
    print('Loading data...')
    dm = SparseCellDataModule(
        train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
        batch_size=128, data_dimensions=1000, input_mode='scrna'
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    print(f'Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}')
    
    # Model
    print('Creating model...')
    model = SVAE(
        data_dimensions=1000, num_layers=3, d_model=256, d_ff=1024,
        h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.2f}M')
    
    # Optimizer + scheduler
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.99), eps=1e-9)
    scheduler = LambdaLR(optimizer, 
        lr_lambda=lambda step: rate(step, 256, factor=1, warmup=4000))
    
    # EMA
    ema_decay = 0.9999
    ema_model = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    # Training
    max_steps = 100000
    save_every = 1000
    output_dir = '/repo/svae_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Training for {max_steps} steps...')
    model.train()
    global_step = 0
    epoch = 0
    t_start = time.time()
    
    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            in_pos, in_val = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            loss_dict, _, _, _ = model.step((in_pos, in_val))
            loss = loss_dict['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # EMA update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ema_model[name] = ema_decay * ema_model[name] + (1 - ema_decay) * param
            
            global_step += 1
            
            if global_step % 100 == 0:
                elapsed = time.time() - t_start
                steps_per_sec = global_step / elapsed
                remaining = (max_steps - global_step) / steps_per_sec / 3600
                print(f'Step {global_step}/{max_steps} | Loss: {loss.item():.4f} | '
                      f'CE: {loss_dict["CE"].item():.4f} | MSE: {loss_dict["MSE"].item():.4f} | '
                      f'KLD: {loss_dict["KLD"].item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | '
                      f'{steps_per_sec:.1f} st/s | ETA: {remaining:.1f}h', flush=True)
            
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f'svae_step{global_step}.ckpt')
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'ema_model': ema_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, ckpt_path)
                print(f'Checkpoint saved: {ckpt_path}', flush=True)
            
            if global_step >= max_steps:
                break
    
    # Save final checkpoint
    final_path = os.path.join(output_dir, 'last.ckpt')
    torch.save({
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'ema_model': ema_model,
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    # Also save in Lightning format for compatibility
    model.to('cpu')
    ckpt_data = {
        'state_dict': model.state_dict(),
        'hyper_parameters': {
            'data_dimensions': 1000, 'num_layers': 3, 'd_model': 256,
            'd_ff': 1024, 'h': 4, 'dropout': 0.1, 'beta': 1e-6,
            'input_mode': 'scrna', 'lr': None
        }
    }
    torch.save(ckpt_data, os.path.join(output_dir, 'last_lightning.ckpt'))
    
    elapsed = time.time() - t_start
    print(f'Training complete! Total time: {elapsed/3600:.1f}h, {elapsed/global_step:.3f}s/step')
    print(f'Final checkpoint: {final_path}')

if __name__ == '__main__':
    main()
