import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial
import math
import pickle
from tqdm import tqdm
import re
import types

from .hypersphere import Hypersphere
from .scheduler_lib import Geometric
from . import distribution
from .sde import LogBridge_Mixture, ContinuousSDE
from .model.dit import MixeDiT
from .model.ema import ExponentialMovingAverage
from . import losses
from .model import utils as mutils
from .utils import seq_utils as sutils

# ==========================================
# Configuration
# ==========================================
class Config:
    def __init__(self):
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.work_dir = "./output"
        
        # Data
        self.data_path = "./training_data/mixedit/data/tokenized_uav.npy"
        self.obs_path = "./training_data/mixedit/data/uav_observations.npy"
        self.vocab_size = 144 #512
        self.seq_len = 1  # As requested
        self.val_split = 0.2
        
        # Training
        self.batch_size = 256 #512 #128 #64 # Memory
        self.lr = 3e-4 #1e-4
        self.n_epochs = 1500 #2500 #1000
        self.log_freq = 100
        self.save_freq = 5 # Epochs
        self.ema_decay = 0.9999
        self.grad_clip = 1.0
        self.loss = "ce"#"focal"#"ce"
        
        # Model (DiT)
        self.hidden_size = 768 #512
        self.n_heads = 12 #8
        self.n_blocks = 12 #6
        self.dropout = 0.0
        self.cond_dim = 128
        self.continuous_dim = 7 # input dim for contin. params
        
        # SDE / Scheduler
        # We assume base 512 for length 1, so tokens=512
        self.tokens = 144 #512 
        self.beta_0 = 0.001
        self.beta_f = 2e-1
        self.step_thr = 0.0 # Threshold for mixture path
        self.preprocess_steps = 2**14 # For SDE moments
        
        # Force add mask token for Mixture Path (512 -> 513 dims)
        self.add_mask_token = True

# ==========================================
# Custom Dataset
# ==========================================
class NumpyDataset(Dataset):
    def __init__(self, npy_file, npy_obs_file):
        # if not os.path.exists(npy_file):
        #     print(f"File {npy_file} not found. Generating dummy data...")
        #     # Generate dummy data: N=1000, 1 column, values 0-511
        #     dummy = np.random.randint(0, 512, size=(1000, 1))
        #     np.save(npy_file, dummy)
            
        self.data = torch.from_numpy(np.load(npy_file)).long()
        self.data, _ = compress_decimal_representation(self.data)
        self.obs = torch.from_numpy(np.load(npy_obs_file, allow_pickle=True).item()["norm_data"])
        print(f"Compressed representation with max token: {max(self.data)}")
        # Ensure shape is [N, 1]
        if len(self.data.shape) == 1:
            self.data = self.data.unsqueeze(-1)
        if len(self.obs.shape) == 3:
            self.obs = self.obs.squeeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return dict to match existing pipeline logic
        return {'input_ids_disc': self.data[idx], 'input_cont': self.obs[idx]}

def compress_to_decimal(dataset):
    """
    Compresses a (B, D) binary dataset by:
    1. Finding and removing columns where all values are 1.
    2. Converting the remaining (B, D_filtered) binary rows to decimal.
    
    Returns:
    - decimals_bx1 (np.array): (B, 1) array of decimal values.
    - all_ones_indices (np.array): 1D array of indices of removed columns.
    - original_D (int): The original dimension D, needed for decompression.
    """
    # Ensure input is a numpy array
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)
        
    B, original_D = dataset.shape
    
    # 1. Find columns to remove (where all values are 1)
    remove_mask = dataset.all(axis=0)
    all_ones_indices = np.where(remove_mask)[0]
    
    # 2. Create the filtered dataset (keeping columns that are NOT all 1s)
    keep_mask = ~remove_mask
    filtered_dataset = dataset[:, keep_mask]
    
    D_filtered = filtered_dataset.shape[1]
    
    # 3. Convert the filtered (B, D_filtered) array to decimal
    if D_filtered == 0:
        # Edge case: All columns were 1s.
        # Decimal representation is 0, and array has shape (B, 1).
        decimals_bx1 = np.zeros((B, 1), dtype=int)
    else:
        # Create the powers-of-2 multiplier: [2**(D_filtered-1), ..., 2**1, 2**0]
        powers = np.arange(D_filtered)[::-1]
        multipliers = 2 ** powers
        
        # Perform dot product to get decimal values
        # (B, D_filtered) @ (D_filtered,) -> (B,)
        decimals = filtered_dataset @ multipliers
        
        # Reshape to (B, 1)
        decimals_bx1 = decimals[:, np.newaxis]
        
    return decimals_bx1, all_ones_indices, original_D

def decompress_from_decimal(decimals_bx1, all_ones_indices, original_D):
    """
    Decompresses a (B, 1) decimal array back into a (B, D) binary array,
    re-inserting the "all 1s" columns at their original positions.
    
    Args:
    - decimals_bx1 (np.array): (B, 1) array of decimal values.
    - all_ones_indices (np.array): 1D array of indices that were removed.
    - original_D (int): The original D dimension of the dataset.
    
    Returns:
    - reconstructed (np.array): The reconstructed (B, D) binary array.
    """
    B = decimals_bx1.shape[0]
    
    # Calculate how many columns the binary data should have
    D_filtered = original_D - len(all_ones_indices)
    
    # Create the output array, initializing with 0s
    reconstructed = np.zeros((B, original_D), dtype=int)
    
    # Step 1: Add back the "all 1s" columns.
    # This is safe even if all_ones_indices is empty.
    reconstructed[:, all_ones_indices] = 1
    
    # Step 2: Add back the filtered data, if any.
    if D_filtered > 0:
        # Squeeze decimals from (B, 1) to (B,)
        decimals = decimals_bx1.ravel() 
        
        # Convert (B,) decimals to (B, D_filtered) binary array
        # m = [2**(D_filtered-1), ..., 2**1, 2**0]
        m = 2 ** np.arange(D_filtered)[::-1]
        
        # This uses numpy broadcasting to convert all rows at once
        # (x // 2**k) % 2 is a standard way to get the k-th bit
        filtered_binary = (decimals[:, np.newaxis] // m) % 2
        
        # Get the mask for columns that were *not* all 1s
        keep_mask = np.ones(original_D, dtype=bool)
        keep_mask[all_ones_indices] = False
        
        # Place the binary data into the correct columns
        reconstructed[:, keep_mask] = filtered_binary
    
    # If D_filtered was 0, we just return the array filled
    # with 1s, which is correct.
    return reconstructed

def compress_decimal_representation(dec_data):
    h, c = np.unique(dec_data, return_counts=True)
    tokenizer = {k:v for k,v in zip(h, range(len(h)))}
    return np.array([tokenizer[int(v[0])] for v in dec_data]).reshape((-1,1)), tokenizer


# ==========================================
# Training Logic
# ==========================================
def main():
    # 1. Initialize Default Config
    default_cfg = Config()
    n_epochs = default_cfg.n_epochs

    work_dir = default_cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # -----------------------------------------------------------------------------
    # STEP 1: Check for Checkpoint & Load Config BEFORE Model Init
    # -----------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    train_list = []
    val_list = []
    val_d_list = []
    checkpoint = None
    
    checkpoint_files = [f for f in os.listdir(work_dir) if f.startswith("checkpoint_ep") and f.endswith(".pth")]
    
    if checkpoint_files:
        epochs = []
        for f in checkpoint_files:
            match = re.search(r"checkpoint_ep(\d+).pth", f)
            if match:
                epochs.append(int(match.group(1)))
        
        if epochs:
            max_epoch = max(epochs)
            resume_path = os.path.join(work_dir, f"checkpoint_ep{max_epoch}.pth")
            
            if os.path.exists(resume_path):
                print(f"Found existing checkpoint: {resume_path}")
                print(f"Loading configuration from checkpoint...")
                
                # Load checkpoint to CPU first to get config
                checkpoint = torch.load(resume_path, map_location=default_cfg.device)
                
                # OVERWRITE cfg with the one from the checkpoint
                # We use SimpleNamespace to convert the dict back to an object with .attributes
                loaded_config_dict = checkpoint['config']
                cfg = types.SimpleNamespace(**loaded_config_dict)
                
                # Ensure work_dir is set correctly (in case paths changed, though usually we trust the script)
                cfg.work_dir = work_dir 
                cfg.device = default_cfg.device # Ensure device is correct for current run
                
                start_epoch = checkpoint.get('epoch', max_epoch)
                train_list = checkpoint.get('train_list', [])
                val_list = checkpoint.get('val_list', [])
                val_d_list = checkpoint.get('val_d_list', [])
                best_val_loss = float('inf') if val_list == 0 else min(val_list) #checkpoint.get('best_val_loss', float('inf'))
                
                print(f"   Resuming from Epoch {start_epoch}")
            else:
                 cfg = default_cfg
                 cfg.work_dir = work_dir
        else:
             cfg = default_cfg
             cfg.work_dir = work_dir
    else:
        cfg = default_cfg
        cfg.work_dir = work_dir

    if not hasattr(cfg, 'loss'):
        cfg.loss = "ce"


    # -----------------------------------------------------------------------------
    # STEP 2: Initialize Everything using the Correct (Loaded or Default) Config
    # -----------------------------------------------------------------------------
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Running on {cfg.device}")
    print(f"Config: hidden_size={cfg.hidden_size}, lr={cfg.lr}, loss={cfg.loss}")

    # Data Loading
    full_dataset = NumpyDataset(cfg.data_path, cfg.obs_path)
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # SDE Setup
    real_vocab_size = cfg.tokens + (1 if cfg.add_mask_token else 0)
    manifold = Hypersphere(real_vocab_size - 1)
    scheduler = Geometric(beta_0=cfg.beta_0, beta_f=cfg.beta_f, weight_type='step')
    batch_dims = (cfg.batch_size, cfg.seq_len, real_vocab_size)
    
    prior_dist = distribution.Mixture(batch_dims=batch_dims, device=cfg.device)
    
    print("Initializing SDE...")
    sde_disc = LogBridge_Mixture(
        manifold=manifold,
        scheduler=scheduler,
        prior_dist=prior_dist,
        device=cfg.device,
        preprocess_steps=cfg.preprocess_steps,
        mix_type="step",
        step_thr=cfg.step_thr,
        dims=batch_dims
    )
    
    # Model Setup (Using cfg parameters which might be from checkpoint)
    model = MixeDiT(
        input_dim=real_vocab_size,
        continuous_dim=cfg.continuous_dim, # cont input dim
        output_dim=real_vocab_size,
        hidden_size=cfg.hidden_size,
        n_heads=cfg.n_heads,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
        n_blocks=cfg.n_blocks,
        length=cfg.seq_len
    ).to(cfg.device)

    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # -----------------------------------------------------------------------------
    # STEP 3: Load State Dicts if Resuming
    # -----------------------------------------------------------------------------
    if checkpoint is not None:
        print("   Loading model weights and optimizer state...")
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Clean up memory
        del checkpoint
        torch.cuda.empty_cache()
    
    sde_cont = ContinuousSDE(schedule="vp")

    # Initialize Loss
    train_loss_calc = losses.mixed_loss_fn(sde_disc, sde_cont, train=True)
    val_loss_calc = losses.mixed_loss_fn(sde_disc, sde_cont, train=False)

    # -----------------------------------------------------------------------------
    # STEP 4: Training Loop
    # -----------------------------------------------------------------------------
    max_epochs = max(n_epochs,cfg.n_epochs)
    print(f"Starting training from epoch {start_epoch+1} to {max_epochs}...")
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss_sum = 0
        train_loss_d_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for batch in pbar:
            inputs_disc = batch['input_ids_disc'].to(cfg.device) # [B, 1]
            inputs_cont = batch['input_cont'].to(cfg.device) # [B, D]
            
            # Prepare inputs: One-hot encoding
            # Map indices (0-511) to OneHot (Size 513, last one is mask)
            # The data only has 0-511, so the 512th index (mask) is 0
            inputs_oh = F.one_hot(inputs_disc, num_classes=real_vocab_size).float()
            
            optimizer.zero_grad()
            
            # Compute Loss
            loss, loss_d, loss_c = train_loss_calc(model, (inputs_oh, inputs_cont))
            
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model.parameters())
            
            train_loss_sum += loss.mean().item()
            train_loss_d_sum += loss_d.mean().item()
            # global_step += 1
            pbar.set_postfix({'loss': loss.mean().item()})
            
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_loss_d = train_loss_d_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_loss_d_sum = 0
        val_loss_c_sum = 0
        with torch.no_grad():
            # Apply EMA for validation
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            
            for batch in val_loader:
                inputs_disc = batch['input_ids_disc'].to(cfg.device) # [B, 1]
                inputs_cont = batch['input_cont'].to(cfg.device) # [B, D]
                inputs_oh = F.one_hot(inputs_disc, num_classes=real_vocab_size).float()
                loss, loss_d, loss_c = val_loss_calc(model, (inputs_oh, inputs_cont))
                val_loss_sum += loss.mean().item()
                val_loss_d_sum += loss_d.mean().item()
                val_loss_c_sum += loss_c.mean().item()
            
            # Restore original params
            ema.restore(model.parameters())
            
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_loss_d = val_loss_d_sum / len(val_loader)
        avg_val_loss_c = val_loss_c_sum / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Loss Disc.: {avg_val_loss_d:.4f} | Val Loss Cont.: {avg_val_loss_c:.4f}")

        train_list.append(avg_train_loss)
        val_list.append(avg_val_loss)
        val_d_list.append(avg_val_loss_d)
        
        # Check if the current validation loss is the best seen so far
        if avg_val_loss < best_val_loss:
            # Update the best loss value
            best_val_loss = avg_val_loss

            # Define the checkpoint path (using 'best' instead of epoch number is common)
            ckpt_path = os.path.join(cfg.work_dir, "checkpoint_best.pth")
            
            # Save the model state
            torch.save({
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'config': cfg.__dict__, # Save config dict
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss, # Save the tracked best loss
                'epoch': epoch + 1, # Save the epoch number
                'train_list': train_list,
                'val_list': val_list,
                'val_d_list': val_d_list

            }, ckpt_path)
            print(f"New best validation loss ({best_val_loss:.4f}) found! Saved checkpoint to {ckpt_path}")

    ckpt_path = os.path.join(cfg.work_dir, f"checkpoint_ep{epoch+1}.pth")
    torch.save({
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'config': cfg.__dict__, # Save config dict
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss, # Save the tracked best loss
            'epoch': epoch + 1, # Save the epoch number
            'train_list': train_list,
            'val_list': val_list,
            'val_d_list': val_d_list
    }, ckpt_path)

if __name__ == "__main__":
    main()