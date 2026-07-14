import logging
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from utils import ess, plot_bias_analysis, sample_categorical_logits
from utils_ising import ising2d_mag

# Import kB for temperature conversion
try:
    import ase.units
    K_B = ase.units.kB  # eV/K
except ImportError:
    K_B = 8.617333262e-5  # eV/K (fallback)



class ReplayBuffer:
    """Experience Replay Buffer for MDNS.
    
    Stores samples (x, beta, h) and allows sampling mixed batches.
    Designed to be stored on CPU to save VRAM, with on-demand move to GPU.
    Supports N-dimensional CV binning via flattening.
    """
    def __init__(self, buffer_size, x_shape, device='cpu', dtype=torch.float32, 
                 cv_min=None, cv_max=None, n_bins=1, strategy='fifo'):
        self.buffer_size = buffer_size
        self.device = device  # Device where samples are returned (usually GPU)
        self.storage_device = 'cpu' # Store on CPU to save VRAM
        self.strategy = strategy
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors on CPU
        # x_shape is (D,)
        self.x = torch.zeros((buffer_size, *x_shape), dtype=torch.long, device=self.storage_device)
        self.beta = torch.zeros(buffer_size, dtype=dtype, device=self.storage_device)
        self.h = torch.zeros(buffer_size, dtype=dtype, device=self.storage_device)
        
        # CV storage: [buffer_size, cv_dim]
        # We need to determine cv_dim from inputs
        
        # Parse CV Binning setup
        def ensure_tensor(val, default):
            if val is None: return torch.tensor([default], dtype=dtype, device=self.storage_device)
            if isinstance(val, (int, float)): return torch.tensor([val], dtype=dtype, device=self.storage_device)
            if isinstance(val, list): return torch.tensor(val, dtype=dtype, device=self.storage_device)
            if isinstance(val, torch.Tensor): return val.to(dtype=dtype, device=self.storage_device)
            return torch.tensor([default], dtype=dtype, device=self.storage_device)
            
        self.cv_min = ensure_tensor(cv_min, -1.0)
        self.cv_max = ensure_tensor(cv_max, 1.0)
        
        # Handle n_bins (can be int or list of ints)
        if isinstance(n_bins, int):
            self.n_bins_per_dim = torch.tensor([n_bins] * self.cv_min.shape[0], dtype=torch.long, device=self.storage_device)
        elif isinstance(n_bins, list):
            self.n_bins_per_dim = torch.tensor(n_bins, dtype=torch.long, device=self.storage_device)
        elif isinstance(n_bins, torch.Tensor):
            self.n_bins_per_dim = n_bins.to(dtype=torch.long, device=self.storage_device)
        else:
            self.n_bins_per_dim = torch.tensor([1] * self.cv_min.shape[0], dtype=torch.long, device=self.storage_device)
            
        self.cv_dim = self.cv_min.shape[0]
        
        # Ensure consistency
        if self.cv_max.shape[0] != self.cv_dim:
             raise ValueError(f"cv_min has dim {self.cv_dim} but cv_max has dim {self.cv_max.shape[0]}")
        if self.n_bins_per_dim.shape[0] != self.cv_dim:
             # Try to broadcast scalar n_bins
             if self.n_bins_per_dim.shape[0] == 1:
                  self.n_bins_per_dim = self.n_bins_per_dim.repeat(self.cv_dim)
             else:
                  raise ValueError(f"cv_min has dim {self.cv_dim} but n_bins has dim {self.n_bins_per_dim.shape[0]}")

        # Compute bin widths
        denom = (self.cv_max - self.cv_min)
        self.bin_width = torch.where(denom > 1e-6, denom / self.n_bins_per_dim.float(), torch.ones_like(denom))
        
        # Compute total bins (product of bins per dim)
        self.total_bins = self.n_bins_per_dim.prod().item()
        
        # Strides for flattening: [N2*N3*..., N3*..., ..., 1]
        self.bin_strides = torch.ones(self.cv_dim, dtype=torch.long, device=self.storage_device)
        cum_prod = 1
        for i in range(self.cv_dim - 1, -1, -1):
            self.bin_strides[i] = cum_prod
            cum_prod *= self.n_bins_per_dim[i].item()
            
        # CV Storage
        self.cv = torch.zeros((buffer_size, self.cv_dim), dtype=dtype, device=self.storage_device)
        self.has_conditions = False 
        
        if self.strategy == 'balanced':
            # Partitioned storage: each bin gets a fixed slice of the buffer
            self.max_per_bin = buffer_size // self.total_bins
            if self.max_per_bin == 0:
                raise ValueError(f"Buffer size {buffer_size} too small for {self.total_bins} bins.")
            
            # Pointers and counts for each bin
            self.bin_ptrs = torch.zeros(self.total_bins, dtype=torch.long, device=self.storage_device)
            self.bin_counts = torch.zeros(self.total_bins, dtype=torch.long, device=self.storage_device)
            self.active_bins = [] # For fast sampling
            
            # Warn if buffer_size is not perfectly divisible
            if buffer_size % self.total_bins != 0:
                 print(f"Warning: Buffer size {buffer_size} not divisible by {self.total_bins}. Using {self.max_per_bin * self.total_bins} slots.")
        else:
            # FIFO strategy
            # Bin storage: list of lists of indices
            self.bins = [[] for _ in range(self.total_bins)]
            
            # Index tracking for O(1) removal
            self.sample_bins = torch.full((buffer_size,), -1, dtype=torch.long, device=self.storage_device)
            self.bin_pos = torch.full((buffer_size,), -1, dtype=torch.long, device=self.storage_device)

    def _get_bin_indices(self, cv):
        """Compute flattened bin indices for given CVs."""
        # cv: [B, D] or [B]
        if cv.ndim == 1 and self.cv_dim == 1:
            cv = cv.unsqueeze(-1)
        elif cv.ndim == 1 and self.cv_dim > 1:
            # Should not happen if correctly passed, but maybe handle error
            raise ValueError(f"CV has dim 1 but buffer expects {self.cv_dim}")
            
        # [B, D]
        bin_indices_per_dim = ((cv - self.cv_min) / self.bin_width).floor().long()
        
        # Clamp each dimension
        for d in range(self.cv_dim):
            bin_indices_per_dim[:, d] = bin_indices_per_dim[:, d].clamp(0, self.n_bins_per_dim[d] - 1)
            
        # Flatten: sum(idx_d * stride_d)
        flat_indices = (bin_indices_per_dim * self.bin_strides).sum(dim=1)
        return flat_indices

    def add(self, x, beta=None, h=None, cv=None):
        """Add a batch of samples to the buffer."""
        # Clean inputs and move to storage device
        x = x.to(self.storage_device)
        
        if self.strategy == 'balanced':
            self._add_balanced(x, beta, h, cv)
        else:
            self._add_fifo(x, beta, h, cv)
            
    def _add_balanced(self, x, beta, h, cv):
        if cv is None:
             # If balanced mode but no CV, dump to bin 0 (fallback)
             # Need vector of zeros
             bin_indices = torch.zeros(x.shape[0], dtype=torch.long, device=self.storage_device)
             cv_to_store = torch.ones((x.shape[0], self.cv_dim), device=self.storage_device) * self.cv_min # Placeholder
        else:
             cv = cv.to(self.storage_device)
             cv_to_store = cv
             bin_indices = self._get_bin_indices(cv)
             
        # We process each bin separately to allow vectorization within bins
        # This is reasonably fast because n_bins is small (approx 10-100)
        unique_bins = torch.unique(bin_indices)
        
        for b in unique_bins:
            # Mask for current bin
            mask = (bin_indices == b)
            
            # Extract samples for this bin
            x_b = x[mask]
            
            n_samples = x_b.shape[0]
            
            # If input samples exceed bin capacity, keep only the latest ones
            # (User feedback: "only take the final n samples")
            if n_samples > self.max_per_bin:
                x_b = x_b[-self.max_per_bin:]
                # Also update beta/h/cv slices if needed (done below lazily via indices)
                # But here we need to slice the mask or indices to match
                # Easiest way: re-slice mask? No, mask is for input batch.
                # Let's just update n_samples and handle slicing carefully.
                start_idx_in_batch = n_samples - self.max_per_bin # Logic is getting complex for batch slicing
                # Simplify: Slice everything now
                mask_indices = torch.nonzero(mask).squeeze(-1)
                # Take last N indices
                selected_indices = mask_indices[-self.max_per_bin:]
                n_samples = self.max_per_bin
                x_b = x[selected_indices]
                # Beta/H/CV for this bin
                mask = torch.zeros_like(mask) # Reset mask
                mask[selected_indices] = True
            
            # Calculate destination indices in global buffer
            # Global index = b * max_per_bin + local_offset
            # We use circular buffer logic within the bin
            
            start_ptr = self.bin_ptrs[b].item()
            indices = torch.arange(start_ptr, start_ptr + n_samples, device=self.storage_device) % self.max_per_bin
            global_indices = b * self.max_per_bin + indices
            
            # Write to buffer
            self.x[global_indices] = x_b
            if cv_to_store.ndim == 1 and self.cv_dim == 1:
                self.cv[global_indices, 0] = cv_to_store[mask]
            else:
                self.cv[global_indices] = cv_to_store[mask]
            
            if beta is not None:
                self.has_conditions = True
                if isinstance(beta, torch.Tensor):
                    beta_in = beta.to(self.storage_device)
                    if beta_in.ndim == 0: beta_in = beta_in.expand(x.shape[0])
                    self.beta[global_indices] = beta_in[mask]
                else:
                    self.beta[global_indices] = torch.full((n_samples,), beta, device=self.storage_device)
            
            if h is not None:
                self.has_conditions = True
                if isinstance(h, torch.Tensor):
                    h_in = h.to(self.storage_device)
                    if h_in.ndim == 0: h_in = h_in.expand(x.shape[0])
                    self.h[global_indices] = h_in[mask]
                else:
                    self.h[global_indices] = torch.full((n_samples,), h, device=self.storage_device)
            
            # Update pointers
            self.bin_ptrs[b] = (start_ptr + n_samples) % self.max_per_bin
            self.bin_counts[b] = min(self.bin_counts[b].item() + n_samples, self.max_per_bin)
            
        # Update total size (sum of all bin counts)
        self.size = self.bin_counts.sum().item()
        
    def _add_fifo(self, x, beta, h, cv):
        batch_size = x.shape[0]
        
        if batch_size > self.buffer_size:
            # If batch to add is larger than buffer, take last part
            x = x[-self.buffer_size:]
            if beta is not None: beta = beta[-self.buffer_size:]
            if h is not None: h = h[-self.buffer_size:]
            if cv is not None: cv = cv[-self.buffer_size:]
            batch_size = x.shape[0]
            
        # Indices for circular buffer
        indices = torch.arange(self.ptr, self.ptr + batch_size) % self.buffer_size
        
        # Remove overwritten samples from their bins
        # We need to do this element-wise or careful batching because list implementation
        # For Python speed, a loop is acceptable for typical batch sizes (e.g. 128)
        # Vectorized approach is hard because multiple removed items might be in same bin
        
        indices_np = indices.numpy() # CPU
        sample_bins_np = self.sample_bins.numpy()
        bin_pos_np = self.bin_pos.numpy()
        
        for idx in indices_np:
            if self.size == self.buffer_size: # Logic only needed if we are overwriting
                old_bin = sample_bins_np[idx]
                if old_bin != -1: # Should be true if buffer full
                    pos = bin_pos_np[idx]
                    bin_list = self.bins[old_bin]
                    
                    # Swap with last element and pop
                    last_element_idx = bin_list[-1]
                    
                    if last_element_idx != idx:
                         bin_list[pos] = last_element_idx
                         bin_pos_np[last_element_idx] = pos # Update swapped element's pos
                         
                    bin_list.pop()
                    # No need to update bin_pos_np[idx] as it will be overwritten below

        self.x[indices] = x
        
        if cv is not None:
             cv_in = cv.to(self.storage_device)
             
             if cv_in.ndim == 1 and self.cv_dim == 1:
                self.cv[indices, 0] = cv_in
             else:
                self.cv[indices] = cv_in
                
             # Compute new bins
             bin_indices = self._get_bin_indices(cv_in)
             bin_indices = bin_indices.clamp(0, self.total_bins - 1)
        else:
             # Default to bin 0
             bin_indices = torch.zeros(batch_size, dtype=torch.long, device=self.storage_device)

             
        bin_indices_np = bin_indices.numpy()
        
        # Add new samples to bins
        for i, idx in enumerate(indices_np):
            new_bin = bin_indices_np[i]
            
            self.bins[new_bin].append(idx)
            
            # Update tracking
            sample_bins_np[idx] = new_bin
            bin_pos_np[idx] = len(self.bins[new_bin]) - 1

        # Sync back numpy views to tensors (share memory usually but good to be safe)
        # (Tensor.numpy() shares memory for CPU tensors, so modifications are in-place)
        
        if beta is not None:
            self.has_conditions = True
            if isinstance(beta, torch.Tensor):
                beta_in = beta.to(self.storage_device)
                if beta_in.ndim == 0:
                    beta_in = beta_in.expand(batch_size)
                self.beta[indices] = beta_in
            else:
                self.beta[indices] = torch.full((batch_size,), beta, device=self.storage_device)
                
        if h is not None:
            self.has_conditions = True
            if isinstance(h, torch.Tensor):
                h_in = h.to(self.storage_device)
                if h_in.ndim == 0:
                    h_in = h_in.expand(batch_size)
                self.h[indices] = h_in
            else:
                self.h[indices] = torch.full((batch_size,), h, device=self.storage_device)
        
        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size):
        """Sample a batch from the buffer."""
        if self.size == 0:
            return None, None, None
            
        if self.strategy == 'balanced':
            # 1. Identify non-empty bins from counts
            non_empty_bins = torch.nonzero(self.bin_counts > 0).squeeze(-1) # [N_active]
            
            if non_empty_bins.numel() == 0:
                return None, None, None
                
            # 2. Sample bins uniformly (with replacement)
            # Use torch for random choice
            random_indices = torch.randint(0, non_empty_bins.numel(), (batch_size,), device=self.storage_device)
            chosen_bins = non_empty_bins[random_indices] # [B]
            
            # 3. For each chosen bin, sample uniform index within its count
            # Get counts for chosen bins
            chosen_counts = self.bin_counts[chosen_bins]
            
            # Draw random offset for each sample: 0 to count-1
            offsets = (torch.rand(batch_size, device=self.storage_device) * chosen_counts).long()
            
            # Global index = bin * max + offset
            # Note: We don't worry about circular pointer 'start' here because relevant data is always 0..count-1?
            # Wait! In 'add', we write circularly to bin range.
            # BUT we don't zero out old data. 
            # If count < max, data is at [0, count-1].
            # If count == max, data is at [0, max-1] (full).
            # So picking any index 0..count-1 is valid. The 'start_ptr' is just for writing.
            indices = chosen_bins * self.max_per_bin + offsets
            
        else: # FIFO
            # CV-based Sampling:
            # 1. Identify non-empty bins
            non_empty_bins = [i for i, b in enumerate(self.bins) if len(b) > 0]
            if not non_empty_bins:
                 return None, None, None
                 
            # 2. Sample bins uniformly
            # samples_per_bin = batch_size // len(non_empty_bins) 
            # (This approach might be tricky if batch_size is small or bins are many. 
            #  Better to just sample batch_size bin indices uniformly with replacement)
            
            chosen_bins = np.random.choice(non_empty_bins, size=batch_size, replace=True)
            
            indices = torch.empty(batch_size, dtype=torch.long, device=self.storage_device)
            
            # 3. For each chosen bin, sample a random element from it
            # Optimization: group by bin to avoid calling random choice too many times? 
            # Actually random choice from list is fast.
            
            for i, bin_idx in enumerate(chosen_bins):
                 bin_list = self.bins[bin_idx]
                 rand_pos = np.random.randint(0, len(bin_list))
                 indices[i] = bin_list[rand_pos]
        
        x_out = self.x[indices].to(self.device)
        
        if self.has_conditions:
            beta_out = self.beta[indices].to(self.device)
            h_out = self.h[indices].to(self.device)
        else:
            beta_out = None
            h_out = None
            
        return x_out, beta_out, h_out
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        state = {
            # Configuration
            'buffer_size': self.buffer_size,
            'x_shape': tuple(self.x.shape[1:]),  # Exclude batch dimension
            'device': self.device,
            'storage_device': self.storage_device,
            'dtype': self.beta.dtype,  # Use beta to get dtype
            'strategy': self.strategy,
            # 'n_bins': self.n_bins, # REMOVED: Replaced by n_bins_per_dim
            'n_bins_per_dim': self.n_bins_per_dim.clone(),
            'cv_min': self.cv_min,
            'cv_max': self.cv_max,
            'bin_width': self.bin_width,
            
            # State variables
            'ptr': self.ptr,
            'size': self.size,
            'has_conditions': self.has_conditions,
            
            # Data tensors
            'x': self.x.clone(),
            'beta': self.beta.clone(),
            'h': self.h.clone(),
            'cv': self.cv.clone(),
        }
        
        # Strategy-specific state
        if self.strategy == 'balanced':
            state['max_per_bin'] = self.max_per_bin
            state['bin_ptrs'] = self.bin_ptrs.clone()
            state['bin_counts'] = self.bin_counts.clone()
            # Convert active_bins list to tensor for serialization
            if self.active_bins:
                state['active_bins'] = torch.tensor(self.active_bins, dtype=torch.long)
            else:
                state['active_bins'] = torch.tensor([], dtype=torch.long)
        else:  # FIFO
            # Convert bins (list of lists) to a format that can be serialized
            # Store as a list of tensors
            state['bins'] = [torch.tensor(bin_list, dtype=torch.long) for bin_list in self.bins]
            state['sample_bins'] = self.sample_bins.clone()
            state['bin_pos'] = self.bin_pos.clone()
        
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint.
        
        Args:
            state_dict: Dictionary containing buffer state. If None or missing keys,
                       will gracefully skip loading (for backward compatibility).
        """
        if state_dict is None:
            return  # Backward compatibility: skip if None
        
        # Load configuration and verify compatibility
        if 'buffer_size' in state_dict:
            if state_dict['buffer_size'] != self.buffer_size:
                raise ValueError(f"Buffer size mismatch: checkpoint has {state_dict['buffer_size']}, "
                               f"but current buffer has {self.buffer_size}")
        
        if 'x_shape' in state_dict:
            if state_dict['x_shape'] != tuple(self.x.shape[1:]):
                raise ValueError(f"x_shape mismatch: checkpoint has {state_dict['x_shape']}, "
                               f"but current buffer has {tuple(self.x.shape[1:])}")
        
        if 'strategy' in state_dict:
            if state_dict['strategy'] != self.strategy:
                raise ValueError(f"Strategy mismatch: checkpoint has {state_dict['strategy']}, "
                               f"but current buffer has {self.strategy}")
        
        if 'n_bins_per_dim' in state_dict:
             if not torch.equal(state_dict['n_bins_per_dim'], self.n_bins_per_dim):
                  # Could also be loose check if we just want to load data?
                  # But consistency is important for bin mapping
                  raise ValueError(f"n_bins_per_dim mismatch: checkpoint has {state_dict['n_bins_per_dim']}, "
                                   f"but current buffer has {self.n_bins_per_dim}")
        elif 'n_bins' in state_dict:
             # Legacy check: if stored as scalar n_bins
             legacy_n_bins = state_dict['n_bins']
             # Check if current config matches this scalar (all dims equal)
             if not (self.n_bins_per_dim == legacy_n_bins).all():
                 raise ValueError(f"Legacy n_bins mismatch: checkpoint has {legacy_n_bins}, "
                                   f"but current buffer has {self.n_bins_per_dim}")
        
        # Load state variables
        if 'ptr' in state_dict:
            self.ptr = state_dict['ptr']
        if 'size' in state_dict:
            self.size = state_dict['size']
        if 'has_conditions' in state_dict:
            self.has_conditions = state_dict['has_conditions']
        
        # Load data tensors (only if they exist in checkpoint)
        if 'x' in state_dict:
            self.x.copy_(state_dict['x'])
        if 'beta' in state_dict:
            self.beta.copy_(state_dict['beta'])
        if 'h' in state_dict:
            self.h.copy_(state_dict['h'])
        if 'cv' in state_dict:
            self.cv.copy_(state_dict['cv'])
        
        # Load strategy-specific state
        if self.strategy == 'balanced':
            if 'bin_ptrs' in state_dict:
                self.bin_ptrs.copy_(state_dict['bin_ptrs'])
            if 'bin_counts' in state_dict:
                self.bin_counts.copy_(state_dict['bin_counts'])
            if 'active_bins' in state_dict:
                # Convert tensor back to list
                active_bins_tensor = state_dict['active_bins']
                if active_bins_tensor.numel() > 0:
                    self.active_bins = active_bins_tensor.tolist()
                else:
                    self.active_bins = []
        else:  # FIFO
            if 'bins' in state_dict:
                # Convert list of tensors back to list of lists
                bins_tensors = state_dict['bins']
                if len(bins_tensors) != self.total_bins:
                    raise ValueError(f"Number of bins mismatch: checkpoint has {len(bins_tensors)}, "
                                   f"but current buffer has {self.total_bins}")
                self.bins = [tensor.tolist() for tensor in bins_tensors]
            if 'sample_bins' in state_dict:
                self.sample_bins.copy_(state_dict['sample_bins'])
            if 'bin_pos' in state_dict:
                self.bin_pos.copy_(state_dict['bin_pos'])
            
            # Legacy fix: If switching from scalar CV to tensor CV in buffer state
            if 'cv' in state_dict and state_dict['cv'].ndim == 1 and self.cv_dim == 1:
                self.cv[:, 0].copy_(state_dict['cv'])
            elif 'cv' in state_dict:
                # Handle dimension mismatch if buffer size changed or cv_dim changed?
                # Assume match for now, or copy flexible
                pass


def compute_model_log_prob(model, x, beta=None, h=None):
    """Compute the model component of log_rnd for samples x.
    
    This function iteratively computes the accumulation term:
       sum_{t} [ -log(V-1) - log P(x_t | x_{<t}) ]
    which matches the calculation in the rnd() sampling function.
    
    Args:
        model: Autoregressive model
        x: [B, D] input samples (fully observed)
        beta: [B] or scalar
        h: [B] or scalar
        
    Returns:
        log_rnd_model_term: [B] The accumulated model term of log_rnd.
           To get full log_rnd, add reward_fn(x): 
           log_rnd = log_rnd_model_term + reward_fn(x)
    """
    if hasattr(model, 'module'):
        model = model.module
        
    batch_size = x.shape[0]
    device = x.device
    
    # Initialize mock 'x' with masks to simulate generation
    # We use the actual token values from input x but revealed iteratively
    x_curr = torch.full((batch_size, model.length), model.vocab_size-1, device=device, dtype=torch.int64)
    
    # Random order for each sample (same as rnd)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    
    batch_arange = torch.arange(batch_size, device=device)
    log_rnd_term = torch.zeros(batch_size, device=device)
    
    for d in range(model.length-1, -1, -1):
        # Forward pass on current partially masked sequence
        if beta is not None or h is not None:
            logits = model(x_curr, beta=beta, h=h)[:, :, :-1]
        else:
            logits = model(x_curr)[:, :, :-1]
            
        # Check if raw logits or log probs
        if (logits > 0).any(): 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            log_probs = logits

        # Identify which token is being "revealed" at this step
        # In rnd(), 'update' is sampled. Here, 'update' is the true token from x.
        update_pos = jump_pos[:, d] # [B]
        update_val = x[batch_arange, update_pos] # [B]
        
        # Accumulate: -log(V-1) - log P(x_t | ...)
        # Note: model.vocab_size usually includes mask, so vocab_size-1 is number of real tokens.
        log_rnd_term += -np.log(model.vocab_size-1) - log_probs[batch_arange, update_pos, update_val]
        
        # Reveal the token in x_curr for next step
        x_curr[batch_arange, update_pos] = update_val
        
    return log_rnd_term


def save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, path, bias_potential=None, replay_buffer=None):
    """Helper function to save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'losses': losses,
        'ess_train': ess_train,
        'ess_eval': ess_eval,
        'bias_potential': bias_potential.state_dict() if bias_potential is not None else None,
        'replay_buffer': replay_buffer.state_dict() if replay_buffer is not None else None,
        'cfg': cfg
    }, path)


def rnd(model, reward_model, batch_size, device='cuda:0', beta_batch=None, h_batch=None, J=1):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    
    Args:
        model: The model
        reward_model: Function that takes (x, beta, J, h) and returns rewards [B]
        batch_size: Batch size
        device: Device
        beta_batch: [B] tensor of beta values, or None for single beta (backward compatible)
        h_batch: [B] tensor of field values, or None for single field (backward compatible)
        J: Scalar interaction strength
        
    Returns:
        x: the final samples, [B, D]
        log_rnd: the log RND along this trajectory, [B]
    """
    if hasattr(model, 'module'):
        model = model.module
    
    x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    log_rnd = torch.zeros(batch_size, device=device) # [B]
    for d in range(model.length-1, -1, -1):
        # Pass beta and h to model if provided
        if beta_batch is not None or h_batch is not None:
            logits = model(x, beta=beta_batch, h=h_batch)[:, :, :-1] # [B, D, N-1]
        else:
            logits = model(x)[:, :, :-1] # [B, D, N-1] - backward compatible
        
        # Both vit_rope (Ising) and MultiOutputTransformer (CuAu) now return log probabilities
        # For backward compatibility with old models that might return raw logits, check and convert if needed
        if (logits > 0).any():
            # Raw logits detected (old model format) - apply log_softmax to get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # For sampling, we can use either raw logits or log_probs with Gumbel-max
            # Using raw logits is more standard, but log_probs also works (argmax is shift-invariant)
            logits_for_sampling = logits
        else:
            # Already log probabilities (expected for modern models: vit_rope and MultiOutputTransformer)
            log_probs = logits
            logits_for_sampling = logits
        
        update = sample_categorical_logits(
            logits_for_sampling[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size-1) - log_probs[batch_arange, jump_pos[:, d], update]
    
    # Compute reward with per-sample temperatures/fields if provided
    if beta_batch is not None or h_batch is not None:
        # Ensure tensors are on correct device
        if beta_batch is not None and isinstance(beta_batch, torch.Tensor):
            beta_batch = beta_batch.to(device)
        if h_batch is not None and isinstance(h_batch, torch.Tensor):
            h_batch = h_batch.to(device)
        log_rnd += reward_model(x, beta=beta_batch, h=h_batch if h_batch is not None else 0, J=J)
    else:
        # Backward compatible: use default scalar values
        log_rnd += reward_model(x) # [B]

    return x, log_rnd


@torch.no_grad()
def sampling(model, batch_size, rounds=1, device='cuda:0'):
    """Any order autoregressive sampling"""
    if hasattr(model, 'module'):
        model = model.module
    batch_arange = torch.arange(batch_size, device=device)
    all_samples = []
    for _ in tqdm(range(rounds), leave=False):
        x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
        jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
        for d in tqdm(range(model.length-1, -1, -1), leave=False):
            logits = model.logits(x)[:, :, :-1] # [B, D, N-1], not log-softmaxed but fine
            update = sample_categorical_logits(
                logits[batch_arange, jump_pos[:, d]]) # [B]
            x[batch_arange, jump_pos[:, d]] = update
        all_samples.append(x)
    return torch.cat(all_samples)


def loss_ce(log_rnd):
    """Cross entropy loss KL(P^*||P^u)"""
    weights = log_rnd.detach().softmax(dim=-1)
    return (log_rnd * weights).sum()


def loss_lv(log_rnd):
    r"""Log variance loss Var_{P^\bar{u}}\log\frac{dP^*}{dP^u}"""
    return log_rnd.var()


def loss_re_rf(log_rnd, const=0):
    r"""Relative entropy loss KL(P^u||P^*) with REINFORCE trick"""
    reward_term = (-log_rnd.detach() + const)
    reward_mean = reward_term.mean()
    reward_var = reward_term.var()
    return (-log_rnd * (reward_term - reward_mean) / (reward_var + 1e-8)).mean()


def loss_wdce(model, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, beta_batch=None, h_batch=None):
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
    
    log_rnd: [B]; x: [B, D] (no mask)
    num_replicates: R, number of replicates of each row in x
    weight_func: w(lambda) for each sample, 1/lambda by default
    beta_batch: [B] tensor of beta values (optional, for conditioning)
    h_batch: [B] tensor of field values (optional, for conditioning)
    """
    if hasattr(model, 'module'):
        model = model.module
    
    batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, D]
    batch_weights = log_rnd.detach_().softmax(dim=-1).repeat_interleave(num_replicates, dim=0) # [B*R]
    lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None] # [B*R, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, batch)
    
    # Handle beta/h conditioning - replicate them to match perturbed_batch
    if beta_batch is not None or h_batch is not None:
        beta_perturbed = beta_batch.repeat_interleave(num_replicates, dim=0) if beta_batch is not None else None
        h_perturbed = h_batch.repeat_interleave(num_replicates, dim=0) if h_batch is not None else None
        logits = model(perturbed_batch, beta=beta_perturbed, h=h_perturbed)
    else:
        logits = model(perturbed_batch)  # Backward compatible
    
    losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights * batch_weights).mean()


def loss_dce(model, x, weight_func=lambda l: 1/l, beta_batch=None, h_batch=None):
    r"""
    Denoising cross entropy loss, x [B, D] are ground truth samples
    weight_func: w(lambda) for each sample, 1/lambda by default
    beta_batch: [B] tensor of beta values (optional, for conditioning)
    h_batch: [B] tensor of field values (optional, for conditioning)
    """
    lamda = torch.rand(x.shape[0], device=x.device) # [B]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B]
    masked_index = torch.rand(*x.shape, device=x.device) < lamda[..., None] # [B, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, x)
    
    # Handle beta/h conditioning
    if beta_batch is not None or h_batch is not None:
        logits = model(perturbed_batch, beta=beta_batch, h=h_batch)
    else:
        logits = model(perturbed_batch)  # Backward compatible
    
    losses = torch.zeros(*x.shape, device=x.device, dtype=logits.dtype) # [B, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=x[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights).mean()


def log_validation_metrics(
    wandb_run,
    logp_x: torch.Tensor,
    logf_t: torch.Tensor,
    vfe: torch.Tensor,
    beta_batch: Optional[torch.Tensor],
    h_batch: Optional[torch.Tensor],
    step: int,
    log_rnd: Optional[torch.Tensor] = None,
    log_kwargs: Optional[dict] = None,
) -> None:
    """Log validation metrics per condition and overall, similar to snowyflow.
    
    Args:
        wandb_run: Wandb run object for logging
        logp_x: Log probability of data under model, shape [B]
        logf_t: Log forward probability from MDNS, shape [B]
        vfe: Variational free energy (logp_x - logf_t), shape [B]
        beta_batch: Beta values [B] or None for single condition
        h_batch: Field values [B] or None for single condition
        step: Step/epoch number for logging
        log_rnd: Log RND values, shape [B] (optional)
        log_kwargs: Additional keyword arguments to pass to wandb.log
    """
    if log_kwargs is None:
        log_kwargs = {}
    
    # Extract temperature and field batches (use defaults if None)
    batch_size = logf_t.shape[0]
    if beta_batch is not None:
        temp_batch = 1.0 / beta_batch.cpu().numpy()  # [B]
    else:
        temp_batch = np.zeros(batch_size)  # Default: single temp (will be 0.0)
    
    if h_batch is not None:
        field_batch = h_batch.cpu().numpy()  # [B]
    else:
        field_batch = np.zeros(batch_size)  # Default: single field (will be 0.0)
    
    # Get unique temperature/field combinations
    conditions = list(zip(temp_batch, field_batch))
    unique_conditions = sorted(set(conditions))
    
    # Separate dictionaries for main val panel and conditions panel
    val_log_data = {}  # Overall metrics for main val panel
    conditions_log_data = {}  # Per-condition metrics for separate panel
    
    # Group metrics by condition
    for temp_val, field_val in unique_conditions:
        condition_name = f"T{temp_val:04.0f}_F{field_val:+.3f}"
        
        # Find indices for this condition
        mask = np.array([(t == temp_val and f == field_val) for t, f in conditions])
        
        # Extract metrics for this condition
        logf_t_cond = logf_t[mask]
        logp_x_cond = logp_x[mask]
        vfe_cond = vfe[mask]
        
        # Log per-condition metrics to separate panel (val_conditions/)
        conditions_log_data[f"logf_t_mean/{condition_name}"] = logf_t_cond.mean().item()
        conditions_log_data[f"logf_t_std/{condition_name}"] = logf_t_cond.std().item()
        conditions_log_data[f"logp_x_mean/{condition_name}"] = logp_x_cond.mean().item()
        conditions_log_data[f"logp_x_std/{condition_name}"] = logp_x_cond.std().item()
        conditions_log_data[f"vfe_mean/{condition_name}"] = vfe_cond.mean().item()
        conditions_log_data[f"vfe_std/{condition_name}"] = vfe_cond.std().item()
        
        # Log log_rnd per condition if provided
        if log_rnd is not None:
            log_rnd_cond = log_rnd[mask]
            conditions_log_data[f"log_rnd_mean/{condition_name}"] = log_rnd_cond.mean().item()
            conditions_log_data[f"log_rnd_std/{condition_name}"] = log_rnd_cond.std().item()
            conditions_log_data[f"log_rnd_min/{condition_name}"] = log_rnd_cond.min().item()
            conditions_log_data[f"log_rnd_max/{condition_name}"] = log_rnd_cond.max().item()
    
    # Log temperature and field statistics to conditions panel (only if they vary)
    if beta_batch is not None:
        conditions_log_data["temp/min"] = float(temp_batch.min())
        conditions_log_data["temp/max"] = float(temp_batch.max())
        conditions_log_data["temp/mean"] = float(temp_batch.mean())
        conditions_log_data["temp/std"] = float(temp_batch.std())
    
    if h_batch is not None:
        conditions_log_data["field/min"] = float(field_batch.min())
        conditions_log_data["field/max"] = float(field_batch.max())
        conditions_log_data["field/mean"] = float(field_batch.mean())
        conditions_log_data["field/std"] = float(field_batch.std())
    
    # Add overall statistics to main val panel (always log these)
    val_log_data["logf_t_mean/overall"] = logf_t.mean().item()
    val_log_data["logf_t_std/overall"] = logf_t.std().item()
    val_log_data["logp_x_mean/overall"] = logp_x.mean().item()
    val_log_data["logp_x_std/overall"] = logp_x.std().item()
    val_log_data["vfe_mean/overall"] = vfe.mean().item()
    val_log_data["vfe_std/overall"] = vfe.std().item()
    
    # Log log_rnd overall statistics if provided
    if log_rnd is not None:
        val_log_data["log_rnd_mean/overall"] = log_rnd.mean().item()
        val_log_data["log_rnd_std/overall"] = log_rnd.std().item()
        val_log_data["log_rnd_min/overall"] = log_rnd.min().item()
        val_log_data["log_rnd_max/overall"] = log_rnd.max().item()
    
    # Prefix keys appropriately for separate panels
    val_log_data_prefixed = {f"val/{k}": v for k, v in val_log_data.items()}
    conditions_log_data_prefixed = {f"val_conditions/{k}": v for k, v in conditions_log_data.items()}
    
    # Log to separate panels
    wandb_run.log(val_log_data_prefixed, step=step, **log_kwargs)
    if conditions_log_data_prefixed:  # Only log if there are per-condition metrics
        wandb_run.log(conditions_log_data_prefixed, step=step, **log_kwargs)


def _compute_log_stats(x, log_rnd, reward_fn, model, beta_batch=None, h_batch=None, J=1,
                       bias_potential=None, cv_compute_fn=None):
    """Compute logf_t and logp_x given samples and RND values.
    
    Args:
        x: [B, D] samples
        log_rnd: [B] log RND values
        reward_fn: Function that takes (x, beta, J, h) and returns rewards
        model: Model (for getting vocab_size)
        beta_batch: [B] tensor of beta values, or None for scalar beta
        h_batch: [B] tensor of field values, or None for scalar field
        J: Scalar interaction strength
    """
    # Compute logf_t using per-sample betas/fields if provided
    if beta_batch is not None or h_batch is not None:
        logf_t_vals = reward_fn(x, beta=beta_batch, h=h_batch if h_batch is not None else 0, J=J, use_bias=False)
    else:
        logf_t_vals = reward_fn(x, use_bias=False)  # Use default scalar values
    num_states = getattr(model, "vocab_size", 3) - 1  # exclude mask token
    data_dim = x.shape[1]
    uniform_prior_term = -torch.log(torch.tensor(float(num_states), device=x.device))
    uniform_prior_total = uniform_prior_term * data_dim
    logp_x_vals = uniform_prior_total + logf_t_vals - log_rnd
    
    # Correct logp_x vals if bias_potential is provided
    # logp_x_effective = logp_model(x) + beta * V(s)
    # This recovers the effective sampling probability with respect to the unbiased Hamiltonian
    # assuming logf_t_vals is unbiased (which it is, see use_bias=False above)
    if bias_potential is not None:
        with torch.no_grad():
            if cv_compute_fn is not None:
                s = cv_compute_fn(x)  # Use provided CV computation function
            else:
                # Backward compatible: default to Ising magnetization
                x_spins = 2 * x - 1
                s = ising2d_mag(x_spins)

            bias_vals = bias_potential.evaluate(s) # [B]
            
            # Apply beta correction
            if beta_batch is not None:
                logp_x_vals = logp_x_vals - beta_batch * bias_vals
            else:
                 # Assume standard T from bias potential or model
                 # Train Ising sets bias_potential.T = 1/beta
                 beta = 1.0 / bias_potential.T
                 logp_x_vals = logp_x_vals - beta * bias_vals
                 
    return logf_t_vals, logp_x_vals


def _visualize_lattices(samples, L, n_rows=2, n_cols=5, max_samples=16, 
                        beta_batch=None, h_batch=None, q=None):
    """Visualize Ising or Potts lattices in a grid.
    
    Args:
        samples: Tensor of shape [B, L*L] or [B, L, L]
        L: Lattice dimension
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        max_samples: Maximum number of samples to visualize
        beta_batch: [B] tensor of beta values (optional, for sampling from each temp)
        h_batch: [B] tensor of field values (optional, for sampling from each field)
        q: Number of states (for Potts model). If None or 2, uses binary Ising visualization.
    """
    # Reshape if needed: [B, L*L] -> [B, L, L]
    if samples.ndim == 2:
        B = samples.shape[0]
        samples = samples.view(B, L, L)
    
    # Convert to float and CPU for matplotlib
    samples = samples.float().cpu()
    
    # If beta_batch and h_batch are provided, sample from each unique temp/field combination
    if beta_batch is not None and h_batch is not None:
        # Convert beta to temperature for grouping
        temp_batch = 1.0 / beta_batch.cpu().numpy()
        h_batch_cpu = h_batch.cpu().numpy()
        
        # Find unique temp/field combinations
        # Round to avoid floating point precision issues
        temp_rounded = np.round(temp_batch, decimals=4)
        h_rounded = np.round(h_batch_cpu, decimals=4)
        
        # Create combination keys
        combinations = [(t, h) for t, h in zip(temp_rounded, h_rounded)]
        unique_combos = list(set(combinations))
        
        # Sample at least a few samples from each combination
        samples_to_plot_indices = []
        samples_per_combo = max(1, max_samples // max(len(unique_combos), 1))
        
        for combo in unique_combos:
            # Find indices matching this combination
            matching_indices = np.array([i for i, c in enumerate(combinations) if c == combo])
            # Sample up to samples_per_combo from this combination
            n_samples_from_combo = min(samples_per_combo, len(matching_indices))
            if n_samples_from_combo > 0:
                if len(matching_indices) == 1:
                    # Only one matching index, just use it
                    selected_indices = matching_indices
                else:
                    selected_indices = np.random.choice(
                        matching_indices, size=n_samples_from_combo, replace=False
                    )
                samples_to_plot_indices.extend(selected_indices.tolist())
        
        # Limit total number of samples
        n_plots = min(n_rows * n_cols, len(samples_to_plot_indices), max_samples)
        samples_to_plot_indices = samples_to_plot_indices[:n_plots]
        samples_to_plot = samples[samples_to_plot_indices]
    else:
        # Original behavior: just take first N samples
        n_plots = min(n_rows * n_cols, max_samples, samples.shape[0])
        samples_to_plot = samples[:n_plots]
    
    # Create colormap based on number of states
    if q is None or q == 2:
        # Binary Ising (blue for 0, pink for 1)
        palette = ["#1f77b4", "#e377c2"]
        vmax = 1
    else:
        # Potts model with q states
        # Use distinct colors for each state
        if q == 3:
            palette = ["#540D6E", "#EE4266", "#FFD23F"]  # Purple, Pink, Yellow
        elif q == 4:
            palette = ["#540D6E", "#EE4266", "#FFD23F", "#26547c"]  # Add blue
        else:
            # For q > 4, use a colormap that can handle more states
            from matplotlib.cm import get_cmap
            cmap_obj = get_cmap('tab10')
            palette = [cmap_obj(i / max(q-1, 1)) for i in range(q)]
        vmax = q - 1
    
    cmap = ListedColormap(palette)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    for i in range(n_plots):
        ax = axes[i]
        sample_np = samples_to_plot[i].numpy()
        ax.imshow(sample_np, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
        ax.axis("off")
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    return fig


def train(model, optimizer, reward_fn, args, device, num_epochs = 10000, ema=None,
          losses=None, ess_train=None, ess_eval=None, wandb_run=None, L=None,
          bias_potential=None, current_temps=None, current_fields=None, rng=None, save_dir=None, cfg_dict=None,
          validation_plot_callback=None, cv_compute_fn=None,
          buffer_size=0, buffer_ratio=0.0, buffer_n_bins=1, buffer_strategy='fifo', plot_bias_fn=None,
          buffer_state_dict=None, start_epoch=0, save_best_metric='none'):
    loss_fn = {'ce': loss_ce, 'lv': loss_lv, 're_rf': loss_re_rf,
               'wdce': loss_wdce}.get(args.loss_fn)
    if loss_fn is None:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    # continue recording the metrics from the last training
    losses = [] if losses is None else losses.copy()
    ess_train = [] if ess_train is None else ess_train.copy()
    ess_eval = [] if ess_eval is None else ess_eval.copy()
    # Best checkpoint tracking
    best_loss = float('inf')
    best_epoch = 0
    pbar = tqdm(range(num_epochs), initial=start_epoch, total=start_epoch+num_epochs)
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # Initialize Replay Buffer
    replay_buffer = None
    if buffer_size > 0:
        # Determine dimension D from model
        D = model.length
        # Parse CV arguments carefully
        # They might be strings (comma separated), floats, or lists
        def parse_arg_list(arg):
            if isinstance(arg, str):
                try:
                    return [float(x) for x in arg.split(',')]
                except ValueError:
                    return [-1.0] # Fallback
            elif isinstance(arg, (list, tuple)):
                return arg
            elif isinstance(arg, (int, float)):
                return [arg]
            else:
                return [-1.0]

        cv_min_raw = args.cv_min if hasattr(args, 'cv_min') else -1.0
        cv_max_raw = args.cv_max if hasattr(args, 'cv_max') else 1.0
        
        cv_min_val = parse_arg_list(cv_min_raw)
        cv_max_val = parse_arg_list(cv_max_raw)
        
        # Similar for n_bins? Usually integer, but could be list?
        # args.buffer_n_bins is usually int. If user wants per-dim bins, they might need to update arg parser.
        # For now assume scalar n_bins applies to all dims or user passes list manually if calling function.
        buffer_n_bins_val = buffer_n_bins 
        
        replay_buffer = ReplayBuffer(buffer_size, (D,), device=device,
                                     cv_min=cv_min_val, cv_max=cv_max_val, n_bins=buffer_n_bins_val, strategy=buffer_strategy)
        print(f"Initialized ReplayBuffer with size {buffer_size}, mixing ratio {buffer_ratio}, n_bins {buffer_n_bins}, strategy {buffer_strategy}")
        
        # Load buffer state from checkpoint if provided
        if buffer_state_dict is not None:
            try:
                replay_buffer.load_state_dict(buffer_state_dict)
                print(f"Loaded ReplayBuffer state from checkpoint (size={replay_buffer.size})")
            except Exception as e:
                print(f"Warning: Could not load ReplayBuffer state from checkpoint: {e}")
                print("Continuing with empty buffer.")

    # Handle multiple temperatures/fields
    from utils_ising import get_temp_field_batch, sample_temp_field
    use_multi_temp_field = hasattr(args, 'temps') and args.temps is not None and (
        len(args.temps) > 1 or (hasattr(args, 'fields') and args.fields is not None and len(args.fields) > 1)
    )
    
    # Initialize temps/fields for dynamic sampling
    # Store starting temps/fields for validation (fixed, not dynamically sampled)
    starting_temps = None
    starting_fields = None
    if use_multi_temp_field:
        if current_temps is None:
             current_temps = np.array(args.temps)
        if current_fields is None:
             current_fields = np.array(args.fields) if hasattr(args, 'fields') and args.fields is not None else np.array([0.0])
        # Store starting values for validation
        starting_temps = current_temps.copy()
        starting_fields = current_fields.copy()
        if rng is None:
            rng = np.random.default_rng(seed=args.seed if args.seed is not None else 42)
    else:
        current_temps = None
        # current_fields = None # Already passed or initialized
        if rng is None:
             rng = None

    x_saved, log_rnd_saved = None, None
    
    for epoch in pbar:
        model.train(); optimizer.zero_grad(); info = {}

        # Generate temperature/field batches if using multiple temps/fields
        beta_batch = None
        h_batch = None
        log_dict_temp_field = {}  # Initialize for logging
        if use_multi_temp_field:
            # Handle dynamic sampling if enabled
            if hasattr(args, 'sample_delta_temp') and args.sample_delta_temp:
                # Use provided min/max temps if available, otherwise compute from current temps
                if hasattr(args, 'min_temp') and args.min_temp is not None:
                    min_temp = args.min_temp
                else:
                    min_temp = current_temps.min() - args.delta_temp
                if hasattr(args, 'max_temp') and args.max_temp is not None:
                    max_temp = args.max_temp
                else:
                    max_temp = current_temps.max() + args.delta_temp
                current_temps = sample_temp_field(
                    current_temps, args.delta_temp, min_temp, max_temp, rng
                )
            if hasattr(args, 'sample_delta_field') and args.sample_delta_field:
                min_field = current_fields.min() - args.delta_field
                max_field = current_fields.max() + args.delta_field
                current_fields = sample_temp_field(
                    current_fields, args.delta_field, min_field, max_field, rng
                )
            
            # Generate batch of temps/fields
            beta_batch, h_batch, _, _, _ = get_temp_field_batch(
                current_temps, current_fields, args.batch_size
            )
            beta_batch = beta_batch.to(device)
            h_batch = h_batch.to(device)
            
            # Log temperature and field statistics
            if beta_batch is not None:
                # Convert beta back to temperature for logging (beta = 1/T, so T = 1/beta)
                temp_batch = 1.0 / beta_batch.cpu().numpy()
                log_dict_temp_field = {
                    "train/temp_min": float(temp_batch.min()),
                    "train/temp_max": float(temp_batch.max()),
                    "train/temp_mean": float(temp_batch.mean()),
                    "train/temp_std": float(temp_batch.std()),
                    "train/field_min": float(h_batch.cpu().min().item()),
                    "train/field_max": float(h_batch.cpu().max().item()),
                    "train/field_mean": float(h_batch.cpu().mean().item()),
                    "train/field_std": float(h_batch.cpu().std().item()),
                }
        loss = None
        if args.loss_fn == 'wdce':
            with torch.no_grad():
                if x_saved is None or epoch % args.resample_every_n_step == 0:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device,
                                     beta_batch=beta_batch, h_batch=h_batch, J=args.J if hasattr(args, 'J') else 1)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                    is_fresh_sample = True
                else:
                    x, log_rnd = x_saved, log_rnd_saved
                    is_fresh_sample = False
            # loss will be computed below after optional replay-buffer mixing
        else:
            x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device,
                             beta_batch=beta_batch, h_batch=h_batch, J=args.J if hasattr(args, 'J') else 1)
            loss = loss_fn(log_rnd)
            is_fresh_sample = True

        # Update bias after sampling (on-policy update)
        # CRITICAL FIX: Only update bias if we actually generated new samples!
        # Otherwise we build a huge wall at the same spot for N steps (sloshing).
        
        # Calculate CV if needed (for bias or buffer)
        s_cv = None
        if is_fresh_sample and (bias_potential is not None or replay_buffer is not None):
             with torch.no_grad():
                if cv_compute_fn is not None:
                    s_cv = cv_compute_fn(x)  # Use provided CV computation function
                else:
                    # Backward compatible: default to Ising magnetization
                    x_spins = 2 * x - 1
                    s_cv = ising2d_mag(x_spins)
        
        if bias_potential is not None and is_fresh_sample:
             if s_cv is not None:
                bias_potential.update(s_cv)

        # Add to Replay Buffer - ONLY FRESH SAMPLES
        if replay_buffer is not None and is_fresh_sample:
            replay_buffer.add(x, beta=beta_batch, h=h_batch, cv=s_cv)
            
        # Prepare Training Batch (Mix Fresh + Replay)
        x_train, log_rnd_train = x, log_rnd
        beta_train, h_train = beta_batch, h_batch
        
        # Only mix if we have fresh samples (user requirement for WDCE)
        if is_fresh_sample and replay_buffer is not None and replay_buffer.size >= args.batch_size:
            n_replay = int(args.batch_size * buffer_ratio)
            if n_replay > 0 and n_replay < args.batch_size:
                # Slice fresh samples
                n_fresh = args.batch_size - n_replay
                x_f = x[:n_fresh]
                log_rnd_f = log_rnd[:n_fresh]
                beta_f = beta_batch[:n_fresh] if beta_batch is not None else None
                h_f = h_batch[:n_fresh] if h_batch is not None else None
                
                # Sample from Buffer
                x_r, beta_r, h_r = replay_buffer.sample(n_replay)
                
                # RECALCULATE log_rnd for Replay Samples
                # log_rnd = log_reward(current_bias) + log_model_term
                # Note: compute_model_log_prob returns sum[ -log(V-1) - log P(x_t) ]
                # So we simply add it to log_reward.
                with torch.no_grad():
                    # Recalculate Reward with CURRENT bias
                    J_val = args.J if hasattr(args, 'J') else 1
                    log_reward_r = reward_fn(x_r, beta=beta_r, h=h_r, J=J_val, use_bias=True)
                    
                    # Recalculate Model Term with CURRENT model
                    log_model_term_r = compute_model_log_prob(model, x_r, beta=beta_r, h=h_r)
                    
                    log_rnd_r = log_reward_r + log_model_term_r
                
                # Combine
                x_train = torch.cat([x_f, x_r], dim=0)
                log_rnd_train = torch.cat([log_rnd_f, log_rnd_r], dim=0)
                
                if beta_batch is not None:
                    # If beta_r came back as None (shouldn't if buffer works right), handle it
                    if beta_r is None: # Fallback
                         if beta_batch is not None:
                             beta_r = beta_batch[:n_replay] 
                    beta_train = torch.cat([beta_f, beta_r], dim=0)
                elif beta_r is not None:
                     # Fresh was None but Replay has Beta? (Unlikely in consistent run)
                     pass

                if h_batch is not None:
                    if h_r is None: 
                        if h_batch is not None:
                            h_r = h_batch[:n_replay]
                    h_train = torch.cat([h_f, h_r], dim=0)

        # Recalculate loss on mixed batch (overwriting previous loss calculation)
        if args.loss_fn == 'wdce':
            loss = loss_wdce(model, log_rnd_train, x_train,
                                num_replicates=args.wdce_num_replicates,
                                beta_batch=beta_train, h_batch=h_train)
        else:
            loss = loss_fn(log_rnd_train)

        # Synchronize loss across processes
        
        logf_t_vals, logp_x_vals = _compute_log_stats(x_train, log_rnd_train, reward_fn, model,
                                                       beta_batch=beta_train, h_batch=h_train,
                                                       J=args.J if hasattr(args, 'J') else 1,
                                                       bias_potential=bias_potential,
                                                       cv_compute_fn=cv_compute_fn)
        vfe = logp_x_vals - logf_t_vals  # variational free energy
        ess_train.append(ess(log_rnd_train))
        info['ess_train'] = ess_train[-1]
        info['loss'] = loss.item()
        losses.append(loss.item())

        if wandb_run is not None:
            log_dict = {
                "train/loss": loss.item(),
                "train/avg_free_energy": vfe.mean().item(),
                "train/std_free_energy": vfe.std().item(),
                "train/avg_logp_x": logp_x_vals.mean().item(),
                "train/std_logp_x": logp_x_vals.std().item(),
                "train/avg_logf_t": logf_t_vals.mean().item(),
                "train/std_logf_t": logf_t_vals.std().item(),
                "train/avg_log_rnd": log_rnd_train.mean().item(),
                "train/std_log_rnd": log_rnd_train.std().item(),
                "train/min_log_rnd": log_rnd_train.min().item(),
                "train/max_log_rnd": log_rnd_train.max().item(),
            }
            
            # Log bias stats
            if bias_potential is not None:
                # We can't plot the whole grid every step easily, just stats
                # Using get_bias_grid_np() if available or accessing internal
                if hasattr(bias_potential, 'bias_grid'):
                    bias_grid = bias_potential.bias_grid.detach().cpu().numpy()
                    log_dict["bias/max_height"] = bias_grid.max()
                    log_dict["bias/mean_height"] = bias_grid.mean()
                    # Fraction of visited states (nonzero bias)
                    log_dict["bias/coverage"] = (bias_grid > 1e-6).mean()
                    
                    # Plot bias analysis every 100 epochs (REMOVED: Moved to validation)
                    # if epoch % 100 == 0:
                    #     fig_bias = plot_bias_analysis(bias_potential, epoch, s_batch=s)
                    #     if fig_bias is not None:
                    #         log_dict["bias/analysis_plot"] = wandb.Image(fig_bias)
                    #         plt.close(fig_bias)
            
            # Add temperature and field statistics if available
            if log_dict_temp_field:
                log_dict.update(log_dict_temp_field)
            
            # Log lattice visualization periodically (every 100 epochs)
            if L is not None and epoch % 100 == 0:
                try:
                    # Get q from cfg_dict if available (for Potts model)
                    q = cfg_dict.get('q', None) if cfg_dict is not None else None
                    fig = _visualize_lattices(x_train, L, n_rows=2, n_cols=5, max_samples=10,
                                              beta_batch=beta_train, h_batch=h_train, q=q)
                    log_dict["train/samples"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    # Silently skip visualization if there's an error
                    pass
            
            wandb_run.log(log_dict, step=start_epoch + epoch)
        
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm_clip)
        optimizer.step()
        if ema is not None: ema.update(model.parameters())
        
        pbar.set_postfix(loss=info['loss'], ess=info['ess_train'])

        # Save checkpoint periodically
        if save_dir is not None and getattr(args, 'save_every', 0) > 0 and (start_epoch + epoch + 1) % args.save_every == 0:
            save_path = f"{save_dir}/ckpt_{start_epoch + epoch + 1}.pth"
            save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg_dict, save_path, bias_potential, replay_buffer)

        # Track and save best checkpoint by loss
        if save_dir is not None and save_best_metric == 'loss' and loss is not None:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = start_epoch + epoch + 1
                best_path = f"{save_dir}/best_model.pth"
                save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg_dict, best_path, bias_potential, replay_buffer)

        # Evaluate periodically
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Generate eval temp/field batches if using multiple temps/fields
                # Use starting temps/fields (fixed) for validation, not dynamically sampled ones
                eval_beta_batch = None
                eval_h_batch = None
                if use_multi_temp_field:
                    eval_beta_batch, eval_h_batch, _, _, _ = get_temp_field_batch(
                        starting_temps, starting_fields, args.eval_batch_size
                    )
                    eval_beta_batch = eval_beta_batch.to(device)
                    eval_h_batch = eval_h_batch.to(device)
                
                x, log_rnd = rnd(model, reward_fn, args.eval_batch_size, device=device,
                                 beta_batch=eval_beta_batch, h_batch=eval_h_batch, 
                                 J=args.J if hasattr(args, 'J') else 1)
                eval_ess = ess(log_rnd)
                ess_eval.append(eval_ess)
                # Evaluate metrics (always compute, log if wandb enabled)
                logf_t_vals, logp_x_vals = _compute_log_stats(x, log_rnd, reward_fn, model,
                                                               beta_batch=eval_beta_batch, h_batch=eval_h_batch,
                                                               J=args.J if hasattr(args, 'J') else 1,
                                                               bias_potential=bias_potential,
                                                               cv_compute_fn=cv_compute_fn)
                vfe = logp_x_vals - logf_t_vals  # variational free energy
                
                if wandb_run is not None:
                    # Use the new per-condition logging function (similar to snowyflow)
                    log_validation_metrics(
                        wandb_run=wandb_run,
                        logp_x=logp_x_vals,
                        logf_t=logf_t_vals,
                        vfe=vfe,
                        beta_batch=eval_beta_batch,
                        h_batch=eval_h_batch,
                        log_rnd=log_rnd,
                        step=start_epoch + epoch,
                    )
                    
                    # Also log ESS (log_rnd stats are included in log_validation_metrics)
                    wandb_run.log({"val/ess": eval_ess}, step=start_epoch + epoch)
                
                # Call validation plotting callback if provided (e.g. for energy/conc distributions)
                if validation_plot_callback is not None:
                    try:
                        # Convert beta/h to temps/fields for plotting
                        if eval_beta_batch is not None:
                            # beta = 1/(kB*T), so T = 1/(kB*beta)
                            if isinstance(eval_beta_batch, torch.Tensor):
                                plot_temps = 1.0 / (eval_beta_batch * K_B)
                            else:
                                plot_temps = torch.full((x.shape[0],), 1.0 / (eval_beta_batch * K_B), device=device, dtype=torch.float32)
                        else:
                            # Single temp case - need to get from args or use default
                            if hasattr(args, 'temp_min') and args.num_temps == 1:
                                plot_temps = torch.full((x.shape[0],), args.temp_min, device=device, dtype=torch.float32)
                            else:
                                plot_temps = torch.ones(x.shape[0], device=device, dtype=torch.float32)
                        
                        if eval_h_batch is not None:
                            plot_fields = eval_h_batch
                        else:
                            plot_fields = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
                        
                        validation_plot_callback(
                            x=x,
                            temps=plot_temps,
                            fields=plot_fields,
                            wandb_run=wandb_run,
                            step=start_epoch + epoch,
                        )
                    except Exception as e:
                        logging.warning(f"Validation plotting callback failed: {e}")
                
                # Log lattice visualization during evaluation
                if L is not None and wandb_run is not None:
                    try:
                        # Get q from cfg_dict if available (for Potts model)
                        q = cfg_dict.get('q', None) if cfg_dict is not None else None
                        fig = _visualize_lattices(x, L, n_rows=2, n_cols=5, max_samples=10,
                                                  beta_batch=eval_beta_batch, h_batch=eval_h_batch, q=q)
                        wandb_run.log({"val/samples": wandb.Image(fig)}, step=start_epoch + epoch)
                        plt.close(fig)
                    except Exception as e:
                        # Silently skip visualization if there's an error
                        pass

                # Plot bias analysis during validation (uses eval_batch_size)
                if bias_potential is not None:
                    try:
                        # Compute CV using provided function or default to Ising
                        if cv_compute_fn is not None:
                            s_eval = cv_compute_fn(x)  # Use provided CV computation function
                        else:
                            # Backward compatible: default to Ising magnetization
                            x_spins_eval = 2 * x - 1
                            s_eval = ising2d_mag(x_spins_eval)
                        
                        # Compute biased reward: R_biased = R_unbiased - beta * V(s)
                        # logf_t_vals contains R_unbiased values for the batch
                        
                        # Get V(s)
                        with torch.no_grad():
                            v_eval = bias_potential.evaluate(s_eval.to(device))
                            
                            # Get beta (use eval_beta_batch if available, else 1/T from bias_pot)
                            if eval_beta_batch is not None:
                                beta_val = eval_beta_batch
                            else:
                                beta_val = 1.0 / bias_potential.T
                                
                            biased_reward_vals = logf_t_vals - beta_val * v_eval
                            
                        # Sample from Replay Buffer for visualization if available
                        s_buffer = None
                        if replay_buffer is not None and replay_buffer.size > 0:
                            n_sample = min(args.eval_batch_size, replay_buffer.size)
                            x_buf, _, _ = replay_buffer.sample(n_sample)
                            with torch.no_grad():
                                if cv_compute_fn is not None:
                                    s_buffer = cv_compute_fn(x_buf.to(device))
                                else:
                                    # Backward compatible: default to Ising magnetization
                                    x_spins_buf = 2 * x_buf.to(device) - 1
                                    s_buffer = ising2d_mag(x_spins_buf)
                        
                        # Use custom plot function if provided, else default
                        plot_fn = plot_bias_fn if plot_bias_fn is not None else plot_bias_analysis
                        
                        fig_bias = plot_fn(bias_potential, epoch, s_batch=s_eval, 
                                                      biased_reward=biased_reward_vals, 
                                                      s_buffer=s_buffer,
                                                      num_sites=getattr(model, 'length', None))
                        if fig_bias is not None:
                            if wandb_run is not None:
                                wandb_run.log({"val/bias_analysis_plot": wandb.Image(fig_bias)}, step=start_epoch + epoch)
                            
                            # Also save locally if save_dir is available
                            if save_dir is not None:
                                plot_path = f"{save_dir}/bias_analysis_epoch_{start_epoch + epoch}.png"
                                fig_bias.savefig(plot_path, dpi=150)
                                # if (start_epoch + epoch) % 100 == 0: # Print only occasionally to avoid spam
                                #     print(f"Saved bias analysis plot to {plot_path}")
                                    
                            plt.close(fig_bias)
                    except Exception as e:
                        print(f"Error plotting bias analysis during val: {e}")
            model.train()
            
    return model, optimizer, ema, losses, ess_train, ess_eval


def train_ddp(model, optimizer, reward_fn, args, device, num_epochs = 10000, ema=None, scheduler=None,
          losses=None, ess_train=None, ess_eval=None):
    loss_fn = {'ce': loss_ce, 'lv': loss_lv, 're_rf': loss_re_rf,
               'wdce': loss_wdce}.get(args.loss_fn)
    if loss_fn is None:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    # continue recording the metrics from the last training
    losses = [] if losses is None else losses.copy()
    ess_train = [] if ess_train is None else ess_train.copy()
    ess_eval = [] if ess_eval is None else ess_eval.copy()
    
    # Only show progress bar on rank 0
    if dist.get_rank() == 0:
        pbar = tqdm(range(num_epochs))
    else:
        pbar = range(num_epochs)
        
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    x_saved, log_rnd_saved = None, None
    
    for epoch in pbar:
        model.train(); optimizer.zero_grad(); info = {}

        if args.loss_fn == 'wdce':
            with torch.no_grad():
                if x_saved is None or epoch % args.resample_every_n_step == 0:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    # Note: train_ddp doesn't have multi-temp/field setup, so beta_batch/h_batch will be None
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                else:
                    x, log_rnd = x_saved, log_rnd_saved

            # beta_batch and h_batch will be None in train_ddp (no multi-temp/field support yet)
            loss = loss_wdce(model, log_rnd, x,
                                num_replicates=args.wdce_num_replicates,
                                beta_batch=None, h_batch=None)
        else:
            x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
            loss = loss_fn(log_rnd)
                
        # Synchronize loss across processes
        dist.all_reduce(loss)
        
        # to average over all processes and accommodate for the reduced batch size
        loss = loss / dist.get_world_size() / dist.get_world_size()
        
        ess_train.append(ess(log_rnd))
        info['ess_train'] = ess_train[-1]
        info['loss'] = loss.item()
        losses.append(loss.item())
        
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm_clip)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        if ema is not None: ema.update(model.parameters())
        
        # Update progress bar only on rank 0
        if dist.get_rank() == 0:
            pbar.set_postfix(loss=info['loss'], ess=info['ess_train'])
        
        # Evaluate periodically
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x, log_rnd = rnd(model, reward_fn, args.eval_batch_size, device=device)
                eval_ess = ess(log_rnd)
                # Synchronize evaluation metrics across processes
                ess_eval.append(eval_ess)
            model.train()
            
    return model, optimizer, ema, losses, ess_train, ess_eval