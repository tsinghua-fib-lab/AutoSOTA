import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math
import copy
from scipy.stats import spearmanr
from tqdm import tqdm
import os

class CNNProbe(nn.Module):
    def __init__(self, input_dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.layer_pre_head = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Length, Dim)
        x_t = x.permute(0, 2, 1) 
        out_t = self.layer_pre_head(x_t)
        out = out_t.permute(0, 2, 1) # (B, L, D)        
        pooled = out.mean(dim=1)
        return self.head(pooled)

def estimate_memmap_size(N, L, D):
    """Returns estimated size in GB."""
    # float32 = 4 bytes
    total_bytes = N * L * D * 4
    return total_bytes / (1024 ** 3)

# DMS wrapper
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, indices=None):
        """
        Args:
            embeddings: The master memmap (N_total, Length, Dim)
            labels: The labels for this specific split (N_subset,)
            indices: The global indices corresponding to this split (N_subset,)
        """
        self.embeddings = embeddings 
        self.labels = labels
        self.indices = indices # If None, assumes embeddings is already sliced

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Map subset index -> global index
        global_idx = self.indices[idx] if self.indices is not None else idx        
        if isinstance(self.embeddings, (np.memmap, np.ndarray)):
            emb = torch.from_numpy(self.embeddings[global_idx].copy())
        else:
            emb = self.embeddings[global_idx]
            
        return emb, self.labels[idx]
    
def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings_stack = torch.stack([torch.as_tensor(e) for e in embeddings]) # (B, L, D)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return embeddings_stack, labels

def get_lr_multiplier(training_step, warmup_steps, total_steps, max_lr, min_lr):
    if training_step < warmup_steps:
        return float(training_step) / float(max(1, warmup_steps))
    if training_step > total_steps:
        return min_lr / max_lr
    ratio_post_warmup = (training_step - warmup_steps) / (total_steps - warmup_steps)
    cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_post_warmup))
    target_lr = min_lr + cosine_scaler * (max_lr - min_lr)
    return target_lr / max_lr

def train_probe_cnn(
    train_dataset, 
    val_dataset, 
    input_dim, 
    device, 
    batch_size=128,
    total_steps=10000, 
    max_lr=3e-4, 
    min_lr=1e-5, 
    weight_decay=5e-2, 
    dropout=0.1, 
    kernel_size=7, 
    warmup_steps=100, 
    patience=10, 
    num_eval_steps=10,
    recycle_data=True, 
    # --- RESUME ARGS ---
    probe=None,
    optimizer=None,
    scheduler=None,
    start_step=0,
    best_val_loss=float('inf'),
    best_val_corr=0,
    patience_counter=0,
    best_state=None
):
    """
    Train CNN Probe.
    Returns: (probe, best_corr, training_state_dict, stop_early_flag)
    """
    # 1. Initialize or Use Existing
    if probe is None:
        probe = CNNProbe(input_dim, kernel_size=kernel_size, dropout=dropout).to(device)
    else:
        probe = probe.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(probe.parameters(), lr=max_lr, weight_decay=weight_decay)
    if scheduler is None:
        lr_lambda_fn = lambda step: get_lr_multiplier(step, warmup_steps, total_steps, max_lr, min_lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)
    loss_fn = nn.MSELoss()
    safe_drop_last = (len(train_dataset) >= batch_size)
    loader_kwargs = {
        "num_workers": 4, 
        "pin_memory": True, 
        "persistent_workers": True
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=safe_drop_last, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False, **loader_kwargs)

    # 2. Train
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training CNN")
    global_step = start_step
    train_iter = iter(train_loader)
    stop_early = False
    while global_step < total_steps:
        probe.train()
        
        # If recycle, feed training data back into iterator
        try:
            b_emb, b_y = next(train_iter)
        except StopIteration:
            if recycle_data: 
                train_iter = iter(train_loader)
                try:
                    b_emb, b_y = next(train_iter)
                except StopIteration:
                    print("WARNING: Dataset too small or empty even after restart. Breaking.")
                    break
            else:
                break 
            
        b_emb, b_y = b_emb.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
        
        preds = probe(b_emb)
        loss = loss_fn(preds, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        global_step += 1
        pbar.update(1)

        # Early stopping / Eval
        if global_step % num_eval_steps == 0:
            probe.eval()
            v_loss_accum = 0
            all_v_preds = []
            all_v_labels = []
            
            with torch.no_grad():
                for v_emb, v_y in val_loader:
                    v_emb, v_y = v_emb.to(device), v_y.to(device)
                    v_preds = probe(v_emb)
                    v_loss_accum += loss_fn(v_preds, v_y).item() * v_y.size(0)
                    all_v_preds.append(v_preds.cpu().numpy())
                    all_v_labels.append(v_y.cpu().numpy())
            
            all_v_preds = np.concatenate(all_v_preds).flatten()
            all_v_labels = np.concatenate(all_v_labels).flatten()
            avg_v_loss = v_loss_accum / len(val_dataset)
            corr, _ = spearmanr(all_v_labels, all_v_preds)
            
            pbar.set_postfix({"MSE": f"{avg_v_loss:.4f}", "rho": f"{corr:.3f}"})
            
            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                best_val_corr = corr
                patience_counter = 0 
                best_state = copy.deepcopy(probe.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                pbar.write(f"Early stopping triggered at step {global_step}")
                stop_early = True
                break

    # Save state for resuming
    state = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "start_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_corr": best_val_corr,
        "patience_counter": patience_counter,
        "best_state": best_state
    }
    
    if stop_early and best_state:
        probe.load_state_dict(best_state)
    
    if 'train_iter' in locals(): del train_iter
    if 'train_loader' in locals(): del train_loader
    if 'val_loader' in locals(): del val_loader
        
    return probe, best_val_corr, state, stop_early

def precompute_embeddings(
    sequences, 
    inference, 
    dms_name, 
    target_layer=-1, 
    indices=None, 
    suffix="", 
    batch_size=32, 
    cache_dir="embeddings_cache"
):
    """
    Computes MLP outputs for DMS.
    """
    os.makedirs(cache_dir, exist_ok=True)    
    seqs_to_process = sequences[indices] if indices is not None else sequences
    N = len(seqs_to_process)
    
    T_biological = len(seqs_to_process[0]) 
    D = inference.model.embed_dim
    if target_layer < 0:
        target_layer = inference.model.num_layers + target_layer

    # If suffix, only compute embeddings for those indices
    file_label = f"{dms_name}_{suffix}_L{target_layer}" if suffix else f"{dms_name}_all_L{target_layer}"
    path_data = os.path.join(cache_dir, f"{file_label}_mlp_seq.dat")
    shape = (N, T_biological, D)

    if os.path.exists(path_data):
        try:
            fp = np.memmap(path_data, dtype='float32', mode='r', shape=shape)
            print(f"[{dms_name}] Loading cached {file_label}...")
            return fp
        except ValueError:
            print(f"WARNING: Corrupted cache found at {path_data}. Deleting and recomputing...")
            os.remove(path_data)
        except Exception as e:
            print(f"WARNING: Error loading {path_data}: {e}. Recomputing...")
            os.remove(path_data)
    
    fp = np.memmap(path_data, dtype='float32', mode='w+', shape=shape)
    desc = f"Comp. {file_label}"    
    for i in tqdm(range(0, N, batch_size), desc=desc, leave=False):
        batch_seqs = seqs_to_process[i : i + batch_size]
        with torch.no_grad():
            tokens = inference.tokenize(batch_seqs)
            B, T_full = tokens.shape
            
            # Collect MLP output
            _, _, x_mlp_stack_flat, _, _ = inference.collector.collect(tokens)
            x_mlp_stack = x_mlp_stack_flat.view(B, T_full, -1, D)
            x_layer = x_mlp_stack[:, :, target_layer, :]
            
            # Strip CLS/EOS (1:-1)
            x_clean = x_layer[:, 1:-1, :]
            fp[i : i + len(batch_seqs)] = x_clean.cpu().numpy()
            
    fp.flush()
    return fp

def evaluate_circuit(discoverer, probe, data, y, circuit_nodes, batch_size=8, cnn=False, gt_embeddings=None, **kwargs):
    """
    Unified evaluation function for Regression (Spearman) + Reconstruction (NMSE).
    
    Args:
        discoverer: CLT/PLT Discoverer instance.
        probe: Trained probe (CNN or Linear).
        seqs: List of raw protein strings.
        y: Ground truth labels (regression targets).
        circuit_nodes: Active circuit nodes dict.
        gt_embeddings: (Optional) Ground truth embeddings for NMSE calculation. 
                       Can be a numpy array, memmap, or list of tensors.
        cnn: Boolean, auto-detected if not provided.
        **kwargs: specific flags like 'sequential', 'source', 'freeze_attention'.
        
    Returns:
        dict: {'spearman': float, 'nmse': float}
    """
    discoverer.pl_module.eval()
    probe.eval()

    preds_list = []
    total_squared_error = 0.0
    total_squared_norm = 0.0
    with torch.no_grad():
        pbar = tqdm(range(0, len(y), batch_size), desc="Evaluating Circuit", leave=False, disable=not tqdm.write)
        for i in pbar:
            batch_seqs = data[i : i+batch_size]  

            # 1. Reconstruct
            use_mean_pool = not cnn   
            recon_emb = discoverer.reconstruct_layer_embeddings(
                batch_seqs, 
                active_nodes=circuit_nodes, 
                mean_pool=use_mean_pool,
                **kwargs
            ).detach()

            # Probe Evaluation (Spearman)
            if probe is not None:
                if cnn:
                    probe_input = recon_emb[:, 1:-1, :]
                else:
                    probe_input = recon_emb
                output = probe(probe_input)
                preds_list.append(output)


            # NMSE Evaluation
            if gt_embeddings is not None:
                if hasattr(gt_embeddings, '__getitem__'):
                        batch_gt_np = gt_embeddings[i : i + batch_size]
                else:
                    batch_gt_np = gt_embeddings[i : i + batch_size] # List
                batch_gt = torch.as_tensor(batch_gt_np, device=recon_emb.device).float().detach()
                if recon_emb.ndim == 3 and batch_gt.ndim == 3:
                    if batch_gt.shape[1] == recon_emb.shape[1] - 2:
                        recon_for_nmse = recon_emb[:, 1:-1, :]
                    else:
                        recon_for_nmse = recon_emb
                else:
                    recon_for_nmse = recon_emb
                total_squared_error += torch.sum((recon_for_nmse - batch_gt)**2).item()
                total_squared_norm += torch.sum(batch_gt**2).item()
                del batch_gt
            del recon_emb
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
              
    results = {}
    if probe is not None and len(preds_list) > 0:
        all_preds = torch.cat(preds_list).cpu().numpy().flatten()
        min_len = min(len(all_preds), len(y))
        preds_slice = all_preds[:min_len]
        y_slice = y[:min_len]
        if len(np.unique(preds_slice)) <= 1:
            results['spearman'] = 0.0
        else:
            results['spearman'] = float(spearmanr(y_slice, preds_slice)[0])
    else:
        results['spearman'] = 0.0        
    if gt_embeddings is not None:
        results['nmse'] = total_squared_error / (total_squared_norm + 1e-9)
    else:
        results['nmse'] = float('nan')

    return results

def evaluate_probe_direct(probe, dataset_or_embs, y, device, batch_size=64):
    """
    Evaluate the probe directly on EmbeddingDataset.
    """
    probe.eval()
    preds = []
    total_steps = len(y) // batch_size + (1 if len(y) % batch_size != 0 else 0)
    pbar = tqdm(total=total_steps, desc="Direct Probe Eval", leave=False)
    # All embeddings present in one dataset
    if isinstance(dataset_or_embs, Dataset):
        loader = DataLoader(
            dataset_or_embs, 
            batch_size=batch_size, 
            collate_fn=collate_fn, 
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )
        with torch.no_grad():
            for b_emb, _ in loader:
                b_emb = b_emb.to(device)
                p = probe(b_emb)
                preds.append(p.flatten())
                pbar.update(1)

    # Chunk strategy
    else:
        N = len(y)
        is_memmap = isinstance(dataset_or_embs, np.memmap)
        with torch.no_grad():
            for i in range(0, N, batch_size):
                if is_memmap:
                    batch_data = dataset_or_embs[i : i+batch_size].copy()
                    b_emb = torch.from_numpy(batch_data).to(device, non_blocking=True)
                else:
                    batch_list = dataset_or_embs[i : i+batch_size]
                    b_emb = torch.stack([torch.as_tensor(e) for e in batch_list]).to(device)
                p = probe(b_emb) 
                preds.append(p.flatten())
                pbar.update(1)
                
    if len(preds) > 0:
        all_preds = torch.cat(preds).cpu().numpy().flatten()
        min_len = min(len(all_preds), len(y))
        preds_slice = all_preds[:min_len]
        y_slice = y[:min_len]
        if len(np.unique(preds_slice)) <= 1:
            return 0.0
        rho, _ = spearmanr(y_slice, preds_slice)
        return float(rho) if not np.isnan(rho) else 0.0
    
    return 0.0