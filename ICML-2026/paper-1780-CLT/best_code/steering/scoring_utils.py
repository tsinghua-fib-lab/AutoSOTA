
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import List, Callable, Any, Union
import os
import math

# Using CNNProbe from function_circuit/function_utils.py
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

def load_cnn(seq_len_unused, model_path, device):
    """
    Load the CNN Probe model.
    """
    # We assume model_path points to a .pt file containing the state_dict or the model.
    # We need input_dim. Usually 1280 for ESM-2 650M.
    # If the file contains the model object, we just load it.
    # If state_dict, we instantiate CNNProbe.
    
    # Heuristic: Try loading.
    data = torch.load(model_path, map_location=device)
    
    if isinstance(data, nn.Module):
        model = data
    elif isinstance(data, dict) and 'state_dict' in data:
         # Need params. Assume 1280dim if not specified? 
         # Or check keys.
         # For now, assume 1280.
         model = CNNProbe(input_dim=1280) # Default
         model.load_state_dict(data['state_dict'])
    elif isinstance(data, dict):
         # Maybe pure state dict
         # Check weight shape of head
         if 'head.weight' in data:
             input_dim = data['head.weight'].shape[1]
             model = CNNProbe(input_dim=input_dim)
             model.load_state_dict(data)
         else:
             # Fallback
             model = CNNProbe(input_dim=1280)
             try:
                 model.load_state_dict(data)
             except:
                 print("Warning: Could not load state dict cleanly into CNNProbe.")
    else:
        # Fallback
        model = data
        
    model.to(device).eval()
    return model

def score_cnn(sequences, model, tokenizer=None, esm_model=None, batch_size=32, layer=5):
    """
    Score sequences using the CNN Probe.
    Requires esm_model and tokenizer to compute embeddings first.
    """
    if tokenizer is None or esm_model is None:
        raise ValueError("score_cnn requires tokenizer and esm_model for embedding-based probes.")
        
    embeddings = get_sequence_embeddings(sequences, esm_model, tokenizer, batch_size=batch_size, layer=layer)
    # embeddings: (N, L, D) - padded numpy array? Or list of arrays?
    # CNNProbe expects (Batch, Length, Dim) tensor.
    # Since sequences can be diff lengths, usually we pad or process 1 by 1 or batch with padding.
    # get_sequence_embeddings below returns a list of (L, D) arrays usually if lengths differ, 
    # or a padded array if we handle it.
    
    # If get_sequence_embeddings returns a stacked array (implying same length), we are good.
    # If sequences differ in length, we must pad.
    
    device = next(model.parameters()).device
    preds = []
    
    # Convert/Pad
    # Assume get_sequence_embeddings returns list of arrays [L, D]
    if isinstance(embeddings, list):
        # Batch process
        for i in range(0, len(embeddings), batch_size):
            batch_list = embeddings[i:i+batch_size]
            # Pad
            max_len = max([b.shape[0] for b in batch_list])
            dim = batch_list[0].shape[1]
            batch_tensor = torch.zeros(len(batch_list), max_len, dim, device=device)
            for j, arr in enumerate(batch_list):
                l = arr.shape[0]
                batch_tensor[j, :l, :] = torch.from_numpy(arr).to(device)
            
            with torch.no_grad():
                out = model(batch_tensor)
                preds.append(out.cpu().numpy().flatten())
    else:
        # Array (N, L, D)
        for i in range(0, len(embeddings), batch_size):
            batch_np = embeddings[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_np).to(device)
            with torch.no_grad():
                out = model(batch_tensor)
                preds.append(out.cpu().numpy().flatten())
                
    return np.concatenate(preds)

def calculate_relative_fitness(mutant_scores, wt_sequence, score_function, model, tokenizer=None, esm_model=None, output_dir=None):
    """
    Calculates Relative Fitness scores.
    """
    # 1. Calculate Wildtype (W) score
    if tokenizer and esm_model:
        wt_score_raw = score_function([wt_sequence], model, tokenizer=tokenizer, esm_model=esm_model)
    else:
        wt_score_raw = score_function([wt_sequence], model)
    
    if isinstance(wt_score_raw, (np.ndarray, list)) and len(wt_score_raw) > 0:
        wildtype_score = float(wt_score_raw[0])
    else:
        wildtype_score = float(wt_score_raw)
        
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            wt_score_path = os.path.join(output_dir, "WT.npy")
            np.save(wt_score_path, wildtype_score)
        except Exception as e:
            print(f"  Warning: Could not save WT score. Error: {e}")

    mutant_scores = np.asarray(mutant_scores)
    log_relative_fitness = mutant_scores - wildtype_score
    relative_fitness_scores = np.exp(log_relative_fitness)
    
    return relative_fitness_scores

# --- Embedding Extraction Utilities ---

def get_sequence_embeddings(sequences, esm_model, tokenizer, batch_size=32, layer=5):
    """
    Returns list of embeddings (L, D) for each sequence.
    Does NOT mean pool.
    """
    from steering.gen_utils import get_layer_activations
    
    device = next(esm_model.parameters()).device
    if device.type == "cuda": torch.cuda.empty_cache()
    
    out_list = []
    
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start+batch_size]
        # get_layer_activations usually returns list of tensors? 
        # Or tensor if padded.
        # Based on gen_utils in this project: 
        # get_layer_activations(tokenizer, plm, seqs, layer, x) -> activations (padded tensor), mask
        
        acts, mask = get_layer_activations(tokenizer, esm_model, batch, layer=layer, device=device)
        
        # Unpack
        for i in range(len(batch)):
            # Trim CLS/EOS (1:-1)
            valid_len = mask[i].sum()
            # 1:-1 means start at 1, end at valid_len-1
            # Check length
            if valid_len <= 2:
                # Should not happen for proteins
                seq_emb = np.zeros((1, acts.shape[-1]))
            else:
                seq_emb = acts[i, 1:valid_len-1, :].cpu().numpy()
            out_list.append(seq_emb)
            
    return out_list

def _get_sequence_embeddings_fallback(sequences, esm_model, tokenizer, batch_size, layer):
    # Minimal implementation if gen_utils not found
    device = next(esm_model.parameters()).device
    out_list = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start+batch_size]
        # Tokenizer returns tokens directly (matching training format)
        batch_tokens = tokenizer(batch).to(device)
        padding_idx = tokenizer.padding_idx if hasattr(tokenizer, 'padding_idx') else tokenizer.alphabet.padding_idx
        mask = (batch_tokens != padding_idx).long()
        
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)
            acts = results["representations"][layer]
        
        for i in range(len(batch)):
            l = mask[i].sum()
            seq_emb = acts[i, 1:l-1, :].cpu().numpy()
            out_list.append(seq_emb)
    return out_list


# --- Legacy / Linear Probe Utils ---

def load_probe_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        ridge_model = model_data['ridge_model']
    else:
        ridge_model = model_data
    
    return ridge_model

def score_probe_model(sequences, probe_model, tokenizer, esm_model, sae_model=None, batch_size=32, layer=5, embedding_dim=3200, logits=False):
    ridge_model = probe_model
    if logits:
        embeddings = get_logits_embeddings(sequences, esm_model, tokenizer, batch_size=batch_size, embedding_dim=embedding_dim)
    else:
        embeddings = get_embeddings(sequences, esm_model, tokenizer, sae_model, batch_size=batch_size, layer=layer, embedding_dim=embedding_dim)
    predictions = ridge_model.predict(embeddings)
    return predictions

def get_embeddings(sequences, esm_model, tokenizer, sae_model=None, batch_size=32, layer=5, embedding_dim=3200):
    """
    Mean pooled embeddings for linear probe.
    Handles DataParallel for both ESM (via get_sequence_embeddings) and SAE.
    """
    from steering.gen_utils import SAEInference
    import torch.nn as nn
    
    seq_embs = get_sequence_embeddings(sequences, esm_model, tokenizer, batch_size, layer)
    
    # Pool
    pooled = []
    device = next(esm_model.parameters()).device
    
    # Wrap SAE in SAEInference (SAE may already be wrapped in DataParallel)
    sae_inference = None
    if sae_model:
        # Unwrap if already wrapped in DataParallel (SAEInference expects unwrapped model)
        sae_to_wrap = sae_model.module if isinstance(sae_model, nn.DataParallel) else sae_model
        sae_inference = SAEInference(sae_to_wrap)
        # If SAE was already wrapped, wrap SAEInference too
        if isinstance(sae_model, nn.DataParallel):
            sae_inference = nn.DataParallel(sae_inference)
        sae_inference.to(device).eval()
    
    for emb in seq_embs:
        emb_tensor = torch.from_numpy(emb).to(device) # Move to device for SAE
        if sae_inference is not None:
            # Use SAE with DataParallel support
            # emb_tensor is (L, D_esm), need to add batch dimension
            emb_batch = emb_tensor.unsqueeze(0)  # (1, L, D_esm)
            sae_out = sae_inference(emb_batch)  # (1, L, D_sae)
            sae_out = sae_out.squeeze(0)  # (L, D_sae)
            pooled.append(sae_out.mean(0).detach().cpu().numpy())
        else:
             pooled.append(emb_tensor.mean(0).detach().cpu().numpy())
             
    return np.array(pooled)

def get_logits_embeddings(seqs, esm_model, tokenizer, batch_size=64, embedding_dim=None):
    device = next(esm_model.parameters()).device
    if embedding_dim is None:
        embedding_dim = esm_model.config.vocab_size
    out_NV = [] 
    if device.type=="cuda": torch.cuda.empty_cache()
    for start in tqdm(range(0,len(seqs),batch_size),desc="Computing logits",leave=False):
        batch_B = seqs[start:start+batch_size]
        # Tokenizer returns tokens directly (matching training format)
        batch_tokens = tokenizer(batch_B).to(device)
        
        with torch.inference_mode():
            outputs = esm_model(batch_tokens)
            logits_BTV = outputs.logits
        for i in range(logits_BTV.size(0)): 
            seq_logits_TV = logits_BTV[i,1:-1] # drop CLS/EOS
            out_NV.append(seq_logits_TV.mean(0).detach().cpu().numpy() if seq_logits_TV.numel() else
                       torch.zeros(embedding_dim).numpy())
    return np.vstack(out_NV)
