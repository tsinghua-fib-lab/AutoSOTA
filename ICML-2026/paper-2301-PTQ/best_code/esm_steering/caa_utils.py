import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import esm
from esm.model.esm2 import ESM2

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# L: Number of layers of LM
# T: sequence length
# H: Embedding Dimension of LM (d_model)
# V: model vocab dimension
# ──────────────────────────────────────────────────────────────────────────────

# go back 1 folder and then load from circuit_utils, the esm_activation file
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Direct import - add circuit_utils to path and import
circuit_utils_path = os.path.join(repo_root, 'circuit_utils')
if circuit_utils_path not in sys.path:
    sys.path.insert(0, circuit_utils_path)

from esm_activation import ESMInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_weights_path = "../models/esm2_t6_8M_UR50D.pt"
# Lazy initialization - only create if not provided
_esm_inference = None

def get_esm_inference(esm_inference=None, esm_weights_path=None):
    """Get ESMInference instance, creating if necessary. Only creates once.
    Wraps model in DataParallel if multiple GPUs are available."""
    global _esm_inference
    if esm_inference is not None:
        return esm_inference
    if _esm_inference is None:
        weights = esm_weights_path or globals()['esm_weights_path']
        esm_filename = os.path.basename(weights)
        num_layers = int(esm_filename.split("_")[1][1:])
        d_model = 320 if "8M" in esm_filename else 480 if "35M" in esm_filename else None
        print(f"Loading ESM model from {weights}...")
        _esm_inference = ESMInference(device, weights, num_layers=num_layers, d_model=d_model)
        print("ESM model loaded successfully.")
    else:
        # Model already loaded, just return it
        pass
    return _esm_inference

def get_caa_vector(pos_seqs, neg_seqs, num_layers=None, batch_size=128):
    '''
    Gets the steering vector using CAA method.
    pos_seqs: list of positive sequences (all same length)
    neg_seqs: list of negative sequences (all same length)
    num_layers: Number of layers in ESM model
    batch_size: number of sequences to process at once (to avoid OOM)
    Returns:
        steering_vector: List length L, each entry (H,) numpy array (no CLS/EOS)
    '''
    esm_inference = get_esm_inference()
    if num_layers is None:
        num_layers = len(esm_inference.model.layers)
    print(f"Pos seqs: {len(pos_seqs)}, Neg seqs: {len(neg_seqs)}")
    # Process positive sequences in batches
    positive_vecs = []
    for layer in range(num_layers):
        sum_pos_H = None
        for i in range(0, len(pos_seqs), batch_size):
            batch = pos_seqs[i:i+batch_size]
            batch_emb = esm_inference.get_embeddings(batch, target_layer=layer, mean_pool=True, source="layer_output", keep_cls_eos=False)
            # length of sequences in batches print
            if sum_pos_H is None:
                sum_pos_H = batch_emb.sum(axis=0)
            else:
                sum_pos_H += batch_emb.sum(axis=0)
        positive_vecs.append(torch.from_numpy(sum_pos_H / len(pos_seqs)).to(device))
    positive_vecs_LH = torch.stack(positive_vecs)
    
    # Process negative sequences in batches
    negative_vecs = []
    for layer in range(num_layers):
        sum_neg_H = None
        for i in range(0, len(neg_seqs), batch_size):
            batch = neg_seqs[i:i+batch_size]
            batch_emb = esm_inference.get_embeddings(batch, target_layer=layer, mean_pool=True, source="layer_output", keep_cls_eos=False)
            if sum_neg_H is None:
                sum_neg_H = batch_emb.sum(axis=0)
            else:
                sum_neg_H += batch_emb.sum(axis=0)
        negative_vecs.append(torch.from_numpy(sum_neg_H / len(neg_seqs)).to(device))
    negative_vecs_LH = torch.stack(negative_vecs)

    steering_vector_LH = positive_vecs_LH - negative_vecs_LH
    return steering_vector_LH

def steer_caa_sequence(wt_seq, alphas, steering_vector_LH):
    """
    wt_seq: str
    alphas: list[float] length B
    steering_vector_LH: (L, H) numpy or torch
    returns logits_BTV: (B, T, V) including CLS/EOS positions (same as model.lm_head input)
    """
    esm_inference = get_esm_inference()

    # Unwrap DataParallel if needed
    model = esm_inference.model.module if isinstance(esm_inference.model, nn.DataParallel) else esm_inference.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    B = len(alphas)
    alphas_1B1 = torch.tensor(alphas, device=device, dtype=dtype).view(1, B, 1)  # (1, B, 1)

    with torch.no_grad():
        tokens_1T = esm_inference.tokenize([wt_seq]).to(device)  # (1, T)

        # Embed: (1, T, H)
        hidden_1TH = model.embed_scale * model.embed_tokens(tokens_1T)

        # Transformer expects (T, B, H)
        hidden_T1H = hidden_1TH.transpose(0, 1)        # (T, 1, H)
        hidden_TBH = hidden_T1H.repeat(1, B, 1)        # (T, B, H)

        # Apply steering at each layer (pre-layer), with tokenwise norm preservation
        L = len(model.layers)
        for l, layer_module in enumerate(model.layers):
            pre_norm_TB1 = torch.norm(hidden_TBH, dim=-1, keepdim=True)  # (T, B, 1)

            v_H = torch.as_tensor(steering_vector_LH[l], device=device, dtype=dtype)  # (H,)
            hidden_TBH[1:-1, :, :] += alphas_1B1 * v_H.view(1, 1, -1)                  # broadcast to (T,B,H)

            hidden_TBH = hidden_TBH / (torch.norm(hidden_TBH, dim=-1, keepdim=True) + 1e-10) * pre_norm_TB1

            hidden_TBH, _ = layer_module(hidden_TBH, self_attn_padding_mask=None)

        hidden_TBH = model.emb_layer_norm_after(hidden_TBH)
        hidden_BTH = hidden_TBH.transpose(0, 1)  # (B, T, H)
        logits_BTV = model.lm_head(hidden_BTH)   # (B, T, V)

    return logits_BTV

def score_sequences_mlp(sequences, cnn_model, batch_size=32):
    """
    Score sequences using MLP output embeddings (source="mlp_output").
    Matches run_probe_steering.py logic but batched.
    
    Args:
        sequences: List of strings (should be same length for correct batching)
        cnn_model: PyTorch model (CNNProbe)
        batch_size: Batch size for inference
    
    Returns:
        scores: (N,) numpy array of scores
    """
    inference = get_esm_inference()
    device = next(cnn_model.parameters()).device
    
    scores = []
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        try: 
            # Get embeddings: (B, T, H) - unpooled, no CLS/EOS
            # Using mlp_output to match training of eval models
            embs_np = inference.get_embeddings(
                batch, 
                target_layer=-1, 
                mean_pool=False, 
                source="layer_output", 
                keep_cls_eos=False
            )
            
            # Convert to tensor for CNNProbe
            # CNNProbe expects (B, T, H)
            embs_tensor = torch.tensor(embs_np, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                batch_scores = cnn_model(embs_tensor) # (B, 1)
                
            scores.extend(batch_scores.flatten().cpu().numpy().tolist())
        except Exception as e:
            print(f"Error in score_sequences_mlp batch {i}: {e}")
            scores.extend([np.nan] * len(batch))
        
    return np.array(scores)