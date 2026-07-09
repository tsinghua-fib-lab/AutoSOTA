import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import esm
from esm.model.esm2 import ESM2

# Import SparseAutoencoder
try:
    sae_path = Path(__file__).resolve().parent.parent / "training_sae"
    if str(sae_path) not in sys.path:
        sys.path.append(str(sae_path))
    from sae_model import SparseAutoencoder
except ImportError:
    print("Warning: Could not import SparseAutoencoder from ../training_sae")
    SparseAutoencoder = None

def load_esm_model(weights_path, device, num_layers=6, d_model=320):
    """Load ESM2 model from checkpoint weights. Matches CLT/PLT/SAE training modules.
    Wraps model in DataParallel if multiple GPUs are available."""
    import os
    
    # Convert string device to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CRITICAL: ESM weights not found at {weights_path}")
    
    print(f"Loading ESM weights from {weights_path}...")
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=num_layers,
        embed_dim=d_model,
        attention_heads=20,
        alphabet=alphabet,
        token_dropout=False
    )
    
    data = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" in data:
        data = data["model"]
    
    # Clean up key prefixes from different training scripts
    ckpt = {}
    for k, v in data.items():
        new_key = k.replace("encoder.sentence_encoder.", "").replace("encoder.", "")
        ckpt[new_key] = v
    
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    critical_missing = [k for k in missing if "layers" in k]
    if len(critical_missing) > 0:
        raise RuntimeError(f"CRITICAL: Missing layers: {critical_missing[:5]}")
    
    model.to(device).eval()
    
    # Wrap in DataParallel if multiple GPUs available
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    if num_gpus > 1:
        print(f"Wrapping ESM model in DataParallel for {num_gpus} GPUs")
        model = nn.DataParallel(model)
    else:
        print(f"Using single GPU/CPU: {device}")
    
    print("SUCCESS: ESM weights loaded correctly.")
    return model, alphabet


class LayerExtractor(nn.Module):
    """Extract activations from a specific ESM layer using native ESM API."""
    def __init__(self, model, layer_idx):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        
    def forward(self, tokens, *args, **kwargs):
        """tokens: (B, T) token IDs. Accepts *args, **kwargs for DataParallel compatibility."""
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[self.layer_idx], return_contacts=False)
            return results["representations"][self.layer_idx]

def get_layer_activations(tokenizer, plm, seqs, layer, device=None):
    """
    Get layer activations using ESM's native format (matching training).
    tokenizer: ESMTokenizerWrapper (has batch_converter and padding_idx)
    """
    if device is None:
        device = next(plm.parameters()).device

    # Tokenize using ESM's native format (like training)
    batch_tokens = tokenizer(seqs).to(device)
    attention_mask = (batch_tokens != tokenizer.padding_idx).long()
    
    # Unwrap/Wrap for DataParallel
    model_to_use = plm.module if isinstance(plm, nn.DataParallel) else plm
    extractor = LayerExtractor(model_to_use, layer)
    
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        extractor = nn.DataParallel(extractor)
    
    extractor.to(device)
    with torch.no_grad():
        activations = extractor(batch_tokens)
        
    return activations, attention_mask

def get_embeddings(seqs, esm_model, tokenizer, device, layer=5, batch_size=32):
    """
    Compute embeddings using ESM's native format (matching train_dms_probe.py).
    tokenizer: ESMTokenizerWrapper
    """
    sae_acts = []
    esm_model.eval()
    
    if device.type == "cuda": torch.cuda.empty_cache()
    
    # Handle DataParallel for ESM (unwrap if needed for direct calls)
    model_to_use = esm_model.module if isinstance(esm_model, nn.DataParallel) else esm_model
    
    for start in tqdm(range(0, len(seqs), batch_size), desc="Computing embeddings", leave=False):
        batch_seqs = seqs[start:start+batch_size]
        
        # 1. Tokenize using ESM's native format (like training)
        batch_tokens = tokenizer(batch_seqs).to(device)
        
        # 2. Get ESM layer activations (DataParallel handled automatically if esm_model is wrapped)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)
            batch_acts = results["representations"][layer]
            
            # Create mask
            batch_mask = (batch_tokens != tokenizer.padding_idx)
            
            # Remove CLS/EOS for SAE input
            core_acts = batch_acts[:, 1:-1, :].contiguous()
            core_mask = batch_mask[:, 1:-1].contiguous()
            # If no SAE, return ESM activations mean-pooled
            for i in range(len(batch_seqs)):
                valid_len = core_mask[i].sum()
                if valid_len > 0:
                    feats = core_acts[i, :valid_len, :]
                    sae_acts.append(feats.mean(0).cpu().numpy())
                else:
                    dim = core_acts.shape[-1]
                    sae_acts.append(np.zeros(dim))
                    
    return np.stack(sae_acts)

class ESMTokenizerWrapper:
    """
    Simple wrapper for ESM's batch_converter to match training format.
    Returns tokens directly (like training modules and ESMInference).
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.padding_idx = alphabet.padding_idx
    
    def __call__(self, seqs):
        """
        Tokenize sequences using ESM's native format (matching training/clt_module.py).
        
        Args:
            seqs: List of strings
        
        Returns:
            tokens: (B, T) tensor of token IDs (same format as training)
        """
        data = [("protein", seq) for seq in seqs]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens
