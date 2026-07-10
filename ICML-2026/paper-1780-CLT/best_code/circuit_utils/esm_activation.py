import torch
import os
import esm
from esm.model.esm2 import ESM2

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class ESM2ActivationCollector:
    """
    Grabs MLP inputs, outputs, and layer embeddings.
    More complicated than the version in training/clt_module.py
    """
    def __init__(self, esm2_model, alphabet, target_layers=None):
        self.model = esm2_model
        self.alphabet = alphabet
        self.target_layers = target_layers if target_layers else list(range(len(esm2_model.layers)))
        self.activations = {} 
        self.hooks = []
        self.cache = {} # Storage for pre-computed activations

    def clear_cache(self):
        """Removes cached tensors to free up GPU memory."""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_hook(self, key, scale=1.0, transpose=False, capture_input=False):
        """
        Creates a hook that:
        1. Extracts tensor from tuple (if needed)
        2. Scales the tensor (for embeddings)
        3. Transposes (T, B, H) -> (B, T, H) (for layers)
        """
        def hook(module, input, output):
            if capture_input:
                data = input[0]
            else:
                if isinstance(output, tuple):
                    data = output[0]
                else:
                    data = output
            if scale != 1.0:
                data = data * scale                
            # (T, B, H) -> (B, T, H)
            if transpose:
                data = data.transpose(0, 1)                
            self.activations[key] = data.detach().to("cpu", dtype=torch.float32, non_blocking=True)
        return hook

    def register_hooks(self):
        self.remove_hooks()
        # 1. Hook Embeddings (B, T, H)
        scale = self.model.embed_scale
        embed_hook = self.model.embed_tokens.register_forward_hook(
            self._make_hook(-1, scale=scale, transpose=False)
        ) 
        self.hooks.append(embed_hook)

        # 2. Hook Layers (T, B, H) -> (B, T, H)
        for layer_idx in self.target_layers:
            layer_module = self.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(
                self._make_hook(layer_idx, scale=1.0, transpose=True)
            )
            self.hooks.append(hook)
        
        # 3. Hook MLP Inputs (residual stream before and after final_layer_norm)
        for layer_idx in self.target_layers:
            mlp_input = self.model.layers[layer_idx].final_layer_norm
            # 3a. Hook the INPUT to final_layer_norm (residual stream before normalization)
            hook = mlp_input.register_forward_hook(
                self._make_hook(f"mlp_input_{layer_idx}", scale=1.0, transpose=True, capture_input=True)
            )
            self.hooks.append(hook)
            # 3b. Hook the OUTPUT of final_layer_norm (post-LN residual stream used for CLT inference)
            hook = mlp_input.register_forward_hook(
                self._make_hook(f"clt_input_{layer_idx}", scale=1.0, transpose=True, capture_input=False)
            )
            self.hooks.append(hook)
        
        # 4. Hook MLP Outputs (fc2 output, after MLP)
        for layer_idx in self.target_layers:
            fc2 = self.model.layers[layer_idx].fc2
            hook = fc2.register_forward_hook(
                self._make_hook(f"mlp_{layer_idx}", scale=1.0, transpose=True)
            )
            self.hooks.append(hook)

    def collect(self, input_ids, cache_key=None):
        """
        input_ids: (B, T)
        Returns:
        x_stack_flat_SLH: (B*T, L+1, H) - embeddings + full layer outputs
        x_mlp_input_stack_flat_SLH: (B*T, L, H) - pre-LN residual stream (MLP inputs)
        x_mlp_stack_flat_SLH: (B*T, L, H) - MLP outputs
        x_clt_input_stack_flat_SLH: (B*T, L, H) - post-LN residual stream for CLT inference
        mask_S: (B*T,)
        cache_key: Unique identifier for the batch (e.g., a hash of input_ids).
        """
        if cache_key is not None and cache_key in self.cache:
            device = input_ids.device
            return tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t 
                         for t in self.cache[cache_key])
        
        self.activations.clear()
        with torch.no_grad():
            self.model(tokens=input_ids, repr_layers=[])            
        if not self.activations:
            raise RuntimeError("Collector failed: No activations captured.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        B, T = input_ids.shape
        device = input_ids.device
        def get_flat_stack(keys, depth):
            if not keys: return None
            # Stack on CPU
            stack_cpu = torch.stack([self.activations[k] for k in keys], dim=1) 
            # (B, L, T, H) -> (B, T, L, H) -> (S, L, H)
            # Move to GPU and cast back to float32 only for the specific batch being processed
            return stack_cpu.permute(0, 2, 1, 3).reshape(B * T, depth, -1).to(device=device, dtype=torch.float32)
        
        # 1. Main trajectory: embeddings + full layer outputs (integer keys)
        traj_keys = sorted([k for k in self.activations.keys() if isinstance(k, int)])
        x_stack_flat_SLH = get_flat_stack(traj_keys, len(traj_keys))
        
        # 2. MLP inputs (pre-LN residual stream, mlp_input_0, mlp_input_1, ...)
        mlp_in_keys = sorted([k for k in self.activations.keys() if "mlp_input_" in str(k)], 
                             key=lambda x: int(str(x).split("_")[-1]))
        x_mlp_input_stack_flat_SLH = get_flat_stack(mlp_in_keys, len(mlp_in_keys))
        
        # 3. MLP outputs (mlp_0, mlp_1, ...)
        mlp_keys = sorted([k for k in self.activations.keys() if "mlp_" in str(k) and "input" not in str(k)], 
                          key=lambda x: int(str(x).split("_")[-1]))
        x_mlp_stack_flat_SLH = get_flat_stack(mlp_keys, len(mlp_keys))

        # 4. CLT inputs (post-LN residual stream, clt_input_0, clt_input_1, ...)
        clt_keys = sorted([k for k in self.activations.keys() if "clt_input_" in str(k)], 
                          key=lambda x: int(str(x).split("_")[-1]))
        x_clt_input_stack_flat_SLH = get_flat_stack(clt_keys, len(clt_keys))

        # 5. Mask
        mask_BT = (input_ids != self.alphabet.padding_idx) & \
                  (input_ids != self.alphabet.cls_idx) & \
                  (input_ids != self.alphabet.eos_idx)
        mask_S = mask_BT.view(-1)

        results = (
            x_stack_flat_SLH,
            x_mlp_input_stack_flat_SLH,
            x_mlp_stack_flat_SLH,
            x_clt_input_stack_flat_SLH,
            mask_S,
        )

        if cache_key is not None:
            self.cache[cache_key] = tuple(t.to("cpu", non_blocking=True) if torch.is_tensor(t) else t 
                for t in results)
            
        return results
    
    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []


class ESMInference:
    """
    Wrapper to get embeddings for training the probe.
    """
    def __init__(self, device, esm_weights_path=None, num_layers=6, d_model=320):
        self.device = device
        if esm_weights_path is None:
            esm_weights_path = os.environ.get("ESM_WEIGHTS")
        
        print(f"Initializing ESM2 on {device}...")
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = ESM2(
            num_layers=num_layers, 
            embed_dim=d_model, 
            attention_heads=20, 
            alphabet=self.alphabet, 
            token_dropout=False
        )
        
        self._load_esm_weights(esm_weights_path)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False
        
        self.collector = ESM2ActivationCollector(self.model, self.alphabet)
        self.collector.register_hooks()

    def _load_esm_weights(self, esm_pretrained: str):
        if not os.path.exists(esm_pretrained):
            raise FileNotFoundError(f"CRITICAL: ESM weights not found at {esm_pretrained}")

        print(f"Loading ESM weights from {esm_pretrained}...")
        data = torch.load(esm_pretrained, map_location="cpu", weights_only=False)
        if "model" in data: data = data["model"]

        ckpt = {}
        for k, v in data.items():
            new_key = k.replace("encoder.sentence_encoder.", "").replace("encoder.", "")
            ckpt[new_key] = v

        self.model.load_state_dict(ckpt, strict=False)
        print("SUCCESS: ESM weights loaded correctly.")

    def tokenize(self, seqs):
        data = [("protein", seq) for seq in seqs]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.to(self.device)

    def get_embeddings(self, batch_seqs, target_layer=-1, mean_pool=True, source="layer_output", keep_cls_eos=False):
        """
        Returns Final Layer Embeddings.
        If keep_cls_eos is True, the CLS and EOS tokens are not masked out.
        
        Args:
            target_layer: Which layer to extract. -1 means last layer (default).
            mean_pool: 
                If True: returns (B, H) - Mean over valid tokens (excludes CLS/EOS/PAD).
                If False: returns (B, T, H) - Full sequence with CLS/EOS/PAD masked to 0.
            source: 
                "layer_output": The standard residual stream output of the layer.
                                          Applying final layer norm.
                "mlp_output": The direct output of the MLP block (fc2).
                              Does NOT apply final layer norm (raw activations).
        """
        tokens = self.tokenize(batch_seqs)

        # 1. Collect
        x_stack_flat_SLH, _, x_mlp_stack_flat_SLH, _, mask_flat = self.collector.collect(tokens)
        B, T = tokens.shape
        H = x_stack_flat_SLH.shape[-1]

        # 2. Select Source Stack
        if source == "layer_output":
            # Reshape (S, L+1, H) -> (B, T, L+1, H)
            stack_reshaped = x_stack_flat_SLH.view(B, T, -1, H)
            # Extract target_layer
            if target_layer >= 0:
                stack_idx = target_layer + 1  # offset for embeddings at index 0
            else:
                stack_idx = target_layer  
            raw_output_BTH = stack_reshaped[:, :, stack_idx, :]
            # Apply Final Layer Norm
            if target_layer == -1:
                final_output_BTH = self.model.emb_layer_norm_after(raw_output_BTH)
            else:
                final_output_BTH = raw_output_BTH

        elif source == "mlp_output":
            # Reshape (S, L, H) -> (B, T, L, H)
            stack_reshaped = x_mlp_stack_flat_SLH.view(B, T, -1, H)
            # Extract target_layer
            raw_output_BTH = stack_reshaped[:, :, target_layer, :]
            final_output_BTH = raw_output_BTH

        else:
            raise ValueError(f"Unknown source '{source}'. Use 'layer_output' or 'mlp_output'.")

        # 3. Semantic Masking (Exclude CLS, EOS, PAD)
        mask_BT = (tokens != self.alphabet.cls_idx) & \
                  (tokens != self.alphabet.eos_idx) & \
                  (tokens != self.alphabet.padding_idx)
        mask_BT1 = mask_BT.float().unsqueeze(-1)  # (B, T, 1)

        # 4. Pooling or masked sequence
        if mean_pool:
            sum_emb_BH = (final_output_BTH * mask_BT1).sum(dim=1)
            sum_mask_B1 = mask_BT1.sum(dim=1).clamp(min=1e-9)
            mean_emb_BH = sum_emb_BH / sum_mask_B1
            return mean_emb_BH.cpu().numpy()
        else:
            # Return full sequence with CLS/EOS/PAD positions zeroed out
            if keep_cls_eos:
                return final_output_BTH.cpu().numpy()
            else:
                masked_emb = final_output_BTH * mask_BT1
                return masked_emb.cpu().numpy()