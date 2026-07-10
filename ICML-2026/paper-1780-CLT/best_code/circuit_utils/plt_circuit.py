import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import hashlib
try:
    sys.path.append('../training_transcoder')
    from plt_module import PLTLightningModule
except ImportError:
    PLTLightningModule = None
try:
    from .clt_circuit import CircuitDiscovererCLT
except ImportError:
    try:
        from clt_circuit import CircuitDiscovererCLT
    except ImportError:
        CircuitDiscovererCLT = None
try:
    from .esm_activation import ESM2ActivationCollector
except ImportError:
    try:
        from esm_activation import ESM2ActivationCollector
    except ImportError:
        ESM2ActivationCollector = None

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class CircuitDiscovererPLT(CircuitDiscovererCLT):
    """
    PLT Circuit Discoverer (Sequential / Backprop Through Time).
    Subclass of CircuitDiscovererCLT with PLT-specific reconstruction logic.
    """
    def __init__(self, device, ckpt_path=None, esm_weights_path=None):
        if PLTLightningModule is None:
            raise ImportError("Could not import PLTLightningModule. Check sys.path.")
        self.device = device
        self.ckpt_path = ckpt_path or os.environ.get("PLT_CHECKPOINT")
        self.esm_weights = esm_weights_path or os.environ.get("ESM_WEIGHTS")
        if not self.ckpt_path: 
            raise ValueError("PLT_CHECKPOINT env var not set")
        if not self.esm_weights:
            raise ValueError("ESM_WEIGHTS env var not set")
        print(f"Loading PLT from {self.ckpt_path}...")
        self.pl_module = PLTLightningModule.load_from_checkpoint(
            self.ckpt_path, map_location=device, esm2_weight=self.esm_weights, weights_only=False
        )
        self.pl_module.to(device)
        self.pl_module.eval()
        self.plt = self.pl_module.plt
        self.num_layers = self.plt.num_layers
        self.esm = self.pl_module.esm_model
        self.alphabet = self.pl_module.alphabet
        
        # Setup collector
        if ESM2ActivationCollector is None:
            raise ImportError("Could not import ESM2ActivationCollector.")
        self.collector = ESM2ActivationCollector(self.esm, self.alphabet)
        self.collector.register_hooks()

    def clear_cache(self):
        """Public method to clear the collector's cache."""
        self.collector.clear_cache()

    def _run_plt_forward(self, x_stack, active_nodes=None, retain_grad=False, freeze_attention=True, sequential=True, padding_mask=None):
        """
        PLT Sequential Forward Pass.
        x_stack: (B, L+1, T, H) - Contains embeddings through all layers
        Returns: (B, T, H), List of Latents
        """
        # sequential = True is a dummy variable for compatibility with CLT
        self.plt.eval()
        latents_list = []

        # Pre-calculate which latents influence each layer
        node_masks = None
        if active_nodes is not None:
            node_masks = []
            for l in range(self.num_layers):
                m = torch.zeros(self.plt.d_hidden, device=self.device)
                if l in active_nodes and len(active_nodes[l]) > 0:
                    m[list(active_nodes[l])] = 1.0
                node_masks.append(m.view(1, 1, -1))

        # PLT starts with Layer 0 (embeddings)
        x_curr_BTH = x_stack[:, 0, :, :]  # (B, T, H)
        # 1. Transpose for PLT: (B, T, H) -> (T, B, H)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        last_recon_TBH = None
        
        for l in range(self.num_layers):
            layer = self.esm.layers[l]

            # 2. Get ESM attention
            residual = x_curr_TBH
            if freeze_attention:
                # Use Ground Truth input for Attention
                x_gt_BTH = x_stack[:, l, :, :]
                x_gt_TBH = x_gt_BTH.transpose(0, 1)
                x_ln_gt = layer.self_attn_layer_norm(x_gt_TBH)
                x_attn_out, _ = layer.self_attn(
                    query=x_ln_gt, key=x_ln_gt, value=x_ln_gt,
                    key_padding_mask=padding_mask, need_weights=False
                )
                x_TBH = x_curr_TBH + x_attn_out
            else:
                x_TBH = layer.self_attn_layer_norm(x_curr_TBH)
                x_TBH, _ = layer.self_attn(
                    query=x_TBH, 
                    key=x_TBH, 
                    value=x_TBH,
                    key_padding_mask=padding_mask, 
                    need_weights=False, 
                    attn_mask=None
                )
                x_TBH = residual + x_TBH

            # 3. Encode PLT replacement for MLP (residual)
            residual = x_TBH
            x_mlp_in_TBH = layer.final_layer_norm(x_TBH)
            x_norm_TBH, mu, std = self.plt.LN(x_mlp_in_TBH)
            x_norm_TBH = x_norm_TBH - self.plt.b_pre[l]
            enc_TBD = self.plt.encoders[l](x_norm_TBH) + self.plt.b_enc[l]
            latents_TBD = self.plt.topK_activation(enc_TBD, k=self.plt.k)
            if retain_grad:
                latents_TBD.retain_grad()

            # 4. Apply Sparse Mask (Ablation)
            if node_masks is not None:
                latents_TBD = latents_TBD * node_masks[l]
                
            latents_list.append(latents_TBD)

            # 5. Decode and denormalize
            recon_TBH = (latents_TBD @ self.plt.decoders[l]) + self.plt.b_pre[l]
            recon_TBH = recon_TBH * std + mu
            last_recon_TBH = recon_TBH

            # 6. Update stream to next layer
            x_curr_TBH = residual + recon_TBH

        # Return (B, T, H)
        x_curr_BTH = x_curr_TBH.transpose(0, 1)
        recon_mlp_BTH = last_recon_TBH.transpose(0, 1)

        return x_curr_BTH, latents_list, recon_mlp_BTH

    def _get_reconstruction_with_layer_embedding(self, tokens, active_nodes=None, retain_grad=False, freeze_attention=True, source="mlp_output", sequential=True):
        """
        Runs PLT to get reconstruction of layer embedding.
        Overrides parent CLT method with PLT-specific logic.

        Args:
            source: 
                "mlp_output": Returns ONLY the reconstructed MLP component. (Default)
                "layer_output": Returns the full stream with final layer norm.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        # 0. Generate cache key
        token_bytes = tokens.cpu().numpy().tobytes()
        cache_key = hashlib.md5(token_bytes).hexdigest()
        
        # 1. Collect activations
        x_stack_SLH, _, x_mlp_out_SLH, x_clt_in_SLH, _ = self.collector.collect(tokens, cache_key=cache_key)
        self.collector.remove_hooks()
        if retain_grad:
            x_stack_SLH = x_stack_SLH.detach()
            x_mlp_out_SLH = x_mlp_out_SLH.detach()
            x_clt_in_SLH = x_clt_in_SLH.detach()
        # x_stack_SLH: (B*T, L+1, H)
        # x_mlp_out_SLH: (B*T, L, H)
        # x_clt_in_SLH: (B*T, L, H)
        B, T = tokens.shape
        depth, H = x_stack_SLH.shape[1:]
        padding_mask = (tokens == self.pl_module.alphabet.padding_idx) # (B, T)]

        # Reshape: (B*T, L+1, H) -> (B, T, L+1, H) -> (B, L+1, T, H)
        x_stack_reshaped_BLTH = x_stack_SLH.view(B, T, depth, H).permute(0, 2, 1, 3)
        # 2. Run PLT forward
        recon_stream_BTH, latents_list_L, recon_mlp_BTH = self._run_plt_forward(
            x_stack_reshaped_BLTH, 
            active_nodes=active_nodes, 
            retain_grad=retain_grad,
            freeze_attention=freeze_attention,
            padding_mask=padding_mask
        ) # each entry of latents_list_L is (T, B, D)
        self.collector.register_hooks()
        
        # 3. Determine Output
        if source == "mlp_output":
            return recon_mlp_BTH, latents_list_L
        elif source == "layer_output":
            return self.esm.emb_layer_norm_after(recon_stream_BTH), latents_list_L
        else:
            raise ValueError(f"Unknown source: {source}")

    def get_gradients(self, batch_seqs, torch_probe, cnn=False, freeze_attention=True, source="mlp_output", sequential=True):
        """
        Compute attribution of probe score w.r.t PLT Latents.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        self.plt.zero_grad()
        torch_probe.zero_grad()

        # 1. Forward Pass
        tokens_BT = self.pl_module.tokenize(batch_seqs)
        modified_emb_BTH, latents_list = self._get_reconstruction_with_layer_embedding(tokens_BT, retain_grad=True, freeze_attention=freeze_attention, source=source)
        self.collector.remove_hooks()
        
        # 2. Masking
        mask_BT = (tokens_BT != self.alphabet.cls_idx) & \
               (tokens_BT != self.alphabet.eos_idx) & \
               (tokens_BT != self.alphabet.padding_idx)
        mask_BT1 = mask_BT.float().unsqueeze(-1)
        
        # 3. Probe Forward
        if cnn:
            masked_emb_BTH = modified_emb_BTH * mask_BT1
            probe_input_BH = masked_emb_BTH[:, 1:-1, :] # (B, T-2, H) 
            # since we are removing CLS/EOS it's T-2
        else:
            probe_input_BH = (modified_emb_BTH * mask_BT1).sum(dim=1) / mask_BT1.sum(dim=1).clamp(min=1e-9)

        output = torch_probe(probe_input_BH)
        loss = output.sum()
        loss.backward()
        
        # 4. Collect Attribution
        results = {}
        for l, latents_TBD in enumerate(latents_list):
            if latents_TBD.grad is not None:
                attr = torch.abs(latents_TBD * latents_TBD.grad)
                results[l] = attr.sum(dim=(0, 1)).detach().cpu().numpy()
        self.collector.register_hooks()

        return results

    def run_ablation(self, batch_seqs, torch_probe, active_nodes=None, cnn=False, freeze_attention=True, source="mlp_output", sequential=True):
        """
        Run PLT forward pass with sparse circuit mask applied.
        """
        # sequential = True is a dummy variable for compatibility with CLT
        with torch.no_grad():
            tokens = self.pl_module.tokenize(batch_seqs)
            self.collector.remove_hooks()
            modified_emb_BTH, _ = self._get_reconstruction_with_layer_embedding(tokens, active_nodes=active_nodes, freeze_attention=freeze_attention, source=source)
            
            mask_BT = (tokens != self.alphabet.cls_idx) & \
                   (tokens != self.alphabet.eos_idx) & \
                   (tokens != self.alphabet.padding_idx)
            mask_BT1 = mask_BT.float().unsqueeze(-1)
            
            if cnn:
                masked_emb_BTH = modified_emb_BTH * mask_BT1
                probe_input = masked_emb_BTH[:, 1:-1, :] # removes CLS/EOS
                # probe_input is (B, T-2, H)
            else:
                probe_input = (modified_emb_BTH * mask_BT1).sum(dim=1) / mask_BT1.sum(dim=1).clamp(min=1e-9)
                # probe_input is (B, H)
            self.collector.register_hooks()
            
            return torch_probe(probe_input)