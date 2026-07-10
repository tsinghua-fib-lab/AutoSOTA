import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import hashlib
try:
    sys.path.append('../training')
    from clt_module import CLTLightningModule
except ImportError:
    CLTLightningModule = None
try:
    sys.path.append('../training_block')
    from block_clt_module import CLTLightningModule as BlockCLTLightningModule
except ImportError:
    BlockCLTLightningModule = None
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

class CircuitDiscovererCLT:
    def __init__(self, device, ckpt_path=None, esm_weights_path=None, model_type="clt"):
        self.device = device
        self.ckpt_path = ckpt_path or os.environ.get("CLT_CHECKPOINT")
        self.esm_weights = esm_weights_path or os.environ.get("ESM_WEIGHTS")
        if not self.ckpt_path or not self.esm_weights:
            raise ValueError("CLT_CHECKPOINT or ESM_WEIGHTS not set.")
        if model_type == "block_clt":
            if BlockCLTLightningModule is None:
                raise ImportError("Could not import BlockCLTLightningModule. Check sys.path.")
            ModuleClass = BlockCLTLightningModule
        else:
            if CLTLightningModule is None:
                raise ImportError("Could not import CLTLightningModule. Check sys.path.")
            ModuleClass = CLTLightningModule
        print(f"Loading {model_type} from {self.ckpt_path}...")
        try:
            self.pl_module = ModuleClass.load_from_checkpoint(
                self.ckpt_path, map_location=device, esm2_weight=self.esm_weights, weights_only=False
            )
        except Exception as e:
            raise ValueError(f"Could not load {model_type} from {self.ckpt_path}")
        self.pl_module.to(device)
        self.pl_module.eval()
        self.clt = self.pl_module.clt
        self.num_layers = self.clt.num_layers
        self.esm = self.pl_module.esm_model
        if ESM2ActivationCollector is None:
            raise ImportError("Could not import ESM2ActivationCollector.")
        self.collector = ESM2ActivationCollector(self.esm, self.pl_module.alphabet)
        self.collector.register_hooks()

    def clear_cache(self):
        """Public method to clear the collector's cache."""
        self.collector.clear_cache()

    def _run_clt_sequential(self, x_stack, active_nodes=None, retain_grad=False, freeze_attention=True, padding_mask=None):
        """
        CLT Sequential Forward Pass.
        x_stack: (B, L+1, T, H) - Contains embeddings through all layers
        Returns:
            x_curr_BTH: Modified stream (Input + Attn + MLP_Recon)
            latents_list_L: List of latents
            recon_mlp_BTH: The MLP reconstruction of the final layer (just the MLP part)

        Args:
            freeze_attention (bool): If True, calculates Attention using Ground Truth inputs (x_stack).
                                     If False, calculates Attention using the drifting stream (x_curr).
        """
        self.clt.eval()
        latents_list_L = []

        # Pre-calculate which latents influence each layer
        node_masks = None
        if active_nodes is not None:
            node_masks = []
            for l in range(self.num_layers):
                m = torch.zeros(self.clt.d_hidden, device=self.device)
                if l in active_nodes and len(active_nodes[l]) > 0:
                    m[list(active_nodes[l])] = 1.0
                node_masks.append(m.view(1, 1, -1))

        # CLT starts with Layer 0 (embeddings)
        x_curr_BTH = x_stack[:, 0, :, :]  # (B, T, H)
        # 1. Transpose for CLT: (B, T, H) -> (T, B, H)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        last_recon_TBH = None
        
        for l in range(self.num_layers):
            layer = self.esm.layers[l]

            # 2. Get ESM attention
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
                residual = x_curr_TBH
                x_ln = layer.self_attn_layer_norm(x_curr_TBH)
                x_attn_out, _ = layer.self_attn(
                    query=x_ln, key=x_ln, value=x_ln,
                    key_padding_mask=padding_mask, need_weights=False
                )
                x_TBH = residual + x_attn_out

            # 3. Encode CLT replacement for MLP (residual)
            residual = x_TBH
            x_mlp_in_TBH = layer.final_layer_norm(x_TBH)
            x_norm_TBH, mu, std = self.clt.LN(x_mlp_in_TBH)
            x_norm_TBH = x_norm_TBH - self.clt.b_pre[l]
            enc_TBD = self.clt.encoders[l](x_norm_TBH) + self.clt.b_enc[l]
            latents_TBD = self.clt.topK_activation(enc_TBD, k=self.clt.k)
            if retain_grad:
                latents_TBD.retain_grad()

            # 4. Apply Sparse Mask (Ablation)
            if node_masks is not None:
                latents_TBD = latents_TBD * node_masks[l]

            latents_list_L.append(latents_TBD)

            # 5. Decode and denormalize (reconstruct the MLP output at layer 'l' using latents from 0...l)
            recon_TBH = torch.zeros_like(x_norm_TBH)
            for src in range(l + 1):
                key = f"{src}_{l}"
                if key in self.clt.decoders:
                    recon_TBH = recon_TBH + (latents_list_L[src] @ self.clt.decoders[key])
            recon_TBH = recon_TBH + self.clt.b_pre[l]
            recon_TBH = recon_TBH * std + mu
            last_recon_TBH = recon_TBH

            # 6. Update stream to next layer 
            x_curr_TBH = residual + recon_TBH

        # Return (B, T, H)
        x_curr_BTH = x_curr_TBH.transpose(0, 1)
        recon_mlp_BTH = last_recon_TBH.transpose(0, 1)

        return x_curr_BTH, latents_list_L, recon_mlp_BTH

    def _run_clt_direct(self, x_clt_input_flat, active_nodes=None, retain_grad=False):
        """
        CLT Forward pass to get to the last layer with ground-truth MLP inputs at each layer.
        x_clt_input_flat: (B*T, L, H) = (S, L, H)
        """
        S, L, H = x_clt_input_flat.shape
        latents_list_L = []
        mu_list_L, std_list_L = [], []

        # Pre-calculate which latents influence each layer
        node_masks = None
        if active_nodes is not None:
            node_masks = []
            for l in range(self.num_layers):
                m = torch.zeros(self.clt.d_hidden, device=self.device)
                if l in active_nodes and len(active_nodes[l]) > 0:
                    m[list(active_nodes[l])] = 1.0
                node_masks.append(m.view(1, 1, -1))

        for l in range(L):
            x_in_SH = x_clt_input_flat[:, l, :]
            x_norm_SH, mu, std = self.clt.LN(x_in_SH)
            mu_list_L.append(mu)
            std_list_L.append(std)
            x_norm_SH = x_norm_SH - self.clt.b_pre[l]
            enc_SD = self.clt.encoders[l](x_norm_SH) + self.clt.b_enc[l]
            latents_SD = self.clt.topK_activation(enc_SD, k=self.clt.k)
            if retain_grad: latents_SD.retain_grad()

            if node_masks is not None:
                latents_SD = latents_SD * node_masks[l]
                
            latents_list_L.append(latents_SD)

        target_layer = self.num_layers - 1
        recon_accum_SH = torch.zeros_like(x_clt_input_flat[:, 0, :])
        for src in range(target_layer + 1):
            key = f"{src}_{target_layer}"
            if key in self.clt.decoders:
                recon_accum_SH = recon_accum_SH + (latents_list_L[src] @ self.clt.decoders[key])
            
        recon_accum_SH = recon_accum_SH + self.clt.b_pre[target_layer]
        recon_accum_SH = recon_accum_SH * std_list_L[target_layer] + mu_list_L[target_layer]
        return recon_accum_SH, latents_list_L

    def _get_reconstruction_with_layer_embedding(self, tokens, active_nodes=None, retain_grad=False, sequential=False, freeze_attention=True, source="mlp_output"):
        """
        Runs CLT to get reconstruction of layer embedding.
        Handles data collection and dispatch to Sequential or Direct runners.
        
        Args:
            source: 
                "mlp_output": Returns ONLY the reconstructed MLP component. (Default)
                "layer_output": Returns the full stream with final layer norm.
        
        Returns:
            modified_emb_BTH: (B, T, H) - Modified embeddings after CLT reconstruction
            latents_list_L: List[Tensor] - List of latents for each layer
        """
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

        B, T = tokens.shape
        H = x_stack_SLH.shape[-1]
        padding_mask = (tokens == self.pl_module.alphabet.padding_idx) # (B, T)

        # 2. Run CLT
        if sequential:
            # Prepare stack for Sequential Runner
            # x_stack_SLH: (B*T, L+1, H) -> (B, T, L+1, H) -> (B, L+1, T, H)
            depth = x_stack_SLH.shape[1]
            x_stack_reshaped_BLTH = x_stack_SLH.view(B, T, depth, H).permute(0, 2, 1, 3)
            modified_stream_BTH, latents_list_L, recon_mlp_BTH = self._run_clt_sequential(
                x_stack_reshaped_BLTH,
                active_nodes=active_nodes,
                retain_grad=retain_grad,
                freeze_attention=freeze_attention,
                padding_mask=padding_mask
            )
        else:
            # Use Direct Runner 
            recon_mlp_flat_SH, latents_list_L = self._run_clt_direct(
                x_clt_in_SLH, 
                active_nodes=active_nodes, 
                retain_grad=retain_grad
                )
            B, T = tokens.shape
            H = x_stack_SLH.shape[-1]
            target_layer = self.num_layers - 1
            
            orig_layer_out_BTH = x_stack_SLH[:, -1, :].view(B, T, H)
            orig_mlp_BTH = x_mlp_out_SLH[:, target_layer, :].view(B, T, H)
            recon_mlp_BTH = recon_mlp_flat_SH.view(B, T, H) # (B*T, H) -> (B, T, H)

            if source == "layer_output":
                target_layer = self.num_layers - 1
                orig_layer_out_BTH = x_stack_SLH[:, -1, :].view(B, T, H)
                orig_mlp_BTH = x_mlp_out_SLH[:, target_layer, :].view(B, T, H)
                modified_stream_BTH = orig_layer_out_BTH - orig_mlp_BTH + recon_mlp_BTH
            else:
                modified_stream_BTH = None
        self.collector.register_hooks()
                    
        # 3. Determine source
        if source == "mlp_output":
            return recon_mlp_BTH, latents_list_L
        elif source == "layer_output":
            return self.esm.emb_layer_norm_after(modified_stream_BTH), latents_list_L
        else:
            raise ValueError(f"Unknown source: {source}")
        
    def reconstruct_layer_embeddings(self, batch_seqs, active_nodes=None, mean_pool=True, sequential=False, freeze_attention=True, source="mlp_output"):
        """
        Runs intervention and returns modified embeddings.
        
        Args:
            batch_seqs: List[str]
            active_nodes: Dict or None
            mean_pool: 
                If True: returns (B, H)
                If False: returns (B, T, H) with CLS/EOS/PAD masked to 0.
            sequential:
                If True: uses Sequential Runner
                If False: uses Direct Runner
        """
        with torch.no_grad():
            tokens = self.pl_module.tokenize(batch_seqs)
            modified_emb_BTH, _ = self._get_reconstruction_with_layer_embedding(
                tokens, active_nodes=active_nodes, sequential=sequential, freeze_attention=freeze_attention, source=source
            )
            
            mask_BT = (tokens != self.pl_module.alphabet.cls_idx) & \
                   (tokens != self.pl_module.alphabet.eos_idx) & \
                   (tokens != self.pl_module.alphabet.padding_idx)
            mask_BT1 = mask_BT.float().unsqueeze(-1)
            
            if mean_pool:
                sum_emb = (modified_emb_BTH * mask_BT1).sum(dim=1)
                sum_mask = mask_BT1.sum(dim=1).clamp(min=1e-9)
                return sum_emb / sum_mask
            else:
                masked_emb = modified_emb_BTH * mask_BT1
                return masked_emb[:, 1:-1, :]

    def get_gradients(self, batch_seqs, torch_probe, cnn=False, sequential=False, freeze_attention=True, source="mlp_output"):
        """
        Compute attribution (Gradient * Activation) for both Linear (mean_pool) and CNN (sequence) probes.
        """
        self.clt.zero_grad()
        torch_probe.zero_grad()
        
        # 1. Forward Pass
        tokens = self.pl_module.tokenize(batch_seqs)
        modified_emb_BTH, latents_list_L = self._get_reconstruction_with_layer_embedding(
            tokens, active_nodes=None, retain_grad=True, sequential=sequential, freeze_attention=freeze_attention, source=source
        )
        self.collector.remove_hooks()
            
        # 2. Probe Forward/Backward
        mask_BT = (tokens != self.pl_module.alphabet.cls_idx) & \
               (tokens != self.pl_module.alphabet.eos_idx) & \
               (tokens != self.pl_module.alphabet.padding_idx)
        mask_BT1 = mask_BT.float().unsqueeze(-1)
        if cnn:
            masked_emb = modified_emb_BTH * mask_BT1
            probe_input = masked_emb[:, 1:-1, :]
        else:
            probe_input = (modified_emb_BTH * mask_BT1).sum(dim=1) / mask_BT1.sum(dim=1).clamp(min=1e-9)
        output = torch_probe(probe_input)
        loss = output.sum()
        loss.backward()

        # 3. Collect Attribution
        results = {}
        for l, latents in enumerate(latents_list_L):
            if latents.grad is not None:
                # Sequential: (T, B, D) -> 3D
                # Direct:     (S, D)  -> 2D
                attr = torch.abs(latents * latents.grad)      
                if attr.ndim == 3:
                    # Sum over Time (0) and Batch (1) -> (D,)
                    score = attr.sum(dim=(0, 1))
                elif attr.ndim == 2:
                    # Sum over Spatial (0) -> (D,)
                    score = attr.sum(dim=0)
                else:
                    # Fallback for unexpected shapes
                    score = attr.sum()
                    
                results[l] = score.detach().cpu().numpy()
        del latents_list_L
        del modified_emb_BTH
        self.collector.register_hooks()
                
        return results
    
    def run_ablation(self, batch_seqs, torch_probe, active_nodes=None, cnn=False, sequential=False, freeze_attention=True, source="mlp_output"):
        """
        Runs inference with specific circuit nodes active.
        """
        with torch.no_grad():
            tokens = self.pl_module.tokenize(batch_seqs)
            self.collector.remove_hooks()
            modified_emb_BTH, _ = self._get_reconstruction_with_layer_embedding(
                tokens, active_nodes=active_nodes, sequential=sequential, freeze_attention=freeze_attention, source=source
            )
            mask_BT = (tokens != self.pl_module.alphabet.cls_idx) & \
                   (tokens != self.pl_module.alphabet.eos_idx) & \
                   (tokens != self.pl_module.alphabet.padding_idx)
            mask_BT1 = mask_BT.float().unsqueeze(-1)
            
            if cnn:
                masked_emb = modified_emb_BTH * mask_BT1
                probe_input = masked_emb[:, 1:-1, :]
            else:
                probe_input = (modified_emb_BTH * mask_BT1).sum(dim=1) / mask_BT1.sum(dim=1).clamp(min=1e-9)
            self.collector.register_hooks()
            
            return torch_probe(probe_input)