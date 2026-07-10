import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# from steering.full_replacement_models import FullReplacementModel, FullCLTReplacementModel, FullPLTReplacementModel
from full_replacement_models import FullReplacementModel, FullCLTReplacementModel, FullPLTReplacementModel


# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# A: Number of attention heads in the model
# d: Attention head dimension
# a: A * B
# ──────────────────────────────────────────────────────────────────────────────

class LocalReplacementModel(FullReplacementModel):
    '''
    Implements a local replacement model, based off of:

    https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-local-replacement
    '''
    def __init__(self, pl_module, device, base_prompt: str):
        super().__init__(pl_module, device)
        self.base_prompt = base_prompt
        self.base_variances = {}
        self.base_attn_patterns = {}
        self.base_error_vectors = {}
        self.residual_cache = {}
        self.latents_cache = {}
        self.base_embedding = None # (T, H)
        self.compute_base_values()

    def compute_base_values(self):
        '''
        Compute variance vectors, attention outputs, and error vectors for the base prompt.
        Dynamically updates self.base_variances, self.base_attn_patterns, self.base_error_vectors,
        self.residual_cache, and self.latents_cache.
        '''
        self.esm.eval()

        self.base_variances = {}
        self.base_attn_patterns = {}
        self.base_error_vectors = {}
        self.residual_cache = {}
        self.latents_cache = {}
        latents_list = []

        batch_tokens = self.tokenize([self.base_prompt])  # (1, T)
        batch_tokens = batch_tokens.to(self.device)
        padding_mask = batch_tokens.eq(self.alphabet.padding_idx)  # (1, T)

        # Start with embeddings
        x_1TH = self.esm.embed_scale * self.esm.embed_tokens(batch_tokens)
        x_T1H = x_1TH.transpose(0, 1).requires_grad_(True)  # (T, 1, H)

        base_emb_T1H = x_T1H.detach().clone()
        self.base_embedding = base_emb_T1H.squeeze(1) # (T, H)

        for l in range(self.num_layers):
            layer = self.esm.layers[l]
            resid = x_T1H

            # 1. Compute variance at input to attention
            var_1T = x_1TH.detach().var(dim=-1, unbiased=False)  # (1, T)
            self.base_variances[('attn', l)] = var_1T.squeeze(0)  # (T,)

            # 2. Save the attention pattern for this layer, no grad here!
            with torch.no_grad():
                x_ln_T1H = layer.self_attn_layer_norm(x_T1H)
                _, attn_pattern_A1TT = layer.self_attn(
                    query=x_ln_T1H, key=x_ln_T1H, value=x_ln_T1H,
                    key_padding_mask=padding_mask, need_head_weights=True
                )

                A = layer.self_attn.num_heads
                T, B, H = x_ln_T1H.shape[0], x_ln_T1H.shape[1], x_ln_T1H.shape[2]

                # Store attention pattern by squeezing the batch dimension
                self.base_attn_patterns[l] = attn_pattern_A1TT.squeeze(1)  # (A, T, T)

            # 3. Now run attention again, this time using frozen LN and pattern:
            x_ln_T1H = self.layernorm_fixed(resid, l, ln_type="attn")

            _, values_aTd = layer.self_attn(
                query=x_ln_T1H, key=x_ln_T1H, value=x_ln_T1H,
                key_padding_mask=padding_mask, before_softmax=True
            )
            # Use the stored attention pattern: (A, T, T) -> (B*A, T, T)
            attn_ATT = self.base_attn_patterns[l]  # (A, T, T)
            attn_ATT = attn_ATT.unsqueeze(0).expand(B, -1, -1, -1)  # (B, A, T, T)
            attn_aATT = attn_ATT.reshape(B * A, T, T)  # (B*A, T, T)
            attn_out = torch.bmm(attn_aATT, values_aTd)  # (B*A, T, d)
            attn_out = attn_out.transpose(0, 1).contiguous().view(T, B, H)  # (T, B, H)
            attn_out = layer.self_attn.out_proj(attn_out)

            x_T1H = resid + attn_out
            # Store residual stream after attention
            self.residual_cache[l] = x_T1H
            x_T1H.retain_grad()

            # 4. Compute variance at input to MLP
            x_1TH = x_T1H.transpose(0, 1)  # (1, T, H)
            var_1T = x_1TH.detach().var(dim=-1, unbiased=False)  # (1, T)
            self.base_variances[('mlp', l)] = var_1T.squeeze(0)  # (T,)

            resid = x_T1H
            x_ln_T1H = self.layernorm_fixed(resid, l, ln_type="mlp")
            # Cache variance at input to CLT layernorm (same as MLP LN output)
            var_clt_1T = x_ln_T1H.detach().var(dim=-1, unbiased=False)  # (T, 1)
            self.base_variances[('clt', l)] = var_clt_1T.squeeze(0)  # (T,)

            with torch.no_grad():
                mlp_out = layer.fc2(F.gelu(layer.fc1(x_ln_T1H)))

            x_T1H = resid + mlp_out
            # Store pre-mlp residual
            with torch.no_grad():
                latents_T1D, _, mu, std = self._encode_latents(l, x_ln_T1H)
                latents_list.append(latents_T1D)
                self.latents_cache[l] = latents_T1D
                clt_recon_T1H = self._decode_latents(l, latents_list)
                clt_recon_T1H = clt_recon_T1H + self.model.b_pre[l]
                clt_recon_T1H = clt_recon_T1H * std + mu

                # Compute error: MLP - CLT
                self.base_error_vectors[l] = (mlp_out - clt_recon_T1H).squeeze(1).detach()  # (T, H)

            x_1TH = x_T1H.transpose(0, 1)  # Keep x_1TH in sync for next iteration's attn variance

    def update_base_prompt(self, new_base_prompt):
        """
        Re-compute base values for a new sequence without full re-initialization.
        
        This allows using a different sequence as the base prompt for computing
        frozen attention, fixed variances, and error correction vectors.
        
        Args:
            new_base_prompt: New sequence string to use as base prompt
        """
        self.base_prompt = new_base_prompt
        self.compute_base_values()

    def layer_forward(self, l, x_prev_TBH, latents_list_L, ablate_nodes=None, circuit=None, alphas=None, before=False, padding_mask=None, **kwargs):
        """
        Forward pass for a single layer l.

        Args:
            l: Layer index (0 to num_layers-1)
            x_prev_TBH: Output from previous layer (T, B, H) - this is x_curr_TBH from end of previous layer
            latents_list_L: List of latents from layers 0 to l-1
            ablate_nodes: Node indices to zero-ablate for this specific layer (set/list of indices) or None
            circuit: Dict mapping layer indices to lists of node indices to steer, or None
            alphas: List of scalar magnitudes (hyperparameters)
            before: Whether to steer latents before or after applying TopK
            padding_mask: Boolean mask (B, T) where True = padding token to ignore in attention
            cache: If True, caches the residual (x^l) for each layer
        Returns:
            x_curr_TBH: Output for this layer (residual + recon_TBH) - (T, B, H)
            latents_TBD: Latents for this layer (T, B, D)
            recon_TBH: MLP reconstruction for this layer (T, B, H)
            gt_mlp_TBH: Ground truth MLP output for final layer (residual + gt_mlp), None for other layers

        Notes:
        - If a node is both highlighted in the ablation and steering circuit, it is ablated.
        """
        layer = self.esm.layers[l]
        
        # 1. Use frozen attention from base prompt
        residual = x_prev_TBH
        x_ln_TBH = self.layernorm_fixed(x_prev_TBH, l, ln_type='attn')
        _, values_aTd = layer.self_attn(
            query=x_ln_TBH, key=x_ln_TBH, value=x_ln_TBH,
            key_padding_mask=padding_mask, before_softmax=True
        )
        T, B, H = x_ln_TBH.shape[0], x_ln_TBH.shape[1], x_ln_TBH.shape[2]
        A = layer.self_attn.num_heads
        d = H // A
        assert values_aTd.shape == (B * A, T, d), f"values_aTd shape: {values_aTd.shape} != (B * A, T, d): {B * A, T, d}"

        # frozen patterns: (A, T, T) -> (B*A, T, T)
        attn_ATT = self.base_attn_patterns[l]              # (A, T, T)
        attn_ATT = attn_ATT.unsqueeze(0).expand(B, -1, -1, -1)    # (B, A, T, T)
        attn_aATT = attn_ATT.reshape(B * A, T, T)                  # (B*A, T, T)

        # identical to ESM: attn = bmm(attn_probs, v)
        attn_aTd = torch.bmm(attn_aATT, values_aTd)                      # (B*A, T, d)

        # merge heads -> (T, B, H)
        attn_TBH = attn_aTd.transpose(0, 1).contiguous().view(T, B, H)
        x_attn_out = layer.self_attn.out_proj(attn_TBH)

        x_TBH = residual + x_attn_out

        # 2. Encode replacement for MLP (residual)
        residual = x_TBH
        x_mlp_in_TBH = self.layernorm_fixed(x_TBH, l, ln_type='mlp')

        # 2a. Compute ground truth MLP output for final layer
        with torch.no_grad():
            gt_mlp_TBH = layer.fc2(F.gelu(layer.fc1(x_mlp_in_TBH)))


        latents_TBD, enc_TBD, mu, std = self._encode_latents(l, x_mlp_in_TBH)
        B = x_mlp_in_TBH.shape[1]
       
        if circuit is not None and l in circuit:
            assert alphas.shape[0] == B
            node_indices = list(circuit[l])
            if len(node_indices) > 0:
                # Apply multiplier across the whole latent
                target_tensor = enc_TBD if before else latents_TBD
                with torch.no_grad():
                    current_max = target_tensor.amax(dim=(0, 2))
                    injection_values = (current_max * alphas).view(1, B, 1)
                target_tensor = target_tensor.clone()
                target_tensor[:, :, node_indices] = injection_values
            if before:
                # apply TopK and continue
                latents_TBD = self.model.topK_activation(target_tensor, k=self.model.k)

        # 3. Apply Zero Ablation
        if ablate_nodes is not None:
            # ablate_nodes contains indices to zero out
            node_indices = list(ablate_nodes)
            if len(node_indices) > 0:
                mask = torch.ones(latents_TBD.shape[-1], device=self.device)  # (1, 1, D)
                mask[node_indices] = 0.0
                latents_TBD = latents_TBD * mask.view(1, 1, -1)

        # 4. Decode and denormalize (reconstruct the MLP output)
        # Append current latents to the list for decoding
        current_latents_list = latents_list_L + [latents_TBD]
        recon_TBH = self._decode_latents(l, current_latents_list)
        recon_TBH = recon_TBH + self.model.b_pre[l]
        recon_TBH = recon_TBH * std + mu

        # 5. Update stream to next layer
        x_curr_TBH = residual + recon_TBH
        gt_mlp_TBH = residual + gt_mlp_TBH
        # add the error vector to the residual stream
        # stack the base_error_vector on itself B times and reshape to (T, B, H)
        x_curr_TBH += self.base_error_vectors[l].unsqueeze(1).expand(-1, B, -1)

        return x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH

    def forward(self, batch_seqs, ablate_nodes=None, stop=None, **kwargs):
        return super().forward(batch_seqs, ablate_nodes=ablate_nodes, freeze_attention=False, stop=stop)

    def forward_steered(self, seq, circuit, before=False, ablate_nodes=None, alphas=None, **kwargs):
        return super().forward_steered(seq, circuit=circuit, before=before, ablate_nodes=ablate_nodes, alphas=alphas, freeze_attention=False, add_correction=False)
    
    def layernorm_fixed(self, x_TBH, layer_idx, token_idx = None, ln_type='mlp'):
        '''
        Apply layernorm with the fixed variance from the base prompt.
        Uses the pre-computed variance vectors stored in self.base_variances.

        Args:
            x_TBH: input tensor (T, B, H)
            layer_idx: layer index (int)
            ln_type: 'attn' for self_attn_layer_norm or 'mlp' for final_layer_norm
            or 'clt' for CLT encoder LN
            token_idx: token index (int) 
            if token_idx is given, it is assumed that T = 1, B = 1

        Returns:
            ln_result_TBH: layernorm output (T, B, H)
        '''
        if ln_type == 'clt' and getattr(self.model, 'skip_ln', False):
            return x_TBH

        # Select the appropriate layernorm
        if ln_type == 'attn':
            ln = self.esm.layers[layer_idx].self_attn_layer_norm
        elif ln_type == 'mlp':
            ln = self.esm.layers[layer_idx].final_layer_norm
        elif ln_type == 'clt':
            ln = None

        if ln is not None:
            eps = ln.eps
            beta = ln.bias  # (H,)
            gamma = ln.weight  # (H,)
        else:
            eps = 1e-5
            beta = 0
            gamma = 1

        var_T = self.base_variances[(ln_type, layer_idx)]
        if token_idx is not None:
            mu = x_TBH.mean() 
            fixed_std = torch.sqrt(var_T[token_idx] + eps)
            y = gamma * ((x_TBH - mu) / fixed_std) + beta
        else:
            mu = x_TBH.mean(dim=-1, keepdim=True)  # (T, B, 1)
            fixed_std = torch.sqrt(var_T.view(-1, 1, 1) + eps)  # (T, 1, 1)
            y = gamma * ((x_TBH - mu) / fixed_std) + beta
        return y
    
    def compute_edge_weight(self, source: tuple, target: tuple, rerun_base=False):
        """
        Implements edge weights as seen in 
        https://transformer-circuits.pub/2025/attribution-graphs/methods.html#appendix-attribution-graph-computation
        source=(Ls, ts, fs), target=(Lt, tt, ft)
        rerun_base: if True, re-runs the base values computation
        """
        if rerun_base:
            self.compute_base_values() # shouldn't really be necessary, but just in case
        src_layer, src_token, src_feature = source
        tgt_layer, tgt_token, tgt_feature = target

        # adjust for CLS token
        src_token += 1
        tgt_token += 1

        assert src_layer < tgt_layer, "Source layer must be less than target layer"

        try:
            a_s = self.latents_cache[src_layer][src_token, 0, src_feature].item()
        except KeyError:
            self.compute_base_values()
            a_s = self.latents_cache[src_layer][src_token, 0, src_feature].item()

        target_weight_H = self.model.encoders[tgt_layer].weight[tgt_feature].detach()  # (H,)

        source_weights_H = {
            ell: self.model.decoders[f"{src_layer}_{ell}"].detach()[src_feature]  # (H,)
            for ell in range(src_layer, tgt_layer)
        }
        assert tgt_layer in self.residual_cache, f"Target layer {tgt_layer} not in residual cache"
        resid_cache = self.residual_cache

        # target scalar at target node (layer residual stream here)
        # this is essentially the pre-activation feature we are interested in.
        # now we need to backpropagate and find dz_t/da_s
        residual_pre_lns = resid_cache[tgt_layer][tgt_token, 0]
        # then compute the MLP LN
        residual_pre_lns = self.layernorm_fixed(residual_pre_lns, tgt_layer, token_idx=tgt_token, ln_type="mlp")
        # then compute the CLT ln
        residual_pre_lns = self.layernorm_fixed(residual_pre_lns, tgt_layer, token_idx=tgt_token, ln_type="clt")
        pre_act = torch.dot(residual_pre_lns, target_weight_H)

        self.esm.zero_grad(set_to_none=True)
        
        # Enable anomaly detection for debugging gradient computation
        # torch.autograd.set_detect_anomaly(True)
        # clear the gradient of the residual cache:
        for v in resid_cache.values():
            v.grad = None
        pre_act.backward(retain_graph=True)

        acc = 0.0
        for ell in range(src_layer, tgt_layer):
            grad_H = resid_cache[ell].grad[src_token, 0]         # (H,)
            acc += torch.dot(source_weights_H[ell], grad_H).item()

        return a_s * acc

    def compute_target_batched(self, source_batch, target: tuple, rerun_base=False):
        """
        Gives the edge weight to a single target feature from a batch of source features
        source_batch: list of tuples (src_layer, src_token, src_feature)
        target=(Lt, tt, ft)
        rerun_base: if True, re-runs the base values computation
        """

        # adjust for CLS token
        source_batch = [(src_layer, src_token + 1, src_feature) for src_layer, src_token, src_feature in source_batch]
        target = (target[0], target[1] + 1, target[2])

        if rerun_base:
            self.compute_base_values() # shouldn't really be necessary, but just in case

        tgt_layer, tgt_token, tgt_feature = target
        
        # let B be len(source_batch), then we'll have B a_s values
        a_s = []
        for src_layer, src_token, src_feature in source_batch:
            try:
                a_s.append(self.latents_cache[src_layer][src_token, 0, src_feature].item())
            except KeyError:
                self.compute_base_values()
                a_s.append(self.latents_cache[src_layer][src_token, 0, src_feature].item())

        target_weight_H = self.model.encoders[tgt_layer].weight[tgt_feature].detach()  # (H,)

        weights_dict_B = []
        for src_layer, src_token, src_feature in source_batch:
            source_weights_H = {
                ell: self.model.decoders[f"{src_layer}_{ell}"].detach()[src_feature]  # (H,)
                for ell in range(src_layer, tgt_layer)
            }
            weights_dict_B.append(source_weights_H)

        assert tgt_layer in self.residual_cache, f"Target layer {tgt_layer} not in residual cache"
        resid_cache = self.residual_cache

        # target scalar at target node (layer residual stream here)
        # this is essentially the pre-activation feature we are interested in.
        # now we need to backpropagate and find dz_t/da_s
        residual_pre_lns = resid_cache[tgt_layer][tgt_token, 0]
        # then compute the MLP LN
        residual_pre_lns = self.layernorm_fixed(residual_pre_lns, tgt_layer, token_idx=tgt_token, ln_type="mlp")
        # then compute the CLT ln
        residual_pre_lns = self.layernorm_fixed(residual_pre_lns, tgt_layer, token_idx=tgt_token, ln_type="clt")
        pre_act = torch.dot(residual_pre_lns, target_weight_H)

        self.esm.zero_grad(set_to_none=True)
        
        # Enable anomaly detection for debugging gradient computation
        # torch.autograd.set_detect_anomaly(True)
        # clear the gradient of the residual cache:
        for v in resid_cache.values():
            v.grad = None
        pre_act.backward(retain_graph=True)

        edge_weights_to_tgt_B = []

        for i, source_weights_H in enumerate(weights_dict_B):
            acc = 0.0
            src_layer, src_token, src_feature = source_batch[i]
            for ell in range(src_layer, tgt_layer):
                grad_H = resid_cache[ell].grad[src_token, 0]         # (H,)
                acc += torch.dot(source_weights_H[ell], grad_H).item()
            edge_weights_to_tgt_B.append(a_s[i] * acc)

        return edge_weights_to_tgt_B

# mixin classes
class LocalCLTReplacementModel(FullCLTReplacementModel, LocalReplacementModel):
    def __init__(self, pl_module, device, base_prompt: str):
        LocalReplacementModel.__init__(self, pl_module, device, base_prompt=base_prompt)
        self.clt = self.model 
