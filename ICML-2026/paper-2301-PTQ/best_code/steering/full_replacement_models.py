import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import ESM2ActivationCollector for direct replacement model
# Use absolute path based on script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
_circuit_utils_dir = os.path.join(_parent_dir, 'circuit_utils')

try:
    if _circuit_utils_dir not in sys.path:
        sys.path.append(_circuit_utils_dir)
    from esm_activation import ESM2ActivationCollector
except ImportError:
    try:
        if _parent_dir not in sys.path:
            sys.path.append(_parent_dir)
        from circuit_utils.esm_activation import ESM2ActivationCollector
    except ImportError:
        ESM2ActivationCollector = None

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: CLT/PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# V: LM vocabulary size
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────

class FullReplacementModel(nn.Module):
    '''
    Base class for replacement models that use encoder/decoders to replace MLP, as
    described in:
    https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-replacement

    '''
    def __init__(self, pl_module, device):
        super().__init__()
        self.pl_module = pl_module
        self.device = device

        # Extract components from pl_module
        self.model = pl_module.clt if hasattr(pl_module, 'clt') else pl_module.plt
        self.esm = pl_module.esm_model
        self.alphabet = pl_module.alphabet
        self.num_layers = self.model.num_layers

        # Setup activation collector for freeze_attention support
        if ESM2ActivationCollector is not None:
            self.collector = ESM2ActivationCollector(self.esm, self.alphabet)
            self.collector.register_hooks()

    def tokenize(self, seqs):
        """
        Tokenize a list of sequence strings using pl_module's tokenize method.

        Args:
            seqs: List of sequence strings

        Returns:
            batch_tokens: (B, T) token tensor
        """
        return self.pl_module.tokenize(seqs)

    def _encode_latents(self, l, x_mlp_in_TBH):
        """
        Encode MLP input to latents.

        Args:
            l: Current layer index
            x_mlp_in_TBH: MLP input after layer norm (T, B, H)

        Returns:
            latents_TBD: Encoded latents after topK (T, B, D)
            enc_TBD: Raw encoder activations before topK (T, B, D)
            mu: Mean from LayerNorm
            std: Std from LayerNorm
        """
        x_norm_TBH, mu, std = self.model.LN(x_mlp_in_TBH)
        x_norm_TBH = x_norm_TBH - self.model.b_pre[l]
        enc_TBD = self.model.encoders[l](x_norm_TBH) + self.model.b_enc[l]
        latents_TBD = self.model.topK_activation(enc_TBD, k=self.model.k)
        return latents_TBD, enc_TBD, mu, std

    def _decode_latents(self, l, current_latents_list):
        """
        Decode latents to reconstruct MLP output. Override this in subclasses.

        Args:
            l: Current layer index
            current_latents_list: List of latents from layers 0 to l (inclusive)

        Returns:
            recon_TBH: Reconstructed MLP output (T, B, H)
        """
        raise NotImplementedError("Subclasses must implement _decode_latents")
    
    def forward_steered(self, seq, circuit, before=False, ablate_nodes=None, alphas=None, freeze_attention=False, add_correction=True):
        """
        Steered Forward Pass using a circuit.

        Args:
            seq: Input sequence string
            alphas: List of scalar magnitudes (hyperparameters)
            circuit: Dict mapping layer indices to lists of node indices to steer
            before: Whether to steer latents before or after applying TopK
            ablate_nodes: Dict mapping layer indices to lists of node indices to zero-ablate, or None
            freeze_attention: If True, uses ground truth attention to update stream

        Returns:
            x_curr_BTH: Final layer replacement model output (Input + Attn + MLP_Recon)
            latents_list_L: List of latents
            recon_mlp_BTH: The MLP reconstruction of the final layer (just the MLP part)
            gt_mlp_BTH: Ground truth MLP output of final layer for sanity checking
            mask_BT: Boolean mask (B, T) where True = valid token, False = special token
        """
        self.model.eval()
        if isinstance(alphas, (float, int)):
            alphas = [alphas]
        if not isinstance(alphas, torch.Tensor):
            alphas = torch.tensor(alphas, device=self.device, dtype=torch.float32)
        else:
            alphas = alphas.to(self.device, dtype=torch.float32)
        if alphas.ndim == 0:       # 0-dim scalar tensor -> (1,)
            alphas = alphas.unsqueeze(0)
        elif alphas.ndim > 1:      # (1, B) or (B, 1) -> (B,)
            alphas = alphas.reshape(-1)
        B = alphas.shape[0]

        latents_list_L = []

        # check if the circuit doesn't have int keys, if so convert them
        if not all(isinstance(k, int) for k in circuit.keys()):
            circuit = {int(k): v for k, v in circuit.items()}
        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        # Get the CLT's reconstruction error
        if freeze_attention and add_correction:
            clt_out_1TH, _, _, gt_mlp_1TH, _ = self.forward([seq], ablate_nodes=ablate_nodes, freeze_attention = True)
        elif add_correction:
            _, _, _, gt_mlp_1TH, _ = self.forward([seq], ablate_nodes=ablate_nodes, freeze_attention = True)
            clt_out_1TH, _, _, _, _ = self.forward([seq], ablate_nodes=ablate_nodes, freeze_attention = freeze_attention)
        
        if add_correction:
            # gt_mlp_1TH: ACTUAL last layer output of model
            # clt_out_1TH: CLT approximation of last layer output
            recon_error_1TH = gt_mlp_1TH - clt_out_1TH
            # stack recon_error_1TH to be (B, T, H), eg just repeat it B times
            recon_error_BTH = recon_error_1TH.expand(B, -1, -1)
            # why do this? we know that clt_out_BTH is going to be an inaccurate approximation 
            # of the true steered last-layer output because the CLT isn't 100% accurate, so adding
            # back the recon error should give us a better output. this is common in SAE steering

        # Tokenize sequences and get embeddings
        tokens_BT = self.tokenize([seq] * B)
        with torch.no_grad():
            # convert from tokens to embds
            x_curr_BTH = self.esm.embed_scale * self.esm.embed_tokens(tokens_BT)

        # Create mask to exclude special tokens (CLS, EOS, padding)
        mask_BT = (tokens_BT != self.alphabet.cls_idx) & \
                  (tokens_BT != self.alphabet.eos_idx) & \
                  (tokens_BT != self.alphabet.padding_idx)

        # 1. Transpose: (B, T, H) -> (T, B, H)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        last_recon_TBH = None
        T = tokens_BT.shape[1]  
        H = self.esm.embed_dim

        if freeze_attention:
            # Collect ground truth embeddings for frozen attention
            if not hasattr(self, 'collector') or self.collector is None:
                raise RuntimeError("freeze_attention requires ESM2ActivationCollector but it's not available")
            x_stack_SLH, _, _, _, _ = self.collector.collect(tokens_BT)
        else:
            x_stack_SLH = None

        for l in range(self.num_layers):
            layer_ablate_nodes = None
            if ablate_nodes is not None and l in ablate_nodes:
                layer_ablate_nodes = ablate_nodes[l]

            # Get ground truth for this layer if freezing attention
            # x_stack_SLH is (B*T, L+1, H) with B-major ordering, so reshape to (B, T, H) then transpose
            x_gt_BTH = x_stack_SLH[:, l, :].view(B, T, H).transpose(0, 1) if x_stack_SLH is not None else None

            # Call the individual layer forward function
            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = self.layer_forward(
                l, x_curr_TBH, latents_list_L,
                circuit=circuit, alphas=alphas, before=before, ablate_nodes=layer_ablate_nodes, x_gt=x_gt_BTH,
            )

            latents_list_L.append(latents_TBD)
            last_recon_TBH = recon_TBH

        # Return (B, T, H)
        x_curr_BTH = x_curr_TBH.transpose(0, 1)
        recon_mlp_BTH = last_recon_TBH.transpose(0, 1)
        gt_mlp_BTH = gt_mlp_TBH.transpose(0, 1)

        # add the recon error to the final layer
        if add_correction:
            x_curr_BTH = x_curr_BTH + recon_error_BTH

        return x_curr_BTH, latents_list_L, recon_mlp_BTH, gt_mlp_BTH, mask_BT

    def layer_forward(self, l, x_prev_TBH, latents_list_L, x_gt=None, ablate_nodes=None, circuit=None, alphas=None, before=False, padding_mask=None):
        """
        Forward pass for a single layer l.

        Args:
            l: Layer index (0 to num_layers-1)
            x_prev_TBH: Output from previous layer (T, B, H) - this is x_curr_TBH from end of previous layer
            x_gt: Ground truth output from previous layer (T, B, H) (if not None, freezes attn)
            latents_list_L: List of latents from layers 0 to l-1
            ablate_nodes: Node indices to zero-ablate for this specific layer (set/list of indices) or None
            circuit: Dict mapping layer indices to lists of node indices to steer, or None
            alphas: Tensor (B,) — strength per batch element
            before: Whether to steer latents before or after applying TopK
            padding_mask: Boolean mask (B, T) where True = padding token to ignore in attention

        Returns:
            x_curr_TBH: Output for this layer (residual + recon_TBH) - (T, B, H)
            latents_TBD: Latents for this layer (T, B, D)
            recon_TBH: MLP reconstruction for this layer (T, B, H)
            gt_mlp_TBH: Ground truth MLP output for final layer (residual + gt_mlp), None for other layers

        Notes:
        - If a node is both highlighted in the ablation and steering circuit, it is ablated.
        - For a sanity check, if you set freeze_attention = True, that will use the ground-truth
        attention value to update the residual stream. If you then use the MLP output (gt_mlp_BTH)
        of the forward pass, this should be the same as the ESM final layer output.
        """
        layer = self.esm.layers[l]

        # 1. Get ESM attention
        # When x_gt is provided (freeze_attention), use ground truth for attention computation
        # but use our reconstruction (x_prev_TBH) for the residual to avoid double-counting
        residual = x_prev_TBH
        if x_gt is not None:
            x_ln = layer.self_attn_layer_norm(x_gt)
        else:
            x_ln = layer.self_attn_layer_norm(x_prev_TBH)

        x_attn_out, _ = layer.self_attn(
            query=x_ln, key=x_ln, value=x_ln,
            key_padding_mask=padding_mask, need_weights=False
        )

        x_TBH = residual + x_attn_out

        # 2. Encode replacement for MLP (residual)
        residual = x_TBH
        x_mlp_in_TBH = layer.final_layer_norm(x_TBH)

        # 2a. Compute ground truth MLP output for final layer
        with torch.no_grad():
            gt_mlp_TBH = layer.fc2(F.gelu(layer.fc1(x_mlp_in_TBH)))

        latents_TBD, enc_TBD, mu, std = self._encode_latents(l, x_mlp_in_TBH)
        B = x_mlp_in_TBH.shape[1]
        # enc_TBD is before applying ReLU/TopK
        if circuit is not None and l in circuit:
            assert alphas.shape[0] == B
            node_indices = list(circuit[l])
            if len(node_indices) > 0:
                # Apply multiplier across the whole latent
                target_tensor = enc_TBD if before else latents_TBD
                current_max = target_tensor.amax(dim=(0, 2))
                injection_values = (current_max * alphas).view(1, B, 1)
                target_tensor[:, :, node_indices] = injection_values
            if before:
                # apply TopK and continue
                latents_TBD = self.model.topK_activation(target_tensor, k=self.model.k)

        # 3. Apply Zero Ablation
        if ablate_nodes is not None:
            # ablate_nodes contains indices to zero out
            node_indices = list(ablate_nodes)
            if len(node_indices) > 0:
                mask = torch.ones(latents_TBD.shape[-1], device=self.device) # (1, 1, D)
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

        return x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH

    def forward(self, batch_seqs, ablate_nodes=None, freeze_attention=False, stop=None):
        """
        Sequential Forward Pass.

        Args:
            batch_seqs: List of sequence strings
            ablate_nodes: Dict mapping layer indices to sets/lists of node indices to ablate, or None
            freeze_attention: If True, uses the ground truth attention to update stream
            stop: Layer index to stop at, if None, stops at the last layer
            cache: If True, caches the residual (x^l) for each layer
        Returns:
            x_curr_BTH: Final layer replacement model output (Input + Attn + MLP_Recon)
            latents_list_L: List of latents
            recon_mlp_BTH: The MLP reconstruction of the final layer (just the MLP part)
            gt_mlp_BTH: Ground truth MLP output of final layer for sanity checking
            mask_BT: Boolean mask (B, T) where True = valid token, False = special token
        """
        self.model.eval()
        latents_list_L = []
        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        # Tokenize sequences and get embeddings
        tokens_BT = self.tokenize(batch_seqs)
        # with torch.no_grad():
        #     # convert from tokens to embds
        #     x_curr_BTH = self.esm.embed_scale * self.esm.embed_tokens(tokens_BT)
        x_curr_BTH = self.esm.embed_scale * self.esm.embed_tokens(tokens_BT)

        # Create mask to exclude special tokens (CLS, EOS, padding) for output
        mask_BT = (tokens_BT != self.alphabet.cls_idx) & \
                  (tokens_BT != self.alphabet.eos_idx) & \
                  (tokens_BT != self.alphabet.padding_idx)

        # Create padding mask for attention (only padding tokens, not CLS/EOS)
        padding_mask = (tokens_BT == self.alphabet.padding_idx)

        # 1. Transpose: (B, T, H) -> (T, B, H)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)
        last_recon_TBH = None
        B, T = tokens_BT.shape
        H = self.esm.embed_dim

        if freeze_attention:
            # Collect ground truth embeddings for frozen attention
            if not hasattr(self, 'collector') or self.collector is None:
                raise RuntimeError("freeze_attention requires ESM2ActivationCollector but it's not available")
            x_stack_SLH, _, _, _, _ = self.collector.collect(tokens_BT)
        else:
            x_stack_SLH = None

        for l in range(self.num_layers):
            # Extract ablate nodes for this specific layer
            layer_ablate_nodes = None
            if ablate_nodes is not None and l in ablate_nodes:
                layer_ablate_nodes = ablate_nodes[l]

            # Get ground truth for this layer if freezing attention
            # x_stack_SLH is (B*T, L+1, H) with B-major ordering, so reshape to (B, T, H) then transpose
            x_gt = x_stack_SLH[:, l, :].view(B, T, H).transpose(0, 1) if x_stack_SLH is not None else None

            # Call the individual layer forward function
            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = self.layer_forward(
                l, x_curr_TBH, latents_list_L,
                ablate_nodes=layer_ablate_nodes, x_gt=x_gt, padding_mask=padding_mask
            )

            latents_list_L.append(latents_TBD)
            last_recon_TBH = recon_TBH
            
            if stop is not None and l == stop:
                break

        # Return (B, T, H)
        x_curr_BTH = x_curr_TBH.transpose(0, 1)
        recon_mlp_BTH = last_recon_TBH.transpose(0, 1)
        gt_mlp_BTH = gt_mlp_TBH.transpose(0, 1)

        return x_curr_BTH, latents_list_L, recon_mlp_BTH, gt_mlp_BTH, mask_BT

    def get_sequences(self, embeddings_BTH, mask_BT, select_indices=None, valid_muts=None, wt=None, max_mutations=None, cosine_similarity=True, cosine_threshold=0.98, min_position=None, max_position=None):
        """
        Convert embeddings to sequences, optionally argmaxing only over valid mutations at selected positions.

        Args:
            embeddings_BTH: Embeddings tensor (B, T, H)
            mask_BT: Boolean mask (B, T) where True = valid token
            select_indices: Indices to mutate (zero-indexed). If None, all positions use full argmax.
            valid_muts: Dict {index: [valid_amino_acids]} for each position. If provided with select_indices,
                argmax is restricted to these tokens only.
            wt: Wildtype sequence string. Required if select_indices is not None or max_mutations is not None.
            max_mutations: Maximum number of mutations from wildtype to allow (int). If there are more 
                mutations than this limit, only the top max_mutations positions (by logit confidence) 
                are kept; the rest revert to wildtype. If None, no limit.
            cosine_similarity: If True, reverts token to WT if cosine sim(steered_emb, wt_emb) >= cosine_threshold.
            cosine_threshold: Threshold for similarity (default 0.98). High similarity -> Revert to WT.

        Returns:
            sequences: List of sequence strings
            logits_BTV: Full logits tensor
        """
        if select_indices is not None and wt is None:
            raise ValueError("wt must be provided if select_indices is not None")
        if max_mutations is not None and wt is None:
            raise ValueError("wt must be provided if max_mutations is not None")
        if select_indices is not None and valid_muts is not None:
            if sorted(valid_muts.keys()) != sorted(select_indices):
                raise ValueError("valid_muts and select_indices must have the same indices")

        alphabet = self.alphabet
        wt_list = list(wt) if wt is not None else None
        self.esm.eval()
        with torch.no_grad():
            # 1. Apply Layer Norm to Input Embeddings (Steered)
            if hasattr(self.esm, "emb_layer_norm_after"):
                embeddings_BTH = self.esm.emb_layer_norm_after(embeddings_BTH)
            elif hasattr(self.esm, "layer_norm"):
                embeddings_BTH = self.esm.layer_norm(embeddings_BTH)

            # 2. Compute WT Embeddings for Similarity Check (if enabled)
            sim_mask_BT = None
            if cosine_similarity:
                # Tokenize WT (1, T)
                wt_tokens = self.tokenize([wt]).to(embeddings_BTH.device)
                
                # Get WT Representations
                res = self.esm(wt_tokens, repr_layers=[self.esm.num_layers], return_contacts=False)
                wt_emb = res["representations"][self.esm.num_layers] # (1, T, H)
                
                # Apply Same Layer Norm to WT
                if hasattr(self.esm, "emb_layer_norm_after"):
                    wt_emb = self.esm.emb_layer_norm_after(wt_emb)
                elif hasattr(self.esm, "layer_norm"):
                    wt_emb = self.esm.layer_norm(wt_emb)

                # Compute Cosine Similarity (B, T)
                # We want to know where similarity is HIGH ( >= 0.95) to force WT
                sim_BT = torch.nn.functional.cosine_similarity(embeddings_BTH, wt_emb, dim=-1)
                sim_mask_BT = (sim_BT >= cosine_threshold) # True = Too similar, revert to WT

            # 3. Compute Logits
            logits_BTV = self.esm.lm_head(embeddings_BTH)

            # if hasattr(self.esm, "emb_layer_norm_after"):
            #     embeddings_BTH = self.esm.emb_layer_norm_after(embeddings_BTH)
            # elif hasattr(self.esm, "layer_norm"):
            #     embeddings_BTH = self.esm.layer_norm(embeddings_BTH)
            # logits_BTV = self.esm.lm_head(embeddings_BTH)

        token_ids_BT = torch.argmax(logits_BTV, dim=-1)

        sequences = []
        for b in range(logits_BTV.shape[0]):
            valid_positions = torch.where(mask_BT[b])[0]
            valid_token_ids = token_ids_BT[b][mask_BT[b]]
            tokens = [alphabet.get_tok(t.item()) for t in valid_token_ids]

            # if select_indices is not None:
            #     for i in range(len(tokens)):
            #         if i not in select_indices:
            #             tokens[i] = wt_list[i]
            #         elif valid_muts is not None:
            #             valid_idxs = [alphabet.get_idx(aa) for aa in valid_muts[i]]
            #             best_idx = logits_BTV[b, valid_positions[i], valid_idxs].argmax().item()
            #             tokens[i] = valid_muts[i][best_idx]
            window_active = (min_position is not None and max_position is not None)
            if window_active:
                lo = min_position - 1
                hi = max_position - 1
            # Iterate over positions to apply constraints
            for i in range(len(tokens)):
                global_idx = valid_positions[i]
                if window_active and (i < lo or i > hi):
                    tokens[i] = wt_list[i]
                    continue
                
                # Check 1: Cosine Similarity Constraint
                # If the embedding is too similar to WT (>= 0.95), we force WT and skip argmax logic
                if cosine_similarity and sim_mask_BT[b, global_idx]:
                    tokens[i] = wt_list[i]
                    continue 

                # Check 2: Select Indices / Valid Muts Constraint
                if select_indices is not None:
                    if i not in select_indices:
                        tokens[i] = wt_list[i]
                    elif valid_muts is not None:
                        # Restricted Argmax
                        valid_idxs = [alphabet.get_idx(aa) for aa in valid_muts[i]]
                        best_idx = logits_BTV[b, global_idx, valid_idxs].argmax().item()
                        tokens[i] = valid_muts[i][best_idx]

            # Apply max_mutations constraint if specified
            if max_mutations is not None and wt_list is not None:
                # Find all positions that differ from wildtype
                mutation_positions = []
                mutation_logits = []
                for i in range(len(tokens)):
                    if tokens[i] != wt_list[i]:
                        mutation_positions.append(i)
                        # Get the logit for the chosen token at this position
                        chosen_tok_idx = alphabet.get_idx(tokens[i])
                        mutation_logits.append(logits_BTV[b, valid_positions[i], chosen_tok_idx].item())
                
                # If more mutations than allowed, keep only the top ones by logit confidence
                if len(mutation_positions) > max_mutations:
                    # Sort by logit value (descending) and keep top max_mutations
                    sorted_indices = sorted(range(len(mutation_logits)), key=lambda x: mutation_logits[x], reverse=True)
                    positions_to_keep = set(mutation_positions[idx] for idx in sorted_indices[:max_mutations])
                    
                    # Revert positions not in the top max_mutations to wildtype
                    for i in mutation_positions:
                        if i not in positions_to_keep:
                            tokens[i] = wt_list[i]

            sequences.append(''.join(tokens))

        return sequences, logits_BTV
    
class FullCLTReplacementModel(FullReplacementModel):
    '''
    Cross-Layer Transformer (CLT) replacement model.
    Uses latents from ALL previous layers (0...l) to reconstruct layer l's MLP output.
    '''
    def __init__(self, pl_module, device):
        print("Initializing FullCLTReplacementModel")
        super().__init__(pl_module, device)
        self.clt = self.model  # Alias for backwards compatibility

    def _decode_latents(self, l, current_latents_list):
        """
        Cross-layer decoding: Use latents from all layers 0 to l.
        """
        # Initialize output tensor using latents shape
        T, B, D = current_latents_list[l].shape
        recon_TBH = torch.zeros(T, B, self.esm.embed_dim, device=self.device)
        for src in range(l + 1):
            key = f"{src}_{l}"
            if key in self.model.decoders:
                recon_TBH = recon_TBH + (current_latents_list[src] @ self.model.decoders[key])
        return recon_TBH

class FullPLTReplacementModel(FullReplacementModel):
    '''
    Per-Layer Transformer (PLT) replacement model.
    Uses only the current layer's latents to reconstruct layer l's MLP output.
    '''
    def __init__(self, pl_module, device):
        super().__init__(pl_module, device)
        self.plt = self.model  # Alias for backwards compatibility

    def _decode_latents(self, l, current_latents_list):
        """
        Per-layer decoding: Use only current layer's latents.
        """
        return current_latents_list[l] @ self.model.decoders[l]


class FullCLTDirectReplacementModel(FullReplacementModel):
    '''
    CLT Direct replacement model that uses ground-truth MLP inputs at each layer.
    Unlike sequential processing, this encodes all layers in parallel using ground-truth
    inputs, then reconstructs only the final layer. This means the MLP output should be
    the actual ground-truth output of ESM (since it's just running ESM w/no corruptions).

    This requires ESM2ActivationCollector to gather activations from all layers.
    '''
    def __init__(self, pl_module, device):
        super().__init__(pl_module, device)
        self.clt = self.model  # Alias for backwards compatibility

        # Verify collector is available (initialized by base class)
        if not hasattr(self, 'collector') or self.collector is None:
            raise ImportError("ESM2ActivationCollector required for FullCLTDirectReplacementModel")

    def _decode_latents(self, l, current_latents_list):
        """
        Cross-layer decoding: Use latents from all layers 0 to l.
        """
        # Create output tensor from latents shape (S, H)
        recon_SH = torch.zeros(current_latents_list[0].shape[0], self.esm.embed_dim, device=self.device)

        for src in range(l + 1):
            key = f"{src}_{l}"
            if key in self.clt.decoders:
                recon_SH = recon_SH + (current_latents_list[src] @ self.clt.decoders[key])

        return recon_SH

    def forward(self, batch_seqs, ablate_nodes=None):
        """
        Direct Forward Pass using ground-truth MLP inputs.

        Args:
            batch_seqs: List of sequence strings
            ablate_nodes: Dict mapping layer indices to sets/lists of node indices to ablate, or None

        Returns:
            modified_stream_BTH: Modified stream (orig_layer_out - orig_mlp + recon_mlp)
            latents_list_L: List of latents for each layer
            recon_mlp_BTH: The reconstructed MLP output of the final layer
            gt_mlp_BTH: Ground truth MLP output of final layer
            mask_BT: Boolean mask (B, T) where True = valid token, False = special token
        """
        self.clt.eval()
        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        # Tokenize sequences
        tokens_BT = self.tokenize(batch_seqs)

        # Create mask to exclude special tokens (CLS, EOS, padding)
        mask_BT = (tokens_BT != self.alphabet.cls_idx) & \
                  (tokens_BT != self.alphabet.eos_idx) & \
                  (tokens_BT != self.alphabet.padding_idx)

        # 1. Collect activations from ESM
        x_stack_SLH, _, x_mlp_out_SLH, x_clt_in_SLH, _ = self.collector.collect(tokens_BT)

        B, T = tokens_BT.shape
        S, L, H = x_clt_in_SLH.shape

        # 2. Encode all layers in parallel using ground-truth inputs
        latents_list_L = []

        for l in range(L):
            x_in_SH = x_clt_in_SLH[:, l, :]
            x_norm_SH, mu, std = self.clt.LN(x_in_SH)
            # mu and std will naturally be from the final layer after the loop
            x_norm_SH = x_norm_SH - self.clt.b_pre[l]
            enc_SD = self.clt.encoders[l](x_norm_SH) + self.clt.b_enc[l]
            latents_SD = self.clt.topK_activation(enc_SD, k=self.clt.k)

            # Apply ablation
            if ablate_nodes is not None and l in ablate_nodes:
                node_indices = list(ablate_nodes[l])
                print("ablating nodes: ", node_indices)
                if len(node_indices) > 0:
                    node_mask = torch.ones(latents_SD.shape[-1], device=self.device)
                    node_mask[node_indices] = 0.0
                    latents_SD = latents_SD * node_mask

            latents_list_L.append(latents_SD)

        # 3. Decode final layer using all latents
        recon_SH = self._decode_latents(self.num_layers - 1, latents_list_L)

        # Denormalize (mu and std are from final layer)
        recon_SH = recon_SH + self.clt.b_pre[-1]
        recon_SH = recon_SH * std + mu

        # 4. Reshape and compute modified stream
        orig_layer_out_BTH = x_stack_SLH[:, -1, :].view(B, T, H)
        orig_mlp_BTH = x_mlp_out_SLH[:, -1, :].view(B, T, H)
        recon_mlp_BTH = recon_SH.view(B, T, H)

        # Modified stream = original stream - original MLP + reconstructed MLP
        # since the layer output is x^l + y^l, we are doing:
        # (x^l + y^l) - y^l + \hat{y}^l = x^l + \hat{y}^l
        modified_stream_BTH = orig_layer_out_BTH - orig_mlp_BTH + recon_mlp_BTH

        return modified_stream_BTH, latents_list_L, recon_mlp_BTH, orig_layer_out_BTH, mask_BT

    def forward_steered(self, seq, circuit, before=False, ablate_nodes=None, alphas=None, freeze_attention=False, add_correction=True):
        """
        Steered Forward Pass using a circuit.

        Args:
            seq: Input sequence string
            alphas: List of scalars (hyperparameter), where B = len(alphas)
            circuit: Dict mapping layer indices to lists of node indices to steer
            before: Whether to steer latents before or after applying TopK
            ablate_nodes: Dict mapping layer indices to lists of node indices to zero-ablate, or None
            add_correction: Whether to add the correction term (recon_error_1TH) to the modified stream
        Returns:
            x_curr_BTH: Final layer replacement model output (Input + Attn + MLP_Recon)
            latents_list_L: List of latents
            recon_mlp_BTH: The MLP reconstruction of the final layer (just the MLP part)
            gt_mlp_BTH: Ground truth MLP output of final layer for sanity checking
            mask_BT: Boolean mask (B, T) where True = valid token, False = special token
        """
        self.model.eval()
        if isinstance(alphas, (float, int)):
            alphas = [alphas]
        if not isinstance(alphas, torch.Tensor):
            alphas = torch.tensor(alphas, device=self.device, dtype=torch.float32)
        else:
            alphas = alphas.to(self.device, dtype=torch.float32)
        if alphas.ndim == 0:       # 0-dim scalar tensor -> (1,)
            alphas = alphas.unsqueeze(0)
        elif alphas.ndim > 1:      # (1, B) or (B, 1) -> (B,)
            alphas = alphas.reshape(-1)
        B = alphas.shape[0]

        # check if the circuit doesn't have int keys, if so convert them
        if not all(isinstance(k, int) for k in circuit.keys()):
            circuit = {int(k): v for k, v in circuit.items()}

        if ablate_nodes is not None and not all(isinstance(k, int) for k in ablate_nodes.keys()):
            ablate_nodes = {int(k): v for k, v in ablate_nodes.items()}

        if add_correction:
            # Get the CLT's reconstruction error
            clt_out_1TH, _, _, gt_mlp_1TH, _ = self.forward([seq], ablate_nodes=ablate_nodes)
            # gt_mlp_1TH: ACTUAL last layer output of model
            # clt_out_1TH: CLT approximation of last layer output
            recon_error_1TH = gt_mlp_1TH - clt_out_1TH
            # stack recon_error_1TH to be (B, T, H), eg just repeat it B times
            recon_error_BTH = recon_error_1TH.expand(B, -1, -1)
            # if the CLT has perfect recon; this should be all 0s

        # why do this? we know that clt_out_1TH is going to be an inaccurate approximation 
        # of the true steered last-layer output because the CLT isn't 100% accurate, so adding
        # back the recon error should give us a better output. this is common in SAE steering;
        # makes sense it would probably works for CLT!

        # Tokenize sequences and get embeddings
        tokens_BT = self.tokenize([seq] * B)
        with torch.no_grad():
            # convert from tokens to embds
            x_curr_BTH = self.esm.embed_scale * self.esm.embed_tokens(tokens_BT)

        # Create mask to exclude special tokens (CLS, EOS, padding)
        mask_BT = (tokens_BT != self.alphabet.cls_idx) & \
                    (tokens_BT != self.alphabet.eos_idx) & \
                    (tokens_BT != self.alphabet.padding_idx)
        x_stack_SLH, _, x_mlp_out_SLH, x_clt_in_SLH, _ = self.collector.collect(tokens_BT)

        T = tokens_BT.shape[1]

        S, L, H = x_clt_in_SLH.shape

        latents_list_L = []
        for l in range(L):
            x_in_SH = x_clt_in_SLH[:, l, :]
            x_norm_SH, mu, std = self.clt.LN(x_in_SH)
            # mu and std will naturally be from the final layer after the loop
            x_norm_SH = x_norm_SH - self.clt.b_pre[l]
            enc_SD = self.clt.encoders[l](x_norm_SH) + self.clt.b_enc[l]
            latents_SD = self.clt.topK_activation(enc_SD, k=self.clt.k)

            if l in circuit:
                node_indices = list(circuit[l])
                if len(node_indices) > 0:
                    alpha_S = alphas.repeat_interleave(T) # (S,)
                    target_tensor_SD = enc_SD if before else latents_SD
                    current_max_S = target_tensor_SD.amax(dim=1)
                    injection_values_S1 = (current_max_S * alpha_S).unsqueeze(1)
                    target_tensor_SD[:, node_indices] = injection_values_S1
                if before:
                    # apply TopK and continue
                    latents_SD = self.clt.topK_activation(target_tensor_SD, k=self.clt.k)

            # Apply ablation
            if ablate_nodes is not None and l in ablate_nodes:
                node_indices = list(ablate_nodes[l])
                if len(node_indices) > 0:
                    node_mask = torch.ones(latents_SD.shape[-1], device=self.device)
                    node_mask[node_indices] = 0.0
                    latents_SD = latents_SD * node_mask

            latents_list_L.append(latents_SD)

        recon_SH = self._decode_latents(self.num_layers - 1, latents_list_L)

        recon_SH = recon_SH + self.clt.b_pre[-1]
        recon_SH = recon_SH * std + mu

        orig_layer_out_BTH = x_stack_SLH[:, -1, :].view(B, T, H)
        orig_mlp_BTH = x_mlp_out_SLH[:, -1, :].view(B, T, H)
        recon_mlp_BTH = recon_SH.view(B, T, H)

        # Modified stream = original stream - original MLP + reconstructed MLP
        # since the layer output is x^l + y^l, we are doing:
        # (x^l + y^l) - y^l + \hat{y}^l = x^l + \hat{y}^l + (y^l - \hat{y}^l)
        # the (y^l - \hat{y}^l) term is the reconstruction error for steering
        if add_correction:
            modified_stream_BTH = orig_layer_out_BTH - orig_mlp_BTH + recon_mlp_BTH + recon_error_BTH
        else:
            modified_stream_BTH = orig_layer_out_BTH - orig_mlp_BTH + recon_mlp_BTH

        return modified_stream_BTH, latents_list_L, recon_mlp_BTH, orig_layer_out_BTH, mask_BT
