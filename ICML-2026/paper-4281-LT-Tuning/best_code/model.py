# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Optional
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.gpt2 import GPT2LMHeadModel
import gc
from safetensors.torch import load_file
Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "last_hidden_state", "attentions"]
)
MAX_THINKING_AUTOREGRESS = 32


class LT_Tuning_Model(nn.Module):

    def __init__(
        self,
        base_causallm,
        thinking_token_id,
        eos_token_id,
        use_thinking_mlp: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        mlp_activation: str = "gelu",
        hidden_state_layer_index: int = -1,
        stage_mode: str = "explicit",  # explicit, hidden_state, soft_fusion
        fusion_alpha: float = 0.6,  # weight for hidden state in soft fusion
        fusion_top_p: float = 0.9,  # top-p threshold for token selection
        fusion_temperature: float = 1.0,  # temperature for softmax
        **kwargs,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
            self.lm_head = self.base_causallm.lm_head
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            self.lm_head = self.base_causallm.lm_head
        self.thinking_token_id = thinking_token_id
        self.eos_token_id = eos_token_id
        self.gen_forward_cnt = 0

        self.use_thinking_mlp = use_thinking_mlp
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_activation = mlp_activation
        self.hidden_state_layer_index = hidden_state_layer_index
        
        # Stage control
        self.stage_mode = stage_mode
        # Use register_buffer for fusion_alpha to avoid meta device issues
        # It's not a learnable parameter anyway
        self.register_buffer("fusion_alpha", torch.tensor(fusion_alpha))
        self.fusion_top_p = fusion_top_p
        self.fusion_temperature = fusion_temperature

        if self.use_thinking_mlp:
            input_dim = self.embedding.embedding_dim
            hidden_dim = self.mlp_hidden_dim or input_dim
            activation_module = self._get_activation(self.mlp_activation)
            self.thinking_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_module,
                nn.Linear(hidden_dim, input_dim),
                nn.LayerNorm(input_dim),
            )
        else:
            self.thinking_mlp = None

    @property
    def config(self):
        """Expose base model's config for HuggingFace Trainer/DeepSpeed compatibility."""
        return self.base_causallm.config

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Load LT_Tuning_Model model from pretrained weights.
        
        Supports two formats:
        1. Standard HuggingFace format: has config.json, weights are base_causallm's
        2. LT_Tuning_Model checkpoint format: single model.safetensors with LT_Tuning_Model structure
        """
        print("Using from_pretrained scheme for LTF models...")
        print(f"Loading model from: {model_path}")
        print(f"Target device: {device}")
        
        config_path = os.path.join(model_path, "config.json")
        single_safetensor = os.path.join(model_path, "model.safetensors")
        
        # Detect format: if config.json exists, it's HuggingFace format
        # If only model.safetensors exists without config.json, it's LT_Tuning_Model checkpoint format
        has_config = os.path.exists(config_path)
        has_single_safetensor = os.path.exists(single_safetensor)
        
        # Check if it's LT_Tuning_Model checkpoint format (no config but has model.safetensors)
        is_LT_Tuning_Model_checkpoint = has_single_safetensor and not has_config
        
        if is_LT_Tuning_Model_checkpoint:
            print("Detected LT_Tuning_Model checkpoint format (no config.json, has model.safetensors)")
            return cls._load_from_LT_Tuning_Model_checkpoint(model_path, device, **kwargs)
        else:
            print("Detected HuggingFace format (has config.json)")
            return cls._load_from_hf_format(model_path, device, **kwargs)
    
    @classmethod
    def _load_from_hf_format(
        cls,
        model_path: str,
        device: str,
        **kwargs
    ):
        """Load from standard HuggingFace format (base_causallm weights)."""
        # Get attention implementation from kwargs (default: sdpa for wider compatibility)
        attn_implementation = kwargs.pop("attn_implementation", "sdpa")
        
        base_causallm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        print(f"Base model loaded: {type(base_causallm).__name__} (attn: {attn_implementation})")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        t_id = kwargs.get("thinking_token_id")
        if t_id is None:
            if "<thinking>" in tokenizer.get_vocab():
                t_id = tokenizer.convert_tokens_to_ids("<thinking>")
            else:
                t_id = tokenizer.unk_token_id
                
        if "thinking_token_id" in kwargs:
            model = cls(
                base_causallm=base_causallm,
                **kwargs
            )
        else:
            model = cls(
                base_causallm=base_causallm,
                thinking_token_id=t_id,
                **kwargs
            )
        custom_weights_path = os.path.join(model_path, "LT_Tuning_Model_custom_weights.pt")
        if os.path.exists(custom_weights_path):
            print(f"Loading custom LT_Tuning_Model weights from {custom_weights_path}")
            custom_state = torch.load(custom_weights_path, map_location=device)
            model.load_state_dict(custom_state, strict=False)
        
        print(f"Successfully loaded model (HF format) from {model_path}.")
        model.eval()
        
        return model
    
    @classmethod
    def _load_from_LT_Tuning_Model_checkpoint(
        cls,
        model_path: str,
        device: str,
        **kwargs
    ):
        """Load from LT_Tuning_Model checkpoint format (full LT_Tuning_Model structure in model.safetensors)."""
        base_model_path = kwargs.pop("base_model_name_or_path", None)

        print(f"Loading base model from: {base_model_path}")
        
        # Get attention implementation from kwargs (default: sdpa for wider compatibility)
        attn_implementation = kwargs.pop("attn_implementation", "sdpa")
        base_causallm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        print(f"Base model structure loaded from: {base_model_path} (attn: {attn_implementation})")
        
        checkpoint_file = os.path.join(model_path, "model.safetensors")
        checkpoint_state = load_file(checkpoint_file)
        checkpoint_vocab_size = None
        for key in ["base_causallm.model.embed_tokens.weight", "embedding.weight"]:
            if key in checkpoint_state:
                checkpoint_vocab_size = checkpoint_state[key].shape[0]
                break
        
        if checkpoint_vocab_size is None:
            raise RuntimeError(f"Cannot determine vocab size from checkpoint at {model_path}")
        
        print(f"Checkpoint vocab size: {checkpoint_vocab_size}")

        tokenizer_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        thinking_token = kwargs.pop("thinking_token", "<thinking>")
        use_unk_for_thinking = kwargs.pop("use_unk_for_thinking", False)
        
        if use_unk_for_thinking:
            if tokenizer.unk_token_id is None:
                raise ValueError("Tokenizer has no unk_token but use_unk_for_thinking=True")
            t_id = tokenizer.unk_token_id
            print(f"Using UNK token as thinking token: {t_id}")
        else:
            if thinking_token in tokenizer.get_vocab():
                t_id = tokenizer.convert_tokens_to_ids(thinking_token)
                print(f"Thinking token '{thinking_token}' already in vocab at index {t_id}")
            else:
                newly_added = tokenizer.add_tokens([thinking_token])
                t_id = tokenizer.convert_tokens_to_ids(thinking_token)
                print(f"Added thinking token '{thinking_token}' at index {t_id} (newly_added={newly_added})")
        
        current_vocab_size = base_causallm.get_input_embeddings().num_embeddings
        if current_vocab_size != checkpoint_vocab_size:
            print(f"Resizing embeddings from {current_vocab_size} to {checkpoint_vocab_size} to match checkpoint")
            base_causallm.resize_token_embeddings(checkpoint_vocab_size)
            
            if len(tokenizer) != checkpoint_vocab_size:
                print(f"Warning: tokenizer size {len(tokenizer)} != checkpoint vocab size {checkpoint_vocab_size}")
                if len(tokenizer) < checkpoint_vocab_size:
                    num_to_add = checkpoint_vocab_size - len(tokenizer)
                    print(f"Adding {num_to_add} placeholder tokens to match checkpoint")
                    tokenizer.add_tokens([f"<placeholder_{i}>" for i in range(num_to_add)])
        
        t_id = kwargs.get("thinking_token_id", t_id)
        
        if "thinking_token_id" in kwargs:
            model = cls(
                base_causallm=base_causallm,
                **kwargs
            )
        else:
            model = cls(
                base_causallm=base_causallm,
                thinking_token_id=t_id,
                **kwargs
            )
        
        print("Model skeleton initialized, loading weights from checkpoint...")
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        if missing_keys:
            print(f"Missing keys (expected for non-trained params): {len(missing_keys)}")
            if len(missing_keys) > 0:
                print(f"Sample missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        model = model.to(device)
        if model.thinking_mlp is not None:
            model.thinking_mlp = model.thinking_mlp.to(device)
            print(f"Moved thinking_mlp to device: {device}")
        
        print(f"Successfully loaded model (LT_Tuning_Model checkpoint) from {model_path}.")
        model.eval()
        
        return model

    def update_stage_config(self, stage_mode: str, fusion_alpha: float = None, 
                           fusion_top_p: float = None, fusion_temperature: float = None):
        """Update stage-specific configuration dynamically."""
        self.stage_mode = stage_mode
        if fusion_alpha is not None:
            self.fusion_alpha = torch.tensor(fusion_alpha, device=self.fusion_alpha.device, dtype=self.fusion_alpha.dtype)
        if fusion_top_p is not None:
            self.fusion_top_p = fusion_top_p
        if fusion_temperature is not None:
            self.fusion_temperature = fusion_temperature

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation for thinking MLP: {name}")

    def _apply_transform(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.use_thinking_mlp and self.thinking_mlp is not None:
            return self.thinking_mlp(hidden_state)
        return hidden_state
    
    def _soft_fusion_embedding(self, hidden_state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Fuse hidden state with top-p sampled token embeddings."""
        # Apply temperature scaling
        scaled_logits = logits / self.fusion_temperature
        
        # Mask out thinking token from distribution
        masked_logits = scaled_logits.clone()
        masked_logits[self.thinking_token_id] = float('-inf')
        
        # Compute softmax probabilities
        probs = torch.softmax(masked_logits, dim=-1)
        
        # Apply top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff for top-p
        cutoff_mask = cumsum_probs <= self.fusion_top_p
        if cutoff_mask.sum() == 0:  # Include at least one token
            cutoff_mask[0] = True
        
        # Zero out probabilities below threshold
        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_indices[cutoff_mask]] = sorted_probs[cutoff_mask]
        filtered_probs = filtered_probs / filtered_probs.sum()  # Renormalize
        
        # Weighted sum of embeddings
        token_embed_component = filtered_probs @ self.embedding.weight
        
        # Combine with hidden state using fusion_alpha
        alpha = self.fusion_alpha.item()
        fused_embed = alpha * hidden_state + (1 - alpha) * token_embed_component
        
        return fused_embed
    
    def get_fusion_stats(self, hidden_state: torch.Tensor, logits: torch.Tensor, tokenizer=None):
        """Get statistics about fusion for debugging/monitoring (call with no_grad)."""
        with torch.no_grad():
            scaled_logits = logits / self.fusion_temperature
            masked_logits = scaled_logits.clone()
            masked_logits[self.thinking_token_id] = float('-inf')
            probs = torch.softmax(masked_logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumsum_probs <= self.fusion_top_p
            if cutoff_mask.sum() == 0:
                cutoff_mask[0] = True
            
            stats = {
                'top_token_prob': sorted_probs[0].item(),
                'num_tokens_in_top_p': cutoff_mask.sum().item(),
                'fusion_alpha': self.fusion_alpha.item(),
                'entropy': -(probs * torch.log(probs + 1e-10)).sum().item(),
            }
            
            if tokenizer is not None:
                top_5_tokens = [(tokenizer.decode([idx.item()]), prob.item()) 
                               for idx, prob in zip(sorted_indices[:5], sorted_probs[:5])]
                stats['top_5_tokens'] = top_5_tokens
            
            return stats

    def _select_hidden_state(self, hidden_states) -> torch.Tensor:
        index = self.hidden_state_layer_index
        if index is None:
            index = -1
        if index < 0:
            index = len(hidden_states) + index
        if index < 0 or index >= len(hidden_states):
            raise IndexError(
                f"hidden_state_layer_index {self.hidden_state_layer_index} out of range for hidden_states of length {len(hidden_states)}"
            )
        return hidden_states[index]

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels = None,
        position_ids = None,
        return_probing_info = False,
        output_attentions = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        embed_dim = self.embedding.embedding_dim
        
        # Storage for attention outputs from all segments
        all_attentions = [] if output_attentions else None

        thinking_positions_batch = []
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        attention_mask_bool = attention_mask.bool()
        for batch_idx in range(batch_size):
            thinking_mask = (
                (input_ids[batch_idx] == self.thinking_token_id)
                & attention_mask_bool[batch_idx]
            )
            positions = torch.nonzero(thinking_mask, as_tuple=False).squeeze(-1)
            thinking_positions_batch.append(positions.tolist())

        boundary_positions = set()
        for positions in thinking_positions_batch:
            for pos in positions:
                if 0 < pos < seq_len:
                    boundary_positions.add(pos)
        boundary_positions = sorted(boundary_positions)
        boundary_positions.append(seq_len)
        current_start = 0
        kv_cache = None
        outputs = None
        logits_segments = []
        segment_embeds_accum = []
        thinking_sets = [set(lst) for lst in thinking_positions_batch]
        thinking_replacements = {}

        selected_hidden_state = None

        for boundary_end in boundary_positions:
            if boundary_end <= current_start:
                continue

            segment_length = boundary_end - current_start
            if segment_length <= 0:
                current_start = boundary_end
                continue

            segment_slice = slice(current_start, boundary_end)
            segment_ids = input_ids[:, segment_slice]

            segment_embeds = torch.zeros(
                batch_size,
                segment_length,
                embed_dim,
                device=segment_ids.device,
                dtype=self.embedding.weight.dtype,
            )

            for offset in range(segment_length):
                absolute_pos = current_start + offset
                token_column = segment_ids[:, offset]
                thinking_mask = token_column == self.thinking_token_id

                if (~thinking_mask).any():
                    non_thinking_ids = token_column[~thinking_mask]
                    non_thinking_embeds = self.embedding(non_thinking_ids)
                    segment_embeds[~thinking_mask, offset, :] = non_thinking_embeds

                if thinking_mask.any():
                    thinking_indices = torch.nonzero(thinking_mask, as_tuple=False).flatten()
                    for batch_idx in thinking_indices.tolist():
                        replacement = thinking_replacements.get((batch_idx, absolute_pos))
                        if replacement is None:
                            raise RuntimeError(
                                "Missing replacement embedding for thinking token at position"
                                f" {absolute_pos} in batch {batch_idx}."
                            )
                        segment_embeds[batch_idx, offset, :] = replacement
                        del thinking_replacements[(batch_idx, absolute_pos)]

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=segment_embeds,
                    attention_mask=attention_mask[:, segment_slice],
                    position_ids=position_ids[:, segment_slice],
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=True,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=segment_embeds,
                    attention_mask=attention_mask[:, :boundary_end],
                    position_ids=position_ids[:, segment_slice],
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=True,
                )
            
            # Collect attention weights if requested
            if output_attentions and outputs.attentions is not None:
                all_attentions.append(outputs.attentions)

            logits_segments.append(outputs.logits)
            segment_embeds_accum.append(segment_embeds)
            hidden_states = self._select_hidden_state(outputs.hidden_states)
            selected_hidden_state = hidden_states
            kv_cache = outputs.past_key_values

            if boundary_end < seq_len:
                batch_indices_to_update = [
                    batch_idx
                    for batch_idx in range(batch_size)
                    if boundary_end in thinking_sets[batch_idx]
                ]

                if batch_indices_to_update and segment_length > 0:
                    hidden_idx = segment_length - 1
                    for batch_idx in batch_indices_to_update:
                        prev_hidden = hidden_states[batch_idx, hidden_idx, :]
                        
                        # Stage-specific embedding generation
                        if self.stage_mode == "hidden_state":
                            # Stage 1: Use hidden state directly (with optional MLP), <thinking> has no embedding
                            replacement = self._apply_transform(prev_hidden)
                        elif self.stage_mode == "soft_fusion":
                            # Stage 2: Fuse hidden state with predicted token embeddings
                            prev_logits = outputs.logits[batch_idx, hidden_idx, :]
                            replacement = self._soft_fusion_embedding(prev_hidden, prev_logits)
                            replacement = self._apply_transform(replacement)
                        else:
                            # Stage 0 (explicit) or default: normal hidden state
                            replacement = self._apply_transform(prev_hidden)
                        
                        thinking_replacements[(batch_idx, boundary_end)] = replacement
                        thinking_sets[batch_idx].remove(boundary_end)

            current_start = boundary_end

        if outputs is None:
            raise RuntimeError("No forward pass executed in LT_Tuning_Model model.")

        logits = torch.cat(logits_segments, dim=1)
        inputs_embeds = torch.cat(segment_embeds_accum, dim=1)
        self.gen_forward_cnt += len(logits_segments)
        del logits_segments
        del segment_embeds_accum
        gc.collect()
        torch.cuda.empty_cache()
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = None

        if selected_hidden_state is None:
            raise RuntimeError("Failed to select hidden state during forward pass in LT_Tuning_Model.")
        merged_attentions = None
        if output_attentions and all_attentions:
            num_layers = len(all_attentions[0])
            merged_attentions = []
            
            for layer_idx in range(num_layers):
                # Get attention from all segments for this layer
                layer_attentions = [seg_attn[layer_idx] for seg_attn in all_attentions]
                
                # Build full attention matrix by padding and concatenating
                batch_size, num_heads = layer_attentions[0].shape[:2]
                
                # Calculate total sequence length from the last segment's key dimension
                total_seq_len = layer_attentions[-1].shape[-1]
                
                # Create full attention matrix
                full_attention = torch.zeros(
                    batch_size, num_heads, total_seq_len, total_seq_len,
                    dtype=layer_attentions[0].dtype,
                    device=layer_attentions[0].device
                )
                
                # Fill in attention values from each segment
                query_start = 0
                for seg_attn in layer_attentions:
                    seg_query_len = seg_attn.shape[2]
                    seg_key_len = seg_attn.shape[3]
                    
                    # The segment attends to positions [0, seg_key_len)
                    # Query positions are [query_start, query_start + seg_query_len)
                    full_attention[:, :, query_start:query_start + seg_query_len, :seg_key_len] = seg_attn
                    query_start += seg_query_len
                
                merged_attentions.append(full_attention)
            
            merged_attentions = tuple(merged_attentions)

        # Prepare output with optional probing info
        output = Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=selected_hidden_state,
            attentions=merged_attentions,
        )
        
        if return_probing_info:
            # Directly extract logits from final logits tensor
            # Assume batch_size = 1 for probing
            probing_info = {
                'thinking_positions': thinking_positions_batch[0],
                'logits_before_thinking': [],  # Logits at position before <thinking>
                'logits_at_thinking': [],   # Logits at <thinking> position after transformation
            }
            
            for thinking_pos in thinking_positions_batch[0]:
                if thinking_pos > 0 and thinking_pos < logits.shape[1]:
                    # Logits at position before <thinking>
                    probing_info['logits_before_thinking'].append({
                        'thinking_position': thinking_pos,
                        'logits': logits[0, thinking_pos - 1].clone().detach()
                    })
                    # Logits at <thinking> position (after latent forward)
                    probing_info['logits_at_thinking'].append({
                        'thinking_position': thinking_pos,
                        'logits': logits[0, thinking_pos].clone().detach()
                    })
            
            return output, probing_info
        
        return output
    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        if self.thinking_mlp is not None:
            self.thinking_mlp.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        without_thinking_token=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            labels=None,
            position_ids=torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds
        past_key_values = outputs.past_key_values
        selected_hidden_state = outputs.last_hidden_state

        # get the first token using the current hidden state
        if without_thinking_token:
            outputs.logits[0, -1, self.thinking_token_id] = float('-inf')

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        if next_token == self.thinking_token_id:
            prev_hidden = selected_hidden_state[:, -1, :]
            if self.stage_mode == "soft_fusion":
                prev_logits = outputs.logits[0, -1, :]
                fused = self._soft_fusion_embedding(prev_hidden, prev_logits)
                new_token_embed = self._apply_transform(fused).unsqueeze(1)
            else:
                # hidden_state mode or default: use hidden state as embedding
                new_token_embed = self._apply_transform(prev_hidden).unsqueeze(1)
        else:
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)

        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            self.gen_forward_cnt += 1
            if without_thinking_token:
                outputs.logits[0, -1, self.thinking_token_id] = float('-inf')
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            past_key_values = outputs.past_key_values
            selected_hidden_state = self._select_hidden_state(outputs.hidden_states)
            if next_token == self.thinking_token_id:
                prev_hidden = selected_hidden_state[:, -1, :]
                if self.stage_mode == "soft_fusion":
                    prev_logits = outputs.logits[0, -1, :]
                    fused = self._soft_fusion_embedding(prev_hidden, prev_logits)
                    new_token_embed = self._apply_transform(fused).unsqueeze(1)
                else:
                    # hidden_state mode or default: use hidden state as embedding
                    new_token_embed = self._apply_transform(prev_hidden).unsqueeze(1)
            else:
                new_token_embed = self.embedding(
                    torch.tensor(next_token, device=input_ids.device)
                ).view(1, 1, -1)

            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while self.gen_forward_cnt < max_new_tokens + MAX_THINKING_AUTOREGRESS:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        generated_tokens = torch.tensor(
            tokens, dtype=input_ids.dtype, device=input_ids.device
        ).view(1, -1)

        if output_embedding:
            # for analysis purpose
            return generated_tokens, new_inputs_embeds

        return generated_tokens


# Alias for backward compatibility with evaluation scripts
SoftSeft = LT_Tuning_Model
