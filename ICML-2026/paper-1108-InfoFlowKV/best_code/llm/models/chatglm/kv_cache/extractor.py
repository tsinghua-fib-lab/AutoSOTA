"""KV cache extraction utilities."""

import torch
from typing import List, Tuple, Optional, Any
from transformers.cache_utils import DynamicCache

from .base import KVCacheData


class KVCacheExtractor:
    """Extract KV cache from context passages."""

    def __init__(self, model, tokenizer, model_type: str = "glm"):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            model_type: Model type ("qwen", "glm", "llama")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.device = next(model.parameters()).device

        # Get model config
        config = model.config
        self.num_layers = getattr(config, "num_layers",
                                  getattr(config, "num_hidden_layers", 0))
        self.num_heads = getattr(config, "num_attention_heads", 0)
        self.num_kv_heads = getattr(config, "multi_query_group_num",
                                    getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = getattr(config, "hidden_size", 0) // max(1, self.num_heads)
        self.kv_head_dim = getattr(config, "kv_channels", self.head_dim)
        
        # Get decoder to skip lm_head
        self.decoder = getattr(model, "model", None) \
            or getattr(model, "transformer", None) \
            or getattr(model, "base_model", None)
        assert self.decoder is not None, "Cannot find model decoder"

    def _get_prefix_tokens(self) -> List[int]:
        """Get ChatGLM prefix tokens: [gMASK], <sop>, <|user|>, \n."""
        prefix_tokens = self.tokenizer.get_prefix_tokens()
        user_token = self.tokenizer.convert_tokens_to_ids("<|user|>")
        newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        return prefix_tokens + [user_token] + newline_tokens

    def _remove_rope_glm(self, x: torch.Tensor, rope_cache: torch.Tensor,
                        position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Remove GLM's RoPE (inverse transformation)."""
        B, H, T, D = x.shape
        rot_dim = rope_cache.size(1) * 2
        
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        
        rope = rope_cache[position_ids]
        cos = rope[..., 0]
        sin = rope[..., 1]
        
        x_rot = x_rot.reshape(B, H, T, rot_dim // 2, 2)
        x_rot_0 = x_rot[..., 0]
        x_rot_1 = x_rot[..., 1]
        
        # Inverse rotation
        x_rot_inv_0 = x_rot_0 * cos + x_rot_1 * sin
        x_rot_inv_1 = -x_rot_0 * sin + x_rot_1 * cos
        
        x_rot_inv = torch.stack([x_rot_inv_0, x_rot_inv_1], dim=-1)
        x_rot_inv = x_rot_inv.reshape(B, H, T, rot_dim)
        
        return torch.cat([x_rot_inv, x_pass], dim=-1)

    def _apply_rope_glm(self, x: torch.Tensor, rope_cache: torch.Tensor,
                       position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply GLM's RoPE."""
        B, H, T, D = x.shape
        rot_dim = rope_cache.size(1) * 2
        
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        
        rope = rope_cache[position_ids]
        cos = rope[..., 0]
        sin = rope[..., 1]
        
        x_rot = x_rot.reshape(B, H, T, rot_dim // 2, 2)
        x_rot_0 = x_rot[..., 0]
        x_rot_1 = x_rot[..., 1]
        
        # Apply rotation
        x_rot_new_0 = x_rot_0 * cos - x_rot_1 * sin
        x_rot_new_1 = x_rot_0 * sin + x_rot_1 * cos
        
        x_rot_new = torch.stack([x_rot_new_0, x_rot_new_1], dim=-1)
        x_rot_new = x_rot_new.reshape(B, H, T, rot_dim)
        
        return torch.cat([x_rot_new, x_pass], dim=-1)

    @torch.no_grad()
    def extract_from_passages(
        self,
        passages: List[str],
        batch_size: int = 1,
    ) -> Tuple[List[List[Tuple[torch.Tensor, torch.Tensor]]], List[int], List[List[int]]]:
        """
        Extract KV cache chunks from passages in batches.

        Args:
            passages: List of text passages
            batch_size: Batch size for processing

        Returns:
            kv_chunks: List of per-layer (K, V) tuples for each passage
            seq_lens: Length of each passage's KV cache
        """
        device = self.device
        kv_chunks = [None] * len(passages)
        seq_lens = [0] * len(passages)
        ids_each: List[List[int]] = [None] * len(passages)

        prefix_token_ids = self._get_prefix_tokens()
        prefix_len = len(prefix_token_ids)

        for s in range(0, len(passages), batch_size):
            batch_texts = passages[s:s + batch_size]

            # Sort within batch by length to minimize padding
            batch_indices = list(range(len(batch_texts)))
            batch_indices.sort(key=lambda i: len(batch_texts[i]))
            batch_texts_sorted = [batch_texts[i] for i in batch_indices]

            # Tokenize batch
            enc = self.tokenizer(
                batch_texts_sorted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False
            )

            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            # Add prefix tokens
            B = input_ids.size(0)
            prefix_ids_1 = torch.tensor(prefix_token_ids, device=device, dtype=input_ids.dtype).unsqueeze(0)  # (1, P)
            prefix_ids = prefix_ids_1.expand(B, -1)  # view, no copy
            prefix_mask = torch.ones((B, prefix_len), dtype=attn_mask.dtype, device=device)
            input_ids = torch.cat([prefix_ids, input_ids], dim=1)
            attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)

            # Forward pass to get KV cache (using decoder to skip lm_head)
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
            )

            past = outputs.past_key_values
            lens = attn_mask.sum(dim=1).tolist()

            # Extract KV for each passage in batch
            for b in range(B):
                original_idx = batch_indices[b]  # Map back to original position
                idx_global = s + original_idx
                if idx_global >= len(passages):
                    break

                if idx_global == 0:
                    # First chunk: keep the prefix tokens
                    T_b = int(lens[b])
                    seq_lens[idx_global] = T_b
                    
                    # Extract token ids from input_ids (includes prefix)
                    ids_each[idx_global] = input_ids[b, :T_b].tolist()

                    per_layer = []
                    for l in range(self.num_layers):
                        if isinstance(past, DynamicCache):
                            K = past.key_cache[l][b:b + 1, :, :T_b, :]
                            V = past.value_cache[l][b:b + 1, :, :T_b, :]
                        else:
                            K = past[l][0][b:b + 1, :, :T_b, :]
                            V = past[l][1][b:b + 1, :, :T_b, :]
                        per_layer.append((K, V))
                    kv_chunks[idx_global] = per_layer
                else:
                    # Following chunks: remove prefix tokens
                    T_b = int(lens[b]) - prefix_len
                    seq_lens[idx_global] = T_b
                    
                    # Extract token ids from input_ids (strip prefix)
                    ids_each[idx_global] = input_ids[b, prefix_len:prefix_len + T_b].tolist()

                    if T_b <= 0:
                        per_layer = [
                            (torch.empty((1, self.num_kv_heads, 0, self.kv_head_dim),
                                        device=device, dtype=torch.bfloat16),
                             torch.empty((1, self.num_kv_heads, 0, self.kv_head_dim),
                                        device=device, dtype=torch.bfloat16))
                            for _ in range(self.num_layers)
                        ]
                        kv_chunks[idx_global] = per_layer
                        continue

                    per_layer = []
                    for l in range(self.num_layers):
                        if isinstance(past, DynamicCache):
                            K = past.key_cache[l][b:b + 1, :, prefix_len:prefix_len + T_b, :]
                            V = past.value_cache[l][b:b + 1, :, prefix_len:prefix_len + T_b, :]
                        else:
                            K = past[l][0][b:b + 1, :, prefix_len:prefix_len + T_b, :]
                            V = past[l][1][b:b + 1, :, prefix_len:prefix_len + T_b, :]
                        per_layer.append((K, V))
                    kv_chunks[idx_global] = per_layer

            del outputs, past

        return kv_chunks, seq_lens, ids_each

    @torch.no_grad()
    def extract_without_RoPE_correction(
        self,
        context: str,
        default_split: bool = True,
        chunk_size: int = 512,
        batch_size: int = 1,
    ) -> KVCacheData:
        """
        Extract KV cache from full context without RoPE correction.

        Args:
            context: Full context string
            default_split: If True, use marker-based split; if False, use fixed-length split
            chunk_size: Token length for each chunk when default_split=False
            batch_size: Batch size for processing

        Returns:
            KVCacheData containing the extracted cache (as DynamicCache)
        """
        # Split context
        if default_split:
            passages = self._default_split(context, chunk_size)
        else:
            passages = self._fixed_length_split(context, chunk_size)

        # Extract KV chunks and per-passage token ids (ids_each)
        kv_chunks, seq_lens, ids_each = self.extract_from_passages(passages, batch_size)

        # Concatenate all tokens
        from itertools import chain
        ids_cat = list(chain.from_iterable(ids_each))
        input_ids = torch.tensor([ids_cat], dtype=torch.long, device=self.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Combine KV chunks into DynamicCache
        dyn = DynamicCache()
        dyn.key_cache = []
        dyn.value_cache = []

        for layer_idx in range(self.num_layers):
            K_list = [kv_chunks[i][layer_idx][0] for i in range(len(kv_chunks))]
            V_list = [kv_chunks[i][layer_idx][1] for i in range(len(kv_chunks))]
            K_full = torch.cat(K_list, dim=2)  # Concatenate along seq dimension
            V_full = torch.cat(V_list, dim=2)
            dyn.key_cache.append(K_full)
            dyn.value_cache.append(V_full)

        return KVCacheData(
            past_key_values=dyn,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_lens=seq_lens,
        )

    def _default_split(self, context: str, chunk_size: int = 1024) -> List[str]:
        """Default context splitting by markers.

        Tries in order: passage markers, paragraph breaks, bracket markers.
        Falls back to fixed-length splitting if no passage structure is found.
        """
        # 1. Check for LONGBENCH passage markers
        if '\nPassage ' in context:
            passages = context.split('\nPassage ')
            result = []
            for i, passage in enumerate(passages):
                if i == 0:
                    if passage.strip():
                        result.append(passage.strip())
                else:
                    result.append(f"Passage {passage.strip()}")
            if len(result) >= 2:
                return result

        # 2. Try paragraph breaks (double newlines)
        if '\n\n' in context:
            paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
            if len(paragraphs) >= 2:
                return paragraphs

        # 3. Fallback: split by Chinese/English bracket markers
        passages = []
        for line in context.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('【') or line.startswith('['):
                passages.append(line)
            elif passages:
                passages[-1] += '\n' + line
            else:
                passages.append(line)
        result = [p for p in passages if p]
        if len(result) >= 2:
            return result

        # No passage structure found — fall back to fixed-length splitting
        print(f"    [WARNING] No default passage structure found (context: {len(context)} chars). Falling back to fixed-length splitting ({chunk_size} tokens).")
        return self._fixed_length_split(context, chunk_size)

    def _fixed_length_split(self, context: str, chunk_size: int = 512) -> List[str]:
        """Split context into fixed-length chunks by tokens."""
        # Tokenize entire context
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks

    def _correct_rope_positions(
        self,
        kv_chunks: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        seq_lens: List[int],
        offsets: List[int],
        total_len: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Correct RoPE positions for concatenated KV chunks (GLM version).
        
        Args:
            kv_chunks: Raw KV chunks from extract_from_passages
            seq_lens: Sequence length for each chunk
            offsets: Starting position offset for each chunk
            total_len: Total sequence length
            
        Returns:
            K_final: List of corrected key tensors per layer
            V_final: List of value tensors per layer (copied)
        """
        device = self.device
        prefix_len = len(self._get_prefix_tokens())
        
        # Allocate final KV cache
        K_final = [
            torch.empty((1, self.num_kv_heads, total_len, self.kv_head_dim),
                       device=device, dtype=torch.bfloat16)
            for _ in range(self.num_layers)
        ]
        V_final = [
            torch.empty((1, self.num_kv_heads, total_len, self.kv_head_dim),
                       device=device, dtype=torch.bfloat16)
            for _ in range(self.num_layers)
        ]
        
        # Get RoPE cache for GLM
        rope_cache = self.model.transformer.rotary_pos_emb(total_len).to(
            device=device, dtype=self.model.dtype
        )
        
        # Correct RoPE for each chunk
        prefix_len = len(self._get_prefix_tokens())
        for chunk_idx, layer_kv in enumerate(kv_chunks):
            T_i = int(seq_lens[chunk_idx])
            off = int(offsets[chunk_idx])
            if T_i == 0:
                continue
            
            for layer_idx in range(self.num_layers):
                K_chunk, V_chunk = layer_kv[layer_idx]
                
                # Remove old RoPE
                if chunk_idx == 0:
                    position_ids_old = torch.arange(T_i, device=device)
                else:
                    position_ids_old = torch.arange(T_i, device=device) + prefix_len
                
                K_removed = self._remove_rope_glm(K_chunk, rope_cache, position_ids_old)
                
                # Apply new RoPE
                position_ids_new = torch.arange(off, off + T_i, device=device)
                K_rebased = self._apply_rope_glm(K_removed, rope_cache, position_ids_new)
                K_final[layer_idx][:, :, off:off+T_i, :].copy_(K_rebased)
                
                # Copy V without modification
                V_final[layer_idx][:, :, off:off+T_i, :].copy_(V_chunk)
        
        return K_final, V_final

    @torch.no_grad()
    def extract_with_rope_correction(
        self,
        context: str,
        default_split: bool = True,
        chunk_size: int = 512,
        batch_size: int = 1,
    ) -> KVCacheData:
        """
        Extract KV cache with RoPE position correction for concatenated chunks.
        
        Use this when you need correct positional encoding for concatenated passages.
        For no correction, use extract_full_context() instead.
        
        Args:
            context: Full context string
            default_split: If True, use marker-based split; if False, use fixed-length split
            chunk_size: Token length for each chunk when default_split=False
            batch_size: Batch size for processing
            
        Returns:
            KVCacheData with corrected RoPE positions (as DynamicCache)
        """
        from itertools import chain
        
        device = self.device
        
        # Split context based on split mode
        if default_split:
            passages = self._default_split(context, chunk_size)
        else:
            passages = self._fixed_length_split(context, chunk_size)
        
        # Extract KV chunks
        kv_chunks, seq_lens, ids_each = self.extract_from_passages(passages, batch_size=batch_size)
        
        # Calculate offsets
        offsets = []
        cur = 0
        for L_i in seq_lens:
            offsets.append(cur)
            cur += int(L_i)
        total_len = cur
        
        # Correct RoPE positions
        K_final, V_final = self._correct_rope_positions(kv_chunks, seq_lens, offsets, total_len)
        
        # Create DynamicCache
        dyn = DynamicCache()
        dyn.key_cache = [K.contiguous() for K in K_final]
        dyn.value_cache = [V.contiguous() for V in V_final]
        
        # Create combined input_ids
        ids_cat = list(chain.from_iterable(ids_each))
        input_ids = torch.tensor([ids_cat], dtype=torch.long, device=device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return KVCacheData(
            past_key_values=dyn,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_lens=seq_lens,
        )