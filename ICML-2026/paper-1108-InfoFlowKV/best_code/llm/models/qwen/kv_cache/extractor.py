"""KV cache extraction utilities."""

import torch
from typing import List, Tuple, Optional, Any
from transformers.cache_utils import DynamicCache
from flash_attn import flash_attn_varlen_func

from .base import KVCacheData


class KVCacheExtractor:
    """Extract KV cache from context passages."""

    def __init__(self, model, tokenizer, model_type: str = "qwen"):
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
        """Get Qwen prefix tokens (one BOS/EOS token)."""
        return [151643]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @torch.no_grad()
    def _apply_rope(
        self,
        x,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor = None,  
    ):
        if position_ids is not None:
            if position_ids.dim() == 1:                       # [K]
                cos= cos.index_select(1, position_ids)
                sin= sin.index_select(1, position_ids)
            if position_ids.dim() == 2:                       # [B,K]
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)   # [B,K,D]
                cos, sin = torch.gather(cos, 1, idx), torch.gather(sin, 1, idx)

        cos = cos.unsqueeze(1)  
        sin = sin.unsqueeze(1)

        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    @torch.no_grad()
    def _remove_rope(
        self,
        x,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor = None,   
    ):
        
        if position_ids is not None:
            if position_ids.dim() == 1:                       # [K]
                cos= cos.index_select(1, position_ids)
                sin= sin.index_select(1, position_ids)
            if position_ids.dim() == 2:                       # [B,K]
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)   # [B,K,D]
                cos, sin = torch.gather(cos, 1, idx), torch.gather(sin, 1, idx)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
        x = (x * cos) - (self._rotate_half(x) * sin)
        return x

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
            batch_size: Batch size for processing (default=1 for safety)

        Returns:
            kv_chunks: List of per-layer (K, V) tuples for each passage
            seq_lens: Length of each passage's KV cache
        """
        self.tokenizer.padding_side = "right"
        device = self.device
        kv_chunks = [None] * len(passages)
        seq_lens = [0] * len(passages)
        ids_each: List[List[int]] = [None] * len(passages)

        prefix_token_ids = self._get_prefix_tokens()
        prefix_len = len(prefix_token_ids)

        for s in range(0, len(passages), batch_size):
            batch_texts = passages[s:s + batch_size]

            # Ensure tokenizer has pad_token for batch processing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
    def extract_from_passages_varlen(
        self,
        passages: List[str],
    ) -> Tuple[List[List[Tuple[torch.Tensor, torch.Tensor]]], List[int], List[List[int]]]:
        """
        Extract KV cache from passages using flash_attn_varlen (no padding).

        This method packs all sequences together and uses flash_attn_varlen_func
        to avoid padding overhead.

        Args:
            passages: List of text passages

        Returns:
            kv_chunks: List of per-layer (K, V) tuples for each passage
            seq_lens: Length of each passage's KV cache
            ids_each: Token IDs for each passage
        """
        device = self.device
        dtype = self.model.dtype

        prefix_token_ids = self._get_prefix_tokens()
        prefix_len = len(prefix_token_ids)

        # Tokenize all passages without padding
        all_input_ids = []
        seq_lens = []
        ids_each = []

        for i, passage in enumerate(passages):
            tokens = self.tokenizer(passage, add_special_tokens=False, return_tensors="pt")
            passage_ids = tokens["input_ids"][0].tolist()

            if i == 0:
                # First passage: include prefix
                full_ids = prefix_token_ids + passage_ids
            else:
                # Other passages: include prefix for attention but track separately
                full_ids = prefix_token_ids + passage_ids

            all_input_ids.extend(full_ids)
            seq_lens.append(len(full_ids) if i == 0 else len(passage_ids))
            ids_each.append(full_ids if i == 0 else passage_ids)

        # Create packed input tensor
        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device).unsqueeze(0)
        total_tokens = len(all_input_ids)

        # Create cu_seqlens for varlen attention
        # Each passage is processed independently
        passage_lens_with_prefix = []
        for i, passage in enumerate(passages):
            tokens = self.tokenizer(passage, add_special_tokens=False, return_tensors="pt")
            plen = len(tokens["input_ids"][0]) + prefix_len
            passage_lens_with_prefix.append(plen)

        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(passage_lens_with_prefix), 0)),
            dtype=torch.int32, device=device
        )
        max_seqlen = max(passage_lens_with_prefix)

        # Get model components
        embed_layer = self.model.get_input_embeddings()
        layers = self.decoder.layers

        # Compute embeddings
        hidden_states = embed_layer(input_ids).to(dtype)  # [1, total_tokens, hidden]
        hidden_states = hidden_states.squeeze(0)  # [total_tokens, hidden]

        # Compute RoPE for all positions
        position_ids = torch.arange(total_tokens, device=device).unsqueeze(0)
        cos, sin = self.decoder.rotary_emb(hidden_states.unsqueeze(0), position_ids)
        cos = cos.squeeze(0)  # [total_tokens, head_dim]
        sin = sin.squeeze(0)

        # But we need per-passage position IDs (reset for each passage)
        per_passage_positions = []
        for plen in passage_lens_with_prefix:
            per_passage_positions.extend(list(range(plen)))
        position_ids_packed = torch.tensor(per_passage_positions, dtype=torch.long, device=device)

        # Recompute RoPE with per-passage positions
        cos, sin = self.decoder.rotary_emb(hidden_states.unsqueeze(0), position_ids_packed.unsqueeze(0))
        cos = cos.squeeze(0)  # [total_tokens, head_dim]
        sin = sin.squeeze(0)

        # Storage for KV cache per layer
        kv_per_layer = []

        # Timing accumulators
        import time
        time_proj = 0
        time_rope = 0
        time_attn = 0
        time_mlp = 0
        time_clone = 0
        time_gqa = 0

        is_cuda = str(self.device).startswith("cuda")

        # Forward through each layer
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn

            # Layer norm + Q/K/V projections
            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t0 = time.perf_counter()

            normed = layer.input_layernorm(hidden_states)

            # Compute Q, K, V
            q = attn.q_proj(normed)  # [total_tokens, num_heads * head_dim]
            k = attn.k_proj(normed)  # [total_tokens, num_kv_heads * head_dim]
            v = attn.v_proj(normed)  # [total_tokens, num_kv_heads * head_dim]

            # Reshape for attention: [total_tokens, num_heads, head_dim]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.kv_head_dim)
            v = v.view(total_tokens, self.num_kv_heads, self.kv_head_dim)

            # Apply Q/K norms if present
            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm") and attn.k_norm is not None:
                k = attn.k_norm(k)

            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t1 = time.perf_counter()
            time_proj += (_t1 - _t0)

            # Apply RoPE
            q = self._apply_rope_packed(q, cos, sin)
            k = self._apply_rope_packed(k, cos, sin)

            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t2 = time.perf_counter()
            time_rope += (_t2 - _t1)

            # Store K, V before attention (this is what we extract)
            kv_per_layer.append((k.clone(), v.clone()))

            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t3 = time.perf_counter()
            time_clone += (_t3 - _t2)

            # Flash attention with varlen (no padding!)
            # flash_attn v2 supports GQA natively - no need to expand k/v
            _t3b = time.perf_counter()
            time_gqa += (_t3b - _t3)

            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=True
            )  # [total_tokens, num_heads, head_dim]

            # Reshape and project output
            attn_output = attn_output.view(total_tokens, -1)  # [total_tokens, hidden]
            attn_output = attn.o_proj(attn_output).to(dtype)

            # Residual connection
            hidden_states = hidden_states + attn_output

            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t4 = time.perf_counter()
            time_attn += (_t4 - _t3b)

            # MLP
            residual = hidden_states
            mlp_input = layer.post_attention_layernorm(hidden_states)
            mlp_output = layer.mlp(mlp_input).to(dtype)
            hidden_states = residual + mlp_output

            if is_cuda:
                torch.cuda.synchronize(self.device)
            _t5 = time.perf_counter()
            time_mlp += (_t5 - _t4)

        # Print layer timing breakdown
        total_layer_time = time_proj + time_rope + time_clone + time_gqa + time_attn + time_mlp
        print(f"      [VARLEN LAYERS] Proj: {time_proj*1000:.1f}ms | RoPE: {time_rope*1000:.1f}ms | Clone: {time_clone*1000:.1f}ms | GQA: {time_gqa*1000:.1f}ms | Attn: {time_attn*1000:.1f}ms | MLP: {time_mlp*1000:.1f}ms | Total: {total_layer_time*1000:.1f}ms")

        # Split KV cache by passage
        kv_chunks = [None] * len(passages)
        offset = 0

        for i, plen in enumerate(passage_lens_with_prefix):
            per_layer = []
            for layer_idx in range(self.num_layers):
                k_all, v_all = kv_per_layer[layer_idx]

                if i == 0:
                    # First passage: keep all including prefix
                    k_chunk = k_all[offset:offset + plen].unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, plen, head_dim]
                    v_chunk = v_all[offset:offset + plen].unsqueeze(0).transpose(1, 2)
                    seq_lens[i] = plen
                else:
                    # Other passages: skip prefix
                    k_chunk = k_all[offset + prefix_len:offset + plen].unsqueeze(0).transpose(1, 2)
                    v_chunk = v_all[offset + prefix_len:offset + plen].unsqueeze(0).transpose(1, 2)
                    seq_lens[i] = plen - prefix_len

                per_layer.append((k_chunk, v_chunk))

            kv_chunks[i] = per_layer
            offset += plen

        return kv_chunks, seq_lens, ids_each

    def _apply_rope_packed(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to packed tensor [total_tokens, num_heads, head_dim]."""
        # x: [total_tokens, num_heads, head_dim]
        # cos, sin: [total_tokens, head_dim]
        cos = cos.unsqueeze(1)  # [total_tokens, 1, head_dim]
        sin = sin.unsqueeze(1)
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    @torch.no_grad()
    def extract_without_RoPE_correction(
        self,
        context: str,
        default_split: bool = True,
        chunk_size: int = 512,
        batch_size: int = 1,
        use_varlen: bool = False,
    ) -> KVCacheData:
        """
        Extract KV cache from full context without RoPE correction.

        Args:
            context: Full context string
            default_split: If True, use marker-based split; if False, use fixed-length split
            chunk_size: Token length for each chunk when default_split=False
            batch_size: Batch size for processing
            use_varlen: If True, use flash_attn_varlen for extraction (no padding)

        Returns:
            KVCacheData containing the extracted cache (as DynamicCache)
        """
        # Split context
        if default_split:
            passages = self._default_split(context, chunk_size)
        else:
            passages = self._fixed_length_split(context, chunk_size)

        # Extract KV chunks and per-passage token ids (ids_each)
        if use_varlen:
            kv_chunks, seq_lens, ids_each = self.extract_from_passages_varlen(passages)
        else:
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
    @torch.no_grad()
    def extract_with_rope_correction(
        self,
        context: str,
        default_split: bool = True,
        chunk_size: int = 512,
        batch_size: int = 1,
        use_varlen: bool = False,
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
            use_varlen: If True, use flash_attn_varlen for extraction (no padding)

        Returns:
            KVCacheData with corrected RoPE positions (as DynamicCache)
        """
        import time
        from itertools import chain

        device = self.device
        device_str = str(device)

        # Phase 1: Split context
        t0 = time.perf_counter()
        if default_split:
            passages = self._default_split(context, chunk_size)
        else:
            passages = self._fixed_length_split(context, chunk_size)
        t1 = time.perf_counter()
        split_ms = (t1 - t0) * 1000

        # Phase 2: Extract KV chunks (forward pass)
        if use_varlen:
            kv_chunks, seq_lens, ids_each = self.extract_from_passages_varlen(passages)
        else:
            kv_chunks, seq_lens, ids_each = self.extract_from_passages(passages, batch_size=batch_size)

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        forward_ms = (t2 - t1) * 1000

        # Calculate offsets
        offsets = []
        cur = 0
        for L_i in seq_lens:
            offsets.append(cur)
            cur += int(L_i)
        total_len = cur

        # Phase 3: Correct RoPE positions
        K_final, V_final = self._correct_rope_positions(kv_chunks, seq_lens, offsets, total_len)

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        rope_ms = (t3 - t2) * 1000

        # Phase 4: Create DynamicCache
        dyn = DynamicCache()
        dyn.key_cache = [K.contiguous() for K in K_final]
        dyn.value_cache = [V.contiguous() for V in V_final]

        # Create combined input_ids
        ids_cat = list(chain.from_iterable(ids_each))
        input_ids = torch.tensor([ids_cat], dtype=torch.long, device=device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        cache_ms = (t4 - t3) * 1000

        total_ms = (t4 - t0) * 1000
        print(f"    [EXTRACT] Split: {split_ms:.1f}ms | Forward: {forward_ms:.1f}ms | RoPE: {rope_ms:.1f}ms | Cache: {cache_ms:.1f}ms | Total: {total_ms:.1f}ms (tokens: {total_len})")

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
        Correct RoPE positions for concatenated KV chunks.
        
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
        
        # Get RoPE embeddings for Qwen
        position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            cos_full, sin_full = self.model.model.rotary_emb(
                x=torch.empty(1, 1, 1, 1, device=device, dtype=self.model.dtype),
                position_ids=position_ids
            )
        else:
            raise ValueError("Cannot find rotary_emb for Qwen model")
        
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
                
                K_removed = self._remove_rope(K_chunk, cos_full, sin_full, position_ids_old)
                
                # Apply new RoPE
                position_ids_new = torch.arange(off, off + T_i, device=device)
                K_rebased = self._apply_rope(K_removed, cos_full, sin_full, position_ids_new)
                K_final[layer_idx][:, :, off:off+T_i, :].copy_(K_rebased)
                
                # Copy V without modification
                V_final[layer_idx][:, :, off:off+T_i, :].copy_(V_chunk)
        
        return K_final, V_final

