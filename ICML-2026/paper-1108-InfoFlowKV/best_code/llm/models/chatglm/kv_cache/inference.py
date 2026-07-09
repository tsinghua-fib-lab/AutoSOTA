"""Inference with KV cache."""

import time
import torch
from typing import Dict, Optional, Any
from transformers.cache_utils import DynamicCache

from .base import KVCacheData


# GLM stop tokens: <|endoftext|>, <|user|>, <|observation|>
GLM_STOP_TOKEN_IDS = {151329, 151336, 151338}


class KVCacheInference:
    """
    Run generation with a pre-filled KV cache.

    Supports TTFT (Time To First Token) measurement and comparison
    with baseline inference.
    """

    def __init__(self, model, tokenizer, model_type: str = "glm"):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            model_type: Model type ("qwen", "glm", "llama")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model_type = model_type

    def _tokenize_query_for_cache(self, query_text: str) -> torch.Tensor:
        """
        Tokenize query for appending to cached context.
        For GLM, adds <|assistant|> suffix since context already has [gMASK]<sop><|user|> prefix.

        Args:
            query_text: The query text to tokenize

        Returns:
            Tensor of token IDs [1, seq_len]
        """
        query_tokens = self.tokenizer.encode(query_text, add_special_tokens=False)
        assistant_token = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        query_tokens = query_tokens + [assistant_token]
        return torch.tensor([query_tokens], dtype=torch.long)

    @torch.no_grad()
    def generate(
        self,
        kv_data: KVCacheData,
        query_input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        start_time: Optional[float] = None,
    ) -> Dict:
        """
        Generate with pre-filled KV cache.

        Args:
            kv_data: KVCacheData containing pre-filled cache and metadata
            query_input_ids: Query input_ids [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            start_time: Optional start time from perf_counter(). If provided,
                       TTFT and total_time will be calculated from this time.
                       If None, will start timing internally.

        Returns:
            Dict with:
                - text: Generated text
                - ttft_ms: Time to first token in milliseconds
                - total_time_ms: Total generation time
                - tokens_generated: Number of tokens generated
                - tokens_per_second: Generation speed
        """
        device = self.device
        kv_cache = kv_data.past_key_values
        query_input_ids = query_input_ids.to(device)

        # Start timing if not provided
        if start_time is None:
            device_str = str(device)
            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

        # Use the query_input_ids (query prompt) directly with the KV cache (context)
        # For ChatGLM, we need position_ids for the query
        context_len = kv_data.total_len
        cache_len = kv_cache.key_cache[0].size(2) if hasattr(kv_cache, 'key_cache') else context_len
        query_len = query_input_ids.size(1)
        position_ids = torch.arange(cache_len, cache_len + query_len, device=device).unsqueeze(0)
        
        outputs = self.model(
            input_ids=query_input_ids,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        first_token = self._sample_token(logits, do_sample, temperature, top_p)
        past_kv = outputs.past_key_values

        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        ttft_time = time.perf_counter() - start_time

        # Continue generation
        generated_ids = [first_token]
        current_token = first_token
        next_pos = cache_len + query_len  # Position after initial query

        for _ in range(max_new_tokens - 1):
            # Check GLM stop tokens
            if current_token.item() in GLM_STOP_TOKEN_IDS:
                break

            next_pos_tensor = torch.tensor([[next_pos]], device=device, dtype=torch.long)
            outputs = self.model(
                input_ids=current_token,
                past_key_values=past_kv,
                position_ids=next_pos_tensor,
                use_cache=True,
                return_dict=True,
            )
            next_pos += 1

            logits = outputs.logits
            current_token = self._sample_token(logits[:, -1, :], do_sample, temperature, top_p)

            generated_ids.append(current_token)
            past_kv = outputs.past_key_values

        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        total_time = time.perf_counter() - start_time

        # Decode
        all_gen_ids = torch.cat(generated_ids, dim=1)
        generated_text = self.tokenizer.decode(all_gen_ids[0], skip_special_tokens=True)
        num_tokens = len(generated_ids)
        gen_time = total_time - ttft_time

        return {
            "text": generated_text,
            "ttft_ms": ttft_time * 1000,
            "total_time_ms": total_time * 1000,
            "tokens_generated": num_tokens,
            "tokens_per_second": num_tokens / gen_time if gen_time > 0 else 0,
        }

    def _sample_token(
        self,
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_baseline(
        self,
        inputs: torch.Tensor,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Dict:
        """
        Generate without KV cache (baseline for comparison).
        Uses improved implementation with proper TTFT measurement and model-specific handling.

        Args:
            inputs: Input token IDs [batch, seq_len] or dict with "input_ids"
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (currently uses greedy)
            temperature: Sampling temperature (not used in greedy)
            top_p: Top-p sampling parameter (not used in greedy)

        Returns:
            Dict with same format as generate()
        """
        device = self.device
        
        # Handle both tensor and dict inputs
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(device)
        else:
            input_ids = inputs.to(device)
        
        # Generate with TTFT measurement
        with torch.no_grad():
            device_str = str(device)
            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            
            # First token generation (prefill)
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # [1, V]
            past_key_values = outputs.past_key_values
            
            # Greedy decoding (argmax)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            ttft = (time.perf_counter() - start_time) * 1000.0
            
            generated = next_token  # [1, 1]
            
            # ChatGLM EOS tokens
            eos_token_id = [151329, 151336, 151338]
            
            # Generate remaining tokens
            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
                generated = torch.cat([generated, next_token], dim=-1)  # [1, L_gen+1]
                
                # Check EOS
                if eos_token_id and next_token.item() in eos_token_id:
                    break
            
            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            total_time = (time.perf_counter() - start_time) * 1000.0
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Decode full text (input + generated)
        all_ids = torch.cat([input_ids, generated], dim=1)
        full_text = self.tokenizer.decode(all_ids[0], skip_special_tokens=True)
        
        num_tokens = generated.shape[1]
        gen_time = (total_time - ttft) / 1000.0  # Convert to seconds
        
        return {
            "text": generated_text.strip(),
            "full_text": full_text,
            "ttft_ms": ttft,
            "total_time_ms": total_time,
            "tokens_generated": num_tokens,
            "tokens_per_second": num_tokens / gen_time if gen_time > 0 else 0,
            "context_tokens": input_ids.shape[1],
        }

    def _sample_token(
        self,
        logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample a token from logits.

        Args:
            logits: Logits tensor [1, vocab_size]
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Sampled token tensor [1, 1]
        """
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


