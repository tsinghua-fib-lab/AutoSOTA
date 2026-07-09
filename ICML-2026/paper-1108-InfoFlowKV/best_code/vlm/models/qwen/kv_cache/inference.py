"""Inference with recomputed KV cache."""

import time
import torch
from typing import Dict
from transformers.cache_utils import DynamicCache

from .base import KVCacheData


class KVCacheInference:
    """
    Run generation with a pre-filled KV cache.

    Supports TTFT (Time To First Token) measurement and comparison
    with baseline inference.

    Usage:
        inference = KVCacheInference(model, processor)
        result = inference.generate(kv_data.past_key_values, query_text, kv_data.seq_len)
    """

    def __init__(self, model, processor):
        """
        Args:
            model: The Qwen3-VL model
            processor: The processor for tokenization
        """
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

    @torch.no_grad()
    def generate(
        self,
        kv_cache: DynamicCache,
        input_ids: torch.Tensor,
        context_len: int,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Dict:
        """
        Generate with pre-filled KV cache using model.generate().

        The context should already include the generation prompt (e.g., <|im_start|>assistant).
        This method continues generation from the cache.

        Args:
            kv_cache: Pre-filled and optionally recomputed KV cache
            input_ids: Original input_ids [batch, seq_len]
            context_len: Length of context (for position offset)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Dict with:
                - text: Generated text
                - ttft_ms: Time to first token in milliseconds
                - total_time_ms: Total generation time
                - tokens_generated: Number of tokens generated
                - tokens_per_second: Generation speed
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Start timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        # Crop cache by 1 and re-run last token to continue generation
        # This ensures the model generates new tokens after the prefilled context
        last_token = input_ids[:, -1:]
        cache_seq_len = context_len - 1

        for layer in kv_cache.layers:
            if hasattr(layer, 'keys') and layer.keys is not None and layer.keys.numel() > 0:
                layer.keys = layer.keys[..., :cache_seq_len, :]
            if hasattr(layer, 'values') and layer.values is not None and layer.values.numel() > 0:
                layer.values = layer.values[..., :cache_seq_len, :]

        # Create cache_position for the last token position
        cache_position = torch.arange(
            cache_seq_len, cache_seq_len + 1, device=device, dtype=torch.long
        )

        # Use model.generate() with past_key_values
        # This properly handles all the internal state including position_ids
        gen_kwargs = {
            "max_new_tokens": max_new_tokens + 1,  # +1 because we're starting from last_token
            "do_sample": do_sample,
            "use_cache": True,
            "past_key_values": kv_cache,
            "cache_position": cache_position,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # Generate starting from last token
        generated_ids = self.model.generate(
            input_ids=last_token,
            **gen_kwargs,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - start_time

        # Extract generated tokens (skip the input last_token)
        new_tokens = generated_ids[:, 1:]  # Skip the input token

        # Decode
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        full_ids = torch.cat([input_ids, new_tokens], dim=1)
        full_text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        num_tokens = new_tokens.shape[1]

        # Estimate TTFT (we can't measure it exactly with generate())
        ttft_estimate = total_time / (num_tokens + 1) if num_tokens > 0 else total_time

        return {
            "text": generated_text,
            "full_text": full_text,
            "ttft_ms": ttft_estimate * 1000,
            "total_time_ms": total_time * 1000,
            "tokens_generated": num_tokens,
            "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
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
        inputs: Dict[str, torch.Tensor],
        query_text: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> Dict:
        """
        Generate without KV cache (baseline for comparison).

        Args:
            inputs: Full processor output (context + query)
            query_text: Query text (for extracting response)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with same format as generate()
        """
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Start timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        # Use model.generate() for baseline
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - start_time

        # Extract generated tokens
        new_tokens = generated_ids[:, input_len:]

        # Decode
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        num_tokens = new_tokens.shape[1]

        # Estimate TTFT
        ttft_estimate = total_time / (num_tokens + 1) if num_tokens > 0 else total_time

        return {
            "text": generated_text,
            "full_text": full_text,
            "ttft_ms": ttft_estimate * 1000,
            "total_time_ms": total_time * 1000,
            "tokens_generated": num_tokens,
            "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
            "context_tokens": input_len,
        }
