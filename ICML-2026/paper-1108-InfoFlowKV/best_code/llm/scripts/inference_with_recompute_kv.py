#!/usr/bin/env python3
"""
Unified KV cache recomputation with multiple strategies for both Qwen and ChatGLM.

Supports:
- baseline: Just baseline inference
- no_recompute: Use extracted cache without cross attention between chunks directly without recomputation
- 1_layer_guided: Standard recomputation with importance scoring (norm, attn, entropy, etc.)
- cacheblend: CacheBlend strategy (Layer 0: full, Layer 1: full + select top 15%, Layer 2+: selective)
- 2_layer_guided: Extract without RoPE correction, then reorder_and_rebase before recompute

Usage:
    python scripts/inference_with_recompute_kv.py configs/musique_eval.yaml
"""

import sys
import os
import yaml
import time
import json
import gc
import numpy as np
from tqdm import tqdm
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from benchmarks import get_dataset
from models.chatglm.kv_cache import RecomputeConfig

_CUSTOM_QUERY_TEMPLATE = None



def get_model_components(model_name, device, model, tokenizer, num_recompute_chunks=2, recompute_attention_mode="flashinfer"):
    """Get model-specific components based on model type."""
    model_name_lower = model_name.lower()
    
    if "glm" in model_name_lower or "chatglm" in model_name_lower:
        from models.chatglm.kv_cache import (
            KVCacheExtractor,
            ImportanceScorer,
            KVCacheRecomputer,
            KVCacheInference,
        )
        model_type = "glm"
        
        # Get GLM-specific config parameters
        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "multi_query_group_num",
                               getattr(config, "num_key_value_heads", num_heads))
        head_dim = getattr(config, "kv_channels",
                          config.hidden_size // num_heads)
    elif "llama" in model_name_lower:
        from models.llama.kv_cache import (
            KVCacheExtractor,
            ImportanceScorer,
            KVCacheRecomputer,
            KVCacheInference,
        )
        model_type = "llama"
        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = config.hidden_size // num_heads
    else:
        from models.qwen.kv_cache import (
            KVCacheExtractor,
            ImportanceScorer,
            KVCacheRecomputer,
            KVCacheInference,
        )
        model_type = "qwen"
        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = config.hidden_size // num_heads
    
    # Pass optimization params only to models that support it (Qwen)
    if model_type == "qwen":
        recomputer = KVCacheRecomputer(
            model, tokenizer, model_type,
            num_recompute_chunks=num_recompute_chunks,
            recompute_attention_mode=recompute_attention_mode,
        )
    else:
        recomputer = KVCacheRecomputer(model, tokenizer, model_type)

    return {
        "extractor": KVCacheExtractor(model, tokenizer, model_type),
        "recomputer": recomputer,
        "inference": KVCacheInference(model, tokenizer, model_type),
        "model_type": model_type,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim
    }


def get_chat_prefix(model_type: str) -> str:
    """Get chat template prefix to prepend to context."""
    if model_type == "qwen":
        return "<|im_start|>user\n"
    return ""


def get_chat_suffix(model_type: str) -> str:
    """Get chat template suffix to append after query.
    Includes <think>\\n\\n</think>\\n to skip Qwen3's thinking phase."""
    if model_type == "qwen":
        return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return ""


def build_query_prompt(query: str, model_type: str) -> str:
    """
    Build query prompt based on model type.

    From small_model_guided:
    "Answer the question based on the given passages. Only give me the answer
    and do not output any other words. The answer should be within 5 words."
    """
    if _CUSTOM_QUERY_TEMPLATE:
        return _CUSTOM_QUERY_TEMPLATE.format(query=query)
    # All models use the same query prompt format
    return f"Answer the question based on the given passages. Only give me the answer and do not output any other words. The answer should be within 5 words.\nQuestion: {query}\nAnswer:"


def build_full_prompt(context: str, query: str, model_type: str, tokenizer) -> str:
    """Build full prompt with context and query, wrapped in chat template."""
    query_prompt = build_query_prompt(query, model_type)
    content = f"{context}\n{query_prompt}"

    if model_type == "llama":
        prefix_tokens = [128000, 128006, 882, 128007, 271]
        prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
        return prefix_text + content
    elif model_type == "qwen":
        return get_chat_prefix(model_type) + content + get_chat_suffix(model_type)
    elif model_type == "glm":
        return content  # GLM handles its own prefix tokens
    return content




def make_baseline_fn(model, tokenizer, inference, model_type, max_new_tokens):
    """
    Baseline: no cache, full forward pass.
    For GLM: uses model.generate() directly (matches glm_recompute behavior).
    For other models: delegates to inference.generate_baseline().
    """
    def fn(sample):
        device = next(model.parameters()).device
        context = sample.get('context', '')
        query = sample.get('input', '')

        if model_type == "glm":
            # GLM: Must have prefix tokens [gMASK]<sop><|user|>\n to match extraction structure
            # But NO <|assistant|> at end (apply_chat_template adds it, which is wrong)
            prefix_tokens = tokenizer.get_prefix_tokens()  # [gMASK], <sop>
            user_token = tokenizer.convert_tokens_to_ids("<|user|>")
            newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]

            # Build query prompt
            query_prompt = build_query_prompt(query, model_type)
            full_text = f"{context}\n{query_prompt}"
            content_tokens = tokenizer.encode(full_text, add_special_tokens=False)

            # Combine: prefix + content (no <|assistant|> at end)
            full_tokens = prefix_tokens + [user_token, newline_token] + content_tokens
            enc = {"input_ids": torch.tensor([full_tokens]), "attention_mask": torch.ones(1, len(full_tokens), dtype=torch.long)}
            inputs = {k: v.to(device) for k, v in enc.items()}

            device_str = str(device)
            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            if device_str.startswith("cuda"):
                torch.cuda.synchronize(device)
            total_time = (time.perf_counter() - start_time) * 1000.0

            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            tokens_generated = outputs.shape[1] - input_len

            return {
                "prediction": response.strip(),
                "ttft_ms": total_time,  # For baseline, TTFT ≈ total time (no separate measurement)
                "total_time_ms": total_time,
                "tokens_generated": tokens_generated,
                "strategy": "baseline",
            }

        # Other model types: use tokenize + generate_baseline()
        # Official LongBench v2 truncation: first half + last half (preserves query at end)
        full_prompt = build_full_prompt(context, query, model_type, tokenizer)
        if model_type == "qwen":
            input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
            max_len = 131072 - 32  # reserve for generation
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]
            enc = {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long)}
        elif model_type == "llama":
            enc = tokenizer(
                full_prompt,
                return_tensors="pt",
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Use inference.generate_baseline() for consistent implementation
        result = inference.generate_baseline(
            inputs=enc["input_ids"],
            max_new_tokens=max_new_tokens,
        )

        print(f"  [BASELINE] Input: {result['input_tokens']} tokens | Prefill: {result['prefill_ms']:.1f}ms | Decode: {result['decode_ms']:.1f}ms | TTFT: {result['ttft_ms']:.1f}ms")

        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "prefill_ms": result["prefill_ms"],
            "decode_ms": result["decode_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "input_tokens": result["input_tokens"],
            "strategy": "baseline",
        }
    return fn


def make_no_recompute_fn(model, tokenizer, extractor, inference, model_type, max_new_tokens, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False):
    """No recompute: use extracted cache directly."""
    def fn(sample):
        device = next(model.parameters()).device

        # Ensure CUDA is synchronized before extraction
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)

        # Phase 1: Extraction
        t0 = time.perf_counter()
        start_time = t0

        # Extract cache with RoPE correction
        context = get_chat_prefix(model_type) + sample.get('context', '')
        kv_data = extractor.extract_with_rope_correction(context, default_split=default_split, chunk_size=chunk_size, batch_size=batch_size, use_varlen=use_varlen)

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        extraction_ms = (t1 - t0) * 1000

        # Build query prompt
        query = sample.get('input', '')
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)

        # Phase 2: Inference
        result = inference.generate(
            kv_data,
            query_input_ids,
            max_new_tokens=max_new_tokens,
            start_time=start_time,
        )

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        inference_ms = (t2 - t1) * 1000

        print(f"  [NO_RECOMPUTE] Extract: {extraction_ms:.1f}ms | Inference: {inference_ms:.1f}ms | TTFT: {result['ttft_ms']:.1f}ms")

        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "strategy": "no_recompute",
            "extraction_ms": extraction_ms,
        }
    return fn


def make_1_layer_recompute_fn(
    model, tokenizer, extractor, recomputer, inference, model_type, components,
    config: RecomputeConfig, max_new_tokens, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False
):
    """Recomputation with 1 layer importance scoring."""
    # Import ImportanceScorer based on model type
    if model_type == "glm":
        from models.chatglm.kv_cache import ImportanceScorer
    elif model_type == "llama":
        from models.llama.kv_cache import ImportanceScorer
    else:
        from models.qwen.kv_cache import ImportanceScorer
    
    # Get model config parameters from components
    num_heads = components.get("num_heads")
    num_kv_heads = components.get("num_kv_heads")
    
    # Create scorer with method, layer_indices, and model-specific params
    scorer = ImportanceScorer(
        model, 
        method=config.method, 
        layer_indices=config.layer_indices,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads
    )
    
    def fn(sample):
        device = next(model.parameters()).device

        # Ensure CUDA is synchronized before extraction
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)

        # Phase 1: Extraction
        t0 = time.perf_counter()
        start_time = t0

        # Extract context KV cache
        context = get_chat_prefix(model_type) + sample.get('context', '')
        kv_data = extractor.extract_with_rope_correction(context, default_split=default_split, chunk_size=chunk_size, batch_size=batch_size, use_varlen=use_varlen)
        first_chunk_len = kv_data.chunk_lens[0]
        # For llama, only exclude actual prefix tokens (5 tokens) not entire first chunk
        actual_exclude = 5 if model_type == "llama" else first_chunk_len

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        extraction_ms = (t1 - t0) * 1000

        # Build query prompt
        query = sample.get('input', '')
        query_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)

        # Phase 2: Scoring
        t2 = time.perf_counter()
        scores = scorer.compute(
            kv_data=kv_data,
            query_input_ids=query_input_ids,  # Use inference-consistent prompt for scoring
        )

        # Select positions based on config
        if config.recompute_k is not None:
            recompute_indices = scorer.select_positions(
                scores,
                k=config.recompute_k,
                exclude_first_tokens=actual_exclude,
                descending=config.sort_descending,
            )
        else:
            recompute_indices = scorer.select_positions(
                scores,
                ratio=config.recompute_ratio,
                exclude_first_tokens=actual_exclude,
                descending=config.sort_descending,
            )

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        scoring_ms = (t3 - t2) * 1000

        # Phase 3: Recompute
        updated_kv_data = recomputer.recompute(kv_data, recompute_indices, descending=config.sort_descending)

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        recompute_ms = (t4 - t3) * 1000

        # Phase 4: Inference (first token generation)
        result = inference.generate(
            updated_kv_data,
            query_input_ids,
            max_new_tokens=max_new_tokens,
            start_time=start_time,
        )

        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        t5 = time.perf_counter()
        inference_ms = (t5 - t4) * 1000

        # Print timing breakdown
        total_ms = (t5 - t0) * 1000
        ttft_ms = result['ttft_ms']
        print(f"  [GUIDED_RECOMPUTE] Extract: {extraction_ms:.1f}ms | Score: {scoring_ms:.1f}ms | Recompute: {recompute_ms:.1f}ms | Inference: {inference_ms:.1f}ms | TTFT: {ttft_ms:.1f}ms")

        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "recompute_positions": len(recompute_indices) if recompute_indices is not None and len(recompute_indices) > 0 else 0,
            "strategy": f"standard_{config.method}",
            "extraction_ms": extraction_ms,
            "scoring_ms": scoring_ms,
            "recompute_ms": recompute_ms,
            "inference_ms": inference_ms,
        }
    return fn


def make_cacheblend_fn(
    model, tokenizer, extractor, recomputer, inference, model_type,
    config: RecomputeConfig, max_new_tokens, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False
):
    """CacheBlend: special layer-wise strategy."""
    def fn(sample):
        device = next(model.parameters()).device

        # Ensure CUDA is synchronized before extraction
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        # Start timing from extraction
        start_time = time.perf_counter()

        # Extract cache with RoPE correction
        context = get_chat_prefix(model_type) + sample.get('context', '')
        kv_data = extractor.extract_with_rope_correction(context, default_split=default_split, chunk_size=chunk_size, batch_size=batch_size, use_varlen=use_varlen)


        # CacheBlend recomputation - returns KVCacheData
        updated_kv_data = recomputer.recompute_cacheblend(
            kv_data,
            recompute_ratio=config.recompute_ratio
        )
        
        # Build query prompt
        query = sample.get('input', '')
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)
        
        # Generate with cacheblend cache, passing start_time
        result = inference.generate(
            updated_kv_data,
            query_input_ids,
            max_new_tokens=max_new_tokens,
            start_time=start_time,
        )
        
        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "context_tokens": updated_kv_data.total_len,
            "strategy": "cacheblend",
        }
    return fn


def make_lego_fn(
    model, tokenizer, extractor, recomputer, inference, model_type,
    config: RecomputeConfig, max_new_tokens, strategy_name: str = "lego", batch_size=1, default_split=True, chunk_size=1024, use_varlen=False
):
    """Lego/Lego2: select first k tokens or top p% tokens from each passage."""
    def fn(sample):
        device = next(model.parameters()).device

        # Ensure CUDA is synchronized before extraction
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)
        # Start timing from extraction
        start_time = time.perf_counter()

        # Extract cache with RoPE correction - returns KVCacheData with seq_len as chunk_lens list
        context = sample.get('context', '')
        kv_data = extractor.extract_with_rope_correction(context, default_split=default_split, chunk_size=chunk_size, batch_size=batch_size, use_varlen=use_varlen)


        # Get chunk lengths
        chunk_lens = kv_data.chunk_lens
        
        # Build recompute indices: first k tokens (or top_p%) from each chunk
        important_positions = []
        offset = 0
        for L in chunk_lens:
            if config.recompute_k is not None:
                take = min(config.recompute_k, L)
            else:
                take = max(1, int(L * config.recompute_ratio))
            important_positions.extend(range(offset, offset + take))
            offset += L
        
        # Recompute cache at important positions
        updated_kv_data = recomputer.recompute(kv_data, important_positions)
        
        # Build query prompt
        query = sample.get('input', '')
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)
        
        # Generate with recomputed cache, passing start_time
        result = inference.generate(
            updated_kv_data,
            query_input_ids,
            max_new_tokens=max_new_tokens,
            start_time=start_time,
        )
        
        # Generate strategy name for output
        if config.recompute_k is not None:
            strategy_label = f"{strategy_name}_k{config.recompute_k}"
        else:
            strategy_label = f"{strategy_name}_{int(config.recompute_ratio*100)}pct"
        
        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "recompute_positions": len(important_positions),
            "strategy": strategy_label,
        }
    return fn


def make_double_guided_fn(
    model, tokenizer, extractor, recomputer, inference, model_type, components,
    config: RecomputeConfig, max_new_tokens, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False
):
    """Double guided: extract without RoPE correction, reorder, then recompute."""
    # Import ImportanceScorer based on model type
    if model_type == "glm":
        from models.chatglm.kv_cache import ImportanceScorer
    elif model_type == "llama":
        from models.llama.kv_cache import ImportanceScorer
    else:
        from models.qwen.kv_cache import ImportanceScorer
    
    # Get model config parameters from components
    num_heads = components.get("num_heads")
    num_kv_heads = components.get("num_kv_heads")
    
    # Create scorer with model-specific params
    scorer = ImportanceScorer(
        model, 
        method=config.method, 
        layer_indices=config.layer_indices,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads
    )
    
    def fn(sample):
        device = next(model.parameters()).device

        # Ensure CUDA is synchronized before extraction
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(device)

        # Start timing from extraction
        start_time = time.perf_counter()

        # Extract WITHOUT RoPE correction
        context = sample.get('context', '')
        kv_data = extractor.extract_without_RoPE_correction(context, default_split=default_split, chunk_size=chunk_size, batch_size=batch_size, use_varlen=use_varlen)


        # Build query prompt
        query = sample.get('input', '')
        query_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)
        
        # Score to find important positions (scorer internally clones kv_data to avoid modification)
        scores = scorer.compute(
            kv_data=kv_data,
            query_input_ids=query_input_ids,  # Use inference-consistent prompt for scoring
        )
        # Select positions based on config
        if config.recompute_k is not None:
            important_positions = scorer.select_positions(
                scores,
                k=config.recompute_k,
            )
        else:
            important_positions = scorer.select_positions(
                scores,
                ratio=config.recompute_ratio,
            )
        
        # Reorder and rebase KV cache
        reordered_kv_data = recomputer.reorder_and_rebase_kv(
            kv_data,
            important_positions,
            put_higher_ratio_to_tail=True
        )
        # Score to find important positions (scorer internally clones kv_data to avoid modification)
        scores = scorer.compute(
            kv_data=reordered_kv_data,
            query_input_ids=query_input_ids,  # Use inference-consistent prompt for scoring
        )

        if config.recompute_k is not None:
            important_positions = scorer.select_positions(
                scores,
                k=config.recompute_k,
            )
        else:
            important_positions = scorer.select_positions(
                scores,
                ratio=config.recompute_ratio,
            )
        
        # Recompute at important positions
        updated_kv_data = recomputer.recompute(
            reordered_kv_data,
            important_positions
        )
        
        # Build query prompt
        query = sample.get('input', '')
        query_prompt = build_query_prompt(query, model_type) + get_chat_suffix(model_type)
        query_inputs = tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)
        query_input_ids = query_inputs.input_ids.to(device)
        
        # Generate with reordered and recomputed cache, passing start_time
        result = inference.generate(
            updated_kv_data,
            query_input_ids,
            max_new_tokens=max_new_tokens,
            start_time=start_time,
        )
        
        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "context_tokens": kv_data.total_len,
            "recompute_positions": len(important_positions) if important_positions is not None else 0,
            "strategy": "double_guided",
        }
    return fn


def make_inference_fn(model, tokenizer, components, model_type, max_new_tokens,
                      strat_name, method, ratio, k, layer_indices, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False, sort_descending=False):
    """
    Create an inference function for a specific strategy.

    Returns a function that takes only a sample and returns prediction results.
    """
    def inference_fn(sample):
        # Select strategy function and call it directly
        if strat_name == "baseline":
            fn = make_baseline_fn(model, tokenizer, components["inference"],
                                    model_type, max_new_tokens)
        elif strat_name == "no_recompute":
            fn = make_no_recompute_fn(model, tokenizer, components["extractor"],
                                        components["inference"], model_type, max_new_tokens, batch_size, default_split, chunk_size, use_varlen)
        elif strat_name == "guided_recompute":
            cfg = RecomputeConfig(recompute_ratio=ratio, method=method, layer_indices=layer_indices, sort_descending=sort_descending)
            fn = make_1_layer_recompute_fn(
                model, tokenizer, components["extractor"],
                components["recomputer"], components["inference"],
                model_type, components, cfg, max_new_tokens, batch_size, default_split, chunk_size, use_varlen
            )
        elif strat_name == "double_guided":
            cfg = RecomputeConfig(recompute_ratio=ratio, method=method, layer_indices=layer_indices)
            fn = make_double_guided_fn(
                model, tokenizer, components["extractor"],
                components["recomputer"], components["inference"],
                model_type, components, cfg, max_new_tokens, batch_size, default_split, chunk_size, use_varlen
            )
        elif strat_name == "cacheblend":
            cfg = RecomputeConfig(recompute_ratio=ratio)
            fn = make_cacheblend_fn(
                model, tokenizer, components["extractor"],
                components["recomputer"], components["inference"],
                model_type, cfg, max_new_tokens, batch_size, default_split, chunk_size, use_varlen
            )
        elif strat_name == "lego":
            cfg = RecomputeConfig(recompute_k=k)
            fn = make_lego_fn(
                model, tokenizer, components["extractor"],
                components["recomputer"], components["inference"],
                model_type, cfg, max_new_tokens, strategy_name="lego", batch_size=batch_size, default_split=default_split, chunk_size=chunk_size, use_varlen=use_varlen
            )
        elif strat_name == "lego2":
            cfg = RecomputeConfig(recompute_ratio=ratio)
            fn = make_lego_fn(
                model, tokenizer, components["extractor"],
                components["recomputer"], components["inference"],
                model_type, cfg, max_new_tokens, strategy_name="lego2", batch_size=batch_size, default_split=default_split, chunk_size=chunk_size, use_varlen=use_varlen
            )
        else:
            raise ValueError(f"Unknown strategy: {strat_name}")
        
        # Run the strategy function (only pass sample)
        result = fn(sample)
        
        # Pass through all fields from strategy result (including component timings)
        output = {
            'prediction': result['prediction'],
            'ttft_ms': result.get('ttft_ms', 0.0),
            'recompute_positions': result.get('recompute_positions', None),
            'extraction_ms': result.get('extraction_ms', None),
            'scoring_ms': result.get('scoring_ms', None),
            'recompute_ms': result.get('recompute_ms', None),
            'inference_ms': result.get('inference_ms', None),
            'seq_len': result.get('seq_len', None),
        }
        return output
    
    return inference_fn


def prepare_strategy_functions(model, tokenizer, components, model_type, max_new_tokens,
                               strategies, top_p, lego_k, layer_indices, batch_size=1, default_split=True, chunk_size=1024, use_varlen=False, sort_descending=False):
    """
    Prepare all strategy inference functions and their metadata.

    Args:
        model: The language model
        tokenizer: The tokenizer
        components: Model-specific components dict
        model_type: Model type string
        max_new_tokens: Maximum tokens to generate
        strategies: List of strategy configs
        top_p: Default recompute ratio
        lego_k: Default lego k value
        layer_indices: Layer indices for importance scoring
        batch_size: Batch size for parallel extraction
        use_varlen: Use flash_attn_varlen for extraction (no padding)

    Returns:
        Tuple of (strategy_fns, strategy_metadata) dicts
    """
    strategy_fns = {}
    strategy_metadata = {}
    
    for strategy in strategies:
        strategy_name = strategy['name']
        method = strategy.get('method', None)
        ratio = strategy.get('ratio', top_p)
        k = strategy.get('k', lego_k)
        
        # Create strategy key
        strategy_key = f"{strategy_name}_{method}" if method else strategy_name
        
        # Create inference function
        infer_fn = make_inference_fn(
            model, tokenizer, components, model_type, max_new_tokens,
            strategy_name, method, ratio, k, layer_indices, batch_size, default_split, chunk_size, use_varlen, sort_descending
        )
        
        strategy_fns[strategy_key] = infer_fn
        strategy_metadata[strategy_key] = {
            'strategy': strategy_name,
            'method': method,
            'top_p': ratio,
            'lego_k': k,
        }
        print(f"Prepared strategy: {strategy_key}")
    
    return strategy_fns, strategy_metadata


def report_results(all_results: Dict, model_name: str, model_output_dir: str):
    """
    Print detailed performance comparison and metrics for all strategies.
    
    Args:
        all_results: Dict of {strategy_key: {'results': [...], 'summary': {...}}}
        model_name: Name of the model being evaluated
        model_output_dir: Directory where results are saved
    """
    # Print performance comparison table
    print(f"\n{'='*100}")
    print("DETAILED PERFORMANCE COMPARISON")
    print(f"{'='*100}")
    print(f"{'Strategy':<30s} {'Accuracy':<12s} {'Avg F1':<10s} {'TTFT (ms)':<15s} {'Total Time (ms)':<15s}")
    print(f"{'-'*100}")
    
    for strategy_key, output in all_results.items():
        summary = output['summary']
        strategy_label = strategy_key.replace('_', ' ').title()
        
        # Extract metrics
        accuracy = summary['accuracy']
        avg_f1 = summary['avg_f1']
        avg_ttft = summary.get('avg_ttft_ms', 0.0)
        
        # Try to get total_time from results
        results = output['results']
        total_times = [r.get('total_time_ms', 0) for r in results if 'total_time_ms' in r]
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0.0
        
        print(f"{strategy_label:<30s} {accuracy:>9.2f}%  {avg_f1:>9.4f}  {avg_ttft:>14.2f}  {avg_total_time:>14.2f}")
    
    print(f"{'='*100}")
    
    # Print detailed metrics
    print(f"\n{'='*100}")
    print("DETAILED METRICS BY STRATEGY")
    print(f"{'='*100}")
    
    for strategy_key, output in all_results.items():
        summary = output['summary']
        strategy_label = strategy_key.replace('_', ' ').title()
        
        print(f"\n{strategy_label}:")
        print(f"  Dataset: {summary['dataset']}")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Correct (F1>0.5): {summary['correct']}/{summary['total_samples']}")
        print(f"  Accuracy: {summary['accuracy']:.2f}%")
        print(f"  Average F1 Score: {summary['avg_f1']:.4f}")
        print(f"  Average TTFT: {summary.get('avg_ttft_ms', 0):.2f} ms")
        
        # Calculate total_time from results if not in summary
        if 'avg_total_time_ms' in summary:
            print(f"  Average Total Time: {summary['avg_total_time_ms']:.2f} ms")
        else:
            results = output['results']
            total_times = [r.get('total_time_ms', 0) for r in results if 'total_time_ms' in r]
            if total_times:
                avg_total = sum(total_times) / len(total_times)
                print(f"  Average Total Time: {avg_total:.2f} ms")
        
        if 'avg_tokens_generated' in summary:
            print(f"  Average Tokens Generated: {summary['avg_tokens_generated']:.2f}")
        
        if 'avg_recompute_positions' in summary:
            print(f"  Average Recompute Positions: {summary['avg_recompute_positions']:.2f}")
    
    print(f"\n{'='*100}")
    print(f"✅ Results for {model_name} saved to {model_output_dir}/")
    print(f"{'='*100}")




def main():
    # Clean up GPU memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get config file path from command line
    if len(sys.argv) < 2:
        print("Usage: python inference_with_recompute_kv.py <config_file.yaml>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load config
    print(f"Loading config from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract all parameters from config
    # Support both 'model' (single) and 'models' (list)
    if 'models' in config:
        model_paths = config['models']
    elif 'model' in config:
        model_paths = [config['model']]
    else:
        raise ValueError("Config must contain either 'model' or 'models'")
    
    dataset_name = config.get('dataset')
    if not dataset_name:
        raise ValueError("'dataset' must be specified in config file")
    
    num_samples = config.get('num_samples', None)
    warmup_samples = config.get('warmup_samples', 1)
    max_new_tokens = config.get('max_new_tokens', 32)
    output_dir = config.get('output_dir', 'results')
    
    strategies = config.get('strategies', [])
    top_p = config.get('top_p', 0.15)
    lego_k = config.get('lego_k', 5)
    batch_sizes_config = config.get('batch_size', 1)
    # Support both single value and list
    batch_sizes = batch_sizes_config if isinstance(batch_sizes_config, list) else [batch_sizes_config]
    
    # Handle default_split with case-insensitive string support
    default_split_raw = config.get('default_split', True)
    if isinstance(default_split_raw, str):
        default_split = default_split_raw.lower() in ('true', '1', 'yes')
    else:
        default_split = bool(default_split_raw)
    
    chunk_size = config.get('chunk_size', 1024)
    layer_indices = config.get('layer_indices', None)
    use_varlen = config.get('use_varlen', False)
    num_recompute_chunks = config.get('num_recompute_chunks', 2)
    recompute_attention_mode = config.get('recompute_attention_mode', 'flashinfer')

    global _CUSTOM_QUERY_TEMPLATE
    _CUSTOM_QUERY_TEMPLATE = config.get('query_template', None)

    # Device configuration
    device_str = config.get('device', 'auto')
    
    print(f"{'='*60}")
    print(f"KV Cache Recomputation Evaluation")
    print(f"{'='*60}")
    print(f"Models: {len(model_paths)}")
    for i, mp in enumerate(model_paths, 1):
        print(f"  {i}. {mp}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device_str}")
    print(f"Num Samples: {num_samples if num_samples else 'All'} ({warmup_samples} warmup)")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Top P: {top_p}")
    print(f"Lego K: {lego_k}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Context Split: {'Passage markers' if default_split else f'Fixed {chunk_size} tokens'}")
    print(f"Use Varlen Attention: {use_varlen}")
    print(f"Num Recompute Chunks: {num_recompute_chunks}")
    print(f"Recompute Attention Mode: {recompute_attention_mode}")
    print(f"Strategies: {len(strategies)}")
    print(f"{'='*60}")
    
    # Loop over each model
    for model_idx, model_path in enumerate(model_paths, 1):
        print(f"\n{'#'*60}")
        print(f"# MODEL {model_idx}/{len(model_paths)}: {model_path}")
        print(f"{'#'*60}")
        print(f"\nLoading model {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Configure device_map based on device setting
        if device_str == 'auto':
            device_map = 'auto'
        else:
            # Use specific device (e.g., cuda:5, cpu)
            device_map = {"": device_str}
        
        # Load dataset first (shared across all strategies)
        # We need to get device first, so load a dummy model or use config
        temp_device = torch.device(device_str if device_str != 'auto' else 'cuda:0')
        dataset = get_dataset(dataset_name, input_dir='inputs', device=temp_device)
        length_filter = config.get('length_filter', None)
        dataset.load(num_samples=num_samples, length_filter=length_filter)

        # Truncate contexts that exceed max_context_tokens
        max_context_tokens = config.get('max_context_tokens', None)
        if max_context_tokens and dataset.data:
            truncated = 0
            for sample in dataset.data:
                ctx = sample.get('context', '')
                token_ids = tokenizer.encode(ctx, add_special_tokens=False)
                if len(token_ids) > max_context_tokens:
                    sample['context'] = tokenizer.decode(token_ids[:max_context_tokens], skip_special_tokens=True)
                    truncated += 1
            if truncated:
                print(f"Truncated {truncated}/{len(dataset.data)} contexts to {max_context_tokens} tokens")

        all_batch_results = {}
        model_type = None  # Will be set when model is loaded

        # Loop over each batch_size
        for batch_idx, batch_size in enumerate(batch_sizes, 1):
            print(f"\n{'='*60}")
            print(f"Testing Batch Size: {batch_size} ({batch_idx}/{len(batch_sizes)})")
            print(f"{'='*60}")

            all_results = {}
            all_metadata = {}
            local_layer_indices = layer_indices  # Copy to avoid modifying across iterations

            # Load model once with FA2 for all strategies.
            # Scoring uses selective eager patching (only scoring layers compute
            # attention weights; other layers stay on flash attention).
            print(f"\n--- Loading model with FA2 for all strategies ---")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            model.eval()
            device = next(model.parameters()).device
            components = get_model_components(model_path, device, model, tokenizer, num_recompute_chunks, recompute_attention_mode)
            model_type = components["model_type"]

            # Set default layer_indices based on model type if not specified
            if local_layer_indices is None:
                if model_type in ["qwen", "llama"]:
                    local_layer_indices = [22, 23, 24, 25]
                    print(f"Using default layer_indices for {model_type}: {local_layer_indices}")

            sort_descending = config.get('sort_descending', False)
            strategy_fns, strategy_metadata = prepare_strategy_functions(
                model, tokenizer, components, model_type, max_new_tokens,
                strategies, top_p, lego_k, local_layer_indices, batch_size, default_split, chunk_size, use_varlen, sort_descending
            )

            # Run each strategy on all samples separately to avoid GPU memory fragmentation
            all_results = {}
            all_metadata = strategy_metadata
            for key, fn in strategy_fns.items():
                print(f"\n--- Running strategy: {key} on all samples ---")
                result = dataset.evaluate({key: fn}, num_samples=num_samples, warmup_samples=warmup_samples)
                all_results.update(result)
                gc.collect()
                torch.cuda.empty_cache()

            # Clean up model
            del model
            del components
            gc.collect()
            torch.cuda.empty_cache()

            # Extract model name for output directory
            model_name = model_path.split('/')[-1]
            batch_suffix = f"_bs{batch_size}" if len(batch_sizes) > 1 else ""
            model_output_dir = f"{output_dir}/{dataset_name}_{model_name}{batch_suffix}"

            # Add metadata and save results for each strategy
            for strategy_key, output in all_results.items():
                # Add strategy metadata to summary
                output['summary'].update(all_metadata[strategy_key])
                output['summary']['model'] = model_path
                output['summary']['model_type'] = model_type
                output['summary']['batch_size'] = batch_size

                # Save individual strategy results
                dataset.save_results(output, label=strategy_key, output_dir=model_output_dir)

            # Report results using dedicated function
            report_results(all_results, model_name, model_output_dir)

        # Clean up tokenizer
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n Cleaned up {model_name} from memory\n")
    
    print(f"\n{'#'*60}")
    print(f"# ALL MODELS COMPLETED")
    print(f"{'#'*60}")
    print(f"✅ All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
