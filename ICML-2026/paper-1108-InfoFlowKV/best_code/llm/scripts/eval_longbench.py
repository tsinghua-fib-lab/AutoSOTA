#!/usr/bin/env python
"""
LongBench evaluation for comparing guided recompute methods.

Uses the existing benchmarks/longbench.py evaluation framework for proper
prompt templates and F1 computation.

Evaluates on real long-context benchmarks:
- HotpotQA: Multi-hop QA
- 2WikiMultihopQA: Multi-hop QA
- MuSiQue: Multi-hop QA

Usage:
    # Single GPU baseline + non-SP guided recompute
    python scripts/eval_longbench.py --model ../models/Qwen3-14B/ --tasks hotpotqa

    # Multi-GPU with SP guided recompute
    torchrun --nproc_per_node=4 scripts/eval_longbench.py \
        --model ../models/Qwen3-14B/ \
        --tasks hotpotqa 2wikimqa \
        --methods baseline sp_guided_recompute
"""

import gc
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime

import torch
import torch.distributed as dist
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation framework (factory supports LongBench v1 + v2)
from benchmarks.base import get_dataset
from benchmarks.longbench import LongBenchDataset  # type hint (V2 inherits from this)

# Qwen3 chat template tokens (matching inference_with_recompute_kv.py)
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_CONTEXT_TOKENS = 131072


def build_truncated_prompt(sample, dataset, tokenizer, device):
    """Build prompt with context truncated to fit within MAX_CONTEXT_TOKENS.

    Truncates context (from the end) while always preserving the query/suffix.
    This ensures the question is never cut off for long-context samples.
    """
    context = sample.get('context', '')
    raw_input = sample.get('input', '')

    # Tokenize the non-context parts to know how much room context gets
    suffix_text = f"\n{raw_input}\nAnswer:" + CHAT_SUFFIX
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
    prefix_ids = tokenizer(CHAT_PREFIX, add_special_tokens=False).input_ids
    reserved = len(prefix_ids) + len(suffix_ids) + 10  # small buffer

    # Truncate context to fit
    max_context = MAX_CONTEXT_TOKENS - reserved
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    if len(context_ids) > max_context:
        context_ids = context_ids[:max_context]
        context = tokenizer.decode(context_ids, skip_special_tokens=True)

    prompt = CHAT_PREFIX + context + f"\n{raw_input}\nAnswer:" + CHAT_SUFFIX
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return prompt, input_ids


def parse_args():
    parser = argparse.ArgumentParser(description="LongBench Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["hotpotqa"],
        choices=["hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", "longbenchv2"],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["baseline", "guided_recompute"],
        help="Methods to evaluate: baseline, single_gpu_prefill, ring_attention, guided_recompute, sp_guided_recompute, sp_cacheblend, sp_lego",
    )
    parser.add_argument("--recompute_ratio", type=float, default=0.15)
    parser.add_argument("--heads_k_stride", type=int, default=1,
                        help="KV heads per all-gather in ring attention (1=per-head, num_kv_heads=all-at-once)")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per task (0 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--input_dir", type=str, default="inputs", help="Directory containing LongBench JSONL files")
    parser.add_argument("--output", type=str, default="results/longbench_eval")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def create_inference_fn_baseline(model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int):
    """Create inference function for baseline (full forward pass)."""

    def infer(sample: Dict) -> Dict:
        _, input_ids = build_truncated_prompt(sample, dataset, tokenizer, device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][input_ids.shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {"prediction": answer}

    return infer


def create_inference_fn_single_gpu_prefill(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, eos_token_ids: set = None
):
    """Create inference function for single-GPU full prefill (baseline for speedup).

    This is the simplest baseline: full forward pass on a single GPU.
    Used to measure speedup of distributed methods.
    """
    import time
    from transformers.cache_utils import DynamicCache

    def infer(sample: Dict) -> Dict:
        _, input_ids = build_truncated_prompt(sample, dataset, tokenizer, device)
        seq_len = input_ids.shape[1]

        # Measure prefill time
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            # Full prefill forward pass
            outputs = model(
                input_ids,
                use_cache=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values

        torch.cuda.synchronize(device)
        prefill_time_ms = (time.perf_counter() - t0) * 1000

        # Generation
        generated_ids = []
        current_pos = torch.tensor([[seq_len]], device=device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token)

                if eos_token_ids and next_token.item() in eos_token_ids:
                    break

                outputs = model(
                    input_ids=next_token,
                    position_ids=current_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = outputs.past_key_values
                current_pos = current_pos + 1

        if generated_ids:
            generated = torch.cat(generated_ids, dim=1)
            answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        else:
            answer = ""

        del outputs, past_kv
        gc.collect()
        torch.cuda.empty_cache()
        return {"prediction": answer, "prefill_time_ms": prefill_time_ms, "seq_len": seq_len}

    return infer


def create_inference_fn_guided_recompute(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, recompute_ratio: float
):
    """Create inference function for single-GPU guided recompute.

    Uses the existing single-GPU pipeline (KVCacheExtractor → ImportanceScorer →
    KVCacheRecomputer → KVCacheInference) directly, avoiding reimplementation.
    """
    from models.qwen.kv_cache import KVCacheExtractor, ImportanceScorer, KVCacheRecomputer, KVCacheInference

    extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    scorer = ImportanceScorer(model, method="norm", layer_indices=[22, 23, 24, 25])
    recomputer = KVCacheRecomputer(model, tokenizer, model_type="qwen")
    inference = KVCacheInference(model, tokenizer, model_type="qwen")

    def infer(sample: Dict) -> Dict:
        context_part = sample.get('context', '')
        raw_query = sample.get('input', '')

        # Score with raw query only (matching inference_with_recompute_kv.py)
        scorer_query_ids = tokenizer(raw_query, return_tensors="pt").input_ids.to(device)

        # Generate with full prompt template + chat suffix to skip thinking
        query_prompt = (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words. "
            "The answer should be within 5 words.\n"
            f"Question: {raw_query}\nAnswer:"
        )
        gen_query_ids = tokenizer(query_prompt + CHAT_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Extract with chat prefix → Score → Recompute → Generate
        kv_data = extractor.extract_with_rope_correction(CHAT_PREFIX + context_part)
        first_chunk_len = kv_data.chunk_lens[0]
        scores = scorer.compute(kv_data, query_input_ids=scorer_query_ids)
        important_indices = scorer.select_positions(
            scores, ratio=recompute_ratio, exclude_first_tokens=first_chunk_len
        )
        updated_kv = recomputer.recompute(kv_data, important_indices)
        del kv_data, scores

        result = inference.generate(updated_kv, gen_query_ids, max_new_tokens=max_new_tokens)
        del updated_kv
        gc.collect()
        torch.cuda.empty_cache()

        return {"prediction": result["text"]}

    return infer


def create_inference_fn_ring_attention(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, config, group, eos_token_ids: set = None
):
    """Create inference function for ring attention (official library).

    This is the proper ring attention baseline:
    1. Prefill with ring attention (sequence distributed across GPUs)
    2. All-gather KV cache to rank 0
    3. Generate on rank 0 with full KV cache

    Ring attention setup (substitute_hf_flash_attn) must be done by the caller.
    """
    from ring_flash_attn import update_ring_flash_attn_params
    from ring_flash_attn.adapters.hf_adapter import use_ring_attn
    from transformers.cache_utils import DynamicCache

    world_size = config.world_size
    rank = config.rank

    def all_gather_kv_cache(local_kv, seq_len, world_size, rank, device, group):
        """All-gather KV cache from all GPUs to reconstruct full cache."""
        if isinstance(local_kv, DynamicCache):
            key_cache = local_kv.key_cache
            value_cache = local_kv.value_cache
        else:
            key_cache = [kv[0] for kv in local_kv]
            value_cache = [kv[1] for kv in local_kv]

        num_layers = len(key_cache)

        # Get local sequence lengths
        local_len = key_cache[0].shape[2]
        local_lens_tensor = torch.tensor([local_len], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_lens_tensor, group=group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)
        total_len = sum(all_lens)

        # Build full cache on all ranks (needed for generation)
        full_cache = DynamicCache()
        full_cache.key_cache = []
        full_cache.value_cache = []

        for layer_idx in range(num_layers):
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
            B, H, local_T, D = k.shape

            # Pad to max_len for all_gather (requires same tensor size)
            k_padded = torch.zeros(B, H, max_len, D, device=device, dtype=k.dtype)
            v_padded = torch.zeros(B, H, max_len, D, device=device, dtype=v.dtype)
            k_padded[:, :, :local_T, :] = k
            v_padded[:, :, :local_T, :] = v

            # All-gather with same-sized tensors
            k_gathered = [torch.zeros_like(k_padded) for _ in range(world_size)]
            v_gathered = [torch.zeros_like(v_padded) for _ in range(world_size)]

            dist.all_gather(k_gathered, k_padded.contiguous(), group=group)
            dist.all_gather(v_gathered, v_padded.contiguous(), group=group)

            # Concatenate and trim to actual lengths
            k_full = torch.cat([k_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
            v_full = torch.cat([v_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

            full_cache.key_cache.append(k_full)
            full_cache.value_cache.append(v_full)

        full_cache._seen_tokens = total_len
        return full_cache

    def infer(sample: Dict) -> Dict:
        import time

        # Enable ring attention for prefill
        use_ring_attn(True)

        _, input_ids = build_truncated_prompt(sample, dataset, tokenizer, device)
        seq_len = input_ids.shape[1]

        # Pad sequence length to be divisible by world_size (required by ring-flash-attention)
        if seq_len % world_size != 0:
            pad_len = world_size - (seq_len % world_size)
            input_ids = torch.cat([
                input_ids,
                torch.full((1, pad_len), tokenizer.pad_token_id, device=device, dtype=input_ids.dtype)
            ], dim=1)
            padded_seq_len = seq_len + pad_len
        else:
            padded_seq_len = seq_len

        # Setup position_ids and cu_seqlens
        position_ids = torch.arange(padded_seq_len, device=device).unsqueeze(0)
        cu_seqlens = torch.tensor([0, padded_seq_len], device=device, dtype=torch.int32)
        update_ring_flash_attn_params(cu_seqlens, group)

        # Chunk input across GPUs (using padded length)
        chunk_size = padded_seq_len // world_size  # Now evenly divisible
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size

        input_ids_chunk = input_ids[:, start_idx:end_idx]
        position_ids_chunk = position_ids[:, start_idx:end_idx]

        # --- TTFT timing: prefill + all-gather ---
        torch.cuda.synchronize(device)
        t_ttft_start = time.perf_counter()

        # Forward pass with ring attention (prefill)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_chunk,
                position_ids=position_ids_chunk,
                use_cache=True,
            )

        # Get first token from the last REAL token position (not pad position)
        if rank == world_size - 1:
            last_rank_start = (world_size - 1) * chunk_size
            last_real_local_idx = seq_len - 1 - last_rank_start
            next_token = outputs.logits[:, last_real_local_idx, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        dist.broadcast(next_token, src=world_size - 1, group=group)

        # All-gather KV cache from all ranks
        full_kv = all_gather_kv_cache(outputs.past_key_values, padded_seq_len, world_size, rank, device, group)

        torch.cuda.synchronize(device)
        ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

        # Trim padding from KV cache (pad entries are at the end, from last rank)
        if padded_seq_len != seq_len:
            for layer_idx in range(len(full_kv.key_cache)):
                full_kv.key_cache[layer_idx] = full_kv.key_cache[layer_idx][:, :, :seq_len, :].contiguous()
                full_kv.value_cache[layer_idx] = full_kv.value_cache[layer_idx][:, :, :seq_len, :].contiguous()
            full_kv._seen_tokens = seq_len

        # Free extraction outputs to reclaim local KV cache memory
        del outputs
        torch.cuda.empty_cache()

        # Disable ring attention for generation - use standard attention
        # Generation only happens on rank 0 with the full KV cache
        use_ring_attn(False)

        if rank == 0:
            generated_ids = [next_token.clone()]
            past_kv = full_kv
            current_pos = torch.tensor([[seq_len]], device=device)

            with torch.no_grad():
                for _ in range(max_new_tokens - 1):
                    if eos_token_ids and next_token.item() in eos_token_ids:
                        break

                    gen_out = model(
                        input_ids=next_token,
                        position_ids=current_pos,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = gen_out.past_key_values
                    current_pos = current_pos + 1
                    next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids.append(next_token.clone())

            generated = torch.cat(generated_ids, dim=1)
            answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            del past_kv, full_kv
        else:
            answer = ""
            # Free gathered KV on non-generating ranks immediately
            del full_kv

        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Keep ring attention OFF (safe default; re-enabled at start of next infer call)
        # use_ring_attn stays False from the generation disable above

        # Synchronize all ranks before returning
        dist.barrier(group=group)

        return {"prediction": answer, "ttft_ms": ttft_ms, "seq_len": seq_len}

    return infer


def create_inference_fn_sp_guided_recompute(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, config, recompute_ratio: float, eos_token_ids: set = None
):
    """Create inference function for multi-GPU SP guided recompute.

    Uses the distributed pipeline with independent chunk extraction:
    1. Independent extraction - each GPU processes only its chunk (no ring attention)
    2. Distributed scoring - each GPU scores its local KV, all-gather top-k
    3. Distributed recompute - each GPU recomputes its important positions
       (pre-gathers full KV once for cross-GPU attention context)
    4. All-gather final KV for generation (reuses pre-gathered KV, no extra comm)
    5. Generate on rank 0

    Key insight: extraction is cheap (independent chunks, no cross-GPU comm).
    The recompute step fixes quality at important positions using full-context
    attention via pre-gathered KV.
    """
    from transformers.cache_utils import DynamicCache
    from models.qwen.kv_cache import KVCacheExtractor
    from models.parallel import (
        DistributedExtractor,
        DistributedScorer,
        RingAttentionRecomputer,
    )
    from models.parallel.recomputer import all_gather_kv

    world_size = config.world_size
    rank = config.rank

    # Create distributed components
    base_extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    extractor = DistributedExtractor(base_extractor, config)
    scorer = DistributedScorer(model, config, method="norm", layer_indices=[22, 23, 24, 25])
    recomputer = RingAttentionRecomputer(model, config, use_ring_attention=True)

    def infer(sample: Dict) -> Dict:
        import time

        context_part = sample.get('context', '')
        raw_query = sample.get('input', '')

        # For scoring: use just the raw query text (matching single-GPU pipeline
        # in inference_with_recompute_kv.py:330 which tokenizes only the question)
        scorer_query_ids = tokenizer(raw_query, return_tensors="pt").input_ids.to(device)

        # For generation: use the same prompt template + chat suffix to skip thinking
        query_prompt = (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words. "
            "The answer should be within 5 words.\n"
            f"Question: {raw_query}\nAnswer:"
        )
        gen_query_ids = tokenizer(query_prompt + CHAT_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Add chat prefix to context for extraction (matching inference_with_recompute_kv.py)
        context_ids = tokenizer(CHAT_PREFIX + context_part, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).input_ids.to(device)
        seq_len = context_ids.shape[1]

        # --- TTFT timing: extraction + scoring + recompute + all-gather ---
        torch.cuda.synchronize(device)
        t_ttft_start = time.perf_counter()

        # Step 1: Independent chunk extraction (no ring attention, no cross-GPU comm)
        # Each GPU processes only its local chunk. Cheap: O((T/N)^2) attention per GPU.
        local_kv = extractor.extract_distributed(context_ids)

        # Step 2: Distributed scoring (use raw query, matching single-GPU)
        # Exclude rank 0's local tokens from recomputation selection,
        # matching single-GPU exclude_first_tokens=first_chunk_len behavior
        exclude_first = local_kv.local_seq_len if rank == 0 else 0
        # Broadcast rank 0's value so all ranks agree
        exclude_tensor = torch.tensor([exclude_first], device=device, dtype=torch.long)
        dist.broadcast(exclude_tensor, src=0, group=config.process_group)
        exclude_first = int(exclude_tensor.item())

        local_important, global_important = scorer.score_distributed(
            local_kv, scorer_query_ids, top_ratio=recompute_ratio,
            exclude_first_tokens=exclude_first,
        )

        # Step 3: Distributed recompute
        updated_kv = recomputer.recompute_distributed(
            local_kv, local_important, global_important
        )

        # Free local_kv (superseded by updated_kv)
        del local_kv

        # Step 4: All-gather final KV for generation
        full_kv = all_gather_kv(updated_kv, config)

        torch.cuda.synchronize(device)
        ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

        # Free local updated KV
        del updated_kv
        torch.cuda.empty_cache()

        # Step 5: Generate on rank 0
        if rank == 0:
            generation_cache = DynamicCache()
            generation_cache.key_cache = full_kv.key_cache
            generation_cache.value_cache = full_kv.value_cache
            cache_len = full_kv.key_cache[0].shape[2]
            generation_cache._seen_tokens = cache_len

            with torch.no_grad():
                generated_ids = gen_query_ids.clone()
                past_kv = generation_cache
                next_pos = cache_len  # Track position correctly

                for step in range(max_new_tokens):
                    if step == 0:
                        input_ids = generated_ids
                        position_ids = torch.arange(
                            next_pos, next_pos + generated_ids.shape[1], device=device
                        ).unsqueeze(0)
                        next_pos += generated_ids.shape[1]
                    else:
                        input_ids = generated_ids[:, -1:]
                        position_ids = torch.tensor([[next_pos]], device=device)
                        next_pos += 1

                    gen_out = model(
                        input_ids=input_ids, position_ids=position_ids,
                        past_key_values=past_kv, use_cache=True,
                    )
                    next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    past_kv = gen_out.past_key_values
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            answer = tokenizer.decode(generated_ids[0][gen_query_ids.shape[1]:], skip_special_tokens=True).strip()
            del past_kv, generation_cache, full_kv
        else:
            answer = ""
            # Free gathered KV on non-generating ranks immediately
            del full_kv

        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

        return {"prediction": answer, "ttft_ms": ttft_ms, "seq_len": seq_len}

    return infer


def create_inference_fn_sp_cacheblend(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, config, recompute_ratio: float, eos_token_ids: set = None
):
    """Create inference function for multi-GPU CacheBlend.

    Uses distributed extraction + CacheBlend recomputation (V-diff selection at Layer 1).
    No importance scorer needed — CacheBlend selects positions internally.
    """
    from transformers.cache_utils import DynamicCache
    from models.qwen.kv_cache import KVCacheExtractor
    from models.parallel import DistributedExtractor, RingAttentionRecomputer
    from models.parallel.recomputer import all_gather_kv

    world_size = config.world_size
    rank = config.rank

    base_extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    extractor = DistributedExtractor(base_extractor, config)
    recomputer = RingAttentionRecomputer(model, config, use_ring_attention=True)

    def infer(sample: Dict) -> Dict:
        import time

        context_part = sample.get('context', '')
        raw_query = sample.get('input', '')

        query_prompt = (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words. "
            "The answer should be within 5 words.\n"
            f"Question: {raw_query}\nAnswer:"
        )
        gen_query_ids = tokenizer(query_prompt + CHAT_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        context_ids = tokenizer(CHAT_PREFIX + context_part, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).input_ids.to(device)

        torch.cuda.synchronize(device)
        t_ttft_start = time.perf_counter()

        local_kv = extractor.extract_distributed(context_ids)

        updated_kv = recomputer.recompute_distributed_cacheblend(
            local_kv, recompute_ratio=recompute_ratio
        )

        full_kv = all_gather_kv(updated_kv, config)

        torch.cuda.synchronize(device)
        ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

        del updated_kv
        torch.cuda.empty_cache()

        if rank == 0:
            generation_cache = DynamicCache()
            generation_cache.key_cache = full_kv.key_cache
            generation_cache.value_cache = full_kv.value_cache
            cache_len = full_kv.key_cache[0].shape[2]
            generation_cache._seen_tokens = cache_len

            with torch.no_grad():
                generated_ids = gen_query_ids.clone()
                past_kv = generation_cache
                next_pos = cache_len

                for step in range(max_new_tokens):
                    if step == 0:
                        input_ids = generated_ids
                        position_ids = torch.arange(
                            next_pos, next_pos + generated_ids.shape[1], device=device
                        ).unsqueeze(0)
                        next_pos += generated_ids.shape[1]
                    else:
                        input_ids = generated_ids[:, -1:]
                        position_ids = torch.tensor([[next_pos]], device=device)
                        next_pos += 1

                    gen_out = model(
                        input_ids=input_ids, position_ids=position_ids,
                        past_key_values=past_kv, use_cache=True,
                    )
                    next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    past_kv = gen_out.past_key_values
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            answer = tokenizer.decode(generated_ids[0][gen_query_ids.shape[1]:], skip_special_tokens=True).strip()
            del past_kv, generation_cache, full_kv
        else:
            answer = ""
            del full_kv

        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        return {"prediction": answer, "ttft_ms": ttft_ms}

    return infer


def create_inference_fn_sp_lego(
    model, tokenizer, dataset: LongBenchDataset, device: str, max_new_tokens: int, config, recompute_ratio: float, eos_token_ids: set = None
):
    """Create inference function for multi-GPU LEGO.

    Uses distributed extraction + LEGO position selection (first recompute_ratio
    of each GPU's local chunk) + standard distributed recomputation.
    No importance scorer needed — positions are deterministic.
    """
    from transformers.cache_utils import DynamicCache
    from models.qwen.kv_cache import KVCacheExtractor
    from models.parallel import DistributedExtractor, RingAttentionRecomputer
    from models.parallel.recomputer import all_gather_kv, allgather_positions

    world_size = config.world_size
    rank = config.rank

    base_extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    extractor = DistributedExtractor(base_extractor, config)
    recomputer = RingAttentionRecomputer(model, config, use_ring_attention=True)

    def infer(sample: Dict) -> Dict:
        import time

        context_part = sample.get('context', '')
        raw_query = sample.get('input', '')

        query_prompt = (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words. "
            "The answer should be within 5 words.\n"
            f"Question: {raw_query}\nAnswer:"
        )
        gen_query_ids = tokenizer(query_prompt + CHAT_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        context_ids = tokenizer(CHAT_PREFIX + context_part, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).input_ids.to(device)

        torch.cuda.synchronize(device)
        t_ttft_start = time.perf_counter()

        local_kv = extractor.extract_distributed(context_ids)

        # LEGO: select first recompute_ratio of each GPU's local chunk
        local_T = local_kv.local_seq_len
        take = max(1, int(local_T * recompute_ratio))
        local_important = torch.arange(take, device=device, dtype=torch.long)

        global_important = allgather_positions(
            local_important, local_kv.global_offset, config
        )

        updated_kv = recomputer.recompute_distributed(
            local_kv, local_important, global_important
        )

        full_kv = all_gather_kv(updated_kv, config)

        torch.cuda.synchronize(device)
        ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

        del updated_kv
        torch.cuda.empty_cache()

        if rank == 0:
            generation_cache = DynamicCache()
            generation_cache.key_cache = full_kv.key_cache
            generation_cache.value_cache = full_kv.value_cache
            cache_len = full_kv.key_cache[0].shape[2]
            generation_cache._seen_tokens = cache_len

            with torch.no_grad():
                generated_ids = gen_query_ids.clone()
                past_kv = generation_cache
                next_pos = cache_len

                for step in range(max_new_tokens):
                    if step == 0:
                        input_ids = generated_ids
                        position_ids = torch.arange(
                            next_pos, next_pos + generated_ids.shape[1], device=device
                        ).unsqueeze(0)
                        next_pos += generated_ids.shape[1]
                    else:
                        input_ids = generated_ids[:, -1:]
                        position_ids = torch.tensor([[next_pos]], device=device)
                        next_pos += 1

                    gen_out = model(
                        input_ids=input_ids, position_ids=position_ids,
                        past_key_values=past_kv, use_cache=True,
                    )
                    next_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    past_kv = gen_out.past_key_values
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            answer = tokenizer.decode(generated_ids[0][gen_query_ids.shape[1]:], skip_special_tokens=True).strip()
            del past_kv, generation_cache, full_kv
        else:
            answer = ""
            del full_kv

        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        return {"prediction": answer, "ttft_ms": ttft_ms}

    return infer


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    if is_main:
        print("=" * 70)
        print("LongBench Evaluation (using official framework)")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Tasks: {args.tasks}")
        print(f"Methods: {args.methods}")
        print(f"Recompute ratio: {args.recompute_ratio}")
        print(f"Max samples per task: {args.max_samples}")
        print(f"Input directory: {args.input_dir}")
        print(f"World size: {world_size}")
        print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_main:
        print("\nLoading model...")

    # Enable YaRN RoPE scaling for 131072 context (Qwen3 native 32K → 128K with YaRN)
    rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        rope_scaling=rope_scaling,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect all EOS token IDs from generation config (Qwen3 has multiple: <|im_end|> and <|endoftext|>)
    eos_token_ids = set()
    if hasattr(model, 'generation_config') and model.generation_config.eos_token_id is not None:
        eid = model.generation_config.eos_token_id
        if isinstance(eid, list):
            eos_token_ids.update(eid)
        else:
            eos_token_ids.add(eid)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    if is_main:
        print(f"EOS token IDs: {eos_token_ids}")

    # Setup distributed config and ring attention
    config = None
    ring_group = None
    if world_size > 1:
        from models.parallel.config import DistributedConfig
        config = DistributedConfig.from_env(recompute_ratio=args.recompute_ratio)

        # Setup process group for distributed methods
        dist_methods = {"ring_attention", "sp_guided_recompute", "sp_cacheblend", "sp_lego"}
        if dist_methods & set(args.methods):
            ring_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
            config.process_group = ring_group

            # Ring attention substitution only needed for ring_attention method
            if "ring_attention" in args.methods:
                from ring_flash_attn import substitute_hf_flash_attn
                from ring_flash_attn.adapters.hf_adapter import use_ring_attn

                substitute_hf_flash_attn(ring_group, heads_k_stride=args.heads_k_stride)
                use_ring_attn(False)  # Default OFF; enabled only during ring_attention prefill

    # Run evaluations for each task
    all_results = {}

    for task_name in args.tasks:
        if is_main:
            print(f"\n{'='*70}")
            print(f"Task: {task_name}")
            print(f"{'='*70}")

        # Load dataset using existing framework
        dataset = get_dataset(task_name, input_dir=args.input_dir, device=device)
        dataset.load(num_samples=args.max_samples or None)

        if is_main:
            print(f"Loaded {len(dataset.data)} samples")

        # Create inference functions for each method
        inference_fns = {}

        methods_to_run = args.methods

        for method in methods_to_run:
            # Skip multi-GPU methods on single GPU
            if method in ["sp_guided_recompute", "ring_attention", "sp_cacheblend", "sp_lego"] and world_size == 1:
                if is_main:
                    print(f"  {method}: Skipped (requires multi-GPU)")
                continue

            if method == "baseline":
                inference_fns[method] = create_inference_fn_baseline(
                    model, tokenizer, dataset, device, args.max_new_tokens
                )
            elif method == "single_gpu_prefill":
                inference_fns[method] = create_inference_fn_single_gpu_prefill(
                    model, tokenizer, dataset, device, args.max_new_tokens, eos_token_ids
                )
            elif method == "guided_recompute":
                inference_fns[method] = create_inference_fn_guided_recompute(
                    model, tokenizer, dataset, device, args.max_new_tokens, args.recompute_ratio
                )
            elif method == "ring_attention":
                inference_fns[method] = create_inference_fn_ring_attention(
                    model, tokenizer, dataset, device, args.max_new_tokens, config, ring_group, eos_token_ids
                )
            elif method == "sp_guided_recompute":
                inference_fns[method] = create_inference_fn_sp_guided_recompute(
                    model, tokenizer, dataset, device, args.max_new_tokens, config, args.recompute_ratio, eos_token_ids
                )
            elif method == "sp_cacheblend":
                inference_fns[method] = create_inference_fn_sp_cacheblend(
                    model, tokenizer, dataset, device, args.max_new_tokens, config, args.recompute_ratio, eos_token_ids
                )
            elif method == "sp_lego":
                inference_fns[method] = create_inference_fn_sp_lego(
                    model, tokenizer, dataset, device, args.max_new_tokens, config, args.recompute_ratio, eos_token_ids
                )

        if not inference_fns:
            if is_main:
                print("  No methods to evaluate")
            continue

        # Run evaluation using existing framework
        # This uses proper F1 computation with parse_generation() and normalize_answer()
        results = dataset.evaluate(
            inference_fns=inference_fns,
            num_samples=args.max_samples or None,
            warmup_samples=1,
        )

        all_results[task_name] = results

        # Print results
        if is_main:
            print(f"\n  Results for {task_name}:")
            for method, result in results.items():
                summary = result["summary"]
                print(f"    {method}:")
                print(f"      F1: {summary['avg_f1']*100:.2f}%")
                print(f"      Accuracy: {summary['accuracy']:.2f}%")
                print(f"      Samples: {summary['total_samples']}")
                if "avg_ttft_ms" in summary:
                    print(f"      Avg TTFT: {summary['avg_ttft_ms']:.1f}ms")

            # Save results
            dataset.save_results(results[list(results.keys())[0]], label=list(results.keys())[0], output_dir=args.output)

    # Print summary table
    if is_main:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"{'Task':<15} {'Method':<25} {'F1 (%)':<10} {'Accuracy (%)':<12} {'TTFT (ms)':<10}")
        print("-" * 80)

        for task_name, task_results in all_results.items():
            for method, result in task_results.items():
                summary = result["summary"]
                ttft_str = f"{summary['avg_ttft_ms']:.0f}" if "avg_ttft_ms" in summary else "-"
                print(f"{task_name:<15} {method:<25} {summary['avg_f1']*100:<10.2f} {summary['accuracy']:<12.2f} {ttft_str:<10}")

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output) / f"combined_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        json_results = {}
        for task_name, task_results in all_results.items():
            json_results[task_name] = {}
            for method, result in task_results.items():
                json_results[task_name][method] = result["summary"]

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nCombined results saved to: {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
