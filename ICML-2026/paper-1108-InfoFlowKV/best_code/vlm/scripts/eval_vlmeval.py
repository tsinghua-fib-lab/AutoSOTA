#!/usr/bin/env python3
"""
Evaluation script using VLMEvalKit with KV cache recomputation.

Usage:
    # Baseline (no chunking, native generation)
    python scripts/eval_vlmeval.py --data ChartQA --strategy no_recompute --chunk_k 0

    # No recompute with chunking (degraded quality)
    python scripts/eval_vlmeval.py --data ChartQA --strategy no_recompute --chunk_k 4

    # Guided recompute with FlashInfer attention
    python scripts/eval_vlmeval.py --data ChartQA --strategy guided_recompute --chunk_k 4 --attention_mode flashinfer

    # LEGO baseline
    python scripts/eval_vlmeval.py --data ChartQA --strategy lego --chunk_k 4

    # CacheBlend baseline
    python scripts/eval_vlmeval.py --data ChartQA --strategy cacheblend --chunk_k 4

    # Multiple datasets
    python scripts/eval_vlmeval.py --data ChartQA OCRBench DocVQA --strategy guided_recompute --chunk_k 4
"""

import os
import sys

# Add VLMEvalKit to path
sys.path.insert(0, "/path/to/VLMEvalKit")
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from datetime import datetime

from vlmeval.dataset import build_dataset
from vlmeval.smp import dump, load

from models.qwen.kv_cache import (
    VLMKVCacheExtractor,
    ImportanceScorer,
    KVCacheRecomputer,
    KVCacheInference,
    RecomputeConfig,
)


class Qwen3VLWithKVRecompute:
    """Qwen3-VL model with KV cache recomputation strategies for VLMEvalKit."""

    def __init__(
        self,
        model_path: str,
        strategy: str = "no_recompute",
        recompute_ratio: float = 0.15,
        method: str = "norm",
        chunk_k: int = 0,
        attention_mode: str = "flashinfer",
        max_new_tokens: int = 128,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.strategy = strategy
        self.recompute_ratio = recompute_ratio
        self.method = method
        self.chunk_k = chunk_k
        self.attention_mode = attention_mode
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # Load model and processor
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"[1/4] Loading model: {model_path}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        print(f"[2/4] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        print(f"[3/4] Initializing KV cache components...")
        self.extractor = VLMKVCacheExtractor(self.model)
        self.scorer = ImportanceScorer(self.model, method=self.method)
        self.recomputer = KVCacheRecomputer(
            self.model,
            recompute_attention_mode=self.attention_mode,
        )
        self.kv_inference = KVCacheInference(self.model, self.processor)

        self.config = RecomputeConfig(
            strategy=self.strategy,
            recompute_ratio=self.recompute_ratio,
            method=self.method,
        )

        print(f"[4/4] Ready. Strategy: {strategy}, chunk_k: {chunk_k}, ratio: {recompute_ratio}, attn_mode: {attention_mode}")
        torch.cuda.empty_cache()

    def generate(self, message, dataset=None):
        """Generate response for VLMEvalKit format message."""
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        # Build messages from VLMEvalKit format
        content = []
        for item in message:
            if item['type'] == 'image':
                img_path = item['value']
                if os.path.exists(img_path):
                    content.append({'type': 'image', 'image': f'file://{img_path}'})
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})

        messages = [{'role': 'user', 'content': content}]

        # Process inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            return_tensors='pt',
        )
        inputs = inputs.to(self.device)

        # Generate using KV cache recomputation pipeline
        if self.chunk_k == 0 and self.strategy == "no_recompute":
            # Native generation (baseline)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        else:
            with torch.no_grad():
                # Extract KV cache with chunked prefill
                kv_data = self.extractor.extract(
                    inputs,
                    chunk_k=self.chunk_k if self.chunk_k > 0 else None,
                )

                # At chunk_k=0 or 1 (single chunk = full attention), skip recomputation.
                # The extracted KV cache is already correct with no cross-chunk artifacts.
                if self.chunk_k <= 1:
                    updated_cache = kv_data.past_key_values
                elif self.strategy == "no_recompute":
                    updated_cache = self.recomputer.recompute_noop(kv_data, return_kv_data=False)
                elif self.strategy == "lego":
                    updated_cache = self.recomputer.recompute_lego(kv_data, ratio=self.recompute_ratio, return_kv_data=False)
                elif self.strategy == "cacheblend":
                    updated_cache = self.recomputer.recompute_cacheblend(kv_data, self.recompute_ratio, return_kv_data=False)
                else:  # guided_recompute
                    scores = self.scorer.compute(kv_data)
                    recompute_indices = self.scorer.select_positions(scores, self.config, kv_data.image_ranges)
                    updated_cache = self.recomputer.recompute(kv_data, recompute_indices, return_kv_data=False)

                # Generate with updated cache
                result = self.kv_inference.generate(
                    updated_cache,
                    inputs.input_ids,
                    kv_data.seq_len,
                    max_new_tokens=self.max_new_tokens,
                )
                return result["text"]

        response = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response


def run_evaluation(model, dataset, dataset_name, output_dir, num_samples=None):
    """Run evaluation on dataset using VLMEvalKit's evaluation methods."""
    from tqdm import tqdm

    results = []
    data = dataset.data if hasattr(dataset, 'data') else dataset

    total = len(data)
    if num_samples is not None:
        total = min(num_samples, total)

    print(f"\nEvaluating {dataset_name}")
    print(f"Total samples: {total}" + (f" (limited from {len(data)})" if num_samples else ""))

    for i in tqdm(range(total), desc="Processing"):
        item = data.iloc[i] if hasattr(data, 'iloc') else data[i]

        # Build message in VLMEvalKit format
        message = dataset.build_prompt(item)

        # Generate response
        prediction = model.generate(message, dataset=dataset_name)

        # Store result - include all original columns plus prediction
        result = dict(item) if hasattr(item, 'items') else item.to_dict()
        result['prediction'] = prediction

        results.append(result)

        if i < 3 or (i + 1) % 50 == 0:
            print(f"\n[{i}] Pred: {prediction[:80]}...")

    # Convert to DataFrame for VLMEvalKit compatibility
    result_df = pd.DataFrame(results)

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = os.path.join(output_dir, f"{dataset_name}_{timestamp}.xlsx")
    result_df.to_excel(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # Run VLMEvalKit evaluation
    try:
        eval_result = dataset.evaluate(pred_file)
        print(f"\n{'=' * 60}")
        print(f"RESULTS for {dataset_name}")
        print('=' * 60)
        if isinstance(eval_result, dict):
            for key, value in eval_result.items():
                print(f"{key}: {value}")
        else:
            print(eval_result)
        return eval_result
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate with VLMEvalKit + KV Recompute")
    parser.add_argument("--model", type=str, default="/path/to/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data", type=str, nargs="+", required=True,
                       help="Dataset names (e.g., ChartQA, OCRBench, DocVQA)")
    parser.add_argument("--strategy", type=str, default="no_recompute",
                       choices=["no_recompute", "lego", "cacheblend", "guided_recompute"])
    parser.add_argument("--recompute_ratio", type=float, default=0.15)
    parser.add_argument("--method", type=str, default="norm", choices=["norm", "mass", "entropy", "vatp", "combined"])
    parser.add_argument("--chunk_k", type=int, default=0)
    parser.add_argument("--attention_mode", type=str, default="flashinfer", choices=["flashinfer", "sdpa", "math"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./vlmeval_output")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for quick testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("VLMEvalKit Evaluation with KV Cache Recomputation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Chunk K: {args.chunk_k}")
    print(f"Recompute Ratio: {args.recompute_ratio}")
    print(f"Scoring Method: {args.method}")
    print(f"Attention Mode: {args.attention_mode}")
    print(f"Datasets: {args.data}")
    print("=" * 60)

    # Initialize model once
    model = Qwen3VLWithKVRecompute(
        model_path=args.model,
        strategy=args.strategy,
        recompute_ratio=args.recompute_ratio,
        method=args.method,
        chunk_k=args.chunk_k,
        attention_mode=args.attention_mode,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    # Evaluate each dataset
    all_results = {}
    for dataset_name in args.data:
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {dataset_name}")
        print("=" * 60)

        try:
            dataset = build_dataset(dataset_name)
            if dataset is None:
                print(f"Failed to build dataset: {dataset_name}")
                continue

            result = run_evaluation(model, dataset, dataset_name, args.output_dir, args.num_samples)
            all_results[dataset_name] = result

        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"\n{name}:")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
