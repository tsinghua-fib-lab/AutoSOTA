import os
import sys

# Ensure project root is importable regardless of CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import random
import time
from itertools import islice
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, utils

import llm_attr
import llm_attr_eval
from exp.exp2 import dataset_utils as ds_utils


utils.logging.set_verbosity_error()


def _first_json_obj(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _load_ruler_examples(args) -> Tuple[str, List[ds_utils.CachedExample]]:
    ds_arg = args.dataset
    cache_dir = Path(args.data_root)

    # 1) If dataset points to an existing file, detect cache vs raw RULER.
    p = Path(ds_arg)
    if p.exists():
        obj = _first_json_obj(p)
        if "prompt" in obj:
            return p.stem, ds_utils.load_cached(p, sample=args.sample, seed=args.seed)
        if "input" in obj and "needle_spans" in obj:
            return p.stem, ds_utils.load_ruler(p, sample=args.sample, seed=args.seed)
        raise SystemExit(
            f"Unsupported JSONL schema for recovery_ruler: {p}. "
            "Expected either exp2 cache (has 'prompt') or raw RULER JSONL (has 'input'+'needle_spans')."
        )

    # 2) Prefer exp2 cache under --data_root by dataset name.
    cached = cache_dir / f"{ds_arg}.jsonl"
    if cached.exists():
        return ds_arg, ds_utils.load_cached(cached, sample=args.sample, seed=args.seed)

    # 3) Fall back to raw RULER resolution by name.
    resolved = ds_utils.dataset_from_name(ds_arg)
    if resolved is None:
        raise SystemExit(f"Could not resolve RULER dataset name '{ds_arg}'.")
    return ds_arg, ds_utils.load_ruler(resolved, sample=args.sample, seed=args.seed)


def _resolve_indices_to_explain_token_span(
    attr_result: llm_attr.LLMAttributionResult, indices_to_explain: list[int] | None
) -> list[int]:
    if (
        isinstance(indices_to_explain, list)
        and len(indices_to_explain) == 2
        and all(isinstance(x, int) and x >= 0 for x in indices_to_explain)
        and indices_to_explain[0] <= indices_to_explain[1]
    ):
        return indices_to_explain

    gen_len = int(attr_result.attribution_matrix.shape[0])
    if gen_len <= 0:
        return [0, 0]

    # Default: explain the full generation excluding the appended EOS token.
    end_tok = max(0, gen_len - 2)
    return [0, end_tok]


def run_attribution(
    testing_dict, example: ds_utils.CachedExample, batch_size: int, target: Optional[str]
) -> tuple[List[torch.Tensor], dict | None]:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]
    attr_func = testing_dict["attr_func"]

    if "IG" in attr_func:
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)
        attr = llm_attributor.calculate_IG_per_generation(
            example.prompt,
            20,
            tokenizer.eos_token_id,
            batch_size=batch_size,
            target=target,
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if "perturbation" in attr_func:
        llm_attributor = llm_attr.LLMPerturbationAttribution(model, tokenizer)
        if attr_func == "perturbation_all":
            attr = llm_attributor.calculate_feature_ablation_sentences(
                example.prompt, baseline=tokenizer.eos_token_id, measure="log_loss", target=target
            )
        elif attr_func == "perturbation_CLP":
            attr = llm_attributor.calculate_feature_ablation_sentences(
                example.prompt, baseline=tokenizer.eos_token_id, measure="KL", target=target
            )
        elif attr_func == "perturbation_REAGENT":
            attr = llm_attributor.calculate_feature_ablation_sentences_mlm(example.prompt, target=target)
        else:
            raise ValueError(f"Unsupported perturbation attr_func {attr_func}")
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if "attention" in attr_func:
        llm_attributor = llm_attr.LLMAttentionAttribution(model, tokenizer)
        llm_attributor_ig = llm_attr.LLMGradientAttribtion(model, tokenizer)
        attr = llm_attributor.calculate_attention_attribution(example.prompt, target=target)
        if attr_func == "attention_I_G":
            attr_b = llm_attributor_ig.calculate_IG_per_generation(
                example.prompt, 20, tokenizer.eos_token_id, batch_size=batch_size, target=target
            )
            attr.attribution_matrix = attr.attribution_matrix * attr_b.attribution_matrix
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_all_positions":
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        attr = llm_attributor.calculate_ifr_for_all_positions(example.prompt, target=target)
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_all_positions_output_only":
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        sink_span = tuple(example.sink_span) if example.sink_span else None
        attr = llm_attributor.calculate_ifr_for_all_positions_output_only(
            example.prompt,
            target=target,
            sink_span=sink_span,
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_span":
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        span = example.sink_span if example.sink_span else None
        attr = llm_attributor.calculate_ifr_span(example.prompt, target=target, span=tuple(span) if span else None)
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_multi_hop":
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        attr = llm_attributor.calculate_ifr_multi_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_in_all_gen":
        import ft_ifr_improve

        llm_attributor = ft_ifr_improve.LLMIFRAttributionInAllGen(model, tokenizer)
        attr = llm_attributor.calculate_ifr_in_all_gen(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "ifr_multi_hop_stop_words":
        import ft_ifr_improve

        llm_attributor = ft_ifr_improve.LLMIFRAttributionImproved(model, tokenizer)
        attr = llm_attributor.calculate_ifr_multi_hop_stop_words(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        extra = {
            "keep_prompt_token_indices": ft_ifr_improve.keep_token_indices(list(attr.prompt_tokens)),
        }
        return list(attr.get_all_token_attrs(token_span)), extra

    if attr_func == "ifr_multi_hop_both":
        import ft_ifr_improve

        llm_attributor = ft_ifr_improve.LLMIFRAttributionBoth(model, tokenizer)
        attr = llm_attributor.calculate_ifr_multi_hop_both(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        extra = {
            "keep_prompt_token_indices": ft_ifr_improve.keep_token_indices(list(attr.prompt_tokens)),
        }
        return list(attr.get_all_token_attrs(token_span)), extra

    if attr_func == "ifr_multi_hop_split_hop":
        import ft_ifr_improve

        llm_attributor = ft_ifr_improve.LLMIFRAttributionSplitHop(model, tokenizer)
        attr = llm_attributor.calculate_ifr_multi_hop_split_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "basic":
        llm_attributor = llm_attr.LLMBasicAttribution(model, tokenizer)
        attr = llm_attributor.calculate_basic_attribution(example.prompt, target=target)
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "attnlrp":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        sink_span = getattr(example, "sink_span", None)
        thinking_span = getattr(example, "thinking_span", None)
        attr = llm_attributor.calculate_attnlrp_ft_hop0(
            example.prompt,
            target=target,
            sink_span=tuple(sink_span) if sink_span else None,
            thinking_span=tuple(thinking_span) if thinking_span else None,
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "attnlrp_aggregated":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated(example.prompt, target=target)
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    if attr_func == "attnlrp_aggregated_multi_hop":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated_multi_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict.get("n_hops", 1),
        )
        token_span = _resolve_indices_to_explain_token_span(attr, example.indices_to_explain)
        return list(attr.get_all_token_attrs(token_span)), None

    raise ValueError(f"Unsupported attribution function '{attr_func}'.")


def evaluate_dataset_recovery_ruler(testing_dict, dataset_name: str, examples: List[ds_utils.CachedExample]) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    tokenizer = testing_dict["tokenizer"]
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(testing_dict["model"], tokenizer)

    results: List[np.ndarray] = []
    durations: List[float] = []
    skipped = 0

    num_examples = testing_dict["num_examples"]
    total = min(len(examples), num_examples)
    iterator = islice(examples, total)

    description = f"Recovery@10pct {testing_dict['model_name']} {dataset_name} {testing_dict['attr_func']}"
    for ex in tqdm(iterator, desc=description, total=total):
        needle_spans = (ex.metadata or {}).get("needle_spans")
        if not isinstance(needle_spans, list) or not needle_spans:
            raise SystemExit(
                "recovery_ruler only supports RULER examples with metadata.needle_spans; "
                f"dataset={dataset_name} has missing/empty needle_spans."
            )

        gold_prompt = ds_utils.ruler_gold_prompt_token_indices(ex, tokenizer)
        if not gold_prompt:
            skipped += 1
            continue

        # Batch size is set based on the max_input_len (same policy as faithfulness).
        target = ex.target
        if target is None:
            generation, full_output = llm_evaluator.response(ex.prompt)
            target = generation
            response_len = len(tokenizer(full_output).input_ids)
        else:
            response_len = len(tokenizer(llm_evaluator.format_prompt(" " + ex.prompt) + target).input_ids)
        batch_size = max(1, math.floor((testing_dict["max_input_len"] - 100) / max(1, response_len)))

        sample_start = time.perf_counter()
        attr_list, extra = run_attribution(testing_dict, ex, batch_size, target)
        durations.append(time.perf_counter() - sample_start)

        seq_attr = attr_list[0]
        prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G
        if prompt_len <= 0:
            skipped += 1
            continue

        if testing_dict["attr_func"] in ("ifr_multi_hop_stop_words", "ifr_multi_hop_both") and extra is not None:
            import ft_ifr_improve

            keep_prompt_token_indices = extra.get("keep_prompt_token_indices") or []
            gold_filtered = [idx for idx in gold_prompt if int(idx) in set(int(x) for x in keep_prompt_token_indices)]
            if not gold_filtered:
                skipped += 1
                continue
            scores = [
                ft_ifr_improve.evaluate_attr_recovery_skip_tokens(
                    attr[:, :prompt_len],
                    keep_prompt_token_indices=keep_prompt_token_indices,
                    gold_prompt_token_indices=gold_prompt,
                    top_fraction=0.1,
                )
                for attr in attr_list
            ]
        else:
            scores = [
                llm_evaluator.evaluate_attr_recovery(
                    attr,
                    prompt_len=prompt_len,
                    gold_prompt_token_indices=gold_prompt,
                    top_fraction=0.1,
                )
                for attr in attr_list
            ]
        results.append(np.asarray(scores, dtype=np.float64))

    scores = np.stack(results, axis=0) if results else np.zeros((0, 3), dtype=np.float64)
    used = int(scores.shape[0])
    mean = scores.mean(0) if used else np.full((3,), np.nan, dtype=np.float64)
    std = scores.std(0) if used else np.full((3,), np.nan, dtype=np.float64)
    avg_time = float(np.mean(durations)) if durations else 0.0
    return mean, std, avg_time, used, int(skipped)


def load_model(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except Exception:
            gpu_idx = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main(args) -> None:
    if args.cuda is not None and isinstance(args.cuda, str) and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = "auto"
    elif args.cuda is not None and isinstance(args.cuda, str) and args.cuda.strip() != "":
        try:
            idx = int(args.cuda)
        except Exception:
            idx = 0
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    else:
        device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    if args.model == "llama-1B":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        max_input_len = 5500
    elif args.model == "llama-3B":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        max_input_len = 4800
    elif args.model == "llama-8B":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        max_input_len = 3500
    elif args.model == "qwen-1.7B":
        model_name = "Qwen/Qwen3-1.7B"
        max_input_len = 5500
    elif args.model == "qwen-4B":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
        max_input_len = 3500
    elif args.model == "qwen-8B":
        model_name = "Qwen/Qwen3-8B"
        max_input_len = 3000
    elif args.model == "qwen-32B":
        model_name = "Qwen/Qwen3-32B"
        max_input_len = 1500
    elif args.model == "gemma-12B":
        model_name = "gemma/gemma-3-12b-it"
        max_input_len = 1500
    elif args.model == "gemma-27B":
        model_name = "gemma/gemma-3-27b-it"
        max_input_len = 2000
    else:
        model_name = args.model_path if args.model_path is not None else args.model
        max_input_len = 2000

    model, tokenizer = load_model(model_name if args.model_path is None else args.model_path, device)

    dataset_name, examples = _load_ruler_examples(args)

    testing_dict = {
        "model": model,
        "model_name": args.model,
        "tokenizer": tokenizer,
        "dataset_name": dataset_name,
        "attr_func": args.attr_func,
        "num_examples": args.num_examples,
        "max_input_len": max_input_len,
        "n_hops": args.n_hops,
    }

    mean, std, avg_time, used, skipped = evaluate_dataset_recovery_ruler(testing_dict, dataset_name, examples)

    out_dir = Path("./test_results") / "attribution_recovery" / dataset_name / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{args.attr_func}_{args.num_examples}_examples.csv"
    with open(out_dir / file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Recovery@10pct"])
        writer.writerow(["Seq Attr Recovery Mean", mean[0]])
        writer.writerow(["Row Attr Recovery Mean", mean[1]])
        writer.writerow(["Recursive Attr Recovery Mean", mean[2]])
        writer.writerow(["Seq Attr Recovery Std", std[0]])
        writer.writerow(["Row Attr Recovery Std", std[1]])
        writer.writerow(["Recursive Attr Recovery Std", std[2]])
        writer.writerow(["Examples Used", used])
        writer.writerow(["Examples Skipped", skipped])
        writer.writerow(["Avg Sample Time (s)", avg_time])

    print(f"[{dataset_name}] {args.attr_func} -> {out_dir/file_name} (used={used} skipped={skipped} avg {avg_time:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RULER-only token-level attribution recovery evaluation (Recall@10pct).")
    parser.add_argument("--num_examples", type=int, default=100, help="How many examples to evaluate.")
    parser.add_argument("--sample", type=int, default=None, help="Optional subsample before num_examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="qwen-8B")
    parser.add_argument("--model_path", type=str, default=None, help="Optional local model path to load.")
    parser.add_argument("--attr_func", type=str, default="ifr_multi_hop")
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, help="RULER dataset name or JSONL path (raw or exp2 cache).")
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Cache directory to search by dataset name.")
    parser.add_argument("--n_hops", type=int, default=3)

    args, _ = parser.parse_known_args()
    main(args)
