import argparse
import json
import os
import sys
import yaml
import re
from typing import List, Dict, Optional
from datetime import datetime
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from model import SoftSeft
from utils import Config, set_seed
from eval.dataset import load_test_dataset, NAME_REPO_MAPPING
from eval.utils import (
    extract_answer_from_output,
    normalize_answer,
    check_answer_correct,
    apply_chat_template_if_needed
)


def _init_distributed(debug: bool = False):
    """Initialize distributed training environment."""
    # Check if we're in a distributed environment (launched with torchrun/torch.distributed.launch)
    # Only consider distributed if WORLD_SIZE > 1 (multiple GPUs)
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size_env > 1
    
    if debug or not is_distributed:
        # Single GPU mode - no distributed initialization needed
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return rank, world_size, local_rank, device

    # Distributed mode - initialize process group
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://", timeout=timedelta(hours=1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if current process is main."""
    return get_rank() == 0


def print_rank_0(msg: str):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_results_from_all_ranks(local_results: List[Dict]) -> List[Dict]:
    """Gather results from all ranks to rank 0."""
    world_size = get_world_size()
    
    if world_size == 1:
        return local_results
    
    if dist.is_initialized():
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        
        if is_main_process():
            all_results = []
            for results in gathered_results:
                all_results.extend(results)
            return all_results
    
    return local_results


def load_model_and_tokenizer(configs: Config, device: torch.device):
    """Load model and tokenizer using SoftSeft.from_pretrained()."""
    print_rank_0(f"Loading model from {configs.model_path}")
    # import pdb; pdb.set_trace()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Get thinking_token_id
    thinking_token = getattr(configs, "thinking_token", "<thinking>")
    use_unk_for_thinking = getattr(configs, "use_unk_for_thinking", False)
    
    if use_unk_for_thinking:
        if tokenizer.unk_token_id is None:
            raise ValueError("Tokenizer has no unk_token but use_unk_for_thinking=True")
        thinking_token_id = tokenizer.unk_token_id
    else:
        if thinking_token not in tokenizer.get_vocab():
            raise ValueError(f"Thinking token '{thinking_token}' not found in tokenizer vocabulary. "
                           "Please ensure the model was trained with this token.")
        thinking_token_id = tokenizer.convert_tokens_to_ids(thinking_token)
    
    print_rank_0(f"Thinking token: '{thinking_token}' (id={thinking_token_id})")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "thinking_token_id": thinking_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "base_model_name_or_path": configs.model_name_or_path,
        "device": str(device),
    }
    
    # Add optional parameters if specified in config
    if hasattr(configs, "thinking_use_mlp"):
        model_kwargs["use_thinking_mlp"] = configs.thinking_use_mlp

    if hasattr(configs, "thinking_mlp_hidden_dim"):
        model_kwargs["mlp_hidden_dim"] = configs.thinking_mlp_hidden_dim

    if hasattr(configs, "thinking_mlp_activation"):
        model_kwargs["mlp_activation"] = configs.thinking_mlp_activation

    if hasattr(configs, "thinking_hidden_state_layer"):
        raw_layer = configs.thinking_hidden_state_layer
        if isinstance(raw_layer, str):
            raw_layer = -1 if raw_layer.lower() == "none" else int(raw_layer)
        model_kwargs["hidden_state_layer_index"] = int(raw_layer)

    if hasattr(configs, "eval_stage_mode"):
        model_kwargs["stage_mode"] = configs.eval_stage_mode
    else:
        model_kwargs["stage_mode"] = "common"

    # Handle fusion_alpha - may be a list (one per stage) or a single value
    if hasattr(configs, "fusion_alpha"):
        fusion_alpha_cfg = configs.fusion_alpha
        if isinstance(fusion_alpha_cfg, list):
            # Select fusion_alpha based on eval_stage_mode
            stage_mode = model_kwargs.get("stage_mode", "common")
            stage_modes = getattr(configs, "stage_modes", ["common", "hidden_state", "soft_fusion"])
            try:
                stage_idx = stage_modes.index(stage_mode)
                fusion_alpha_value = fusion_alpha_cfg[stage_idx] if stage_idx < len(fusion_alpha_cfg) else fusion_alpha_cfg[-1]
            except (ValueError, IndexError):
                fusion_alpha_value = fusion_alpha_cfg[-1]  # Default to last value
            print_rank_0(f"fusion_alpha: selected {fusion_alpha_value} for stage_mode '{stage_mode}'")
            model_kwargs["fusion_alpha"] = fusion_alpha_value
        else:
            model_kwargs["fusion_alpha"] = fusion_alpha_cfg

    if hasattr(configs, "fusion_top_p"):
        model_kwargs["fusion_top_p"] = configs.fusion_top_p

    if hasattr(configs, "fusion_temperature"):
        model_kwargs["fusion_temperature"] = configs.fusion_temperature
    
    print_rank_0(f"Model loading kwargs: {model_kwargs}")
    
    # Load model using from_pretrained
    model = SoftSeft.from_pretrained(
        model_path=configs.model_path,
        **model_kwargs
    )
    
    # Set dtype if specified (also ensure device)
    if getattr(configs, "bf16", False):
        model = model.to(device=device, dtype=torch.bfloat16)
        print_rank_0(f"Converted model to bfloat16 on {device}")
    elif getattr(configs, "fp16", False):
        model = model.to(device=device, dtype=torch.float16)
        print_rank_0(f"Converted model to float16 on {device}")
    else:
        # Ensure on correct device even without dtype conversion
        model = model.to(device)
    
    print_rank_0(f"Model loaded: {type(model).__name__}, dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")
    
    return model, tokenizer, thinking_token_id


def evaluate_single_dataset(
    model,
    tokenizer,
    dataset_name: str,
    configs: Config,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Dict:
    """Evaluate model on a single dataset."""
    print_rank_0(f"\n{'='*60}")
    print_rank_0(f"Evaluating on {dataset_name}")
    print_rank_0(f"{'='*60}")
    
    # Load dataset
    data = load_test_dataset(dataset_name)
    
    # Limit samples in debug mode
    if getattr(configs, "debug", False):
        data = data[:32]
    
    # Add index to each sample
    for idx, sample in enumerate(data):
        sample["idx"] = idx
    
    print_rank_0(f"Loaded {len(data)} samples from {dataset_name}")
    
    # Distributed: split samples across GPUs
    total_samples = len(data)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    local_indices = list(range(start_idx, end_idx))
    
    if world_size > 1:
        print(f"[Rank {rank}] Evaluating samples {start_idx} to {end_idx-1} ({len(local_indices)} samples)")
    
    # Evaluation parameters
    max_new_tokens = getattr(configs, "max_new_tokens", 1024)
    
    # Local counters
    local_correct = 0
    local_total = len(local_indices)
    local_detailed_results = []
    
    # Progress bar
    pbar = tqdm(local_indices, desc=f"[Rank {rank}] {dataset_name}", disable=(rank != 0))
    ADD_INDICATOR = (configs.eval_stage_mode != "soft_fusion")
    with torch.no_grad():
        for idx in pbar:
            sample = data[idx]
            # Build input using chat template
            message = [{"role": "user", "content": sample["question"]}]
            input_text = apply_chat_template_if_needed(tokenizer, message)
            input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                without_thinking_token=False, # enable latent reasoning at test time
            )
            
            # Decode output
            text_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract answer
            answer_output = extract_answer_from_output(text_output)
            
            # Compare with ground truth
            gt_answer = sample["answer"]
            is_correct = check_answer_correct(answer_output, gt_answer)
            
            if is_correct:
                local_correct += 1
            
            # Store detailed result
            if getattr(configs, "save_detailed_results", True):
                detailed_result = {
                    "sample_idx": idx,
                    "question": sample["question"],
                    "ground_truth_answer": normalize_answer(gt_answer),
                    "predicted_answer": normalize_answer(answer_output),
                    "raw_predicted_answer": answer_output,
                    "is_correct": is_correct,
                    "output_text": text_output,
                    "rank": rank,
                }
                local_detailed_results.append(detailed_result)
            
            # Print some examples (only rank 0, first few samples) - full input/output for inspection
            num_examples_to_print = getattr(configs, "num_examples_to_print", 10)
            if rank == 0 and idx < num_examples_to_print:
                print_rank_0(f"\n{'='*80}")
                print_rank_0(f"[{dataset_name} Sample {idx}] - FULL INPUT/OUTPUT")
                print_rank_0(f"{'='*80}")
                print_rank_0(f"\n>>> QUESTION (raw):\n{sample['question']}")
                print_rank_0(f"\n>>> MODEL INPUT (after chat template):\n{input_text}")
                print_rank_0(f"\n>>> MODEL OUTPUT (full, skip_special_tokens=False):\n{text_output}")
                print_rank_0(f"\n>>> EXTRACTED ANSWER: '{answer_output}'")
                print_rank_0(f">>> GROUND TRUTH: '{gt_answer}' | PREDICTED: '{answer_output}' | CORRECT: {is_correct}")
                print_rank_0(f"{'='*80}\n")
            
            # Update progress bar
            if rank == 0:
                current_acc = local_correct / (idx - start_idx + 1) if (idx - start_idx + 1) > 0 else 0
                pbar.set_description(f"[Rank 0] {dataset_name} Acc: {current_acc:.4f}")
    
    pbar.close()
    
    # Aggregate results across all ranks
    correct_tensor = torch.tensor([local_correct], device=device, dtype=torch.float32)
    total_tensor = torch.tensor([local_total], device=device, dtype=torch.float32)
    
    correct_tensor = all_reduce_sum(correct_tensor)
    total_tensor = all_reduce_sum(total_tensor)
    
    total_correct = int(correct_tensor.item())
    total_evaluated = int(total_tensor.item())
    
    accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0
    
    # Gather detailed results
    all_results = gather_results_from_all_ranks(local_detailed_results)
    if is_main_process():
        all_results.sort(key=lambda x: x["sample_idx"])
    
    result = {
        "dataset_name": dataset_name,
        "total_samples": total_evaluated,
        "correct": total_correct,
        "accuracy": accuracy,
        "detailed_results": all_results if is_main_process() else [],
    }
    
    print_rank_0(f"\n{dataset_name} Results: {total_correct}/{total_evaluated} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return result


def evaluate_all_datasets(configs: Config):
    """Main evaluation function - evaluate on all datasets."""
    # Initialize distributed environment
    rank, world_size, local_rank, device = _init_distributed(getattr(configs, "debug", False))
    
    print_rank_0("=" * 80)
    print_rank_0(f"LT_Tuning Evaluation - All Datasets")
    print_rank_0(f"World size: {world_size}")
    print_rank_0("=" * 80)
    
    if is_main_process():
        print_rank_0(f"Config: {configs.__dict__}")
    
    # Set seed
    set_seed(getattr(configs, "seed", 42))
    
    # Load model and tokenizer (once for all datasets)
    # import pdb; pdb.set_trace()

    model, tokenizer, thinking_token_id = load_model_and_tokenizer(configs, device)
    model.eval()
    
    # Get datasets to evaluate
    datasets_to_eval = getattr(configs, "datasets", list(NAME_REPO_MAPPING.keys()))
    if isinstance(datasets_to_eval, str):
        datasets_to_eval = [datasets_to_eval]
    
    print_rank_0(f"\nDatasets to evaluate: {datasets_to_eval}")
    
    # Evaluate on each dataset
    all_results = {}
    accuracies = []
    
    for dataset_name in datasets_to_eval:
        if dataset_name not in NAME_REPO_MAPPING:
            print_rank_0(f"Warning: Unknown dataset '{dataset_name}', skipping...")
            continue
        
        result = evaluate_single_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            configs=configs,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        
        all_results[dataset_name] = result
        accuracies.append(result["accuracy"])
    
    # Calculate average performance
    if accuracies:
        average_accuracy = sum(accuracies) / len(accuracies)
    else:
        average_accuracy = 0.0
    
    # Print final summary
    print_rank_0("\n" + "=" * 80)
    print_rank_0("FINAL EVALUATION SUMMARY")
    print_rank_0("=" * 80)
    print_rank_0(f"{'Dataset':<20} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print_rank_0("-" * 52)
    
    for dataset_name, result in all_results.items():
        print_rank_0(f"{dataset_name:<20} {result['correct']:>10} {result['total_samples']:>10} {result['accuracy']*100:>11.2f}%")
    
    print_rank_0("-" * 52)
    print_rank_0(f"{'Average':<20} {'':<10} {'':<10} {average_accuracy*100:>11.2f}%")
    print_rank_0("=" * 80)
    
    # Save results
    if is_main_process() and getattr(configs, "save_detailed_results", True):
        output_dir = getattr(configs, "output_dir", 
                           os.path.join(getattr(configs, "save_path", "results"), 
                                       getattr(configs, "name", "eval_lt_tuning")))
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_path": configs.model_path,
            "max_new_tokens": getattr(configs, "max_new_tokens", 1024),
            "datasets_evaluated": list(all_results.keys()),
            "per_dataset_results": {
                name: {
                    "total_samples": r["total_samples"],
                    "correct": r["correct"],
                    "accuracy": r["accuracy"],
                }
                for name, r in all_results.items()
            },
            "average_accuracy": average_accuracy,
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, "eval_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print_rank_0(f"\nSummary saved to: {summary_file}")
        
        # Save detailed results for each dataset
        for dataset_name, result in all_results.items():
            if result.get("detailed_results"):
                detail_file = os.path.join(output_dir, f"eval_{dataset_name.replace('-', '_')}_detailed.json")
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "dataset": dataset_name,
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "correct": result["correct"],
                        "detailed_results": result["detailed_results"],
                    }, f, indent=2, ensure_ascii=False)
                print_rank_0(f"Detailed results for {dataset_name} saved to: {detail_file}")
    
    sys.stdout.flush()
    
    return all_results, average_accuracy


def main():
    parser = argparse.ArgumentParser(description="LT_Tuning Evaluation - Multi-Dataset Benchmark")
    parser.add_argument("config_file", type=str, help="Path to YAML config file")
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="Local rank for distributed training (passed by launcher)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Specific datasets to evaluate (default: all)")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer samples")
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    
    configs = Config(config_dict)
    
    # Override with command line arguments
    if args.datasets:
        configs.datasets = args.datasets
    if args.debug:
        configs.debug = True
    
    # Validate required fields
    if not hasattr(configs, "model_path") or configs.model_path is None:
        # raise ValueError("Config must specify 'model_path' for loading the trained model")
        configs.model_path = os.path.join(configs.save_path, configs.name)
    
    # Set defaults for optional fields
    if not hasattr(configs, "save_path"):
        configs.save_path = "results"
    if not hasattr(configs, "name"):
        configs.name = "eval_lt_tuning"
    if not hasattr(configs, "seed"):
        configs.seed = 42
    if not hasattr(configs, "debug"):
        configs.debug = False
    if not hasattr(configs, "max_new_tokens"):
        configs.max_new_tokens = 1024
    if not hasattr(configs, "save_detailed_results"):
        configs.save_detailed_results = True
    if not hasattr(configs, "datasets"):
        configs.datasets = list(NAME_REPO_MAPPING.keys())
    
    evaluate_all_datasets(configs)


if __name__ == "__main__":
    main()
