"""
Steered evaluation on MATH500 using HuggingFace generate() with baukit hooks.
Applies DynaSteer steering vectors during inference.
"""
import os
import sys
import json
import pickle as pkl
import torch
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset
from tqdm import tqdm

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("Please install math-verify: pip install math-verify")

from baukit import TraceDict

INSTRUCTION = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: \boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


def compute_score(model_output: str, ground_truth: str) -> bool:
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        return ret_score
    except BaseException as e:
        print(f"[Warning] math_verify failed: {repr(e)}")
        return 0.0


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Compute Shannon entropy from logits."""
    logits_f = logits.to(torch.float32)
    log_pd = torch.nn.functional.log_softmax(logits_f, dim=-1)
    pd = torch.exp(log_pd)
    ent = -torch.sum(pd * log_pd, dim=-1)
    return ent.item()


def load_steering_configs(pkl_path: str):
    """Load steering configs from solve_steering.py output."""
    with open(pkl_path, 'rb') as f:
        configs = pkl.load(f)
    return configs


def build_hook_map(steering_configs, model_config, alpha=1.0, top_k=10):
    """
    Build a mapping from hook name to steering vector info.
    Returns: dict of hook_name -> list of steer_info dicts
    Each steer_info: {'head_idx': int or None, 'vector': numpy array, 'probe_acc': float}
    For attention: vector is 128-dim (head_dim), applied to specific head position
    For mlp/layer: vector is 2048-dim (hidden_size), applied to full output
    """
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // n_heads

    all_steer_info = []

    for act_type in ['attention', 'mlp', 'layer']:
        if act_type not in steering_configs:
            continue
        for unit_cfg in steering_configs[act_type]:
            layer = unit_cfg['layer']
            head = unit_cfg.get('head')

            # Use the best cluster's w_lda
            if unit_cfg.get('clusters'):
                best_cluster = max(unit_cfg['clusters'], key=lambda c: c.get('probe_acc', 0))
                w_lda = np.array(best_cluster['w_lda'])
                w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-8)

                all_steer_info.append({
                    'act_type': act_type,
                    'layer': layer,
                    'head': head,
                    'vector': w_lda * alpha,
                    'probe_acc': unit_cfg.get('probe_acc', 0),
                    'lda_fdr': unit_cfg.get('lda_fdr', 0),
                })

    # Sort by probe_acc and take top units
    all_steer_info.sort(key=lambda x: x['probe_acc'], reverse=True)

    # Build hook map with top units
    hook_map = {}
    for info in all_steer_info[:top_k]:
        act_type = info['act_type']
        layer = info['layer']
        head = info['head']
        vec = info['vector']

        if act_type == 'attention':
            hook_name = f"model.layers.{layer}.self_attn.attn_retain"
            # Expand per-head vector to full hidden_size
            full_vec = np.zeros(model_config.hidden_size, dtype=vec.dtype)
            full_vec[head * head_dim : (head + 1) * head_dim] = vec
            if hook_name not in hook_map:
                hook_map[hook_name] = full_vec
            else:
                hook_map[hook_name] += full_vec  # combine multiple heads
        elif act_type == 'mlp':
            hook_name = f"model.layers.{layer}.mlp"
            if hook_name not in hook_map:
                hook_map[hook_name] = vec
            else:
                hook_map[hook_name] += vec
        elif act_type == 'layer':
            hook_name = f"model.layers.{layer}.layer_retain"
            if hook_name not in hook_map:
                hook_map[hook_name] = vec
            else:
                hook_map[hook_name] += vec

    # Re-normalize combined vectors
    for hook_name in hook_map:
        hook_map[hook_name] = hook_map[hook_name] / (np.linalg.norm(hook_map[hook_name]) + 1e-8) * alpha

    print(f"Top steering units (top {top_k} by probe_acc):")
    for info in all_steer_info[:10]:
        head_str = f"_H{info['head']}" if info['head'] is not None else ""
        print(f"  {info['act_type']}_L{info['layer']}{head_str}: probe_acc={info['probe_acc']:.4f}, lda_fdr={info['lda_fdr']:.4f}")

    return hook_map


def make_steer_edit_function(hook_map, device, dtype):
    """Create edit function for baukit TraceDict that applies steering."""
    # Pre-convert vectors to tensors
    tensor_map = {}
    for hook_name, vec in hook_map.items():
        tensor_map[hook_name] = torch.tensor(vec, device=device, dtype=dtype)

    def edit_fn(name, output):
        if name in tensor_map:
            steer_vec = tensor_map[name]
            # output shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            # Steer only the last position
            if output.dim() == 3:
                # (batch, seq_len, hidden)
                try:
                    output = output.clone()
                    output[:, -1, :] += steer_vec.to(output.dtype)
                except Exception as e:
                    print(f"[DEBUG] Hook {name}: output shape={output.shape}, vec shape={steer_vec.shape}, error={e}")
                    raise
            elif output.dim() == 2:
                # (batch, hidden)
                try:
                    output = output.clone()
                    output += steer_vec.to(output.dtype)
                except Exception as e:
                    print(f"[DEBUG] Hook {name}: output shape={output.shape}, vec shape={steer_vec.shape}, error={e}")
                    raise
        return output

    return edit_fn


def generate_with_steering(model, tokenizer, prompt_text, hook_map,
                           max_new_tokens=2048, temperature=0.6, top_p=0.95, top_k=20,
                           entropy_threshold=None, steer_alpha=1.0, steering_decay=0.0):
    """
    Generate text with steering applied.
    Uses a simple custom generation loop.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Build hook names
    hook_names = list(hook_map.keys())

    if not hook_names:
        # No steering - fall back to normal generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p if temperature > 0 else 1.0,
                top_k=top_k if temperature > 0 else 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        return generated

    # Create edit function
    edit_fn = make_steer_edit_function(hook_map, device, dtype)

    # Custom generation loop with steering
    generated_ids = []
    past_key_values = None
    current_ids = input_ids
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id

    # Pre-compute step-dependent alpha multipliers for steering decay
    import math
    step_scales = [1.0] * max_new_tokens
    if steering_decay > 0:
        for s in range(max_new_tokens):
            progress = s / max(1, max_new_tokens - 1)
            step_scales[s] = max(0.3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Apply step-dependent decay to steering vectors
            if steering_decay > 0 and hook_names:
                scale = step_scales[step]
                decayed_hook_map = {}
                for name, vec in hook_map.items():
                    decayed_hook_map[name] = vec * (1.0 - steering_decay * (1.0 - scale))
                edit_fn = make_steer_edit_function(decayed_hook_map, device, dtype)

            # Run forward pass with steering hooks
            with TraceDict(model, hook_names, edit_output=edit_fn, retain_input=False, clone=False) as td:
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_idx, top_k_vals)

            # Apply top-p
            if top_p < 1.0 and temperature > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_ids.append(next_token.item())

            # Check for EOS
            if next_token.item() == eos_token_id:
                break

            # Update for next step
            past_key_values = outputs.past_key_values
            current_ids = next_token

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/models/Qwen3-1.7B")
    parser.add_argument("--steering_config", type=str, default=None,
                       help="Path to steering_configs.pkl from solve_steering.py")
    parser.add_argument("--output_dir", type=str, default="/repo/results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--steer_alpha", type=float, default=1.0,
                       help="Steering coefficient alpha")
    parser.add_argument("--steering_decay", type=float, default=0.0,
                       help="Steering strength decay factor (0=none, 0.02=mild)")
    parser.add_argument("--top_units", type=int, default=10,
                       help="Number of top steering units to use")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_steer", action="store_true", default=False,
                       help="Run without steering (baseline)")
    args = parser.parse_args()

    set_seed(42)

    print(f"Loading model from {args.model}...")
    from custom_hf.modeling_qwen3 import CustomQwen3ForCausalLM

    model = CustomQwen3ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    # Load steering configs
    hook_map = {}
    if args.steering_config and not args.no_steer:
        print(f"Loading steering configs from {args.steering_config}...")
        steer_configs = load_steering_configs(args.steering_config)
        hook_map = build_hook_map(steer_configs, model.config, alpha=args.steer_alpha, top_k=args.top_units)
        print(f"Loaded {len(hook_map)} steering hooks")
        for name in sorted(hook_map.keys()):
            print(f"  {name}")

    # Load dataset
    print("Loading MATH500...")
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Dataset size: {len(ds)}")

    model_name_label = args.model.split("/")[-1]
    steer_label = "steered" if hook_map else "baseline"
    output_dir = os.path.join(args.output_dir, model_name_label, f"steer_eval_{steer_label}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    correct = 0

    for i, example in enumerate(tqdm(ds, desc="Evaluating")):
        problem = example["problem"]
        ground_truth = example["answer"]

        # Build prompt
        if "Qwen3" in args.model:
            messages = [{"role": "user", "content": INSTRUCTION.format(problem=problem)}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            messages = [{"role": "user", "content": INSTRUCTION.format(problem=problem)}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        try:
            generated = generate_with_steering(
                model, tokenizer, prompt_text, hook_map,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                steer_alpha=args.steer_alpha,
                steering_decay=args.steering_decay,
            )
        except Exception as e:
            print(f"\n[Error] Generation failed for problem {i}: {e}")
            generated = ""

        score = compute_score(model_output=generated[-500:], ground_truth=ground_truth)
        if score:
            correct += 1

        results.append({
            "problem_id": i,
            "problem": problem,
            "model_pred": generated,
            "ground_truth": ground_truth,
            "score": float(score),
        })

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            print(f"\n  Progress: {i+1}/{len(ds)}, Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    acc = correct / len(results)
    print(f"\n=== STEERED EVALUATION RESULTS ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Correct: {correct}/{len(results)}")
    print(f"Steering: {'enabled' if hook_map else 'disabled'}")
    if hook_map:
        print(f"Alpha: {args.steer_alpha}")

    output_file = os.path.join(output_dir, f"results_steered_a{args.steer_alpha}_t{args.temperature}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    summary = {
        "model": args.model,
        "dataset": "MATH500",
        "steering": bool(hook_map),
        "alpha": args.steer_alpha if hook_map else 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "accuracy": acc,
        "correct": correct,
        "total": len(results),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
