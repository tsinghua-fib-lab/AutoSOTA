#!/usr/bin/env python3
"""
Fine-tune Pythia-14M with configurable attention alignment regularization.

V3: Fixed alignment loss to properly backpropagate gradients through attention.

L_total = L_LM + lambda * KL(attn_L3H3 || induction_template)

The induction template is derived from the transformer_lens detection pattern
for each sequence, representing the "ideal" induction head attention.
"""

import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
)
from transformer_lens.head_detector import get_induction_head_detection_pattern

from eval_induction_final import compute_induction_score


# Synthetic data templates
XML_TEMPLATES = [
    "<doc>\n<item>value_{i}</item>\n</doc>\n<doc>\n<item>value_{i}</item>\n</doc>",
    '<div class="row"><span>data_{i}</span></div>\n<div class="row"><span>data_{i}</span></div>',
    '<entry id="{i}"><name>item_{i}</name></entry>\n<entry id="{i}"><name>item_{i}</name></entry>',
]

LATEX_TEMPLATES = [
    "\\begin{{equation}}\nx_{{{i}}} = y_{{{i}}}\n\\end{{equation}}\n\\begin{{equation}}\nx_{{{i}}} = y_{{{i}}}\n\\end{{equation}}",
    "\\section{{Section {i}}}\nContent for section {i}.\n\\section{{Section {i}}}\nContent for section {i}.",
    "\\textbf{{term_{i}}} is defined as \\textit{{definition_{i}}}.\n\\textbf{{term_{i}}} is defined as \\textit{{definition_{i}}}.",
]

CODE_TEMPLATES = [
    "def func_{i}(x):\n    return x + {i}\n\ndef func_{i}(x):\n    return x + {i}",
    "for i in range({i}):\n    print(\"Iteration {i}\")\nfor i in range({i}):\n    print(\"Iteration {i}\")",
    "class MyClass{i}:\n    def __init__(self):\n        self.val = {i}\n\nclass MyClass{i}:\n    def __init__(self):\n        self.val = {i}",
]

CHAR_REP_TEMPLATES = [
    "AAAA BBBB CCCC DDDD AAAA BBBB CCCC DDDD",
    "XY XY XY XY ZW ZW ZW ZW XY XY XY XY ZW ZW ZW ZW",
]

ALL_TEMPLATES = XML_TEMPLATES + LATEX_TEMPLATES + CODE_TEMPLATES + CHAR_REP_TEMPLATES


def generate_synthetic_sequences(tokenizer, n_samples, seq_len, seed):
    rng = np.random.default_rng(seed)
    samples = []
    template_idx = 0
    while len(samples) < n_samples:
        template = ALL_TEMPLATES[template_idx % len(ALL_TEMPLATES)]
        i_val = int(rng.integers(1, 1000))
        text = template.format(i=i_val)
        tokens = tokenizer.encode(text)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            repeat_unit = tokens[:max(1, len(tokens) // 2)]
            while len(tokens) < seq_len:
                needed = seq_len - len(tokens)
                tokens.extend(repeat_unit[:needed])
        samples.append(torch.tensor(tokens, dtype=torch.long))
        template_idx += 1
    return samples


class SyntheticDataset(IterableDataset):
    def __init__(self, tokenizer, n_samples, seq_len, seed):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        samples = generate_synthetic_sequences(
            self.tokenizer, self.n_samples, self.seq_len, self.seed)
        for s in samples:
            yield s


def collate_batch(batch, pad_token_id):
    max_len = max(len(s) for s in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = True
    labels = padded.clone()
    labels[~mask] = -100
    return {"input_ids": padded, "labels": labels, "attention_mask": mask}


def compute_combined_loss(model, input_ids, layer, head, lambda_reg):
    """
    Compute LM loss + alignment loss in a single forward pass.
    The alignment loss uses attention patterns captured during the LM forward pass.

    Returns: (total_loss, lm_loss_value, align_loss_value)
    """
    inner = model.module if hasattr(model, "module") else model

    # Forward pass for LM loss (this also populates hook points with attention patterns)
    lm_loss = model(input_ids, return_type="loss")

    # Extract attention pattern from hook point
    # HookedTransformer stores patterns at blocks[l].attn.hook_pattern
    attn_pattern = inner.blocks[layer].attn.hook_pattern  # [batch, n_heads, seq, seq]
    head_pattern = attn_pattern[0, head]  # [seq_len, seq_len] - first batch item

    # Get induction detection pattern
    det_pattern = get_induction_head_detection_pattern(input_ids[0].cpu())
    det_pattern = det_pattern.to(input_ids.device)

    # Compute KL divergence: KL(det_pattern || head_attention)
    # Normalize both to probability distributions
    hp_prob = head_pattern / (head_pattern.sum() + 1e-10)
    dp_prob = det_pattern / (det_pattern.sum() + 1e-10)

    # KL(dp || hp) = sum(dp * log(dp/hp))
    kl_div = (dp_prob * torch.log((dp_prob + 1e-10) / (hp_prob + 1e-10))).sum()

    align_loss = lambda_reg * kl_div
    total_loss = lm_loss + align_loss

    return total_loss, lm_loss.item(), align_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/paper_data/pythia-14m-step2000")
    parser.add_argument("--output-dir", default="/repo/outputs/finetune_v3")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda-reg", type=float, default=0.0,
                        help="Weight of induction alignment loss (0 = LM-only)")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm_label = "LM+align" if args.lambda_reg > 0 else "LM-only"
    print("Device: {}".format(device))
    print("Config: mode={}, n_steps={}, lr={}, lambda={}, batch={}".format(
        lm_label, args.n_steps, args.lr, args.lambda_reg, args.batch_size))

    os.makedirs(args.output_dir, exist_ok=True)

    OFFICIAL_MODEL_NAMES.append(args.model_path)
    MODEL_ALIASES[args.model_path] = ["local-model"]
    make_model_alias_map()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=torch.float32,
        tokenizer=tokenizer,
        device=device,
    )
    # Enable attention result caching so hook_pattern is populated
    model.cfg.use_attn_result = True
    model.train()

    model.eval()
    print("\n=== Baseline Induction Score ===")
    baseline = compute_induction_score(
        model, args.layer, args.head,
        n_sequences=args.n_eval_seqs, seed=12345, n_seeds=3)
    print("  Baseline: {:.6f} +/- {:.6f}".format(
        baseline["induction_score"], baseline["induction_score_std"]))
    model.train()

    dataset = SyntheticDataset(tokenizer, args.n_samples, args.seq_len, args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.n_steps)

    results = {"baseline": baseline["induction_score"], "checkpoints": []}
    global_step = 0
    total_lm = 0.0
    total_align = 0.0
    data_iter = iter(dataloader)
    best_score = baseline["induction_score"]
    best_step = 0

    print("\n=== Training: {} steps, {} ===".format(args.n_steps, lm_label))
    started = time.time()

    while global_step < args.n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)

        optimizer.zero_grad()

        if args.lambda_reg > 0:
            total_loss, lm_val, align_val = compute_combined_loss(
                model, input_ids, args.layer, args.head, args.lambda_reg)
            total_align += align_val
        else:
            total_loss = model(input_ids, return_type="loss")
            lm_val = total_loss.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_lm += lm_val
        global_step += 1

        if global_step % 5 == 0 or global_step == args.n_steps:
            avg_lm = total_lm / global_step
            lr_now = scheduler.get_last_lr()[0]
            align_str = ", align={:.4f}".format(total_align / global_step) if args.lambda_reg > 0 else ""
            print("  Step {}/{}: lm_loss={:.4f}{}, lr={:.2e}".format(
                global_step, args.n_steps, avg_lm, align_str, lr_now))

        if global_step % args.eval_every == 0 or global_step == args.n_steps:
            model.eval()
            score = compute_induction_score(
                model, args.layer, args.head,
                n_sequences=args.n_eval_seqs, seed=12345, n_seeds=3)
            delta = score["induction_score"] - baseline["induction_score"]
            mark = " *** NEW BEST" if score["induction_score"] > best_score else ""
            print("  >>> Step {}: Induction Score = {:.6f} (delta: {:+.6f}){}".format(
                global_step, score["induction_score"], delta, mark))

            if score["induction_score"] > best_score:
                best_score = score["induction_score"]
                best_step = global_step

            ckpt_dir = os.path.join(args.output_dir, "step_{}".format(global_step))
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(ckpt_dir)

            results["checkpoints"].append({
                "step": global_step, "induction_score": score["induction_score"],
                "induction_score_std": score["induction_score_std"],
                "lm_loss": total_lm / global_step,
            })
            model.train()

    elapsed = time.time() - started
    print("\nTraining complete: {:.1f}s".format(elapsed))
    print("Best: {:.6f} at step {}".format(best_score, best_step))

    # Final 5-seed eval
    model.eval()
    final_score = compute_induction_score(
        model, args.layer, args.head,
        n_sequences=args.n_eval_seqs, seed=12345, n_seeds=5)
    print("\n=== Final (5-seed) ===")
    print("  Baseline: {:.6f}".format(baseline["induction_score"]))
    print("  Final:    {:.6f}".format(final_score["induction_score"]))
    print("  Delta:    {:+.6f} ({:+.2f}%)".format(
        final_score["induction_score"] - baseline["induction_score"],
        100 * (final_score["induction_score"] - baseline["induction_score"]) / baseline["induction_score"]))

    results["final"] = final_score["induction_score"]
    results["final_std"] = final_score["induction_score_std"]
    results["best"] = best_score
    results["best_step"] = best_step
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to {}".format(args.output_dir))
    return results


if __name__ == "__main__":
    main()
