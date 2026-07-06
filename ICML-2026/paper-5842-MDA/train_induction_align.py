#!/usr/bin/env python3
"""
Fine-tune Pythia-14M with attention map alignment regularization (IDEA-03).

Adds an auxiliary KL-divergence loss between L3H3's attention map and the
ideal induction detection pattern. This explicitly shapes attention toward
induction behavior during fine-tuning.

L_total = L_LM + lambda * KL(attn_L3H3 || induction_template)

The induction template is derived from the transformer_lens detection pattern
for each sequence, representing the "ideal" induction head attention.
"""

import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse, math
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
from model.hooks import PatternCache, setup_pattern_hooks

from eval_induction_final import compute_induction_score


# Synthetic data templates with strong structural repetition
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
    "for i in range({i}):\n    print(f\"Iteration {{i}}\")\nfor i in range({i}):\n    print(f\"Iteration {{i}}\")",
    "class MyClass{i}:\n    def __init__(self):\n        self.val = {i}\n\nclass MyClass{i}:\n    def __init__(self):\n        self.val = {i}",
]

CHAR_REP_TEMPLATES = [
    "AAAA BBBB CCCC DDDD AAAA BBBB CCCC DDDD",
    "XY XY XY XY ZW ZW ZW ZW XY XY XY XY ZW ZW ZW ZW",
]

ALL_TEMPLATES = XML_TEMPLATES + LATEX_TEMPLATES + CODE_TEMPLATES + CHAR_REP_TEMPLATES


def generate_synthetic_sequences(tokenizer, n_samples, seq_len, seed):
    """Generate synthetic sequences with diverse structural repetition patterns."""
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


def compute_induction_alignment_loss(model, input_ids, layer, head, lambda_reg):
    """
    Compute KL divergence between L3H3 attention and the induction detection pattern.

    The detection pattern marks positions where token[q] == token[k-1],
    normalized to form a target probability distribution.

    Returns: lambda_reg * mean_kl_divergence across the batch
    """
    inner = model.module if hasattr(model, "module") else model
    batch_size = input_ids.shape[0]

    total_kl = 0.0
    valid_count = 0

    for b in range(batch_size):
        tokens = input_ids[b:b+1]
        seq_len = tokens.shape[1]

        # Get induction detection pattern
        det_pattern = get_induction_head_detection_pattern(tokens[0].cpu())
        if det_pattern.sum() == 0:
            continue
        det_pattern = det_pattern.to(tokens.device)

        # Get attention pattern for the target head
        pat_cache = PatternCache()
        hooks = setup_pattern_hooks(inner, layer, pat_cache)

        try:
            with torch.no_grad():
                _ = model(tokens)

            hp = pat_cache.pattern[0, head]  # [seq_len, seq_len]

            # Normalize both to probability distributions
            hp_prob = hp / (hp.sum() + 1e-10)
            dp_prob = det_pattern / (det_pattern.sum() + 1e-10)

            # KL divergence: KL(dp || hp) = sum(dp * log(dp/hp))
            kl = (dp_prob * torch.log((dp_prob + 1e-10) / (hp_prob + 1e-10))).sum()
            total_kl += kl.item()
            valid_count += 1
        finally:
            try:
                inner.reset_hooks(hooks)
            except Exception:
                pass

    if valid_count > 0:
        return lambda_reg * (total_kl / valid_count)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/paper_data/pythia-14m-step2000")
    parser.add_argument("--output-dir", default="/repo/outputs/finetune_align")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda-reg", type=float, default=0.05,
                        help="Weight of induction alignment loss")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Config: n_samples={}, seq_len={}, batch_size={}, n_steps={}, lr={}, lambda={}".format(
        args.n_samples, args.seq_len, args.batch_size, args.n_steps, args.lr, args.lambda_reg))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
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
    model.cfg.use_attn_result = False
    model.train()

    # Evaluate baseline
    model.eval()
    print("\n=== Baseline Induction Score ===")
    baseline = compute_induction_score(
        model, args.layer, args.head,
        n_sequences=args.n_eval_seqs,
        seed=12345,
        n_seeds=3,
    )
    print("  Baseline: {:.6f} +/- {:.6f}".format(
        baseline["induction_score"], baseline["induction_score_std"]))
    model.train()

    # Create dataset
    dataset = SyntheticDataset(
        tokenizer, args.n_samples, args.seq_len, args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.n_steps,
    )

    # Training loop
    results = {"baseline": baseline["induction_score"], "checkpoints": []}
    global_step = 0
    total_lm_loss = 0.0
    total_align_loss = 0.0
    data_iter = iter(dataloader)

    print("\n=== Training for {} steps with alignment loss (lambda={}) ===".format(
        args.n_steps, args.lambda_reg))
    started = time.time()

    while global_step < args.n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # LM loss
        lm_loss = model(input_ids, return_type="loss")

        # Alignment loss: KL divergence between L3H3 attention and induction template
        align_loss = compute_induction_alignment_loss(
            model, input_ids, args.layer, args.head, args.lambda_reg)

        total_loss = lm_loss + align_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_lm_loss += lm_loss.item()
        total_align_loss += align_loss if isinstance(align_loss, float) else 0.0
        global_step += 1

        if global_step % 10 == 0:
            avg_lm = total_lm_loss / global_step
            avg_align = total_align_loss / global_step
            lr_now = scheduler.get_last_lr()[0]
            print("  Step {}/{}: lm_loss={:.4f}, align={:.4f}, lr={:.2e}".format(
                global_step, args.n_steps, avg_lm, avg_align, lr_now))

        if global_step % args.eval_every == 0:
            model.eval()
            score = compute_induction_score(
                model, args.layer, args.head,
                n_sequences=args.n_eval_seqs,
                seed=12345,
                n_seeds=3,
            )
            delta = score["induction_score"] - baseline["induction_score"]
            print("  >>> Step {}: Induction Score = {:.6f} (baseline: {:.6f}, delta: {:+.6f})".format(
                global_step, score["induction_score"], baseline["induction_score"], delta))

            ckpt_dir = os.path.join(args.output_dir, "step_{}".format(global_step))
            os.makedirs(ckpt_dir, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(ckpt_dir)

            results["checkpoints"].append({
                "step": global_step,
                "induction_score": score["induction_score"],
                "induction_score_std": score["induction_score_std"],
                "lm_loss": total_lm_loss / global_step,
                "align_loss": total_align_loss / global_step,
            })

            model.train()

    elapsed = time.time() - started
    print("\nTraining complete: {:.1f}s".format(elapsed))

    model.eval()
    final_score = compute_induction_score(
        model, args.layer, args.head,
        n_sequences=args.n_eval_seqs,
        seed=12345,
        n_seeds=5,
    )
    print("\n=== Final Induction Score ===")
    print("  Baseline: {:.6f}".format(baseline["induction_score"]))
    print("  Final:    {:.6f}".format(final_score["induction_score"]))
    print("  Delta:    {:+.6f}".format(
        final_score["induction_score"] - baseline["induction_score"]))

    results["final"] = final_score["induction_score"]
    results["final_std"] = final_score["induction_score_std"]
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(final_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_dir)

    print("\nSaved to {}".format(args.output_dir))
    return results


if __name__ == "__main__":
    main()
