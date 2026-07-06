#!/usr/bin/env python3
"""
Fine-tune Pythia-14M to strengthen induction heads using synthetic data
with strong structural repetition patterns.

Strategy:
- Load pythia-14m-step2000 (peak induction score checkpoint)
- Generate synthetic sequences with diverse repetition patterns
- Fine-tune with causal LM loss for a small number of steps
- Save checkpoints and evaluate induction score

Patterns based on paper findings: XML, LaTeX, code, and character repetition
are the most effective for accelerating induction head formation.
"""

import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse, math
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
)

# Import our eval function
from eval_induction_final import compute_induction_score


# Synthetic data templates with strong structural repetition
# Each template creates "A B A B" style patterns that reward induction
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
    """Streaming dataset of synthetic sequences."""
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
    """Pad batch to max length and create labels."""
    max_len = max(len(s) for s in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = True
    labels = padded.clone()
    labels[~mask] = -100
    return {"input_ids": padded, "labels": labels, "attention_mask": mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/paper_data/pythia-14m-step2000")
    parser.add_argument("--output-dir", default="/repo/outputs/finetune")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Config: n_samples={}, seq_len={}, batch_size={}, n_steps={}, lr={}".format(
        args.n_samples, args.seq_len, args.batch_size, args.n_steps, args.lr))

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

    # Evaluate baseline induction score
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
    total_loss = 0.0
    data_iter = iter(dataloader)

    print("\n=== Training for {} steps ===".format(args.n_steps))
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

        # Causal LM loss via HookedTransformer
        loss = model(input_ids, return_type="loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % 10 == 0:
            avg_loss = total_loss / global_step
            lr_now = scheduler.get_last_lr()[0]
            print("  Step {}/{}: loss={:.4f}, lr={:.2e}".format(
                global_step, args.n_steps, avg_loss, lr_now))

        # Evaluate periodically
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

            # Save checkpoint
            ckpt_dir = os.path.join(args.output_dir, "step_{}".format(global_step))
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save as HuggingFace format
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(ckpt_dir)
            # Config contains non-serializable objects (torch.device); skip

            results["checkpoints"].append({
                "step": global_step,
                "induction_score": score["induction_score"],
                "induction_score_std": score["induction_score_std"],
                "loss": total_loss / global_step,
            })

            model.train()

    elapsed = time.time() - started
    print("\nTraining complete: {:.1f}s".format(elapsed))

    # Final evaluation
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

    # Save final checkpoint
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(final_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_dir)
    # Config contains non-serializable objects (torch.device); skip

    print("\nSaved to {}".format(args.output_dir))
    return results


if __name__ == "__main__":
    main()
