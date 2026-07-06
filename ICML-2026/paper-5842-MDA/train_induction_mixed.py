#!/usr/bin/env python3
"""Mixed-data fine-tuning: pure induction + text templates."""
import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from eval_induction_final import compute_induction_score

XML_TEMPLATES = [
    "<doc>\n<item>value_{i}</item>\n</doc>\n<doc>\n<item>value_{i}</item>\n</doc>",
    "<div class=\"row\"><span>data_{i}</span></div>\n<div class=\"row\"><span>data_{i}</span></div>",
]
LATEX_TEMPLATES = [
    "\\begin{{equation}}\nx_{{{i}}} = y_{{{i}}}\n\\end{{equation}}\n\\begin{{equation}}\nx_{{{i}}} = y_{{{i}}}\n\\end{{equation}}",
]
CODE_TEMPLATES = [
    "def func_{i}(x):\n    return x + {i}\n\ndef func_{i}(x):\n    return x + {i}",
]
ALL_TEMPLATES = XML_TEMPLATES + LATEX_TEMPLATES + CODE_TEMPLATES

def generate_mixed_data(n_samples, seq_len, vocab_size, seed, tokenizer, pure_ratio=0.5):
    rng = np.random.default_rng(seed)
    data = []
    half = seq_len // 2
    template_idx = 0
    for i in range(n_samples):
        if rng.random() < pure_ratio:
            # Pure induction: random repeated prefix
            prefix = rng.integers(0, vocab_size, size=half)
            seq = np.concatenate([prefix, prefix])
        else:
            # Text template
            template = ALL_TEMPLATES[template_idx % len(ALL_TEMPLATES)]
            template_idx += 1
            text = template.format(i=int(rng.integers(1, 1000)))
            tokens = tokenizer.encode(text)
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            elif len(tokens) < seq_len:
                tokens = tokens + [tokenizer.pad_token_id or 0] * (seq_len - len(tokens))
            seq = np.array(tokens, dtype=np.int64)[:seq_len]
        data.append(torch.tensor(seq, dtype=torch.long))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/paper_data/pythia-14m-step2000")
    parser.add_argument("--output-dir", default="/repo/outputs/ft_mixed")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pure-ratio", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Config: n_steps={}, lr={}, batch={}, pure_ratio={}".format(args.n_steps, args.lr, args.batch_size, args.pure_ratio))
    os.makedirs(args.output_dir, exist_ok=True)

    OFFICIAL_MODEL_NAMES.append(args.model_path)
    MODEL_ALIASES[args.model_path] = ["local-model"]
    make_model_alias_map()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained_no_processing(args.model_path, dtype=torch.float32, tokenizer=tokenizer, device=device)
    model.cfg.use_attn_result = True
    model.train()

    model.eval()
    print("\n=== Baseline ===")
    baseline = compute_induction_score(model, args.layer, args.head, n_sequences=args.n_eval_seqs, seed=12345, n_seeds=3)
    print("  Baseline: {:.6f}".format(baseline["induction_score"]))
    model.train()

    data = generate_mixed_data(args.n_samples, args.seq_len, int(model.cfg.d_vocab), args.seed, tokenizer, args.pure_ratio)
    inputs = torch.stack([d[:-1] for d in data])
    targets = torch.stack([d[1:] for d in data])
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.n_steps)

    global_step = 0
    total_loss = 0.0
    data_iter = iter(dataloader)
    print("\n=== Training: {} steps ===".format(args.n_steps))
    started = time.time()

    while global_step < args.n_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        global_step += 1
        if global_step % 20 == 0 or global_step == args.n_steps:
            print("  Step {}/{}: loss={:.4f}, lr={:.2e}".format(global_step, args.n_steps, total_loss/global_step, scheduler.get_last_lr()[0]))

    model.eval()
    final_score = compute_induction_score(model, args.layer, args.head, n_sequences=args.n_eval_seqs, seed=12345, n_seeds=5)
    print("\n=== Final (5-seed) ===")
    print("  Baseline: {:.6f}".format(baseline["induction_score"]))
    print("  Final:    {:.6f}".format(final_score["induction_score"]))
    print("  Delta:    {:+.6f} ({:+.2f}%)".format(final_score["induction_score"] - baseline["induction_score"], 100 * (final_score["induction_score"] - baseline["induction_score"]) / baseline["induction_score"]))
    print("  Elapsed:  {:.1f}s".format(time.time() - started))

if __name__ == "__main__":
    main()
