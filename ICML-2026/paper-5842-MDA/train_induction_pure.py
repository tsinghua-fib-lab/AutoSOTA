#!/usr/bin/env python3
"""Pure induction-pattern fine-tuning: train on repeated token sequences."""
import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
from eval_induction_final import compute_induction_score

def generate_pure_induction_data(n_samples, seq_len, vocab_size, seed):
    """Generate repeated-prefix sequences: [A0..A(N/2-1), A0..A(N/2-1)]"""
    rng = np.random.default_rng(seed)
    half = seq_len // 2
    data = []
    for _ in range(n_samples):
        prefix = rng.integers(0, vocab_size, size=half)
        seq = np.concatenate([prefix, prefix])
        data.append(torch.tensor(seq, dtype=torch.long))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/paper_data/pythia-14m-step2000")
    parser.add_argument("--output-dir", default="/repo/outputs/ft_pure")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-eval-seqs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Config: n_steps={}, lr={}, batch={}, seq_len={}".format(args.n_steps, args.lr, args.batch_size, args.seq_len))
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
    print("  Baseline: {:.6f} +/- {:.6f}".format(baseline["induction_score"], baseline["induction_score_std"]))
    model.train()

    # Generate pure induction data
    data = generate_pure_induction_data(args.n_samples, args.seq_len, int(model.cfg.d_vocab), args.seed)
    # Shift for causal LM: input = tokens[:-1], target = tokens[1:]
    inputs = torch.stack([d[:-1] for d in data])
    targets = torch.stack([d[1:] for d in data])
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.n_steps)

    results = {"baseline": baseline["induction_score"], "checkpoints": []}
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

        # Forward through HookedTransformer
        logits = model(x)
        # Causal LM loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % 2 == 0 or global_step == args.n_steps:
            lr_now = scheduler.get_last_lr()[0]
            print("  Step {}/{}: loss={:.4f}, lr={:.2e}".format(global_step, args.n_steps, total_loss/global_step, lr_now))

        if global_step % args.eval_every == 0 or global_step == args.n_steps:
            model.eval()
            score = compute_induction_score(model, args.layer, args.head, n_sequences=args.n_eval_seqs, seed=12345, n_seeds=3)
            delta = score["induction_score"] - baseline["induction_score"]
            mark = " *** NEW BEST" if score["induction_score"] > baseline["induction_score"] else ""
            print("  >>> Step {}: Induction Score = {:.6f} (delta: {:+.6f}){}".format(global_step, score["induction_score"], delta, mark))
            model.train()

    elapsed = time.time() - started
    print("\nTraining complete: {:.1f}s".format(elapsed))

    model.eval()
    final_score = compute_induction_score(model, args.layer, args.head, n_sequences=args.n_eval_seqs, seed=12345, n_seeds=5)
    print("\n=== Final (5-seed) ===")
    print("  Baseline: {:.6f}".format(baseline["induction_score"]))
    print("  Final:    {:.6f}".format(final_score["induction_score"]))
    print("  Delta:    {:+.6f} ({:+.2f}%)".format(final_score["induction_score"] - baseline["induction_score"], 100 * (final_score["induction_score"] - baseline["induction_score"]) / baseline["induction_score"]))

    results["final"] = final_score["induction_score"]
    results["final_std"] = final_score["induction_score_std"]
    results["elapsed_seconds"] = round(elapsed, 1)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to {}".format(args.output_dir))

if __name__ == "__main__":
    main()
