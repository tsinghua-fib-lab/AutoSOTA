"""Scalability and Correctness experiment from Section 4.
Compares ProjectedTracInSDI (sketched) against FullGradientTracInSDI (exact baseline)
on a 135.1M Looped GPT-2 with synthetic random data.
Measures: relative SDI error, relative TracIn error, runtime overhead."""

import argparse
import json
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sdi import CheckpointSpec, ProjectedTracInSDI, FullGradientTracInSDI

# Import the model from local file
sys.path.insert(0, "/repo")
from looped_gpt2 import LoopedGPT2


class RandomTokenDataset(Dataset):
    """Synthetic random token sequences."""
    def __init__(self, n, seq_len, vocab_size, seed):
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab_size, (n, seq_len), generator=g)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return {"tokens": self.tokens[idx]}


def collate(batch):
    return {"tokens": torch.stack([b["tokens"] for b in batch], dim=0)}


def per_example_loss(model, batch):
    """Next-token prediction loss, per-example."""
    device = next(model.parameters()).device
    tokens = batch["tokens"].to(device)
    logits = model(tokens)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)
    return per_pos.mean(dim=1)


def relative_error(estimated, exact):
    """||estimated - exact||_F / ||exact||_F"""
    diff = (estimated - exact).norm().item()
    norm_exact = exact.norm().item()
    if norm_exact < 1e-12:
        return float("nan")
    return diff / norm_exact


def main():
    parser = argparse.ArgumentParser(description="Scalability experiment for SDI paper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tau", type=int, default=32)
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--n-train", type=int, default=8, help="Number of train examples")
    parser.add_argument("--n-query", type=int, default=4, help="Number of query examples")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for full gradient")
    parser.add_argument("--sketch-batch-size", type=int, default=4, help="Batch size for sketched")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of independent trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/repo/outputs/scalability_metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("SDI Scalability and Correctness Experiment")
    print("=" * 60)
    print("Device: %s" % device)
    print("GPU: %s" % (torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"))
    print("Settings: vocab=%d, seq_len=%d, tau=%d, sketch_dim=%d" % (
        args.vocab_size, args.seq_len, args.tau, args.sketch_dim))
    print("Data: n_train=%d, n_query=%d" % (args.n_train, args.n_query))
    print("Batch: full_grad=%d, sketched=%d" % (args.batch_size, args.sketch_batch_size))
    print("Trials: %d" % args.n_trials)

    all_results = []
    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        torch.manual_seed(trial_seed)
        print("\n--- Trial %d/%d (seed=%d) ---" % (trial + 1, args.n_trials, trial_seed))

        # Create model
        print("Building model...")
        model = LoopedGPT2(
            vocab_size=args.vocab_size,
            d_model=768,
            n_head=12,
            n_prelude=2,
            n_recurrent=4,
            n_coda=2,
            tau=args.tau,
            seq_len=args.seq_len,
            dropout=0.0,
        )
        model.to(device)
        model.eval()

        # Create synthetic data
        train_ds = RandomTokenDataset(args.n_train, args.seq_len, args.vocab_size, trial_seed)
        query_ds = RandomTokenDataset(args.n_query, args.seq_len, args.vocab_size, trial_seed)

        train_loader_full = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        query_loader_full = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        train_loader_sketch = DataLoader(train_ds, batch_size=args.sketch_batch_size, shuffle=False, collate_fn=collate)
        query_loader_sketch = DataLoader(train_ds, batch_size=args.sketch_batch_size, shuffle=False, collate_fn=collate)

        # Save a "checkpoint" (random init)
        ckpt_path = Path("/tmp/ckpt_trial_%d.pt" % trial)
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        ckpt_spec = CheckpointSpec(path=str(ckpt_path), weight=1.0)

        # Target modules: recurrent core + injection adapter
        target_modules = [model.injection] + list(model.recurrent)

        # --- Full Gradient Baseline ---
        print("Computing full-gradient baseline...")
        t0 = time.time()
        full_est = FullGradientTracInSDI(
            model=model,
            target_modules=target_modules,
            loss_reduction="sum",
            batch_first=True,
            expected_steps=args.tau,
            allow_mixed_steps=False,
            strict=True,
        )
        full_out = full_est.influence_across_checkpoints(
            checkpoints=[ckpt_spec],
            train_loader=train_loader_full,
            query_loader=query_loader_full,
            loss_fn=per_example_loss,
            mode="sdi",
            train_chunk_size=args.batch_size,
            query_chunk_size=args.batch_size,
        )
        full_time = time.time() - t0
        full_est.close()
        print("  Full gradient time: %.2f s" % full_time)

        # Move full results to CPU
        exact_sdi = full_out.sdi.cpu()  # (n_train, n_query, steps)
        exact_tracin = full_out.tracin.cpu()
        del full_out, full_est

        # --- Sketched (Projected) SDI ---
        torch.manual_seed(trial_seed)  # reset seed for deterministic sketch hashes
        print("Computing sketched SDI...")
        t0 = time.time()
        sketch_est = ProjectedTracInSDI(
            model=model,
            target_modules=target_modules,
            projection_size=args.sketch_dim,
            seed=trial_seed,
            loss_reduction="sum",
            batch_first=True,
            expected_steps=args.tau,
            allow_mixed_steps=False,
            strict=True,
        )
        sketch_out = sketch_est.influence_across_checkpoints(
            checkpoints=[ckpt_spec],
            train_loader=train_loader_sketch,
            query_loader=query_loader_sketch,
            loss_fn=per_example_loss,
            mode="sdi",
            train_chunk_size=args.sketch_batch_size,
            query_chunk_size=args.sketch_batch_size,
        )
        sketch_time = time.time() - t0
        sketch_est.close()
        print("  Sketched SDI time: %.2f s" % sketch_time)

        sketch_sdi = sketch_out.sdi.cpu()
        sketch_tracin = sketch_out.tracin.cpu()
        del sketch_out, sketch_est

        # --- Compute relative errors ---
        sdi_err = relative_error(sketch_sdi, exact_sdi)
        tracin_err = relative_error(sketch_tracin, exact_tracin)
        runtime_overhead = sketch_time  # total sketched computation time

        print("  Relative SDI Error:   %.6f" % sdi_err)
        print("  Relative TracIn Error: %.6f" % tracin_err)
        print("  Runtime overhead:      %.3f s" % runtime_overhead)

        all_results.append({
            "trial": trial + 1,
            "seed": trial_seed,
            "relative_sdi_error": sdi_err,
            "relative_tracin_error": tracin_err,
            "runtime_overhead_s": runtime_overhead,
            "full_gradient_time_s": full_time,
            "sketch_time_s": sketch_time,
        })

        # Cleanup
        del model, exact_sdi, exact_tracin, sketch_sdi, sketch_tracin
        torch.cuda.empty_cache()

    # Aggregate
    import math as _math
    sdi_errors = [r["relative_sdi_error"] for r in all_results]
    tracin_errors = [r["relative_tracin_error"] for r in all_results]
    runtimes = [r["runtime_overhead_s"] for r in all_results]

    mean_sdi = sum(sdi_errors) / len(sdi_errors)
    mean_tracin = sum(tracin_errors) / len(tracin_errors)
    mean_runtime = sum(runtimes) / len(runtimes)
    std_sdi = _math.sqrt(sum((x - mean_sdi)**2 for x in sdi_errors) / len(sdi_errors)) if len(sdi_errors) > 1 else 0.0
    std_tracin = _math.sqrt(sum((x - mean_tracin)**2 for x in tracin_errors) / len(tracin_errors)) if len(tracin_errors) > 1 else 0.0
    std_runtime = _math.sqrt(sum((x - mean_runtime)**2 for x in runtimes) / len(runtimes)) if len(runtimes) > 1 else 0.0

    metrics = {
        "relative_sdi_error_mean": round(mean_sdi, 6),
        "relative_sdi_error_std": round(std_sdi, 6),
        "relative_tracin_error_mean": round(mean_tracin, 6),
        "relative_tracin_error_std": round(std_tracin, 6),
        "runtime_overhead_mean_s": round(mean_runtime, 6),
        "runtime_overhead_std_s": round(std_runtime, 6),
        "n_trials": args.n_trials,
        "settings": {
            "model_params": "134.6M (untied weights)",
            "d_model": 768,
            "n_head": 12,
            "n_prelude": 2,
            "n_recurrent": 4,
            "n_coda": 2,
            "tau": args.tau,
            "sketch_dim": args.sketch_dim,
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "n_train": args.n_train,
            "n_query": args.n_query,
            "batch_size_full": args.batch_size,
            "batch_size_sketch": args.sketch_batch_size,
            "precision": "float32",
        },
        "per_trial": all_results,
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Relative SDI Error:   %.6f +/- %.6f" % (mean_sdi, std_sdi))
    print("Relative TracIn Error: %.6f +/- %.6f" % (mean_tracin, std_tracin))
    print("Runtime Overhead:      %.3f +/- %.3f s" % (mean_runtime, std_runtime))
    print("=" * 60)

    # Paper reference values
    print("\nPaper reference values:")
    print("  Relative SDI Error:   0.0388 +/- 0.0030")
    print("  Relative TracIn Error: 0.0220 +/- 0.0052")
    print("  Runtime Overhead:      2.55 +/- 0.002 s")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved to %s" % output_path)

    return metrics


if __name__ == "__main__":
    main()
