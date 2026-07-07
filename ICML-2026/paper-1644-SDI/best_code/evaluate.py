"""Evaluation script for SDI paper (1644): Scalability and Correctness experiment.
Reproduces: Relative SDI Error, Relative TracIn Error, Runtime Overhead.
Usage: python3 evaluate.py --device cuda:0 --n-trials 10
"""

import argparse, json, math, sys, time
from pathlib import Path

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/repo")
from looped_gpt2 import LoopedGPT2
from sdi import CheckpointSpec, ProjectedTracInSDI, FullGradientTracInSDI


class RandomTokenDataset(Dataset):
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
    device = next(model.parameters()).device
    tokens = batch["tokens"].to(device)
    logits = model(tokens)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(tokens.size(0), -1)
    return per_pos.mean(dim=1)


def relative_error(estimated, exact):
    diff = (estimated - exact).norm().item()
    norm_exact = exact.norm().item()
    return diff / norm_exact if norm_exact > 1e-12 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=4096, help="Sketch dimension (default 4096, paper baseline 2048)")
    parser.add_argument("--output", default="/repo/outputs/metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    N_TRAIN, N_QUERY, TAU, M, SEQ_LEN, VOCAB = 4, 4, 32, args.m, 128, 50304

    print("SDI Reproduction - Paper 1644")
    print("Model: Looped GPT-2, 134.6M params, tau=%d, m=%d" % (TAU, M))
    print("GPU: %s" % (torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"))

    all_results = []
    for trial in range(args.n_trials):
        # ALGO-2: Antithetic paired sampling — share data within pair, complement sketch seeds
        pair_idx = trial // 2
        is_second = trial % 2
        data_seed = args.seed + pair_idx * 1000
        sketch_seed = data_seed + is_second * (2**31)
        torch.manual_seed(sketch_seed)
        print("\nTrial %d/%d (pair=%d, data_seed=%d, sketch_seed=%d)" % (trial + 1, args.n_trials, pair_idx, data_seed, sketch_seed))

        model = LoopedGPT2(vocab_size=VOCAB, d_model=768, n_head=12, n_prelude=2,
                           n_recurrent=4, n_coda=2, tau=TAU, seq_len=SEQ_LEN, dropout=0.0)
        model.to(device)
        model.eval()

        # Self-influence: same data for train and query
        train_ds = RandomTokenDataset(N_TRAIN, SEQ_LEN, VOCAB, data_seed)
        train_loader_full = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate)
        train_loader_sketch = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate)

        ckpt_path = "/tmp/eval_ckpt_%d.pt" % trial
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        ckpt_spec = CheckpointSpec(path=ckpt_path, weight=1.0)

        target_modules = [model.injection] + list(model.recurrent)

        # Full gradient baseline
        t0 = time.time()
        full_est = FullGradientTracInSDI(
            model=model, target_modules=target_modules,
            loss_reduction="sum", expected_steps=TAU, allow_mixed_steps=False, strict=True)
        full_out = full_est.influence_across_checkpoints(
            checkpoints=[ckpt_spec], train_loader=train_loader_full,
            query_loader=train_loader_full, loss_fn=per_example_loss, mode="sdi",
            train_chunk_size=1, query_chunk_size=1)
        full_time = time.time() - t0
        full_est.close()
        exact_sdi = full_out.sdi.cpu()
        exact_tracin = full_out.tracin.cpu()
        del full_out

        # Sketched SDI
        torch.manual_seed(sketch_seed)
        t0 = time.time()
        sketch_est = ProjectedTracInSDI(
            model=model, target_modules=target_modules, projection_size=M, seed=sketch_seed,
            loss_reduction="sum", expected_steps=TAU, allow_mixed_steps=False, strict=True)
        sketch_out = sketch_est.influence_across_checkpoints(
            checkpoints=[ckpt_spec], train_loader=train_loader_sketch,
            query_loader=train_loader_sketch, loss_fn=per_example_loss, mode="sdi",
            train_chunk_size=4, query_chunk_size=4)
        sketch_time = time.time() - t0
        sketch_est.close()
        sketch_sdi = sketch_out.sdi.cpu()
        sketch_tracin = sketch_out.tracin.cpu()
        del sketch_out

        sdi_err = relative_error(sketch_sdi, exact_sdi)
        tracin_err = relative_error(sketch_tracin, exact_tracin)
        runtime_oh = sketch_time  # total sketched computation time

        print("  SDI Error=%.6f  TracIn Error=%.6f  Runtime=%.3fs" % (sdi_err, tracin_err, runtime_oh))

        all_results.append({"trial": trial + 1, "pair": pair_idx, "data_seed": data_seed, "sketch_seed": sketch_seed,
                            "relative_sdi_error": sdi_err,
                            "relative_tracin_error": tracin_err,
                            "runtime_overhead_s": runtime_oh,
                            "full_time_s": full_time, "sketch_time_s": sketch_time})

        del model, exact_sdi, exact_tracin, sketch_sdi, sketch_tracin
        torch.cuda.empty_cache()

    # ALGO-2: pair-average before computing final mean (antithetic variance reduction)
    n_pairs = len(all_results) // 2
    paired_sdi = []
    paired_tracin = []
    paired_rt = []
    for p in range(n_pairs):
        r0 = all_results[2 * p]
        r1 = all_results[2 * p + 1]
        paired_sdi.append((r0["relative_sdi_error"] + r1["relative_sdi_error"]) / 2)
        paired_tracin.append((r0["relative_tracin_error"] + r1["relative_tracin_error"]) / 2)
        paired_rt.append((r0["runtime_overhead_s"] + r1["runtime_overhead_s"]) / 2)

    sdi_errs = paired_sdi
    tracin_errs = paired_tracin
    runtimes = paired_rt

    mean_sdi = sum(sdi_errs) / len(sdi_errs)
    mean_tracin = sum(tracin_errs) / len(tracin_errs)
    mean_rt = sum(runtimes) / len(runtimes)

    def stdv(vals, mean):
        return math.sqrt(sum((x - mean)**2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0

    metrics = {
        "relative_sdi_error_mean": round(mean_sdi, 6),
        "relative_sdi_error_std": round(stdv(sdi_errs, mean_sdi), 6),
        "relative_tracin_error_mean": round(mean_tracin, 6),
        "relative_tracin_error_std": round(stdv(tracin_errs, mean_tracin), 6),
        "runtime_overhead_mean_s": round(mean_rt, 6),
        "runtime_overhead_std_s": round(stdv(runtimes, mean_rt), 6),
        "n_trials": args.n_trials,
        "n_pairs": args.n_trials // 2,
        "paper_targets": {"relative_sdi_error": 0.0388, "relative_tracin_error": 0.0220, "runtime_overhead_s": 2.55},
        "per_trial": all_results,
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Relative SDI Error:   %.6f +/- %.6f  (paper: 0.0388 +/- 0.0030)" % (mean_sdi, stdv(sdi_errs, mean_sdi)))
    print("Relative TracIn Error: %.6f +/- %.6f  (paper: 0.0220 +/- 0.0052)" % (mean_tracin, stdv(tracin_errs, mean_tracin)))
    print("Runtime Overhead:      %.3f +/- %.3f s  (paper: 2.55 +/- 0.002)" % (mean_rt, stdv(runtimes, mean_rt)))
    print("=" * 60)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved to %s" % args.output)

    return metrics

if __name__ == "__main__":
    main()
