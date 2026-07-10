"""
MNIST Evaluation Script for Paper 2727
Reproduces: KarcherFlow Attention on MNIST with hidden_dim=4
Paper result: 85.52% +/- 2.50
Eval command: python3 /repo/eval_mnist.py --model kf_attention --hidden-dim 4 --epochs 14 --trials 5 --seed 42
"""
import os, sys, json, math, argparse
import numpy as np

sys.path.insert(0, "/repo/ml/hyphop")
os.chdir("/repo/ml/hyphop")

import torch
from test_mnist import main as run_mnist
from unittest.mock import patch


def parse_args():
    ap = argparse.ArgumentParser(description="MNIST KarcherFlow evaluation")
    ap.add_argument("--model", default="kf_attention",
                    choices=["kf_attention", "kf_layer", "kf_pooling",
                             "hf_attention", "hf_layer", "hf_pooling",
                             "ein_attention", "ein_layer", "ein_pooling"])
    ap.add_argument("--hidden-dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-interval", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    trial_accs = []

    for trial_idx in range(args.trials):
        seed = args.seed + trial_idx
        print("Trial {}/{} seed={} ...".format(trial_idx+1, args.trials, seed),
              flush=True)

        test_args = [
            "test_mnist.py",
            "--model", args.model,
            "--hidden-dim", str(args.hidden_dim),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--gamma", str(args.gamma),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
            "--log-interval", str(args.log_interval),
            "--device", args.device,
        ]
        if args.beta is not None:
            test_args += ["--beta", str(args.beta)]

        torch.manual_seed(seed)
        with patch.object(sys, "argv", test_args):
            acc = run_mnist()
        trial_accs.append(acc)
        print("Trial {} accuracy: {:.2f}%".format(trial_idx+1, acc))

    mean_acc = float(np.mean(trial_accs))
    std_acc = float(np.std(trial_accs, ddof=1))
    print("\nRESULT: accuracy={:.4f} accuracy_std={:.4f}".format(mean_acc, std_acc))
    return mean_acc


if __name__ == "__main__":
    main()
