import os, sys, json, math, time
import numpy as np

sys.path.insert(0, "/repo/ml/hyphop")
os.chdir("/repo/ml/hyphop")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from test_mnist import main as run_mnist
from unittest.mock import patch

model_name = "kf_attention"
hidden_dim = 4
epochs = 14
lr = 0.001
gamma = 0.96
batch_size = 64
num_trials = 5
base_seed = 42

results = {
    "model": model_name,
    "hidden_dim": hidden_dim,
    "epochs": epochs,
    "lr": lr,
    "gamma": gamma,
    "batch_size": batch_size,
    "beta": 1.0 / math.sqrt(hidden_dim),
    "trials": [],
}

trial_accs = []
for trial_idx in range(num_trials):
    seed = base_seed + trial_idx
    sep = "=" * 60
    print("\n" + sep)
    print("Trial {}/{} (seed={})".format(trial_idx+1, num_trials, seed))
    print(sep)

    test_args = [
        "test_mnist.py",
        "--model", model_name,
        "--hidden-dim", str(hidden_dim),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--gamma", str(gamma),
        "--seed", str(seed),
        "--log-interval", "200",
        "--device", "auto",
    ]
    torch.manual_seed(seed)
    with patch.object(sys, "argv", test_args):
        acc = run_mnist()
    trial_accs.append(acc)
    results["trials"].append({"seed": seed, "accuracy": round(acc, 4)})
    print("Trial {} final accuracy: {:.2f}%".format(trial_idx+1, acc))

mean_acc = float(np.mean(trial_accs))
std_acc = float(np.std(trial_accs, ddof=1))

results["mean_accuracy"] = round(mean_acc, 4)
results["std_accuracy"] = round(std_acc, 4)

print("\n" + "=" * 60)
print("FINAL RESULTS: {:.2f}% +/- {:.2f}%".format(mean_acc, std_acc))
print("=" * 60)

os.makedirs("/repo/results", exist_ok=True)
with open("/repo/results/mnist_d4_kfattention.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to /repo/results/mnist_d4_kfattention.json")
