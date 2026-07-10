import os, sys
sys.path.insert(0, "/repo/ml/hyphop")
os.chdir("/repo/ml/hyphop")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
torch.manual_seed(42)

from test_mnist import main as run_mnist
from unittest.mock import patch

test_args = [
    "test_mnist.py",
    "--model", "kf_attention",
    "--hidden-dim", "4",
    "--epochs", "1",
    "--lr", "0.001",
    "--gamma", "0.96",
    "--seed", "42",
    "--log-interval", "200",
    "--device", "auto",
]
with patch.object(sys, "argv", test_args):
    acc = run_mnist()
print(f"Quick test acc: {acc:.2f}%")
