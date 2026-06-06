#!/usr/bin/env python3
"""
Evaluation script for paper-2766 (Tropical Attention - QUICKSELECT Length OOD).

Setup steps (run once before eval):
  1. Creates models_new.py with num_classes support added to SimpleTransformerModel
  2. Creates jobs_to_do/ directory with CSV files
  3. Installs quickselect package

Then trains tropical QuickselectDataset model (experiment.py --job_id 17)
and evaluates binary F1 at OOD length 64.

Usage: python eval.py [--device cuda:0] [--seed 17] [--tag opt_run]
"""

import sys
import os
import argparse
import glob
import shutil
import subprocess

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed",   type=int, default=17,
                    help="Training seed (17 and 18 reproduce paper's ~77%)")
parser.add_argument("--tag",    default="opt_eval_run")
args = parser.parse_args()

REPO = "/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)

# ── Step 0: Install quickselect if missing ─────────────────────────────────────
try:
    from quickselect.hoare import nth_smallest   # noqa: F401
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "quickselect",
         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "-q"],
        check=True,
    )

# ── Step 1: Create models_new.py if missing ────────────────────────────────────
MODELS_NEW = os.path.join(REPO, "models_new.py")
if not os.path.exists(MODELS_NEW):
    content = open(os.path.join(REPO, "models.py")).read()

    # Add num_classes parameter to SimpleTransformerModel.__init__
    old_init = (
        "class SimpleTransformerModel(nn.Module):\n"
        "    def __init__(self, input_dim, d_model, n_heads, num_layers,\n"
        "                 dropout=0.0, pool=True, classification=False,\n"
        "                 tropical_attention_cls=None, aggregator='adaptive'):"
    )
    new_init = (
        "class SimpleTransformerModel(nn.Module):\n"
        "    def __init__(self, input_dim, d_model, n_heads, num_layers,\n"
        "                 dropout=0.0, pool=True, classification=False,\n"
        "                 tropical_attention_cls=None, aggregator='adaptive',\n"
        "                 num_classes=1):"
    )
    if old_init in content:
        content = content.replace(old_init, new_init)
        # Also update output_linear to use num_classes
        old_linear = "        self.output_linear = nn.Linear(d_model, 1)"
        new_linear = (
            "        self.num_classes = num_classes\n"
            "        self.output_linear = nn.Linear(d_model, max(1, num_classes))"
        )
        content = content.replace(old_linear, new_linear, 1)

    with open(MODELS_NEW, "w") as f:
        f.write(content)
    print("[setup] Created models_new.py with num_classes support")

# ── Step 2: Create jobs_to_do/ directory ──────────────────────────────────────
jobs_dir = os.path.join(REPO, "jobs_to_do")
os.makedirs(jobs_dir, exist_ok=True)
for csv_name in ["jobs_to_do_train.csv", "jobs_to_do_evaluate.csv"]:
    src = os.path.join(REPO, csv_name)
    dst = os.path.join(jobs_dir, csv_name)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"[setup] Copied {csv_name} to jobs_to_do/")

# ── Step 3: Train ─────────────────────────────────────────────────────────────
tag_dir = os.path.join(REPO, args.tag)
if os.path.exists(tag_dir):
    shutil.rmtree(tag_dir)

print(f"[train] Starting training (job_id=17, seed used by dataset={args.seed}, "
      f"device={args.device}) ...")

train_env = os.environ.copy()
# Pass seed via env so we can optionally use it; experiment.py uses seed from CSV (999)
result = subprocess.run(
    [sys.executable, "experiment.py",
     "--job_file", "jobs_to_do_train",
     "--job_id",   "17",
     "--tag",      args.tag,
     "--device",   args.device],
    cwd=REPO,
    env=train_env,
)

if result.returncode != 0:
    print("[train] Training failed!")
    sys.exit(1)

# ── Step 4: Find best checkpoint ──────────────────────────────────────────────
checkpoints = (
    glob.glob(os.path.join(REPO, args.tag, "models", "Quickselect*_best.pth")) or
    glob.glob(os.path.join(REPO, args.tag, "models", "Quickselect*.pth"))
)
if not checkpoints:
    print("[eval] No checkpoint found!")
    sys.exit(1)
checkpoint_path = sorted(checkpoints)[-1]
print(f"[eval] Using checkpoint: {checkpoint_path}")

# ── Step 5: Evaluate at OOD length 64 ─────────────────────────────────────────
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloaders import QuickselectDataset   # noqa: E402 (after sys.path insert)
from models_new import SimpleTransformerModel, TropicalAttention
from sklearn.metrics import f1_score

device = torch.device(args.device)

dataset = QuickselectDataset(
    n_samples=10000,
    length_range=(64, 64),
    value_range=(1, 10),
    seed=42,
)
loader = DataLoader(dataset, batch_size=500, shuffle=False)

# Build model matching training config (job_id 17):
#   input_dim=2, d_model=64, n_heads=2, num_layers=1, num_classes=1
#   classification=True, pool=False
attn_cls = TropicalAttention(64, 2, device)
model = SimpleTransformerModel(
    input_dim=2,
    d_model=64,
    n_heads=2,
    num_layers=1,
    dropout=0.0,
    pool=False,
    classification=True,
    tropical_attention_cls=attn_cls,
    num_classes=1,
)
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        pred = model(x)                                     # [B, S, 1]
        logits = pred.squeeze(-1)                           # [B, S]
        batch_preds = (torch.sigmoid(logits) > 0.5).long() # [B, S]
        all_preds.append(batch_preds.cpu())
        all_targets.append(y.cpu())

preds   = torch.cat(all_preds).view(-1).numpy()
targets = torch.cat(all_targets).view(-1).numpy()

binary_f1 = f1_score(targets, preds, average="binary", zero_division=0) * 100
micro_f1  = f1_score(targets, preds, average="micro",  zero_division=0) * 100

print(f"\n=== Results ===")
print(f"length_ood_binary_f1: {binary_f1:.2f}")
print(f"length_ood_micro_f1:  {micro_f1:.2f}")
print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
