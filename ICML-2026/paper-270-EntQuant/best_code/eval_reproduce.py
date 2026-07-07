#!/usr/bin/env python3
"""Reproduction script for EntQuant C4 Perplexity on LLaMA-2 7B at 3-bit.

Reproduces the rubric metric:
  Paper: Float8@2bits - Entropy Coding Enables Data-Free Model Compression
  Metric: C4 Perplexity
  Model: LLaMA-2 7B
  Setting: EntQuant Float8 3-bit (reg_param=14.5, lr=1.0, spike_threshold=inf)
  Paper value: 7.55
  Our value: 7.5491

Usage:
  # Set up environment
  export PATH=/repo/.venv/bin:/opt/conda/envs/py311/bin:$PATH
  export CC=gcc-10 CXX=g++-10
  export CUDA_HOME=/opt/conda/envs/py311
  export CPLUS_INCLUDE_PATH=/opt/conda/envs/py311/targets/x86_64-linux/include
  export HF_HOME=/autosota_cache/hf
  export HF_ENDPOINT=https://hf-mirror.com

  # Run (ensure C4 data is downloaded first)
  python3 eval_reproduce.py

  # Or with custom settings:
  WEIGHT_QTYPE=qint8 LAMBDA=14.5 LR=1.0 NORM_P=2.0 python3 eval_reproduce.py
"""

import os, sys, json, gzip, random, logging
import torch
from tqdm import tqdm

# Environment setup (idempotent - respects pre-set values)
os.environ.setdefault("HF_HOME", "/autosota_cache/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CC", "gcc-10")
os.environ.setdefault("CXX", "g++-10")
os.environ.setdefault("CUDA_HOME", "/opt/conda/envs/py311")
os.environ.setdefault(
    "CPLUS_INCLUDE_PATH",
    "/opt/conda/envs/py311/targets/x86_64-linux/include"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)

from entquant import EntQuantModel
from entquant.quantization.optimizer import SymmetricEntropyOptimizer, WrappedAbsmaxOptimizer
from transformers import AutoTokenizer

# ---- Configuration ----
MODEL = os.environ.get("MODEL_PATH", "/models/Llama-2-7b-hf")
DTYPE = torch.bfloat16
WEIGHT_QTYPE = os.environ.get("WEIGHT_QTYPE", "qfloat8")  # qfloat8 or qint8
LAMBDA = float(os.environ.get("LAMBDA", "14.5"))   # ~3-bit
LR = float(os.environ.get("LR", "1.0"))
NORM_P = float(os.environ.get("NORM_P", "1.0"))  # Lp norm for reconstruction loss
MAXITERS = int(os.environ.get("MAXITERS", "500"))  # max LBFGS iterations
HISTORY_SIZE = int(os.environ.get("HISTORY_SIZE", "100"))  # LBFGS history size
CTX_LENGTH = int(os.environ.get("CTX_LENGTH", "2048"))
VAL_PATH = os.environ.get(
    "C4_VAL_PATH",
    "/autosota_cache/hf/datasets_local/c4-validation.00000-of-00008.json.gz"
)

print(f"EntQuant Reproduction: model={MODEL}, qtype={WEIGHT_QTYPE}")
print(f"  lambda={LAMBDA}, lr={LR}, norm_p={NORM_P}, maxiters={MAXITERS}, history_size={HISTORY_SIZE}, ctx_length={CTX_LENGTH}")
print(f"  C4 data: {VAL_PATH}")

# Step 1: Tokenizer
print("\n[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Step 2: Quantize model
print(f"[2/3] Quantizing with EntQuant {WEIGHT_QTYPE}...")
optimizer = SymmetricEntropyOptimizer(lr=LR, reg_param=LAMBDA, norm_p=NORM_P, maxiters=MAXITERS, history_size=HISTORY_SIZE)
optimizer_fb = WrappedAbsmaxOptimizer()

model = EntQuantModel.from_pretrained(
    MODEL, quantize=True, compress=False,
    weight_qtype=WEIGHT_QTYPE, dtype=DTYPE,
    optimizer=optimizer, optimizer_fallback=optimizer_fb,
    device_map="cuda",
)
print("Quantization complete.")

# Step 3: C4 Perplexity
print(f"[3/3] Evaluating C4 perplexity (ctx_length={CTX_LENGTH})...")

val_data = []
with gzip.open(VAL_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        val_data.append(json.loads(line))
print(f"Loaded {len(val_data)} C4 validation samples")

random.seed(0)
valenc = []
for _ in range(256):
    while True:
        i = random.randint(0, len(val_data) - 1)
        tmp = tokenizer(val_data[i]["text"], return_tensors="pt")
        if tmp.input_ids.shape[1] >= CTX_LENGTH:
            break
    i = random.randint(0, tmp.input_ids.shape[1] - CTX_LENGTH - 1)
    valenc.append(tmp.input_ids[:, i:i + CTX_LENGTH])

input_ids = torch.hstack(valenc)
n_iterations = input_ids.numel() // CTX_LENGTH
nlls = []

inner_model = model.model
for i in tqdm(range(n_iterations), desc="Evaluating PPL"):
    batch = input_ids[:, (i * CTX_LENGTH):((i + 1) * CTX_LENGTH)].to(inner_model.device)
    lm_logits = inner_model(batch).logits
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:]
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    nlls.append(loss.float() * CTX_LENGTH)

ppl = torch.exp(torch.stack(nlls).sum() / (n_iterations * CTX_LENGTH))
result = ppl.item()
print(f"\nC4 Perplexity: {result:.4f}")
print("=" * 60)
