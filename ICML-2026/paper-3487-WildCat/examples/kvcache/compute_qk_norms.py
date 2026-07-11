"""
Compute average entry growth parameter gamma(n) as a function of the sequence length n
for Qwen2.5-7B on the QASPER-E dataset.

For a batch of sequences with queries of shape (batch, num_heads, seq_len, head_dim):
    q_scale = queries.square().sum(dim=-1).sqrt().amax(dim=-1).mean(0)  -> (num_heads,)
    k_scale = keys.square().sum(dim=-1).sqrt().amax(dim=-1).mean(0)     -> (num_kv_heads,)

These are computed at each power-of-2 prefix length n, averaged over layers and heads,
and plotted as a function of n on a log2 x-axis.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "Xnhyacinth/LongBench"
DATASET_CONFIG = "qasper_e"
POWERS_OF_TWO = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
MAX_SEQ_LEN = max(POWERS_OF_TWO)
NUM_SAMPLES = 10                 # number of unique contexts to average over
OUTPUT_PATH = "qk_norms_qwen7b_longbench_qasper_e.png"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",  # ensures standard nn.Linear hooks fire
)
model.eval()

cfg = model.config
num_heads    = cfg.num_attention_heads
num_kv_heads = cfg.num_key_value_heads
head_dim     = cfg.hidden_size // num_heads
num_layers   = cfg.num_hidden_layers
print(f"  layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

# ---------------------------------------------------------------------------
# Forward hooks on q_proj / k_proj
# ---------------------------------------------------------------------------
# Buffers – one slot per layer, overwritten on every forward pass
layer_q: list[torch.Tensor | None] = [None] * num_layers
layer_k: list[torch.Tensor | None] = [None] * num_layers


def make_q_hook(idx: int):
    def hook(module, input, output):
        b, s, _ = output.shape
        # -> (batch, num_heads, seq_len, head_dim)
        layer_q[idx] = output.detach().cpu().reshape(b, s, num_heads, head_dim).transpose(1, 2)
    return hook


def make_k_hook(idx: int):
    def hook(module, input, output):
        b, s, _ = output.shape
        # -> (batch, num_kv_heads, seq_len, head_dim)
        layer_k[idx] = output.detach().cpu().reshape(b, s, num_kv_heads, head_dim).transpose(1, 2)
    return hook


hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(i)))
    hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))

# ---------------------------------------------------------------------------
# Dataset – collect unique contexts long enough for MAX_SEQ_LEN tokens
# ---------------------------------------------------------------------------
print(f"\nLoading dataset: {DATASET_NAME} (config={DATASET_CONFIG})")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

long_contexts: list[str] = []
seen: set[str] = set()
for row in dataset:
    ctx = row["context"]
    if ctx in seen:
        continue
    seen.add(ctx)
    token_len = tokenizer(ctx, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    if token_len >= MAX_SEQ_LEN:
        long_contexts.append(ctx)
    if len(long_contexts) >= NUM_SAMPLES:
        break

print(f"Found {len(long_contexts)} contexts with >= {MAX_SEQ_LEN} tokens")
if len(long_contexts) == 0:
    raise RuntimeError(
        f"No contexts long enough in {DATASET_NAME}/{DATASET_CONFIG}. "
        "Try a larger config, e.g. '8192'."
    )

# ---------------------------------------------------------------------------
# Main loop – forward pass per context, accumulate norms per prefix length
# ---------------------------------------------------------------------------
# q_norms_all[n]  ->  list of per-layer-averaged scalars, one per sample
q_norms_all: dict[int, list] = {n: [] for n in POWERS_OF_TWO}
k_norms_all: dict[int, list] = {n: [] for n in POWERS_OF_TWO}
gamma_all: dict[int, list] = {n: [] for n in POWERS_OF_TWO}

for ctx_idx, context in enumerate(long_contexts):
    print(f"  sample {ctx_idx + 1}/{len(long_contexts)}", flush=True)

    input_ids = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )["input_ids"].to(device)

    with torch.no_grad():
        model(input_ids)
    del input_ids
    torch.cuda.empty_cache()

    for n in POWERS_OF_TWO:
        q_layer_vals = []
        k_layer_vals = []
        gamma_vals = []

        for lq, lk in zip(layer_q, layer_k):
            if lq is None or lk is None:
                continue

            # lq: (batch, num_heads,    seq_len, head_dim)
            # lk: (batch, num_kv_heads, seq_len, head_dim)
            q_prefix = lq[:, :, :n, :].float()
            k_prefix = lk[:, :, :n, :].float()
            # mean-center keys per layer
            k_prefix = k_prefix - k_prefix.mean(dim=2, keepdim=True)  

            # (batch, heads, n, d) -> (batch, heads, n) -> (batch, heads)
            q_scale = q_prefix.square().sum(dim=-1).sqrt().amax(dim=-1)
            k_scale = k_prefix.square().sum(dim=-1).sqrt().amax(dim=-1)

            g = num_heads // num_kv_heads  
            qk_scale = q_scale * k_scale.repeat_interleave(g)  # (num_heads,)

            gamma = qk_scale.mean()/np.sqrt(head_dim)/np.log(n)  # scalar

            # scalar: mean over heads
            gamma_vals.append(gamma.detach().cpu())

        # mean over layers
        gamma_all[n].append(gamma_vals)

    for i in range(num_layers):
        layer_q[i] = None
        layer_k[i] = None

# Remove hooks
for h in hooks:
    h.remove()

# ---------------------------------------------------------------------------
# Aggregate over samples
# ---------------------------------------------------------------------------
gamma_n = [float(np.mean(gamma_all[n])) for n in POWERS_OF_TWO]

print("|--:|-------:|")
for n, gamma in zip(POWERS_OF_TWO, gamma_n):
    print(f"| {n} | {gamma:.2f} |")