"""
Scale calibration script v2 for EfficientQAT.
Fine-tunes quantization scales using wikitext2 TRAIN data (not test data).
Saves updated scales to /model/Llama-2-7b-EfficientQAT-w2g128-scales.pt

Usage: python calibrate_scales_v2.py [n_steps] [lr] [n_samples]
"""
import sys
sys.path.insert(0, '/repo')

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import os

from quantize.int_linear_real import load_quantized_model, QuantLinear
from accelerate import infer_auto_device_map, dispatch_model

# Config from command line
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
N_CALIB_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 128

MODEL_PATH = '/model/Llama-2-7b-EfficientQAT-w2g128'
SCALES_OUT = MODEL_PATH + '-scales.pt'
SEQLEN = 512  # shorter seqlen for calibration to save memory
SEED = 42

print(f"Calibration config: N_STEPS={N_STEPS}, LR={LR}, N_CALIB={N_CALIB_SAMPLES}")

print("Loading model...")
model, tokenizer = load_quantized_model(MODEL_PATH, wbits=2, group_size=128)

# Get device map and dispatch model
block_class_name = model.model.layers[0].__class__.__name__
device_map = infer_auto_device_map(model, max_memory={i: "70GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
model = dispatch_model(model, device_map=device_map)

# Load wikitext2 TRAIN data (never test data)
print("Loading wikitext2 TRAIN data...")
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
print(f"Train tokens: {trainenc.input_ids.shape[1]}")

random.seed(SEED)
torch.manual_seed(SEED)

# Create calibration samples - use only first 90% of train to avoid any leakage
max_start = int(trainenc.input_ids.shape[1] * 0.9) - SEQLEN - 1
calib_data = []
for _ in range(N_CALIB_SAMPLES):
    i = random.randint(0, max_start)
    calib_data.append(trainenc.input_ids[:, i:i+SEQLEN])
print(f"Created {len(calib_data)} calibration sequences of length {SEQLEN}")

# Make only scales trainable
for name, param in model.named_parameters():
    param.requires_grad = ('scales' in name)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable scale params: {n_trainable:,}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.0, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=LR/20)

loss_fct = nn.CrossEntropyLoss()
model.config.use_cache = False

print(f"Starting calibration: {N_STEPS} steps, LR={LR}")

losses = []
for step in range(N_STEPS):
    batch_idx = step % N_CALIB_SAMPLES
    input_ids = calib_data[batch_idx].to('cuda:0')

    optimizer.zero_grad()

    # Forward pass with gradient tracking for scales
    outputs = model(input_ids)
    logits = outputs.logits  # [1, seqlen, vocab]

    # Compute language modeling loss (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=0.5
    )
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if step % 20 == 0 or step == N_STEPS - 1:
        avg_loss = sum(losses[-20:]) / len(losses[-20:])
        print(f"Step {step+1}/{N_STEPS}: loss={loss.item():.4f} (avg20={avg_loss:.4f}), lr={scheduler.get_last_lr()[0]:.2e}")

# Save updated scales to the scales file
print("\nSaving updated scales...")
scale_dict = {}
for name, module in model.named_modules():
    if isinstance(module, QuantLinear):
        scale_dict[name] = module.scales.data.detach().to(torch.float16).cpu()

torch.save(scale_dict, SCALES_OUT)
print(f"Saved {len(scale_dict)} scale tensors to {SCALES_OUT}")
print(f"Final avg loss (last 20 steps): {sum(losses[-20:])/len(losses[-20:]):.4f}")
print("Done! Run eval to check PPL improvement.")
