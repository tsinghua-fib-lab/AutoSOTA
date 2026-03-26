"""
Scale calibration script for EfficientQAT.
Fine-tunes quantization scales using wikitext2 TRAIN data.
Saves updated scales to /model/Llama-2-7b-EfficientQAT-w2g128-scales.pt
"""
import sys
sys.path.insert(0, '/repo')

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import math
import os

from quantize.int_linear_real import load_quantized_model, QuantLinear

# Config
MODEL_PATH = '/model/Llama-2-7b-EfficientQAT-w2g128'
SCALES_OUT = MODEL_PATH + '-scales.pt'
SEQLEN = 2048
# Use wikitext2 TRAIN data (not test data!)
N_CALIB_SAMPLES = 64  # number of sequences for calibration
N_STEPS = 50
LR = 5e-5
SEED = 42

print("Loading model...")
model, tokenizer = load_quantized_model(MODEL_PATH, wbits=2, group_size=128)
model.eval()

# Load wikitext2 TRAIN data
print("Loading wikitext2 TRAIN data...")
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
print(f"Train tokens: {trainenc.input_ids.shape[1]}")

random.seed(SEED)
torch.manual_seed(SEED)

# Create calibration samples (use only the first 90% of train tokens to avoid leakage)
val_sample_ratio = 0.9
max_start = int(trainenc.input_ids.shape[1] * val_sample_ratio) - SEQLEN - 1
calib_loader = []
for _ in range(N_CALIB_SAMPLES):
    i = random.randint(0, max_start)
    calib_loader.append(trainenc.input_ids[:, i:i+SEQLEN])
print(f"Created {len(calib_loader)} calibration samples")

# Get device map and dispatch model
from accelerate import infer_auto_device_map, dispatch_model
block_class_name = model.model.layers[0].__class__.__name__
device_map = infer_auto_device_map(model, max_memory={i: "70GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
model = dispatch_model(model, device_map=device_map)

# Make only scales trainable
for name, param in model.named_parameters():
    if 'scales' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Count trainable params
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable scale params: {n_trainable}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.0,
    betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=LR/10)

loss_fct = nn.CrossEntropyLoss()
model.config.use_cache = False

print(f"Starting calibration: {N_STEPS} steps, LR={LR}")
best_loss = float('inf')
best_scales = None

for step in range(N_STEPS):
    # Pick a random batch of calibration samples
    batch_idx = random.randint(0, N_CALIB_SAMPLES - 1)
    input_ids = calib_loader[batch_idx].to('cuda:0')
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    with torch.no_grad():
        hidden_states = model.model(input_ids)[0]
    
    # Actually we need gradients through the full model
    # Let's do it properly
    optimizer.zero_grad()
    
    # Re-do forward with gradient tracking
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.scales.requires_grad_(True)
    
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 
        max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    
    if step % 10 == 0 or step == N_STEPS - 1:
        print(f"Step {step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
    
    if loss.item() < best_loss:
        best_loss = loss.item()

# Save updated scales
print("Saving updated scales...")
scale_dict = {}
for name, module in model.named_modules():
    if isinstance(module, QuantLinear):
        scale_dict[name] = module.scales.data.to(torch.float16).cpu()
        
torch.save(scale_dict, SCALES_OUT)
print(f"Saved {len(scale_dict)} scale tensors to {SCALES_OUT}")
print(f"Best training loss: {best_loss:.4f}")
print("Done! Run eval to check PPL.")
