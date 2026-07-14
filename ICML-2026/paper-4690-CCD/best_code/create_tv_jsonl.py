"""Streamlined script: generate images for TV prompts, match templates, create JSONL."""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import utils.processing_utils as processing_utils

MODEL_PATH = "/models/stable-diffusion-v1-4"
PARQUET_FILE = "groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet"
OUT_JSONL = "sdv1-4_bb_attack_gt_verify_TV.jsonl"
MVRV_JSONL = "sdv1-4_bb_attack_gt_verify_MVRV.jsonl"
TEMPLATE_DIR = "templates"
N_SEEDS = 4
STEPS = 50

device = "cuda"
dtype = torch.float16

# Load templates and masks
print("Loading templates...")
template_parquet = pd.read_parquet(f"{TEMPLATE_DIR}/metadata.parquet")
mask_files = list(template_parquet["mask_file"])
template_files = list(template_parquet["img_file"])

template_imgs = []
mask_imgs = []
for imgf, maskf in tqdm(zip(template_files, mask_files), total=len(template_files), desc="Loading templates"):
    img = processing_utils.pil_img_to_torch(Image.open(imgf).resize((256, 256)))
    mask = processing_utils.pil_img_to_torch(Image.open(maskf).resize((256, 256)))
    template_imgs.append(img)
    mask_imgs.append(mask)

# Convert to tensors for batch computation
template_tensors = torch.stack(template_imgs).to(device)
mask_tensors = torch.stack(mask_imgs).to(device)
print(f"Loaded {len(template_imgs)} templates")

# Load model
print("Loading SD v1.4...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

# Load parquet
d = pd.read_parquet(PARQUET_FILE)
tv_rows = d[d["overfit_type"] == "TV"]
mv_rows = d[d["overfit_type"] == "MV"]
rv_rows = d[d["overfit_type"] == "RV"]

print(f"TV prompts: {len(tv_rows)}, MV: {len(mv_rows)}, RV: {len(rv_rows)}")

# Process TV prompts
tv_entries = []
verb_thresh = 2.5e3

for idx, row in tqdm(tv_rows.iterrows(), total=len(tv_rows), desc="TV prompts"):
    caption = row["caption"]
    gen_seeds_list = []
    
    for seed in range(N_SEEDS):
        generator = torch.Generator(device).manual_seed(seed)
        image = pipe(caption, num_inference_steps=STEPS, generator=generator).images[0]
        img_tensor = processing_utils.pil_img_to_torch(image.resize((256, 256)))
        img_tensor_gpu = img_tensor.to(device)
        
        # Batch masked MSE against all templates
        masked_diff = (template_tensors * mask_tensors - img_tensor_gpu.unsqueeze(0) * mask_tensors) ** 2
        mses = masked_diff.sum(dim=(1, 2, 3)) / (mask_tensors.mean(dim=(1, 2, 3)) + 1e-8)
        
        best_mse = mses.min().item()
        best_tidx = mses.argmin().item()
        
        if best_mse < verb_thresh:
            gen_seeds_list.append([seed, best_tidx])
    
    if gen_seeds_list:
        tv_entries.append({
            "caption": caption,
            "gen_seeds": gen_seeds_list,
            "overfit_type": "TV",
        })
        if len(tv_entries) % 50 == 0:
            print(f"  Found {len(tv_entries)} TV entries so far...")

print(f"\nTotal TV entries: {len(tv_entries)}")
total_tv_samples = sum(len(e["gen_seeds"]) for e in tv_entries)
print(f"Total TV samples: {total_tv_samples}")

# Save TV JSONL
with open(OUT_JSONL, "w") as f:
    for entry in tv_entries:
        f.write(json.dumps(entry) + "\n")
print(f"Saved {OUT_JSONL}")

# Also save MVRV (MV+RV) entries
mvrv_entries = []
for idx, row in pd.concat([mv_rows, rv_rows]).iterrows():
    seeds = row["gen_seeds"]
    seed_list = []
    if hasattr(seeds, "__len__") and len(seeds) > 0:
        for s in seeds[:N_SEEDS]:
            seed_list.append([int(s), -1])
    mvrv_entries.append({
        "caption": row["caption"],
        "gen_seeds": seed_list,
        "overfit_type": row["overfit_type"],
    })

with open(MVRV_JSONL, "w") as f:
    for entry in mvrv_entries:
        f.write(json.dumps(entry) + "\n")
print(f"Saved {MVRV_JSONL} ({len(mvrv_entries)} entries)")
