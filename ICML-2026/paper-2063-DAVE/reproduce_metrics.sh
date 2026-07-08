#!/bin/bash
# DAVE Reproduction: compute Vendi and CLIP scores for SANA1.5 + DAVE vs baseline
# Paper: "Breaking the Lock-in" (ICML 2026)
# Run from /repo inside container autosota_repro_paper_2063

cd /repo
python3 -c "
import json, torch, numpy as np
from dave_sana import create_dave_sana_pipeline
from vendi_score.vendi import score_X
import open_clip
from PIL import Image
from pathlib import Path

# Config
MODEL_PATH = '/models/SANA1.5_1.6B_1024px_diffusers'
OUTPUT_DIR = Path('/repo/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet classes (subset for validation)
CLASSES = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
           'electric ray', 'stingray', 'cock', 'hen', 'ostrich',
           'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting',
           'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
           'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl']
N_SAMPLES = 5
SEED = 42

# Create pipeline
print('Loading DAVE-SANA pipeline...')
pipe = create_dave_sana_pipeline(
    MODEL_PATH, target_blocks=[13], dave_scale=0.2, tau=0.2, guidance_scale=4.5,
    torch_dtype=torch.bfloat16,
)

# Load CLIP model for evaluation
print('Loading CLIP ViT-B/32...')
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='/models/clip/ViT-B-32.pt', device='cuda'
)

def generate_images(pipe, prompts, n_per_prompt, use_dave, label):
    images, img_prompts = [], []
    for pi, prompt in enumerate(prompts):
        for si in range(n_per_prompt):
            seed = SEED + pi * n_per_prompt + si
            gen = torch.Generator('cuda').manual_seed(seed)
            result = pipe(prompt, use_dave=use_dave, num_inference_steps=20, generator=gen)
            images.append(result.images[0])
            img_prompts.append(prompt)
    print(f'  {label}: {len(images)} images generated')
    return images, img_prompts

def compute_metrics(images, prompts, label):
    # CLIP embeddings
    img_tensors = torch.stack([clip_preprocess(img) for img in images]).to('cuda')
    n = len(img_tensors)
    embeddings = []
    bs = 32
    for i in range(0, n, bs):
        with torch.no_grad():
            emb = clip_model.encode_image(img_tensors[i:i+bs])
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # CLIP Score
    text_tokens = open_clip.get_tokenizer('ViT-B-32')(prompts).to('cuda')
    text_embs = []
    for i in range(0, n, bs):
        with torch.no_grad():
            te = clip_model.encode_text(text_tokens[i:i+bs])
            te = te / te.norm(dim=-1, keepdim=True)
        text_embs.append(te.cpu())
    text_embs = torch.cat(text_embs, dim=0)
    clip_score = (embeddings * text_embs).sum(dim=-1).mean().item()

    # Vendi Score
    vendi = score_X(embeddings.numpy())

    print(f'  {label}: CLIP={clip_score:.4f}, Vendi={vendi:.4f}')
    return clip_score, vendi

# Generate
prompts = [f'a photo of a {c}' for c in CLASSES]
print(f'Generating {len(CLASSES)} classes x {N_SAMPLES} samples = {len(CLASSES)*N_SAMPLES} images each...')

print('Generating baseline...')
base_imgs, base_prompts = generate_images(pipe, prompts, N_SAMPLES, use_dave=False, label='Baseline')

print('Generating DAVE...')
dave_imgs, dave_prompts = generate_images(pipe, prompts, N_SAMPLES, use_dave=True, label='DAVE')

# Compute metrics
print('Computing metrics...')
base_clip, base_vendi = compute_metrics(base_imgs, base_prompts, 'Baseline')
dave_clip, dave_vendi = compute_metrics(dave_imgs, dave_prompts, 'DAVE')

# Save results
results = {
    'config': {
        'model': 'SANA1.5_1.6B',
        'target_block': 13, 'dave_scale': 0.2, 'tau': 0.2, 'guidance_scale': 4.5,
        'n_classes': len(CLASSES), 'n_samples_per_class': N_SAMPLES,
        'total_images': len(CLASSES) * N_SAMPLES,
        'image_size': '1024x1024',
    },
    'metrics': {
        'baseline': {'CLIP': round(base_clip, 4), 'Vendi': round(base_vendi, 4)},
        'dave': {'CLIP': round(dave_clip, 4), 'Vendi': round(dave_vendi, 4)},
    },
    'paper_reference': {
        'SANA1.5_DAVE_CLIP': 0.2885,
        'SANA1.5_DAVE_Vendi': 2.20,
    },
}
path = OUTPUT_DIR / 'metrics.json'
path.write_text(json.dumps(results, indent=2))
print(f'\\nResults saved to {path}')
print(json.dumps(results['metrics'], indent=2))
"
