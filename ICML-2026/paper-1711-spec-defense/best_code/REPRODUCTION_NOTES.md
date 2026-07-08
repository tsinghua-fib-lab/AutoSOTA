# CSR Reproduction Notes — Paper 1711

## Reproduced Metrics
- Clean Accuracy (ImageNet, ViT-B/16, CSR): 63.20% (paper: 62.5%, CI: [61.25, 63.75])
- Robust Accuracy (ImageNet, ViT-B/16, CSR, PGD eps=1/255): 55.86% (paper: 58.9%, CI: [53.6, 59.43])

## Evaluation Settings
- Model: CLIP ViT-B/16 (openai/clip-vit-base-patch16, local path: /models/clip-vit-base-patch16)
- Dataset: ImageNet validation (10,000 images from 1,000 classes, 10 per class)
- Attack: 10-step PGD, epsilon=1/255
- CSR params: r=40, tau=0.85, noise_budget=4/255, step_size=2/255, N=3
- Filter: Gaussian low-pass filter
- Zero-shot evaluation with "a photo of a {class}" prompts
- Batch size: 32, num_workers: 0

## Setup Changes
1. Patched csr/config.py HF_MODEL_CONFIGS to use local paths (/models/clip-vit-base-patch16)
2. Changed configs/default.yaml: data_root=/repo/data, num_workers=0
3. Fixed labels.csv: replaced underscores with spaces in class names
4. Created ImageNet dataset structure at /repo/data/General/ImageNet/ with symlinks

## Key Files
- Evaluation: scripts/evaluate.py
- CSR Defense: csr/defense/csr_fast.py (FastCSRDefense)
- Config: csr/config.py, configs/default.yaml
- Adversarial generation: scripts/generate_adv.py

## Eval Command
python scripts/evaluate.py --model CLIP-B-16 --defense fast_csr --device cuda:0 --sample_n 1000 --datasets General/ImageNet --adv_root ./outputs/adv_samples/PGD --adv_attack PGD --lpf_radius 40 --detect_thresh 0.85 --purify_steps 3 --batch_size 32

## Known Optimization Levers
- lpf_radius (default 40): Gaussian filter cutoff. Higher = less filtering
- detect_thresh (default 0.85): Detection threshold. Lower = more samples classified as adversarial
- purify_steps (default 3): PGD purification iterations. More steps = potentially better purification
- purify_eps (default 4/255): Purification perturbation budget
- purify_alpha (default 2/255): Purification step size
- filter_type: gaussian, butterworth, ideal
