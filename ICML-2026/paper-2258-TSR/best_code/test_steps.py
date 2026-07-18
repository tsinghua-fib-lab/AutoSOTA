#!/usr/bin/env python3
"""Quick test of different step counts to match paper's DDIM baseline."""
import torch, numpy as np, os, sys
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, '/repo')
from diffusers import MarigoldDepthPipeline, DDIMScheduler

ETH3D_H, ETH3D_W = 4032, 6048

def load_depth_binary(path):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(ETH3D_H, ETH3D_W).copy()
    data[~np.isfinite(data)] = 0.0
    return data

def compute_errors(gt, pred, valid_mask=None):
    if valid_mask is None: valid_mask = gt > 0
    gt, pred = gt[valid_mask], pred[valid_mask]
    pred = np.clip(pred, 1e-6, None)
    thresh = np.maximum(gt/pred, pred/gt)
    return dict(abs_rel=float(np.mean(np.abs(gt-pred)/gt)), delta1=float((thresh<1.25).mean()))

def align_depth(gt, pred, mask=None):
    if mask is not None: gt_m, pred_m = gt[mask], pred[mask]
    else: m = gt>0; gt_m, pred_m = gt[m], pred[m]
    A = np.stack([pred_m, np.ones_like(pred_m)], axis=1)
    s, t = np.linalg.lstsq(A, gt_m, rcond=None)[0]
    return s*pred+t

dtype = torch.float16
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=dtype, variant='fp16', local_files_only=True)
pipe = pipe.to('cuda')

pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

test_pairs = pairs[:10]  # Test on first 10 images

for steps in [1, 5, 10, 20, 50]:
    pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')

    all_metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(tqdm(test_pairs, desc=f'Steps={steps}')):
        rgb_path = os.path.join('/datasets/marigold_eval/eth3d', rgb_rel)
        depth_path = os.path.join('/datasets/marigold_eval/eth3d', depth_rel)
        input_image = Image.open(rgb_path).convert('RGB')
        gt_depth = load_depth_binary(depth_path)
        valid_mask = gt_depth > 0

        generator = torch.Generator(device='cuda')
        generator.manual_seed(42 + i)

        with torch.no_grad():
            pipe_out = pipe(input_image, num_inference_steps=steps, ensemble_size=1,
                          processing_resolution=768, match_input_resolution=True,
                          batch_size=1, generator=generator)
        pred_depth = pipe_out.prediction[0, :, :, 0]
        pred_aligned = align_depth(gt_depth, pred_depth, valid_mask)
        if pred_aligned.shape != gt_depth.shape:
            import cv2
            pred_aligned = cv2.resize(pred_aligned, (gt_depth.shape[1], gt_depth.shape[0]))
        metrics = compute_errors(gt_depth, pred_aligned, valid_mask)
        all_metrics.append(metrics)

    avg_absrel = np.mean([m['abs_rel'] for m in all_metrics]) * 100
    avg_d1 = np.mean([m['delta1'] for m in all_metrics]) * 100
    print(f'  Steps={steps:2d}: AbsRel={avg_absrel:.2f}%, delta1={avg_d1:.2f}%')

print(f'\nPaper ref: DDIM=7.10% AbsRel, 90.4% d1')
