import torch, numpy as np, os, sys
from PIL import Image
sys.path.insert(0, '/repo')
from diffusers import MarigoldDepthPipeline, DDIMScheduler

ETH3D_H, ETH3D_W = 4032, 6048

def load_depth_binary(path):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(ETH3D_H, ETH3D_W).copy()
    data[~np.isfinite(data)] = 0.0
    return data

# Test: what if we DON'T do per-image alignment?
# Marigold is affine-invariant, so we need some alignment.
# Maybe the paper uses a different approach.

dtype = torch.float16
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=dtype, variant='fp16', local_files_only=True)
# Use trailing timestep spacing for 1-step inference
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
pipe = pipe.to('cuda')

pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

test_pairs = pairs[:10]

for steps in [1, 10, 20]:
    all_absrel = []
    all_d1 = []
    for i, (rgb_rel, depth_rel) in enumerate(test_pairs):
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

        # Align with least squares
        gt_m = gt_depth[valid_mask]
        pred_m = pred_depth[valid_mask]
        A = np.stack([pred_m, np.ones_like(pred_m)], axis=1)
        s, t = np.linalg.lstsq(A, gt_m, rcond=None)[0]
        pred_aligned = s * pred_depth + t

        if pred_aligned.shape != gt_depth.shape:
            import cv2
            pred_aligned = cv2.resize(pred_aligned, (gt_depth.shape[1], gt_depth.shape[0]))

        # Clip
        pred_aligned = np.clip(pred_aligned, 1e-6, None)

        # Compute metrics
        gt_valid = gt_depth[valid_mask]
        pred_valid = pred_aligned[valid_mask]

        absrel = float(np.mean(np.abs(gt_valid - pred_valid) / gt_valid))
        thresh = np.maximum(gt_valid/pred_valid, pred_valid/gt_valid)
        d1 = float((thresh < 1.25).mean())

        all_absrel.append(absrel)
        all_d1.append(d1)

    avg_absrel = np.mean(all_absrel) * 100
    avg_d1 = np.mean(all_d1) * 100
    print(f'Steps={steps:2d} (trailing): AbsRel={avg_absrel:.2f}%, d1={avg_d1:.2f}%')
