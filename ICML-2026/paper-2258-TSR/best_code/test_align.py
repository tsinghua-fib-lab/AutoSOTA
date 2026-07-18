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

def align_depth_ls(gt_arr, pred_arr, valid_mask_arr, max_resolution=None):
    gt = gt_arr.squeeze()
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()
    
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(gt.shape))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float()).bool().numpy()
    
    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))
    A = np.concatenate([pred_masked, np.ones_like(pred_masked)], axis=-1)
    scale, shift = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    return pred_arr * float(scale) + float(shift), float(scale), float(shift)

def compute_errors(gt, pred, valid_mask=None):
    if valid_mask is None: valid_mask = gt > 0
    gt, pred = gt[valid_mask], pred[valid_mask]
    pred = np.clip(pred, 1e-6, None)
    thresh = np.maximum(gt/pred, pred/gt)
    return dict(abs_rel=float(np.mean(np.abs(gt-pred)/gt)), delta1=float((thresh<1.25).mean()))

dtype = torch.float16
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=dtype, variant='fp16', local_files_only=True)

pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2: pairs.append((parts[0], parts[1]))

test_pairs = pairs[:10]

# Test: default leading spacing, 1 step, with and without max_res
pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
pipe = pipe.to('cuda')

for max_res in [None, 1024]:
    all_metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(test_pairs):
        rgb_path = os.path.join('/datasets/marigold_eval/eth3d', rgb_rel)
        depth_path = os.path.join('/datasets/marigold_eval/eth3d', depth_rel)
        input_image = Image.open(rgb_path).convert('RGB')
        gt_depth = load_depth_binary(depth_path)
        valid_mask = gt_depth > 0

        gen = torch.Generator(device='cuda')
        gen.manual_seed(42 + i)

        with torch.no_grad():
            pipe_out = pipe(input_image, num_inference_steps=1, ensemble_size=1,
                          processing_resolution=768, match_input_resolution=True,
                          batch_size=1, generator=gen)
        pred_depth = pipe_out.prediction[0, :, :, 0]
        pred_aligned, _, _ = align_depth_ls(gt_depth, pred_depth, valid_mask, max_resolution=max_res)
        if pred_aligned.shape != gt_depth.shape:
            import cv2
            pred_aligned = cv2.resize(pred_aligned, (gt_depth.shape[1], gt_depth.shape[0]))
        all_metrics.append(compute_errors(gt_depth, pred_aligned, valid_mask))

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print(f'max_res={max_res}: AbsRel={avg["abs_rel"]*100:.2f}%, d1={avg["delta1"]*100:.2f}%')

print(f'\nPaper DDIM: AbsRel=7.10%, d1=90.4%')
