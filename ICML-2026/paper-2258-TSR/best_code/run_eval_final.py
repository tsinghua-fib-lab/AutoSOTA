import torch, numpy as np, os, sys, argparse
from PIL import Image
sys.path.insert(0, '/repo')
from TSR_diffusers.TSR_schedulers import TSR_DDIMScheduler
from diffusers import MarigoldDepthPipeline, DDIMScheduler

ETH3D_H, ETH3D_W = 4032, 6048

def load_depth_binary(path):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(ETH3D_H, ETH3D_W).copy()
    data[~np.isfinite(data)] = 0.0
    return data

def align_depth_ls(gt_arr, pred_arr, valid_mask_arr, max_resolution=1024):
    gt = gt_arr.squeeze(); pred = pred_arr.squeeze(); mask = valid_mask_arr.squeeze()
    sf = np.min(max_resolution / np.array(gt.shape))
    if sf < 1:
        ds = torch.nn.Upsample(scale_factor=sf, mode='nearest')
        gt = ds(torch.as_tensor(gt).unsqueeze(0)).numpy()
        pred = ds(torch.as_tensor(pred).unsqueeze(0)).numpy()
        mask = ds(torch.as_tensor(mask).unsqueeze(0).float()).bool().numpy()
    gm = gt[mask].reshape((-1, 1)); pm = pred[mask].reshape((-1, 1))
    A = np.concatenate([pm, np.ones_like(pm)], axis=-1)
    s, t = np.linalg.lstsq(A, gm, rcond=None)[0]
    return pred_arr * float(s) + float(t)

def compute_errors(gt, pred, valid_mask=None):
    if valid_mask is None: valid_mask = gt > 0
    gt, pred = gt[valid_mask], pred[valid_mask]
    pred = np.clip(pred, 1e-6, None)
    thresh = np.maximum(gt/pred, pred/gt)
    return {
        'abs_rel': float(np.mean(np.abs(gt-pred)/gt)),
        'delta1': float((thresh<1.25).mean()),
        'delta2': float((thresh<1.25**2).mean()),
        'delta3': float((thresh<1.25**3).mean()),
        'rmse': float(np.sqrt(np.mean((gt-pred)**2))),
    }

def run_eval(pipe, pairs, data_dir, steps, seed, max_res=1024):
    all_metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(pairs):
        rgb_path = os.path.join(data_dir, rgb_rel)
        depth_path = os.path.join(data_dir, depth_rel)
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue
        try:
            img = Image.open(rgb_path).convert('RGB')
            gt = load_depth_binary(depth_path)
        except Exception:
            continue
        vm = gt > 0
        if vm.sum() < 100: continue
        gen = torch.Generator(device='cuda'); gen.manual_seed(seed + i)
        with torch.no_grad():
            out = pipe(img, num_inference_steps=steps, ensemble_size=1,
                      processing_resolution=768, match_input_resolution=True,
                      batch_size=1, generator=gen)
        pred = out.prediction[0,:,:,0]
        pred = align_depth_ls(gt, pred, vm, max_resolution=max_res)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        all_metrics.append(compute_errors(gt, pred, vm))
    return all_metrics

# Load pairs
pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2: pairs.append((parts[0], parts[1]))

data_dir = '/datasets/marigold_eval/eth3d'

# Load base pipeline
dtype = torch.float16
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=dtype, variant='fp16', local_files_only=True)
pipe = pipe.to('cuda')

# --- DDIM baseline ---
print('=== DDIM Baseline (no TSR) ===')
pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
ddim_metrics = run_eval(pipe, pairs, data_dir, steps=10, seed=42, max_res=1024)
avg = {k: np.mean([m[k] for m in ddim_metrics]) for k in ddim_metrics[0]}
print('DDIM AbsRel: {:.2f}%  d1: {:.2f}%  d2: {:.2f}%  d3: {:.2f}%  RMSE: {:.4f}'.format(
    avg['abs_rel']*100, avg['delta1']*100, avg['delta2']*100, avg['delta3']*100, avg['rmse']))

# --- TSR with different k values ---
for k in [1.5, 3.0, 5.0, 7.0, 10.0]:
    orig = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
    tsr = TSR_DDIMScheduler.from_config(orig.config, k=k, psr_sigma=1.0, orig_scheduler=orig)
    pipe.scheduler = tsr
    tsr_metrics = run_eval(pipe, pairs, data_dir, steps=10, seed=42, max_res=1024)
    avg = {k: np.mean([m[k] for m in tsr_metrics]) for k in tsr_metrics[0]}
    print('TSR k={:.1f}: AbsRel: {:.2f}%  d1: {:.2f}%  d2: {:.2f}%  d3: {:.2f}%  RMSE: {:.4f}'.format(
        k, avg['abs_rel']*100, avg['delta1']*100, avg['delta2']*100, avg['delta3']*100, avg['rmse']))

print('\nPaper ref: TSR=6.68% AbsRel, 95.7% d1  |  DDIM=7.10% AbsRel, 90.4% d1')
