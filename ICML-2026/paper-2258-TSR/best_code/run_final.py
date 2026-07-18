import torch, numpy as np, os, sys, json
from PIL import Image
from tqdm import tqdm
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
    }

def run(pipe, pairs, data_dir, steps, seed):
    metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(tqdm(pairs, desc='Eval')):
        rp = os.path.join(data_dir, rgb_rel); dp = os.path.join(data_dir, depth_rel)
        if not os.path.exists(rp) or not os.path.exists(dp): continue
        try:
            img = Image.open(rp).convert('RGB'); gt = load_depth_binary(dp)
        except: continue
        vm = gt > 0
        if vm.sum() < 100: continue
        gen = torch.Generator(device='cuda'); gen.manual_seed(seed + i)
        with torch.no_grad():
            out = pipe(img, num_inference_steps=steps, ensemble_size=10,
                      processing_resolution=768, match_input_resolution=True, batch_size=1, generator=gen)
        pred = out.prediction[0,:,:,0]
        pred = align_depth_ls(gt, pred, vm)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        metrics.append(compute_errors(gt, pred, vm))
    return metrics

# Load data
pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2: pairs.append((parts[0], parts[1]))
print('Total images:', len(pairs))
data_dir = '/datasets/marigold_eval/eth3d'

# Load pipeline
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=torch.float16, variant='fp16', local_files_only=True)
pipe = pipe.to('cuda')

# DDIM baseline
print('\n=== DDIM Baseline ===')
pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
ddim = run(pipe, pairs, data_dir, steps=10, seed=42)
dd_a = np.mean([m['abs_rel'] for m in ddim]) * 100
dd_d = np.mean([m['delta1'] for m in ddim]) * 100
print('DDIM: AbsRel={:.2f}%  d1={:.2f}%'.format(dd_a, dd_d))

# TSR k=1.5, sigma=1.0
print('\n=== TSR k=1.5 sigma=0.5 ===')
orig = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
tsr = TSR_DDIMScheduler.from_config(orig.config, k=1.5, psr_sigma=0.5, orig_scheduler=orig, use_adaptive_k=False, k_power=0.5)
pipe.scheduler = tsr
tsr_m = run(pipe, pairs, data_dir, steps=10, seed=42)
ts_a = np.mean([m['abs_rel'] for m in tsr_m]) * 100
ts_d = np.mean([m['delta1'] for m in tsr_m]) * 100
print('TSR:  AbsRel={:.2f}%  d1={:.2f}%'.format(ts_a, ts_d))

print('\n=== Summary ===')
print('DDIM baseline: AbsRel={:.2f}%  d1={:.2f}%'.format(dd_a, dd_d))
print('TSR k=1.5:     AbsRel={:.2f}%  d1={:.2f}%'.format(ts_a, ts_d))
print('Improvement:   AbsRel {:.3f}%  d1 {:.3f}%'.format(dd_a - ts_a, ts_d - dd_d))
print('\nPaper ref: DDIM=7.10% AbsRel, 90.4% d1 | TSR=6.68% AbsRel, 95.7% d1')
print('Paper CI: AbsRel [6.666, 6.82], d1 [95.6, 95.71]')
print()

# Save results
results = {
    'ddim': {'abs_rel_pct': dd_a, 'delta1_pct': dd_d},
    'tsr': {'abs_rel_pct': ts_a, 'delta1_pct': ts_d, 'k': 1.5, 'sigma': 1.0},
    'n_images': len(pairs),
    'steps': 10,
    'seed': 42,
}
with open('/repo/output/eth3d_eval/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to /repo/output/eth3d_eval/results.json')
