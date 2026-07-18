import torch, numpy as np, os, sys
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
    return dict(abs_rel=float(np.mean(np.abs(gt-pred)/gt)), delta1=float((thresh<1.25).mean()))

pairs = []
with open('/repo/eth3d_filename_list_available.txt') as f:
    for line in f:
        line = line.strip()
        if line and line[0] != '#':
            parts = line.split()
            if len(parts) >= 2: pairs.append((parts[0], parts[1]))

data_dir = '/datasets/marigold_eval/eth3d'
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=torch.float16, variant='fp16', local_files_only=True)
pipe = pipe.to('cuda')

for steps in [2, 3, 5]:
    print('\n=== Steps={} ==='.format(steps))
    
    # DDIM baseline
    pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
    ddim_m = []
    for i, (r,d) in enumerate(pairs):
        rp = os.path.join(data_dir, r); dp = os.path.join(data_dir, d)
        if not os.path.exists(rp) or not os.path.exists(dp): continue
        img = Image.open(rp).convert('RGB'); gt = load_depth_binary(dp); vm = gt > 0
        if vm.sum() < 100: continue
        gen = torch.Generator(device='cuda'); gen.manual_seed(42+i)
        with torch.no_grad():
            out = pipe(img, num_inference_steps=steps, ensemble_size=1, processing_resolution=768,
                      match_input_resolution=True, batch_size=1, generator=gen)
        pred = align_depth_ls(gt, out.prediction[0,:,:,0], vm)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        ddim_m.append(compute_errors(gt, pred, vm))
    dd_a = np.mean([m['abs_rel'] for m in ddim_m])*100
    dd_d = np.mean([m['delta1'] for m in ddim_m])*100
    print('DDIM:        AbsRel={:.2f}% d1={:.2f}%'.format(dd_a, dd_d))
    
    # TSR with k=5.0
    orig = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
    tsr = TSR_DDIMScheduler.from_config(orig.config, k=5.0, psr_sigma=1.0, orig_scheduler=orig)
    pipe.scheduler = tsr
    tsr_m = []
    for i, (r,d) in enumerate(pairs):
        rp = os.path.join(data_dir, r); dp = os.path.join(data_dir, d)
        if not os.path.exists(rp) or not os.path.exists(dp): continue
        img = Image.open(rp).convert('RGB'); gt = load_depth_binary(dp); vm = gt > 0
        if vm.sum() < 100: continue
        gen = torch.Generator(device='cuda'); gen.manual_seed(42+i)
        with torch.no_grad():
            out = pipe(img, num_inference_steps=steps, ensemble_size=1, processing_resolution=768,
                      match_input_resolution=True, batch_size=1, generator=gen)
        pred = align_depth_ls(gt, out.prediction[0,:,:,0], vm)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        tsr_m.append(compute_errors(gt, pred, vm))
    ts_a = np.mean([m['abs_rel'] for m in tsr_m])*100
    ts_d = np.mean([m['delta1'] for m in tsr_m])*100
    print('TSR k=5.0:   AbsRel={:.2f}% d1={:.2f}%'.format(ts_a, ts_d))
    print('Improvement: AbsRel {:.2f}%, d1 {:.2f}%'.format(dd_a - ts_a, ts_d - dd_d))

print('\nPaper: DDIM=7.10% AbsRel, 90.4% d1 | TSR=6.68% AbsRel, 95.7% d1')
