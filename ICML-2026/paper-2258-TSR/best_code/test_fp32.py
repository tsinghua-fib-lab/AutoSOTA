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
    gt = gt_arr.squeeze(); pred = pred_arr.squeeze(); mask = valid_mask_arr.squeeze()
    if max_resolution is not None:
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

test_pairs = pairs[:5]  # 5 images for speed

# Test fp32
print('=== Testing fp32 precision ===')
pipe = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=torch.float32, variant=None, local_files_only=True)
pipe.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
pipe = pipe.to('cuda')

for steps in [10, 20]:
    all_metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(test_pairs):
        rgb_path = os.path.join('/datasets/marigold_eval/eth3d', rgb_rel)
        depth_path = os.path.join('/datasets/marigold_eval/eth3d', depth_rel)
        img = Image.open(rgb_path).convert('RGB')
        gt = load_depth_binary(depth_path)
        vm = gt > 0
        gen = torch.Generator(device='cuda'); gen.manual_seed(42 + i)
        with torch.no_grad():
            out = pipe(img, num_inference_steps=steps, ensemble_size=1,
                      processing_resolution=768, match_input_resolution=True,
                      batch_size=1, generator=gen)
        pred = out.prediction[0,:,:,0]
        pred = align_depth_ls(gt, pred, vm, max_resolution=1024)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        all_metrics.append(compute_errors(gt, pred, vm))
    avg_a = np.mean([m['abs_rel'] for m in all_metrics]) * 100
    avg_d = np.mean([m['delta1'] for m in all_metrics]) * 100
    print('fp32 steps={}: AbsRel={:.2f}% d1={:.2f}%'.format(steps, avg_a, avg_d))

# Also test with different processing_res
print('\n=== Testing lower processing_res ===')
pipe2 = MarigoldDepthPipeline.from_pretrained(
    '/models/marigold-v1-0', torch_dtype=torch.float16, variant='fp16', local_files_only=True)
pipe2.scheduler = DDIMScheduler.from_pretrained('/models/marigold-v1-0', subfolder='scheduler')
pipe2 = pipe2.to('cuda')

for res in [384, 512, 768]:
    all_metrics = []
    for i, (rgb_rel, depth_rel) in enumerate(test_pairs):
        rgb_path = os.path.join('/datasets/marigold_eval/eth3d', rgb_rel)
        depth_path = os.path.join('/datasets/marigold_eval/eth3d', depth_rel)
        img = Image.open(rgb_path).convert('RGB')
        gt = load_depth_binary(depth_path)
        vm = gt > 0
        gen = torch.Generator(device='cuda'); gen.manual_seed(42 + i)
        with torch.no_grad():
            out = pipe2(img, num_inference_steps=10, ensemble_size=1,
                       processing_resolution=res, match_input_resolution=True,
                       batch_size=1, generator=gen)
        pred = out.prediction[0,:,:,0]
        pred = align_depth_ls(gt, pred, vm, max_resolution=1024)
        if pred.shape != gt.shape:
            import cv2; pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        all_metrics.append(compute_errors(gt, pred, vm))
    avg_a = np.mean([m['abs_rel'] for m in all_metrics]) * 100
    avg_d = np.mean([m['delta1'] for m in all_metrics]) * 100
    print('res={}: AbsRel={:.2f}% d1={:.2f}%'.format(res, avg_a, avg_d))

print('\nPaper DDIM: AbsRel=7.10%, d1=90.4%')
