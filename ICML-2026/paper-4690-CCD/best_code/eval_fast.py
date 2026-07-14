"""Fast sigma sweep evaluation."""
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import json, glob, sys
import pandas as pd

def fast_eval(data_dir, label):
    map_files = sorted(glob.glob(f'{data_dir}/prompt_*_cov.npy'))
    print(f'\n=== {label}: {len(map_files)} TV maps ===')
    maps_raw = np.array([np.load(f) for f in map_files])

    maps_256 = np.zeros((len(maps_raw), 256, 256), dtype=np.float32)
    for i in range(len(maps_raw)):
        img = Image.fromarray(maps_raw[i].astype(np.float32), mode='F')
        img = img.resize((256, 256), Image.BILINEAR)
        maps_256[i] = np.array(img)

    maps_log = np.log1p(maps_256)
    lo_v = np.percentile(maps_log, 2, axis=(1,2), keepdims=True)
    hi_v = np.percentile(maps_log, 98, axis=(1,2), keepdims=True)
    maps_clipped = np.clip(maps_log, lo_v, hi_v)
    denom = hi_v - lo_v; denom[denom <= 1e-8] = 1.0
    norm_maps = np.clip((maps_clipped - lo_v) / denom, 0, 1)

    metadata = pd.read_parquet('templates/metadata.parquet')
    with open('sdv1-4_bb_attack_gt_verify_TV.jsonl') as f:
        tv_jsonl = [json.loads(line) for line in f]
    from utils.data_loaders import load_tv_data
    _, masks = load_tv_data(data_dir, tv_jsonl, metadata, metric_name='cov')
    masks = np.array(masks)

    thresholds = np.linspace(-1e-6, 1+1e-6, 201)
    best_overall_iou = 0; best_overall_config = ''

    for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        if sigma > 0:
            smoothed = np.array([gaussian_filter(m, sigma=sigma) for m in norm_maps])
        else:
            smoothed = norm_maps

        best_iou = 0; best_miou = 0; best_acc = 0
        for th in thresholds:
            pred = (smoothed > th)
            gt = (masks > 0.5)
            inter = np.logical_and(pred, gt).sum(axis=(1,2))
            union = np.logical_or(pred, gt).sum(axis=(1,2))
            iou_per = np.ones(len(inter)); valid = union > 0
            iou_per[valid] = inter[valid] / union[valid]
            miou = iou_per.mean()
            if miou > best_miou: best_miou = miou
            g_iou = inter.sum() / union.sum() if union.sum() > 0 else 1.0
            if g_iou > best_iou: best_iou = g_iou
            acc = (pred == gt).mean()
            if acc > best_acc: best_acc = acc

        if best_iou > best_overall_iou:
            best_overall_iou = best_iou
            best_overall_config = f'sigma={sigma:.1f}'
        print(f'  sigma={sigma:.1f}: IoU={best_iou:.4f}, mIoU={best_miou:.4f}, ACC={best_acc:.4f}')

    print(f'  BEST: {best_overall_config} -> IoU={best_overall_iou:.4f}')
    return best_overall_iou

fast_eval('metrics_outputs_v1/TV_metric_maps', 'K=16')
fast_eval('metrics_outputs_v1/TV_metric_maps_k24', 'K=24')
fast_eval('metrics_outputs_v1/TV_metric_maps_k32', 'K=32')
