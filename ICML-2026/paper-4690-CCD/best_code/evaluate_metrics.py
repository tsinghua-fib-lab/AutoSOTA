
import os
import glob
import numpy as np
import pandas as pd
import argparse
from PIL import Image

from utils.data_loaders import load_tv_data, load_nmem_data

def scan_thresholds(maps, masks, name="Dataset"):
    maps = np.array(maps)
    masks = np.array(masks)
    if len(maps) == 0:
        print(f"\n[{name}] No data found.")
        return
        
    print(f"\n[{name}] Total Samples: {len(maps)}")

    # IDEA-04: Per-sample rank-based normalization
    # Try different normalization strategies
    norm_strategies = {}
    maps_log_all = np.log1p(maps)

    # Strategy A: Global min-max (baseline)
    min_val = maps_log_all.min()
    max_val = maps_log_all.max()
    norm_strategies["global"] = np.clip(
        (maps_log_all - min_val) / (max_val - min_val + 1e-8), 0, 1)

    # Strategy B: Per-sample percentile normalization with various bounds
    for lo, hi in [(1, 99), (0.5, 99.5), (2, 98)]:
        key = f"persample_p{lo}_{hi}"
        lo_v = np.percentile(maps_log_all, lo, axis=(1, 2), keepdims=True)
        hi_v = np.percentile(maps_log_all, hi, axis=(1, 2), keepdims=True)
        maps_clipped = np.clip(maps_log_all, lo_v, hi_v)
        denom = hi_v - lo_v
        denom[denom <= 1e-8] = 1.0  # avoid division by zero
        normed = np.clip((maps_clipped - lo_v) / denom, 0, 1)
        norm_strategies[key] = normed

    # IDEA-06: Gaussian pre-smoothing sweep (extended range)
    from scipy.ndimage import gaussian_filter
    sigma_candidates = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Only use best norm strategy + swap for ALL
    norm_strategies_restricted = {"persample_p2_98": norm_strategies["persample_p2_98"],
                                   "global": norm_strategies["global"]}

    eps = 1e-6
    thresholds = np.linspace(0.0 - eps, 1.0 + eps, 201)
    
    from scipy.ndimage import binary_opening

    # Use best config from iter-3 as base (per-sample [2,98] percentile norm)
    base_key = "persample_p2_98"
    base_norm = norm_strategies[base_key]
    base_sigma = 2.0
    smoothed = np.array([gaussian_filter(m, sigma=base_sigma) for m in base_norm])

    best_miou = 0.0
    best_global_iou = 0.0
    best_acc = 0.0
    best_cc = 0

    # IDEA-07: Connected-component filter sweep on top of best base config
    cc_candidates = [0, 1, 2]  # 0=no filtering, 1=3x3 opening, 2=5x5 opening

    for th in thresholds:
        pred_raw = (smoothed > th)

        for cc_iter in cc_candidates:
            if cc_iter == 0:
                pred = pred_raw
            elif cc_iter == 1:
                pred = np.array([binary_opening(p, structure=np.ones((3,3)), iterations=1) for p in pred_raw])
            else:
                pred = np.array([binary_opening(p, structure=np.ones((5,5)), iterations=1) for p in pred_raw])

            gt = (masks > 0.5)

            inter_per_sample = np.logical_and(pred, gt).sum(axis=(1, 2))
            union_per_sample = np.logical_or(pred, gt).sum(axis=(1, 2))

            iou_per_sample = np.ones_like(inter_per_sample, dtype=float)
            valid = (union_per_sample > 0)
            iou_per_sample[valid] = inter_per_sample[valid] / union_per_sample[valid]
            miou = iou_per_sample.mean()
            if miou > best_miou:
                best_miou = miou
                best_cc = cc_iter

            total_inter = inter_per_sample.sum()
            total_union = union_per_sample.sum()
            g_iou = total_inter / total_union if total_union > 0 else 1.0
            if g_iou > best_global_iou:
                best_global_iou = g_iou
                best_cc = cc_iter

            acc = (pred == gt).mean()
            if acc > best_acc:
                best_acc = acc
                best_cc = cc_iter

    print(f"  IoU (Global):  {best_global_iou:.4f}")
    print(f"  mIoU (Mean):   {best_miou:.4f}")
    print(f"  Accuracy:      {best_acc:.4f}")
    print(f"  Best config:   norm={base_key}, sigma={base_sigma:.1f}, cc_opening={best_cc}")

def evaluate_metrics(data_dir, metric_suffix="score_diff", dataset_path="sdv1-4_bb_attack_gt_verify_TV.jsonl"):
    print(f"Evaluating Metrics for metric: {metric_suffix}")
    
    try:
        import pandas as pd
        metadata_path = "templates/metadata.parquet"
        nmem_file = "data/sd1_nmem.txt" if "sdv1" in dataset_path else "data/sd2_nmem.txt"
        
        metadata = pd.read_parquet(metadata_path)
        
        import json
        with open(dataset_path, "r") as f:
            tv_jsonl = [json.loads(line) for line in f]
            
        nmem_prompts = []
        if os.path.exists(nmem_file):
            with open(nmem_file, "r") as f:
                nmem_prompts = [line.strip() for line in f if line.strip()]
                
        # 1. Load TV data
        tv_maps, tv_masks = load_tv_data(data_dir, tv_jsonl, metadata, metric_name=metric_suffix)
        
        # 2. Load Nmem data
        nmem_maps, nmem_masks = load_nmem_data(data_dir, nmem_prompts, metric_name=metric_suffix)
        
        # 3. Load MVRV data (if available)
        mvrv_dataset_path = dataset_path.replace("_TV.jsonl", "_MVRV.jsonl")
        mvrv_dir = data_dir.replace("TV_metric_maps", "MVRV_metric_maps")
        mvrv_maps, mvrv_masks = [], []
        
        if os.path.exists(mvrv_dataset_path) and os.path.exists(mvrv_dir):
            with open(mvrv_dataset_path, "r") as f:
                mvrv_jsonl = [json.loads(line) for line in f]
            mvrv_maps, mvrv_masks = load_tv_data(mvrv_dir, mvrv_jsonl, metadata, metric_name=metric_suffix)
        
        # --- Evaluate TV Only ---
        scan_thresholds(tv_maps, tv_masks, name="TV Only")
        
        # --- Evaluate ALL (TV + MV/RV + Nmem) ---
        # pos_maps = list(tv_maps) + list(mvrv_maps)
        # pos_masks = list(tv_masks) + list(mvrv_masks)
        # neg_maps = list(nmem_maps)
        # neg_masks = list(nmem_masks)
        # 
        # if len(pos_maps) > 0 and len(neg_maps) > 0:
        #     import random
        #     min_count = min(len(pos_maps), len(neg_maps))
        #     
        #     pos_combined = list(zip(pos_maps, pos_masks))
        #     neg_combined = list(zip(neg_maps, neg_masks))
        #     
        #     random.seed(42)
        #     random.shuffle(pos_combined)
        #     random.shuffle(neg_combined)
        #     
        #     bal_pos_maps, bal_pos_masks = zip(*pos_combined[:min_count])
        #     bal_neg_maps, bal_neg_masks = zip(*neg_combined[:min_count])
        #     
        #     all_maps = list(bal_pos_maps) + list(bal_neg_maps)
        #     all_masks = list(bal_pos_masks) + list(bal_neg_masks)
        #     
        #     scan_thresholds(all_maps, all_masks, name=f"ALL (Balanced 1:1, {min_count} Pos vs {min_count} Neg)")
        # else:
        
        # Currently, data sizes are small (MVRV=32), so we evaluate all without balancing.
        all_maps = list(tv_maps) + list(nmem_maps) + list(mvrv_maps)
        all_masks = list(tv_masks) + list(nmem_masks) + list(mvrv_masks)
        scan_thresholds(all_maps, all_masks, name="ALL (Unbalanced)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="metrics_outputs_v1/TV_metric_maps")
    parser.add_argument("--metric_name", type=str, default="cov", help="Name of the metric to evaluate (e.g. cov, score_diff, cov_bad)")
    parser.add_argument("--dataset", type=str, default="sdv1-4_bb_attack_gt_verify_TV.jsonl", help="Dataset jsonl file")
    args = parser.parse_args()
    evaluate_metrics(args.data_dir, metric_suffix=args.metric_name, dataset_path=args.dataset)
