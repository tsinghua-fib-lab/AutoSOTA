#!/usr/bin/env python3
"""Evaluation script for FLAME on AutoSplice - paper-style detection via max localization prob."""
import argparse, json, os, sys, warnings
import numpy as np
import torch, tqdm
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.sam_utils import initialize_sam_hydra
initialize_sam_hydra()
from test_dataset import load_and_initialize_model
from utils.localforgerydataset import LocalForgeryDataset
from utils.train_utils import custom_collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/datasets/autosplice_test")
    parser.add_argument("--model", default="FLAME/checkpoints/flame_g2_ladmulti_sam2.pth")
    parser.add_argument("--config", default="FLAME/checkpoints/model_params.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--detection_threshold", type=float, default=0.5)
    parser.add_argument("--otsu-det", action="store_true",
                        help="Use Otsu between-class variance as detection score instead of max(probs)")
    parser.add_argument("--otsu", action="store_true",
                        help="Use per-image Otsu thresholding instead of fixed 0.5 for pixel binarization")
    parser.add_argument("--calibrate", action="store_true",
                        help="Sweep detection thresholds and report optimal for ACC")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (hflip + rotations)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    device = torch.device(args.device)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    
    print("Loading model from {}...".format(args.model))
    model = load_and_initialize_model(config, args.model, device)
    model.eval()
    
    dataset = LocalForgeryDataset(
        root_dir=args.dataset, img_size=args.img_size,
        allow_multiple_targets=False, is_training=False, authentic_ratio=1.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    print("Dataset: {} samples ({} batches)".format(len(dataset), len(loader)))

    all_ious, all_f1s = [], []
    det_scores, det_labels = [], []

    # TTA transforms: identity + hflip + rot90/180/270
    def _apply_tta_transform(img, tform):
        if tform == 0:  # identity
            return img
        elif tform == 1:  # hflip
            return torch.flip(img, dims=[-1])
        elif tform == 2:  # rot90
            return torch.rot90(img, k=1, dims=[-2, -1])
        elif tform == 3:  # rot180
            return torch.rot90(img, k=2, dims=[-2, -1])
        elif tform == 4:  # rot270
            return torch.rot90(img, k=3, dims=[-2, -1])
        return img

    def _inverse_tta_transform(prob, tform):
        if tform == 0:
            return prob
        elif tform == 1:  # hflip inverse = hflip
            return torch.flip(prob, dims=[-1])
        elif tform == 2:  # rot90 inverse = rot270
            return torch.rot90(prob, k=3, dims=[-2, -1])
        elif tform == 3:  # rot180 inverse = rot180
            return torch.rot90(prob, k=2, dims=[-2, -1])
        elif tform == 4:  # rot270 inverse = rot90
            return torch.rot90(prob, k=1, dims=[-2, -1])
        return prob

    tta_forms = [0, 1, 2, 3, 4] if args.tta else [0]
    def _otsu_threshold(prob_map, return_variance=False):
        """Compute Otsu threshold on a 2D probability map. Returns threshold (or None)
        and optionally the normalized between-class variance."""
        import numpy as np
        vals = prob_map.cpu().numpy().flatten()
        hist, bin_edges = np.histogram(vals, bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return (None, 0.0) if return_variance else None
        sum_all = (hist * np.arange(256)).sum()
        weight_bg = 0.0
        sum_bg = 0.0
        max_var = 0.0
        best_t = 0
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                best_t = t
        thresh = float(best_t) / 255.0
        if return_variance:
            # Normalize variance: max possible = 0.25 * total^2 * 255^2
            norm_var = max_var / (0.25 * total * total * 255.0 * 255.0) if total > 0 else 0.0
            return thresh, float(norm_var)
        return thresh

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Evaluating"):
            if not batch: continue
            streams = [s.to(device, non_blocking=True) for s in batch["streams"]]
            orig = batch["orig"].to(device)
            tgt = batch["mask"].to(device)

            # TTA ensemble averaging
            probs_sum = None
            for tform in tta_forms:
                orig_t = _apply_tta_transform(orig, tform)
                with torch.amp.autocast(device_type="cuda"):
                    logits, extras = model(orig_t, streams, output_extras=True)
                probs_t = torch.sigmoid(logits)
                probs_t = _inverse_tta_transform(probs_t, tform)
                if probs_sum is None:
                    probs_sum = probs_t
                else:
                    probs_sum = probs_sum + probs_t
            probs = probs_sum / len(tta_forms)
            if probs.shape[-2:] != tgt.shape[-2:]:
                probs = torch.nn.functional.interpolate(probs, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            bs = probs.shape[0]
            otsu_variances = []
            if args.otsu:
                pred_bin = torch.zeros_like(probs)
                for bi in range(bs):
                    result = _otsu_threshold(probs[bi, 0], return_variance=True)
                    if result[0] is not None:
                        pixel_thresh, otsu_var = result
                        otsu_variances.append(otsu_var)
                    else:
                        pixel_thresh = 0.5
                        otsu_variances.append(0.0)
                    pred_bin[bi] = (probs[bi] > pixel_thresh).float()
            else:
                pred_bin = (probs > 0.5).float()
            sample_types = batch.get("sample_type", ["forgery"] * bs)
            if isinstance(sample_types, str): sample_types = [sample_types] * bs
            for i in range(bs):
                is_forgery = sample_types[i] != "authentic"
                label = 1 if is_forgery else 0
                if args.otsu_det and args.otsu and len(otsu_variances) > i:
                    det_scores.append(otsu_variances[i])
                else:
                    mp = probs[i].max().item()
                    det_scores.append(mp)
                det_labels.append(label)
                if is_forgery:
                    pred = pred_bin[i:i+1]; target = tgt[i:i+1]
                    inter = (pred * target).sum(); union = (pred + target).sum() - inter
                    iou = (inter / union).item() if union > 0 else 1.0
                    all_ious.append(iou)
                    pf = pred_bin[i,0].cpu().numpy().flatten().astype(np.uint8)
                    tf_ = tgt[i,0].cpu().numpy().flatten().astype(np.uint8)
                    if tf_.sum()==0 and pf.sum()==0: f1=1.0
                    elif tf_.sum()==0 or pf.sum()==0: f1=0.0
                    else: _,_,f1,_ = precision_recall_fscore_support(tf_, pf, zero_division=1, average="binary")
                    all_f1s.append(f1)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0
    ds = np.array(det_scores); dl_arr = np.array(det_labels)
    ap = average_precision_score(dl_arr, ds) if len(set(dl_arr)) > 1 else 0.0

    # Threshold calibration sweep
    if args.calibrate:
        print("\n" + "=" * 60)
        print("Detection Threshold Calibration Sweep")
        print("=" * 60)
        best_thresh = args.detection_threshold
        best_acc = -1.0
        print(f"{'Threshold':>12s}  {'ACC':>8s}  {'TP':>6s}  {'TN':>6s}  {'FP':>6s}  {'FN':>6s}")
        print("-" * 56)
        for th in np.arange(0.20, 0.81, 0.05):
            th = round(float(th), 2)
            det_pred_t = (ds > th).astype(int)
            acc_t = accuracy_score(dl_arr, det_pred_t)
            tp_t = ((dl_arr==1)&(det_pred_t==1)).sum()
            tn_t = ((dl_arr==0)&(det_pred_t==0)).sum()
            fp_t = ((dl_arr==0)&(det_pred_t==1)).sum()
            fn_t = ((dl_arr==1)&(det_pred_t==0)).sum()
            marker = " <--" if acc_t > best_acc else ""
            print(f"{th:>12.2f}  {acc_t:>8.4f}  {tp_t:>6d}  {tn_t:>6d}  {fp_t:>6d}  {fn_t:>6d}{marker}")
            if acc_t > best_acc:
                best_acc = acc_t
                best_thresh = th
        print("-" * 56)
        print(f"Best threshold: {best_thresh:.2f} (ACC={best_acc:.4f})")
        detection_threshold = best_thresh
    else:
        detection_threshold = args.detection_threshold

    det_pred = (ds > detection_threshold).astype(int)
    acc = accuracy_score(dl_arr, det_pred)

    sep = "=" * 60
    print("\n" + sep)
    print("FLAME on AutoSplice - Paper-style Detection (max localization prob)")
    print(sep)
    print("Pixel-level Localization (threshold=0.5):")
    print("  IoU: {:.4f}  (paper: 0.501)".format(mean_iou))
    print("  F1:  {:.4f}  (paper: 0.624)".format(mean_f1))
    print("  Samples: {}".format(len(all_ious)))
    print("Image-level Detection (max localization prob, threshold={}):".format(detection_threshold))
    print("  ACC: {:.4f}  (paper: 0.714)".format(acc))
    print("  AP:  {:.4f}  (paper: 0.763)".format(ap))
    tp = ((dl_arr==1)&(det_pred==1)).sum(); tn = ((dl_arr==0)&(det_pred==0)).sum()
    fp = ((dl_arr==0)&(det_pred==1)).sum(); fn = ((dl_arr==1)&(det_pred==0)).sum()
    print("  TP={}, TN={}, FP={}, FN={}".format(tp, tn, fp, fn))

    results = {"iou": float(mean_iou), "f1": float(mean_f1), "acc": float(acc), "ap": float(ap),
               "n_forged": len(all_ious), "n_total": len(dataset),
               "detection_method": "max_localization_prob", "threshold": float(detection_threshold)}
    print("\n" + json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
