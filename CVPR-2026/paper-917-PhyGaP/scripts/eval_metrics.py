# calculate psnr, ssim, and lpips metrics between predicted and ground truth 
# if the predicted images and ground truth images contain background, do not provide the background masks
# The background of the predicted images and ground truth images should be the same, black by default.
# input: predicted images folder, ground truth images folder, mask folder
# output: metrics printed to console
import os
import sys
import argparse
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

def calculate_psnr(pred, gt, mask=None):
    # pred, gt: torch.Tensor, shape (H, W) or (C, H, W), values in [0, 255] or [0, 1]
    if pred.max() <= 1.0:
        pred = pred * 255.0
    if gt.max() <= 1.0:
        gt = gt * 255.0

    pred = pred.float()
    gt = gt.float()

    if mask is not None:
        mask = mask.float()
        
        mse = torch.sum(((gt - pred) ** 2)) / torch.sum(mask*3)
    else:
        mse = torch.mean((gt - pred) ** 2)

    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))

def calculate_ssim(pred, gt):
    return ssim(pred, gt).item()

def calculate_lpips(pred, gt):
    return lpips(pred, gt, net_type='vgg').item()

def load_images(folder, exts=['png']):

    images = []
    if not os.path.isdir(folder):
        return images
    for ext in exts:
        for file in sorted(os.listdir(folder)):
            if file.lower().endswith(ext):
                img_path = os.path.join(folder, file)
                img = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
                images.append(img)
    return images

def evaluate_metrics(pred_folder, gt_folder, mask_folder=None):
    # Load as lists first to validate counts before stacking
    pred_list = load_images(pred_folder)  # list of CxHxW tensors
    gt_list = load_images(gt_folder)

    if len(pred_list) != len(gt_list):
        print(f"ERROR: #pred ({len(pred_list)}) != #gt ({len(gt_list)}). Folders: pred={pred_folder}, gt={gt_folder}")
        raise ValueError("Number of predicted images and ground truth images must match.")

    mask_list = None
    if mask_folder and os.path.isdir(mask_folder):
        mask_list = load_images(mask_folder,exts=['jpg','png'])
        if len(mask_list) != len(pred_list):
            print(f"ERROR: #mask ({len(mask_list)}) != #pred/gt ({len(pred_list)}). Mask folder: {mask_folder}")
            raise ValueError("Number of mask images must match number of predicted/ground-truth images.")

    # Now stack into tensors for vectorized processing and clamp to [0,1]
    pred_images = torch.clamp(torch.stack(pred_list), 0.0, 1.0)
    gt_images = torch.clamp(torch.stack(gt_list), 0.0, 1.0)
    mask_images = None if mask_list is None else torch.clamp(torch.stack(mask_list), 0.0, 1.0)

    psnrs = []
    ssims = []
    lpipss = []

    if mask_images is None:
        iterator = zip(pred_images, gt_images)
    else:
        iterator = zip(pred_images, gt_images, mask_images)

    for items in tqdm.tqdm(iterator):
        if mask_images is None:
            pred, gt = items
            m = None
        else:
            pred, gt, mask = items
            # Build a 1xHxW binary mask and apply immediately to pred/gt
            if mask.shape[0] == 4:
                m = mask[3:4, :, :]
            elif mask.shape[0] == 3:
                # luminance
                m = (0.299 * mask[0:1] + 0.587 * mask[1:2] + 0.114 * mask[2:3])
            elif mask.shape[0] >= 1:
                m = mask[0:1, :, :]
            else:
                m = None
            if m is not None:
                # m = (m > 0.5).float()
                # pred = pred * m
                if m is not None:
                    m = (m > 0.5).float()
                    # resize mask to match pred/gt if needed (nearest to preserve binary)
                    if m.shape[1] != pred.shape[1] or m.shape[2] != pred.shape[2]:
                        import torch.nn.functional as F
                        m = F.interpolate(m.unsqueeze(0), size=(pred.shape[1], pred.shape[2]), mode='nearest')[0]
                    pred = pred * m
                    gt = gt * m
        # import pdb; pdb.set_trace()
        psnr_value = calculate_psnr(pred, gt, m if 'm' in locals() else None)
        ssim_value = calculate_ssim(pred, gt)
        lpips_value = calculate_lpips(pred, gt)

        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        lpipss.append(lpips_value)

    print(f'PSNR: {np.mean(psnrs):.4f}, SSIM: {np.mean(ssims):.4f}, LPIPS: {np.mean(lpipss):.4f}')
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss), len(psnrs)


# -------------------------- Normal-specific utilities --------------------------

SUPPORTED_EXTS = [".png", ".jpg", ".jpeg"]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _list_files(folder: str, exts: Optional[List[str]] = None) -> List[str]:
    if not os.path.isdir(folder):
        return []
    if exts is None:
        exts = SUPPORTED_EXTS
    exts = tuple([e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts])
    return sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts])


def _normalize_numeric_stem(stem: str) -> str:
    # If stem is purely digits (with potential leading zeros), normalize by int casting
    if stem.isdigit():
        try:
            return str(int(stem))
        except Exception:
            return stem
    # Otherwise, try to extract a trailing number token and normalize it, fallback to original
    # e.g., img_0001 -> img_1
    num = ""
    i = len(stem) - 1
    while i >= 0 and stem[i].isdigit():
        num = stem[i] + num
        i -= 1
    if num:
        prefix = stem[: i + 1]
        return f"{prefix}{int(num)}"
    return stem


def _build_key_map(folder: str, files: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        key = _normalize_numeric_stem(stem)
        mapping[key] = f
    return mapping


def _read_image_hwc_float01(path: str) -> np.ndarray:
    # Use PIL for broad format support of L/RGB/RGBA; returns float32 HxWxC in [0,1]
    img = Image.open(path)
    img = img.convert("RGBA") if img.mode in ("LA", "P") else img.convert("RGB") if img.mode != "RGBA" else img
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # If RGBA, keep all 4 channels (for mask we may use alpha); for normals we only need first 3
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _read_mask_bool(path: str) -> np.ndarray:
    arr = _read_image_hwc_float01(path)
    if arr.shape[-1] == 4:
        alpha = arr[..., 3]
        mask = alpha > 0.5
    elif arr.shape[-1] == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        mask = gray > 0.5
    else:
        mask = arr[..., 0] > 0.5
    return mask


def _decode_normal_from_img(img_hwc_float01: np.ndarray) -> np.ndarray:
    # Take first 3 channels
    rgb = img_hwc_float01[..., :3]
    n = rgb * 2.0 - 1.0
    # Normalize to unit
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n = n / norm
    return n


def ensure_gt_normals(exp_root: str,
                      data_root: str,
                      data_normal_folder: str = "normal",
                      exp_normal_folder: str = "normal",
                      gt_normal_folder: str = "gt_normal",
                      exts: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """
    Ensure exp_root/gt_normal exists. If not present or empty, copy matching files from
    data_root/{data_normal_folder} that correspond (by name ignoring zero padding width)
    to files in exp_root/{exp_normal_folder}. Returns (gt_dir, sorted_exp_files).
    """
    # import pdb; pdb.set_trace()
    if exts is None:
        exts = [e[1:] if e.startswith('.') else e for e in SUPPORTED_EXTS]
    
    exp_norm_dir = os.path.join(exp_root, exp_normal_folder)
    gt_dir = os.path.join(exp_root, gt_normal_folder)
    _ensure_dir(exp_norm_dir)
    exp_files = _list_files(exp_norm_dir, exts)
    # import pdb;pdb.set_trace()
    # Create/verify gt dir
    need_copy = (not os.path.isdir(gt_dir)) or (len(_list_files(gt_dir, exts)) == 0)
    if need_copy:
        _ensure_dir(gt_dir)
        src_dir = os.path.join(data_root, data_normal_folder)
        src_files = _list_files(src_dir, exts)
        src_map = _build_key_map(src_dir, src_files)

        copied, missing = 0, 0
        # import pdb;pdb.set_trace()
        for f in exp_files:
            key = _normalize_numeric_stem(os.path.splitext(f)[0])
            src_name = src_map.get(key)
            if src_name is None:
                missing += 1
                continue
            shutil.copyfile(os.path.join(src_dir, src_name), os.path.join(gt_dir, f))
            copied += 1
        print(f"[gt_normal] prepared at {gt_dir} | copied: {copied}, missing: {missing}")
    else:
        print(f"[gt_normal] found existing at {gt_dir}")

    return gt_dir, exp_files


def ensure_masks(exp_root: str,
                 data_root: Optional[str],
                 exp_mask_folder: str = "mask",
                 data_mask_folder: str = "masks",
                 ref_folders: Optional[List[str]] = None,
                 exts: Optional[List[str]] = None) -> Tuple[str, int, int]:
    """
    Ensure masks exist under exp_root/exp_mask_folder. If the folder is missing or empty,
    copy masks from data_root/data_mask_folder to match the filenames in a reference folder
    within exp_root (e.g., 'pred' preferred, then 'normal', then 'gt'), using zero-padding-agnostic
    numeric matching like ensure_gt_normals.

    Returns (exp_mask_dir, copied_count, missing_count).
    """
    # import pdb; pdb.set_trace()
    if data_root is None:
        # Nothing to copy from
        exp_mask_dir = exp_mask_folder if os.path.isabs(exp_mask_folder) else os.path.join(exp_root, exp_mask_folder)
        os.makedirs(exp_mask_dir, exist_ok=True)
        return exp_mask_dir, 0, 0

    if exts is None:
        exts = [e[1:] if e.startswith('.') else e for e in SUPPORTED_EXTS]
    # Where to put masks
    exp_mask_dir = exp_mask_folder if os.path.isabs(exp_mask_folder) else os.path.join(exp_root, exp_mask_folder)
    os.makedirs(exp_mask_dir, exist_ok=True)
    # If already has files, skip
    existing = _list_files(exp_mask_dir, exts)
    if len(existing) > 0:
        return exp_mask_dir, 0, 0

    # Source masks
    src_dir = os.path.join(data_root, data_mask_folder)
    src_files = _list_files(src_dir, exts)
    src_map = _build_key_map(src_dir, src_files)

    # Reference files to determine which masks to copy
    if ref_folders is None:
        ref_folders = ["pred", "normal", "gt"]
    ref_files: List[str] = []
    ref_dir_used = None
    for rf in ref_folders:
        cand = os.path.join(exp_root, rf) if not os.path.isabs(rf) else rf
        files = _list_files(cand, exts)
        if len(files) > 0:
            ref_files = files
            ref_dir_used = cand
            break

    if not ref_files:
        print(f"[mask] No reference files found in {ref_folders} under {exp_root}. Skipping mask copy.")
        return exp_mask_dir, 0, 0

    copied, missing = 0, 0
    for f in ref_files:
        key = _normalize_numeric_stem(os.path.splitext(f)[0])
        src_name = src_map.get(key)
        if src_name is None:
            missing += 1
            continue
        src_path = os.path.join(src_dir, src_name)
        # Always place as PNG to be compatible with loaders expecting .png
        dst_name = os.path.splitext(f)[0] + ".png"
        dst_path = os.path.join(exp_mask_dir, dst_name)
        # If source already PNG, copy bytes; else re-encode to PNG via PIL for safety
        if os.path.splitext(src_name)[1].lower() == ".png":
            shutil.copyfile(src_path, dst_path)
        else:
            try:
                img = Image.open(src_path)
                img.save(dst_path, format="PNG")
            except Exception:
                # Fallback to raw copy with original extension
                dst_path = os.path.join(exp_mask_dir, src_name)
                shutil.copyfile(src_path, dst_path)
        copied += 1

    print(f"[mask] prepared at {exp_mask_dir} | copied: {copied}, missing: {missing} | ref={ref_dir_used}")
    return exp_mask_dir, copied, missing


def evaluate_normal_metrics(exp_root: str,
                            data_root: str,
                            data_normal_folder: str = "normal",
                            exp_normal_folder: str = "normal",
                            gt_normal_folder: str = "gt_normal",
                            mask_folder: str = "mask",
                            exts: Optional[List[str]] = None,
                            mask_exts: Optional[List[str]] = None) -> Tuple[float, float, float, int]:
    if exts is None:
        exts = [e[1:] if e.startswith('.') else e for e in SUPPORTED_EXTS]
    if mask_exts is None:
        mask_exts = exts

    gt_dir, exp_files = ensure_gt_normals(
        exp_root=exp_root,
        data_root=data_root,
        data_normal_folder=data_normal_folder,
        exp_normal_folder=exp_normal_folder,
        gt_normal_folder=gt_normal_folder,
        exts=exts,
    )

    exp_norm_dir = os.path.join(exp_root, exp_normal_folder)
    mask_dir = os.path.join(exp_root, mask_folder) if not os.path.isabs(mask_folder) else mask_folder

    cos_dists: List[float] = []
    maes: List[float] = []
    accs_thresh_11_25: List[float] = []  # threshold accuracy @ 11.25° per-image

    valid_img_count = 0
    # import pdb;pdb.set_trace()
    for fname in tqdm.tqdm(exp_files, desc="Normals Eval"):
        pred_path = os.path.join(exp_norm_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.isfile(gt_path):
            # Skip if GT missing (already reported during copy)
            continue

        pred_img = _read_image_hwc_float01(pred_path)
        gt_img = _read_image_hwc_float01(gt_path)
        if pred_img.shape[:2] != gt_img.shape[:2]:
            # Resize gt to pred size if needed
            gt_pil = Image.fromarray((gt_img[..., :3] * 255.0).astype(np.uint8))
            gt_pil = gt_pil.resize((pred_img.shape[1], pred_img.shape[0]), Image.BILINEAR)
            gt_img = np.array(gt_pil).astype(np.float32) / 255.0
            if gt_img.ndim == 2:
                gt_img = gt_img[:, :, None]

        # Optional mask per file (same filename in mask_dir if present)
        mask_bool: Optional[np.ndarray] = None
        if os.path.isdir(mask_dir):
            mask_path = None
            for ext in mask_exts:
                candidate = os.path.join(mask_dir, os.path.splitext(fname)[0] + (ext if ext.startswith('.') else f'.{ext}'))
                if os.path.isfile(candidate):
                    mask_path = candidate
                    break
            if mask_path is not None:
                mask_bool = _read_mask_bool(mask_path)
                if mask_bool.shape != pred_img.shape[:2]:
                    m_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))
                    m_pil = m_pil.resize((pred_img.shape[1], pred_img.shape[0]), Image.NEAREST)
                    mask_bool = np.array(m_pil) > 127

        n_pred = _decode_normal_from_img(pred_img)
        n_gt = _decode_normal_from_img(gt_img)

        # Build valid region
        if mask_bool is None:
            valid = np.ones(n_pred.shape[:2], dtype=bool)
        else:
            valid = mask_bool.astype(bool)

        # Flatten masked pixels
        vp = n_pred[valid]
        vg = n_gt[valid]
        if vp.size == 0:
            continue
        # Cosine similarity
        dots = np.abs(np.sum(vp * vg, axis=-1))
        dots = np.clip(dots, -1.0, 1.0)
        cos_dist = 1.0 - dots  # cosine distance per-pixel
        ang = np.degrees(np.arccos(dots))  # degrees

        cos_dists.append(float(cos_dist.mean()))
        maes.append(float(ang.mean()))
        # Threshold accuracy @ 11.25°: fraction of valid pixels with angular error <= 11.25°
        acc_11_25 = float(np.mean(ang <= 11.25))
        accs_thresh_11_25.append(acc_11_25)
        valid_img_count += 1

    mean_cos = float(np.mean(cos_dists)) if cos_dists else float('nan')
    mean_mae = float(np.mean(maes)) if maes else float('nan')
    mean_acc_11_25 = float(np.mean(accs_thresh_11_25)) if accs_thresh_11_25 else float('nan')
    print(f"Normal Cosine Distance (mean): {mean_cos:.6f} | MAE (deg, mean): {mean_mae:.4f} | Acc@11.25°: {mean_acc_11_25:.4f}")
    return mean_cos, mean_mae, mean_acc_11_25, valid_img_count


def _write_metrics_csv(csv_path: str,
                       model_name: Optional[str],
                       dataset_name: Optional[str],
                       exp_name: Optional[str],
                       type_str: str,
                       normal_cos: Optional[float],
                       normal_mae: Optional[float],
                       normal_acc_11_25: Optional[float],
                       rgb_psnr: Optional[float],
                       rgb_ssim: Optional[float],
                       rgb_lpips: Optional[float]) -> None:
    if not csv_path:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    import csv
    header = [
        "model_name",
        "dataset_name",
        "exp_name",
        "type",
        "normal_cosine_distance_mean",
        "normal_mae_deg_mean",
        "normal_acc_11_25",
        "psnr",
        "ssim",
        "lpips",
    ]
    row = [
        model_name or "",
        dataset_name or "",
        exp_name or "",
        type_str,
        None if normal_cos is None else float(normal_cos),
        None if normal_mae is None else float(normal_mae),
        None if normal_acc_11_25 is None else float(normal_acc_11_25),
        None if rgb_psnr is None else float(rgb_psnr),
        None if rgb_ssim is None else float(rgb_ssim),
        None if rgb_lpips is None else float(rgb_lpips),
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image metrics and/or normal metrics.")
    # Normal metrics args
    parser.add_argument("--data_root", type=str, help="Dataset root containing GT normals (source of copies).", required=False)
    parser.add_argument("--exp_root", type=str, help="Experiment root containing predicted normals and masks.", required=False)
    parser.add_argument("--data_normal_folder", type=str, default="normal", help="Folder under data_root that holds GT normals to copy from.")
    parser.add_argument("--exp_normal_folder", type=str, default="normal", help="Folder under exp_root that holds predicted normals.")
    parser.add_argument("--gt_normal_folder", type=str, default="gt_normal", help="Target folder under exp_root to place/carry GT normals.")
    parser.add_argument("--mask_folder", type=str, default="mask", help="Mask folder (default relative to exp_root). If absolute, used as-is.")
    parser.add_argument("--data_mask_folder", type=str, default="masks", help="Folder under data_root that holds GT masks to copy from when exp masks are missing.")
    parser.add_argument("--exts", type=str, nargs="+", default=["png", "jpg", "jpeg"], help="Normal image filename extensions to consider.")

    # No explicit RGB folder inputs to avoid interference; we infer from exp_root
    parser.add_argument("--metrics_csv", type=str, default=None, help="CSV file to append metrics into (no overwrite, header added if new file).")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for CSV output.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name for CSV output.")
    parser.add_argument("--skip_normal", action="store_true", help="Skip normal metrics and GT normal copying.")

    args = parser.parse_args()

    ran_any = False

    # Evaluate train/test splits if present, else evaluate total
    normal_results = {}
    rgb_results = {}
    counts = {}
    # import pdb; pdb.set_trace()
    if args.exp_root:
        exp_root_norm = os.path.normpath(args.exp_root)
        exp_name = os.path.basename(exp_root_norm)
        splits = []
        for sp in ["test"]:#,"train"
            if os.path.isdir(os.path.join(args.exp_root, sp)):
                splits.append(sp)
        # import pdb;pdb.set_trace()
        if splits:
            ran_any = True
            
            for sp in splits:
                exp_split_root = os.path.join(args.exp_root, sp)
                # Source data_root for this split: prefer data_root/sp if exists
                data_split_root = args.data_root
                if args.data_root:
                    cand = os.path.join(args.data_root, sp)
                    data_split_root = cand if os.path.isdir(cand) else args.data_root

                # Ensure masks present for this split before evaluation
                ensure_masks(
                    exp_root=exp_split_root,
                    data_root=data_split_root,
                    exp_mask_folder=args.mask_folder,
                    data_mask_folder=args.data_mask_folder,
                    ref_folders=["pred", args.exp_normal_folder, "gt"],
                    exts=args.exts,
                )
                
                # Normal metrics per split (copy only matching filenames)
                n_cos, n_mae, n_acc, n_cnt = (None, None, None, 0)
                if (not args.skip_normal) and (data_split_root is not None):
                    n_cos, n_mae, n_acc, n_cnt = evaluate_normal_metrics(
                        exp_root=exp_split_root,
                        data_root=data_split_root,
                        data_normal_folder=args.data_normal_folder,
                        exp_normal_folder=args.exp_normal_folder,
                        gt_normal_folder=args.gt_normal_folder,
                        mask_folder=args.mask_folder,
                        exts=args.exts,
                    )
                normal_results[sp] = (n_cos, n_mae, n_acc)
                counts[("normal", sp)] = n_cnt

                # RGB metrics per split
                pred_dir = os.path.join(exp_split_root, 'pred')
                gt_dir = os.path.join(exp_split_root, 'gt')
                r_psnr, r_ssim, r_lpips, r_cnt = (None, None, None, 0)
                if os.path.isdir(pred_dir) and os.path.isdir(gt_dir):
                    rgb_mask_dir = args.mask_folder
                    if rgb_mask_dir and not os.path.isabs(rgb_mask_dir):
                        rgb_mask_dir = os.path.join(exp_split_root, rgb_mask_dir)
                    r_psnr, r_ssim, r_lpips, r_cnt = evaluate_metrics(pred_dir, gt_dir, rgb_mask_dir)
                # import pdb;pdb.set_trace()
                rgb_results[sp] = (r_psnr, r_ssim, r_lpips)
                counts[("rgb", sp)] = r_cnt

                # CSV row per split
                if args.metrics_csv:
                    _write_metrics_csv(
                        csv_path=args.metrics_csv,
                        model_name=args.model_name,
                        dataset_name=args.dataset_name,
                        exp_name=exp_name,
                        type_str=sp,
                        normal_cos=n_cos,
                        normal_mae=n_mae,
                        normal_acc_11_25=n_acc,
                        rgb_psnr=r_psnr,
                        rgb_ssim=r_ssim,
                        rgb_lpips=r_lpips,
                    )

            # Weighted total across splits
            def _weighted_avg(vals_counts):
                num = 0.0
                den = 0
                for v, c in vals_counts:
                    if v is None or c is None or c == 0:
                        continue
                    num += v * c
                    den += c
                return (num / den) if den > 0 else None, den

            total_n_cos, n_den = _weighted_avg([(normal_results[sp][0], counts[("normal", sp)]) for sp in splits])
            total_n_mae, _ = _weighted_avg([(normal_results[sp][1], counts[("normal", sp)]) for sp in splits])
            total_n_acc, _ = _weighted_avg([(normal_results[sp][2], counts[("normal", sp)]) for sp in splits])
            total_r_psnr, r_den = _weighted_avg([(rgb_results[sp][0], counts[("rgb", sp)]) for sp in splits])
            total_r_ssim, _ = _weighted_avg([(rgb_results[sp][1], counts[("rgb", sp)]) for sp in splits])
            total_r_lpips, _ = _weighted_avg([(rgb_results[sp][2], counts[("rgb", sp)]) for sp in splits])

            if total_n_cos is not None or total_r_psnr is not None:
                print("==== TOTAL (train+test) ====")
                if total_n_cos is not None:
                    print(f"Normal Cosine Distance (mean): {total_n_cos:.6f} | MAE (deg, mean): {total_n_mae:.4f} | Acc@11.25°: {total_n_acc if total_n_acc is not None else float('nan'):.4f} | images: {n_den}")
                if total_r_psnr is not None:
                    print(f"PSNR: {total_r_psnr:.4f}, SSIM: {total_r_ssim:.4f}, LPIPS: {total_r_lpips:.4f} | images: {r_den}")

            # if args.metrics_csv:
            #     _write_metrics_csv(
            #         csv_path=args.metrics_csv,
            #         model_name=args.model_name,
            #         dataset_name=args.dataset_name,
            #         exp_name=exp_name,
            #         type_str="total",
            #         normal_cos=total_n_cos,
            #         normal_mae=total_n_mae,
            #         normal_acc_11_25=total_n_acc,
            #         rgb_psnr=total_r_psnr,
            #         rgb_ssim=total_r_ssim,
            #         rgb_lpips=total_r_lpips,
            #     )
        else:
            # No splits: evaluate as total at root level
            # Ensure masks present at root before evaluation
            ensure_masks(
                exp_root=args.exp_root,
                data_root=args.data_root,
                exp_mask_folder=args.mask_folder,
                data_mask_folder=args.data_mask_folder,
                ref_folders=["pred", args.exp_normal_folder, "gt"],
                exts=args.exts,
            )
            n_cos = n_mae = n_acc = None
            if (not args.skip_normal) and args.data_root:
                n_cos, n_mae, n_acc, _ = evaluate_normal_metrics(
                    exp_root=args.exp_root,
                    data_root=args.data_root,
                    data_normal_folder=args.data_normal_folder,
                    exp_normal_folder=args.exp_normal_folder,
                    gt_normal_folder=args.gt_normal_folder,
                    mask_folder=args.mask_folder,
                    exts=args.exts,
                )
            pred_dir = os.path.join(args.exp_root, 'pred')
            gt_dir = os.path.join(args.exp_root, 'gt')
            r_psnr = r_ssim = r_lpips = None
            if os.path.isdir(pred_dir) and os.path.isdir(gt_dir):
                rgb_mask_dir = args.mask_folder
                if rgb_mask_dir and not os.path.isabs(rgb_mask_dir):
                    rgb_mask_dir = os.path.join(args.exp_root, rgb_mask_dir)
                r_psnr, r_ssim, r_lpips, _ = evaluate_metrics(pred_dir, gt_dir, rgb_mask_dir)
            
            ran_any = True

            # if args.metrics_csv:
            #     _write_metrics_csv(
            #         csv_path=args.metrics_csv,
            #         model_name=args.model_name,
            #         dataset_name=args.dataset_name,
            #         exp_name=exp_name,
            #         type_str="total",
            #         normal_cos=n_cos,
            #         normal_mae=n_mae,
            #         normal_acc_11_25=n_acc,
            #         rgb_psnr=r_psnr,
            #         rgb_ssim=r_ssim,
            #         rgb_lpips=r_lpips,
            #     )

    if not ran_any:
        parser.print_help()
