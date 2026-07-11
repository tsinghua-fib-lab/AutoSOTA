import os
import cv2
import timm
import torch
import argparse
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from skimage import filters, morphology
from scipy.ndimage import binary_fill_holes
from timm import create_model
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import extract_feat, cal_sim
from utils import is_mask_valid
from utils import generate_ref_mask, remove_lines, merge_points
from utils import sample_foreground_points, sample_background_points
from utils import OptimalTransport
from utils import DenseCRF

 
def OT_mask(img, S, ot_fn, crf_fn, K, Hf, Wf):
    T_full = ot_fn.solve(S)  # [M, 2K+1]
    T_proto = T_full[..., :-1]
    slack   = T_full[:, -1]  

    M = Hf*Wf
    S = S.view(M, 2, K)       # [M, 2, K]
    T = T_proto.view(M, 2, K)       # [M, 2, K]
    
    heatmaps = []
    for c in range(2):
        # weighted sum over that class's K clusters
        weighted = T[:,c,:] * S[:,c,:]     # [M, K]
        hm_flat = weighted.sum(dim=1)      # [M]
        if c == 0:
            hm_flat = hm_flat + slack
        hm = hm_flat.view(Hf, Hf)          # [Hf, Wf]
        heatmaps.append(hm)
    
    refined = torch.stack(heatmaps, dim=0)  # [2, Hf, Wf]
    height, width = img.size
    up = F.interpolate(refined.unsqueeze(0), (height, width), mode='bilinear', align_corners=False)
    up = up.squeeze(0).cpu().numpy().copy()
    img = np.array(img).copy()
    crf_out = crf_fn(img, up)

    mask = crf_out[1] > filters.threshold_otsu(crf_out[1])
    mask = remove_lines(mask)
    mask = morphology.remove_small_holes(mask, area_threshold=40)
    mask = morphology.remove_small_objects(mask, min_size=30)

    return mask


def get_points(fname, args):
    img_id = os.path.splitext(fname)[0]
    wkdir = os.path.join(args.output_dir, img_id)
    os.makedirs(wkdir, exist_ok=True) 

    img_path = os.path.join(args.input_dir, fname)
    img_rgb = Image.open(img_path).convert('RGB')
    
    # ===== generate high confident reference masks =====
    mask_fg, mask_bg, high_conf = generate_ref_mask(img_rgb,keep_ratio=args.mask_ratio)

    if high_conf:
        # ===== extract features and perform OT =====
        feature_map = extract_feat(
            img_rgb, model, transform, args.tile_size, args.stride, device
        )
        _, Hf, Wf = feature_map.shape
        
        S = cal_sim(feature_map, mask_fg, mask_bg, K=args.k_num)
        crf_fn = DenseCRF()
        best_rho = args.rho
        ot_fn = OptimalTransport(rho=best_rho)
        best_mask = OT_mask(img_rgb, S, ot_fn, crf_fn, K=args.k_num, Hf=Hf, Wf=Wf) 

        # =============== grow mask =================
        rho_start = best_rho + args.rho_step
        ref_area = mask_fg.sum()
        for rho in np.arange(rho_start, 1.0001, args.rho_step):
            ot_fn = OptimalTransport(rho=float(rho))
            mask = OT_mask(img_rgb, S, ot_fn, crf_fn, K=args.k_num, Hf=Hf, Wf=Wf)
            coverage = (mask & mask_fg).sum() / ref_area
            if coverage < 0.95:
                best_mask = mask
                best_rho = rho
                continue
            if is_mask_valid(mask, best_mask):
                best_mask = mask
                best_rho = rho
            else:
                break
            
        full_rho = min(1.0, best_rho+2*args.rho_step)
        ot_full = OptimalTransport(rho=full_rho)
        mask_full = OT_mask(img_rgb, S, ot_full, crf_fn, K=args.k_num, Hf=Hf, Wf=Wf)
        mask_neg = ~binary_fill_holes(mask_full)

    else:
        best_mask = mask_fg
        mask_full = mask_fg
        mask_neg  = mask_bg

    if args.save_ref_mask:
        mask_fg_path = os.path.join(wkdir, "mask_fg.png")
        cv2.imwrite(mask_fg_path, (mask_fg*255).astype(np.uint8))
        mask_bg_path = os.path.join(wkdir, "mask_bg.png")
        cv2.imwrite(mask_bg_path, (mask_bg*255).astype(np.uint8))
        mask_refine_path = os.path.join(wkdir, 'mask_refine.png')
        cv2.imwrite(mask_refine_path, (best_mask*255).astype(np.uint8))

    # ===== sample points =====
    init_pts = sample_foreground_points(mask_fg)
    heatmap_pts = sample_foreground_points(best_mask)
    pos = np.vstack([init_pts, heatmap_pts])
    pos_merged = merge_points(pos, threshold=3)
    neg = sample_background_points(mask_neg)

    if args.save_points:
        os.makedirs(wkdir, exist_ok=True)
        pd.DataFrame(pos_merged, columns=["y","x"])\
            .to_csv(os.path.join(wkdir, "pos_points.csv"), index=False)
        pd.DataFrame(neg, columns=["y","x"])\
            .to_csv(os.path.join(wkdir, "neg_points.csv"), index=False)

# ========================= load feature extraction model ========================
# timm
def _timm_from_name(model_name: str, device: torch.device, timm_kwargs=None):
    if not timm_kwargs:
        timm_kwargs = {}

    model = create_model(
        model_name, pretrained=True, **timm_kwargs
    ).to(device).eval()

    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)
    return model, transform

def _timm_from_local(checkpoint_path: str, timm_model_name: str, device: torch.device, timm_kwargs=None):
    """Load a timm model from a local safetensors checkpoint."""
    from safetensors.torch import load_file
    if not timm_kwargs:
        timm_kwargs = {}

    model = create_model(
        timm_model_name, pretrained=False, **timm_kwargs
    ).to(device).eval()

    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)

    # Build transform from model config
    cfg = resolve_data_config(model.default_cfg, model=model)
    transform = create_transform(**cfg)
    return model, transform

def _warmup(model_name: str, timm_kwargs=None):
    try:
        if not timm_kwargs:
            timm_kwargs = {}
        m = create_model(
        model_name, pretrained=True, **timm_kwargs
        )
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] Warmup download failed for '{model_name}': {e}")

def _warmup_local(timm_model_name: str, checkpoint_path: str, timm_kwargs=None):
    """Warmup for local checkpoint: just create model arch to trigger any downloads."""
    try:
        if not timm_kwargs:
            timm_kwargs = {}
        m = create_model(timm_model_name, pretrained=False, **timm_kwargs)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] Warmup failed for local model '{timm_model_name}': {e}")

# load model
device = None
model  = None
transform = None

# timm
def init_worker(model_name, timm_kwargs=None, local_checkpoint=None):
    global device, model, transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_checkpoint:
        model, transform = _timm_from_local(local_checkpoint, model_name, device, timm_kwargs)
    else:
        model, transform = _timm_from_name(model_name, device, timm_kwargs)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)


def _map(fname_and_args):
    fname, args = fname_and_args
    get_points(fname, args)
    return fname

if __name__ == "__main__":
    # login()   
    parser = argparse.ArgumentParser(description="Patch-based feature extraction")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="result")
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--k_num", type=int, default=3)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--rho_step", type=float, default=0.05)
    parser.add_argument("--model_name", default="vit_large_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="Path to a local .safetensors checkpoint (bypasses HF download)")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--cpu_num", type=int, default=4)
    parser.add_argument("--save_points", action="store_true")
    parser.add_argument("--save_ref_mask", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set global seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Timm kwargs (empty dict for standard timm models like DINOv2;
    # only needed for custom architectures like UNI2-h)
    timm_kwargs = {}

    mp.set_start_method("spawn", force=True)
    try:
        ifile_path = args.index_file

        with open(ifile_path) as f:
            indices = [ln.strip() for ln in f if ln.strip()]

    except FileNotFoundError:
        print(f"[ERROR] Index file not found at: {ifile_path}")
        exit(1)
    except IndexError:
        print(f"[ERROR] The --input_dir '{args.input_dir}' is not deep enough. It must be at least two levels inside the data root.")
        exit(1)

    if args.local_checkpoint:
        _warmup_local(args.model_name, args.local_checkpoint, timm_kwargs)
    else:
        _warmup(args.model_name, timm_kwargs)

    jobs = [(fname, args) for fname in indices]
    initargs = (args.model_name, timm_kwargs, args.local_checkpoint)
    with mp.Pool(
        processes=args.cpu_num,
        initializer=init_worker,
        initargs=initargs
    ) as pool:
        for _ in tqdm(pool.imap_unordered(_map, jobs), total=len(jobs)):
            pass