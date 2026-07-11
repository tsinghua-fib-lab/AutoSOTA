import os
import cv2
import torch
import argparse
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils import save_json, post_process_nms
from utils import is_candidate_mask, keep_and_smooth
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def split_image(img, tile_size: int, stride: int):
    h, w = img.shape[:2]
    patches, coords = [], []
    for top in range(0, h - tile_size + 1, stride):
        for left in range(0, w - tile_size + 1, stride):
            patches.append(img[top:top+tile_size, left:left+tile_size])
            coords.append((top, left))
    return patches, coords


def generate_instance(img_rgb, pos_df, neg_df, predictor, args):
    """
    generate instance masks from patches
    """
    final_mask = []
    final_scores = []
    patches, coords = split_image(img_rgb, args.patch_size, args.stride)
    H, W, _ = img_rgb.shape

    pos_init = []
    if pos_df is not None:
        for _, row in pos_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < W and 0 <= y < H:
                pos_init.append((y, x))

    neg_init = []
    if neg_df is not None:
        for _, row in neg_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < W and 0 <= y < H:
                neg_init.append((y, x))

    border_offset = 5
    min_pixel = 50

    for patch_img, (top, left) in zip(patches, coords):
        pos = [
            (y-top, x-left)
            for (y,x) in pos_init
            if top <= y < top+args.patch_size and left <= x < left+args.patch_size
        ]
        pts_pos = np.array(pos, dtype=int)

        neg = [
            (y-top, x-left)
            for (y,x) in neg_init
            if top <= y < top+args.patch_size and left <= x < left+args.patch_size
        ]
        pts_neg = np.array(neg, dtype=int)

        neg_tree = cKDTree(pts_neg) if pts_neg.size else None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(patch_img)

        patch_masks = []
        patch_scores = []
        
        for (y0, x0) in pts_pos:
            # build prompt lists
            coords_list = [(x0, y0)]
            labels_list = [1]
            if neg_tree:
                dists, inds = neg_tree.query((y0, x0),
                                             k=min(args.k_neg, len(pts_neg)))
                for idx in np.atleast_1d(inds):
                    yn, xn = pts_neg[idx]
                    coords_list.append((xn, yn))
                    labels_list.append(0)

            mask, score, _ = predictor.predict(
                point_coords = np.array(coords_list),
                point_labels = np.array(labels_list),
                multimask_output=False
            )
            m = keep_and_smooth(mask[0])

            if not is_candidate_mask(
                m, left, top, H, W,
                min_pixel = min_pixel,
                neg_pts=pts_neg,
                border_offset= border_offset,
            ):
                continue

            patch_masks.append(m)
            patch_scores.append(score[0])

        if len(patch_masks) == 0:
            continue
        
        soft_masks_patch, soft_scores_patch = post_process_nms(
            patch_masks,
            patch_scores,
            patch_img,
            h_weight=args.h_weight,
        )

        h_p, w_p = args.patch_size, args.patch_size
        for mask, score in zip(soft_masks_patch, soft_scores_patch):
            full = np.zeros((H, W), dtype=bool)
            full[top:top+h_p, left:left+w_p] = mask
            final_mask.append(full)
            final_scores.append(score)

    return final_mask, final_scores

def process_image(fname, args, predictor):
    img_name, _ = os.path.splitext(fname)
    img_path = os.path.join(args.input_dir, fname)
    output_dir = os.path.join(args.output_dir, img_name)
    os.makedirs(output_dir, exist_ok=True)

    # load & prep
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pos_csv = os.path.join(output_dir, 'pos_points.csv')
    pos_df = pd.read_csv(pos_csv)
    neg_csv = os.path.join(output_dir, 'neg_points.csv')
    neg_df = pd.read_csv(neg_csv)
    initial_masks, initial_scores = generate_instance(
        img_rgb, pos_df, neg_df, predictor, args
    )

    soft_masks, soft_scores = post_process_nms(
        initial_masks,
        initial_scores,
        img_rgb,
        h_weight=args.h_weight,
    )

    save_json(soft_masks,
                os.path.join(output_dir, "soft_masks.json"))


_predictor = None  

def init_worker(model_cfg, checkpoint):
    global _predictor
    sam_model = build_sam2(model_cfg, checkpoint)
    _predictor = SAM2ImagePredictor(sam_model)

def _map(job_and_args) -> None:
    fname, args = job_and_args
    global _predictor
    process_image(fname, args, _predictor)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Use SAM to generate instance masks and save as RLE-encoded COCO style with post-processing.")
    p.add_argument("--input_dir", required=True,
                   help="Path to the original RGB image directory.")
    p.add_argument("--output_dir", required=True,
                   help="Path to the directory of per-image semantic mask folders.")
    p.add_argument("--index_file", type=str, required=True,
                   help="Name of the index file.")
    p.add_argument("--patch_size", type=int, default=512,
                   help="Size of image/mask patches.")
    p.add_argument("--stride", type=int, default=256,
                   help="Stride between patches.")
    p.add_argument("--k_neg", type=int, default=2,
                   help="Number of negative points per positive.")
    p.add_argument("--cpu_num", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--nms_sigma", type=float, default=0.5, help="Soft-NMS Gaussian sigma")
    p.add_argument("--containment_thresh", type=float, default=0.9, help="Containment threshold for mask suppression")
    p.add_argument("--h_weight", type=float, default=0.3, help="Weight for H-channel in combined score")

    args = p.parse_args()

    # Set global seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        ifile_path = args.index_file

        with open(ifile_path, 'r') as f:
            fnames_to_process = f.read().splitlines()
    
    except FileNotFoundError:
        print(f"[ERROR] Index file not found at: {ifile_path}")
        exit(1)
    except IndexError:
        print(f"[ERROR] The --input_dir '{args.input_dir}' is not deep enough. It must be at least two levels inside the data root.")
        exit(1)

    n = len(fnames_to_process)
    jobs = [(fname, args) for fname in fnames_to_process]

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "/repo/sam2/checkpoints/sam2.1_hiera_large.pt"

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.cpu_num, initializer=init_worker, initargs=(model_cfg, checkpoint)) as pool:
        list(tqdm(pool.imap(_map, jobs), total=n))
    