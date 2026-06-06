# mask-leval evaluation
import sys
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import math
import numpy as np
import cv2
from glob import glob
from PIL import Image
import os
from scipy.optimize import linear_sum_assignment
import pandas
import re

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def evaluate_unsupervised(all_gt_masks, all_res_masks, metric, all_void_masks=None):
    j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    for ii in range(all_gt_masks.shape[0]):
        for jj in range(all_res_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
    if 'J' in metric and 'F' in metric:
        all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
    else:
        all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
    row_ind, col_ind = linear_sum_assignment(-all_metrics)
    return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def load_mask(path):
    image = cv2.imread(path)

    contains_black = np.any(np.all(image == [0, 0, 0], axis=2))
    
    if contains_black:
        binary_mask = np.any(image > 0, axis=2)
    else:
        white_threshold = 250
        mask = np.all(image > white_threshold, axis=2)
        binary_mask = ~mask

    binary_mask = binary_mask.astype(np.uint8)
    
    return binary_mask

def read_masks_fbms(mask_dir, indices=None):
    # Get all potential mask file paths in the directory
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.jpg"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.jpeg"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.bmp")))

    # Convert to zero-based indexing for direct access
    if indices is not None:
        selected_paths = [mask_paths[i] for i in indices if 0 <= i < len(mask_paths)]
    else:
        selected_paths = mask_paths

    mask_list = []
    for path in selected_paths:
        mask_img = load_mask(path)
        mask_img = (mask_img > 0).astype(np.uint8)
        mask_list.append(mask_img)

    # Stack mask list along a new axis if there are masks, otherwise create an empty array
    dynamic_mask = np.stack(mask_list, axis=0) if mask_list else None

    return dynamic_mask

def read_masks(mask_dir, exp_masks=None):
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png"))) + sorted(glob(os.path.join(mask_dir, "*.jpg")) + sorted(glob(os.path.join(mask_dir, "*.jpeg")))) +sorted(glob(os.path.join(mask_dir, "*.bmp")))
    mask_list = []
        
    for path in mask_paths:
        mask_img, p = load_ann_png(path)
        mask_img = (mask_img > 0).astype(np.uint8)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        mask_list.append(mask_img)
    if not mask_list:
        dynamic_mask = np.zeros_like(exp_masks, dtype=np.uint8)
    else:
        dynamic_mask = np.stack(mask_list, axis=0)
    
    return dynamic_mask

def extract_frame_number(filename):
    # Try to match the number after an underscore, if present
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # If no underscore number, match the first number in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def get_matching_pred_indices(pred_dir, gt_dir):
    # Get and sort all pred and gt mask paths
    pred_paths = sorted(glob(os.path.join(pred_dir, "*.png"))) + sorted(glob(os.path.join(pred_dir, "*.jpg")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))

    # Extract frame numbers from pred mask filenames
    pred_indices = [extract_frame_number(os.path.basename(path)) for path in pred_paths]

    # Extract frame numbers from gt mask filenames and find matching indices in pred masks
    matching_pred_indices = []
    for gt_path in gt_paths:
        gt_index = extract_frame_number(os.path.basename(gt_path))
        if gt_index in pred_indices:
            matching_pred_indices.append(pred_indices.index(gt_index))  # Get index in pred list

    return matching_pred_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--res_dir', type=str,default="current_work_dir/exp_res/sam_res/ablation/no_tracks/initial_preds")
    parser.add_argument('--eval_seq_list', type=str,default="current_work_dir/baseline/DAVIS/ImageSets/2017/moving_val.txt")
    parser.add_argument('--eval_dir', type=str,default="current_work_dir/baseline/DAVIS/Annotations_unsupervised/480p")
    parser.add_argument('--img_dir', type=str,default="current-data-dir/baseline/davis/Testset")

    args = parser.parse_args()
    
    # eval_seq_path = "baseline/DAVIS/ImageSets/2016/moving_val.txt"
    eval_seq_path = args.eval_seq_list
    eval_dir = args.eval_dir
    # eval_dir = "baseline/DAVIS/Annotations_unsupervised/480p"
    if args.eval_seq_list is None:
        eval_seq_name = [name for name in os.listdir(args.res_dir) if os.path.isdir(os.path.join(args.res_dir, name))]
    else:
        eval_seq_path = args.eval_seq_list
        with open(eval_seq_path, 'r') as file:
            eval_seq_name = [line.strip() for line in file]
    
    metric=('J', 'F')

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for seq in tqdm(eval_seq_name):
        # seq = eval_seq_name[11]
        gt_dir = os.path.join(eval_dir, seq)
        res_dir = os.path.join(args.res_dir, seq)

        print(seq, eval_seq_name)
        
        if "FBMS" in args.eval_dir:
            gt_masks = read_masks_fbms(gt_dir)
            
            img_dir = os.path.join(args.img_dir,seq)
            gt_indices = get_matching_pred_indices(img_dir, gt_dir)
            
            pred_masks = read_masks_fbms(res_dir, indices=gt_indices)
        else:
            gt_masks = read_masks(gt_dir)
            pred_masks = read_masks(res_dir, gt_masks)

        # ==================== START: NEW MODIFICATION ====================
        # 检查GT掩码和预测掩码是否被成功加载
        # 一个更稳健的检查是检查数组的维度或大小
        if gt_masks is None or gt_masks.size == 0:
            print(f"\n!!! WARNING: No Ground Truth masks found for sequence: '{seq}'. Skipping. !!!")
            continue # 直接跳到下一个序列

        if pred_masks is None or pred_masks.size == 0:
            print(f"\n!!! WARNING: No prediction masks found for sequence: '{seq}'. Skipping. !!!")
            continue # 直接跳到下一个序列
        # ===================== END: NEW MODIFICATION =====================


        # if gt_masks.shape[0] != pred_masks.shape[0]:
        #     gt_masks = gt_masks[:-1]
        print(seq)
        min_shape = min(gt_masks.shape[0], pred_masks.shape[0])
        gt_masks = gt_masks[:min_shape]
        pred_masks = pred_masks[:min_shape]
        
        gt_masks = np.expand_dims(gt_masks, axis=0)
        pred_masks = np.expand_dims(pred_masks, axis=0)
        
        j_metrics_res, f_metrics_res = evaluate_unsupervised(gt_masks, pred_masks, metric=metric)
        
        for ii in range(gt_masks.shape[0]):
            seq_name = f'{seq}_{ii+1}'
            if 'J' in metric:
                [JM, JR, JD] = db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
                metrics_res['J']["M_per_object"][seq_name] = JM
            if 'F' in metric:
                [FM, FR, FD] = db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
                metrics_res['F']["M_per_object"][seq_name] = FM
                
    J, F = metrics_res['J'], metrics_res['F']

    seq_names = list(J['M_per_object'].keys())
    sys.stdout.write("----------------Global results in CSV---------------\n")
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    table_g = pandas.DataFrame(data=np.reshape(g_res, [1, len(g_res)]), columns=g_measures)
    table_g.to_csv(sys.stdout, index=False, float_format="%0.5f")
    save_path_g = os.path.join(args.res_dir, "table_g.csv")
    table_g.to_csv(save_path_g, index=False, float_format="%0.5f")

    sys.stdout.write("\n\n------------Per sequence results in CSV-------------\n")
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pandas.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    table_seq.to_csv(sys.stdout, index=False, float_format="%0.5f")
    save_path_s = os.path.join(args.res_dir, "table_seq.csv")
    table_seq.to_csv(save_path_s, index=False, float_format="%0.5f")
