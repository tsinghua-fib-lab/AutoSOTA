import numpy as np
import itertools
from skimage.color import rgb2hed
from collections import defaultdict
from .uf import UnionFind

def compute_iou(masks1, masks2):
    masks1_flat = masks1.reshape(masks1.shape[0], -1)
    masks2_flat = masks2.reshape(masks2.shape[0], -1)

    intersection = masks1_flat.astype(np.float32) @ masks2_flat.astype(np.float32).T

    sum1 = masks1_flat.sum(axis=1)[:, np.newaxis]
    sum2 = masks2_flat.sum(axis=1)[np.newaxis, :]
    union = sum1 + sum2 - intersection
    union = np.clip(union, a_min=1e-8, a_max=None)
    
    return intersection / union

def compute_iou_pairwise(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def merge_overlaps(masks, scores, iou_thresh=0.8, grid_size=128):
    """
    merge overlapping masks
    """
    n_masks = len(masks)
    (H, W) = masks[0].shape
    if n_masks == 0:
        return masks, scores

    uf = UnionFind(n_masks)

    # Calculate bounding boxes
    bboxes = np.zeros((n_masks, 4), dtype=np.int32)
    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)
        if rows.size > 0:
            bboxes[i, 0], bboxes[i, 1] = rows.min(), rows.max()
            bboxes[i, 2], bboxes[i, 3] = cols.min(), cols.max()

    # assign masks to grid cells
    cells = defaultdict(list)
    for i in range(n_masks):
        y0, y1, x0, x1 = bboxes[i]
        for gy in range(int(y0 // grid_size), int(y1 // grid_size) + 1):
            for gx in range(int(x0 // grid_size), int(x1 // grid_size) + 1):
                cells[(gy, gx)].append(i)

    # test overlaps in each cell
    for mask_indices in cells.values():
        if len(mask_indices) < 2:
            continue

        for i, j in itertools.combinations(mask_indices, 2):
            y0_i, y1_i, x0_i, x1_i = bboxes[i]
            y0_j, y1_j, x0_j, x1_j = bboxes[j]
            # check overlap
            if (y1_i < y0_j or y1_j < y0_i or x1_i < x0_j or x1_j < x0_i):
                continue

            iou = compute_iou_pairwise(masks[i], masks[j])
            if iou > iou_thresh:
                uf.union(i, j)

    # collect groups
    roots = [uf.find(i) for i in range(n_masks)]
    unique_roots = list(set(roots))
    remap = {r: idx for idx, r in enumerate(unique_roots)}

    # use mask with the highest score as result
    best_idx = {}
    for i in range(n_masks):
        g = remap[uf.find(i)]
        if g not in best_idx or scores[i] > scores[best_idx[g]]:
            best_idx[g] = i

    merged_masks = []
    merged_scores = []
    for group_id in sorted(best_idx.keys()):
        idx = best_idx[group_id]
        merged_masks.append(masks[idx])
        merged_scores.append(scores[idx])

    return merged_masks, merged_scores


def bbox_iou_fast(b1, b2):
    if b1 is None or b2 is None:
        return False
    y1min, y1max, x1min, x1max = b1
    y2min, y2max, x2min, x2max = b2
    inter_w = max(0, min(x1max, x2max) - max(x1min, x2min))
    inter_h = max(0, min(y1max, y2max) - max(y1min, y2min))
    return (inter_w > 0) and (inter_h > 0)


def soft_nms(masks, scores, score_thresh=0.5, sigma=0.2,
             containment_thresh=0.95, alpha=0.5):
    """
    Applies Soft-NMS with Containment Penalty
    """
    num_masks = len(masks)
    if num_masks == 0:
        return [], []

    # Pre-compute bounding boxes for acceleration
    bboxes = []
    areas = []
    for m in masks:
        ys, xs = np.where(m)
        if ys.size == 0:
            bboxes.append(None) # Mark empty masks
            areas.append(0)
        else:
            bboxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
            areas.append(ys.size)

    # Apply containment penalty to scores before NMS starts
    penalized_scores = list(scores)
    if num_masks > 1:
        for i in range(num_masks):
            if bboxes[i] is None:
                continue

            n_contained = 0
            for j in range(num_masks):
                if i == j or bboxes[j] is None or areas[i] < areas[j]:
                    continue
                if not bbox_iou_fast(bboxes[i], bboxes[j]):
                    continue
                
                intersection = np.logical_and(masks[i], masks[j]).sum()
                if areas[j] > 0:
                    containment_ratio = intersection / areas[j]
                    if containment_ratio > containment_thresh and scores[j] > 0.5:
                        n_contained += 1

            if n_contained > 1:
                penalty_factor = np.tanh(alpha * n_contained)
                penalized_scores[i] *= (1.0 - penalty_factor)
  
    idxs = list(range(len(masks)))
    final_masks = []
    final_scores = []

    while idxs:
        # Select the mask with highest current score
        current = max(idxs, key=lambda i: penalized_scores[i])
        final_masks.append(masks[current])
        final_scores.append(penalized_scores[current])
        idxs.remove(current)
        
        # Decay scores of overlapping masks
        for i in idxs[:]:                       
            if not bbox_iou_fast(bboxes[current], bboxes[i]):
                continue
            iou = compute_iou_pairwise(masks[current], masks[i])   
            penalized_scores[i] *= np.exp(-(iou*iou)/sigma)
            if penalized_scores[i] < score_thresh:
                idxs.remove(i) 
        
    return final_masks, final_scores


def calculate_h_score(masks, h_channel):
    """
    Score each mask by its mean H-channel intensity.
    """
    h_scores = []
    for m in masks:
        m_bool = (m>0.5)
        if np.any(m_bool):
            h_scores.append(h_channel[m_bool].mean())
        else:
            h_scores.append(0.0)
    return h_scores


def post_process_nms(
        masks, 
        scores, 
        original_image, 
        merge_iou_thresh=0.8, 
        nms_score_thresh=0.5, 
        h_weight=0.3,
        containment_alpha=0.5
):
    #  Merge highly overlapping masks
    merged_masks, merged_scores = merge_overlaps(masks, scores, iou_thresh=merge_iou_thresh)

    if len(merged_masks) == 0:
        return [], []
    
    hed = rgb2hed(original_image)
    h_channel = hed[:, :, 0]

    p1, p99 = np.percentile(h_channel, (1, 99))
    if p99 > p1:
        h_channel = np.clip(h_channel, p1, p99)
        h_channel = (h_channel - p1) / (p99 - p1)
    else:
        h_channel = np.zeros_like(h_channel)

    h_scores = calculate_h_score(merged_masks, h_channel)

    combined_scores = []
    for base, h_score in zip(merged_scores, h_scores):
        combined = (
            (1 - h_weight) * base +
            h_weight * h_score
        )
        combined_scores.append(combined)

    soft_masks, soft_scores = soft_nms(
        merged_masks, 
        combined_scores, 
        score_thresh=nms_score_thresh,
        alpha=containment_alpha
    )

    return soft_masks, soft_scores





