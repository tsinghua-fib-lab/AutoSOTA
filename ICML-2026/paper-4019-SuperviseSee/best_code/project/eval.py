import os
import cv2
import argparse
import scipy.sparse
import numpy as np
from PIL import Image


class Evaluator:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.clear()
    
    def clear(self):
        self.meter_dict = {'AJI': [], 'PQ': [], 'DQ': [], 'SQ': [], 'Dice': []}
    
    def add(self, pred_mask, gt_mask):
        # Calculate  Dice score
        dice = self.get_dice(pred_mask > 0, gt_mask > 0)
        self.meter_dict['Dice'].append(dice)
        
        # Compute the pairwise IoU matrix efficiently
        pairwise_iou, true_ids, pred_ids = self._compute_pairwise_iou(pred_mask, gt_mask)

        aji = self._get_aji_from_iou(pairwise_iou, true_ids, pred_ids, gt_mask, pred_mask)
        self.meter_dict['AJI'].append(aji)

        dq, sq, pq = self._get_pq_from_iou(pairwise_iou, true_ids, pred_ids)
        self.meter_dict['PQ'].append(pq)
        self.meter_dict['DQ'].append(dq)
        self.meter_dict['SQ'].append(sq)
        
        return float(aji), float(pq), float(dq), float(sq), float(dice)
            
    def get(self):
        AJI = np.mean(self.meter_dict['AJI'])
        PQ = np.mean(self.meter_dict['PQ'])
        DQ = np.mean(self.meter_dict['DQ'])
        SQ = np.mean(self.meter_dict['SQ'])
        Dice = np.mean(self.meter_dict['Dice'])

        return float(AJI), float(PQ), float(DQ), float(SQ), float(Dice)

    
    def get_dice(self, pred, gt, eps=1e-6):
        inter = np.sum(pred & gt)
        denom = pred.sum() + gt.sum()
        return (2.0 * inter + eps) / (denom + eps)

    def _compute_pairwise_iou(self, pred_mask, gt_mask):
        """
        Computes the pairwise IoU between all instance masks of gt and pred.
        vectorized and avoids slow loops
        """
        true_ids, true_areas = np.unique(gt_mask, return_counts=True)
        pred_ids, pred_areas = np.unique(pred_mask, return_counts=True)

        # Remove background IDs (0)
        true_ids = true_ids[true_ids != 0]
        true_areas = true_areas[1:] if 0 in np.unique(gt_mask) else true_areas
        pred_ids = pred_ids[pred_ids != 0]
        pred_areas = pred_areas[1:] if 0 in np.unique(pred_mask) else pred_areas

        if len(true_ids) == 0 or len(pred_ids) == 0:
            return np.zeros((0, 0)), true_ids, pred_ids

        # Map IDs to continuous indices (0, 1, 2...)
        true_id_map = {tid: i for i, tid in enumerate(true_ids)}
        pred_id_map = {pid: i for i, pid in enumerate(pred_ids)}

        # Flatten masks and filter out background pixels for efficiency
        gt_flat = gt_mask.flatten()
        pred_flat = pred_mask.flatten()
        
        non_bg_mask = (gt_flat != 0) & (pred_flat != 0)
        gt_flat = gt_flat[non_bg_mask]
        pred_flat = pred_flat[non_bg_mask]

        if len(gt_flat) == 0:
             return np.zeros((len(true_ids), len(pred_ids))), true_ids, pred_ids

        # creates a matrix where entry (i, j) is the number of pixels
        # where gt_mask is true_id[i] and pred_mask is pred_id[j].
        row_ind = np.array([true_id_map[i] for i in gt_flat])
        col_ind = np.array([pred_id_map[i] for i in pred_flat])
        
        intersection_matrix = scipy.sparse.coo_matrix(
            (np.ones(len(row_ind)), (row_ind, col_ind)),
            shape=(len(true_ids), len(pred_ids))
        ).toarray()

        # Compute union matrix using broadcasting
        # union(i, j) = area(true_i) + area(pred_j) - intersection(i, j)
        union_matrix = true_areas[:, None] + pred_areas[None, :] - intersection_matrix
        
        # Compute IoU, handling division by zero
        iou_matrix = np.divide(
            intersection_matrix, union_matrix, 
            out=np.zeros_like(intersection_matrix, dtype=float), 
            where=union_matrix != 0
        )
        
        return iou_matrix, true_ids, pred_ids
    
    def _get_aji_from_iou(self, pairwise_iou, true_ids, pred_ids, gt_mask, pred_mask):
        # Edge cases
        if pairwise_iou.size == 0:
            # both empty -> 1.0 ; only one side non-empty -> 0.0
            if len(true_ids) == 0 and len(pred_ids) == 0: 
                return 1.0
            return 0.0

        # Areas per instance id (aligned to true_ids / pred_ids order)
        gt_areas = {tid: np.count_nonzero(gt_mask == tid) for tid in true_ids}
        pr_areas = {pid: np.count_nonzero(pred_mask == pid) for pid in pred_ids}

        # Build all candidate pairs with IoU > 0, sort by IoU desc
        rows, cols = np.where(pairwise_iou > 0)
        pairs = sorted(zip(rows, cols), key=lambda rc: pairwise_iou[rc[0], rc[1]], reverse=True)

        matched_true, matched_pred = set(), set()
        matched_pairs = []
        inter_sum = 0.0
        union_sum = 0.0

        for r, c in pairs:
            if r in matched_true or c in matched_pred:
                continue
            tid = true_ids[r]
            pid = pred_ids[c]
            iou = pairwise_iou[r, c]
            if iou <= 0:
                continue

            Ag = gt_areas[tid]
            Ap = pr_areas[pid]
            inter = iou * (Ag + Ap) / (1.0 + iou)
            uni = Ag + Ap - inter

            inter_sum += inter
            union_sum += uni
            matched_true.add(r)
            matched_pred.add(c)
            matched_pairs.append((tid, pid))

        # Unmatched areas (count entirely in denominator)
        unmatched_gt = set(range(len(true_ids))) - matched_true
        unmatched_pr = set(range(len(pred_ids))) - matched_pred
        for r in unmatched_gt:
            union_sum += gt_areas[true_ids[r]]
        for c in unmatched_pr:
            union_sum += pr_areas[pred_ids[c]]

        return float(inter_sum / (union_sum + 1e-6))


    def _get_pq_from_iou(self, pairwise_iou, true_ids, pred_ids, iou_th=0.5):
        """
        Calculates PQ using the pre-computed pairwise IoU matrix.
        """
        if pairwise_iou.shape[0] == 0 or pairwise_iou.shape[1] == 0:
             if pairwise_iou.shape[0] == 0 and pairwise_iou.shape[1] == 0:
                return 1.0, 1.0, 1.0 # DQ, SQ, PQ
             else:
                return 0.0, 0.0, 0.0
        
        # Match pairs with IoU > threshold
        matches = pairwise_iou > iou_th
        
        # Find unique matches ensuring one-to-one mapping
        # This is a simplification of the Hungarian algorithm for this case
        potential_matches = np.argwhere(matches)
        true_positives = 0
        paired_iou_sum = 0
        
        # Greedily find best matches
        # Sort potential matches by IoU value to prioritize better fits
        sorted_matches = sorted(potential_matches, key=lambda p: pairwise_iou[p[0], p[1]], reverse=True)
        
        matched_true = set()
        matched_pred = set()

        for t_idx, p_idx in sorted_matches:
            if t_idx not in matched_true and p_idx not in matched_pred:
                true_positives += 1
                paired_iou_sum += pairwise_iou[t_idx, p_idx]
                matched_true.add(t_idx)
                matched_pred.add(p_idx)
        
        fp = len(pred_ids) - true_positives
        fn = len(true_ids) - true_positives
        
        dq = true_positives / (true_positives + 0.5 * fp + 0.5 * fn + 1e-6)
        sq = paired_iou_sum / (true_positives + 1e-6) if true_positives > 0 else 0.0
        
        return dq, sq, dq * sq


def main(args):
    evaluator = Evaluator(['background', 'foreground'])

    gt_dir   = args.gt_dir 
    mask_dir = args.mask_dir

    filenames = sorted([
            fn for fn in os.listdir(gt_dir)
            if os.path.isfile(os.path.join(gt_dir, fn))
            and fn.lower().endswith(('.png', '.jpg', '.tif'))
        ])

    for fn in filenames:
        gt_path   = os.path.join(gt_dir,   fn)
        stem, _ = os.path.splitext(fn)
        # pred_path = os.path.join(mask_dir, stem, "soft_mask.png")
        pred_path = os.path.join(mask_dir, stem, "init_mask.png")

        if not os.path.exists(pred_path):
            print(f"{pred_path} not found! Skipping...")
            continue
        
        # Load mask
        gt_img = np.array(Image.open(gt_path))
        pr_img = np.array(Image.open(pred_path))

        gt_mask = (
            gt_img[:, :, 0].astype(np.uint32) * 256 +
            gt_img[:, :, 1].astype(np.uint32)
        )

        pred_mask = (
            pr_img[:, :, 0].astype(np.uint32) * 256 +
            pr_img[:, :, 1].astype(np.uint32)
        )

        aji, pq, dq, sq, dice = evaluator.add(pred_mask, gt_mask)

        print(f"{fn:30s}  AJI={aji:5.3f}  PQ={pq:5.3f}  "
              f"DQ={dq:5.3f}  SQ={sq:5.3f}  Dice={dice:5.3f}")

    (avg_aji, avg_pq, avg_dq, avg_sq, avg_dice) = evaluator.get()

    print("\nOverall average metrics:")
    print(f"AJI={avg_aji:5.3f}  PQ={avg_pq:5.3f}  "
          f"DQ={avg_dq:5.3f}  SQ={avg_sq:5.3f}  Dice={avg_dice:5.3f}"
          )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch-based similarity heatmap and fusion")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument('--mask_dir', required=True)

    args = parser.parse_args()
    main(args)