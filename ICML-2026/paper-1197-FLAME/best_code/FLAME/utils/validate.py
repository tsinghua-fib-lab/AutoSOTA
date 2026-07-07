"""Validation utilities for evaluating FLAME models."""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import torch
import tqdm

from utils.metrics import compute_iou
from utils.tiling import overlap_tile_predict_logits_global_guided


logger = logging.getLogger(__name__)


def validate(
    model,
    loader,
    device,
    save_vis_path=None,
    num_vis_batches=5,
    save_all_vis: bool = False,
    contrastive_blur=False,
    model_outputs_probs=False,
    detection_threshold: float = 0.5,
    *,
    use_tiling: bool = False,
    tile_size: int = 512,
    tile_valid_size: int = 384,
    tile_stride: int = 384,
):
    """
    Validate model with option to accept probabilities directly and split results by dataset
    
    Args:
        model_outputs_probs: If True, model outputs probabilities [0,1] instead of logits
        detection_threshold: Probability threshold for image-level detection classification
        save_all_vis: If True, save visualizations for all batches/samples and ignore num_vis_batches
    """
    model.eval()
    
    # Overall metrics
    ious, fs = [], []
    error_count = 0
    max_errors = 5
    
    # Pristine image tracking
    pristine_correct = 0  # Correctly classified pristine images (pred=0, gt=0)
    pristine_total = 0    # Total pristine images (gt=0)
    
    # Detection metrics tracking
    detection_correct = 0  # Correctly classified detection predictions
    detection_total = 0    # Total samples with detection predictions
    detection_tp = 0       # True positives (forgery detected as forgery)
    detection_tn = 0       # True negatives (authentic detected as authentic)
    detection_fp = 0       # False positives (authentic detected as forgery)
    detection_fn = 0       # False negatives (forgery detected as authentic)
    detection_scores = []  # Detection probabilities for AP
    detection_labels = []  # Detection ground-truth labels for AP
    
    # Dataset-specific metrics
    dataset_metrics = defaultdict(lambda: {
        'ious': [], 'f1s': [],
        'small_ious': [], 'medium_ious': [], 'large_ious': [],
        'small_f1s': [], 'medium_f1s': [], 'large_f1s': [],
        'pristine_correct': 0, 'pristine_total': 0,
        'detection_correct': 0, 'detection_total': 0,
        'detection_tp': 0, 'detection_tn': 0, 'detection_fp': 0, 'detection_fn': 0,
        'detection_scores': [], 'detection_labels': [],
    })
    
    # Pick multiple random batch indices for visualization
    total_batches = len(loader)
    if save_vis_path and save_all_vis:
        vis_batch_indices = set(range(total_batches))
    elif num_vis_batches > 0 and save_vis_path:
        vis_batch_indices = set(random.sample(range(total_batches), min(num_vis_batches, total_batches)))
    else:
        vis_batch_indices = set()
    device = device if isinstance(device, torch.device) else torch.device(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="Validation")):
            try:
                if not batch:
                    logger.warning("Skipping empty batch at index %s", batch_idx)
                    continue
                # New streams format
                streams = [stream.to(device, non_blocking=True) for stream in batch["streams"]]
                orig = batch["orig"]
                tgt = batch["mask"]

                if not use_tiling:
                    orig = orig.to(device)
                    tgt = tgt.to(device)
                    source = batch.get("source", orig).to(device)
                else:
                    # For tiling inference we expect raw (unnormalized) tensors; keep on CPU here.
                    source = batch.get("source", None)
                
                batch_size = len(orig) if isinstance(orig, list) else orig.shape[0]

                # Get dataset names and turn info for this batch
                dataset_names = batch.get("dataset_name", ["unknown"] * batch_size)
                if isinstance(dataset_names, str):
                    dataset_names = [dataset_names] * batch_size
                sample_types = batch.get("sample_type", None)
                sample_types_present = sample_types is not None
                if not sample_types_present:
                    sample_types = ["forgery"] * batch_size
                elif isinstance(sample_types, str):
                    sample_types = [sample_types] * batch_size
                is_authentic = batch.get("is_authentic", None)
                if isinstance(is_authentic, torch.Tensor):
                    is_authentic = is_authentic.detach().cpu().tolist()
                elif isinstance(is_authentic, (bool, np.bool_)):
                    is_authentic = [bool(is_authentic)] * batch_size
                elif is_authentic is not None:
                    is_authentic = list(is_authentic)
                
                # Get turn indices and previous masks for multi-region visualization
                turn_indices = batch.get("turn_idx", [0] * batch_size)
                prev_masks = batch.get("prev_masks", [None] * batch_size)
                
                # Forward pass
                if not use_tiling:
                    global_image = batch.get("global_image")
                    if isinstance(global_image, torch.Tensor):
                        global_image = global_image.to(device, non_blocking=True)
                    else:
                        global_image = None
                    norm_coords = batch.get("norm_coords")
                    if isinstance(norm_coords, torch.Tensor):
                        norm_coords = norm_coords.to(device, non_blocking=True)
                    else:
                        norm_coords = None
                    if model_outputs_probs:
                        if hasattr(model, "set_current_batch"):
                            model.set_current_batch(batch)

                        outputs = model(
                            orig,
                            streams,
                            output_extras=True,
                            global_image=global_image,
                            norm_coords=norm_coords,
                        )
                        if isinstance(outputs, tuple):
                            probs, extras = outputs
                        else:
                            probs = outputs
                            extras = {}
                        logits = None
                    else:
                        if device.type == "cuda":
                            torch.clear_autocast_cache()
                        with torch.amp.autocast(
                            device_type=device.type,
                            enabled=device.type == "cuda",
                            cache_enabled=False,
                        ):
                            logits, extras = model(
                                orig,
                                streams,
                                output_extras=True,
                                global_image=global_image,
                                norm_coords=norm_coords,
                            )
                        probs = torch.sigmoid(logits)

                    coarse_mask = extras.get("coarse_mask", None)
                    mask_diff = extras.get("mask_diff", None)
                    detection_logit = extras.get("detection_logit", None)

                    if probs.shape[-2:] != tgt.shape[-2:]:
                        probs = torch.nn.functional.interpolate(
                            probs,
                            size=tgt.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    # Calculate metrics for each sample in batch
                    batch_ious = []
                    batch_f1s = []
                    pred_bin = (probs > 0.5).float()

                    for i in range(pred_bin.shape[0]):
                        dataset_name = dataset_names[i]

                        # Compute IoU for this sample
                        if model_outputs_probs:
                            probs_clamped = torch.clamp(probs[i : i + 1], 1e-7, 1 - 1e-7)
                            logits_for_iou = torch.log(probs_clamped / (1 - probs_clamped))
                            sample_iou = compute_iou(logits_for_iou, tgt[i : i + 1]).item()
                        else:
                            sample_iou = compute_iou(logits[i : i + 1], tgt[i : i + 1]).item()

                        batch_ious.append(sample_iou)

                        # Compute F1 for this sample
                        pred_flat = pred_bin[i, 0].cpu().numpy().astype(np.uint8).flatten()
                        true_flat = tgt[i, 0].cpu().numpy().astype(np.uint8).flatten()

                        # Check for pristine (mask-empty) images for localization stats
                        is_pristine = (true_flat.sum() == 0)
                        pred_pristine = (pred_flat.sum() == 0)

                        if is_pristine:
                            pristine_total += 1
                            dataset_metrics[dataset_name]["pristine_total"] += 1

                            if pred_pristine:
                                pristine_correct += 1
                                dataset_metrics[dataset_name]["pristine_correct"] += 1

                        if is_authentic is not None and i < len(is_authentic):
                            is_authentic_sample = bool(is_authentic[i])
                        elif sample_types_present:
                            is_authentic_sample = sample_types[i] == "authentic"
                        else:
                            is_authentic_sample = (true_flat.sum() == 0)
                        detection_gt = not is_authentic_sample  # True if sample is forgery
                        if detection_logit is not None:
                            detection_prob = torch.sigmoid(detection_logit[i]).item()
                        else:
                            detection_prob = probs[i].max().item()
                        detection_pred = detection_prob > detection_threshold

                        detection_total += 1
                        dataset_metrics[dataset_name]["detection_total"] += 1
                        detection_scores.append(detection_prob)
                        detection_labels.append(int(detection_gt))
                        dataset_metrics[dataset_name]["detection_scores"].append(detection_prob)
                        dataset_metrics[dataset_name]["detection_labels"].append(int(detection_gt))

                        if detection_pred == detection_gt:
                            detection_correct += 1
                            dataset_metrics[dataset_name]["detection_correct"] += 1

                        # Update confusion matrix
                        if detection_gt and detection_pred:  # True positive
                            detection_tp += 1
                            dataset_metrics[dataset_name]["detection_tp"] += 1
                        elif not detection_gt and not detection_pred:  # True negative
                            detection_tn += 1
                            dataset_metrics[dataset_name]["detection_tn"] += 1
                        elif not detection_gt and detection_pred:  # False positive
                            detection_fp += 1
                            dataset_metrics[dataset_name]["detection_fp"] += 1
                        elif detection_gt and not detection_pred:  # False negative
                            detection_fn += 1
                            dataset_metrics[dataset_name]["detection_fn"] += 1

                        # Check if both are empty (should result in F1 = 1.0)
                        if true_flat.sum() == 0 and pred_flat.sum() == 0:
                            f = 1.0
                        elif true_flat.sum() == 0:  # Pristine image with false positives
                            f = 0.0
                        elif pred_flat.sum() == 0:  # Forgery image with no detection
                            f = 0.0
                        else:
                            precision, recall, f, _ = precision_recall_fscore_support(
                                true_flat, pred_flat, zero_division=1, average="binary"
                            )

                        batch_f1s.append(f)

                        # Calculate forgery ratio for stratification
                        forgery_ratio = tgt[i].sum().item() / tgt[i].numel()

                        if sample_types[i] != "authentic":
                            # Add to overall metrics
                            ious.append(sample_iou)
                            fs.append(f)

                            # Add to dataset-specific metrics
                            dataset_metrics[dataset_name]["ious"].append(sample_iou)
                            dataset_metrics[dataset_name]["f1s"].append(f)

                            # Stratify by forgery size for both overall and dataset-specific
                            if forgery_ratio < 0.05:  # Small forgery (<5%)
                                dataset_metrics[dataset_name]["small_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["small_f1s"].append(f)
                            elif forgery_ratio < 0.20:  # Medium forgery (5-20%)
                                dataset_metrics[dataset_name]["medium_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["medium_f1s"].append(f)
                            else:  # Large forgery (>20%)
                                dataset_metrics[dataset_name]["large_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["large_f1s"].append(f)

                    # Create visualizations for selected batches
                    if batch_idx in vis_batch_indices and save_vis_path:
                        heatmap_prompt = extras.get("heatmap_prompt", None)
                        forensic_features = extras.get("forensic_features", None)
                        display_samples = batch_size if save_all_vis else 1

                        create_batch_visualization(
                            source,
                            orig,
                            streams,
                            tgt,
                            probs,
                            pred_bin,
                            coarse_mask,
                            mask_diff,
                            batch_ious,
                            batch_f1s,
                            batch_idx,
                            save_vis_path,
                            dataset_names,
                            turn_indices,
                            prev_masks,
                            detection_logits=detection_logit,
                            heatmap_prompt=heatmap_prompt,
                            forensic_features=forensic_features,
                            display_samples=display_samples,
                        )
                else:
                    # Tiled inference (variable-sized)
                    if model_outputs_probs:
                        raise ValueError("use_tiling=True does not support model_outputs_probs=True")

                    # Normalize batch access into lists
                    if isinstance(orig, list):
                        orig_list = orig
                    else:
                        orig_list = [orig[i] for i in range(orig.shape[0])]

                    if isinstance(tgt, list):
                        tgt_list = tgt
                    else:
                        tgt_list = [tgt[i] for i in range(tgt.shape[0])]

                    if source is None:
                        source_list = None
                    elif isinstance(source, list):
                        source_list = source
                    else:
                        source_list = [source[i] for i in range(source.shape[0])]

                    batch_ious = []
                    batch_f1s = []

                    save_all_vis_enabled = bool(save_vis_path) and save_all_vis
                    vis_payload = None
                    for i, (orig_i, tgt_i) in enumerate(zip(orig_list, tgt_list)):
                        dataset_name = dataset_names[i]
                        source_i = source_list[i] if source_list is not None else orig_i

                        logits_i, detection_logit_i = overlap_tile_predict_logits_global_guided(
                            model,
                            orig_i,
                            device,
                            tile_size=tile_size,
                            valid_size=tile_valid_size,
                            stride=tile_stride,
                            return_detection_logit=True,
                        )

                        tgt_i = tgt_i.to(device, dtype=torch.float32)
                        if tgt_i.dim() == 3:
                            tgt_i_b = tgt_i.unsqueeze(0)  # [1, 1, H, W]
                        else:
                            tgt_i_b = tgt_i

                        probs_i = torch.sigmoid(logits_i)
                        pred_bin_i = (probs_i > 0.5).float()

                        sample_iou = compute_iou(logits_i, tgt_i_b).item()
                        batch_ious.append(sample_iou)

                        pred_flat = pred_bin_i[0, 0].detach().cpu().numpy().astype(np.uint8).flatten()
                        true_flat = tgt_i_b[0, 0].detach().cpu().numpy().astype(np.uint8).flatten()

                        # Check for pristine (mask-empty) images for localization stats
                        is_pristine = (true_flat.sum() == 0)
                        pred_pristine = (pred_flat.sum() == 0)

                        if is_pristine:
                            pristine_total += 1
                            dataset_metrics[dataset_name]["pristine_total"] += 1

                            if pred_pristine:
                                pristine_correct += 1
                                dataset_metrics[dataset_name]["pristine_correct"] += 1

                        if is_authentic is not None and i < len(is_authentic):
                            is_authentic_sample = bool(is_authentic[i])
                        elif sample_types_present:
                            is_authentic_sample = sample_types[i] == "authentic"
                        else:
                            is_authentic_sample = (true_flat.sum() == 0)
                        detection_gt = not is_authentic_sample  # True if sample is forgery
                        if detection_logit_i is not None:
                            detection_prob = torch.sigmoid(detection_logit_i.squeeze()).item()
                        else:
                            detection_prob = probs_i.max().item()
                        detection_pred = detection_prob > detection_threshold

                        detection_total += 1
                        dataset_metrics[dataset_name]["detection_total"] += 1
                        detection_scores.append(detection_prob)
                        detection_labels.append(int(detection_gt))
                        dataset_metrics[dataset_name]["detection_scores"].append(detection_prob)
                        dataset_metrics[dataset_name]["detection_labels"].append(int(detection_gt))

                        if detection_pred == detection_gt:
                            detection_correct += 1
                            dataset_metrics[dataset_name]["detection_correct"] += 1

                        # Update confusion matrix
                        if detection_gt and detection_pred:  # True positive
                            detection_tp += 1
                            dataset_metrics[dataset_name]["detection_tp"] += 1
                        elif not detection_gt and not detection_pred:  # True negative
                            detection_tn += 1
                            dataset_metrics[dataset_name]["detection_tn"] += 1
                        elif not detection_gt and detection_pred:  # False positive
                            detection_fp += 1
                            dataset_metrics[dataset_name]["detection_fp"] += 1
                        elif detection_gt and not detection_pred:  # False negative
                            detection_fn += 1
                            dataset_metrics[dataset_name]["detection_fn"] += 1

                        # Compute F1 for this sample
                        if true_flat.sum() == 0 and pred_flat.sum() == 0:
                            f = 1.0
                        elif true_flat.sum() == 0:  # Pristine image with false positives
                            f = 0.0
                        elif pred_flat.sum() == 0:  # Forgery image with no detection
                            f = 0.0
                        else:
                            precision, recall, f, _ = precision_recall_fscore_support(
                                true_flat, pred_flat, zero_division=1, average="binary"
                            )

                        batch_f1s.append(f)

                        # Calculate forgery ratio for stratification
                        forgery_ratio = tgt_i_b.sum().item() / tgt_i_b.numel()

                        if sample_types[i] != "authentic":
                            # Add to overall metrics
                            ious.append(sample_iou)
                            fs.append(f)

                            # Add to dataset-specific metrics
                            dataset_metrics[dataset_name]["ious"].append(sample_iou)
                            dataset_metrics[dataset_name]["f1s"].append(f)

                            # Stratify by forgery size for both overall and dataset-specific
                            if forgery_ratio < 0.05:  # Small forgery (<5%)
                                dataset_metrics[dataset_name]["small_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["small_f1s"].append(f)
                            elif forgery_ratio < 0.20:  # Medium forgery (5-20%)
                                dataset_metrics[dataset_name]["medium_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["medium_f1s"].append(f)
                            else:  # Large forgery (>20%)
                                dataset_metrics[dataset_name]["large_ious"].append(sample_iou)
                                dataset_metrics[dataset_name]["large_f1s"].append(f)

                        if save_all_vis_enabled or (
                            vis_payload is None
                            and batch_idx in vis_batch_indices
                            and save_vis_path
                        ):
                            orig_vis = orig_i.detach().to("cpu", dtype=torch.float32)
                            if orig_vis.dim() == 4:
                                orig_vis = orig_vis.squeeze(0)
                            if orig_vis.dim() != 3:
                                raise ValueError(f"Unexpected tiled-orig shape: {tuple(orig_vis.shape)}")

                            source_vis = source_i.detach().to("cpu", dtype=torch.float32)
                            if source_vis.dim() == 4:
                                source_vis = source_vis.squeeze(0)
                            if source_vis.dim() != 3:
                                source_vis = orig_vis

                            gt_vis = tgt_i_b.detach().to("cpu", dtype=torch.float32)
                            prob_vis = probs_i.detach().to("cpu", dtype=torch.float32)
                            bin_vis = pred_bin_i.detach().to("cpu", dtype=torch.float32)
                            det_vis = detection_logit_i.detach().to("cpu", dtype=torch.float32) if detection_logit_i is not None else None
                            coarse_vis = None
                            if hasattr(model, "ferret_backbone") and hasattr(model, "transforms"):
                                try:
                                    orig_for_prompt = orig_i.detach().to("cpu", dtype=torch.float32)
                                    if orig_for_prompt.dim() == 4:
                                        orig_for_prompt = orig_for_prompt.squeeze(0)
                                    orig_for_prompt = model.transforms.transforms(orig_for_prompt)
                                    orig_for_prompt = orig_for_prompt.unsqueeze(0).to(device, non_blocking=True)
                                    with torch.amp.autocast(
                                        device_type=device.type,
                                        enabled=device.type == "cuda",
                                        cache_enabled=False,
                                    ):
                                        _, coarse_vis, _ = model.ferret_backbone(orig_for_prompt)
                                    coarse_vis = coarse_vis.detach().to("cpu", dtype=torch.float32)
                                except Exception:
                                    logger.exception("Failed to compute coarse mask for tiled visualization")

                            vis_payload = {
                                "source": source_vis.unsqueeze(0),
                                "orig": orig_vis.unsqueeze(0),
                                "gt_mask": gt_vis,
                                "pred_prob": prob_vis,
                                "pred_binary": bin_vis,
                                "coarse_mask": coarse_vis,
                                "ious": [sample_iou],
                                "f1s": [f],
                                "dataset_names": [dataset_name],
                                "turn_indices": [turn_indices[i] if turn_indices is not None else 0],
                                "prev_masks": [prev_masks[i] if prev_masks is not None else None],
                                "detection_logits": det_vis,
                            }

                            if save_all_vis_enabled:
                                create_batch_visualization(
                                    vis_payload["source"],
                                    vis_payload["orig"],
                                    streams,
                                    vis_payload["gt_mask"],
                                    vis_payload["pred_prob"],
                                    vis_payload["pred_binary"],
                                    coarse_mask=vis_payload["coarse_mask"],
                                    mask_diff=None,
                                    ious=vis_payload["ious"],
                                    f1s=vis_payload["f1s"],
                                    batch_idx=batch_idx,
                                    save_vis_path=save_vis_path,
                                    dataset_names=vis_payload["dataset_names"],
                                    turn_indices=vis_payload["turn_indices"],
                                    prev_masks=vis_payload["prev_masks"],
                                    detection_logits=vis_payload["detection_logits"],
                                    heatmap_prompt=None,
                                    forensic_features=None,
                                    display_samples=1,
                                    save_name_suffix=f"sample_{i:03d}",
                                )
                                vis_payload = None

                    if vis_payload is not None:
                        create_batch_visualization(
                            vis_payload["source"],
                            vis_payload["orig"],
                            streams,
                            vis_payload["gt_mask"],
                            vis_payload["pred_prob"],
                            vis_payload["pred_binary"],
                            coarse_mask=vis_payload["coarse_mask"],
                            mask_diff=None,
                            ious=vis_payload["ious"],
                            f1s=vis_payload["f1s"],
                            batch_idx=batch_idx,
                            save_vis_path=save_vis_path,
                            dataset_names=vis_payload["dataset_names"],
                            turn_indices=vis_payload["turn_indices"],
                            prev_masks=vis_payload["prev_masks"],
                            detection_logits=vis_payload["detection_logits"],
                            heatmap_prompt=None,
                            forensic_features=None,
                            display_samples=1,
                        )

            except Exception:
                error_count += 1
                if error_count <= max_errors:
                    logger.exception("Error processing validation batch %s", batch_idx)
                    if error_count == max_errors:
                        logger.warning("Suppressing further validation error messages after %s failures.", max_errors)
                continue

    # Handle edge case where no successful batches were processed
    if not ious:
        logger.warning("No localization samples were processed during validation.")

    mean_iou = np.mean(ious) if ious else 0.0
    mean_f1 = np.mean(fs) if fs else 0.0

    summary_lines = [
        "",
        "=" * 80,
        "OVERALL VALIDATION RESULTS",
        "=" * 80,
        f"Overall: IoU={mean_iou:.3f} | F1={mean_f1:.3f} | Total={len(ious)}",
    ]

    if pristine_total > 0:
        pristine_accuracy = pristine_correct / pristine_total
        summary_lines.append(
            f"Pristine Images: {pristine_correct}/{pristine_total} correctly classified as clean ({pristine_accuracy:.3f})"
        )
    else:
        summary_lines.append("Pristine Images: No pristine images found in validation set")

    if detection_total > 0:
        detection_accuracy = detection_correct / detection_total
        detection_precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0.0
        detection_recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0.0
        detection_f1 = (
            2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
            if (detection_precision + detection_recall) > 0
            else 0.0
        )
        authentic_total = detection_tn + detection_fp
        forgery_total = detection_tp + detection_fn
        authentic_accuracy = detection_tn / authentic_total if authentic_total > 0 else 0.0
        forgery_accuracy = detection_tp / forgery_total if forgery_total > 0 else 0.0
        if detection_scores and len(set(detection_labels)) > 1:
            detection_ap = average_precision_score(detection_labels, detection_scores)
        else:
            detection_ap = 0.0
        summary_lines.append(
            f"Classification: ACC={detection_accuracy:.3f} | AP={detection_ap:.3f} | "
            f"Authentic ACC={authentic_accuracy:.3f} | Forgery ACC={forgery_accuracy:.3f}"
        )
        summary_lines.append(
            f"Detection: Accuracy={detection_accuracy:.3f} | Precision={detection_precision:.3f} | "
            f"Recall={detection_recall:.3f} | F1={detection_f1:.3f} | AP={detection_ap:.3f}"
        )
        summary_lines.append(
            f"Detection Confusion: TP={detection_tp}, TN={detection_tn}, FP={detection_fp}, FN={detection_fn}"
        )

    logger.info("\n".join(summary_lines))

    dataset_log_lines = [
        "",
        "=" * 80,
        "DATASET-SPECIFIC VALIDATION RESULTS",
        "=" * 80,
    ]

    dataset_results = {}
    for dataset_name, metrics in dataset_metrics.items():
        if metrics['ious']:  # Only print if we have samples
            dataset_log_lines.append("")
            dataset_log_lines.append(dataset_name.upper())
            dataset_log_lines.append("-" * 60)
            
            # Overall for this dataset
            dataset_iou = np.mean(metrics['ious'])
            dataset_f1 = np.mean(metrics['f1s'])
            dataset_log_lines.append(
                f"Overall: IoU={dataset_iou:.3f} | F1={dataset_f1:.3f} | Count={len(metrics['ious'])}"
            )
            
            # Pristine accuracy for this dataset
            if metrics['pristine_total'] > 0:
                dataset_pristine_acc = metrics['pristine_correct'] / metrics['pristine_total']
                dataset_log_lines.append(
                    f"Pristine: {metrics['pristine_correct']}/{metrics['pristine_total']} correctly classified as clean ({dataset_pristine_acc:.3f})"
                )
            
            # Detection accuracy for this dataset
            if metrics['detection_total'] > 0:
                dataset_detection_acc = metrics['detection_correct'] / metrics['detection_total']
                dataset_detection_precision = metrics['detection_tp'] / (metrics['detection_tp'] + metrics['detection_fp']) if (metrics['detection_tp'] + metrics['detection_fp']) > 0 else 0.0
                dataset_detection_recall = metrics['detection_tp'] / (metrics['detection_tp'] + metrics['detection_fn']) if (metrics['detection_tp'] + metrics['detection_fn']) > 0 else 0.0
                dataset_detection_f1 = 2 * (dataset_detection_precision * dataset_detection_recall) / (dataset_detection_precision + dataset_detection_recall) if (dataset_detection_precision + dataset_detection_recall) > 0 else 0.0
                dataset_authentic_total = metrics['detection_tn'] + metrics['detection_fp']
                dataset_forgery_total = metrics['detection_tp'] + metrics['detection_fn']
                dataset_authentic_acc = metrics['detection_tn'] / dataset_authentic_total if dataset_authentic_total > 0 else 0.0
                dataset_forgery_acc = metrics['detection_tp'] / dataset_forgery_total if dataset_forgery_total > 0 else 0.0
                if metrics['detection_scores'] and len(set(metrics['detection_labels'])) > 1:
                    dataset_detection_ap = average_precision_score(metrics['detection_labels'], metrics['detection_scores'])
                else:
                    dataset_detection_ap = 0.0
                dataset_log_lines.append(
                    f"Classification: ACC={dataset_detection_acc:.3f} | AP={dataset_detection_ap:.3f} | "
                    f"Authentic ACC={dataset_authentic_acc:.3f} | Forgery ACC={dataset_forgery_acc:.3f}"
                )
                dataset_log_lines.append(
                    f"Detection: Accuracy={dataset_detection_acc:.3f} | Precision={dataset_detection_precision:.3f} | "
                    f"Recall={dataset_detection_recall:.3f} | F1={dataset_detection_f1:.3f} | AP={dataset_detection_ap:.3f}"
                )
            
            # Size-stratified for this dataset
            if metrics['small_ious']:
                dataset_log_lines.append(
                    f"Small (<5%):   IoU={np.mean(metrics['small_ious']):.3f} | F1={np.mean(metrics['small_f1s']):.3f} | Count={len(metrics['small_ious'])}"
                )
            if metrics['medium_ious']:
                dataset_log_lines.append(
                    f"Medium (5-20%): IoU={np.mean(metrics['medium_ious']):.3f} | F1={np.mean(metrics['medium_f1s']):.3f} | Count={len(metrics['medium_ious'])}"
                )
            if metrics['large_ious']:
                dataset_log_lines.append(
                    f"Large (>20%):  IoU={np.mean(metrics['large_ious']):.3f} | F1={np.mean(metrics['large_f1s']):.3f} | Count={len(metrics['large_ious'])}"
                )
            
            # Store results for return (including detection metrics)
            dataset_results[dataset_name] = {
                "iou": dataset_iou,
                "f1": dataset_f1,
                "count": len(metrics['ious']),
                "pristine_correct": metrics['pristine_correct'],
                "pristine_total": metrics['pristine_total'],
                "pristine_accuracy": metrics['pristine_correct'] / metrics['pristine_total'] if metrics['pristine_total'] > 0 else 0.0,
                "detection_correct": metrics['detection_correct'],
                "detection_total": metrics['detection_total'],
                "detection_accuracy": metrics['detection_correct'] / metrics['detection_total'] if metrics['detection_total'] > 0 else 0.0,
                "detection_precision": metrics['detection_tp'] / (metrics['detection_tp'] + metrics['detection_fp']) if (metrics['detection_tp'] + metrics['detection_fp']) > 0 else 0.0,
                "detection_recall": metrics['detection_tp'] / (metrics['detection_tp'] + metrics['detection_fn']) if (metrics['detection_tp'] + metrics['detection_fn']) > 0 else 0.0,
                "detection_f1": 0.0,  # Will be calculated below
                "detection_ap": 0.0,
                "classification_accuracy": metrics['detection_correct'] / metrics['detection_total'] if metrics['detection_total'] > 0 else 0.0,
                "classification_ap": 0.0,
                "authentic_accuracy": metrics['detection_tn'] / (metrics['detection_tn'] + metrics['detection_fp']) if (metrics['detection_tn'] + metrics['detection_fp']) > 0 else 0.0,
                "forgery_accuracy": metrics['detection_tp'] / (metrics['detection_tp'] + metrics['detection_fn']) if (metrics['detection_tp'] + metrics['detection_fn']) > 0 else 0.0,
                "detection_tp": metrics['detection_tp'],
                "detection_tn": metrics['detection_tn'],
                "detection_fp": metrics['detection_fp'],
                "detection_fn": metrics['detection_fn'],
                "small_iou": np.mean(metrics['small_ious']) if metrics['small_ious'] else 0.0,
                "small_f1": np.mean(metrics['small_f1s']) if metrics['small_f1s'] else 0.0,
                "small_count": len(metrics['small_ious']),
                "medium_iou": np.mean(metrics['medium_ious']) if metrics['medium_ious'] else 0.0,
                "medium_f1": np.mean(metrics['medium_f1s']) if metrics['medium_f1s'] else 0.0,
                "medium_count": len(metrics['medium_ious']),
                "large_iou": np.mean(metrics['large_ious']) if metrics['large_ious'] else 0.0,
                "large_f1": np.mean(metrics['large_f1s']) if metrics['large_f1s'] else 0.0,
                "large_count": len(metrics['large_ious']),
            }
            
            # Calculate detection F1
            detection_precision = dataset_results[dataset_name]["detection_precision"]
            detection_recall = dataset_results[dataset_name]["detection_recall"]
            if detection_precision + detection_recall > 0:
                dataset_results[dataset_name]["detection_f1"] = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
            if metrics['detection_scores'] and len(set(metrics['detection_labels'])) > 1:
                dataset_results[dataset_name]["detection_ap"] = average_precision_score(
                    metrics['detection_labels'], metrics['detection_scores']
                )
            dataset_results[dataset_name]["classification_ap"] = dataset_results[dataset_name]["detection_ap"]
    
    dataset_log_lines.append("=" * 80)
    logger.info("\n".join(dataset_log_lines))

    # Calculate overall size-stratified metrics for backward compatibility
    all_small_ious = [iou for metrics in dataset_metrics.values() for iou in metrics['small_ious']]
    all_medium_ious = [iou for metrics in dataset_metrics.values() for iou in metrics['medium_ious']]
    all_large_ious = [iou for metrics in dataset_metrics.values() for iou in metrics['large_ious']]
    all_small_f1s = [f1 for metrics in dataset_metrics.values() for f1 in metrics['small_f1s']]
    all_medium_f1s = [f1 for metrics in dataset_metrics.values() for f1 in metrics['medium_f1s']]
    all_large_f1s = [f1 for metrics in dataset_metrics.values() for f1 in metrics['large_f1s']]

    # Calculate overall detection metrics
    overall_detection_accuracy = detection_correct / detection_total if detection_total > 0 else 0.0
    overall_detection_precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0.0
    overall_detection_recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0.0
    overall_detection_f1 = 2 * (overall_detection_precision * overall_detection_recall) / (overall_detection_precision + overall_detection_recall) if (overall_detection_precision + overall_detection_recall) > 0 else 0.0
    overall_authentic_total = detection_tn + detection_fp
    overall_forgery_total = detection_tp + detection_fn
    overall_authentic_accuracy = detection_tn / overall_authentic_total if overall_authentic_total > 0 else 0.0
    overall_forgery_accuracy = detection_tp / overall_forgery_total if overall_forgery_total > 0 else 0.0
    if detection_scores and len(set(detection_labels)) > 1:
        overall_detection_ap = average_precision_score(detection_labels, detection_scores)
    else:
        overall_detection_ap = 0.0
    return {
        "iou": mean_iou, 
        "f1": mean_f1,
        "pristine_correct": pristine_correct,
        "pristine_total": pristine_total,
        "pristine_accuracy": pristine_correct / pristine_total if pristine_total > 0 else 0.0,
        "detection_correct": detection_correct,
        "detection_total": detection_total,
        "detection_accuracy": overall_detection_accuracy,
        "detection_precision": overall_detection_precision,
        "detection_recall": overall_detection_recall,
        "detection_f1": overall_detection_f1,
        "detection_ap": overall_detection_ap,
        "classification_accuracy": overall_detection_accuracy,
        "classification_ap": overall_detection_ap,
        "authentic_accuracy": overall_authentic_accuracy,
        "forgery_accuracy": overall_forgery_accuracy,
        "detection_tp": detection_tp,
        "detection_tn": detection_tn,
        "detection_fp": detection_fp,
        "detection_fn": detection_fn,
        "small_iou": np.mean(all_small_ious) if all_small_ious else 0.0,
        "small_f1": np.mean(all_small_f1s) if all_small_f1s else 0.0,
        "small_count": len(all_small_ious),
        "medium_iou": np.mean(all_medium_ious) if all_medium_ious else 0.0,
        "medium_f1": np.mean(all_medium_f1s) if all_medium_f1s else 0.0,
        "medium_count": len(all_medium_ious),
        "large_iou": np.mean(all_large_ious) if all_large_ious else 0.0,
        "large_f1": np.mean(all_large_f1s) if all_large_f1s else 0.0,
        "large_count": len(all_large_ious),
        "dataset_results": dataset_results,
    }


def create_multi_region_mask_visualization(gt_mask, pred_mask, turn_idx, prev_masks=None):
    """
    Create colored visualizations showing different edited regions
    
    Args:
        gt_mask: Ground truth cumulative mask
        pred_mask: Predicted cumulative mask  
        turn_idx: Current turn index
        prev_masks: List of previous turn masks (if available)
        
    Returns:
        colored_gt: Ground truth with regions colored by turn
        colored_pred: Prediction with connected components colored differently
    """
    # Convert to numpy and ensure binary
    gt_np = (gt_mask.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    pred_np = (pred_mask.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    
    h, w = gt_np.shape
    
    # Create colored ground truth based on turn progression
    colored_gt = np.zeros((h, w, 3))
    
    if prev_masks is not None and len(prev_masks) > 0:
        # Color each turn's contribution differently
        colors = [
            [1.0, 0.0, 0.0],  # Red for turn 0
            [0.0, 1.0, 0.0],  # Green for turn 1  
            [0.0, 0.0, 1.0],  # Blue for turn 2
            [1.0, 1.0, 0.0],  # Yellow for turn 3
            [1.0, 0.0, 1.0],  # Magenta for turn 4
            [0.0, 1.0, 1.0],  # Cyan for turn 5
            [1.0, 0.5, 0.0],  # Orange for turn 6
            [0.5, 0.0, 1.0],  # Purple for turn 7
        ]
        
        # Start with cumulative mask from previous turns
        cumulative_prev = np.zeros_like(gt_np, dtype=np.uint8)
        for i, prev_mask in enumerate(prev_masks):
            if prev_mask is not None:
                prev_np = (prev_mask.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
                if prev_np.shape == gt_np.shape:
                    # Color pixels that are new in this turn
                    new_pixels = prev_np & (~cumulative_prev)  # Use parentheses for clarity
                    color = colors[i % len(colors)]
                    colored_gt[new_pixels > 0] = color
                    cumulative_prev = cumulative_prev | prev_np
        
        # Color current turn's new pixels
        current_new = gt_np & (~cumulative_prev)
        current_color = colors[turn_idx % len(colors)]
        colored_gt[current_new > 0] = current_color
        
    else:
        # If no previous masks available, just use single color
        colored_gt[gt_np > 0] = [1.0, 0.0, 0.0]  # Red
    
    # Create colored prediction based on connected components
    colored_pred = np.zeros((h, w, 3))
    
    if pred_np.max() > 0:
        # Find connected components in prediction
        labeled_pred, num_components = ndimage.label(pred_np > 0)
        
        # Use different colors for different components
        component_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]
        
        for comp_id in range(1, num_components + 1):
            component_mask = labeled_pred == comp_id
            color = component_colors[(comp_id - 1) % len(component_colors)]
            colored_pred[component_mask] = color
    
    return colored_gt, colored_pred


def create_batch_visualization(source, orig, streams, gt_mask, pred_prob, pred_binary,
                               coarse_mask, mask_diff,
                               ious, f1s, batch_idx, save_vis_path,
                               dataset_names=None, turn_indices=None, prev_masks=None,
                               detection_logits=None, heatmap_prompt=None, forensic_features=None,
                               display_samples=None, save_name_suffix=None):  # Add forensic_features parameter
    """Create visualization for an entire batch with multi-region coloring"""
    os.makedirs(save_vis_path, exist_ok=True)
    
    batch_size = source.shape[0]
    if display_samples is None:
        display_samples = min(batch_size, 1)
    else:
        display_samples = min(display_samples, batch_size)
    
    # Number of columns for the panels we actually render.
    num_cols = 8  # source, orig, gt, gt_colored, pred_prob, pred_binary, coarse, lpd
    
    if save_name_suffix:
        batch_save_name = f"batch_{batch_idx:03d}_{save_name_suffix}.png"
    else:
        batch_save_name = f"batch_{batch_idx:03d}_grid.png"
    batch_save_path = os.path.join(save_vis_path, batch_save_name)
    
    fig, axes = plt.subplots(display_samples, num_cols, figsize=(3 * num_cols, 3 * display_samples))
    
    if display_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(display_samples):
        # Get turn info for this sample
        turn_idx = turn_indices[i] if turn_indices is not None else 0
        sample_prev_masks = prev_masks[i] if len(prev_masks) > 0 else None
        
        # Get detection info for this sample
        detection_info = ""
        if detection_logits is not None:
            detection_prob = torch.sigmoid(detection_logits[i]).item()
            detection_pred = detection_prob > 0.5
            is_pristine = (gt_mask[i].sum().item() == 0)
            detection_gt = not is_pristine
            
            # Create detection status string
            pred_label = "FORGERY" if detection_pred else "AUTHENTIC"
            gt_label = "FORGERY" if detection_gt else "AUTHENTIC"
            correct = "✓" if detection_pred == detection_gt else "✗"
            detection_info = f"\nDetection: {pred_label} {correct} (GT: {gt_label}, P={detection_prob:.2f})"
        
        # Only create multi-region visualization for multi-turn samples
        if turn_idx > 0:
            # Create multi-region colored visualizations
            colored_gt, colored_pred = create_multi_region_mask_visualization(
                gt_mask[i], pred_binary[i], turn_idx, sample_prev_masks
            )
        else:
            # For single-turn samples, use None to indicate no multi-region visualization
            colored_gt, colored_pred = None, None
        
        source_norm = normalize_for_display(source[i].cpu())
        orig_norm = normalize_for_display(orig[i].cpu())
        
        col_idx = 0
        
        # Add dataset name, turn info, and detection info to source image title
        dataset_name = dataset_names[i] if dataset_names else "unknown"
        source_title = f'Source (Unedited)\n{dataset_name} Turn {turn_idx}{detection_info}'
        axes[i, col_idx].imshow(source_norm.permute(1, 2, 0))
        axes[i, col_idx].set_title(source_title, fontsize=10)
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        axes[i, col_idx].imshow(orig_norm.permute(1, 2, 0))
        axes[i, col_idx].set_title('Target (Edited)')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        axes[i, col_idx].imshow(gt_mask[i].cpu().squeeze(), cmap='gray')
        axes[i, col_idx].set_title('GT Mask')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # Add colored GT mask showing different turns (only for multi-turn)
        if colored_gt is not None:
            axes[i, col_idx].imshow(colored_gt)
            axes[i, col_idx].set_title('GT Multi-Region')
        else:
            axes[i, col_idx].imshow(gt_mask[i].cpu().squeeze(), cmap='gray')
            axes[i, col_idx].set_title('GT Single-Turn')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        axes[i, col_idx].imshow(pred_prob[i].cpu().squeeze(), cmap='plasma', vmin=0, vmax=1)
        axes[i, col_idx].set_title('Pred Prob')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        axes[i, col_idx].imshow(pred_binary[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, col_idx].set_title(f'Pred Binary (IoU:{ious[i]:.3f})')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        if coarse_mask is not None:
            axes[i, col_idx].imshow(coarse_mask[i].cpu().squeeze(), cmap='plasma', 
                                  vmin=coarse_mask.min().item(), vmax=coarse_mask.max().item())
            axes[i, col_idx].set_title('Coarse Mask')
        else:
            axes[i, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                transform=axes[i, col_idx].transAxes)
            axes[i, col_idx].set_title('Coarse Mask')
        axes[i, col_idx].axis('off')
        col_idx += 1

        # if mask_diff is not None:
        #     axes[i, col_idx].imshow(mask_diff[i].cpu().squeeze(), cmap='plasma', 
        #                           vmin=mask_diff.min().item(), vmax=mask_diff.max().item())
        #     axes[i, col_idx].set_title('Fine Mask')
        # else:
        #     axes[i, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', 
        #                         transform=axes[i, col_idx].transAxes)
        #     axes[i, col_idx].set_title('Fine Mask')
        # axes[i, col_idx].axis('off')
        # col_idx += 1
        
        # Add LPD Input (forensic_features) visualization
        if forensic_features is not None:
            # Forensic features have multiple channels, take the first channel for visualization
            # or create a composite view using mean across channels
            lpd_visual = forensic_features[i].mean(dim=0).detach().float().cpu().squeeze()
            axes[i, col_idx].imshow(lpd_visual, cmap='plasma')
            axes[i, col_idx].set_title('LPD Input')
        else:
            axes[i, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                transform=axes[i, col_idx].transAxes)
            axes[i, col_idx].set_title('LPD Input')
        axes[i, col_idx].axis('off')
        col_idx += 1
        
        # # Add GLP Heatmap Prompt visualization
        # if heatmap_prompt is not None:
        #     axes[i, col_idx].imshow(heatmap_prompt[i].cpu().squeeze(), cmap='plasma', 
        #                           vmin=0, vmax=1)
        #     axes[i, col_idx].set_title('GLP Heatmap')
        # else:
        #     axes[i, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', 
        #                         transform=axes[i, col_idx].transAxes)
        #     axes[i, col_idx].set_title('GLP Heatmap')
        # axes[i, col_idx].axis('off')
    
    avg_iou = np.mean(ious[:display_samples])
    avg_f1 = np.mean(f1s[:display_samples])
    fig.suptitle(f'Batch {batch_idx} - Avg IoU: {avg_iou:.3f} | Avg F1: {avg_f1:.3f}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(batch_save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info("Batch visualization saved to: %s", batch_save_path)


def normalize_for_display(tensor):
    """Normalize tensor for display (0-1 range)"""
    tensor = tensor.clone()
    
    if tensor.dim() == 3:  # [C, H, W]
        for c in range(tensor.shape[0]):
            channel = tensor[c]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            tensor[c] = channel
    else:  # [H, W] 
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    return tensor.clamp(0, 1)
