import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.metrics import compute_all_metrics, log_metrics


logger = logging.getLogger(__name__)


def _tta_forward(model, images, tta_mode='flip'):

    # --- Forward on original images ---
    _features, logits = model(images, return_feature=True)

    if tta_mode in ('flip',):
        # Horizontal flip TTA: average logits from original + flipped.
        images_flipped = torch.flip(images, dims=[-1])
        _feat_f, logits_f = model(images_flipped, return_feature=True)
        logits = (logits + logits_f) / 2.0

    elif tta_mode == 'flip_multiscale':
        # Horizontal flip at 1.0 scale.
        images_flipped = torch.flip(images, dims=[-1])
        _feat_f, logits_f = model(images_flipped, return_feature=True)
        logits = (logits + logits_f) / 2.0

        # Multi-scale: 0.85x and 1.15x center crops.
        B, C, H, W = images.shape
        for scale in (0.85, 1.15):
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            dh = (H - crop_h) // 2
            dw = (W - crop_w) // 2
            cropped = images[:, :, dh:dh + crop_h, dw:dw + crop_w]
            scaled = F.interpolate(
                cropped, size=(H, W), mode='bilinear', align_corners=False,
            )
            _feat_s, logits_s = model(scaled, return_feature=True)
            logits = logits + logits_s
            scaled_f = torch.flip(scaled, dims=[-1])
            _feat_sf, logits_sf = model(scaled_f, return_feature=True)
            logits = logits + logits_sf
        # Divide by 1 (orig) + 1 (flip) + 2x2 (two scales, each flipped) = 6.
        logits = logits / 6.0

    return logits


def evaluate_model(
    model,
    device,
    dataset,
    batch_size,
    num_workers,
    use_tta=False,
    tta_mode='flip',
):

    model.eval()

    subset_metrics: dict[str, dict[str, float]] = {}
    all_labels: list = []
    all_scores: list = []

    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        subset_labels: list = []
        subset_scores: list = []

        with torch.no_grad():
            for batch in subset_loader:
                # Subset-aware datasets may return either
                #     (image, label, subset_name)   or   (image, label).
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(device)
                labels = labels.to(device)

                if use_tta:
                    logits = _tta_forward(model, images, tta_mode=tta_mode)
                else:
                    _features, logits = model(images, return_feature=True)

                logits = logits.view(-1)
                probs = torch.sigmoid(logits)

                subset_labels.extend(labels.cpu().numpy())
                subset_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        subset_y_true = np.array(subset_labels)
        subset_y_score = np.array(subset_scores)

        metrics = compute_all_metrics(subset_y_true, subset_y_score)
        subset_metrics[subset_name] = metrics
        log_metrics(metrics, subset_name)

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    overall_metrics = compute_all_metrics(y_true, y_score)

    return subset_metrics, overall_metrics
