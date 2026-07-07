"""Training script for the FLAME forgery localization model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import warnings
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils.train_utils import custom_collate_fn, get_param_strings
from utils.dataset_config import DatasetManager, load_dataset_config, create_default_config
from utils.sam_utils import initialize_sam_hydra
from utils.validation_manifest import load_validation_manifest


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high") 
# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Initialize Hydra configuration for SAM2
initialize_sam_hydra()

# Set multiprocessing start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

# Configure BatchNorm and activation functions
try:
    BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    relu_inplace = True
except AttributeError:
    BatchNorm2d = torch.nn.BatchNorm2d
    BatchNorm2d_class = BatchNorm2d
    relu_inplace = False

# Import project modules
from model.ema import EMA
from model.adapters import adapter_delta_regularization
from model.forgerylocalizer import ForgeryLocalizer
from utils.localforgerydataset import LocalForgeryDataset
from utils.metrics import sigmoid_focal_loss, dice_loss
from utils.metrics import plot_all_metrics
from utils.validate import validate
from utils.checkpoint_state import build_checkpoint_payload, load_matching_state_dict
from utils.live_reload_diagnostics import compare_models_on_batches

def get_max_streams() -> int:
    """
    Ferret-SAM doesn't use perturbation streams, so always return 0.
    
    Returns:
        Maximum number of streams the model should expect
    """
    return 0


def select_local_input(batch: Dict[str, Any]) -> torch.Tensor:
    """Return the tensor that should enter FerretBackbone/LAD.

    Training samples historically exposed ``local_patch`` after SAM2/ImageNet
    normalization.  The paper-style LAD operator is defined on raw RGB values,
    so prefer the explicit raw patch when the dataset provides it.  Keep the
    old fallback order for compatibility with older configs/checkpoints.
    """
    return batch.get("local_patch_raw", batch.get("local_patch", batch["orig"]))


def parse_adapter_scales(value: Optional[str]) -> Optional[List[int]]:
    """Parse adapter scale CLI strings.

    ``None``, an empty string, or ``all`` means all adapter scales are active.
    Otherwise the value must be a comma-separated subset of ``0,1,2``.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "all", "none"}:
        return None
    scales = [int(part.strip()) for part in text.split(",") if part.strip()]
    invalid = [scale for scale in scales if scale not in {0, 1, 2}]
    if invalid:
        raise ValueError(f"adapter scales must be drawn from 0,1,2; got {invalid}")
    return scales


def parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated float list; empty/all/none disables override."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "all", "none"}:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def summarize_adapter_diagnostics(adapter_diagnostics: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Convert per-scale adapter diagnostics into flat loggable scalars."""
    summary: Dict[str, float] = {}
    for stats in adapter_diagnostics:
        scale = int(stats.get("scale_idx", len(summary)))
        for key in (
            "delta_mse",
            "delta_ratio",
            "cosine",
            "semantic_mean",
            "semantic_std",
            "forensic_mean",
            "forensic_std",
            "adapted_mean",
            "adapted_std",
            "raw_delta_mean",
            "raw_delta_std",
            "gamma",
        ):
            if key in stats:
                value = stats[key]
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().item()
                summary[f"adapter_s{scale}_{key}"] = float(value)
        summary[f"adapter_s{scale}_active"] = float(bool(stats.get("active", True)))
    active_ratios = [
        summary[k] for k in summary if k.endswith("_delta_ratio") and summary.get(k.replace("_delta_ratio", "_active"), 1.0)
    ]
    if active_ratios:
        summary["adapter_delta_ratio_mean"] = float(sum(active_ratios) / len(active_ratios))
    return summary


def apply_ema_weights_to_model(model: nn.Module, ema_state: Dict[str, torch.Tensor]) -> int:
    """Copy EMA tensors into matching model parameters regardless of trainability."""
    model_params = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for name, tensor in ema_state.items():
            param = model_params.get(name)
            if param is None or tuple(param.shape) != tuple(tensor.shape):
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded += 1
    return loaded


def collect_trainable_params(
    model: nn.Module,
    freeze_adapters: bool = False,
    freeze_ferret: bool = False,
    freeze_ferret_unet_only: bool = False,
) -> List[nn.Parameter]:
    """Collect optimizer parameters for SAM-frozen FLAME training.

    When ``freeze_adapters`` is enabled, the residual adapters remain at their
    initialized identity mapping.  This is useful for SAM3 diagnosis/training
    because the frozen SAM3 decoder is sensitive to feature-distribution drift.
    When ``freeze_ferret`` is enabled, the coarse LAD/Ferret prompt remains fixed
    so a second-stage run can fine-tune adapters without moving the prompt head.
    When ``freeze_ferret_unet_only`` is enabled, only the residual-only U-Net
    prompt branch remains trainable, preserving the legacy prompt generator.
    """
    trainable_params: List[nn.Parameter] = []

    adapters = getattr(model, "adapters", None)
    if adapters is not None:
        for param in adapters.parameters():
            param.requires_grad = not freeze_adapters
        if not freeze_adapters:
            trainable_params.extend(adapters.parameters())
    pyramid_adapters = getattr(model, "pyramid_adapters", None)
    if pyramid_adapters is not None:
        for param in pyramid_adapters.parameters():
            param.requires_grad = not freeze_adapters
        if not freeze_adapters:
            trainable_params.extend(pyramid_adapters.parameters())

    ferret_backbone = getattr(model, "ferret_backbone", None)
    if ferret_backbone is not None:
        if freeze_ferret_unet_only:
            for name, param in ferret_backbone.named_parameters():
                trainable = name.startswith("prompt_head_unet")
                param.requires_grad = trainable
                if trainable:
                    trainable_params.append(param)
        else:
            for param in ferret_backbone.parameters():
                param.requires_grad = not freeze_ferret
            if not freeze_ferret:
                trainable_params.extend(ferret_backbone.parameters())

    prompt_calibrator = getattr(model, "prompt_calibrator", None)
    if prompt_calibrator is not None:
        for param in prompt_calibrator.parameters():
            param.requires_grad = True
        trainable_params.extend(prompt_calibrator.parameters())
    prompt_refiner = getattr(model, "prompt_refiner", None)
    if prompt_refiner is not None:
        for param in prompt_refiner.parameters():
            param.requires_grad = True
        trainable_params.extend(prompt_refiner.parameters())
    dual_prompt_fusion_gate = getattr(model, "dual_prompt_fusion_gate", None)
    if dual_prompt_fusion_gate is not None:
        for param in dual_prompt_fusion_gate.parameters():
            param.requires_grad = True
        trainable_params.extend(dual_prompt_fusion_gate.parameters())
    final_logit_calibrator = getattr(model, "final_logit_calibrator", None)
    if final_logit_calibrator is not None:
        for param in final_logit_calibrator.parameters():
            param.requires_grad = True
        trainable_params.extend(final_logit_calibrator.parameters())

    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        for name, param in decoder.named_parameters():
            if "iou_prediction_head" in name and param.requires_grad:
                trainable_params.append(param)

    return trainable_params


def build_optimizer_param_groups(
    model: nn.Module,
    trainable_params: List[nn.Parameter],
    *,
    base_lr: float,
    prompt_calibrator_lr_multiplier: float = 1.0,
    coarse_prompt_head_lr_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups with optional boosted prompt-module LRs."""
    calibrator_multiplier = float(prompt_calibrator_lr_multiplier)
    prompt_head_multiplier = float(coarse_prompt_head_lr_multiplier)
    prompt_modules = [
        module
        for module in (
            getattr(model, "prompt_calibrator", None),
            getattr(model, "prompt_refiner", None),
            getattr(model, "dual_prompt_fusion_gate", None),
            getattr(model, "final_logit_calibrator", None),
        )
        if module is not None
    ]
    calibrator_param_ids = {
        id(p)
        for module in prompt_modules
        for p in module.parameters()
        if p.requires_grad
    }
    if calibrator_multiplier == 1.0:
        calibrator_param_ids.clear()

    prompt_head_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if (
            "ferret_backbone.prompt_head_" in name
            or "ferret_backbone.lad_tau_fusion" in name
            or "ferret_backbone.hybrid_mldc" in name
        )
        and param.requires_grad
    }
    if prompt_head_multiplier == 1.0:
        prompt_head_param_ids.clear()

    boosted_param_ids = calibrator_param_ids | prompt_head_param_ids
    if not boosted_param_ids:
        return [{"params": trainable_params, "lr": float(base_lr)}]

    calibrator_params = [p for p in trainable_params if id(p) in calibrator_param_ids]
    prompt_head_params = [p for p in trainable_params if id(p) in prompt_head_param_ids]
    other_params = [p for p in trainable_params if id(p) not in boosted_param_ids]
    groups: List[Dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr": float(base_lr)})
    if calibrator_params:
        groups.append({"params": calibrator_params, "lr": float(base_lr) * calibrator_multiplier})
    if prompt_head_params:
        groups.append({"params": prompt_head_params, "lr": float(base_lr) * prompt_head_multiplier})
    return groups


def run_live_reload_diagnostic(
    *,
    live_model: ForgeryLocalizer,
    checkpoint_path: str,
    model_params: Dict[str, Any],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str,
    num_batches: int,
) -> Dict[str, Any]:
    """Compare the live EMA-applied model against a fresh checkpoint reload.

    This is a debugging-only gate for SAM3/LAD recovery.  It runs both models
    on the exact same already-collated validation batches, so any non-trivial
    logit difference points at model state/checkpoint/reload differences rather
    than validation sampling.
    """
    from test_dataset import load_and_initialize_model

    logger.info(
        "Running live-vs-reload diagnostic | checkpoint=%s batches=%s",
        checkpoint_path,
        num_batches,
    )
    reload_model = load_and_initialize_model(model_params, checkpoint_path, device, use_ema=False)
    try:
        report = compare_models_on_batches(
            live_model=live_model,
            reload_model=reload_model,
            loader=val_loader,
            device=device,
            num_batches=int(num_batches),
            amp=True,
        )
    finally:
        del reload_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    tensor_summaries = report.get("tensors", {})
    logit_max = tensor_summaries.get("logits", {}).get("max_abs")
    prob_max = tensor_summaries.get("probs", {}).get("max_abs")
    coarse_max = tensor_summaries.get("coarse_mask", {}).get("max_abs")
    logger.info(
        "Live-vs-reload diagnostic saved | path=%s logit_max_abs=%s prob_max_abs=%s coarse_max_abs=%s",
        output_path,
        logit_max,
        prob_max,
        coarse_max,
    )
    return report


def sanitize_experiment_name(name: str, max_length: int = 120) -> str:
    """Convert an experiment name into a filesystem-safe representation.

    Args:
        name: Proposed experiment name.
        max_length: Maximum allowed characters for the sanitized name.

    Returns:
        Sanitized experiment name that avoids OS-reserved characters and long paths.
    """

    safe = re.sub(r"[^A-Za-z0-9_\-]", "-", name)
    safe = re.sub(r"-+", "-", safe)
    safe = safe.strip("-")
    if not safe:
        safe = "DET_experiment"
    if len(safe) > max_length:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:max_length - 9]}_{digest}"
    return safe


def train_epoch(
    model: ForgeryLocalizer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    validate_fn: Optional[callable] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    val_steps: int = 5,
    lambda_focal: float = 20.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    model_save_path: Optional[str] = None,
    best_score: float = 0.0,
    ema: Optional[EMA] = None,
    lambda_iou: float = 1.0,
    train_sam_iou: bool = True,
    contrastive_blur: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
    authentic_ratio: float = 0.0,
    lambda_detection: float = 1.0,
    coarse_loss_weight: float = 0.2,
    adapter_delta_reg_weight: float = 0.0,
    coarse_prompt_calibrator_reg_weight: float = 0.0,
    prompt_bias_supervision_weight: float = 0.0,
    prompt_bias_supervision_loss: str = "smooth_l1",
    prompt_bias_supervision_max_delta_bias: Optional[float] = None,
    dense_prompt_residual_reg_weight: float = 0.0,
    dense_prompt_signed_residual_supervision_weight: float = 0.0,
    dense_prompt_signed_residual_target_source: str = "coarse_mask",
    dense_prompt_signed_residual_loss: str = "smooth_l1",
    dense_prompt_signed_residual_target_scale: float = 1.0,
    dense_prompt_signed_residual_target_mode: str = "hard_error",
    dense_prompt_signed_residual_hard_threshold: float = 0.5,
    dense_prompt_signed_residual_use_gate: bool = False,
    dense_prompt_signed_residual_max_gt_area: Optional[float] = None,
    dense_prompt_unet_identity_weight: float = 0.0,
    dense_prompt_unet_identity_loss: str = "mse",
    dense_prompt_unet_residual_supervision_weight: float = 0.0,
    dense_prompt_unet_residual_loss: str = "balanced_smooth_l1",
    dense_prompt_unet_residual_target_scale: float = 1.0,
    dense_prompt_unet_residual_target_mode: str = "hard_error",
    dense_prompt_unet_residual_hard_threshold: float = 0.5,
    dense_prompt_unet_residual_use_gate: bool = False,
    dense_prompt_unet_residual_max_gt_area: Optional[float] = None,
    coarse_prompt_loss_weight: float = 0.0,
    coarse_prompt_dice_weight: float = 0.0,
    coarse_prompt_false_negative_weight: float = 0.0,
    coarse_prompt_false_positive_weight: float = 0.0,
    coarse_prompt_supervision_source: str = "coarse_prompt",
    coarse_prompt_supervision_max_gt_area: Optional[float] = None,
    prompt_gate_supervision_weight: float = 0.0,
    prompt_gate_target_source: str = "pre_refiner_prompt",
    prompt_gate_target_mode: str = "fn_ratio",
    prompt_gate_sources: str = "refiner",
    prompt_gate_loss: str = "smooth_l1",
    dual_branch_prompt_gate_supervision_weight: float = 0.0,
    dual_branch_prompt_gate_target_source: str = "dense_prompt",
    dual_branch_prompt_gate_target_mode: str = "sample_ratio",
    dual_branch_prompt_gate_hard_threshold: float = 0.5,
    dual_branch_prompt_gate_loss: str = "smooth_l1",
    dual_branch_prompt_residual_supervision_weight: float = 0.0,
    dual_branch_prompt_residual_target_source: str = "dense_prompt",
    dual_branch_prompt_residual_loss: str = "smooth_l1",
    dual_branch_prompt_residual_target_scale: float = 1.0,
    dual_branch_prompt_residual_target_mode: str = "soft_error",
    dual_branch_prompt_residual_hard_threshold: float = 0.5,
    dual_prompt_fusion_supervision_weight: float = 0.0,
    dual_prompt_fusion_oracle_metric: str = "iou",
    dual_prompt_fusion_loss: str = "bce",
    final_logit_calibrator_supervision_weight: float = 0.0,
    final_logit_calibrator_supervision_loss: str = "smooth_l1",
    final_logit_calibrator_supervision_max_delta_bias: Optional[float] = None,
    final_logit_calibrator_oracle_thresholds: Optional[List[float]] = None,
    final_logit_calibrator_oracle_min_threshold: Optional[float] = None,
    final_logit_calibrator_oracle_max_threshold: Optional[float] = None,
    final_logit_calibrator_oracle_false_positive_penalty: float = 0.0,
    final_logit_calibrator_oracle_area_penalty: float = 0.0,
    final_logit_spatial_supervision_weight: float = 0.0,
    final_logit_spatial_supervision_loss: str = "smooth_l1",
    final_logit_spatial_supervision_target_scale: float = 1.0,
    final_logit_spatial_supervision_target_mode: str = "hard_error",
    final_logit_spatial_supervision_hard_threshold: float = 0.5,
    prompt_gate_refiner_max: float = 1.0,
    prompt_gate_dense_max: float = 1.0,
    area_reg_weight: float = 0.0,
    area_reg_target_source: str = "gt",
    area_reg_loss: str = "smooth_l1",
    area_reg_apply_to: str = "coarse",
    area_reg_constant: float = 0.25,
    area_reg_max_gt_area: Optional[float] = None,
    coarse_dice_weight: float = 0.0,
    reload_diagnostics_batches: int = 0,
    checkpoint_save_mode: str = "full",
) -> Tuple[List[float], List[Dict], float, Optional[List[float]]]:
    """
    Train the model for one epoch.
    """
    model.train()
    loss_list = []
    train_ious = [] if train_sam_iou else None
    val_metrics_list = []

    # Initialize running statistics for EMA smoothing
    running_stats = {
        "loss": 0.0,
        "dice": 0.0,
        "focal": 0.0,
        "aux_bce": 0.0,
        "adapter_delta_reg": 0.0,
        "adapter_delta_ratio": 0.0,
        "prompt_calibrator_reg": 0.0,
        "prompt_bias_supervision": 0.0,
        "dense_prompt_residual_reg": 0.0,
        "dense_prompt_signed_residual_loss": 0.0,
        "dense_prompt_unet_identity_loss": 0.0,
        "dense_prompt_unet_residual_loss": 0.0,
        "prompt_loss": 0.0,
        "gate_loss": 0.0,
        "dual_branch_gate_loss": 0.0,
        "dual_branch_residual_loss": 0.0,
        "dual_fusion_loss": 0.0,
        "final_logit_calibrator_supervision": 0.0,
        "final_logit_spatial_supervision": 0.0,
        "area_reg": 0.0,
        "iou_loss": 0.0,
        "sam_iou": 0.0,
        "count": 0,
    }
    
    # Add detection loss to running stats if using detection
    if authentic_ratio > 0.0:
        running_stats["detection_loss"] = 0.0
    
    if train_sam_iou:
        running_stats["actual_iou"] = 0.0

    stats_ema_alpha = 0.05
    pbar = tqdm.tqdm(loader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=False)
    scaler = torch.amp.GradScaler()

    for batch_idx, batch in enumerate(pbar):
        # Skip the first batch to avoid issues with incomplete batches
        if not model.training:
            raise RuntimeError("Model should be in training mode during train_epoch")
        
        # Skip empty batches (all samples were invalid)
        if not batch:
            logger.warning(f"Skipping empty batch at index {batch_idx}")
            continue

        # New streams format
        streams = [stream.to(device, non_blocking=True) for stream in batch["streams"]]

        orig = select_local_input(batch).to(device, non_blocking=True)
        global_image = batch.get("global_image", None)
        norm_coords = batch.get("norm_coords", None)
        if global_image is not None and norm_coords is not None:
            global_image = global_image.to(device, non_blocking=True)
            norm_coords = norm_coords.to(device, non_blocking=True)
        tgt = batch["mask"].to(device, non_blocking=True)
        
        # Check for NaN values in input data
        if torch.isnan(orig).any():
            logger.warning(f"Found NaN values in input image at batch {batch_idx}, replacing with zeros")
            orig = torch.nan_to_num(orig, nan=0.0, posinf=1.0, neginf=0.0)
        
        for i, stream in enumerate(streams):
            if torch.isnan(stream).any():
                logger.warning(f"Found NaN values in stream {i} at batch {batch_idx}, replacing with zeros")
                streams[i] = torch.nan_to_num(stream, nan=0.0, posinf=1.0, neginf=0.0)
        
        if torch.isnan(tgt).any():
            logger.warning(f"Found NaN values in target mask at batch {batch_idx}, replacing with zeros")
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0)
            
        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type=device.type):
            if global_image is not None and norm_coords is not None:
                logits, extras = model(
                    orig,
                    streams,
                    output_extras=True,
                    global_image=global_image,
                    norm_coords=norm_coords,
                )
            else:
                logits, extras = model(orig, streams, output_extras=True)
            # Make the current final prediction available as a detached oracle
            # source for prompt-composer gate/residual supervision.  The target
            # helpers detach prompt sources before deriving FN/FP targets, so
            # this does not create a second-order gradient path through logits.
            extras["final_logits"] = logits
            
            # # Check for NaN values in model outputs before computing loss
            # if torch.isnan(logits).any():
            #     logger.warning(f"Found NaN values in logits at batch {batch_idx}, replacing with zeros")
            #     logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # if 'coarse_mask' in extras and torch.isnan(extras['coarse_mask']).any():
            #     logger.warning(f"Found NaN values in coarse_mask at batch {batch_idx}, replacing with zeros")
            #     extras['coarse_mask'] = torch.nan_to_num(extras['coarse_mask'], nan=0.0, posinf=10.0, neginf=-10.0)
            
            # if 'iou_pred' in extras and torch.isnan(extras['iou_pred']).any():
            #     logger.warning(f"Found NaN values in iou_pred at batch {batch_idx}, replacing with ones")
            #     extras['iou_pred'] = torch.nan_to_num(extras['iou_pred'], nan=0, posinf=0, neginf=0.0)
            
            sample_types = batch.get("sample_type", None)
            is_authentic = batch.get("is_authentic", None)
            sam_iou, aux_bce_l, focal_loss_component, dice_l, actual_iou, iou_loss, detection_loss, loss = compute_loss(
                device, lambda_focal, focal_gamma, focal_alpha,
                lambda_iou, train_sam_iou, tgt, logits, extras,
                authentic_ratio, lambda_detection,
                is_authentic=is_authentic,
                sample_types=sample_types,
                coarse_loss_weight=coarse_loss_weight,
                coarse_dice_weight=coarse_dice_weight,
            )
            adapter_diagnostics = extras.get("adapter_diagnostics", [])
            adapter_delta_reg = adapter_delta_regularization(adapter_diagnostics).to(device=loss.device)
            if adapter_delta_reg_weight > 0.0:
                loss = loss + float(adapter_delta_reg_weight) * adapter_delta_reg

            prompt_calibrator_reg = compute_prompt_calibrator_regularization(
                extras.get("prompt_calibrator")
            ).to(device=loss.device)
            if coarse_prompt_calibrator_reg_weight > 0.0:
                loss = loss + float(coarse_prompt_calibrator_reg_weight) * prompt_calibrator_reg

            prompt_bias_l = compute_prompt_bias_supervision_loss(
                prompt_logits=(
                    extras.get("prompt_calibrator", {}).get("pre_calibrator_prompt")
                    if isinstance(extras.get("prompt_calibrator"), dict)
                    else None
                ),
                target=tgt,
                diagnostics=extras.get("prompt_calibrator"),
                base_scale=1.0,
                base_bias=0.0,
                max_delta_bias=prompt_bias_supervision_max_delta_bias,
                loss_type=prompt_bias_supervision_loss,
            ).to(device=loss.device)
            if prompt_bias_supervision_weight > 0.0:
                loss = loss + float(prompt_bias_supervision_weight) * prompt_bias_l

            dense_prompt_residual_reg = compute_dense_prompt_residual_regularization(extras).to(device=loss.device)
            if dense_prompt_residual_reg_weight > 0.0:
                loss = loss + float(dense_prompt_residual_reg_weight) * dense_prompt_residual_reg

            dense_prompt_signed_residual_l = compute_dense_prompt_signed_residual_supervision_loss(
                extras=extras,
                target=tgt,
                prompt_source=dense_prompt_signed_residual_target_source,
                target_scale=dense_prompt_signed_residual_target_scale,
                loss_type=dense_prompt_signed_residual_loss,
                target_mode=dense_prompt_signed_residual_target_mode,
                hard_threshold=dense_prompt_signed_residual_hard_threshold,
                use_gate=dense_prompt_signed_residual_use_gate,
                max_target_area=dense_prompt_signed_residual_max_gt_area,
            ).to(device=loss.device)
            if dense_prompt_signed_residual_supervision_weight > 0.0:
                loss = loss + (
                    float(dense_prompt_signed_residual_supervision_weight)
                    * dense_prompt_signed_residual_l
                )

            dense_prompt_unet_identity_l = compute_dense_prompt_unet_identity_loss(
                extras,
                loss_type=dense_prompt_unet_identity_loss,
            ).to(device=loss.device)
            if dense_prompt_unet_identity_weight > 0.0:
                loss = loss + float(dense_prompt_unet_identity_weight) * dense_prompt_unet_identity_l

            dense_prompt_unet_residual_l = compute_dense_prompt_unet_residual_supervision_loss(
                extras=extras,
                target=tgt,
                target_scale=dense_prompt_unet_residual_target_scale,
                loss_type=dense_prompt_unet_residual_loss,
                target_mode=dense_prompt_unet_residual_target_mode,
                hard_threshold=dense_prompt_unet_residual_hard_threshold,
                use_gate=dense_prompt_unet_residual_use_gate,
                max_target_area=dense_prompt_unet_residual_max_gt_area,
            ).to(device=loss.device)
            if dense_prompt_unet_residual_supervision_weight > 0.0:
                loss = loss + (
                    float(dense_prompt_unet_residual_supervision_weight)
                    * dense_prompt_unet_residual_l
                )

            prompt_supervision_l = compute_prompt_supervision_loss_from_extras(
                extras=extras,
                prompt_source=coarse_prompt_supervision_source,
                target=tgt,
                bce_weight=coarse_prompt_loss_weight,
                dice_weight=coarse_prompt_dice_weight,
                false_negative_weight=coarse_prompt_false_negative_weight,
                false_positive_weight=coarse_prompt_false_positive_weight,
                max_target_area=coarse_prompt_supervision_max_gt_area,
            ).to(device=loss.device)
            if (
                coarse_prompt_loss_weight > 0.0
                or coarse_prompt_dice_weight > 0.0
                or coarse_prompt_false_negative_weight > 0.0
                or coarse_prompt_false_positive_weight > 0.0
            ):
                loss = loss + prompt_supervision_l

            prompt_gate_l = compute_prompt_gate_supervision_loss(
                extras=extras,
                target=tgt,
                prompt_source=prompt_gate_target_source,
                target_mode=prompt_gate_target_mode,
                gate_sources=prompt_gate_sources,
                refiner_gate_max=prompt_gate_refiner_max,
                dense_gate_max=prompt_gate_dense_max,
                loss_type=prompt_gate_loss,
            ).to(device=loss.device)
            if prompt_gate_supervision_weight > 0.0:
                loss = loss + float(prompt_gate_supervision_weight) * prompt_gate_l

            dual_branch_gate_l = compute_dual_branch_prompt_gate_supervision_loss(
                extras=extras,
                target=tgt,
                prompt_source=dual_branch_prompt_gate_target_source,
                target_mode=dual_branch_prompt_gate_target_mode,
                hard_threshold=dual_branch_prompt_gate_hard_threshold,
                dense_gate_max=prompt_gate_dense_max,
                post_gate_max=prompt_gate_refiner_max,
                loss_type=dual_branch_prompt_gate_loss,
            ).to(device=loss.device)
            if dual_branch_prompt_gate_supervision_weight > 0.0:
                loss = loss + float(dual_branch_prompt_gate_supervision_weight) * dual_branch_gate_l

            dual_branch_residual_l = compute_dual_branch_prompt_residual_supervision_loss(
                extras=extras,
                target=tgt,
                prompt_source=dual_branch_prompt_residual_target_source,
                target_scale=dual_branch_prompt_residual_target_scale,
                loss_type=dual_branch_prompt_residual_loss,
                target_mode=dual_branch_prompt_residual_target_mode,
                hard_threshold=dual_branch_prompt_residual_hard_threshold,
            ).to(device=loss.device)
            if dual_branch_prompt_residual_supervision_weight > 0.0:
                loss = loss + float(dual_branch_prompt_residual_supervision_weight) * dual_branch_residual_l

            dual_fusion_l = compute_dual_prompt_fusion_supervision_loss(
                extras=extras,
                target=tgt,
                metric=dual_prompt_fusion_oracle_metric,
                loss_type=dual_prompt_fusion_loss,
            ).to(device=loss.device)
            if dual_prompt_fusion_supervision_weight > 0.0:
                loss = loss + float(dual_prompt_fusion_supervision_weight) * dual_fusion_l

            final_logit_calibrator_l = compute_final_logit_calibrator_supervision_loss(
                target=tgt,
                diagnostics=extras.get("final_logit_calibrator"),
                thresholds=final_logit_calibrator_oracle_thresholds,
                max_delta_bias=final_logit_calibrator_supervision_max_delta_bias,
                min_threshold=final_logit_calibrator_oracle_min_threshold,
                max_threshold=final_logit_calibrator_oracle_max_threshold,
                false_positive_penalty=final_logit_calibrator_oracle_false_positive_penalty,
                area_penalty=final_logit_calibrator_oracle_area_penalty,
                loss_type=final_logit_calibrator_supervision_loss,
            ).to(device=loss.device)
            if final_logit_calibrator_supervision_weight > 0.0:
                loss = loss + float(final_logit_calibrator_supervision_weight) * final_logit_calibrator_l

            final_logit_spatial_l = compute_final_logit_spatial_error_supervision_loss(
                target=tgt,
                diagnostics=extras.get("final_logit_calibrator"),
                target_scale=final_logit_spatial_supervision_target_scale,
                target_mode=final_logit_spatial_supervision_target_mode,
                hard_threshold=final_logit_spatial_supervision_hard_threshold,
                loss_type=final_logit_spatial_supervision_loss,
            ).to(device=loss.device)
            if final_logit_spatial_supervision_weight > 0.0:
                loss = loss + float(final_logit_spatial_supervision_weight) * final_logit_spatial_l

            area_reg_l = compute_area_ratio_regularization(
                final_logits=logits,
                coarse_logits=extras.get("coarse_mask"),
                dense_prompt_logits=extras.get("dense_prompt_mask"),
                coarse_prompt_logits=extras.get("coarse_prompt"),
                target=tgt,
                apply_to=area_reg_apply_to,
                target_source=area_reg_target_source,
                loss_type=area_reg_loss,
                constant_target=area_reg_constant,
                max_gt_area=area_reg_max_gt_area,
            ).to(device=loss.device)
            if area_reg_weight > 0.0:
                loss = loss + float(area_reg_weight) * area_reg_l
            
            # # Enhanced debug for batch 0
            # if batch_idx == 0:
            #     print(f"=== Batch 0 Loss Debug ===")
            #     print(f"Loss components:")
            #     print(f"  sam_iou: {sam_iou.mean().item():.6f}")
            #     print(f"  aux_bce_l: {aux_bce_l.item():.6f}")
            #     print(f"  focal_loss_component: {focal_loss_component.item():.6f}")
            #     print(f"  dice_l: {dice_l.item():.6f}")
            #     print(f"  iou_loss: {iou_loss.item():.6f}")
            #     print(f"  detection_loss: {detection_loss.item():.6f}")
            #     print(f"  total_loss: {loss.item():.6f}")
            #     print(f"  has_nan: {torch.isnan(loss).any().item()}, has_inf: {torch.isinf(loss).any().item()}")
                
            #     # Check for normalization layers in the model
            #     print(f"\n=== Checking Normalization Layers ===")
            #     def check_norm_layers(model, prefix=''):
            #         for name, module in model.named_children():
            #             full_name = f'{prefix}.{name}' if prefix else name
            #             # Use torch.nn explicitly to avoid NameError
            #             # if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            #             #     print(f"  Found {type(module).__name__} at {full_name}")
            #             # Recursively check submodules
            #             check_norm_layers(module, full_name)
            #     check_norm_layers(model)

        # Define gradient hook function for debugging
        # def grad_hook(name):
        #     def hook(grad):
        #         if torch.isnan(grad).any() or torch.isinf(grad).any():
        #             print(f"\n!!! Gradient Hook Triggered for {name} !!!")
        #             print(f"  Shape: {grad.shape}")
        #             print(f"  Max: {grad.max().item():.6f}, Min: {grad.min().item():.6f}, Mean: {grad.mean().item():.6f}")
        #             print(f"  Is NaN: {torch.isnan(grad).any().item()}, Count: {torch.isnan(grad).sum().item()}")
        #             print(f"  Is Inf: {torch.isinf(grad).any().item()}, Count: {torch.isinf(grad).sum().item()}")
        #             # Print the first few values to see the pattern
        #             print(f"  First 10 values: {grad.flatten()[:10]}")
        #     return hook
        
        # # Register hooks for layers that had NaN gradients in previous runs
        # hooks = []
        # if batch_idx == 0:
        #     print(f"\n=== Adding Gradient Hooks ===")
        #     # SharedAdapter layers
        #     if hasattr(model, 'adapters'):
        #         for i, layer in enumerate(model.adapters.fusion_mlps):
        #             if hasattr(layer, 'bias') and layer.bias is not None:
        #                 hooks.append(layer.bias.register_hook(grad_hook(f'adapters.fusion_mlps.{i}.bias')))
                
        #         for i, layer in enumerate(model.adapters.mlps_bottleneck):
        #             if hasattr(layer[0], 'bias') and layer[0].bias is not None:
        #                 hooks.append(layer[0].bias.register_hook(grad_hook(f'adapters.mlps_bottleneck.{i}.0.bias')))
                
        #         for i, layer in enumerate(model.adapters.mlp_up):
        #             if hasattr(layer, 'bias') and layer.bias is not None:
        #                 hooks.append(layer.bias.register_hook(grad_hook(f'adapters.mlp_up.{i}.bias')))
            
        #     # MaskAdapter layers
        #     if hasattr(model, 'mask_adapter'):
        #         if hasattr(model.mask_adapter.fine_processor, 'coarse_head'):
        #             if hasattr(model.mask_adapter.fine_processor.coarse_head, 'bias') and model.mask_adapter.fine_processor.coarse_head.bias is not None:
        #                 hooks.append(model.mask_adapter.fine_processor.coarse_head.bias.register_hook(grad_hook('mask_adapter.fine_processor.coarse_head.bias')))
            
        #     # Check FerretBackbone normalization layers
        #     if hasattr(model, 'ferret_backbone'):
        #         # print(f"\n=== Checking FerretBackbone Normalization Layers ===")
        #         # # Final check and fix for running_var values
        #         # print(f"  Performing final running_var fix before check...")
        #         # final_fixed = 0
        #         # for name, module in model.ferret_backbone.named_modules():
        #         #     if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'running_var'):
        #         #         if (module.running_var < 1e-6).any():
        #         #             module.running_var = torch.clamp(module.running_var, min=1e-6)
        #         #             final_fixed += 1
        #         # if final_fixed > 0:
        #         #     print(f"  Fixed {final_fixed} more running_var values during check")
                
        #         # Now perform the actual check
        #         for name, module in model.ferret_backbone.named_modules():
        #             if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        #                 print(f"  Found {type(module).__name__} at ferret_backbone.{name}")
        #                 # Register hooks for BN layers' running stats
        #                 if hasattr(module, 'running_mean'):
        #                     print(f"    running_mean: {module.running_mean[:5]}")
        #                 if hasattr(module, 'running_var'):
        #                     print(f"    running_var: {module.running_var[:5]}")
        #                     # Check if any running_var is too close to zero
        #                     if (module.running_var < 1e-6).any():
        #                         print(f"    WARNING: Some running_var values are too small!")
            
        #     print(f"Added {len(hooks)} gradient hooks")

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # # Remove hooks after backward pass
        # for hook in hooks:
        #     hook.remove()
        
        # Unscale gradients for manual gradient operations
        if batch_idx > 2:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # # Initial gradient check immediately after backward pass
        # initial_nan_inf = False
        # for param in model.parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             initial_nan_inf = True
        #             break
        # if initial_nan_inf:
        #     logger.warning(f"NaN/inf gradients detected immediately after backward pass at batch {batch_idx}")
        
        # # Gradient clipping for better stability
        # clip_grad_norm_(model.parameters(), max_norm=1.0)  # Further reduced max_norm for stability
        
        # # Enhanced gradient centralization with safety checks
        # for param in model.parameters():
        #     if param.grad is not None:
        #         # Save original gradient for debugging
        #         original_grad = param.grad.clone()
                
        #         # Calculate mean with safety checks
        #         grad_mean = param.grad.mean(dim=0, keepdim=True)
                
        #         # Check if mean is NaN or inf
        #         if torch.isnan(grad_mean).any() or torch.isinf(grad_mean).any():
        #             # Skip gradient centralization for this parameter if mean is invalid
        #             continue
                
        #         # Perform gradient centralization
        #         param.grad = param.grad - grad_mean
                
        #         # Check if centralization caused NaN/inf
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             # Revert to original gradient if centralization caused issues
        #             param.grad = original_grad
                
        #         # Final safety check and clip after centralization
        #         param.grad = torch.clamp(param.grad, min=-1.0, max=1.0)
        
        # Final gradient check before optimizer step
        # has_nan_grad = False
        # problematic_layers = []
        
        # # First pass: detect and record problematic layers
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             has_nan_grad = True
        #             # Get NaN/inf statistics
        #             nan_count = torch.isnan(param.grad).sum().item()
        #             inf_count = torch.isinf(param.grad).sum().item()
        #             total_count = param.grad.numel()
        #             problematic_layers.append({
        #                 'name': name,
        #                 'nan_count': nan_count,
        #                 'inf_count': inf_count,
        #                 'total_count': total_count,
        #                 'nan_ratio': nan_count / total_count,
        #                 'inf_ratio': inf_count / total_count
        #             })
        
        # # If NaN/inf gradients are detected, reset them to zero to prevent propagation
        # if has_nan_grad:
        #     # Reset problematic gradients
        #     for param in model.parameters():
        #         if param.grad is not None:
        #             param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
        #             param.grad = torch.where(torch.isinf(param.grad), torch.zeros_like(param.grad), param.grad)
            
        #     # Output detailed information about problematic layers
        #     logger.warning(f"NaN/inf gradients detected at batch {batch_idx}, resetting to zero")
        #     logger.warning(f"Found {len(problematic_layers)} layers with NaN/inf gradients:")
            
        #     # Sort by total problematic values ratio
        #     problematic_layers.sort(key=lambda x: (x['nan_ratio'] + x['inf_ratio']), reverse=True)
            
        #     # Limit output to first 5 problematic layers to reduce log noise
        #     for layer in problematic_layers[:5]:
        #         logger.warning(f"  Layer: {layer['name']}")
        #         logger.warning(f"    NaN: {layer['nan_count']}/{layer['total_count']} ({layer['nan_ratio']:.4f})")
        #         logger.warning(f"    Inf: {layer['inf_count']}/{layer['total_count']} ({layer['inf_ratio']:.4f})")
        #         logger.warning(f"    Total problematic: {layer['nan_count'] + layer['inf_count']}/{layer['total_count']} ({(layer['nan_ratio'] + layer['inf_ratio']):.4f})")
        #     if len(problematic_layers) > 5:
        #         logger.warning(f"  ... and {len(problematic_layers) - 5} more layers with NaN/inf gradients")
        
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        optimizer.zero_grad()

        # Update learning rate scheduler
        if scheduler is not None and not skip_lr_sched:
            scheduler.step()

        # Update EMA
        if ema is not None:
            ema.update()

        # Update running statistics and progress bar
        with torch.no_grad():
            # Extract scalar values once and reuse
            loss_val = loss.item()
            focal_val = focal_loss_component.item()
            dice_val = dice_l.item()
            aux_bce_val = aux_bce_l.item()
            adapter_delta_reg_val = adapter_delta_reg.item() if "adapter_delta_reg" in locals() else 0.0
            prompt_calibrator_reg_val = (
                prompt_calibrator_reg.item() if "prompt_calibrator_reg" in locals() else 0.0
            )
            prompt_bias_val = (
                prompt_bias_l.item() if "prompt_bias_l" in locals() else 0.0
            )
            dense_prompt_residual_reg_val = (
                dense_prompt_residual_reg.item() if "dense_prompt_residual_reg" in locals() else 0.0
            )
            dense_prompt_signed_residual_val = (
                dense_prompt_signed_residual_l.item()
                if "dense_prompt_signed_residual_l" in locals()
                else 0.0
            )
            dense_prompt_unet_identity_val = (
                dense_prompt_unet_identity_l.item()
                if "dense_prompt_unet_identity_l" in locals()
                else 0.0
            )
            dense_prompt_unet_residual_val = (
                dense_prompt_unet_residual_l.item()
                if "dense_prompt_unet_residual_l" in locals()
                else 0.0
            )
            prompt_supervision_val = (
                prompt_supervision_l.item() if "prompt_supervision_l" in locals() else 0.0
            )
            prompt_gate_val = (
                prompt_gate_l.item() if "prompt_gate_l" in locals() else 0.0
            )
            dual_branch_gate_val = (
                dual_branch_gate_l.item() if "dual_branch_gate_l" in locals() else 0.0
            )
            dual_branch_residual_val = (
                dual_branch_residual_l.item() if "dual_branch_residual_l" in locals() else 0.0
            )
            dual_fusion_val = (
                dual_fusion_l.item() if "dual_fusion_l" in locals() else 0.0
            )
            final_logit_calibrator_val = (
                final_logit_calibrator_l.item() if "final_logit_calibrator_l" in locals() else 0.0
            )
            final_logit_spatial_val = (
                final_logit_spatial_l.item() if "final_logit_spatial_l" in locals() else 0.0
            )
            area_reg_val = area_reg_l.item() if "area_reg_l" in locals() else 0.0
            iou_loss_val = iou_loss.item()
            sam_iou_val = sam_iou.mean().item()
            detection_loss_val = detection_loss.item() if authentic_ratio > 0.0 else 0.0
            
            lr = optimizer.param_groups[0]["lr"]

            # Calculate gradient norm for debugging (only for first few batches)
            grad_norm = 0.0
            if batch_idx < 10:
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

            # Current stats for this batch
            current_stats = {
                "loss": loss_val,
                "focal": lambda_focal * focal_val,
                "dice": dice_val,
                "aux_bce": coarse_loss_weight * aux_bce_val,
                "adapter_delta_reg": adapter_delta_reg_weight * adapter_delta_reg_val,
                "prompt_calibrator_reg": (
                    coarse_prompt_calibrator_reg_weight * prompt_calibrator_reg_val
                ),
                "prompt_bias_supervision": prompt_bias_supervision_weight * prompt_bias_val,
                "dense_prompt_residual_reg": (
                    dense_prompt_residual_reg_weight * dense_prompt_residual_reg_val
                ),
                "dense_prompt_signed_residual_loss": (
                    dense_prompt_signed_residual_supervision_weight
                    * dense_prompt_signed_residual_val
                ),
                "dense_prompt_unet_identity_loss": (
                    dense_prompt_unet_identity_weight * dense_prompt_unet_identity_val
                ),
                "dense_prompt_unet_residual_loss": (
                    dense_prompt_unet_residual_supervision_weight * dense_prompt_unet_residual_val
                ),
                "prompt_loss": prompt_supervision_val,
                "gate_loss": prompt_gate_supervision_weight * prompt_gate_val,
                "dual_branch_gate_loss": (
                    dual_branch_prompt_gate_supervision_weight * dual_branch_gate_val
                ),
                "dual_branch_residual_loss": (
                    dual_branch_prompt_residual_supervision_weight * dual_branch_residual_val
                ),
                "dual_fusion_loss": dual_prompt_fusion_supervision_weight * dual_fusion_val,
                "final_logit_calibrator_supervision": (
                    final_logit_calibrator_supervision_weight * final_logit_calibrator_val
                ),
                "final_logit_spatial_supervision": (
                    final_logit_spatial_supervision_weight * final_logit_spatial_val
                ),
                "area_reg": area_reg_weight * area_reg_val,
                "iou_loss": lambda_iou * iou_loss_val,
                "sam_iou": sam_iou_val,
            }
            if adapter_diagnostics:
                adapter_summary = summarize_adapter_diagnostics(adapter_diagnostics)
                current_stats["adapter_delta_ratio"] = adapter_summary.get("adapter_delta_ratio_mean", 0.0)
            
            if authentic_ratio > 0.0:
                current_stats["detection_loss"] = lambda_detection * detection_loss_val
            
            if train_sam_iou and actual_iou is not None:
                actual_iou_val = actual_iou.mean().item()
                current_stats["actual_iou"] = actual_iou_val

            # Update running statistics with EMA
            if running_stats["count"] == 0:
                for k, v in current_stats.items():
                    running_stats[k] = v
            else:
                for k, v in current_stats.items():
                    running_stats[k] = (1 - stats_ema_alpha) * running_stats[k] + stats_ema_alpha * v

            running_stats["count"] += 1
            loss_list.append(loss_val)
            
            if train_sam_iou and actual_iou is not None:
                train_ious.append(actual_iou_val)

            # Print detailed debug information every 20 batches and first 5 batches
            if batch_idx % 20 == 0 or batch_idx < 5:
                debug_info = f"Epoch {epoch} Batch {batch_idx}/{len(loader)} | Loss: {loss_val:.6f} | "
                debug_info += f"Focal: {focal_val:.6f} (scaled: {lambda_focal * focal_val:.6f}) | "
                debug_info += f"Dice: {dice_val:.6f} | Aux BCE: {aux_bce_val:.6f} | "
                if adapter_delta_reg_weight > 0.0 or adapter_diagnostics:
                    debug_info += f"AdapterReg: {adapter_delta_reg_val:.6f} | "
                if coarse_prompt_calibrator_reg_weight > 0.0:
                    debug_info += (
                        f"PromptCalReg: {prompt_calibrator_reg_val:.6f} "
                        f"(scaled: {coarse_prompt_calibrator_reg_weight * prompt_calibrator_reg_val:.6f}) | "
                    )
                if prompt_bias_supervision_weight > 0.0:
                    debug_info += (
                        f"PromptBiasLoss: {prompt_bias_val:.6f} "
                        f"(scaled: {prompt_bias_supervision_weight * prompt_bias_val:.6f}) | "
                    )
                if dense_prompt_residual_reg_weight > 0.0:
                    debug_info += (
                        f"DenseResReg: {dense_prompt_residual_reg_val:.6f} "
                        f"(scaled: {dense_prompt_residual_reg_weight * dense_prompt_residual_reg_val:.6f}) | "
                    )
                if dense_prompt_signed_residual_supervision_weight > 0.0:
                    debug_info += (
                        f"DenseSignedResLoss: {dense_prompt_signed_residual_val:.6f} "
                        f"(scaled: {dense_prompt_signed_residual_supervision_weight * dense_prompt_signed_residual_val:.6f}) | "
                    )
                if dense_prompt_unet_identity_weight > 0.0:
                    debug_info += (
                        f"DenseUnetIdentityLoss: {dense_prompt_unet_identity_val:.6f} "
                        f"(scaled: {dense_prompt_unet_identity_weight * dense_prompt_unet_identity_val:.6f}) | "
                    )
                if dense_prompt_unet_residual_supervision_weight > 0.0:
                    debug_info += (
                        f"DenseUnetResidualLoss: {dense_prompt_unet_residual_val:.6f} "
                        f"(scaled: {dense_prompt_unet_residual_supervision_weight * dense_prompt_unet_residual_val:.6f}) | "
                    )
                if coarse_prompt_loss_weight > 0.0 or coarse_prompt_dice_weight > 0.0:
                    debug_info += f"PromptLoss: {prompt_supervision_val:.6f} | "
                if prompt_gate_supervision_weight > 0.0:
                    debug_info += (
                        f"GateLoss: {prompt_gate_val:.6f} "
                        f"(scaled: {prompt_gate_supervision_weight * prompt_gate_val:.6f}) | "
                    )
                if dual_branch_prompt_gate_supervision_weight > 0.0:
                    debug_info += (
                        f"DualBranchGateLoss: {dual_branch_gate_val:.6f} "
                        f"(scaled: {dual_branch_prompt_gate_supervision_weight * dual_branch_gate_val:.6f}) | "
                    )
                if dual_branch_prompt_residual_supervision_weight > 0.0:
                    debug_info += (
                        f"DualBranchResidualLoss: {dual_branch_residual_val:.6f} "
                        f"(scaled: {dual_branch_prompt_residual_supervision_weight * dual_branch_residual_val:.6f}) | "
                    )
                if dual_prompt_fusion_supervision_weight > 0.0:
                    debug_info += (
                        f"DualFusionLoss: {dual_fusion_val:.6f} "
                        f"(scaled: {dual_prompt_fusion_supervision_weight * dual_fusion_val:.6f}) | "
                    )
                if final_logit_calibrator_supervision_weight > 0.0:
                    debug_info += (
                        f"FinalCalLoss: {final_logit_calibrator_val:.6f} "
                        f"(scaled: {final_logit_calibrator_supervision_weight * final_logit_calibrator_val:.6f}) | "
                    )
                if final_logit_spatial_supervision_weight > 0.0:
                    debug_info += (
                        f"FinalSpatialLoss: {final_logit_spatial_val:.6f} "
                        f"(scaled: {final_logit_spatial_supervision_weight * final_logit_spatial_val:.6f}) | "
                    )
                if area_reg_weight > 0.0:
                    debug_info += f"AreaReg: {area_reg_val:.6f} (scaled: {area_reg_weight * area_reg_val:.6f}) | "
                debug_info += f"IoU Loss: {iou_loss_val:.6f} (scaled: {lambda_iou * iou_loss_val:.6f}) | "
                if authentic_ratio > 0.0:
                    debug_info += f"Det Loss: {detection_loss_val:.6f} (scaled: {lambda_detection * detection_loss_val:.6f}) | "
                debug_info += f"LR: {lr:.8f} | "
                if batch_idx < 10:
                    debug_info += f"Grad Norm: {grad_norm:.6f}"
                logger.debug(debug_info)

            # Update progress bar display
            if (batch_idx % 5) == 0 or batch_idx == len(loader) - 1:
                postfix_dict = {
                    "loss": f"{running_stats['loss']:.4f}",
                    "focal": f"{running_stats['focal']:.4f}",
                    "dice": f"{running_stats['dice']:.4f}",
                    "aux": f"{running_stats['aux_bce']:.4f}",
                    "area": f"{running_stats['area_reg']:.4f}",
                    "pcal": f"{running_stats['prompt_calibrator_reg']:.4f}",
                    "pbias": f"{running_stats['prompt_bias_supervision']:.4f}",
                    "dres": f"{running_stats['dense_prompt_residual_reg']:.4f}",
                    "dsres": f"{running_stats['dense_prompt_signed_residual_loss']:.4f}",
                    "uid": f"{running_stats['dense_prompt_unet_identity_loss']:.4f}",
                    "ures": f"{running_stats['dense_prompt_unet_residual_loss']:.4f}",
                    "ploss": f"{running_stats['prompt_loss']:.4f}",
                    "gloss": f"{running_stats['gate_loss']:.4f}",
                    "dbgloss": f"{running_stats['dual_branch_gate_loss']:.4f}",
                    "dbrloss": f"{running_stats['dual_branch_residual_loss']:.4f}",
                    "dfuse": f"{running_stats['dual_fusion_loss']:.4f}",
                    "fcal": f"{running_stats['final_logit_calibrator_supervision']:.4f}",
                    "fsp": f"{running_stats['final_logit_spatial_supervision']:.4f}",
                    "adp": f"{running_stats['adapter_delta_ratio']:.3f}",
                    "lr": f"{lr:.6f}",
                }
                
                if authentic_ratio > 0.0:
                    postfix_dict["det"] = f"{running_stats['detection_loss']:.4f}"
                
                if train_sam_iou:
                    postfix_dict.update({
                        "iou_loss": f"{running_stats['iou_loss']:.4f}",
                        "sam_iou": f"{running_stats['sam_iou']:.4f}",
                        "actual_iou": f"{running_stats['actual_iou']:.4f}",
                    })
                
                pbar.set_postfix(postfix_dict)

            # Validation at specified intervals
            if (
                val_steps != 0
                and val_loader is not None
                and validate_fn is not None
                and (batch_idx + 1) % val_steps == 0
            ):
                logger.info(
                    "Performing validation at batch %s/%s",
                    batch_idx + 1,
                    len(loader),
                )

                # Apply EMA weights for validation
                if ema is not None:
                    ema.apply_shadow()
                    if device.type == "cuda":
                        torch.clear_autocast_cache()

                val_metrics = validate_fn(
                    model, val_loader, device, 
                    save_vis_path=os.path.join(
                        os.path.dirname(model_save_path), "vis", 
                        f"epoch{epoch:03d}_step{batch_idx:06d}"
                    ), 
                    contrastive_blur=contrastive_blur,
                    use_tiling=False,
                )
                
                logger.info(
                    "Validation metrics | F1=%.4f IoU=%.4f",
                    val_metrics['f1'],
                    val_metrics['iou'],
                )
                val_metrics_list.append(val_metrics)

                # Save model if it's the best so far
                if val_metrics["f1"] + val_metrics["iou"] > best_score:
                    best_score = val_metrics["f1"] + val_metrics["iou"]
                    save_mode = str(checkpoint_save_mode).lower()
                    if save_mode == "none":
                        logger.info(
                            "Checkpoint save skipped by checkpoint_save_mode=none | best_score=%.4f",
                            best_score,
                        )
                    else:
                        # Prepare checkpoint data.  Clone tensors while EMA
                        # weights are still applied so restore() cannot mutate the
                        # in-memory payload before torch.save finishes or before
                        # optional reload diagnostics inspect it.
                        checkpoint_data = build_checkpoint_payload(
                            model=model,
                            epoch=epoch,
                            batch=batch_idx,
                            score=best_score,
                            ema_state=ema.state_dict() if ema is not None else None,
                            checkpoint_save_mode=save_mode,
                        )

                        # Save best model
                        torch.save(checkpoint_data, model_save_path)

                        # Save checkpoint with timestamp
                        checkpoint_path = os.path.join(
                            os.path.dirname(model_save_path),
                            "checkpoints",
                            f"model_epoch{epoch}_batch{batch_idx}_score{best_score:.4f}.pth",
                        )
                        torch.save(checkpoint_data, checkpoint_path)
                        
                        # Save parameters alongside checkpoint
                        if model_params is not None:
                            checkpoint_params_path = checkpoint_path.replace('.pth', '_params.json')
                            with open(checkpoint_params_path, 'w') as f:
                                json.dump(model_params, f, indent=2)

                        logger.info(
                            "Model checkpoint updated | path=%s best_score=%.4f mode=%s",
                            model_save_path,
                            best_score,
                            save_mode,
                        )
                        logger.debug("Checkpoint saved to %s", checkpoint_path)

                        if (
                            reload_diagnostics_batches > 0
                            and val_loader is not None
                            and model_params is not None
                        ):
                            diagnostic_path = os.path.join(
                                os.path.dirname(model_save_path),
                                "reload_diagnostics",
                                f"epoch{epoch:03d}_batch{batch_idx:06d}.json",
                            )
                            try:
                                run_live_reload_diagnostic(
                                    live_model=model,
                                    checkpoint_path=checkpoint_path,
                                    model_params=model_params,
                                    val_loader=val_loader,
                                    device=device,
                                    output_path=diagnostic_path,
                                    num_batches=reload_diagnostics_batches,
                                )
                            except Exception:
                                logger.exception("Live-vs-reload diagnostic failed")

                # Restore original weights after validation
                if ema is not None:
                    ema.restore()
                    if device.type == "cuda":
                        torch.clear_autocast_cache()

                model.train()

    # Print epoch summary
    summary = f"\nEpoch {epoch}/{total_epochs} completed. Avg Loss: {np.mean(loss_list):.4f}"
    if train_sam_iou and train_ious:
        summary += f" | Avg Train IoU: {np.mean(train_ious):.4f}"
    logger.info(summary)

    return loss_list, val_metrics_list, best_score, train_ious


def _area_target_from_mask(
    target: torch.Tensor,
    *,
    target_source: str,
    constant_target: float,
) -> torch.Tensor:
    """Return per-sample target area ratios for area calibration."""
    gt_area = (target.detach().float() > 0.5).float().mean(dim=tuple(range(1, target.ndim)))
    if target_source == "gt":
        return gt_area
    if target_source == "batch_gt":
        return gt_area.mean().expand_as(gt_area)
    if target_source == "constant":
        return torch.full_like(gt_area, float(constant_target))
    raise ValueError(f"Unsupported area_reg_target_source: {target_source}")


def compute_area_ratio_regularization(
    *,
    final_logits: torch.Tensor,
    coarse_logits: Optional[torch.Tensor],
    dense_prompt_logits: Optional[torch.Tensor] = None,
    coarse_prompt_logits: Optional[torch.Tensor] = None,
    target: torch.Tensor,
    apply_to: str = "coarse",
    target_source: str = "gt",
    loss_type: str = "smooth_l1",
    constant_target: float = 0.25,
    max_gt_area: Optional[float] = None,
) -> torch.Tensor:
    """Penalize prediction area ratio mismatch against GT/batch/constant area.

    This is intentionally lightweight: it does not replace segmentation losses,
    but counteracts the SAM3+LAD failure mode where the coarse dense prompt and
    final mask converge to almost all-foreground on small-mask samples.
    """
    valid_apply_to = {
        "coarse",
        "final",
        "both",
        "dense_prompt",
        "coarse_and_dense",
        "final_and_dense",
        "coarse_prompt",
        "final_and_prompt",
        "dense_and_prompt",
        "all",
    }
    if apply_to not in valid_apply_to:
        raise ValueError(f"Unsupported area_reg_apply_to: {apply_to}")
    target_area = _area_target_from_mask(
        target,
        target_source=target_source,
        constant_target=constant_target,
    )
    if max_gt_area is not None:
        gt_area = _area_target_from_mask(
            target,
            target_source="gt",
            constant_target=constant_target,
        )
        sample_mask = gt_area <= float(max_gt_area)
        if not bool(sample_mask.any()):
            return final_logits.new_tensor(0.0)
        target_area = target_area[sample_mask]
    else:
        sample_mask = None

    losses = []
    if apply_to in {"final", "both", "final_and_dense", "final_and_prompt", "all"}:
        final_area = torch.sigmoid(final_logits.float()).mean(dim=tuple(range(1, final_logits.ndim)))
        if sample_mask is not None:
            final_area = final_area[sample_mask]
        losses.append(_area_loss(final_area, target_area, loss_type=loss_type))
    if apply_to in {"coarse", "both", "coarse_and_dense", "all"}:
        if coarse_logits is None:
            raise ValueError("coarse area regularization requested but coarse_logits is None")
        coarse_area = torch.sigmoid(coarse_logits.float()).mean(dim=tuple(range(1, coarse_logits.ndim)))
        if sample_mask is not None:
            coarse_area = coarse_area[sample_mask]
        losses.append(_area_loss(coarse_area, target_area, loss_type=loss_type))
    if apply_to in {"dense_prompt", "coarse_and_dense", "final_and_dense", "dense_and_prompt", "all"}:
        if dense_prompt_logits is None:
            raise ValueError("dense-prompt area regularization requested but dense_prompt_logits is None")
        dense_area = torch.sigmoid(dense_prompt_logits.float()).mean(
            dim=tuple(range(1, dense_prompt_logits.ndim))
        )
        if sample_mask is not None:
            dense_area = dense_area[sample_mask]
        losses.append(_area_loss(dense_area, target_area, loss_type=loss_type))
    if apply_to in {"coarse_prompt", "final_and_prompt", "dense_and_prompt", "all"}:
        if coarse_prompt_logits is None:
            raise ValueError("coarse-prompt area regularization requested but coarse_prompt_logits is None")
        prompt_area = torch.sigmoid(coarse_prompt_logits.float()).mean(
            dim=tuple(range(1, coarse_prompt_logits.ndim))
        )
        if sample_mask is not None:
            prompt_area = prompt_area[sample_mask]
        losses.append(_area_loss(prompt_area, target_area, loss_type=loss_type))
    return torch.stack(losses).mean() if losses else final_logits.new_tensor(0.0)


def _area_loss(pred_area: torch.Tensor, target_area: torch.Tensor, *, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred_area, target_area)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred_area, target_area)
    raise ValueError(f"Unsupported area_reg_loss: {loss_type}")


def compute_prompt_calibrator_regularization(
    diagnostics: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Keep dynamic prompt calibration near identity unless evidence warrants drift."""
    if not diagnostics:
        return torch.tensor(0.0)
    scale = diagnostics.get("scale")
    bias = diagnostics.get("bias")
    losses: List[torch.Tensor] = []
    if scale is not None:
        losses.append(torch.log(scale.float().clamp_min(1e-6)).pow(2).mean())
    if bias is not None:
        losses.append(bias.float().pow(2).mean())
    residual = diagnostics.get("refiner_residual")
    if residual is not None:
        losses.append(residual.float().pow(2).mean())
    if not losses:
        ref = next(iter(diagnostics.values()))
        return ref.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_prompt_bias_oracle_target(
    prompt_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    base_scale: float = 1.0,
    base_bias: float = 0.0,
    max_delta_bias: Optional[float] = None,
) -> torch.Tensor:
    """Return a per-sample scalar bias target that matches GT foreground area.

    The oracle is used only during training.  For each sample it chooses the
    prompt threshold whose foreground area matches the GT area, then converts
    that threshold into the additive bias needed after ``base_scale`` and
    ``base_bias``.  This directly supervises a sample-conditioned prompt bias
    policy without leaking GT into inference.
    """
    prompt = prompt_logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if target_f.shape[-2:] != prompt.shape[-2:]:
        target_f = F.interpolate(target_f, size=prompt.shape[-2:], mode="nearest")

    batch_size = int(prompt.shape[0])
    flat_prompt = prompt.flatten(1)
    flat_target = target_f.flatten(1)
    num_pixels = int(flat_prompt.shape[1])
    gt_area = flat_target.mean(dim=1).clamp(1.0 / max(num_pixels, 1), 1.0)
    kth = torch.ceil(gt_area * float(num_pixels)).long().clamp(1, num_pixels)
    sorted_prompt = torch.sort(flat_prompt, dim=1, descending=True).values
    threshold = sorted_prompt[torch.arange(batch_size, device=prompt.device), kth - 1]
    desired_bias = -(float(base_scale) * threshold + float(base_bias))
    if max_delta_bias is not None:
        desired_bias = desired_bias.clamp(
            min=-float(max_delta_bias),
            max=float(max_delta_bias),
        )
    return desired_bias.view(batch_size, 1, 1, 1)


def compute_prompt_bias_supervision_loss(
    *,
    prompt_logits: Optional[torch.Tensor],
    target: torch.Tensor,
    diagnostics: Optional[Dict[str, torch.Tensor]],
    base_scale: float = 1.0,
    base_bias: float = 0.0,
    max_delta_bias: Optional[float] = None,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Supervise dynamic prompt-calibrator bias with a GT-derived oracle."""
    if prompt_logits is None or not diagnostics:
        return target.new_tensor(0.0)
    bias = diagnostics.get("bias")
    if bias is None:
        return target.new_tensor(0.0)
    oracle = compute_prompt_bias_oracle_target(
        prompt_logits,
        target,
        base_scale=base_scale,
        base_bias=base_bias,
        max_delta_bias=max_delta_bias,
    ).to(device=bias.device, dtype=bias.float().dtype)
    bias_f = bias.float()
    if oracle.shape != bias_f.shape:
        oracle = oracle.expand_as(bias_f)
    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(bias_f, oracle)
    if loss_type == "l1":
        return F.l1_loss(bias_f, oracle)
    if loss_type == "mse":
        return F.mse_loss(bias_f, oracle)
    raise ValueError(f"Unsupported prompt_bias_supervision_loss: {loss_type}")


def _default_final_calibrator_thresholds(device: torch.device) -> torch.Tensor:
    return torch.arange(0.10, 0.91, 0.05, device=device, dtype=torch.float32)


def compute_final_logit_bias_oracle_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: Optional[List[float]] = None,
    max_delta_bias: Optional[float] = None,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    false_positive_penalty: float = 0.0,
    area_penalty: float = 0.0,
) -> torch.Tensor:
    """Return the per-sample bias that emulates the best hard threshold.

    The oracle is used only for training the calibrator.  It searches a small
    probability-threshold grid and chooses the threshold that maximizes
    IoU+F1 for each sample.  An additive logit bias of ``-logit(threshold)``
    then makes the standard 0.5 inference threshold equivalent to the selected
    raw probability threshold.
    """
    logits_f = logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if logits_f.shape[-2:] != target_f.shape[-2:]:
        logits_f = F.interpolate(
            logits_f,
            size=target_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    probs = torch.sigmoid(logits_f)
    dims = tuple(range(1, probs.ndim))
    thresholds_t = (
        torch.tensor(thresholds, device=probs.device, dtype=probs.dtype)
        if thresholds is not None
        else _default_final_calibrator_thresholds(probs.device).to(dtype=probs.dtype)
    ).clamp(1e-4, 1.0 - 1e-4)
    if min_threshold is not None:
        thresholds_t = thresholds_t[thresholds_t >= float(min_threshold)]
    if max_threshold is not None:
        thresholds_t = thresholds_t[thresholds_t <= float(max_threshold)]
    if thresholds_t.numel() == 0:
        raise ValueError(
            "No final-logit oracle thresholds remain after applying "
            f"min_threshold={min_threshold} max_threshold={max_threshold}"
        )
    best_score = probs.new_full((probs.shape[0],), -1.0)
    best_threshold = probs.new_full((probs.shape[0],), 0.5)
    total_area = float(torch.ones_like(target_f).flatten(1).shape[1])
    target_area = target_f.sum(dim=dims) / (total_area + 1e-7)
    for threshold in thresholds_t:
        pred = (probs >= threshold).float()
        intersection = (pred * target_f).sum(dim=dims)
        pred_sum = pred.sum(dim=dims)
        target_sum = target_f.sum(dim=dims)
        union = ((pred + target_f) > 0.0).float().sum(dim=dims)
        iou = intersection / (union + 1e-7)
        f1 = 2.0 * intersection / (pred_sum + target_sum + 1e-7)
        pred_area = pred_sum / (total_area + 1e-7)
        fp_area = ((pred * (1.0 - target_f)).sum(dim=dims)) / (total_area + 1e-7)
        score = (
            iou
            + f1
            - float(false_positive_penalty) * fp_area
            - float(area_penalty) * (pred_area - target_area).abs()
        )
        update = score > best_score
        best_score = torch.where(update, score, best_score)
        best_threshold = torch.where(update, threshold.expand_as(best_threshold), best_threshold)
    desired_bias = -torch.logit(best_threshold).view(-1, 1, 1, 1)
    if max_delta_bias is not None:
        desired_bias = desired_bias.clamp(
            min=-float(max_delta_bias),
            max=float(max_delta_bias),
        )
    return desired_bias


def compute_final_logit_calibrator_supervision_loss(
    *,
    target: torch.Tensor,
    diagnostics: Optional[Dict[str, torch.Tensor]],
    thresholds: Optional[List[float]] = None,
    max_delta_bias: Optional[float] = None,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    false_positive_penalty: float = 0.0,
    area_penalty: float = 0.0,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Supervise final-logit calibrator bias with a threshold oracle."""
    if not diagnostics:
        return target.new_tensor(0.0)
    logits = diagnostics.get("pre_calibrator_logits")
    bias = diagnostics.get("bias")
    if logits is None or bias is None:
        return target.new_tensor(0.0)
    oracle = compute_final_logit_bias_oracle_target(
        logits,
        target,
        thresholds=thresholds,
        max_delta_bias=max_delta_bias,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        false_positive_penalty=false_positive_penalty,
        area_penalty=area_penalty,
    ).to(device=bias.device, dtype=bias.float().dtype)
    bias_f = bias.float()
    if oracle.shape != bias_f.shape:
        oracle = oracle.expand_as(bias_f)
    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(bias_f, oracle)
    if loss_type == "l1":
        return F.l1_loss(bias_f, oracle)
    if loss_type == "mse":
        return F.mse_loss(bias_f, oracle)
    raise ValueError(f"Unsupported final_logit_calibrator_supervision_loss: {loss_type}")


def compute_final_logit_spatial_error_supervision_loss(
    *,
    target: torch.Tensor,
    diagnostics: Optional[Dict[str, torch.Tensor]],
    target_scale: float = 1.0,
    target_mode: str = "soft_error",
    hard_threshold: float = 0.5,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Teach a spatial final-logit calibrator where to add/suppress logits.

    Scalar threshold supervision is useful for sample-level calibration but it
    cannot teach local shape corrections.  For spatial calibrators, supervise
    the per-pixel logit delta directly: positive on false-negative regions and
    negative on false-positive regions, with zero target on already-correct
    pixels.
    """
    if not diagnostics:
        return target.new_tensor(0.0)
    logits = diagnostics.get("pre_calibrator_logits")
    delta = diagnostics.get("delta_bias")
    if logits is None or delta is None:
        return target.new_tensor(0.0)
    if delta.ndim != 4 or int(delta.shape[-2]) * int(delta.shape[-1]) <= 1:
        return target.new_tensor(0.0)

    logits_f = logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if logits_f.shape[-2:] != target_f.shape[-2:]:
        logits_f = F.interpolate(logits_f, size=target_f.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits_f)
    mode = str(target_mode).lower()
    if mode in {"soft", "soft_error"}:
        correction_target = target_f * (1.0 - prob) - (1.0 - target_f) * prob
    elif mode in {"hard", "hard_error"}:
        threshold = float(hard_threshold)
        correction_target = (
            target_f * torch.relu(threshold - prob)
            - (1.0 - target_f) * torch.relu(prob - threshold)
        )
    else:
        raise ValueError(f"Unsupported final_logit_spatial_supervision_target_mode: {target_mode}")

    if correction_target.shape[-2:] != delta.shape[-2:]:
        correction_target = F.interpolate(
            correction_target,
            size=delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    correction_target = correction_target.to(device=delta.device, dtype=delta.float().dtype)
    correction_target = correction_target * float(target_scale)
    delta_f = delta.float()
    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(delta_f, correction_target)
    if loss_type == "l1":
        return F.l1_loss(delta_f, correction_target)
    if loss_type == "mse":
        return F.mse_loss(delta_f, correction_target)
    raise ValueError(f"Unsupported final_logit_spatial_supervision_loss: {loss_type}")


def compute_dense_prompt_residual_regularization(
    extras: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize drift between supervised coarse logits and SAM dense-prompt logits.

    In ``split_multiscale`` mode the dense prompt is allowed to learn
    SAM-specific residual semantics, but it should not immediately run far away
    from the stable coarse head.  When both maps come from the same base logits,
    ``dense - coarse`` isolates the residual path and keeps the regularizer from
    fighting the supervised mask compressor.
    """
    coarse = extras.get("coarse_mask") if extras is not None else None
    dense = extras.get("dense_prompt_mask") if extras is not None else None
    if coarse is None and dense is None:
        return torch.tensor(0.0)
    ref = dense if dense is not None else coarse
    if coarse is None or dense is None:
        return ref.new_tensor(0.0)
    if coarse.shape[-2:] != dense.shape[-2:]:
        coarse = F.interpolate(
            coarse,
            size=dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return (dense.float() - coarse.float()).pow(2).mean()


def compute_dense_prompt_signed_residual_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prompt_source: str = "coarse_mask",
    target_scale: float = 1.0,
    loss_type: str = "smooth_l1",
    target_mode: str = "hard_error",
    hard_threshold: float = 0.5,
    use_gate: bool = False,
    max_target_area: Optional[float] = None,
) -> torch.Tensor:
    """Supervise a direct signed dense-prompt residual with local FP/FN errors.

    The direct signed high-resolution head predicts a positive/negative logit
    delta for the SAM dense prompt.  This helper derives a detached oracle from
    a prompt source: false negatives should receive positive deltas; false
    positives should receive negative deltas.  With ``use_gate=True`` the loss
    supervises the actually applied correction ``delta * gate``; otherwise it
    supervises the bounded signed delta itself so the residual branch can learn
    a useful correction field even while the gate is still small.
    """
    if extras is None:
        return target.new_tensor(0.0)
    signed_delta = extras.get("dense_prompt_signed_delta")
    if signed_delta is None:
        return target.new_tensor(0.0)
    prompt_logits = _prompt_source_from_extras(extras, prompt_source)
    if prompt_logits is None:
        return target.new_tensor(0.0)

    prediction = signed_delta.float()
    if use_gate:
        signed_gate = extras.get("dense_prompt_signed_gate")
        if signed_gate is not None:
            gate = signed_gate.float()
            if gate.shape[-2:] != prediction.shape[-2:]:
                gate = F.interpolate(
                    gate,
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = prediction * gate

    prompt = prompt_logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if prompt.shape[-2:] != target_f.shape[-2:]:
        prompt = F.interpolate(
            prompt,
            size=target_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if max_target_area is not None:
        target_area = target_f.mean(dim=tuple(range(1, target_f.ndim)))
        selected = target_area <= float(max_target_area)
        if not bool(selected.any().item()):
            return target.new_tensor(0.0)
        prompt = prompt[selected]
        target_f = target_f[selected]
        prediction = prediction[selected]

    prob = torch.sigmoid(prompt)
    mode = str(target_mode).lower()
    if mode in {"soft", "soft_error"}:
        correction_target = target_f * (1.0 - prob) - (1.0 - target_f) * prob
    elif mode in {"hard", "hard_error"}:
        threshold = float(hard_threshold)
        correction_target = (
            target_f * torch.relu(threshold - prob)
            - (1.0 - target_f) * torch.relu(prob - threshold)
        )
    else:
        raise ValueError(f"Unsupported dense_prompt_signed_residual_target_mode: {target_mode}")
    correction_target = correction_target * float(target_scale)
    if correction_target.shape[-2:] != prediction.shape[-2:]:
        correction_target = F.interpolate(
            correction_target,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    correction_target = correction_target.to(device=prediction.device, dtype=prediction.dtype)

    is_balanced, base_loss_type = _parse_balanced_loss_type(loss_type)
    if is_balanced:
        return _balanced_sparse_target_loss(
            prediction,
            correction_target,
            base_loss_type=base_loss_type,
        )
    if base_loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, correction_target)
    if base_loss_type == "l1":
        return F.l1_loss(prediction, correction_target)
    if base_loss_type == "mse":
        return F.mse_loss(prediction, correction_target)
    raise ValueError(f"Unsupported dense_prompt_signed_residual_loss: {loss_type}")


def compute_dense_prompt_unet_identity_loss(
    extras: Dict[str, torch.Tensor],
    *,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Penalize drift from the pre-U-Net dense prompt teacher.

    The residual-only U-Net branch is intended to be a conservative correction
    over the checkpoint-compatible prompt path.  This loss anchors the final
    dense prompt to the pre-U-Net dense logits and therefore regularizes only
    the new U-Net correction, not the legacy R265 prompt generator.
    """
    if extras is None:
        return torch.tensor(0.0)
    dense = extras.get("dense_prompt_mask")
    teacher = extras.get("dense_prompt_pre_unet")
    if dense is None and teacher is None:
        return torch.tensor(0.0)
    ref = dense if dense is not None else teacher
    if dense is None or teacher is None:
        return ref.new_tensor(0.0)
    dense_f = dense.float()
    teacher_f = teacher.detach().float()
    if teacher_f.shape[-2:] != dense_f.shape[-2:]:
        teacher_f = F.interpolate(
            teacher_f,
            size=dense_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return F.mse_loss(dense_f, teacher_f)
    if loss_type == "l1":
        return F.l1_loss(dense_f, teacher_f)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(dense_f, teacher_f)
    raise ValueError(f"Unsupported dense_prompt_unet_identity_loss: {loss_type}")


def compute_dense_prompt_unet_residual_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    target_scale: float = 1.0,
    loss_type: str = "balanced_smooth_l1",
    target_mode: str = "hard_error",
    hard_threshold: float = 0.5,
    use_gate: bool = False,
    max_target_area: Optional[float] = None,
) -> torch.Tensor:
    """Supervise the U-Net residual from pre-U-Net prompt hard errors.

    False negatives in the teacher prompt receive positive logit deltas and
    false positives receive negative deltas.  The teacher is detached so the
    loss trains only the new U-Net residual branch.
    """
    if extras is None:
        return target.new_tensor(0.0)
    delta = extras.get("dense_prompt_unet_delta")
    teacher = extras.get("dense_prompt_pre_unet")
    if delta is None or teacher is None:
        return target.new_tensor(0.0)

    prediction = delta.float()
    if use_gate:
        gate = extras.get("dense_prompt_unet_gate")
        if gate is not None:
            gate_f = gate.float()
            if gate_f.shape[-2:] != prediction.shape[-2:]:
                gate_f = F.interpolate(
                    gate_f,
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = prediction * gate_f

    target_f = (target.detach().float() > 0.5).float()
    teacher_f = teacher.detach().float()
    if teacher_f.shape[-2:] != target_f.shape[-2:]:
        teacher_f = F.interpolate(
            teacher_f,
            size=target_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if max_target_area is not None:
        target_area = target_f.mean(dim=tuple(range(1, target_f.ndim)))
        selected = target_area <= float(max_target_area)
        if not bool(selected.any().item()):
            return target.new_tensor(0.0)
        teacher_f = teacher_f[selected]
        target_f = target_f[selected]
        prediction = prediction[selected]

    prob = torch.sigmoid(teacher_f)
    mode = str(target_mode).lower()
    if mode in {"soft", "soft_error"}:
        correction_target = target_f * (1.0 - prob) - (1.0 - target_f) * prob
    elif mode in {"hard", "hard_error"}:
        threshold = float(hard_threshold)
        correction_target = (
            target_f * torch.relu(threshold - prob)
            - (1.0 - target_f) * torch.relu(prob - threshold)
        )
    else:
        raise ValueError(f"Unsupported dense_prompt_unet_residual_target_mode: {target_mode}")
    correction_target = correction_target * float(target_scale)
    if correction_target.shape[-2:] != prediction.shape[-2:]:
        correction_target = F.interpolate(
            correction_target,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    correction_target = correction_target.to(device=prediction.device, dtype=prediction.dtype)

    is_balanced, base_loss_type = _parse_balanced_loss_type(loss_type)
    if is_balanced:
        return _balanced_sparse_target_loss(
            prediction,
            correction_target,
            base_loss_type=base_loss_type,
        )
    if base_loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, correction_target)
    if base_loss_type == "l1":
        return F.l1_loss(prediction, correction_target)
    if base_loss_type == "mse":
        return F.mse_loss(prediction, correction_target)
    raise ValueError(f"Unsupported dense_prompt_unet_residual_loss: {loss_type}")


def compute_prompt_gate_oracle_target(
    prompt_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "fn_ratio",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a per-sample soft gate target from prompt/GT mismatch.

    ``fn_ratio`` is the original expansion-gate target: a high value means the
    prompt under-covers the ground truth, while a low value means the prompt is
    mostly false-positive.  This is appropriate for positive-only expansion
    gates.

    ``error_area`` is for signed correction gates such as spatial residual
    refiners.  It opens the gate for both false positives and false negatives,
    allowing the residual branch to learn local suppression or expansion.
    """
    mode = str(mode)
    prompt = prompt_logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if prompt.shape[-2:] != target_f.shape[-2:]:
        prompt = F.interpolate(
            prompt,
            size=target_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    prob = torch.sigmoid(prompt)
    dims = tuple(range(1, prob.ndim))
    if mode == "error_area":
        target_gate = (prob - target_f).abs().mean(dim=dims)
        return target_gate.clamp(0.0, 1.0).view(-1, 1, 1, 1)
    if mode == "fp_pixel_hard":
        return (((prob > 0.5).float() * (1.0 - target_f))).clamp(0.0, 1.0)
    if mode == "pixel_hard_error":
        threshold = 0.5
        target_gate = (
            target_f * torch.relu(threshold - prob)
            + (1.0 - target_f) * torch.relu(prob - threshold)
        )
        return target_gate.clamp(0.0, 1.0)
    if mode not in {"fn_ratio", "fp_ratio"}:
        raise ValueError(f"Unsupported prompt_gate_target_mode: {mode}")
    false_negative_area = (target_f * (1.0 - prob)).mean(dim=dims)
    false_positive_area = ((1.0 - target_f) * prob).mean(dim=dims)
    if mode == "fp_ratio":
        target_gate = false_positive_area / (false_negative_area + false_positive_area + float(eps))
        return target_gate.clamp(0.0, 1.0).view(-1, 1, 1, 1)
    target_gate = false_negative_area / (false_negative_area + false_positive_area + float(eps))
    return target_gate.clamp(0.0, 1.0).view(-1, 1, 1, 1)


def _prompt_source_from_extras(
    extras: Dict[str, torch.Tensor],
    prompt_source: str,
) -> Optional[torch.Tensor]:
    if prompt_source == "coarse_prompt":
        return extras.get("coarse_prompt")
    if prompt_source == "dense_prompt":
        return extras.get("dense_prompt_mask")
    if prompt_source == "dense_prompt_pre_unet":
        return extras.get("dense_prompt_pre_unet")
    if prompt_source == "coarse_mask":
        return extras.get("coarse_mask")
    if prompt_source == "pre_refiner_prompt":
        diagnostics = extras.get("prompt_calibrator")
        if isinstance(diagnostics, dict):
            value = diagnostics.get("pre_refiner_prompt")
            if torch.is_tensor(value):
                return value
        return extras.get("coarse_prompt")
    if prompt_source == "final_logits":
        return extras.get("final_logits")
    if prompt_source == "raw_final_logits":
        return extras.get("raw_final_logits", extras.get("final_logits"))
    raise ValueError(f"Unsupported prompt_gate_target_source: {prompt_source}")


def compute_prompt_gate_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prompt_source: str = "pre_refiner_prompt",
    target_mode: str = "fn_ratio",
    gate_sources: str = "refiner",
    refiner_gate_max: float = 1.0,
    dense_gate_max: float = 1.0,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Supervise prompt expansion gates with a GT-derived FN/FP oracle.

    The gates themselves are available at inference from image/prompt features;
    this loss only uses GT to teach the policy during training.  Gate values are
    normalized by their configured maxima before comparison with the oracle
    target in ``[0, 1]``.
    """
    if extras is None:
        return target.new_tensor(0.0)
    prompt_logits = _prompt_source_from_extras(extras, prompt_source)
    if prompt_logits is None:
        return target.new_tensor(0.0)
    oracle_target = compute_prompt_gate_oracle_target(
        prompt_logits,
        target,
        mode=target_mode,
    ).to(device=target.device)
    requested_sources = {part.strip().lower() for part in str(gate_sources).split(",") if part.strip()}
    if not requested_sources or requested_sources == {"none"}:
        return target.new_tensor(0.0)

    losses: List[torch.Tensor] = []
    if "dense" in requested_sources:
        dense_gate = extras.get("dense_prompt_gate")
        if dense_gate is not None:
            dense_norm = dense_gate.float() / max(float(dense_gate_max), 1e-6)
            losses.append(_gate_loss(dense_norm, oracle_target, loss_type=loss_type))
    if "small_dense" in requested_sources:
        small_gate = extras.get("dense_prompt_small_gate")
        if small_gate is not None:
            small_norm = small_gate.float() / max(float(dense_gate_max), 1e-6)
            losses.append(_gate_loss(small_norm, oracle_target, loss_type=loss_type))
    if "refiner" in requested_sources:
        prompt_diag = extras.get("prompt_calibrator")
        refiner_gate = prompt_diag.get("refiner_gate") if isinstance(prompt_diag, dict) else None
        if refiner_gate is not None:
            refiner_norm = refiner_gate.float() / max(float(refiner_gate_max), 1e-6)
            losses.append(_gate_loss(refiner_norm, oracle_target, loss_type=loss_type))
    unknown = requested_sources.difference({"dense", "small_dense", "refiner", "none"})
    if unknown:
        raise ValueError(f"Unsupported prompt_gate_sources: {sorted(unknown)}")
    if not losses:
        return target.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_dual_branch_prompt_gate_targets(
    prompt_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "sample_ratio",
    hard_threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return separate FG-expansion and BG-suppression gate targets.

    The single dense/refiner gate target mixes two opposing actions.  For the
    dual-branch prompt generator, the foreground branch should open when the
    current prompt under-covers the GT, while the background branch should open
    when the prompt over-covers background.  Targets are detached training
    oracles; inference still uses only image/prompt features.
    """
    prompt = prompt_logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if prompt.shape[-2:] != target_f.shape[-2:]:
        prompt = F.interpolate(
            prompt,
            size=target_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    prob = torch.sigmoid(prompt)
    mode = str(mode).lower()
    if mode in {
        "pixel_soft",
        "pixel_soft_error",
        "balanced_pixel_soft",
        "balanced_pixel_soft_error",
    }:
        fg_target = target_f * (1.0 - prob)
        bg_target = (1.0 - target_f) * prob
        return fg_target.clamp(0.0, 1.0), bg_target.clamp(0.0, 1.0)
    if mode in {
        "pixel_hard",
        "pixel_hard_error",
        "balanced_pixel_hard",
        "balanced_pixel_hard_error",
    }:
        threshold = float(hard_threshold)
        fg_target = target_f * torch.relu(threshold - prob)
        bg_target = (1.0 - target_f) * torch.relu(prob - threshold)
        return fg_target.clamp(0.0, 1.0), bg_target.clamp(0.0, 1.0)
    if mode not in {"sample_ratio", "sample", "area_ratio", "ratio"}:
        raise ValueError(f"Unsupported dual_branch_prompt_gate_target_mode: {mode}")
    dims = tuple(range(1, prob.ndim))
    false_negative_area = (target_f * (1.0 - prob)).mean(dim=dims)
    false_positive_area = ((1.0 - target_f) * prob).mean(dim=dims)
    denom = false_negative_area + false_positive_area + float(eps)
    fg_target = (false_negative_area / denom).clamp(0.0, 1.0).view(-1, 1, 1, 1)
    bg_target = (false_positive_area / denom).clamp(0.0, 1.0).view(-1, 1, 1, 1)
    return fg_target, bg_target


def compute_dual_branch_prompt_gate_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prompt_source: str = "dense_prompt",
    target_mode: str = "sample_ratio",
    hard_threshold: float = 0.5,
    dense_gate_max: float = 1.0,
    post_gate_max: Optional[float] = None,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Supervise dual-branch prompt gates with separate FN/FP oracles."""
    if extras is None:
        return target.new_tensor(0.0)
    prompt_logits = _prompt_source_from_extras(extras, prompt_source)
    fg_gate = extras.get("post_prompt_fg_gate")
    bg_gate = extras.get("post_prompt_bg_gate")
    gate_max = post_gate_max if post_gate_max is not None else dense_gate_max
    if fg_gate is None or bg_gate is None:
        fg_gate = extras.get("dense_prompt_fg_gate")
        bg_gate = extras.get("dense_prompt_bg_gate")
        gate_max = dense_gate_max
    if prompt_logits is None or fg_gate is None or bg_gate is None:
        return target.new_tensor(0.0)
    fg_target, bg_target = compute_dual_branch_prompt_gate_targets(
        prompt_logits,
        target,
        mode=target_mode,
        hard_threshold=hard_threshold,
    )
    if fg_target.shape[-2:] != fg_gate.shape[-2:]:
        fg_target = F.interpolate(fg_target, size=fg_gate.shape[-2:], mode="bilinear", align_corners=False)
    if bg_target.shape[-2:] != bg_gate.shape[-2:]:
        bg_target = F.interpolate(bg_target, size=bg_gate.shape[-2:], mode="bilinear", align_corners=False)
    fg_target = fg_target.to(device=fg_gate.device, dtype=fg_gate.dtype)
    bg_target = bg_target.to(device=bg_gate.device, dtype=bg_gate.dtype)
    max_gate = max(float(gate_max), 1e-6)
    fg_norm = fg_gate.float() / max_gate
    bg_norm = bg_gate.float() / max_gate
    return torch.stack(
        [
            _gate_loss(fg_norm, fg_target, loss_type=loss_type),
            _gate_loss(bg_norm, bg_target, loss_type=loss_type),
        ]
    ).mean()


def compute_dual_branch_prompt_residual_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prompt_source: str = "dense_prompt",
    target_scale: float = 1.0,
    loss_type: str = "smooth_l1",
    target_mode: str = "soft_error",
    hard_threshold: float = 0.5,
) -> torch.Tensor:
    """Teach dual-branch residual maps where to expand and suppress.

    Gate supervision teaches sample-level *whether* to use a branch.  This
    pixel-level oracle teaches the foreground residual to respond on prompt
    false negatives and the background residual to respond on prompt false
    positives.  The oracle is detached and used only during training.
    """
    if extras is None:
        return target.new_tensor(0.0)
    prompt_logits = _prompt_source_from_extras(extras, prompt_source)
    fg_residual = extras.get("post_prompt_fg_residual")
    bg_residual = extras.get("post_prompt_bg_residual")
    if fg_residual is None or bg_residual is None:
        fg_residual = extras.get("dense_prompt_fg_residual")
        bg_residual = extras.get("dense_prompt_bg_residual")
    if prompt_logits is None or fg_residual is None or bg_residual is None:
        return target.new_tensor(0.0)

    prompt = prompt_logits.detach().float()
    target_f = (target.detach().float() > 0.5).float()
    if prompt.shape[-2:] != target_f.shape[-2:]:
        prompt = F.interpolate(prompt, size=target_f.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(prompt)
    target_mode = str(target_mode).lower()
    if target_mode in {"soft", "soft_error"}:
        fg_target = target_f * (1.0 - prob)
        bg_target = (1.0 - target_f) * prob
    elif target_mode in {"hard", "hard_error"}:
        threshold = float(hard_threshold)
        fg_target = target_f * torch.relu(threshold - prob)
        bg_target = (1.0 - target_f) * torch.relu(prob - threshold)
    else:
        raise ValueError(f"Unsupported dual_branch_prompt_residual_target_mode: {target_mode}")

    if fg_target.shape[-2:] != fg_residual.shape[-2:]:
        fg_target = F.interpolate(fg_target, size=fg_residual.shape[-2:], mode="bilinear", align_corners=False)
        bg_target = F.interpolate(bg_target, size=bg_residual.shape[-2:], mode="bilinear", align_corners=False)
    fg_target = fg_target.to(device=fg_residual.device, dtype=fg_residual.dtype) * float(target_scale)
    bg_target = bg_target.to(device=bg_residual.device, dtype=bg_residual.dtype) * float(target_scale)

    is_balanced, base_loss_type = _parse_balanced_loss_type(loss_type)
    if is_balanced:
        fg_loss = _balanced_sparse_target_loss(
            fg_residual,
            fg_target,
            base_loss_type=base_loss_type,
        )
        bg_loss = _balanced_sparse_target_loss(
            bg_residual,
            bg_target,
            base_loss_type=base_loss_type,
        )
    elif base_loss_type == "smooth_l1":
        fg_loss = F.smooth_l1_loss(fg_residual, fg_target)
        bg_loss = F.smooth_l1_loss(bg_residual, bg_target)
    elif base_loss_type == "l1":
        fg_loss = F.l1_loss(fg_residual, fg_target)
        bg_loss = F.l1_loss(bg_residual, bg_target)
    elif base_loss_type == "mse":
        fg_loss = F.mse_loss(fg_residual, fg_target)
        bg_loss = F.mse_loss(bg_residual, bg_target)
    else:
        raise ValueError(f"Unsupported dual_branch_prompt_residual_loss: {loss_type}")
    return torch.stack([fg_loss, bg_loss]).mean()


def _binary_quality_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    metric: str = "iou",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return per-sample hard IoU/F1 for logits against ``target``.

    This is used only to build a detached training oracle for choosing between
    SAM3 prompt interfaces.  It intentionally has no gradient path to either
    branch logits, only to the fusion gate through the supervised gate loss.
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    probs = torch.sigmoid(logits.float())
    target_f = target.float()
    if target_f.shape[-2:] != probs.shape[-2:]:
        target_f = F.interpolate(target_f, size=probs.shape[-2:], mode="nearest")
    pred = (probs >= float(threshold)).float()
    gt = (target_f >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    tp = (pred * gt).sum(dim=dims)
    fp = (pred * (1.0 - gt)).sum(dim=dims)
    fn = ((1.0 - pred) * gt).sum(dim=dims)
    eps = 1e-6
    metric = str(metric).lower()
    if metric == "iou":
        return tp / (tp + fp + fn + eps)
    if metric == "f1":
        return (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    raise ValueError(f"Unsupported dual_prompt_fusion oracle metric: {metric}")


def compute_dual_prompt_fusion_supervision_loss(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    metric: str = "iou",
    loss_type: str = "bce",
) -> torch.Tensor:
    """Supervise ``dual_gated`` to choose the better SAM3 prompt interface.

    Target convention: ``0`` means legacy logits are better, ``1`` means native
    logits are better.  The target is computed from GT during training only and
    is detached from branch logits.
    """
    if extras is None:
        return target.new_tensor(0.0)
    gate = extras.get("dual_prompt_fusion_gate")
    legacy_logits = extras.get("dual_prompt_legacy_logits")
    native_logits = extras.get("dual_prompt_native_logits")
    if gate is None or legacy_logits is None or native_logits is None:
        return target.new_tensor(0.0)
    with torch.no_grad():
        metric_l = str(metric).lower()
        if metric_l in {"pixel_bce", "per_pixel_bce", "spatial_bce"}:
            logits_size = legacy_logits.shape[-2:]
            target_f = target.detach().float()
            if target_f.shape[-2:] != logits_size:
                target_f = F.interpolate(target_f, size=logits_size, mode="nearest")
            legacy_bce = F.binary_cross_entropy_with_logits(
                legacy_logits.detach().float(),
                target_f,
                reduction="none",
            )
            native_bce = F.binary_cross_entropy_with_logits(
                native_logits.detach().float(),
                target_f,
                reduction="none",
            )
            oracle_target = (native_bce < legacy_bce).to(dtype=gate.float().dtype)
            if oracle_target.shape[-2:] != gate.shape[-2:]:
                if gate.shape[-2:] == (1, 1):
                    oracle_target = oracle_target.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
                else:
                    oracle_target = F.interpolate(
                        oracle_target,
                        size=gate.shape[-2:],
                        mode="nearest",
                    )
            oracle_target = oracle_target.to(device=gate.device)
        else:
            legacy_quality = _binary_quality_from_logits(legacy_logits, target, metric=metric_l)
            native_quality = _binary_quality_from_logits(native_logits, target, metric=metric_l)
            oracle_target = (native_quality > legacy_quality).to(dtype=gate.float().dtype)
            oracle_target = oracle_target.view(-1, 1, 1, 1).to(device=gate.device)
    gate_f = gate.float()
    if oracle_target.shape != gate_f.shape:
        oracle_target = oracle_target.expand_as(gate_f)
    loss_type = str(loss_type).lower()
    if loss_type == "bce":
        gate_safe = gate_f.clamp(1e-6, 1.0 - 1e-6)
        return -(
            oracle_target * torch.log(gate_safe)
            + (1.0 - oracle_target) * torch.log(1.0 - gate_safe)
        ).mean()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(gate_f, oracle_target)
    if loss_type == "l1":
        return F.l1_loss(gate_f, oracle_target)
    if loss_type == "mse":
        return F.mse_loss(gate_f, oracle_target)
    raise ValueError(f"Unsupported dual_prompt_fusion loss: {loss_type}")


def _elementwise_supervision_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="none")
    if loss_type == "l1":
        return torch.abs(prediction - target)
    if loss_type == "mse":
        return (prediction - target) ** 2
    raise ValueError(f"Unsupported elementwise supervision loss: {loss_type}")


def _balanced_sparse_target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    base_loss_type: str,
    negative_weight: float = 0.05,
) -> torch.Tensor:
    """Average sparse non-zero targets without letting easy zero pixels dominate.

    Pixel-level endpoint supervision can be extremely sparse when it is derived
    from final false-positive/false-negative regions.  A normal full-image L1
    makes an all-zero gate/residual look good because most pixels have target
    zero.  This loss averages non-zero target pixels separately and keeps only a
    small regularizer on zero-target pixels.
    """
    if target.shape != prediction.shape:
        target = target.expand_as(prediction)
    per_pixel = _elementwise_supervision_loss(prediction, target, loss_type=base_loss_type)
    positive = target.detach().float() > 0.0
    negative = ~positive
    loss = prediction.new_tensor(0.0)
    if bool(positive.any().item()):
        loss = loss + per_pixel[positive].mean()
    if bool(negative.any().item()):
        loss = loss + float(negative_weight) * per_pixel[negative].mean()
    return loss


def _parse_balanced_loss_type(loss_type: str) -> Tuple[bool, str]:
    normalized = str(loss_type).lower()
    if normalized.startswith("balanced_"):
        base = normalized[len("balanced_") :]
        if base not in {"smooth_l1", "l1", "mse"}:
            raise ValueError(f"Unsupported balanced supervision base loss: {loss_type}")
        return True, base
    return False, normalized


def _gate_loss(gate_norm: torch.Tensor, target_gate: torch.Tensor, *, loss_type: str) -> torch.Tensor:
    # Do not clamp ``gate_norm`` here.  The gates are produced by bounded
    # sigmoids, but autocast/FP16 can round ``gate / gate_max`` slightly above
    # 1.0 (e.g. 0.30078 / 0.3).  Clamping would zero the supervision gradient
    # exactly when a saturated gate needs to be pushed back down toward a low
    # oracle target.
    if target_gate.ndim == gate_norm.ndim and target_gate.shape[-2:] != gate_norm.shape[-2:]:
        target_gate = F.interpolate(
            target_gate.float(),
            size=gate_norm.shape[-2:],
            mode="nearest",
        ).to(device=gate_norm.device, dtype=gate_norm.dtype)
    if target_gate.shape != gate_norm.shape:
        target_gate = target_gate.expand_as(gate_norm)
    is_balanced, base_loss_type = _parse_balanced_loss_type(loss_type)
    if is_balanced:
        return _balanced_sparse_target_loss(
            gate_norm,
            target_gate,
            base_loss_type=base_loss_type,
        )
    loss_type = base_loss_type
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(gate_norm, target_gate)
    if loss_type == "l1":
        return F.l1_loss(gate_norm, target_gate)
    if loss_type == "mse":
        return F.mse_loss(gate_norm, target_gate)
    raise ValueError(f"Unsupported prompt_gate_loss: {loss_type}")


def compute_prompt_supervision_loss(
    *,
    coarse_prompt: Optional[torch.Tensor],
    target: torch.Tensor,
    bce_weight: float = 0.0,
    dice_weight: float = 0.0,
    false_negative_weight: float = 0.0,
    false_positive_weight: float = 0.0,
    max_target_area: Optional[float] = None,
) -> torch.Tensor:
    """Auxiliary loss on the calibrated dense prompt sent to SAM."""
    if coarse_prompt is None or (
        bce_weight <= 0.0
        and dice_weight <= 0.0
        and false_negative_weight <= 0.0
        and false_positive_weight <= 0.0
    ):
        return target.new_tensor(0.0)
    prompt = coarse_prompt.float()
    if prompt.shape[-2:] != target.shape[-2:]:
        prompt = F.interpolate(
            prompt,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if max_target_area is not None:
        target_area = (target.detach().float() > 0.5).float().mean(
            dim=tuple(range(1, target.ndim))
        )
        selected = target_area <= float(max_target_area)
        if not bool(selected.any().item()):
            return target.new_tensor(0.0)
        prompt = prompt[selected]
        target = target[selected]
    loss = target.new_tensor(0.0)
    if bce_weight > 0.0:
        loss = loss + float(bce_weight) * F.binary_cross_entropy_with_logits(prompt, target.float())
    if dice_weight > 0.0:
        loss = loss + float(dice_weight) * dice_loss(prompt, target.float())
    if false_negative_weight > 0.0 or false_positive_weight > 0.0:
        target_f = target.float()
        prob = torch.sigmoid(prompt)
        if false_negative_weight > 0.0:
            loss = loss + float(false_negative_weight) * (target_f * (1.0 - prob)).mean()
        if false_positive_weight > 0.0:
            loss = loss + float(false_positive_weight) * ((1.0 - target_f) * prob).mean()
    return loss


def compute_prompt_supervision_loss_from_extras(
    *,
    extras: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prompt_source: str = "coarse_prompt",
    bce_weight: float = 0.0,
    dice_weight: float = 0.0,
    false_negative_weight: float = 0.0,
    false_positive_weight: float = 0.0,
    max_target_area: Optional[float] = None,
) -> torch.Tensor:
    """Auxiliary prompt loss on a configurable model output.

    Historically ``coarse_prompt_*`` losses always supervised the calibrated
    dense prompt sent to SAM.  For teacher-guided diagnostics we also need to
    supervise the upstream raw dense-prompt logits before SAM3 centering/biasing
    to determine whether the source prompt can learn better localization.
    """
    prompt = _prompt_source_from_extras(extras, prompt_source)
    active = (
        bce_weight > 0.0
        or dice_weight > 0.0
        or false_negative_weight > 0.0
        or false_positive_weight > 0.0
    )
    if prompt is None and active:
        raise ValueError(
            f"coarse_prompt_supervision_source={prompt_source!r} is unavailable in extras"
        )
    return compute_prompt_supervision_loss(
        coarse_prompt=prompt,
        target=target,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        false_negative_weight=false_negative_weight,
        false_positive_weight=false_positive_weight,
        max_target_area=max_target_area,
    )


def compute_loss(
    device: torch.device,
    lambda_focal: float,
    focal_gamma: float,
    focal_alpha: float,
    lambda_iou: float,
    train_sam_iou: bool,
    tgt: torch.Tensor,
    logits: torch.Tensor,
    extras: Dict[str, torch.Tensor],
    authentic_ratio: float = 0.0,
    lambda_detection: float = 1.0,
    is_authentic: Optional[List[bool]] = None,
    sample_types: Optional[List[str]] = None,
    coarse_loss_weight: float = 0.2,
    coarse_dice_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute the combined loss for forgery localization training.
    
    Args:
        device: Training device (cuda/cpu)
        lambda_focal: Weight for focal loss
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss
        lambda_iou: Weight for IoU prediction loss
        train_sam_iou: Whether to train SAM IoU head
        tgt: Ground truth masks
        logits: Model predictions
        extras: Additional model outputs (IoU predictions, coarse masks, detection logits)
        authentic_ratio: Ratio of authentic images (for detection loss)
        lambda_detection: Weight for detection loss
        is_authentic: Optional list indicating whether each sample is authentic
        sample_types: Optional list of sample types ("authentic" or "forgery")
        
    Returns:
        Tuple of (sam_iou, aux_bce_loss, focal_loss, dice_loss, actual_iou, iou_loss, detection_loss, total_loss)
    """
    sam_iou = extras["iou_pred"]
    coarse_logits = extras["coarse_mask"]
    
    # Upsample coarse logits to target size
    coarse_up = F.interpolate(
        coarse_logits,
        size=tgt.shape[-2:],
        mode="nearest"
    )
    
    # Do not hard-clamp final logits here: clamping above the bound zeros the
    # gradient and can lock a run into an all-foreground/all-background state.
    # The focal helper has its own local numerical clamp; Dice/sigmoid are
    # stable on the original logits.
    coarse_up = torch.clamp(coarse_up, min=-10.0, max=10.0)
    
    if is_authentic is not None:
        if torch.is_tensor(is_authentic):
            authentic_tensor = is_authentic.to(device=tgt.device, dtype=torch.bool)
        else:
            authentic_tensor = torch.tensor(
                [bool(val) for val in is_authentic],
                device=tgt.device,
                dtype=torch.bool,
            )
        forgery_mask = ~authentic_tensor
    elif sample_types is not None:
        forgery_mask = torch.tensor(
            [sample_type != "authentic" for sample_type in sample_types],
            device=tgt.device,
            dtype=torch.bool,
        )
    else:
        forgery_mask = (tgt.sum(dim=[1, 2, 3]) > 0)

    if forgery_mask.any():
        logits_for_loss = logits[forgery_mask]
        tgt_for_loss = tgt[forgery_mask]
        coarse_up_for_loss = coarse_up[forgery_mask]
        sam_iou_for_loss = sam_iou[forgery_mask]

        aux_bce_l = F.binary_cross_entropy_with_logits(coarse_up_for_loss, tgt_for_loss)
        # Clamp BCE loss to prevent extreme values
        aux_bce_l = torch.clamp(aux_bce_l, min=0.0, max=10.0)
        coarse_dice_l = dice_loss(coarse_up_for_loss, tgt_for_loss)
        coarse_dice_l = torch.clamp(coarse_dice_l, min=0.0, max=10.0)

        focal_loss_component = sigmoid_focal_loss(
            logits_for_loss, tgt_for_loss, alpha=focal_alpha, gamma=focal_gamma
        )
        # Clamp focal loss to prevent extreme values
        focal_loss_component = torch.clamp(focal_loss_component, min=0.0, max=10.0)

        dice_l = dice_loss(logits_for_loss, tgt_for_loss)
        # Clamp dice loss to prevent extreme values
        dice_l = torch.clamp(dice_l, min=0.0, max=10.0)

        seg_loss = (
            lambda_focal * focal_loss_component
            + dice_l
            + float(coarse_loss_weight) * aux_bce_l
            + float(coarse_dice_weight) * coarse_dice_l
        )
        # Clamp seg loss to prevent overflow
        seg_loss = torch.clamp(seg_loss, min=0.0, max=100.0)
    else:
        sam_iou_for_loss = torch.zeros(1, device=device)
        aux_bce_l = torch.tensor(0.0, device=device)
        coarse_dice_l = torch.tensor(0.0, device=device)
        focal_loss_component = torch.tensor(0.0, device=device)
        dice_l = torch.tensor(0.0, device=device)
        seg_loss = torch.tensor(0.0, device=device)
    
    # Calculate actual IoU and IoU prediction loss if training SAM IoU head
    actual_iou = None
    if train_sam_iou and forgery_mask.any():
        with torch.no_grad():
            pred_masks = torch.sigmoid(logits_for_loss) > 0.5
            intersection = (pred_masks & (tgt_for_loss > 0.5)).float().sum(dim=[2, 3])
            union = (pred_masks | (tgt_for_loss > 0.5)).float().sum(dim=[2, 3])
            both_empty = (union == 0)
            actual_iou = torch.where(
                both_empty,
                torch.ones_like(union),
                intersection / (union + 1e-8),
            )
            # Ensure actual_iou is valid
            actual_iou = torch.nan_to_num(actual_iou, nan=0, posinf=0, neginf=0)

        # Always ensure sam_iou is within valid range
        sam_iou_for_loss = torch.clamp(sam_iou_for_loss, min=0.0, max=1.0)

        # Ensure actual_iou is valid and within range
        actual_iou = torch.clamp(actual_iou, min=0.0, max=1.0)

        # Calculate IoU loss with additional safety checks
        iou_loss = F.l1_loss(sam_iou_for_loss.view(-1), actual_iou.view(-1))
        # Clamp IoU loss to prevent extreme values
        iou_loss = torch.clamp(iou_loss, min=0.0, max=10.0)
    else:
        iou_loss = torch.tensor(0.0, device=device)
    
    # Detection loss (only when authentic_ratio > 0)
    detection_loss = torch.tensor(0.0, device=device)
    if authentic_ratio > 0.0 and 'detection_logit' in extras and extras['detection_logit'] is not None:
        detection_logit = extras['detection_logit']  # [B, 1]
        detection_logit = torch.clamp(detection_logit, min=-10.0, max=10.0)
       
        # Derive detection targets from sample types when available.
        with torch.no_grad():
            if is_authentic is not None or sample_types is not None:
                detection_targets = forgery_mask.float()
            else:
                detection_targets = (tgt.sum(dim=[1, 2, 3]) > 0).float()  # [B]
        
        # Binary cross-entropy loss for detection
        detection_loss = F.binary_cross_entropy_with_logits(
            detection_logit.view(-1),
            detection_targets.view(-1),
            reduction='mean',
        )
        # Clamp detection loss to prevent extreme values
        detection_loss = torch.clamp(detection_loss, min=0.0, max=10.0)
    
    # Combine all losses
    if train_sam_iou:
        total_loss = seg_loss + lambda_iou * iou_loss
    else:
        total_loss = seg_loss
    
    # Add detection loss if applicable
    if authentic_ratio > 0.0 and detection_loss.item() > 0.0:
        total_loss = total_loss + lambda_detection * detection_loss
    
    # Final clamp on total loss to prevent nan/inf
    total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
    
    return sam_iou_for_loss, aux_bce_l, focal_loss_component, dice_l, actual_iou, iou_loss, detection_loss, total_loss


# ------------------------------


def main(
    device: torch.device,
    lambda_focal: float,
    focal_gamma: float,
    focal_alpha: float,
    weight_decay: float,
    prompt_dim: int,
    save_path: str,
    downscale: int,
    lambda_iou: float = 1.0,
    ema_decay: float = 0.95,
    img_size: int = 512,
    lr: float = 1e-3,
    dropout_rate: float = 0.1,
    lad_tau: float = 0.004,
    lad_multi_taus: Optional[str] = None,
    forensic_operator: str = "lad",
    scheduler_type: str = "onecycle",
    authentic_ratio: float = 0.0,
    authentic_source_dir: Optional[str] = None,
    val_steps: int = 4000,
    dataset_config: str = None,
    val_manifest: Optional[str] = None,
    lambda_detection: float = 1.0,
    resume_checkpoint: Optional[str] = None,
    train_sam_iou: bool = True,
    train_force_resize: bool = False,
    val_force_resize: bool = False,
    sam_config: Optional[str] = None,
    sam_ckpt: Optional[str] = None,
    sam_backend: str = "sam2",
    freeze_adapters: bool = False,
    freeze_ferret: bool = False,
    freeze_ferret_unet_only: bool = False,
    adapter_residual_scale: float = 1.0,
    adapter_type: str = "shared",
    adapter_scales: Optional[str] = None,
    adapter_gamma_init: float = 0.0,
    adapter_sample_gate: bool = False,
    adapter_sample_gate_scales: Optional[str] = None,
    adapter_sample_gate_max_delta: float = 0.5,
    adapter_forensic_source: str = "final",
    adapter_delta_reg_weight: float = 0.0,
    adapter_diagnostics: bool = False,
    sam3_prompt_mode: str = "legacy",
    coarse_prompt_transform: str = "none",
    coarse_prompt_scale: float = 1.0,
    coarse_prompt_bias: float = 0.0,
    coarse_prompt_calibrator: str = "none",
    coarse_prompt_calibrator_hidden: int = 16,
    coarse_prompt_calibrator_max_delta_scale: float = 1.0,
    coarse_prompt_calibrator_max_delta_bias: float = 2.0,
    coarse_prompt_calibrator_reg_weight: float = 0.0,
    coarse_prompt_calibrator_lr_multiplier: float = 1.0,
    prompt_bias_supervision_weight: float = 0.0,
    prompt_bias_supervision_loss: str = "smooth_l1",
    prompt_bias_supervision_max_delta_bias: Optional[float] = None,
    final_logit_calibrator: str = "none",
    final_logit_calibrator_hidden: int = 16,
    final_logit_calibrator_max_delta_scale: float = 0.0,
    final_logit_calibrator_max_delta_bias: float = 1.0,
    final_logit_calibrator_supervision_weight: float = 0.0,
    final_logit_calibrator_supervision_loss: str = "smooth_l1",
    final_logit_calibrator_supervision_max_delta_bias: Optional[float] = None,
    final_logit_calibrator_oracle_thresholds: Optional[str] = None,
    final_logit_calibrator_oracle_min_threshold: Optional[float] = None,
    final_logit_calibrator_oracle_max_threshold: Optional[float] = None,
    final_logit_calibrator_oracle_false_positive_penalty: float = 0.0,
    final_logit_calibrator_oracle_area_penalty: float = 0.0,
    final_logit_spatial_supervision_weight: float = 0.0,
    final_logit_spatial_supervision_loss: str = "smooth_l1",
    final_logit_spatial_supervision_target_scale: float = 1.0,
    final_logit_spatial_supervision_target_mode: str = "hard_error",
    final_logit_spatial_supervision_hard_threshold: float = 0.5,
    coarse_prompt_refiner: str = "none",
    coarse_prompt_refiner_hidden: int = 8,
    coarse_prompt_refiner_max_residual: float = 1.0,
    coarse_prompt_refiner_gate_init: float = 0.2,
    coarse_prompt_refiner_gate_max: float = 1.0,
    coarse_prompt_refiner_precision_bias: float = -0.10,
    coarse_prompt_refiner_recall_bias: float = 0.45,
    coarse_prompt_head: str = "mask_compressor",
    coarse_prompt_hidden: Optional[int] = None,
    coarse_prompt_dropout: float = 0.0,
    coarse_prompt_gate_init: float = 0.02,
    coarse_prompt_gate_max: float = 1.0,
    coarse_prompt_area_bias: bool = False,
    coarse_prompt_signed_residual_max_delta: float = 0.5,
    coarse_prompt_unet_gate_init: Optional[float] = None,
    coarse_prompt_unet_gate_max: Optional[float] = None,
    coarse_prompt_unet_signed_residual_max_delta: Optional[float] = None,
    coarse_prompt_head_lr_multiplier: float = 1.0,
    dense_prompt_residual_reg_weight: float = 0.0,
    dense_prompt_signed_residual_supervision_weight: float = 0.0,
    dense_prompt_signed_residual_target_source: str = "coarse_mask",
    dense_prompt_signed_residual_loss: str = "smooth_l1",
    dense_prompt_signed_residual_target_scale: float = 1.0,
    dense_prompt_signed_residual_target_mode: str = "hard_error",
    dense_prompt_signed_residual_hard_threshold: float = 0.5,
    dense_prompt_signed_residual_use_gate: bool = False,
    dense_prompt_signed_residual_max_gt_area: Optional[float] = None,
    dense_prompt_unet_identity_weight: float = 0.0,
    dense_prompt_unet_identity_loss: str = "mse",
    dense_prompt_unet_residual_supervision_weight: float = 0.0,
    dense_prompt_unet_residual_loss: str = "balanced_smooth_l1",
    dense_prompt_unet_residual_target_scale: float = 1.0,
    dense_prompt_unet_residual_target_mode: str = "hard_error",
    dense_prompt_unet_residual_hard_threshold: float = 0.5,
    dense_prompt_unet_residual_use_gate: bool = False,
    dense_prompt_unet_residual_max_gt_area: Optional[float] = None,
    coarse_prompt_loss_weight: float = 0.0,
    coarse_prompt_dice_weight: float = 0.0,
    coarse_prompt_false_negative_weight: float = 0.0,
    coarse_prompt_false_positive_weight: float = 0.0,
    coarse_prompt_supervision_source: str = "coarse_prompt",
    coarse_prompt_supervision_max_gt_area: Optional[float] = None,
    prompt_gate_supervision_weight: float = 0.0,
    prompt_gate_target_source: str = "pre_refiner_prompt",
    prompt_gate_target_mode: str = "fn_ratio",
    prompt_gate_sources: str = "refiner",
    prompt_gate_loss: str = "smooth_l1",
    dual_branch_prompt_gate_supervision_weight: float = 0.0,
    dual_branch_prompt_gate_target_source: str = "dense_prompt",
    dual_branch_prompt_gate_target_mode: str = "sample_ratio",
    dual_branch_prompt_gate_hard_threshold: float = 0.5,
    dual_branch_prompt_gate_loss: str = "smooth_l1",
    dual_branch_prompt_residual_supervision_weight: float = 0.0,
    dual_branch_prompt_residual_target_source: str = "dense_prompt",
    dual_branch_prompt_residual_loss: str = "smooth_l1",
    dual_branch_prompt_residual_target_scale: float = 1.0,
    dual_branch_prompt_residual_target_mode: str = "soft_error",
    dual_branch_prompt_residual_hard_threshold: float = 0.5,
    dual_prompt_fusion_supervision_weight: float = 0.0,
    dual_prompt_fusion_oracle_metric: str = "iou",
    dual_prompt_fusion_loss: str = "bce",
    resume_use_ema_weights: bool = False,
    reset_best_score: bool = False,
    coarse_loss_weight: float = 0.2,
    coarse_dice_weight: float = 0.0,
    area_reg_weight: float = 0.0,
    area_reg_target_source: str = "gt",
    area_reg_loss: str = "smooth_l1",
    area_reg_apply_to: str = "coarse",
    area_reg_constant: float = 0.25,
    area_reg_max_gt_area: Optional[float] = None,
    reload_diagnostics_batches: int = 0,
    checkpoint_save_mode: str = "full",
) -> None:
    """
    Main training function for FLAME.
    
    Args:
        device: Training device (cuda/cpu)
        lambda_focal: Weight for focal loss
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss
        weight_decay: Weight decay for optimizer
        prompt_dim: Dimension of SAM prompt embeddings
        save_path: Path to save the trained model
        downscale: Downscale factor for mask adapter
        lambda_iou: Weight for IoU prediction loss
        ema_decay: Decay rate for EMA
        img_size: Input image size
        lr: Learning rate
        dropout_rate: Dropout rate for adapters
        scheduler_type: Type of learning rate scheduler
        authentic_ratio: Ratio of authentic images to include
        authentic_source_dir: Optional; ignored (authentic images use dataset root/source)
        val_steps: Number of training steps between validation runs
        lambda_detection: Weight for auxiliary detection loss when authentic samples are present
        resume_checkpoint: Optional path to a checkpoint for resuming training
        train_force_resize: If True, resize training samples instead of balanced crop
        val_force_resize: If True, resize validation samples instead of balanced crop
        freeze_adapters: If True, keep residual adapters frozen at identity
        freeze_ferret: If True, keep the LAD/Ferret prompt backbone frozen
        adapter_residual_scale: Multiplier for adapter residual deltas
        adapter_type: Adapter architecture to use
        adapter_scales: Comma-separated active adapter scales; all when unset
        adapter_gamma_init: Initial per-scale gamma for gated adapters
        adapter_delta_reg_weight: Weight for adapter feature-drift regularization
        adapter_diagnostics: If True, collect adapter per-scale diagnostics in training
        resume_use_ema_weights: If True, load checkpoint EMA weights into matching model params
        reset_best_score: If True, ignore resumed checkpoint score when deciding saves
        coarse_loss_weight: Weight for auxiliary coarse prompt BCE loss
        reload_diagnostics_batches: If >0, compare live validation weights against a fresh reload after each saved checkpoint
        checkpoint_save_mode: full, trainable_only, or none; use none for short diagnostics on low disk
    """
    if sam_config is None:
        sam_config = "sam2.1_hiera_b+.yaml"
    if sam_ckpt is None:
        sam_ckpt = "sam2configs/sam2.1_hiera_base_plus.pt"

    # Generate experiment name with timestamp and create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_strings = get_param_strings()
    experiment_name_raw = f"{timestamp}_DET_" + "_".join(param_strings)
    experiment_name = sanitize_experiment_name(experiment_name_raw)
    if experiment_name != experiment_name_raw:
        logger.debug("Sanitized experiment name | raw=%s sanitized=%s", experiment_name_raw, experiment_name)
    # Ferret-SAM doesn't use perturbation streams
    max_streams = get_max_streams()
    final_logit_oracle_thresholds = parse_float_list(final_logit_calibrator_oracle_thresholds)
    
    # Enable detection probe when authentic_ratio > 0
    use_detection_probe = authentic_ratio > 0.0
    
    logger.info("Running experiment: %s", experiment_name)
    logger.info(
        "Ferret-SAM configuration | max_streams=%s",
        max_streams,
    )
    logger.info("Authentic ratio: %.3f", authentic_ratio)
    logger.info("Detection probe enabled: %s", use_detection_probe)
    logger.info("Freeze adapters: %s", freeze_adapters)
    logger.info("Freeze Ferret/LAD backbone: %s", freeze_ferret)
    logger.info("Train only Ferret U-Net residual branch: %s", freeze_ferret_unet_only)
    logger.info("Adapter residual scale: %.4f", adapter_residual_scale)
    active_adapter_scales = parse_adapter_scales(adapter_scales)
    active_adapter_sample_gate_scales = parse_adapter_scales(adapter_sample_gate_scales)
    logger.info("Adapter type: %s", adapter_type)
    logger.info("Adapter active scales: %s", active_adapter_scales if active_adapter_scales is not None else "all")
    logger.info(
        "Adapter sample gate: enabled=%s scales=%s max_delta=%.3f",
        adapter_sample_gate,
        active_adapter_sample_gate_scales if active_adapter_sample_gate_scales is not None else "all",
        adapter_sample_gate_max_delta,
    )
    logger.info("Adapter gamma init: %.6f", adapter_gamma_init)
    logger.info("Adapter delta regularization weight: %.6f", adapter_delta_reg_weight)
    logger.info("SAM3 prompt mode: %s", sam3_prompt_mode)
    logger.info("Forensic operator: %s", forensic_operator)
    parsed_lad_multi_taus = parse_float_list(lad_multi_taus)
    logger.info("LAD multi taus: %s", parsed_lad_multi_taus if parsed_lad_multi_taus is not None else "default")
    logger.info(
        "Coarse prompt transform: %s scale=%.4f bias=%.4f",
        coarse_prompt_transform,
        coarse_prompt_scale,
        coarse_prompt_bias,
    )
    logger.info(
        "Coarse prompt calibrator: %s hidden=%s max_delta_scale=%.4f max_delta_bias=%.4f reg_weight=%.6f",
        coarse_prompt_calibrator,
        coarse_prompt_calibrator_hidden,
        coarse_prompt_calibrator_max_delta_scale,
        coarse_prompt_calibrator_max_delta_bias,
        coarse_prompt_calibrator_reg_weight,
    )
    logger.info(
        "Coarse prompt calibrator LR multiplier: %.4f",
        coarse_prompt_calibrator_lr_multiplier,
    )
    logger.info(
        "Prompt bias oracle supervision | weight=%.6f loss=%s max_delta_bias=%s",
        prompt_bias_supervision_weight,
        prompt_bias_supervision_loss,
        prompt_bias_supervision_max_delta_bias,
    )
    logger.info(
        "Final logit calibrator: %s hidden=%s max_delta_scale=%.4f max_delta_bias=%.4f supervision_weight=%.6f loss=%s supervision_max_delta_bias=%s oracle_thresholds=%s oracle_min=%s oracle_max=%s oracle_fp_penalty=%.4f oracle_area_penalty=%.4f",
        final_logit_calibrator,
        final_logit_calibrator_hidden,
        final_logit_calibrator_max_delta_scale,
        final_logit_calibrator_max_delta_bias,
        final_logit_calibrator_supervision_weight,
        final_logit_calibrator_supervision_loss,
        final_logit_calibrator_supervision_max_delta_bias,
        final_logit_oracle_thresholds,
        final_logit_calibrator_oracle_min_threshold,
        final_logit_calibrator_oracle_max_threshold,
        final_logit_calibrator_oracle_false_positive_penalty,
        final_logit_calibrator_oracle_area_penalty,
    )
    logger.info(
        "Final logit spatial supervision | weight=%.6f loss=%s target_scale=%.4f target_mode=%s hard_threshold=%.4f",
        final_logit_spatial_supervision_weight,
        final_logit_spatial_supervision_loss,
        final_logit_spatial_supervision_target_scale,
        final_logit_spatial_supervision_target_mode,
        final_logit_spatial_supervision_hard_threshold,
    )
    logger.info(
        "Coarse prompt refiner: %s hidden=%s max_residual=%.4f gate_init=%.4f gate_max=%.4f precision_bias=%.4f recall_bias=%.4f",
        coarse_prompt_refiner,
        coarse_prompt_refiner_hidden,
        coarse_prompt_refiner_max_residual,
        coarse_prompt_refiner_gate_init,
        coarse_prompt_refiner_gate_max,
        coarse_prompt_refiner_precision_bias,
        coarse_prompt_refiner_recall_bias,
    )
    logger.info(
        "Coarse prompt U-Net residual scales: gate_init=%s gate_max=%s max_delta=%s",
        coarse_prompt_unet_gate_init if coarse_prompt_unet_gate_init is not None else coarse_prompt_gate_init,
        coarse_prompt_unet_gate_max if coarse_prompt_unet_gate_max is not None else coarse_prompt_gate_max,
        (
            coarse_prompt_unet_signed_residual_max_delta
            if coarse_prompt_unet_signed_residual_max_delta is not None
            else coarse_prompt_signed_residual_max_delta
        ),
    )
    logger.info("Dense prompt residual regularization weight: %.6f", dense_prompt_residual_reg_weight)
    logger.info("Coarse loss weight: %.4f", coarse_loss_weight)
    logger.info(
        "Coarse prompt supervision: source=%s max_gt_area=%s bce_weight=%.4f dice_weight=%.4f fn_weight=%.4f fp_weight=%.4f",
        coarse_prompt_supervision_source,
        coarse_prompt_supervision_max_gt_area,
        coarse_prompt_loss_weight,
        coarse_prompt_dice_weight,
        coarse_prompt_false_negative_weight,
        coarse_prompt_false_positive_weight,
    )
    logger.info(
        "Prompt gate supervision | weight=%.6f source=%s mode=%s gates=%s loss=%s",
        prompt_gate_supervision_weight,
        prompt_gate_target_source,
        prompt_gate_target_mode,
        prompt_gate_sources,
        prompt_gate_loss,
    )
    logger.info(
        "Dual-branch prompt gate supervision | weight=%.6f source=%s target_mode=%s hard_threshold=%.3f loss=%s",
        dual_branch_prompt_gate_supervision_weight,
        dual_branch_prompt_gate_target_source,
        dual_branch_prompt_gate_target_mode,
        dual_branch_prompt_gate_hard_threshold,
        dual_branch_prompt_gate_loss,
    )
    logger.info(
        "Dual-branch prompt residual supervision | weight=%.6f source=%s loss=%s target_scale=%.4f target_mode=%s hard_threshold=%.3f",
        dual_branch_prompt_residual_supervision_weight,
        dual_branch_prompt_residual_target_source,
        dual_branch_prompt_residual_loss,
        dual_branch_prompt_residual_target_scale,
        dual_branch_prompt_residual_target_mode,
        dual_branch_prompt_residual_hard_threshold,
    )
    logger.info(
        "Dual prompt fusion supervision | weight=%.6f metric=%s loss=%s",
        dual_prompt_fusion_supervision_weight,
        dual_prompt_fusion_oracle_metric,
        dual_prompt_fusion_loss,
    )
    logger.info(
        "Area regularization | weight=%.6f target=%s loss=%s apply_to=%s constant=%.4f max_gt_area=%s",
        area_reg_weight,
        area_reg_target_source,
        area_reg_loss,
        area_reg_apply_to,
        area_reg_constant,
        area_reg_max_gt_area,
    )

    # Create experiment directory structure
    experiment_dir = os.path.join("results", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "vis"), exist_ok=True)
    
    if authentic_source_dir:
        logger.info(
            "authentic_source_dir is ignored; authentic images use each dataset's root/source folder."
        )
        authentic_source_dir = None

    # Collect all parameters for saving
    model_params = {
        "experiment_name": experiment_name,
        "model_config": {
            "prompt_dim": prompt_dim,
            "downscale": downscale,
            "train_sam_iou": train_sam_iou,
            "dropout_rate": dropout_rate,
            "lad_tau": lad_tau,
            "lad_multi_taus": parsed_lad_multi_taus,
            "forensic_operator": forensic_operator,
            "sam_backend": sam_backend,
            "freeze_adapters": freeze_adapters,
            "freeze_ferret": freeze_ferret,
            "freeze_ferret_unet_only": freeze_ferret_unet_only,
            "adapter_residual_scale": adapter_residual_scale,
            "adapter_type": adapter_type,
            "adapter_scales": active_adapter_scales,
            "adapter_gamma_init": adapter_gamma_init,
            "adapter_sample_gate": adapter_sample_gate,
            "adapter_sample_gate_scales": active_adapter_sample_gate_scales,
            "adapter_sample_gate_max_delta": adapter_sample_gate_max_delta,
            "adapter_forensic_source": adapter_forensic_source,
            "adapter_diagnostics": adapter_diagnostics,
            "sam3_prompt_mode": sam3_prompt_mode,
            "coarse_prompt_transform": coarse_prompt_transform,
            "coarse_prompt_scale": coarse_prompt_scale,
            "coarse_prompt_bias": coarse_prompt_bias,
            "coarse_prompt_calibrator": coarse_prompt_calibrator,
            "coarse_prompt_calibrator_hidden": coarse_prompt_calibrator_hidden,
            "coarse_prompt_calibrator_max_delta_scale": coarse_prompt_calibrator_max_delta_scale,
            "coarse_prompt_calibrator_max_delta_bias": coarse_prompt_calibrator_max_delta_bias,
            "coarse_prompt_calibrator_lr_multiplier": coarse_prompt_calibrator_lr_multiplier,
            "prompt_bias_supervision_weight": prompt_bias_supervision_weight,
            "prompt_bias_supervision_loss": prompt_bias_supervision_loss,
            "prompt_bias_supervision_max_delta_bias": prompt_bias_supervision_max_delta_bias,
            "final_logit_calibrator": final_logit_calibrator,
            "final_logit_calibrator_hidden": final_logit_calibrator_hidden,
            "final_logit_calibrator_max_delta_scale": final_logit_calibrator_max_delta_scale,
            "final_logit_calibrator_max_delta_bias": final_logit_calibrator_max_delta_bias,
            "final_logit_calibrator_supervision_weight": final_logit_calibrator_supervision_weight,
            "final_logit_calibrator_supervision_loss": final_logit_calibrator_supervision_loss,
            "final_logit_calibrator_supervision_max_delta_bias": final_logit_calibrator_supervision_max_delta_bias,
            "final_logit_calibrator_oracle_thresholds": final_logit_oracle_thresholds,
            "final_logit_calibrator_oracle_min_threshold": final_logit_calibrator_oracle_min_threshold,
            "final_logit_calibrator_oracle_max_threshold": final_logit_calibrator_oracle_max_threshold,
            "final_logit_calibrator_oracle_false_positive_penalty": final_logit_calibrator_oracle_false_positive_penalty,
            "final_logit_calibrator_oracle_area_penalty": final_logit_calibrator_oracle_area_penalty,
            "final_logit_spatial_supervision_weight": final_logit_spatial_supervision_weight,
            "final_logit_spatial_supervision_loss": final_logit_spatial_supervision_loss,
            "final_logit_spatial_supervision_target_scale": final_logit_spatial_supervision_target_scale,
            "final_logit_spatial_supervision_target_mode": final_logit_spatial_supervision_target_mode,
            "final_logit_spatial_supervision_hard_threshold": final_logit_spatial_supervision_hard_threshold,
            "coarse_prompt_refiner": coarse_prompt_refiner,
            "coarse_prompt_refiner_hidden": coarse_prompt_refiner_hidden,
            "coarse_prompt_refiner_max_residual": coarse_prompt_refiner_max_residual,
            "coarse_prompt_refiner_gate_init": coarse_prompt_refiner_gate_init,
            "coarse_prompt_refiner_gate_max": coarse_prompt_refiner_gate_max,
            "coarse_prompt_refiner_precision_bias": coarse_prompt_refiner_precision_bias,
            "coarse_prompt_refiner_recall_bias": coarse_prompt_refiner_recall_bias,
            "coarse_prompt_head": coarse_prompt_head,
            "coarse_prompt_hidden": coarse_prompt_hidden,
            "coarse_prompt_dropout": coarse_prompt_dropout,
            "coarse_prompt_gate_init": coarse_prompt_gate_init,
            "coarse_prompt_gate_max": coarse_prompt_gate_max,
            "coarse_prompt_area_bias": coarse_prompt_area_bias,
            "coarse_prompt_signed_residual_max_delta": coarse_prompt_signed_residual_max_delta,
            "coarse_prompt_unet_gate_init": coarse_prompt_unet_gate_init,
            "coarse_prompt_unet_gate_max": coarse_prompt_unet_gate_max,
            "coarse_prompt_unet_signed_residual_max_delta": coarse_prompt_unet_signed_residual_max_delta,
            "coarse_prompt_head_lr_multiplier": coarse_prompt_head_lr_multiplier,
            "dense_prompt_residual_reg_weight": dense_prompt_residual_reg_weight,
            "dense_prompt_signed_residual_supervision_weight": dense_prompt_signed_residual_supervision_weight,
            "dense_prompt_signed_residual_target_source": dense_prompt_signed_residual_target_source,
            "dense_prompt_signed_residual_loss": dense_prompt_signed_residual_loss,
            "dense_prompt_signed_residual_target_scale": dense_prompt_signed_residual_target_scale,
            "dense_prompt_signed_residual_target_mode": dense_prompt_signed_residual_target_mode,
            "dense_prompt_signed_residual_hard_threshold": dense_prompt_signed_residual_hard_threshold,
            "dense_prompt_signed_residual_use_gate": dense_prompt_signed_residual_use_gate,
            "dense_prompt_signed_residual_max_gt_area": dense_prompt_signed_residual_max_gt_area,
            "dense_prompt_unet_identity_weight": dense_prompt_unet_identity_weight,
            "dense_prompt_unet_identity_loss": dense_prompt_unet_identity_loss,
            "dense_prompt_unet_residual_supervision_weight": dense_prompt_unet_residual_supervision_weight,
            "dense_prompt_unet_residual_loss": dense_prompt_unet_residual_loss,
            "dense_prompt_unet_residual_target_scale": dense_prompt_unet_residual_target_scale,
            "dense_prompt_unet_residual_target_mode": dense_prompt_unet_residual_target_mode,
            "dense_prompt_unet_residual_hard_threshold": dense_prompt_unet_residual_hard_threshold,
            "dense_prompt_unet_residual_use_gate": dense_prompt_unet_residual_use_gate,
            "dense_prompt_unet_residual_max_gt_area": dense_prompt_unet_residual_max_gt_area,
        },
        "training_config": {
            "img_size": img_size,
            "lr": lr,
            "lambda_focal": lambda_focal,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "weight_decay": weight_decay,
            "lambda_iou": lambda_iou,
            "scheduler_type": scheduler_type,
            "ema_decay": ema_decay,
            "resume_checkpoint": resume_checkpoint,
            "train_force_resize": train_force_resize,
            "val_force_resize": val_force_resize,
            "val_manifest": val_manifest,
            "coarse_loss_weight": coarse_loss_weight,
            "coarse_dice_weight": coarse_dice_weight,
            "coarse_prompt_loss_weight": coarse_prompt_loss_weight,
            "coarse_prompt_dice_weight": coarse_prompt_dice_weight,
            "coarse_prompt_false_negative_weight": coarse_prompt_false_negative_weight,
            "coarse_prompt_false_positive_weight": coarse_prompt_false_positive_weight,
            "coarse_prompt_supervision_source": coarse_prompt_supervision_source,
            "coarse_prompt_supervision_max_gt_area": coarse_prompt_supervision_max_gt_area,
            "prompt_gate_supervision_weight": prompt_gate_supervision_weight,
            "prompt_gate_target_source": prompt_gate_target_source,
            "prompt_gate_target_mode": prompt_gate_target_mode,
            "prompt_gate_sources": prompt_gate_sources,
            "prompt_gate_loss": prompt_gate_loss,
            "dual_branch_prompt_gate_supervision_weight": dual_branch_prompt_gate_supervision_weight,
            "dual_branch_prompt_gate_target_source": dual_branch_prompt_gate_target_source,
            "dual_branch_prompt_gate_target_mode": dual_branch_prompt_gate_target_mode,
            "dual_branch_prompt_gate_hard_threshold": dual_branch_prompt_gate_hard_threshold,
            "dual_branch_prompt_gate_loss": dual_branch_prompt_gate_loss,
            "dual_branch_prompt_residual_supervision_weight": dual_branch_prompt_residual_supervision_weight,
            "dual_branch_prompt_residual_target_source": dual_branch_prompt_residual_target_source,
            "dual_branch_prompt_residual_loss": dual_branch_prompt_residual_loss,
            "dual_branch_prompt_residual_target_scale": dual_branch_prompt_residual_target_scale,
            "dual_branch_prompt_residual_target_mode": dual_branch_prompt_residual_target_mode,
            "dual_branch_prompt_residual_hard_threshold": dual_branch_prompt_residual_hard_threshold,
            "dual_prompt_fusion_supervision_weight": dual_prompt_fusion_supervision_weight,
            "dual_prompt_fusion_oracle_metric": dual_prompt_fusion_oracle_metric,
            "dual_prompt_fusion_loss": dual_prompt_fusion_loss,
            "adapter_delta_reg_weight": adapter_delta_reg_weight,
            "coarse_prompt_calibrator_reg_weight": coarse_prompt_calibrator_reg_weight,
            "prompt_bias_supervision_weight": prompt_bias_supervision_weight,
            "prompt_bias_supervision_loss": prompt_bias_supervision_loss,
            "prompt_bias_supervision_max_delta_bias": prompt_bias_supervision_max_delta_bias,
            "final_logit_calibrator_supervision_weight": final_logit_calibrator_supervision_weight,
            "final_logit_calibrator_supervision_loss": final_logit_calibrator_supervision_loss,
            "final_logit_calibrator_supervision_max_delta_bias": final_logit_calibrator_supervision_max_delta_bias,
            "final_logit_calibrator_oracle_thresholds": final_logit_oracle_thresholds,
            "final_logit_calibrator_oracle_min_threshold": final_logit_calibrator_oracle_min_threshold,
            "final_logit_calibrator_oracle_max_threshold": final_logit_calibrator_oracle_max_threshold,
            "final_logit_calibrator_oracle_false_positive_penalty": final_logit_calibrator_oracle_false_positive_penalty,
            "final_logit_calibrator_oracle_area_penalty": final_logit_calibrator_oracle_area_penalty,
            "final_logit_spatial_supervision_weight": final_logit_spatial_supervision_weight,
            "final_logit_spatial_supervision_loss": final_logit_spatial_supervision_loss,
            "final_logit_spatial_supervision_target_scale": final_logit_spatial_supervision_target_scale,
            "final_logit_spatial_supervision_target_mode": final_logit_spatial_supervision_target_mode,
            "final_logit_spatial_supervision_hard_threshold": final_logit_spatial_supervision_hard_threshold,
            "dense_prompt_residual_reg_weight": dense_prompt_residual_reg_weight,
            "dense_prompt_signed_residual_supervision_weight": dense_prompt_signed_residual_supervision_weight,
            "dense_prompt_signed_residual_target_source": dense_prompt_signed_residual_target_source,
            "dense_prompt_signed_residual_loss": dense_prompt_signed_residual_loss,
            "dense_prompt_signed_residual_target_scale": dense_prompt_signed_residual_target_scale,
            "dense_prompt_signed_residual_target_mode": dense_prompt_signed_residual_target_mode,
            "dense_prompt_signed_residual_hard_threshold": dense_prompt_signed_residual_hard_threshold,
            "dense_prompt_signed_residual_use_gate": dense_prompt_signed_residual_use_gate,
            "dense_prompt_signed_residual_max_gt_area": dense_prompt_signed_residual_max_gt_area,
            "dense_prompt_unet_identity_weight": dense_prompt_unet_identity_weight,
            "dense_prompt_unet_identity_loss": dense_prompt_unet_identity_loss,
            "dense_prompt_unet_residual_supervision_weight": dense_prompt_unet_residual_supervision_weight,
            "dense_prompt_unet_residual_loss": dense_prompt_unet_residual_loss,
            "dense_prompt_unet_residual_target_scale": dense_prompt_unet_residual_target_scale,
            "dense_prompt_unet_residual_target_mode": dense_prompt_unet_residual_target_mode,
            "dense_prompt_unet_residual_hard_threshold": dense_prompt_unet_residual_hard_threshold,
            "dense_prompt_unet_residual_use_gate": dense_prompt_unet_residual_use_gate,
            "dense_prompt_unet_residual_max_gt_area": dense_prompt_unet_residual_max_gt_area,
            "coarse_prompt_calibrator_lr_multiplier": coarse_prompt_calibrator_lr_multiplier,
            "area_reg_weight": area_reg_weight,
            "area_reg_target_source": area_reg_target_source,
            "area_reg_loss": area_reg_loss,
            "area_reg_apply_to": area_reg_apply_to,
            "area_reg_constant": area_reg_constant,
            "area_reg_max_gt_area": area_reg_max_gt_area,
            "resume_use_ema_weights": resume_use_ema_weights,
            "reset_best_score": reset_best_score,
            "reload_diagnostics_batches": reload_diagnostics_batches,
            "checkpoint_save_mode": checkpoint_save_mode,
        },
        "data_config": {
            "authentic_ratio": authentic_ratio,
        "authentic_source_dir": authentic_source_dir,
        },
        "sam_config": {
            "sam_backend": sam_backend,
            "sam_config_file": sam_config,
            "sam_checkpoint": sam_ckpt,
        }
    }
    
    # Save parameters to JSON file
    params_save_path = os.path.join(experiment_dir, "model_params.json")
    with open(params_save_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    logger.info("Model parameters saved to %s", params_save_path)
    
    # Update save path to be under experiment directory
    model_save_path = os.path.join(experiment_dir, "best_model.pth")

    # Load dataset configuration and create datasets
    try:
        # Use default config path if not specified
        if dataset_config is None:
            dataset_config = os.path.join(os.path.dirname(__file__), "configs/datasets_default.json")
            
        if os.path.exists(dataset_config):
            dataset_manager_config = load_dataset_config(dataset_config)
            logger.info("Loaded dataset configuration from %s", dataset_config)
        else:
            logger.warning("Dataset config file %s not found, using default configuration", dataset_config)
            dataset_manager_config = create_default_config()
    except Exception as e:
        logger.warning("Failed to load dataset config: %s. Using default configuration.", e)
        dataset_manager_config = create_default_config()
    
    # Override batch size from command line args
    dataset_manager_config.train_loader.batch_size = args.batch_size
    
    dataset_manager = DatasetManager(dataset_manager_config)
    
    # Create training datasets
    train_dataset, train_loader = dataset_manager.create_train_datasets(
        img_size=img_size,
        contrastive_blur=False,
        perturbation_type="gaussian_blur/gaussian_noise",
        perturbation_intensity=0.75,
        authentic_ratio=authentic_ratio,
        force_resize=train_force_resize,
    )
    
    validation_manifest = load_validation_manifest(val_manifest) if val_manifest else None
    if validation_manifest is not None:
        logger.info("Loaded fixed validation manifest from %s", val_manifest)

    # Create validation datasets
    val_dataset, val_loader = dataset_manager.create_val_datasets(
        img_size=img_size,
        contrastive_blur=False,
        perturbation_type="gaussian_blur/gaussian_noise",
        perturbation_intensity=0.75,
        authentic_ratio=authentic_ratio,
        force_resize=val_force_resize,
        validation_manifest=validation_manifest,
    )
    
    # Log dataset information
    logger.info(
        "Training samples=%s validation_samples=%s",
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
    )
    checkpoint_data: Optional[Dict[str, Any]] = None
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            logger.info("Loading checkpoint from %s", resume_checkpoint)
            # Keep resumed checkpoints on CPU until tensors are selectively
            # copied into the model.  Loading a full SAM checkpoint directly
            # to CUDA duplicates several GB of weights before model
            # construction and makes concurrent diagnostics unnecessarily
            # fragile.
            checkpoint_data = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            logger.info(
                "Checkpoint metadata | epoch=%s score=%s",
                checkpoint_data.get("epoch"),
                checkpoint_data.get("score"),
            )
        else:
            logger.warning("Checkpoint %s not found; starting from scratch", resume_checkpoint)
    # Initialize Ferret-SAM model
    model = ForgeryLocalizer(
        sam_config=sam_config,
        sam_checkpoint=sam_ckpt,
        sam_backend=sam_backend,
        prompt_dim=prompt_dim,
        output_resolution=(img_size, img_size),
        downscale=downscale,
        train_sam_iou=train_sam_iou,
        dropout_rate=dropout_rate,
        lad_tau=lad_tau,
        lad_multi_taus=parsed_lad_multi_taus,
        forensic_operator=forensic_operator,
        use_detection_probe=use_detection_probe,  # Enable based on authentic_ratio
        adapter_residual_scale=adapter_residual_scale,
        adapter_type=adapter_type,
        adapter_active_scales=active_adapter_scales,
        adapter_gamma_init=adapter_gamma_init,
        adapter_sample_gate=adapter_sample_gate,
        adapter_sample_gate_scales=active_adapter_sample_gate_scales,
        adapter_sample_gate_max_delta=adapter_sample_gate_max_delta,
        adapter_forensic_source=adapter_forensic_source,
        adapter_diagnostics=adapter_diagnostics or adapter_delta_reg_weight > 0.0,
        sam3_prompt_mode=sam3_prompt_mode,
        coarse_prompt_transform=coarse_prompt_transform,
        coarse_prompt_scale=coarse_prompt_scale,
        coarse_prompt_bias=coarse_prompt_bias,
        coarse_prompt_calibrator=coarse_prompt_calibrator,
        coarse_prompt_calibrator_hidden=coarse_prompt_calibrator_hidden,
        coarse_prompt_calibrator_max_delta_scale=coarse_prompt_calibrator_max_delta_scale,
        coarse_prompt_calibrator_max_delta_bias=coarse_prompt_calibrator_max_delta_bias,
        final_logit_calibrator=final_logit_calibrator,
        final_logit_calibrator_hidden=final_logit_calibrator_hidden,
        final_logit_calibrator_max_delta_scale=final_logit_calibrator_max_delta_scale,
        final_logit_calibrator_max_delta_bias=final_logit_calibrator_max_delta_bias,
        coarse_prompt_refiner=coarse_prompt_refiner,
        coarse_prompt_refiner_hidden=coarse_prompt_refiner_hidden,
        coarse_prompt_refiner_max_residual=coarse_prompt_refiner_max_residual,
        coarse_prompt_refiner_gate_init=coarse_prompt_refiner_gate_init,
        coarse_prompt_refiner_gate_max=coarse_prompt_refiner_gate_max,
        coarse_prompt_refiner_precision_bias=coarse_prompt_refiner_precision_bias,
        coarse_prompt_refiner_recall_bias=coarse_prompt_refiner_recall_bias,
        coarse_prompt_head=coarse_prompt_head,
        coarse_prompt_hidden=coarse_prompt_hidden,
        coarse_prompt_dropout=coarse_prompt_dropout,
        coarse_prompt_gate_init=coarse_prompt_gate_init,
        coarse_prompt_gate_max=coarse_prompt_gate_max,
        coarse_prompt_area_bias=coarse_prompt_area_bias,
        coarse_prompt_signed_residual_max_delta=coarse_prompt_signed_residual_max_delta,
        coarse_prompt_unet_gate_init=coarse_prompt_unet_gate_init,
        coarse_prompt_unet_gate_max=coarse_prompt_unet_gate_max,
        coarse_prompt_unet_signed_residual_max_delta=coarse_prompt_unet_signed_residual_max_delta,
    ).to(device)
    
    # Load pretrained FerretNet weights if requested
    if args.load_ferret_weights:
        logger.info("Loading pretrained FerretNet weights from %s", args.ferret_weights_path)
        model.ferret_backbone.load_pretrained_weights(args.ferret_weights_path)
    
    if checkpoint_data and "model" in checkpoint_data:
        load_report = load_matching_state_dict(model, checkpoint_data["model"])
        logger.info("Loaded %s matching checkpoint tensors into model.", load_report["loaded"])
        if load_report["missing_keys"]:
            logger.warning("Missing keys when loading checkpoint: %s", load_report["missing_keys"])
        if load_report["unexpected_keys"]:
            logger.warning("Unexpected keys when loading checkpoint: %s", load_report["unexpected_keys"])
        if load_report["skipped_shape_mismatch"]:
            logger.warning(
                "Skipped shape-mismatched checkpoint keys: %s",
                load_report["skipped_shape_mismatch"],
            )
        if load_report.get("adapted_shape_mismatch"):
            logger.info(
                "Adapted shape-mismatched checkpoint keys: %s",
                load_report["adapted_shape_mismatch"],
            )
        if resume_use_ema_weights and "ema" in checkpoint_data:
            loaded_ema = apply_ema_weights_to_model(model, checkpoint_data["ema"])
            logger.info("Loaded %s matching EMA tensors into model before fine-tuning.", loaded_ema)

    # Freeze SAM components (only train adapters)
    model.encoder.eval()
    model.decoder.eval()
    model.sam_prompt_encoder.eval()
    
    # Freeze only the preloaded parts of FerretBackbone, leave MLDC trainable
    # model.ferret_backbone.eval()
    
    # # Fix running_var values in FerretBackbone after setting to eval mode
    # print(f"\n=== Post-Initialization RunningVar Fix ===")
    # fixed_count = 0
    # for name, module in model.ferret_backbone.named_modules():
    #     if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'running_var'):
    #         # Check if any running_var is too small
    #         if (module.running_var < 1e-6).any():
    #             # Set minimum value to 1e-6
    #             module.running_var = torch.clamp(module.running_var, min=1e-6)
    #             fixed_count += 1
    #             print(f"  Fixed running_var in {name}")
    # if fixed_count > 0:
    #     print(f"  Total fixed: {fixed_count} BatchNorm2d layers")
    # else:
    #     print(f"  No running_var values needed fixing")
    
    # Recalibrate BN statistics using current dataset
    # def recalibrate_bn(model, dataloader, device, num_batches=50):
    #     """
    #     仅用于修复冻结层中损坏的 BN 统计量。
    #     """
    #     print("\n=== 开始重新校准 BatchNorm 统计量 ===")
        
    #     # 1. 强制让 Backbone 处于 Train 模式
    #     # 这会开启 BN 层的统计量更新 (track_running_stats=True)
    #     model.ferret_backbone.train()
        
    #     # 2. 确保不需要梯度（因为我们只更新统计量，不更新权重）
    #     model.ferret_backbone.requires_grad_(False)
        
    #     with torch.no_grad():
    #         for i, batch in enumerate(dataloader):
    #             if i >= num_batches: break # 跑几十个 Batch 就足够收敛了
                
    #             # 3. 准备数据
    #             images = batch['orig'].to(device, non_blocking=True)
                
    #             # 4. 前向传播
    #             # 这一步会自动更新 model.ferret_backbone 里所有 BN 层的 running_mean/var
    #             _ = model.ferret_backbone(images)
                
    #             if i % 10 == 0:
    #                 print(f"  校准进度: {i}/{num_batches}")
    
    #     # 5. 校准完成，切回 Eval 模式（冻结行为）
    #     model.ferret_backbone.eval()
    #     print("=== 校准完成！BN 统计量已修复为当前数据的真实分布 ===")
    
    # # 使用当前的 train_loader 校准 BN 统计量
    # recalibrate_bn(model, train_loader, device, num_batches=50)
    
    # # Freeze all parameters first
    # for p in model.ferret_backbone.parameters():
    #     p.requires_grad = False
    
    # # Make MLDC parameters trainable since they're not from preloaded weights
    # for p in model.ferret_backbone.mldc.parameters():
    #     p.requires_grad = True

    # Setup trainable parameters
    trainable_params = collect_trainable_params(
        model,
        freeze_adapters=freeze_adapters,
        freeze_ferret=freeze_ferret,
        freeze_ferret_unet_only=freeze_ferret_unet_only,
    )
            
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(
            model,
            trainable_params,
            base_lr=lr,
            prompt_calibrator_lr_multiplier=coarse_prompt_calibrator_lr_multiplier,
            coarse_prompt_head_lr_multiplier=coarse_prompt_head_lr_multiplier,
        ),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Create learning rate scheduler
    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            div_factor=25,  # Increased div_factor for more gradual warmup
            final_div_factor=100,  # Increased final_div_factor for smoother decay
            pct_start=0.3,  # Increased warmup to 30% of epochs
            anneal_strategy="cos",
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=lr * 0.001  # Lower minimum learning rate
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Initialize EMA
    ema = EMA(model, decay=ema_decay)
    if checkpoint_data and "ema" in checkpoint_data:
        ema.load_state_dict(checkpoint_data["ema"])
        logger.info("Loaded EMA weights from checkpoint.")

    # Initialize training state
    best_score = 0.0 if reset_best_score else (checkpoint_data.get("score", 0.0) if checkpoint_data else 0.0)
    if reset_best_score and checkpoint_data:
        logger.info("Reset best_score to 0.0 for resumed run checkpoint selection.")
    start_epoch = checkpoint_data.get("epoch", 0) + 1 if checkpoint_data else 1

    # Initialize metrics history for tracking
    metrics_history = {
        'train_losses': [], 'val_ious': [], 'val_f1s': [],
        'small_iou': [], 'medium_iou': [], 'large_iou': [],
        'small_f1': [], 'medium_f1': [], 'large_f1': [],
        'small_count': [], 'medium_count': [], 'large_count': [],
        'dataset_metrics': {},
        'dataset_size_metrics': {},  # NEW: Add dataset-specific size metrics
        'train_iou': [], 'val_iou_epoch': [], 'iou_divergence': []
    }

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        loss_list, val_metrics_list, best_score, train_ious = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            val_loader=val_loader,
            validate_fn=validate,
            epoch=epoch,
            total_epochs=args.epochs,
            val_steps=val_steps,
            lambda_focal=lambda_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            model_save_path=model_save_path,
            best_score=best_score,
            ema=ema,
            lambda_iou=lambda_iou,
            train_sam_iou=train_sam_iou,
            model_params=model_params,
            authentic_ratio=authentic_ratio,  # Pass authentic_ratio
            lambda_detection=lambda_detection,  # Pass detection loss weight
            coarse_loss_weight=coarse_loss_weight,
            coarse_dice_weight=coarse_dice_weight,
            adapter_delta_reg_weight=adapter_delta_reg_weight,
            coarse_prompt_calibrator_reg_weight=coarse_prompt_calibrator_reg_weight,
            prompt_bias_supervision_weight=prompt_bias_supervision_weight,
            prompt_bias_supervision_loss=prompt_bias_supervision_loss,
            prompt_bias_supervision_max_delta_bias=prompt_bias_supervision_max_delta_bias,
            final_logit_calibrator_supervision_weight=final_logit_calibrator_supervision_weight,
            final_logit_calibrator_supervision_loss=final_logit_calibrator_supervision_loss,
            final_logit_calibrator_supervision_max_delta_bias=final_logit_calibrator_supervision_max_delta_bias,
            final_logit_calibrator_oracle_thresholds=final_logit_oracle_thresholds,
            final_logit_calibrator_oracle_min_threshold=final_logit_calibrator_oracle_min_threshold,
            final_logit_calibrator_oracle_max_threshold=final_logit_calibrator_oracle_max_threshold,
            final_logit_calibrator_oracle_false_positive_penalty=final_logit_calibrator_oracle_false_positive_penalty,
            final_logit_calibrator_oracle_area_penalty=final_logit_calibrator_oracle_area_penalty,
            final_logit_spatial_supervision_weight=final_logit_spatial_supervision_weight,
            final_logit_spatial_supervision_loss=final_logit_spatial_supervision_loss,
            final_logit_spatial_supervision_target_scale=final_logit_spatial_supervision_target_scale,
            final_logit_spatial_supervision_target_mode=final_logit_spatial_supervision_target_mode,
            final_logit_spatial_supervision_hard_threshold=final_logit_spatial_supervision_hard_threshold,
            dense_prompt_residual_reg_weight=dense_prompt_residual_reg_weight,
            dense_prompt_signed_residual_supervision_weight=dense_prompt_signed_residual_supervision_weight,
            dense_prompt_signed_residual_target_source=dense_prompt_signed_residual_target_source,
            dense_prompt_signed_residual_loss=dense_prompt_signed_residual_loss,
            dense_prompt_signed_residual_target_scale=dense_prompt_signed_residual_target_scale,
            dense_prompt_signed_residual_target_mode=dense_prompt_signed_residual_target_mode,
            dense_prompt_signed_residual_hard_threshold=dense_prompt_signed_residual_hard_threshold,
            dense_prompt_signed_residual_use_gate=dense_prompt_signed_residual_use_gate,
            dense_prompt_signed_residual_max_gt_area=dense_prompt_signed_residual_max_gt_area,
            dense_prompt_unet_identity_weight=dense_prompt_unet_identity_weight,
            dense_prompt_unet_identity_loss=dense_prompt_unet_identity_loss,
            dense_prompt_unet_residual_supervision_weight=dense_prompt_unet_residual_supervision_weight,
            dense_prompt_unet_residual_loss=dense_prompt_unet_residual_loss,
            dense_prompt_unet_residual_target_scale=dense_prompt_unet_residual_target_scale,
            dense_prompt_unet_residual_target_mode=dense_prompt_unet_residual_target_mode,
            dense_prompt_unet_residual_hard_threshold=dense_prompt_unet_residual_hard_threshold,
            dense_prompt_unet_residual_use_gate=dense_prompt_unet_residual_use_gate,
            dense_prompt_unet_residual_max_gt_area=dense_prompt_unet_residual_max_gt_area,
            coarse_prompt_loss_weight=coarse_prompt_loss_weight,
            coarse_prompt_dice_weight=coarse_prompt_dice_weight,
            coarse_prompt_false_negative_weight=coarse_prompt_false_negative_weight,
            coarse_prompt_false_positive_weight=coarse_prompt_false_positive_weight,
            coarse_prompt_supervision_source=coarse_prompt_supervision_source,
            coarse_prompt_supervision_max_gt_area=coarse_prompt_supervision_max_gt_area,
            prompt_gate_supervision_weight=prompt_gate_supervision_weight,
            prompt_gate_target_source=prompt_gate_target_source,
            prompt_gate_target_mode=prompt_gate_target_mode,
            prompt_gate_sources=prompt_gate_sources,
            prompt_gate_loss=prompt_gate_loss,
            dual_branch_prompt_gate_supervision_weight=dual_branch_prompt_gate_supervision_weight,
            dual_branch_prompt_gate_target_source=dual_branch_prompt_gate_target_source,
            dual_branch_prompt_gate_target_mode=dual_branch_prompt_gate_target_mode,
            dual_branch_prompt_gate_hard_threshold=dual_branch_prompt_gate_hard_threshold,
            dual_branch_prompt_gate_loss=dual_branch_prompt_gate_loss,
            dual_branch_prompt_residual_supervision_weight=dual_branch_prompt_residual_supervision_weight,
            dual_branch_prompt_residual_target_source=dual_branch_prompt_residual_target_source,
            dual_branch_prompt_residual_loss=dual_branch_prompt_residual_loss,
            dual_branch_prompt_residual_target_scale=dual_branch_prompt_residual_target_scale,
            dual_branch_prompt_residual_target_mode=dual_branch_prompt_residual_target_mode,
            dual_branch_prompt_residual_hard_threshold=dual_branch_prompt_residual_hard_threshold,
            dual_prompt_fusion_supervision_weight=dual_prompt_fusion_supervision_weight,
            dual_prompt_fusion_oracle_metric=dual_prompt_fusion_oracle_metric,
            dual_prompt_fusion_loss=dual_prompt_fusion_loss,
            prompt_gate_refiner_max=coarse_prompt_refiner_gate_max,
            prompt_gate_dense_max=coarse_prompt_gate_max,
            area_reg_weight=area_reg_weight,
            area_reg_target_source=area_reg_target_source,
            area_reg_loss=area_reg_loss,
            area_reg_apply_to=area_reg_apply_to,
            area_reg_constant=area_reg_constant,
            area_reg_max_gt_area=area_reg_max_gt_area,
            reload_diagnostics_batches=reload_diagnostics_batches,
            checkpoint_save_mode=checkpoint_save_mode,
        )
        
        # Update metrics history
        metrics_history['train_losses'].extend(loss_list)
        
        # Process validation metrics
        if val_metrics_list:
            epoch_metrics = aggregate_epoch_metrics(val_metrics_list)
            
            # Store overall metrics
            metrics_history['val_ious'].extend([vm["iou"] for vm in val_metrics_list])
            metrics_history['val_f1s'].extend([vm["f1"] for vm in val_metrics_list])
            
            # Store stratified metrics
            for size in ['small', 'medium', 'large']:
                for metric in ['iou', 'f1', 'count']:
                    key = f'{size}_{metric}'
                    metrics_history[key].append(epoch_metrics.get(key, 0.0))
            
            # Store dataset-specific metrics
            if 'dataset_metrics' in epoch_metrics:
                for dataset_name, dataset_result in epoch_metrics['dataset_metrics'].items():
                    if dataset_name not in metrics_history['dataset_metrics']:
                        metrics_history['dataset_metrics'][dataset_name] = {
                            'iou_history': [], 'f1_history': [], 'count_history': []
                        }
                    
                    metrics_history['dataset_metrics'][dataset_name]['iou_history'].append(dataset_result['iou'])
                    metrics_history['dataset_metrics'][dataset_name]['f1_history'].append(dataset_result['f1'])
                    metrics_history['dataset_metrics'][dataset_name]['count_history'].append(dataset_result['count'])
            
            # NEW: Store dataset-specific SIZE metrics
            if 'dataset_size_metrics' in epoch_metrics:
                for dataset_name, size_results in epoch_metrics['dataset_size_metrics'].items():
                    if dataset_name not in metrics_history['dataset_size_metrics']:
                        metrics_history['dataset_size_metrics'][dataset_name] = {
                            f'{size}_{metric}_history': []
                            for size in ['small', 'medium', 'large']
                            for metric in ['iou', 'f1', 'count']
                        }
                    
                    for size in ['small', 'medium', 'large']:
                        for metric in ['iou', 'f1', 'count']:
                            key = f'{size}_{metric}'
                            history_key = f'{size}_{metric}_history'
                            value = size_results.get(key, 0.0)
                            metrics_history['dataset_size_metrics'][dataset_name][history_key].append(value)
            
            # Handle IoU tracking
            if train_ious:
                epoch_train_iou = np.mean(train_ious)
                epoch_val_iou = epoch_metrics.get('overall_iou', 0.0)
                
                metrics_history['train_iou'].append(epoch_train_iou)
                metrics_history['val_iou_epoch'].append(epoch_val_iou)
                metrics_history['iou_divergence'].append(epoch_train_iou - epoch_val_iou)
                
                logger.info(
                    "Epoch %s IoU | train=%.4f val=%.4f divergence=%.4f",
                    epoch,
                    epoch_train_iou,
                    epoch_val_iou,
                    epoch_train_iou - epoch_val_iou,
                )
        
        # Generate metrics visualization
        logger.info(
            "Epoch %s completed. Saving metrics visualization at %s",
            epoch,
            os.path.join(experiment_dir, 'metrics.png'),
        )
        plot_all_metrics(
            metrics_history, 
            True, 
            save_path=os.path.join(experiment_dir, "metrics.png")
        )


def aggregate_epoch_metrics(val_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate validation metrics for one epoch.
    
    Args:
        val_metrics_list: List of validation metric dictionaries
        
    Returns:
        Dictionary containing aggregated metrics
    """
    aggregated = {'overall_iou': 0.0, 'overall_f1': 0.0}
    
    # Collect metrics by size
    size_metrics = {size: {'iou': [], 'f1': [], 'count': []} 
                   for size in ['small', 'medium', 'large']}
    
    # Collect dataset-specific metrics
    dataset_metrics = {}
    
    # NEW: Collect dataset-specific SIZE metrics
    dataset_size_metrics = {}
    
    for vm in val_metrics_list:
        aggregated['overall_iou'] += vm.get('iou', 0.0)
        aggregated['overall_f1'] += vm.get('f1', 0.0)
        
        # Collect size-specific metrics
        for size in ['small', 'medium', 'large']:
            if vm.get(f'{size}_count', 0) > 0:
                size_metrics[size]['iou'].append(vm[f'{size}_iou'])
                size_metrics[size]['f1'].append(vm[f'{size}_f1'])
                size_metrics[size]['count'].append(vm[f'{size}_count'])
        
        # Collect dataset-specific metrics
        if 'dataset_results' in vm:
            for dataset_name, dataset_result in vm['dataset_results'].items():
                if dataset_name not in dataset_metrics:
                    dataset_metrics[dataset_name] = {'iou': [], 'f1': [], 'count': []}
                dataset_metrics[dataset_name]['iou'].append(dataset_result['iou'])
                dataset_metrics[dataset_name]['f1'].append(dataset_result['f1'])
                dataset_metrics[dataset_name]['count'].append(dataset_result['count'])
                
                # NEW: Collect size-specific metrics per dataset
                if dataset_name not in dataset_size_metrics:
                    dataset_size_metrics[dataset_name] = {
                        f'{size}_{metric}': [] 
                        for size in ['small', 'medium', 'large'] 
                        for metric in ['iou', 'f1', 'count']
                    }
                
                # Add size-specific metrics if available
                for size in ['small', 'medium', 'large']:
                    for metric in ['iou', 'f1', 'count']:
                        key = f'{size}_{metric}'
                        if key in dataset_result:
                            dataset_size_metrics[dataset_name][key].append(dataset_result[key])
    
    # Average overall metrics
    n = len(val_metrics_list)
    aggregated['overall_iou'] /= n
    aggregated['overall_f1'] /= n
    
    # Average size-specific metrics
    for size in ['small', 'medium', 'large']:
        for metric in ['iou', 'f1', 'count']:
            values = size_metrics[size][metric]
            aggregated[f'{size}_{metric}'] = np.mean(values) if values else 0.0
    
    # Average dataset-specific metrics
    aggregated['dataset_metrics'] = {}
    for dataset_name, metrics in dataset_metrics.items():
        aggregated['dataset_metrics'][dataset_name] = {
            'iou': np.mean(metrics['iou']) if metrics['iou'] else 0.0,
            'f1': np.mean(metrics['f1']) if metrics['f1'] else 0.0,
            'count': np.mean(metrics['count']) if metrics['count'] else 0.0,
        }
    
    # NEW: Average dataset-specific SIZE metrics
    aggregated['dataset_size_metrics'] = {}
    for dataset_name, size_metrics in dataset_size_metrics.items():
        aggregated['dataset_size_metrics'][dataset_name] = {}
        for size in ['small', 'medium', 'large']:
            for metric in ['iou', 'f1', 'count']:
                key = f'{size}_{metric}'
                values = size_metrics[key]
                aggregated['dataset_size_metrics'][dataset_name][key] = (
                    np.mean(values) if values else 0.0
                )
    
    return aggregated


# Configuration constants
DATA_ROOT = "sam2data"
SAM_CONFIG = "sam2.1_hiera_b+.yaml"
SAM_CKPT = "sam2configs/sam2.1_hiera_base_plus.pt"
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-3
LAMBDA_FOCAL = 10.0
FOCAL_GAMMA = 1.0
FOCAL_ALPHA = 0.55
WEIGHT_DECAY = 0.0001
PROMPT_DIM = 64
SAVE_PATH = "results/sam2_forgery.pth"
DOWNSCALE = 16
LAMBDA_IOU = 0.25
DROPOUT_RATE = 0.2
SCHEDULER_TYPE = "cosine"
CONTRASTIVE_BLUR = False
PERTURBATION_TYPE = "gaussian_blur/gaussian_noise"  # Options: "gaussian_blur", "jpeg_compression", "gaussian_noise", "none", "gaussian_blur/gaussian_noise"
PERTURBATION_INTENSITY = 0.75
VAL_STEPS = 2000  # Number of training steps between validation runs


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="FLAME - Train forgery localization model using SAM2 architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument(
        "--data_root", type=str, default=DATA_ROOT, 
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--dataset_config", type=str, default=os.path.join(os.path.dirname(__file__), "configs/datasets_default.json"),
        help="Path to dataset configuration JSON file"
    )
    parser.add_argument(
        "--val_manifest", type=str, default=None,
        help="Optional fixed validation manifest JSON; overrides validation sample_ratio selections"
    )
    parser.add_argument(
        "--img_size", type=int, default=IMG_SIZE,
        help="Image size for training and validation"
    )
    parser.add_argument(
        "--train_force_resize", action="store_true",
        help="Resize training samples to img_size instead of using balanced crop"
    )
    parser.add_argument(
        "--val_force_resize", action="store_true",
        help="Resize validation samples to img_size instead of using balanced crop"
    )
    
    # SAM configuration
    parser.add_argument(
        "--sam_config", type=str, default=SAM_CONFIG, 
        help="SAM configuration file"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, default=SAM_CKPT, 
        help="SAM checkpoint file"
    )
    parser.add_argument(
        "--sam_backend", type=str, default="sam2", choices=["sam2", "sam3_interactive"],
        help="SAM backend to use"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=LR, 
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=WEIGHT_DECAY,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--scheduler_type", type=str, default=SCHEDULER_TYPE,
        choices=["onecycle", "cosine", "none"],
        help="Type of learning rate scheduler"
    )
    
    # Loss configuration
    parser.add_argument(
        "--lambda_focal", type=float, default=LAMBDA_FOCAL, 
        help="Weight for focal loss"
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=FOCAL_GAMMA, 
        help="Gamma parameter for focal loss"
    )
    parser.add_argument(
        "--focal_alpha", type=float, default=FOCAL_ALPHA, 
        help="Alpha parameter for focal loss"
    )
    parser.add_argument(
        "--lambda_iou", type=float, default=LAMBDA_IOU, 
        help="Weight for IoU prediction loss"
    )
    parser.add_argument(
        "--coarse_loss_weight",
        type=float,
        default=0.2,
        help="Weight for auxiliary coarse prompt BCE-with-logits loss",
    )
    parser.add_argument(
        "--coarse_dice_weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary coarse prompt Dice loss",
    )
    parser.add_argument(
        "--coarse_prompt_loss_weight",
        type=float,
        default=0.0,
        help="BCE-with-logits loss weight on calibrated dense prompt sent to SAM",
    )
    parser.add_argument(
        "--coarse_prompt_dice_weight",
        type=float,
        default=0.0,
        help="Dice loss weight on calibrated dense prompt sent to SAM",
    )
    parser.add_argument(
        "--coarse_prompt_false_negative_weight",
        type=float,
        default=0.0,
        help="Recall-preserving loss weight for foreground missed by calibrated dense prompt",
    )
    parser.add_argument(
        "--coarse_prompt_false_positive_weight",
        type=float,
        default=0.0,
        help="Precision-preserving loss weight for background covered by calibrated dense prompt",
    )
    parser.add_argument(
        "--coarse_prompt_supervision_source",
        type=str,
        default="coarse_prompt",
        choices=[
            "coarse_prompt",
            "dense_prompt",
            "coarse_mask",
            "pre_refiner_prompt",
            "final_logits",
            "raw_final_logits",
        ],
        help=(
            "Logit source supervised by --coarse_prompt_* losses. "
            "Use dense_prompt to teach the raw prompt before SAM3 center/bias."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_supervision_max_gt_area",
        type=float,
        default=None,
        help=(
            "If set, apply --coarse_prompt_* supervision only to samples whose "
            "GT foreground area ratio is at most this value."
        ),
    )
    parser.add_argument(
        "--prompt_gate_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for GT-derived FN/FP oracle supervision on prompt expansion gates.",
    )
    parser.add_argument(
        "--prompt_gate_target_source",
        type=str,
        default="pre_refiner_prompt",
        choices=[
            "pre_refiner_prompt",
            "coarse_prompt",
            "dense_prompt",
            "coarse_mask",
            "final_logits",
            "raw_final_logits",
        ],
        help="Prompt logits used to compute the GT-derived gate oracle target.",
    )
    parser.add_argument(
        "--prompt_gate_target_mode",
        type=str,
        default="fn_ratio",
        choices=["fn_ratio", "fp_ratio", "fp_pixel_hard", "pixel_hard_error", "error_area"],
        help=(
            "Gate oracle target. fn_ratio opens expansion gates for false negatives; "
            "fp_ratio opens conservative-routing gates for false positives; "
            "fp_pixel_hard gives a spatial hard false-positive target; "
            "pixel_hard_error gives a spatial hard false-positive/false-negative target; "
            "error_area opens signed correction gates for both false positives and false negatives."
        ),
    )
    parser.add_argument(
        "--prompt_gate_sources",
        type=str,
        default="refiner",
        help="Comma-separated gates to supervise: refiner,dense,small_dense,none.",
    )
    parser.add_argument(
        "--prompt_gate_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse", "balanced_smooth_l1", "balanced_l1", "balanced_mse"],
        help="Loss type for prompt gate oracle supervision.",
    )
    parser.add_argument(
        "--dual_branch_prompt_gate_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for separate FN/FP oracle supervision on dual-branch prompt FG/BG gates.",
    )
    parser.add_argument(
        "--dual_branch_prompt_gate_target_source",
        type=str,
        default="dense_prompt",
        choices=[
            "pre_refiner_prompt",
            "coarse_prompt",
            "dense_prompt",
            "coarse_mask",
            "final_logits",
            "raw_final_logits",
        ],
        help="Prompt logits used to compute separate dual-branch FG/BG gate targets.",
    )
    parser.add_argument(
        "--dual_branch_prompt_gate_target_mode",
        type=str,
        default="sample_ratio",
        choices=[
            "sample_ratio",
            "pixel_soft_error",
            "pixel_hard_error",
            "balanced_pixel_soft_error",
            "balanced_pixel_hard_error",
        ],
        help=(
            "Target mode for dual-branch prompt gate supervision. sample_ratio is the legacy "
            "per-sample FN/FP area ratio; pixel_* modes supervise local gate maps. "
            "balanced_* aliases use the same target maps and are intended for balanced losses."
        ),
    )
    parser.add_argument(
        "--dual_branch_prompt_gate_hard_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used by --dual_branch_prompt_gate_target_mode=pixel_hard_error.",
    )
    parser.add_argument(
        "--dual_branch_prompt_gate_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse", "balanced_smooth_l1", "balanced_l1", "balanced_mse"],
        help="Loss type for dual-branch prompt gate supervision.",
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for pixel-level false-negative/false-positive supervision on dual-branch residual maps.",
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_target_source",
        type=str,
        default="dense_prompt",
        choices=[
            "pre_refiner_prompt",
            "coarse_prompt",
            "dense_prompt",
            "coarse_mask",
            "final_logits",
            "raw_final_logits",
        ],
        help="Prompt logits used to compute pixel-level dual-branch residual targets.",
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse", "balanced_smooth_l1", "balanced_l1", "balanced_mse"],
        help="Loss type for dual-branch prompt residual supervision.",
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_target_scale",
        type=float,
        default=1.0,
        help="Scale factor for false-negative/false-positive target maps used by dual-branch residual supervision.",
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_target_mode",
        type=str,
        default="soft_error",
        choices=["soft_error", "hard_error"],
        help=(
            "Target map mode for dual-branch residual supervision. soft_error uses all "
            "probability mass; hard_error only supervises confident threshold-crossing "
            "FP/FN pixels to avoid global half-probability suppression."
        ),
    )
    parser.add_argument(
        "--dual_branch_prompt_residual_hard_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used by hard_error dual-branch residual targets.",
    )
    parser.add_argument(
        "--dual_prompt_fusion_supervision_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for training-only oracle supervision of sam3_prompt_mode=dual_gated "
            "legacy/native fusion gate."
        ),
    )
    parser.add_argument(
        "--dual_prompt_fusion_oracle_metric",
        type=str,
        default="iou",
        choices=["iou", "f1", "pixel_bce", "per_pixel_bce", "spatial_bce"],
        help=(
            "Oracle used to decide whether legacy or native branch is better. "
            "iou/f1 produce one target per sample; pixel_bce/per_pixel_bce/spatial_bce "
            "produce per-pixel targets for spatial fusion gates."
        ),
    )
    parser.add_argument(
        "--dual_prompt_fusion_loss",
        type=str,
        default="bce",
        choices=["bce", "smooth_l1", "l1", "mse"],
        help="Loss type for dual prompt fusion oracle supervision.",
    )
    parser.add_argument(
        "--area_reg_weight",
        type=float,
        default=0.0,
        help="Weight for prediction-area calibration regularization",
    )
    parser.add_argument(
        "--area_reg_target_source",
        type=str,
        default="gt",
        choices=["gt", "batch_gt", "constant"],
        help="Target area source for area regularization",
    )
    parser.add_argument(
        "--area_reg_loss",
        type=str,
        default="smooth_l1",
        choices=["l1", "smooth_l1"],
        help="Loss type for prediction-area calibration",
    )
    parser.add_argument(
        "--area_reg_apply_to",
        type=str,
        default="coarse",
        choices=[
            "coarse",
            "final",
            "both",
            "dense_prompt",
            "coarse_and_dense",
            "final_and_dense",
            "coarse_prompt",
            "final_and_prompt",
            "dense_and_prompt",
            "all",
        ],
        help="Which prediction area to regularize; dense_prompt targets the raw SAM dense prompt branch",
    )
    parser.add_argument(
        "--area_reg_constant",
        type=float,
        default=0.25,
        help="Constant target area ratio when --area_reg_target_source=constant",
    )
    parser.add_argument(
        "--area_reg_max_gt_area",
        type=float,
        default=None,
        help=(
            "Only apply area regularization to samples whose GT area ratio is <= this threshold; "
            "unset applies area regularization to all samples"
        ),
    )
    
    # Model configuration
    parser.add_argument(
        "--prompt_dim", type=int, default=PROMPT_DIM, 
        help="Dimension of SAM prompt embeddings"
    )
    parser.add_argument(
        "--downscale", type=int, default=DOWNSCALE,
        help="Downscale factor for mask adapter"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=DROPOUT_RATE,
        help="Dropout rate for adapters"
    )
    parser.add_argument(
        "--lad_tau",
        type=float,
        default=0.004,
        help="Tau for the LAD operator applied to raw RGB local patches",
    )
    parser.add_argument(
        "--lad_multi_taus",
        type=str,
        default=None,
        help="Comma-separated tau values for forensic_operator=lad_multi (default: 0.016,0.032,0.064,0.128)",
    )
    parser.add_argument(
        "--forensic_operator",
        type=str,
        default="lad",
        choices=["lad", "lad_multi", "mldc", "lad_mldc_hybrid"],
        help="Forensic preprocessing operator used by the Ferret backbone.",
    )
    parser.add_argument(
        "--adapter_residual_scale",
        type=float,
        default=1.0,
        help="Multiplier for residual adapter deltas before adding to SAM features",
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="shared",
        choices=["shared", "norm_gated"],
        help="Adapter architecture for SAM feature fusion",
    )
    parser.add_argument(
        "--adapter_scales",
        type=str,
        default=None,
        help="Comma-separated adapter scales to enable (0=image embedding, 1/2=high-res) or 'all'",
    )
    parser.add_argument(
        "--adapter_gamma_init",
        type=float,
        default=0.0,
        help="Initial per-scale gamma for gated adapter variants",
    )
    parser.add_argument(
        "--adapter_sample_gate",
        action="store_true",
        help="Enable a sample-conditioned multiplicative gate for norm_gated adapter deltas.",
    )
    parser.add_argument(
        "--adapter_sample_gate_scales",
        type=str,
        default=None,
        help="Comma-separated adapter scales whose deltas use the sample gate; defaults to all.",
    )
    parser.add_argument(
        "--adapter_sample_gate_max_delta",
        type=float,
        default=0.5,
        help="Maximum relative sample-gate deviation around 1.0 for norm_gated adapters.",
    )
    parser.add_argument(
        "--adapter_forensic_source",
        type=str,
        default="final",
        choices=["final", "final_plus_pyramid"],
        help=(
            "For norm_gated adapters, optionally add an identity-initialized "
            "pyramid adapter branch that consumes Ferret final/high/mid features."
        ),
    )
    parser.add_argument(
        "--adapter_delta_reg_weight",
        type=float,
        default=0.0,
        help="Weight for adapter delta MSE regularization",
    )
    parser.add_argument(
        "--adapter_diagnostics",
        action="store_true",
        help="Collect adapter per-scale diagnostics during training",
    )
    parser.add_argument(
        "--sam3_prompt_mode",
        type=str,
        default="legacy",
        choices=[
            "legacy",
            "native",
            "legacy_dummy_point",
            "native_resize_only",
            "dual_avg",
            "dual_gated",
            "dual_spatial_gated",
        ],
        help=(
            "SAM3 dense-prompt interface: 'legacy' preserves existing checkpoints; "
            "'native' adds SAM3's no-point sparse prompt and native mask resize; "
            "'legacy_dummy_point' adds only SAM3's no-point sparse prompt; "
            "'native_resize_only' uses SAM3 prompt-encoder mask size without the dummy point; "
            "'dual_avg' averages legacy/native decoder logits; "
            "'dual_gated' learns a sample-conditioned legacy/native fusion gate; "
            "'dual_spatial_gated' learns a pixel-conditioned legacy/native fusion gate"
        ),
    )
    parser.add_argument(
        "--coarse_prompt_transform",
        type=str,
        default="none",
        choices=["none", "center", "zscore"],
        help="Optional per-sample transform for dense coarse prompts before SAM prompt encoder",
    )
    parser.add_argument(
        "--coarse_prompt_scale",
        type=float,
        default=1.0,
        help="Scale applied after coarse prompt transform",
    )
    parser.add_argument(
        "--coarse_prompt_bias",
        type=float,
        default=0.0,
        help="Bias applied after coarse prompt transform",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator",
        type=str,
        default="none",
        choices=["none", "stats_mlp", "context_stats_mlp"],
        help="Optional trainable sample-conditioned dense prompt calibrator",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator_hidden",
        type=int,
        default=16,
        help="Hidden dimension for --coarse_prompt_calibrator=stats_mlp",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator_max_delta_scale",
        type=float,
        default=1.0,
        help="Maximum absolute log-scale delta for trainable prompt calibrators",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator_max_delta_bias",
        type=float,
        default=2.0,
        help="Maximum absolute bias delta for trainable prompt calibrators",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator_reg_weight",
        type=float,
        default=0.0,
        help="Regularize dynamic prompt calibrator scale/bias toward identity",
    )
    parser.add_argument(
        "--coarse_prompt_calibrator_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for prompt calibrator parameters",
    )
    parser.add_argument(
        "--prompt_bias_supervision_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for training-only oracle supervision of the dynamic prompt "
            "calibrator bias using GT-area quantile targets."
        ),
    )
    parser.add_argument(
        "--prompt_bias_supervision_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Loss for --prompt_bias_supervision_weight.",
    )
    parser.add_argument(
        "--prompt_bias_supervision_max_delta_bias",
        type=float,
        default=None,
        help=(
            "Optional clamp for the oracle bias target; normally match "
            "--coarse_prompt_calibrator_max_delta_bias."
        ),
    )
    parser.add_argument(
        "--final_logit_calibrator",
        type=str,
        default="none",
        choices=["none", "stats_mlp", "context_stats_mlp", "quantile_mlp", "semantic_spatial_cnn"],
        help="Optional trainable calibrator/refiner for final mask logits.",
    )
    parser.add_argument(
        "--final_logit_calibrator_hidden",
        type=int,
        default=16,
        help="Hidden dimension for --final_logit_calibrator.",
    )
    parser.add_argument(
        "--final_logit_calibrator_max_delta_scale",
        type=float,
        default=0.0,
        help="Maximum absolute log-scale delta for final logit calibrator.",
    )
    parser.add_argument(
        "--final_logit_calibrator_max_delta_bias",
        type=float,
        default=1.0,
        help="Maximum absolute additive bias delta for final logit calibrator.",
    )
    parser.add_argument(
        "--final_logit_calibrator_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for training-only final-logit threshold-oracle bias supervision.",
    )
    parser.add_argument(
        "--final_logit_calibrator_supervision_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Loss type for final-logit calibrator oracle supervision.",
    )
    parser.add_argument(
        "--final_logit_calibrator_supervision_max_delta_bias",
        type=float,
        default=None,
        help="Optional clamp for final-logit oracle bias target; normally match --final_logit_calibrator_max_delta_bias.",
    )
    parser.add_argument(
        "--final_logit_calibrator_oracle_thresholds",
        type=str,
        default=None,
        help="Optional comma-separated probability thresholds for final-logit oracle supervision.",
    )
    parser.add_argument(
        "--final_logit_calibrator_oracle_min_threshold",
        type=float,
        default=None,
        help="Optional lower bound applied to final-logit oracle threshold candidates.",
    )
    parser.add_argument(
        "--final_logit_calibrator_oracle_max_threshold",
        type=float,
        default=None,
        help="Optional upper bound applied to final-logit oracle threshold candidates.",
    )
    parser.add_argument(
        "--final_logit_calibrator_oracle_false_positive_penalty",
        type=float,
        default=0.0,
        help="Penalty weight for false-positive area when selecting the final-logit oracle threshold.",
    )
    parser.add_argument(
        "--final_logit_calibrator_oracle_area_penalty",
        type=float,
        default=0.0,
        help="Penalty weight for |predicted area - target area| when selecting the final-logit oracle threshold.",
    )
    parser.add_argument(
        "--final_logit_spatial_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for per-pixel signed FP/FN supervision on spatial final-logit calibrators.",
    )
    parser.add_argument(
        "--final_logit_spatial_supervision_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Loss type for final-logit spatial error supervision.",
    )
    parser.add_argument(
        "--final_logit_spatial_supervision_target_scale",
        type=float,
        default=1.0,
        help="Scale factor for signed spatial final-logit FP/FN correction targets.",
    )
    parser.add_argument(
        "--final_logit_spatial_supervision_target_mode",
        type=str,
        default="hard_error",
        choices=["soft_error", "hard_error"],
        help=(
            "Target map mode for final-logit spatial supervision. hard_error only supervises "
            "threshold-crossing FP/FN pixels; soft_error uses all probability residual mass."
        ),
    )
    parser.add_argument(
        "--final_logit_spatial_supervision_hard_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used by hard_error final-logit spatial supervision.",
    )
    parser.add_argument(
        "--coarse_prompt_refiner",
        type=str,
        default="none",
        choices=[
            "none",
            "residual_cnn",
            "context_residual_cnn",
            "gated_context_residual_cnn",
            "spatial_context_residual_cnn",
            "raw_blend_context_cnn",
            "precision_recall_endpoint_context_cnn",
            "transform_router_context_cnn",
            "learned_precision_recall_endpoint_context_cnn",
            "teacher_oracle_endpoint_context_cnn",
            "post_dual_branch_context_cnn",
            "feature_guided_post_dual_branch_context_cnn",
            "semantic_guided_post_dual_branch_context_cnn",
        ],
        help="Optional identity-initialized spatial residual refiner for dense prompts",
    )
    parser.add_argument(
        "--coarse_prompt_refiner_hidden",
        type=int,
        default=8,
        help="Hidden channels for --coarse_prompt_refiner=residual_cnn",
    )
    parser.add_argument(
        "--coarse_prompt_refiner_max_residual",
        type=float,
        default=1.0,
        help="Maximum absolute spatial residual for contextual dense-prompt refiners",
    )
    parser.add_argument(
        "--coarse_prompt_refiner_gate_init",
        type=float,
        default=0.2,
        help=(
            "Initial sigmoid gate fraction for --coarse_prompt_refiner=gated_context_residual_cnn; "
            "effective residual multiplier starts at gate_init * gate_max."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_refiner_gate_max",
        type=float,
        default=1.0,
        help="Maximum per-sample residual gate for --coarse_prompt_refiner=gated_context_residual_cnn.",
    )
    parser.add_argument(
        "--coarse_prompt_refiner_precision_bias",
        type=float,
        default=-0.10,
        help=(
            "Precision endpoint bias for --coarse_prompt_refiner=precision_recall_endpoint_context_cnn; "
            "conservative transform bias for --coarse_prompt_refiner=transform_router_context_cnn."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_refiner_recall_bias",
        type=float,
        default=0.45,
        help="Recall endpoint bias for --coarse_prompt_refiner=precision_recall_endpoint_context_cnn.",
    )
    parser.add_argument(
        "--coarse_prompt_head",
        type=str,
        default="mask_compressor",
        choices=[
            "mask_compressor",
            "multiscale",
            "split_multiscale",
            "gated_split_multiscale",
            "gated_split_multiscale_highres",
            "adaptive_tau_fusion_multiscale",
            "dual_branch_multiscale",
            "signed_tribranch_multiscale",
            "signed_tribranch_multiscale_highres",
            "detail_guided_signed_tribranch_multiscale",
            "adaptive_detail_guided_signed_tribranch_multiscale",
            "precision_recall_adaptive_prompt_head",
            "uncertainty_guided_precision_recall_prompt_head",
            "contextual_highres_precision_recall_prompt_head",
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
            "unet_residual_only_prompt_head",
        ],
        help=(
            "Coarse dense-prompt generator head. 'multiscale' changes both supervised coarse "
            "and SAM prompt logits; 'split_multiscale' keeps supervised coarse logits on "
            "mask_compressor and applies the residual only to the SAM dense prompt; "
            "'gated_split_multiscale' adds a sample-conditioned residual gate; "
            "'adaptive_tau_fusion_multiscale' adds identity-initialized spatial tau fusion "
            "before the gated multiscale dense-prompt head; "
            "'dual_branch_multiscale' separates foreground expansion and background suppression; "
            "'signed_tribranch_multiscale' adds spatial core/foreground/background prompt branches; "
            "'signed_tribranch_multiscale_highres' keeps those branches at high-feature resolution; "
            "'detail_guided_signed_tribranch_multiscale' injects zero-initialized LAD-detail "
            "context into the signed tribranch generator so resumed checkpoints are unperturbed; "
            "'adaptive_detail_guided_signed_tribranch_multiscale' additionally adds identity "
            "multi-tau LAD fusion before the Ferret stem; "
            "'precision_recall_adaptive_prompt_head' adds identity-initialized multi-tau/detail "
            "features plus separate local precision suppression, recall expansion, and router maps; "
            "'fpn_highres_precision_recall_prompt_head' adds a high-resolution FPN/local-detail "
            "branch for small-object prompt localization; "
            "'direct_signed_highres_prompt_head' additionally applies a bounded signed local "
            "residual directly to dense prompt logits; "
            "'unet_residual_only_prompt_head' preserves the legacy prompt path and adds only "
            "a bounded U-Net/RGB/MLDC residual correction."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_hidden",
        type=int,
        default=None,
        help="Hidden channels for --coarse_prompt_head=multiscale; defaults to a Ferret-dim-dependent value.",
    )
    parser.add_argument(
        "--coarse_prompt_dropout",
        type=float,
        default=0.0,
        help="Dropout2d probability inside the optional multiscale coarse prompt head.",
    )
    parser.add_argument(
        "--coarse_prompt_gate_init",
        type=float,
        default=0.02,
        help="Initial residual gate value for --coarse_prompt_head=gated_split_multiscale.",
    )
    parser.add_argument(
        "--coarse_prompt_gate_max",
        type=float,
        default=1.0,
        help="Maximum residual gate value for --coarse_prompt_head=gated_split_multiscale.",
    )
    parser.add_argument(
        "--coarse_prompt_area_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a zero-initialized sample-wise area-bias term in gated dense prompts.",
    )
    parser.add_argument(
        "--coarse_prompt_signed_residual_max_delta",
        type=float,
        default=0.5,
        help="Maximum absolute logit delta for --coarse_prompt_head=direct_signed_highres_prompt_head.",
    )
    parser.add_argument(
        "--coarse_prompt_unet_gate_init",
        type=float,
        default=None,
        help=(
            "Initial gate value for U-Net dense-prompt residual heads. "
            "Defaults to --coarse_prompt_gate_init for backward compatibility."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_unet_gate_max",
        type=float,
        default=None,
        help=(
            "Maximum gate value for U-Net dense-prompt residual heads. "
            "Defaults to --coarse_prompt_gate_max; set separately to keep legacy prompt gates unchanged."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_unet_signed_residual_max_delta",
        type=float,
        default=None,
        help=(
            "Maximum absolute logit delta for U-Net dense-prompt residual heads. "
            "Defaults to --coarse_prompt_signed_residual_max_delta."
        ),
    )
    parser.add_argument(
        "--coarse_prompt_head_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for optional coarse prompt-head residual parameters.",
    )
    parser.add_argument(
        "--dense_prompt_residual_reg_weight",
        type=float,
        default=0.0,
        help="L2 regularization weight for dense_prompt_mask - coarse_mask in split prompt experiments.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for direct signed dense-prompt residual FP/FN supervision.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_target_source",
        type=str,
        default="coarse_mask",
        choices=[
            "coarse_prompt",
            "dense_prompt",
            "dense_prompt_pre_unet",
            "coarse_mask",
            "pre_refiner_prompt",
            "final_logits",
            "raw_final_logits",
        ],
        help="Prompt/logit source used to derive signed residual FP/FN targets.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse", "balanced_smooth_l1", "balanced_l1", "balanced_mse"],
        help="Loss type for direct signed dense-prompt residual supervision.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_target_scale",
        type=float,
        default=1.0,
        help="Scale applied to signed dense-prompt residual FP/FN targets.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_target_mode",
        type=str,
        default="hard_error",
        choices=["soft_error", "hard_error"],
        help="Target map mode for signed dense-prompt residual supervision.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_hard_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used by hard_error signed dense-prompt residual supervision.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_use_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Supervise delta*gate instead of the raw signed delta for direct dense-prompt residuals.",
    )
    parser.add_argument(
        "--dense_prompt_signed_residual_max_gt_area",
        type=float,
        default=None,
        help="Optional max GT area for applying direct signed residual supervision to small masks only.",
    )
    parser.add_argument(
        "--dense_prompt_unet_identity_weight",
        type=float,
        default=0.0,
        help="Weight for anchoring final dense prompt logits to the pre-U-Net residual-only teacher.",
    )
    parser.add_argument(
        "--dense_prompt_unet_identity_loss",
        type=str,
        default="mse",
        choices=["mse", "l1", "smooth_l1"],
        help="Loss type for residual-only U-Net dense-prompt identity anchoring.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_supervision_weight",
        type=float,
        default=0.0,
        help="Weight for residual-only U-Net signed FP/FN correction supervision.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_loss",
        type=str,
        default="balanced_smooth_l1",
        choices=["smooth_l1", "l1", "mse", "balanced_smooth_l1", "balanced_l1", "balanced_mse"],
        help="Loss type for residual-only U-Net signed correction supervision.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_target_scale",
        type=float,
        default=1.0,
        help="Scale applied to residual-only U-Net signed FP/FN correction targets.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_target_mode",
        type=str,
        default="hard_error",
        choices=["soft_error", "hard_error"],
        help="Target map mode for residual-only U-Net correction supervision.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_hard_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used by hard_error residual-only U-Net supervision.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_use_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Supervise U-Net delta*gate instead of the raw signed delta.",
    )
    parser.add_argument(
        "--dense_prompt_unet_residual_max_gt_area",
        type=float,
        default=None,
        help="Optional max GT area for applying residual-only U-Net supervision to small masks only.",
    )
    parser.add_argument(
        "--freeze_adapters",
        action="store_true",
        help="Keep residual adapters frozen at their identity initialization",
    )
    parser.add_argument(
        "--freeze_ferret",
        action="store_true",
        help="Freeze the LAD/Ferret coarse prompt backbone for second-stage adapter fine-tuning",
    )
    parser.add_argument(
        "--freeze_ferret_unet_only",
        action="store_true",
        help="Freeze legacy Ferret prompt parameters and train only prompt_head_unet_* residual modules.",
    )
    
    # FerretNet pretrained weights configuration
    parser.add_argument(
        "--load_ferret_weights", type=bool, default=False, 
        help="Whether to load pretrained FerretNet weights"
    )
    parser.add_argument(
        "--ferret_weights_path", type=str, default="",
        help="Path to pretrained FerretNet weights"
    )

    
    # Validation configuration
    parser.add_argument(
        "--val_steps", type=int, default=VAL_STEPS,
        help="Number of training steps between validation runs (0 to disable validation during training)"
    )
    
    # EMA configuration
    parser.add_argument(
        "--ema_decay", type=float, default=0.975, 
        help="Decay rate for EMA"
    )
    
    # Authentic image configuration
    parser.add_argument(
        "--authentic_ratio", type=float, default=0.2,
        help="Ratio of authentic images to include (0.0-1.0)"
    )
    parser.add_argument(
        "--authentic_source_dir", type=str, default=None,
        help="Optional; ignored (authentic images use dataset root/source)"
    )
    # Output configuration
    parser.add_argument(
        "--save_path", type=str, default=SAVE_PATH,
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint for resuming training",
    )
    parser.add_argument(
        "--resume_use_ema_weights",
        action="store_true",
        help="When resuming, copy matching checkpoint EMA tensors into the model before fine-tuning",
    )
    parser.add_argument(
        "--reset_best_score",
        action="store_true",
        help="When resuming, reset best_score to 0 so new validation checkpoints can be saved",
    )
    parser.add_argument(
        "--reload_diagnostics_batches",
        type=int,
        default=0,
        help="Debugging only: after saving a checkpoint, compare live EMA weights with a fresh reload on this many validation batches",
    )
    parser.add_argument(
        "--checkpoint_save_mode",
        type=str,
        default="full",
        choices=["full", "trainable_only", "none"],
        help=(
            "Checkpoint payload mode. Use 'none' for short diagnostic runs on low disk; "
            "'trainable_only' omits frozen SAM tensors."
        ),
    )
    # Detection configuration
    parser.add_argument(
        "--lambda_detection", type=float, default=1.0,
        help="Weight for detection loss (only used when authentic_ratio > 0)"
    )
    
    # SAM IoU training configuration
    parser.add_argument(
        "--train_sam_iou",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to train the SAM IoU head"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    
    # Start Ferret-SAM training
    main(
        device=device,
        lambda_focal=args.lambda_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        weight_decay=args.weight_decay,
        prompt_dim=args.prompt_dim,
        save_path=args.save_path,
        downscale=args.downscale,
        ema_decay=args.ema_decay,
        img_size=args.img_size,
        lr=args.lr,
        lambda_iou=args.lambda_iou,
        dropout_rate=args.dropout_rate,
        lad_tau=args.lad_tau,
        lad_multi_taus=args.lad_multi_taus,
        forensic_operator=args.forensic_operator,
        scheduler_type=args.scheduler_type,
        authentic_ratio=args.authentic_ratio,
        authentic_source_dir=args.authentic_source_dir,
        val_steps=args.val_steps,
        dataset_config=args.dataset_config,
        val_manifest=args.val_manifest,
        lambda_detection=args.lambda_detection,
        resume_checkpoint=args.resume_checkpoint,
        train_sam_iou=args.train_sam_iou,
        train_force_resize=args.train_force_resize,
        val_force_resize=args.val_force_resize,
        sam_config=args.sam_config,
        sam_ckpt=args.sam_ckpt,
        sam_backend=args.sam_backend,
        freeze_adapters=args.freeze_adapters,
        freeze_ferret=args.freeze_ferret,
        freeze_ferret_unet_only=args.freeze_ferret_unet_only,
        adapter_residual_scale=args.adapter_residual_scale,
        adapter_type=args.adapter_type,
        adapter_scales=args.adapter_scales,
        adapter_gamma_init=args.adapter_gamma_init,
        adapter_sample_gate=args.adapter_sample_gate,
        adapter_sample_gate_scales=args.adapter_sample_gate_scales,
        adapter_sample_gate_max_delta=args.adapter_sample_gate_max_delta,
        adapter_forensic_source=args.adapter_forensic_source,
        adapter_delta_reg_weight=args.adapter_delta_reg_weight,
        adapter_diagnostics=args.adapter_diagnostics,
        sam3_prompt_mode=args.sam3_prompt_mode,
        coarse_prompt_transform=args.coarse_prompt_transform,
        coarse_prompt_scale=args.coarse_prompt_scale,
        coarse_prompt_bias=args.coarse_prompt_bias,
        coarse_prompt_calibrator=args.coarse_prompt_calibrator,
        coarse_prompt_calibrator_hidden=args.coarse_prompt_calibrator_hidden,
        coarse_prompt_calibrator_max_delta_scale=args.coarse_prompt_calibrator_max_delta_scale,
        coarse_prompt_calibrator_max_delta_bias=args.coarse_prompt_calibrator_max_delta_bias,
        coarse_prompt_calibrator_reg_weight=args.coarse_prompt_calibrator_reg_weight,
        coarse_prompt_calibrator_lr_multiplier=args.coarse_prompt_calibrator_lr_multiplier,
        prompt_bias_supervision_weight=args.prompt_bias_supervision_weight,
        prompt_bias_supervision_loss=args.prompt_bias_supervision_loss,
        prompt_bias_supervision_max_delta_bias=args.prompt_bias_supervision_max_delta_bias,
        final_logit_calibrator=args.final_logit_calibrator,
        final_logit_calibrator_hidden=args.final_logit_calibrator_hidden,
        final_logit_calibrator_max_delta_scale=args.final_logit_calibrator_max_delta_scale,
        final_logit_calibrator_max_delta_bias=args.final_logit_calibrator_max_delta_bias,
        final_logit_calibrator_supervision_weight=args.final_logit_calibrator_supervision_weight,
        final_logit_calibrator_supervision_loss=args.final_logit_calibrator_supervision_loss,
        final_logit_calibrator_supervision_max_delta_bias=args.final_logit_calibrator_supervision_max_delta_bias,
        final_logit_calibrator_oracle_thresholds=args.final_logit_calibrator_oracle_thresholds,
        final_logit_calibrator_oracle_min_threshold=args.final_logit_calibrator_oracle_min_threshold,
        final_logit_calibrator_oracle_max_threshold=args.final_logit_calibrator_oracle_max_threshold,
        final_logit_calibrator_oracle_false_positive_penalty=args.final_logit_calibrator_oracle_false_positive_penalty,
        final_logit_calibrator_oracle_area_penalty=args.final_logit_calibrator_oracle_area_penalty,
        final_logit_spatial_supervision_weight=args.final_logit_spatial_supervision_weight,
        final_logit_spatial_supervision_loss=args.final_logit_spatial_supervision_loss,
        final_logit_spatial_supervision_target_scale=args.final_logit_spatial_supervision_target_scale,
        final_logit_spatial_supervision_target_mode=args.final_logit_spatial_supervision_target_mode,
        final_logit_spatial_supervision_hard_threshold=args.final_logit_spatial_supervision_hard_threshold,
        coarse_prompt_refiner=args.coarse_prompt_refiner,
        coarse_prompt_refiner_hidden=args.coarse_prompt_refiner_hidden,
        coarse_prompt_refiner_max_residual=args.coarse_prompt_refiner_max_residual,
        coarse_prompt_refiner_gate_init=args.coarse_prompt_refiner_gate_init,
        coarse_prompt_refiner_gate_max=args.coarse_prompt_refiner_gate_max,
        coarse_prompt_refiner_precision_bias=args.coarse_prompt_refiner_precision_bias,
        coarse_prompt_refiner_recall_bias=args.coarse_prompt_refiner_recall_bias,
        coarse_prompt_head=args.coarse_prompt_head,
        coarse_prompt_hidden=args.coarse_prompt_hidden,
        coarse_prompt_dropout=args.coarse_prompt_dropout,
        coarse_prompt_gate_init=args.coarse_prompt_gate_init,
        coarse_prompt_gate_max=args.coarse_prompt_gate_max,
        coarse_prompt_area_bias=args.coarse_prompt_area_bias,
        coarse_prompt_signed_residual_max_delta=args.coarse_prompt_signed_residual_max_delta,
        coarse_prompt_unet_gate_init=args.coarse_prompt_unet_gate_init,
        coarse_prompt_unet_gate_max=args.coarse_prompt_unet_gate_max,
        coarse_prompt_unet_signed_residual_max_delta=args.coarse_prompt_unet_signed_residual_max_delta,
        coarse_prompt_head_lr_multiplier=args.coarse_prompt_head_lr_multiplier,
        dense_prompt_residual_reg_weight=args.dense_prompt_residual_reg_weight,
        dense_prompt_signed_residual_supervision_weight=args.dense_prompt_signed_residual_supervision_weight,
        dense_prompt_signed_residual_target_source=args.dense_prompt_signed_residual_target_source,
        dense_prompt_signed_residual_loss=args.dense_prompt_signed_residual_loss,
        dense_prompt_signed_residual_target_scale=args.dense_prompt_signed_residual_target_scale,
        dense_prompt_signed_residual_target_mode=args.dense_prompt_signed_residual_target_mode,
        dense_prompt_signed_residual_hard_threshold=args.dense_prompt_signed_residual_hard_threshold,
        dense_prompt_signed_residual_use_gate=args.dense_prompt_signed_residual_use_gate,
        dense_prompt_signed_residual_max_gt_area=args.dense_prompt_signed_residual_max_gt_area,
        dense_prompt_unet_identity_weight=args.dense_prompt_unet_identity_weight,
        dense_prompt_unet_identity_loss=args.dense_prompt_unet_identity_loss,
        dense_prompt_unet_residual_supervision_weight=args.dense_prompt_unet_residual_supervision_weight,
        dense_prompt_unet_residual_loss=args.dense_prompt_unet_residual_loss,
        dense_prompt_unet_residual_target_scale=args.dense_prompt_unet_residual_target_scale,
        dense_prompt_unet_residual_target_mode=args.dense_prompt_unet_residual_target_mode,
        dense_prompt_unet_residual_hard_threshold=args.dense_prompt_unet_residual_hard_threshold,
        dense_prompt_unet_residual_use_gate=args.dense_prompt_unet_residual_use_gate,
        dense_prompt_unet_residual_max_gt_area=args.dense_prompt_unet_residual_max_gt_area,
        resume_use_ema_weights=args.resume_use_ema_weights,
        reset_best_score=args.reset_best_score,
        coarse_loss_weight=args.coarse_loss_weight,
        coarse_dice_weight=args.coarse_dice_weight,
        coarse_prompt_loss_weight=args.coarse_prompt_loss_weight,
        coarse_prompt_dice_weight=args.coarse_prompt_dice_weight,
        coarse_prompt_false_negative_weight=args.coarse_prompt_false_negative_weight,
        coarse_prompt_false_positive_weight=args.coarse_prompt_false_positive_weight,
        coarse_prompt_supervision_source=args.coarse_prompt_supervision_source,
        coarse_prompt_supervision_max_gt_area=args.coarse_prompt_supervision_max_gt_area,
        prompt_gate_supervision_weight=args.prompt_gate_supervision_weight,
        prompt_gate_target_source=args.prompt_gate_target_source,
        prompt_gate_target_mode=args.prompt_gate_target_mode,
        prompt_gate_sources=args.prompt_gate_sources,
        prompt_gate_loss=args.prompt_gate_loss,
        dual_branch_prompt_gate_supervision_weight=args.dual_branch_prompt_gate_supervision_weight,
        dual_branch_prompt_gate_target_source=args.dual_branch_prompt_gate_target_source,
        dual_branch_prompt_gate_target_mode=args.dual_branch_prompt_gate_target_mode,
        dual_branch_prompt_gate_hard_threshold=args.dual_branch_prompt_gate_hard_threshold,
        dual_branch_prompt_gate_loss=args.dual_branch_prompt_gate_loss,
        dual_branch_prompt_residual_supervision_weight=args.dual_branch_prompt_residual_supervision_weight,
        dual_branch_prompt_residual_target_source=args.dual_branch_prompt_residual_target_source,
        dual_branch_prompt_residual_loss=args.dual_branch_prompt_residual_loss,
        dual_branch_prompt_residual_target_scale=args.dual_branch_prompt_residual_target_scale,
        dual_branch_prompt_residual_target_mode=args.dual_branch_prompt_residual_target_mode,
        dual_branch_prompt_residual_hard_threshold=args.dual_branch_prompt_residual_hard_threshold,
        dual_prompt_fusion_supervision_weight=args.dual_prompt_fusion_supervision_weight,
        dual_prompt_fusion_oracle_metric=args.dual_prompt_fusion_oracle_metric,
        dual_prompt_fusion_loss=args.dual_prompt_fusion_loss,
        area_reg_weight=args.area_reg_weight,
        area_reg_target_source=args.area_reg_target_source,
        area_reg_loss=args.area_reg_loss,
        area_reg_apply_to=args.area_reg_apply_to,
        area_reg_constant=args.area_reg_constant,
        area_reg_max_gt_area=args.area_reg_max_gt_area,
        reload_diagnostics_batches=args.reload_diagnostics_batches,
        checkpoint_save_mode=args.checkpoint_save_mode,
    )
