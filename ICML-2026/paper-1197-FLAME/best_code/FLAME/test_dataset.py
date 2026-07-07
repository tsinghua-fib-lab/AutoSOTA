#!/usr/bin/env python3
"""Test script for evaluating a trained FLAME forgery localizer model on an entire dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from utils.localforgerydataset import LocalForgeryDataset
from utils.validate import validate
from utils.checkpoint_state import load_matching_state_dict
from utils.train_utils import custom_collate_fn
from model.forgerylocalizer import ForgeryLocalizer
from utils.sam_utils import initialize_sam_hydra

# Initialize Hydra configuration for SAM2
initialize_sam_hydra()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_config(json_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file.
    
    Args:
        json_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing model configuration
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def get_max_streams(contrastive_blur: bool, perturbation_type: str) -> int:
    """Determine the maximum number of streams based on contrastive mode and perturbation type.
    
    Args:
        contrastive_blur: Whether contrastive mode is enabled
        perturbation_type: Type of perturbation being applied
        
    Returns:
        Maximum number of streams the model should expect
    """
    if contrastive_blur:
        # Contrastive mode: always has streams (sharp/clean + perturbations)
        if perturbation_type == "none":
            return 0  # [clean, clean] (identical for consistency)
        elif perturbation_type in ["gaussian_blur", "jpeg_compression", "gaussian_noise"]:
            return 2  # [sharp/clean, perturbation]
        elif perturbation_type == "gaussian_blur/gaussian_noise":
            return 3  # [sharp, blur, noise]
    else:
        # Non-contrastive mode: only perturbations in streams
        if perturbation_type == "none":
            return 0  # Empty streams list - model will use orig as fallback
        elif perturbation_type in ["gaussian_blur", "jpeg_compression", "gaussian_noise"]:
            return 1  # Single perturbation
        elif perturbation_type == "gaussian_blur/gaussian_noise":
            return 2  # Two perturbations [blur, noise]
    return 0


def parse_adapter_scales(value: Any) -> Optional[list[int]]:
    """Normalize saved/CLI adapter scale settings.

    Training stores ``None`` for all scales and a JSON list for ablations.
    Older ad-hoc configs may store strings such as ``"1,2"`` or ``"all"``.
    The eval path must recreate the exact adapter topology before loading a
    checkpoint; otherwise norm-gated adapter checkpoints silently load with
    missing/unexpected keys under ``strict=False`` and the reload metric no
    longer proves anything about the trained model.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none"}:
            return None
        scales = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        scales = [int(scale) for scale in value]
    invalid = [scale for scale in scales if scale not in {0, 1, 2}]
    if invalid:
        raise ValueError(f"adapter scales must be drawn from 0,1,2; got {invalid}")
    return scales


def parse_float_list(value: Any) -> Optional[list[float]]:
    """Normalize optional saved/CLI comma-separated float lists."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none"}:
            return None
        return [float(part.strip()) for part in text.split(",") if part.strip()]
    return [float(part) for part in value]


def _compute_input_perturbation_params(
    perturbation_type: str,
    perturbation_intensity: float
) -> Dict[str, float]:
    """Compute input perturbation parameters based on intensity (0-1.5)."""
    params: Dict[str, float] = {}
    if perturbation_type == "gaussian_blur":
        params["blur_sigma"] = perturbation_intensity
    elif perturbation_type == "jpeg_compression":
        params["jpeg_quality"] = max(10, int(95 - (perturbation_intensity * 56.67)))
    elif perturbation_type == "gaussian_noise":
        params["noise_std"] = perturbation_intensity * 0.2
    elif perturbation_type == "gaussian_blur/gaussian_noise":
        params["blur_sigma"] = perturbation_intensity
        params["noise_std"] = perturbation_intensity * 0.2
    elif perturbation_type == "none":
        pass
    else:
        raise ValueError(f"Unknown input perturbation type: {perturbation_type}")
    return params


def _apply_input_perturbation(
    img_tensor: torch.Tensor,
    perturbation_type: str,
    params: Dict[str, float]
) -> torch.Tensor:
    """Apply a robustness perturbation to a single image tensor."""
    if perturbation_type == "none":
        return img_tensor
    if perturbation_type == "gaussian_blur":
        from utils.perturbations import apply_blur_to_image_tensor
        return apply_blur_to_image_tensor(img_tensor, params["blur_sigma"])
    if perturbation_type == "jpeg_compression":
        from utils.perturbations import apply_jpeg_compression_to_tensor
        return apply_jpeg_compression_to_tensor(img_tensor, params["jpeg_quality"])
    if perturbation_type == "gaussian_noise":
        from utils.perturbations import add_gaussian_noise
        return add_gaussian_noise(img_tensor, std=params["noise_std"])
    if perturbation_type == "gaussian_blur/gaussian_noise":
        from utils.perturbations import apply_blur_to_image_tensor, add_gaussian_noise
        blurred = apply_blur_to_image_tensor(img_tensor, params["blur_sigma"])
        return add_gaussian_noise(blurred, std=params["noise_std"])
    raise ValueError(f"Unknown input perturbation type: {perturbation_type}")


class InputPerturbationDataset(Dataset):
    """Dataset wrapper that applies input perturbations for robustness testing."""

    def __init__(
        self,
        base_dataset: Dataset,
        perturbation_type: str,
        perturbation_intensity: float,
        apply_to_source: bool = False
    ) -> None:
        self.base_dataset = base_dataset
        self.perturbation_type = perturbation_type
        self.perturbation_intensity = perturbation_intensity
        self.apply_to_source = apply_to_source
        self.params = _compute_input_perturbation_params(perturbation_type, perturbation_intensity)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _maybe_apply(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return _apply_input_perturbation(value, self.perturbation_type, self.params)
        if isinstance(value, list):
            return [
                _apply_input_perturbation(v, self.perturbation_type, self.params)
                if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        return value

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        if not sample or sample.get("orig") is None:
            return sample
        sample["orig"] = self._maybe_apply(sample["orig"])
        if self.apply_to_source and sample.get("source") is not None:
            sample["source"] = self._maybe_apply(sample["source"])
        return sample


def load_and_initialize_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = False,
) -> ForgeryLocalizer:
    """Load and initialize the forgery localizer model.
    
    Args:
        config: Model configuration dictionary
        checkpoint_path: Path to the model checkpoint (.pth file)
        device: Device to load the model on
        
    Returns:
        Initialized ForgeryLocalizer model
    """
    model_config = config['model_config']
    sam_config_dict = config['sam_config']
    data_config = config['data_config']
    
    # Get max_streams from configuration
    contrastive_blur = model_config.get('contrastive_blur', False)
    perturbation_type = data_config.get('perturbation_type', 'none')
    max_streams = get_max_streams(contrastive_blur, perturbation_type)
    
    # Determine use_detection_probe based on authentic_ratio
    authentic_ratio = data_config.get('authentic_ratio', 0.0)
    use_detection_probe = authentic_ratio > 0.0
    
    logger.info(f"Initializing model with max_streams={max_streams}, use_detection_probe={use_detection_probe}")
    
    # Get the directory of the test script to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get SAM config/checkpoint/backend from config dictionary.
    sam_config_dict = config.get('sam_config', {})
    sam_backend = sam_config_dict.get('sam_backend', model_config.get('sam_backend', 'sam2'))
    sam_config_file = sam_config_dict.get('sam_config_file', 'sam2.1_hiera_b+.yaml')
    sam_checkpoint_config = sam_config_dict.get(
        'sam_checkpoint',
        os.path.join(script_dir, 'sam2configs/sam2.1_hiera_base_plus.pt'),
    )
    if os.path.isabs(sam_checkpoint_config) or os.path.exists(sam_checkpoint_config):
        sam_checkpoint = sam_checkpoint_config
    else:
        sam_checkpoint = os.path.join(script_dir, sam_checkpoint_config)
    output_img_size = int(config.get('training_config', {}).get('img_size', 512))
    
    logger.info(f"SAM backend: {sam_backend}")
    logger.info(f"SAM config: {sam_config_file}")
    logger.info(f"SAM checkpoint: {sam_checkpoint}")
    logger.info(f"Model output resolution: {output_img_size}")
    
    # Initialize model with the training output resolution.
    model = ForgeryLocalizer(
        sam_config=sam_config_file,
        sam_checkpoint=sam_checkpoint,
        sam_backend=sam_backend,
        prompt_dim=model_config['prompt_dim'],
        output_resolution=(output_img_size, output_img_size),
        downscale=model_config['downscale'],
        train_sam_iou=model_config.get('train_sam_iou', True),
        dropout_rate=model_config['dropout_rate'],
        lad_tau=model_config.get('lad_tau', 0.004),
        lad_multi_taus=parse_float_list(model_config.get('lad_multi_taus')),
        forensic_operator=model_config.get('forensic_operator', 'lad'),
        use_detection_probe=use_detection_probe,
        adapter_residual_scale=model_config.get('adapter_residual_scale', 1.0),
        adapter_type=model_config.get('adapter_type', 'shared'),
        adapter_active_scales=parse_adapter_scales(model_config.get('adapter_scales')),
        adapter_gamma_init=model_config.get('adapter_gamma_init', 0.0),
        adapter_sample_gate=model_config.get('adapter_sample_gate', False),
        adapter_sample_gate_scales=parse_adapter_scales(model_config.get('adapter_sample_gate_scales')),
        adapter_sample_gate_max_delta=model_config.get('adapter_sample_gate_max_delta', 0.5),
        adapter_forensic_source=model_config.get('adapter_forensic_source', 'final'),
        adapter_diagnostics=model_config.get('adapter_diagnostics', False),
        sam3_prompt_mode=model_config.get('sam3_prompt_mode', 'legacy'),
        coarse_prompt_transform=model_config.get('coarse_prompt_transform', 'none'),
        coarse_prompt_scale=model_config.get('coarse_prompt_scale', 1.0),
        coarse_prompt_bias=model_config.get('coarse_prompt_bias', 0.0),
        coarse_prompt_calibrator=model_config.get('coarse_prompt_calibrator', 'none'),
        coarse_prompt_calibrator_hidden=model_config.get('coarse_prompt_calibrator_hidden', 16),
        coarse_prompt_calibrator_max_delta_scale=model_config.get('coarse_prompt_calibrator_max_delta_scale', 1.0),
        coarse_prompt_calibrator_max_delta_bias=model_config.get('coarse_prompt_calibrator_max_delta_bias', 2.0),
        final_logit_calibrator=model_config.get('final_logit_calibrator', 'none'),
        final_logit_calibrator_hidden=model_config.get('final_logit_calibrator_hidden', 16),
        final_logit_calibrator_max_delta_scale=model_config.get('final_logit_calibrator_max_delta_scale', 0.0),
        final_logit_calibrator_max_delta_bias=model_config.get('final_logit_calibrator_max_delta_bias', 1.0),
        coarse_prompt_refiner=model_config.get('coarse_prompt_refiner', 'none'),
        coarse_prompt_refiner_hidden=model_config.get('coarse_prompt_refiner_hidden', 8),
        coarse_prompt_refiner_max_residual=model_config.get('coarse_prompt_refiner_max_residual', 1.0),
        coarse_prompt_refiner_gate_init=model_config.get('coarse_prompt_refiner_gate_init', 0.2),
        coarse_prompt_refiner_gate_max=model_config.get('coarse_prompt_refiner_gate_max', 1.0),
        coarse_prompt_refiner_precision_bias=model_config.get('coarse_prompt_refiner_precision_bias', -0.10),
        coarse_prompt_refiner_recall_bias=model_config.get('coarse_prompt_refiner_recall_bias', 0.45),
        coarse_prompt_head=model_config.get('coarse_prompt_head', 'mask_compressor'),
        coarse_prompt_hidden=model_config.get('coarse_prompt_hidden'),
        coarse_prompt_dropout=model_config.get('coarse_prompt_dropout', 0.0),
        coarse_prompt_gate_init=model_config.get('coarse_prompt_gate_init', 0.02),
        coarse_prompt_gate_max=model_config.get('coarse_prompt_gate_max', 1.0),
        coarse_prompt_area_bias=model_config.get('coarse_prompt_area_bias', False),
        coarse_prompt_signed_residual_max_delta=model_config.get('coarse_prompt_signed_residual_max_delta', 0.5),
        coarse_prompt_unet_gate_init=model_config.get('coarse_prompt_unet_gate_init'),
        coarse_prompt_unet_gate_max=model_config.get('coarse_prompt_unet_gate_max'),
        coarse_prompt_unet_signed_residual_max_delta=model_config.get('coarse_prompt_unet_signed_residual_max_delta'),
        mask_compressor_kernel_size=model_config.get('mask_compressor_kernel_size', 3),
        mask_compressor_output=model_config.get('mask_compressor_output', 'logits'),
        legacy_logit_head=model_config.get('legacy_logit_head', False),
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint_data:
        report = load_matching_state_dict(model, checkpoint_data["model"])
        if report["missing_keys"]:
            logger.warning(f"Missing keys when loading checkpoint: {report['missing_keys']}")
        if report["unexpected_keys"]:
            logger.warning(f"Unexpected keys when loading checkpoint: {report['unexpected_keys']}")
        if report["skipped_shape_mismatch"]:
            logger.warning(f"Skipped shape-mismatched checkpoint keys: {report['skipped_shape_mismatch']}")
        if report.get("adapted_shape_mismatch"):
            logger.info(f"Adapted shape-mismatched checkpoint keys: {report['adapted_shape_mismatch']}")
    else:
        # Checkpoint might be just the state dict
        report = load_matching_state_dict(model, checkpoint_data)
        if report["missing_keys"]:
            logger.warning(f"Missing keys when loading checkpoint: {report['missing_keys']}")
        if report["unexpected_keys"]:
            logger.warning(f"Unexpected keys when loading checkpoint: {report['unexpected_keys']}")
        if report["skipped_shape_mismatch"]:
            logger.warning(f"Skipped shape-mismatched checkpoint keys: {report['skipped_shape_mismatch']}")
        if report.get("adapted_shape_mismatch"):
            logger.info(f"Adapted shape-mismatched checkpoint keys: {report['adapted_shape_mismatch']}")

    if use_ema:
        ema_state = checkpoint_data.get("ema")
        if not isinstance(ema_state, dict):
            raise ValueError(f"--use_ema requested, but checkpoint has no EMA state dict: {checkpoint_path}")
        report = load_matching_state_dict(model, ema_state)
        logger.info(
            "Overlayed EMA state dict - missing=%d unexpected=%d",
            len(report["missing_keys"]),
            len(report["unexpected_keys"]),
        )
        if report["unexpected_keys"]:
            logger.warning(f"Unexpected keys when loading EMA checkpoint: {report['unexpected_keys']}")
        if report["skipped_shape_mismatch"]:
            logger.warning(f"Skipped shape-mismatched EMA keys: {report['skipped_shape_mismatch']}")
        if report.get("adapted_shape_mismatch"):
            logger.info(f"Adapted shape-mismatched EMA keys: {report['adapted_shape_mismatch']}")
    
    logger.info(f"Loaded checkpoint - epoch: {checkpoint_data.get('epoch', 'N/A')}, score: {checkpoint_data.get('score', 'N/A')}")
    
    # Set model to eval mode
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    model.sam_prompt_encoder.eval()
    
    return model


def create_dataloader(
    dataset_dir: str,
    img_size: int,
    perturbation_type: str,
    perturbation_intensity: float,
    contrastive_blur: bool,
    input_perturbation_type: str,
    input_perturbation_intensity: float,
    input_perturb_source: bool,
    authentic_ratio: float,
    batch_size: int,
    num_workers: int
) -> DataLoader:
    """Create a DataLoader for the test dataset.
    
    Args:
        dataset_dir: Directory containing the dataset with source/target/mask subdirectories
        img_size: Target image size
        perturbation_type: Type of perturbation to apply
        perturbation_intensity: Intensity of perturbation
        contrastive_blur: Whether to use contrastive blur
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for the test dataset
    """
    dataset = LocalForgeryDataset(
        root_dir=dataset_dir,
        img_size=img_size,
        allow_multiple_targets=False,
        # contrastive_blur=contrastive_blur,
        is_training=False,  # Use validation mode (center crop)
        # perturbation_type=perturbation_type,
        # perturbation_intensity=perturbation_intensity,
        authentic_ratio=authentic_ratio,
        authentic_source_dir=None,
    )

    if input_perturbation_type != "none":
        dataset = InputPerturbationDataset(
            dataset,
            perturbation_type=input_perturbation_type,
            perturbation_intensity=input_perturbation_intensity,
            apply_to_source=input_perturb_source,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Test FLAME forgery localizer on an entire dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset directory containing source/target/mask subdirectories')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration JSON file')
    parser.add_argument('--output', type=str, default='dataset_results',
                        help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--num_vis_batches', type=int, default=5,
                        help='Number of batches to visualize (ignored if --save_all_vis is set)')
    parser.add_argument('--save_all_vis', action='store_true',
                        help='Save visualization results for all test images')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                        help='Probability threshold for image-level detection classification')
    parser.add_argument('--input_perturbation_type', type=str, default='none',
                        choices=['none', 'gaussian_blur', 'jpeg_compression', 'gaussian_noise',
                                 'gaussian_blur/gaussian_noise'],
                        help='Apply a robustness perturbation to input images before testing')
    parser.add_argument('--input_perturbation_intensity', type=float, default=0.5,
                        help='Perturbation intensity (0.0-1.5). Higher is stronger.')
    parser.add_argument('--input_perturb_source', action='store_true',
                        help='Also perturb source images (visualization only)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Evaluation resize size before inference (default: 512)')
    parser.add_argument('--authentic_ratio', type=float, default=1.0,
                        help='Ratio of authentic source images to append for detection metrics; localization IoU/F1 ignore them')
    parser.add_argument('--use_tiling', action='store_true',
                        help='Use overlap-tile inference instead of single resized-image inference')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Tile size for --use_tiling')
    parser.add_argument('--tile_valid_size', type=int, default=384,
                        help='Central valid tile size for --use_tiling')
    parser.add_argument('--tile_stride', type=int, default=384,
                        help='Tile stride for --use_tiling')
    parser.add_argument('--use_ema', action='store_true',
                        help='After loading the model state, overlay EMA weights from the checkpoint if present')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_model_config(args.config)
    
    # Load model
    model = load_and_initialize_model(config, args.model, device, use_ema=args.use_ema)
    
    # Get data configuration
    data_config = config['data_config']
    training_config = config['training_config']
    model_config = config['model_config']
    
    # Default is 512x512 to match training; can be overridden for reproduction sweeps.
    img_size = args.img_size
    perturbation_type = data_config.get('perturbation_type', 'none')
    perturbation_intensity = data_config.get('perturbation_intensity', 0.5)
    contrastive_blur = model_config.get('contrastive_blur', False)
    
    logger.info(f"Configuration: img_size={img_size}, perturbation_type={perturbation_type}, "
                f"perturbation_intensity={perturbation_intensity}, contrastive_blur={contrastive_blur}")
    if args.input_perturbation_type != "none":
        logger.info(
            "Input perturbation enabled: type=%s intensity=%s apply_to_source=%s",
            args.input_perturbation_type,
            args.input_perturbation_intensity,
            args.input_perturb_source,
        )
    if args.save_all_vis:
        logger.info("Saving visualizations for all test images.")
    
    # Create output directory for visualizations
    os.makedirs(args.output, exist_ok=True)
    
    # Create dataloader for the entire dataset
    dataloader = create_dataloader(
        dataset_dir=args.dataset,
        img_size=img_size,
        perturbation_type=perturbation_type,
        perturbation_intensity=perturbation_intensity,
        contrastive_blur=contrastive_blur,
        input_perturbation_type=args.input_perturbation_type,
        input_perturbation_intensity=args.input_perturbation_intensity,
        input_perturb_source=args.input_perturb_source,
        authentic_ratio=args.authentic_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Run evaluation on the entire dataset
    logger.info(f"Running evaluation on dataset: {args.dataset}")
    results = validate(
        model=model,
        loader=dataloader,
        device=device,
        save_vis_path=args.output,
        num_vis_batches=args.num_vis_batches,
        save_all_vis=args.save_all_vis,
        contrastive_blur=contrastive_blur,
        model_outputs_probs=False,
        detection_threshold=args.detection_threshold,
        use_tiling=args.use_tiling,
        tile_size=args.tile_size,
        tile_valid_size=args.tile_valid_size,
        tile_stride=args.tile_stride,
    )
    
    # Save results to JSON file
    results_file = os.path.join(args.output, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to {results_file}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
