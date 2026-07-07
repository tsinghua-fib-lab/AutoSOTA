"""Test script for evaluating a trained FLAME forgery localizer model on a single image."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
from typing import Dict, Any, Tuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.forgerylocalizer import ForgeryLocalizer
from utils.localforgerydataset import LocalForgeryDataset
from utils.sam_utils import initialize_sam_hydra, get_sam_config_from_json

# Initialize Hydra configuration for SAM2
initialize_sam_hydra()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def load_and_initialize_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device
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
    
    # Get SAM config and checkpoint from config dictionary
    sam_config_dict = config.get('sam_config', {})
    sam_config_file = sam_config_dict.get('sam_config_file', 'sam2.1_hiera_b+.yaml')
    
    # Manually construct the correct sam_checkpoint path
    sam_checkpoint = os.path.join(script_dir, 'sam2configs/sam2.1_hiera_base_plus.pt')
    
    logger.info(f"SAM config: {sam_config_file}")
    logger.info(f"SAM checkpoint: {sam_checkpoint}")
    
    # Initialize model
    model = ForgeryLocalizer(
        sam_config=sam_config_file,
        sam_checkpoint=sam_checkpoint,
        prompt_dim=model_config['prompt_dim'],
        downscale=model_config['downscale'],
        train_sam_iou=model_config.get('train_sam_iou', True),
        dropout_rate=model_config['dropout_rate'],
        use_detection_probe=use_detection_probe,
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint_data:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data["model"], strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    else:
        # Checkpoint might be just the state dict
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    
    logger.info(f"Loaded checkpoint - epoch: {checkpoint_data.get('epoch', 'N/A')}, score: {checkpoint_data.get('score', 'N/A')}")
    
    # Set model to eval mode
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    model.sam_prompt_encoder.eval()
    
    return model


def create_temp_dataset_structure(
    image_path: str,
    mask_path: str = None,
    source_path: str = None
) -> str:
    """Create a temporary dataset structure for LocalForgeryDataset.
    
    Args:
        image_path: Path to the input image (target)
        mask_path: Optional path to ground truth mask
        source_path: Optional path to source image (original unedited)
        
    Returns:
        Path to temporary dataset directory
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="flame_test_")
    
    # Create dataset structure
    target_dir = os.path.join(temp_dir, "target")
    mask_dir = os.path.join(temp_dir, "mask")
    source_dir = os.path.join(temp_dir, "source")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    
    # Copy image to target directory with a standard name
    img_name = "test_image.png"
    shutil.copy(image_path, os.path.join(target_dir, img_name))
    
    # Copy or create dummy mask
    if mask_path and os.path.exists(mask_path):
        shutil.copy(mask_path, os.path.join(mask_dir, img_name))
    else:
        # Create a dummy mask (all zeros) if no mask provided
        img = Image.open(image_path)
        dummy_mask = Image.new('L', img.size, 0)
        dummy_mask.save(os.path.join(mask_dir, img_name))
    
    # Copy source image or use target as source
    if source_path and os.path.exists(source_path):
        shutil.copy(source_path, os.path.join(source_dir, img_name))
    else:
        # Use target image as source if no source provided
        shutil.copy(image_path, os.path.join(source_dir, img_name))
    
    return temp_dir


def load_sample_from_dataset(
    image_path: str,
    mask_path: str,
    source_path: str,
    img_size: int,
    perturbation_type: str,
    perturbation_intensity: float,
    contrastive_blur: bool
) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess a sample using LocalForgeryDataset.
    
    Args:
        image_path: Path to input image (target)
        mask_path: Path to ground truth mask (optional)
        source_path: Path to source image (original unedited)
        img_size: Target image size
        perturbation_type: Type of perturbation to apply
        perturbation_intensity: Intensity of perturbation
        contrastive_blur: Whether to use contrastive blur
        
    Returns:
        Tuple of (orig_tensor, streams_list, mask_tensor, source_tensor, original_image_np, source_image_np, mask_np)
    """
    # Create temporary dataset structure
    temp_dir = create_temp_dataset_structure(image_path, mask_path, source_path)
    
    try:
        # Create dataset
        dataset = LocalForgeryDataset(
            root_dir=temp_dir,
            img_size=img_size,
            allow_multiple_targets=False,
            contrastive_blur=contrastive_blur,
            is_training=False,  # Use validation mode (center crop)
            perturbation_type=perturbation_type,
            perturbation_intensity=perturbation_intensity,
            authentic_ratio=0.0,
            authentic_source_dir=None,
        )
        
        # Get the single sample
        if len(dataset) == 0:
            raise ValueError("Dataset is empty - check image paths")
        
        sample = dataset[0]
        
        # Extract data from sample
        orig_tensor = sample['orig']    # [3, H, W] - target
        streams = sample['streams']     # List of [3, H, W] tensors
        mask_tensor = sample['mask']    # [1, H, W]
        source_tensor = sample['source'] # [3, H, W] - source
        
        # Load original image for visualization
        orig_img = np.array(Image.open(image_path).convert('RGB'))
        
        # Convert source tensor to numpy for visualization using proper normalization
        source_normalized = normalize_for_display(source_tensor)
        source_img = source_normalized.permute(1, 2, 0).cpu().numpy()
        # Convert to 0-255 range for display
        source_img = (source_img * 255).clip(0, 255).astype(np.uint8)
        
        # Load ground truth mask for visualization
        if mask_path and os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert('L'))
        else:
            mask_img = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        
        return orig_tensor, streams, mask_tensor, source_tensor, orig_img, source_img, mask_img
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def visualize_results(
    source_img: np.ndarray,
    target_img: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray = None,
    save_path: str = None,
    detection_prob: float = None,
    forensic_features: torch.Tensor = None,
    heatmap_prompt: torch.Tensor = None
):
    """Visualize the prediction results.
    
    Args:
        source_img: Source image
        target_img: Target image
        prediction: Binary prediction mask
        ground_truth: Optional ground truth mask
        save_path: Path to save the visualization
        detection_prob: Optional detection probability
        forensic_features: Optional LPD input features
        heatmap_prompt: Optional GLP heatmap output
    """
    # Determine number of columns based on available data
    num_cols = 6  # Basic: source, target, gt, pred, lpd, glp
    
    fig, axes = plt.subplots(1, num_cols, figsize=(25, 5))
    
    # Source image
    axes[0].imshow(source_img)
    axes[0].set_title('Source')
    axes[0].axis('off')
    
    # Target image
    axes[1].imshow(target_img)
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    # Ground truth mask over target
    axes[2].imshow(target_img)
    if ground_truth is not None and ground_truth.max() > 0:
        gt_overlay = np.zeros((*ground_truth.shape, 4))
        gt_overlay[ground_truth > 0] = [0, 1, 0, 0.5]  # Green with 50% transparency
        axes[2].imshow(gt_overlay)
    axes[2].set_title('GT Mask over Target')
    axes[2].axis('off')
    
    # Prediction mask over target
    axes[3].imshow(target_img)
    if prediction.max() > 0:
        pred_overlay = np.zeros((*prediction.shape, 4))
        pred_overlay[prediction > 0] = [1, 0, 0, 0.5]  # Red with 50% transparency
        axes[3].imshow(pred_overlay)
    title = 'Prediction over Target'
    if detection_prob is not None:
        title += f'\n(Detection Prob: {detection_prob:.3f})'
    axes[3].set_title(title)
    axes[3].axis('off')
    
    # LPD Input (forensic_features) visualization
    if forensic_features is not None:
        # Forensic features have multiple channels, take the first channel for visualization
        # or create a composite view using mean across channels
        lpd_visual = forensic_features[0].mean(dim=0).cpu().squeeze()
        # Normalize for display
        lpd_visual = (lpd_visual - lpd_visual.min()) / (lpd_visual.max() - lpd_visual.min() + 1e-8)
        axes[4].imshow(lpd_visual, cmap='plasma')
        axes[4].set_title('LPD Input')
    else:
        axes[4].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    transform=axes[4].transAxes)
        axes[4].set_title('LPD Input')
    axes[4].axis('off')
    
    # GLP Heatmap Prompt visualization
    if heatmap_prompt is not None:
        heatmap_visual = heatmap_prompt[0, 0].cpu().numpy()
        axes[5].imshow(heatmap_visual, cmap='plasma', vmin=0, vmax=1)
        axes[5].set_title('GLP Heatmap')
    else:
        axes[5].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    transform=axes[5].transAxes)
        axes[5].set_title('GLP Heatmap')
    axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        prediction: Binary prediction mask
        ground_truth: Binary ground truth mask
        
    Returns:
        Dictionary of metrics
    """
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Compute IoU
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    iou = intersection / (union + 1e-8)
    
    # Compute precision, recall, F1
    tp = intersection
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Test FLAME forgery localizer on a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (target/edited image)')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to source image (original unedited image). If not provided, will use target image as source.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration JSON file')
    parser.add_argument('--mask', type=str, default=None,
                        help='Optional path to ground truth mask for evaluation')
    parser.add_argument('--output', type=str, default='result.png',
                        help='Path to save output visualization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction (default: 0.5)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_model_config(args.config)
    
    # Load model
    model = load_and_initialize_model(config, args.model, device)
    
    # Get data configuration
    data_config = config['data_config']
    training_config = config['training_config']
    model_config = config['model_config']
    
    img_size = training_config.get('img_size', 512)
    perturbation_type = data_config.get('perturbation_type', 'none')
    perturbation_intensity = data_config.get('perturbation_intensity', 0.5)
    contrastive_blur = model_config.get('contrastive_blur', False)
    
    logger.info(f"Configuration: img_size={img_size}, perturbation_type={perturbation_type}, "
                f"perturbation_intensity={perturbation_intensity}, contrastive_blur={contrastive_blur}")
    
    # Load and preprocess using LocalForgeryDataset for consistency with training
    logger.info(f"Loading image from {args.image} using LocalForgeryDataset")
    if args.source:
        logger.info(f"Using source image from {args.source}")
    orig_tensor, streams, mask_tensor, source_tensor, orig_img, source_img, mask_img = load_sample_from_dataset(
        image_path=args.image,
        mask_path=args.mask,
        source_path=args.source,
        img_size=img_size,
        perturbation_type=perturbation_type,
        perturbation_intensity=perturbation_intensity,
        contrastive_blur=contrastive_blur
    )
    
    # Prepare batch
    orig_batch = orig_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    streams_batch = [s.unsqueeze(0).to(device) for s in streams]  # List of [1, 3, H, W]
    
    logger.info(f"Running inference with {len(streams_batch)} stream(s)...")
    
    # Run inference
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            outputs = model(orig_batch, streams_batch, output_extras=True)
            
            if isinstance(outputs, tuple):
                logits, extras = outputs
            else:
                logits = outputs
                extras = {}
            
            # Get probability map
            probs = torch.sigmoid(logits)
            
            # Get detection probability if available
            detection_prob = None
            if 'detection_logit' in extras and extras['detection_logit'] is not None:
                detection_logit = extras['detection_logit']
                detection_prob = torch.sigmoid(detection_logit).item()
                logger.info(f"Detection probability: {detection_prob:.4f}")
            
            # Get forensic_features (LPD input) and heatmap_prompt (GLP output) if available
            forensic_features = extras.get('forensic_features', None)
            heatmap_prompt = extras.get('heatmap_prompt', None)
    
    # Convert to numpy
    probs_np = probs[0, 0].cpu().numpy()
    pred_binary = (probs_np > args.threshold).astype(np.uint8)
    
    logger.info(f"Prediction shape: {pred_binary.shape}")
    logger.info(f"Forgery coverage: {pred_binary.sum() / pred_binary.size * 100:.2f}%")
    
    # Process ground truth mask from dataset (already at correct size)
    ground_truth = None
    if args.mask:
        logger.info(f"Using ground truth mask from dataset")
        ground_truth = mask_tensor[0].cpu().numpy().astype(np.uint8)  # [H, W]
        
        # Compute metrics on the 512x512 versions
        metrics = compute_metrics(pred_binary, ground_truth)
        logger.info(f"Metrics: IoU={metrics['iou']:.4f}, Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Resize prediction, GT, and source to original image size for visualization
    if orig_img.shape[:2] != pred_binary.shape:
        pred_resized = cv2.resize(pred_binary, (orig_img.shape[1], orig_img.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        source_resized = cv2.resize(source_img, (orig_img.shape[1], orig_img.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        if ground_truth is not None:
            gt_resized = cv2.resize(ground_truth, (orig_img.shape[1], orig_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            gt_resized = None
    else:
        pred_resized = pred_binary
        source_resized = source_img
        gt_resized = ground_truth
    
    # Visualize results
    logger.info("Generating visualization...")
    visualize_results(source_resized, orig_img, pred_resized, gt_resized, args.output, detection_prob, forensic_features, heatmap_prompt)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()