"""Evaluation loops for MASS segmentation models.

This module contains the in-context evaluator used by ``evaluate.py`` and the
regular evaluator used by downstream examples. It handles sliding-window
inference, task-embedding ensembles, prediction saving, and Dice/ASD/HD95
metric aggregation.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import logging
from tqdm import tqdm
import time
from pathlib import Path 

from utils.distributed import (
    is_master, 
    get_rank, 
    get_local_rank,
    get_world_size,
    is_distributed, 
    all_gather_object
)

from metrics.dice import calculate_dice_split
from metrics.surface_distance import calculate_surface_distance

class Evaluator:
    """
    Evaluator for 3D medical image segmentation models.
    
    Supports:
    - Standard evaluation
    - In-context evaluation (one-shot and ensemble)
    - Distributed evaluation
    - Mixed precision evaluation
    - Per-class metrics reporting
    - Optional prediction saving for visualization
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        incontext: bool = False,
        save_predictions: bool = False,
        save_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation
            config: Configuration dictionary
            device: Device to run evaluation on
            incontext: Whether to run in-context evaluation
            save_predictions: Whether to save predictions
            save_dir: Directory to save predictions
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device or torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        self.incontext = incontext

        self.save_predictions = save_predictions
        self.save_dir = save_dir

        if self.save_predictions and self.save_dir is None:
            raise ValueError("save_dir must be provided if save_predictions is True")
        if self.save_predictions:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            import SimpleITK as sitk
            self.sitk = sitk
        
        self.eval_config = config.get('evaluation', {})
        self.use_amp = config.get('amp', {}).get('enabled', False)
        self.amp_dtype = config.get('amp', {}).get('dtype', 'float16')
        
        if isinstance(self.amp_dtype, str):
            if self.amp_dtype == 'float16':
                self.amp_dtype = torch.float16
            elif self.amp_dtype == 'bfloat16':
                self.amp_dtype = torch.bfloat16
        
        if incontext:
            self.incontext_config = config.get('incontext_evaluation', {})
            self.ensemble_size = self.incontext_config.get('ensemble_size', 1)
        
        self.calculate_dice_split = calculate_dice_split
        self.calculate_surface_distance = (
            calculate_surface_distance
            if self.eval_config.get('calculate_surface_metrics', False)
            else None
        )
        
        self.sliding_window = self.eval_config.get('sliding_window', False)
        if self.sliding_window:
            self.window_size = self.eval_config.get('window_size', [128, 128, 128])
            self.overlap = self.eval_config.get('overlap', 0.5)
    
    def run(self, learnable_priors=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run evaluation.
        
        Args:
            learnable_priors: Optional learnable priors for finetuning evaluation
            
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        self.model.eval()
        
        self.sample_idx = 0
        self.learnable_priors = learnable_priors

        if learnable_priors is not None:
            return self._run_learnable_priors_evaluation()
        elif self.incontext:
            return self._run_incontext_evaluation()
        else:
            return self._run_standard_evaluation()
    
    def _run_standard_evaluation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run standard evaluation.
        
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        dice_list = []
        asd_list = []
        hd_list = []
        class_present = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating", disable=not is_master()):
                images, labels, tgt_idx, spacing = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        predictions = self._sliding_window_inference(images, tgt_idx)
                else:
                    predictions = self._sliding_window_inference(images, tgt_idx)
                
                metrics = self._calculate_metrics(predictions, labels, tgt_idx, spacing)
                
                if self.save_predictions:
                    self._save_sample(
                        images[0],
                        labels[0], 
                        predictions[0],
                        spacing[0] if isinstance(spacing, list) else spacing,
                        self.sample_idx
                    )
                    self.sample_idx += 1

                dice_list.append(metrics['dice'])
                asd_list.append(metrics['asd'])
                hd_list.append(metrics['hd'])
                class_present.append(metrics['class_present'])
        
        # Aggregate metrics across distributed processes
        if is_distributed():
            all_dice = self._gather_metrics(dice_list)
            all_asd = self._gather_metrics(asd_list)
            all_hd = self._gather_metrics(hd_list)
            all_present = self._gather_metrics(class_present)
        else:
            all_dice = dice_list
            all_asd = asd_list
            all_hd = hd_list
            all_present = class_present
        
        dice_means, asd_means, hd_means = self._compute_class_means(
            all_dice, all_asd, all_hd, all_present
        )
        
        return dice_means, asd_means, hd_means
    
    def _run_incontext_evaluation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run in-context evaluation using reference image-mask pairs stored in the dataset.
        
        This implementation is optimized for the Iris model architecture by processing multiple
        classes at once, improving efficiency over class-by-class processing.
        
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        self.model.eval()
        ensemble_size = getattr(self, 'ensemble_size', 1)
        
        max_classes_per_batch = self.eval_config.get('max_classes_per_batch', 8)
        
        dataset = self.data_loader.dataset
        
        if not hasattr(dataset, 'ref_img_dict') or not hasattr(dataset, 'ref_mask_dict'):
            raise RuntimeError("Dataset doesn't have reference image-mask dictionaries. Make sure you're using a dataset with in-context support.")
        
        all_classes = sorted(list(dataset.ref_img_dict.keys()))
        logging.info(f"Found {len(all_classes)} classes with references for evaluation")
        
        dice_list = []
        asd_list = []
        hd_list = []
        class_present = []
        
        # Encode each class once; every target volume reuses these task priors.
        class_embeddings = {}
        
        with torch.no_grad():
            
            for class_id in all_classes:
                
                num_refs = dataset.get_num_references_for_class(class_id)
                if num_refs == 0:
                    raise RuntimeError(f"No references available for class {class_id}")
                
                num_refs_to_use = min(num_refs, ensemble_size)
                
                embeddings = []
                for i in range(num_refs_to_use):
                    ref_img, ref_mask = dataset.get_reference_for_class(class_id, index=i)
                    ref_img = ref_img.to(self.device)
                    ref_mask = ref_mask.to(self.device)
                    
                    if self.save_predictions and self.save_dir:
                        self._save_reference_images(class_id, ref_img, ref_mask, i, self.save_dir)
                    

                    if self.use_amp:
                        with autocast(device_type='cuda', dtype=self.amp_dtype):
                            ref_feat_list = self.model.encode_image_feature(ref_img.unsqueeze(0))
                            embedding = self.model.encode_visual_prior(ref_feat_list, ref_mask.unsqueeze(0))
                    else:
                        ref_feat_list = self.model.encode_image_feature(ref_img.unsqueeze(0))
                        embedding = self.model.encode_visual_prior(ref_feat_list, ref_mask.unsqueeze(0))
                    embeddings.append(embedding)
                    
                    if ensemble_size == 1:
                        break
                
                if not embeddings:
                    raise RuntimeError(f"Failed to compute embeddings for class {class_id}")
                    
                if len(embeddings) > 1:
                    # Ensemble references are averaged at the task-token level.
                    class_embeddings[class_id] = self._average_task_embeddings(embeddings)
                else:
                    class_embeddings[class_id] = embeddings[0]
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc=f"Evaluation (ensemble={ensemble_size > 1})", disable=not is_master())):
                images, labels, class_targets, spacing = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]
                
                batch_classes = torch.unique(class_targets[class_targets >= 0]).cpu().numpy().astype(int)
                
                for cls_id in batch_classes:
                    if cls_id not in class_embeddings:
                        raise RuntimeError(f"Batch {batch_idx}: Class {cls_id} doesn't have embeddings")
                
                if not batch_classes.size:
                    raise RuntimeError(f"Batch {batch_idx}: No classes found in targets")
                
                batch_size = images.shape[0]
                spatial_dims = images.shape[2:]
                full_predictions = torch.zeros((batch_size, len(batch_classes), *spatial_dims), device=self.device)
                
                for i in range(0, len(batch_classes), max_classes_per_batch):
                    current_classes = batch_classes[i:i+max_classes_per_batch]
                    
                    num_scales = len(class_embeddings[current_classes[0]])
                    current_embeddings = []

                    for scale_idx in range(num_scales):
                        scale_embeddings = []
                        for cls_id in current_classes:
                            scale_embeddings.append(class_embeddings[cls_id][scale_idx])
                        
                        # Concatenate task embeddings so one forward pass
                        # predicts multiple class channels.
                        current_embeddings.append(torch.cat(scale_embeddings, dim=1))

                    if self.use_amp:
                        with autocast(device_type='cuda', dtype=self.amp_dtype):
                            combined_prediction = self._sliding_window_inference_with_prior(images, current_embeddings)
                            
                    else:
                        combined_prediction = self._sliding_window_inference_with_prior(images, current_embeddings)
                        
                        
                    
                    if combined_prediction.shape[1] != len(current_classes):
                        raise RuntimeError(
                            f"Model returned prediction with unexpected shape: {combined_prediction.shape}, "
                            f"expected second dimension to be {len(current_classes)}"
                        )
                    
                    for j, cls_id in enumerate(current_classes):
                        cls_idx = np.where(batch_classes == cls_id)[0][0]
                        full_predictions[:, cls_idx] = combined_prediction[:, j]
                
                if labels.shape[1] < len(batch_classes):
                    raise RuntimeError(
                        f"Labels have incompatible shape: {labels.shape}, "
                        f"expected at least {len(batch_classes)} in dimension 1"
                    )
                
                metrics = self._calculate_metrics(
                    full_predictions, 
                    labels[:, :len(batch_classes)], 
                    torch.tensor(batch_classes, device=self.device), 
                    spacing
                )
                

                if self.save_predictions:
                    self._save_sample(
                        images[0],
                        labels[0], 
                        full_predictions[0],
                        spacing[0] if isinstance(spacing, list) else spacing,
                        self.sample_idx
                    )
                    self.sample_idx += 1
                
                dice_list.append(metrics['dice'])
                asd_list.append(metrics['asd'])
                hd_list.append(metrics['hd'])
                class_present.append(metrics['class_present'])
        
        # Aggregate metrics across distributed processes
        if is_distributed():
            all_dice = self._gather_metrics(dice_list)
            all_asd = self._gather_metrics(asd_list)
            all_hd = self._gather_metrics(hd_list)
            all_present = self._gather_metrics(class_present)
        else:
            all_dice = dice_list
            all_asd = asd_list
            all_hd = hd_list
            all_present = class_present
        
        if not all_dice:
            raise RuntimeError("No metrics collected during evaluation")
        
        dice_means, asd_means, hd_means = self._compute_class_means(
            all_dice, all_asd, all_hd, all_present
        )
        
        return dice_means, asd_means, hd_means

    def _run_learnable_priors_evaluation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run evaluation using learnable priors for finetuning."""
        dice_list = []
        asd_list = []
        hd_list = []
        class_present = []
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating with learnable priors", disable=not is_master()):
                query_img = batch['query_img'].to(self.device)
                query_lab = batch['query_lab'].to(self.device)
                class_indices = batch['class_indices'].to(self.device)
                spacing = batch.get('spacing', [[1.0, 1.0, 1.0]])
                
               
                batch_size = query_img.size(0)
                
                for b in range(batch_size):
                    image = query_img[b:b+1]
                    label = query_lab[b:b+1]
                    sample_indices = class_indices[b]
                    sample_spacing = spacing[b] if isinstance(spacing, list) else spacing
                    
                    encoded_prior_list = []
                    for stage_priors in self.learnable_priors:
                        if sample_indices.numel() > 0:
                            min_index = int(sample_indices.min().item())
                            max_index = int(sample_indices.max().item())
                            if min_index < 0 or max_index >= stage_priors.shape[0]:
                                raise ValueError(
                                    "class_indices are out of bounds for "
                                    "learnable priors during evaluation: "
                                    f"valid=[0, {stage_priors.shape[0] - 1}], "
                                    f"got=[{min_index}, {max_index}]. Check "
                                    "that finetuning priors and validation "
                                    "foreground_classes use the same class set."
                                )
                        sample_priors = stage_priors[sample_indices]
                        sample_priors = sample_priors.unsqueeze(0)
                        encoded_prior_list.append(sample_priors)
                    
                    if self.sliding_window:
                        predictions = self._sliding_window_inference_with_prior(
                            image,
                            encoded_prior_list
                        )
                    else:
                        predictions = model.forward_with_encoded_prior(
                            tgt_img=image,
                            encoded_prior_list=encoded_prior_list
                        )
                    
                    metrics = self._calculate_metrics(
                        predictions,
                        label,
                        None,
                        sample_spacing
                    )
                    
                    dice_list.append(metrics['dice'])
                    asd_list.append(metrics['asd'])
                    hd_list.append(metrics['hd'])
                    class_present.append(metrics['class_present'])
        
        # Aggregate metrics across distributed processes
        if is_distributed():
            all_dice = self._gather_metrics(dice_list)
            all_asd = self._gather_metrics(asd_list)
            all_hd = self._gather_metrics(hd_list)
            all_present = self._gather_metrics(class_present)
        else:
            all_dice = dice_list
            all_asd = asd_list
            all_hd = hd_list
            all_present = class_present
        
        dice_means, asd_means, hd_means = self._compute_class_means(
            all_dice, all_asd, all_hd, all_present
        )
        
        return dice_means, asd_means, hd_means


    def _forward(self, images: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            images: Input images
            tgt_idx: Target indices
            
        Returns:
            Model predictions
        """

        output = self.model(images, None, None, tgt_idx, update_buffer=False)

        return torch.sigmoid(output)

    def _forward_with_prior(self, images: torch.Tensor, prior_embeddings_list: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the model with multiple class priors.
        
        Args:
            images: Input images
            prior_embeddings_list: List of lists of prior embeddings
            
        Returns:
            Model predictions
        """
        output = self.model.forward_with_encoded_prior(images, prior_embeddings_list)

        return torch.sigmoid(output)
    

    
    def _sliding_window_inference(self, images: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        """
        Perform sliding window inference for large volumes.
        If the volume is smaller than the window size, the volume is padded to the window size.
        
        Args:
            images: Input images
            tgt_idx: Target indices
            
        Returns:
            Predictions for the entire volume
        """
        B, C, D, H, W = images.shape
        window_size = self.window_size
        overlap = self.overlap
        
        flag = False
        if D < window_size[0] or H < window_size[1] or W < window_size[2]:
            # Pad small volumes to the trained window size, then crop back.
            flag = True
            diff_D = max(0, window_size[0] - D)
            diff_H = max(0, window_size[1] - H)
            diff_W = max(0, window_size[2] - W)
            
            images = F.pad(images, (0, diff_W, 0, diff_H, 0, diff_D))
            origin_D, origin_H, origin_W = D, H, W
            B, C, D, H, W = images.shape
        
        stride_d = int(window_size[0] * (1 - overlap))
        stride_h = int(window_size[1] * (1 - overlap))
        stride_w = int(window_size[2] * (1 - overlap))
        
        half_win_d = stride_d
        half_win_h = stride_h
        half_win_w = stride_w
        
        n_classes = tgt_idx.shape[1]
        
        output = torch.zeros((B, n_classes, D, H, W), device=images.device)
        count_map = torch.zeros((B, 1, D, H, W), device=images.device)
        
        n_windows_d = max(1, D // half_win_d) if half_win_d > 0 else 1
        n_windows_h = max(1, H // half_win_h) if half_win_h > 0 else 1
        n_windows_w = max(1, W // half_win_w) if half_win_w > 0 else 1
        
        with torch.no_grad():
            for d in range(n_windows_d):
                for h in range(n_windows_h):
                    for w in range(n_windows_w):
                        # Use split_idx to ensure no gaps
                        d_start, d_end = split_idx(half_win_d, D, window_size[0], d)
                        h_start, h_end = split_idx(half_win_h, H, window_size[1], h)
                        w_start, w_end = split_idx(half_win_w, W, window_size[2], w)
                        
                        window = images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        window_output = self._forward(window, tgt_idx)
                        
                        output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += window_output
                        count_map[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        # Average overlapping windows instead of letting dense regions dominate.
        output = output / (count_map + 1e-8)
        
        if flag:
            output = output[:, :, :origin_D, :origin_H, :origin_W]
        
        return output


    def _sliding_window_inference_with_prior(
        self, 
        images: torch.Tensor, 
        prior_embeddings_list: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Perform sliding window inference with multiple class priors at once.
        If the volume is smaller than the window size, the volume is padded to the window size.
        
        Args:
            images: Input images tensor of shape [B, C, D, H, W]
            prior_embeddings_list: List of lists of prior embeddings
            
        Returns:
            Predictions for all classes for the entire volume
        """
        B, C, D, H, W = images.shape
        window_size = self.window_size
        overlap = self.overlap
        
        flag = False
        if D < window_size[0] or H < window_size[1] or W < window_size[2]:
            # Pad small volumes to the trained window size, then crop back.
            flag = True
            diff_D = max(0, window_size[0] - D)
            diff_H = max(0, window_size[1] - H)
            diff_W = max(0, window_size[2] - W)
            
            images = F.pad(images, (0, diff_W, 0, diff_H, 0, diff_D))
            origin_D, origin_H, origin_W = D, H, W
            B, C, D, H, W = images.shape
        
        num_classes = prior_embeddings_list[0].shape[1]
        
        stride_d = int(window_size[0] * (1 - overlap))
        stride_h = int(window_size[1] * (1 - overlap))
        stride_w = int(window_size[2] * (1 - overlap))
        
        half_win_d = stride_d
        half_win_h = stride_h
        half_win_w = stride_w
        
        output = torch.zeros((B, num_classes, D, H, W), device=images.device)
        count_map = torch.zeros((B, 1, D, H, W), device=images.device)
        
        n_windows_d = max(1, D // half_win_d) if half_win_d > 0 else 1
        n_windows_h = max(1, H // half_win_h) if half_win_h > 0 else 1
        n_windows_w = max(1, W // half_win_w) if half_win_w > 0 else 1
        
        with torch.no_grad():
            for d in range(n_windows_d):
                for h in range(n_windows_h):
                    for w in range(n_windows_w):
                        # Use split_idx to ensure no gaps
                        d_start, d_end = split_idx(half_win_d, D, window_size[0], d)
                        h_start, h_end = split_idx(half_win_h, H, window_size[1], h)
                        w_start, w_end = split_idx(half_win_w, W, window_size[2], w)
                        
                        window = images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        window_output = self._forward_with_prior(window, prior_embeddings_list)
                        
                        output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += window_output
                        count_map[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        # Average overlapping windows instead of letting dense regions dominate.
        output = output / (count_map + 1e-8)
        
        if flag:
            output = output[:, :, :origin_D, :origin_H, :origin_W]
        
        return output


    def _calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        tgt_idx: torch.Tensor,
        spacing: Union[List[float], torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            tgt_idx: Target indices
            spacing: Voxel spacing in array-axis order (z, y, x)
            
        Returns:
            Dictionary of calculated metrics
        """
      
        pred_bin = (predictions >= 0.5).to(torch.int8)
        
        # tgt_idx may include -1 padding, so metrics only use real task slots.
        C = torch.max(torch.nonzero(tgt_idx + 1)).item() + 1 if tgt_idx is not None else pred_bin.shape[1]
        
        # Limit predictions and labels to present classes
        pred_bin = pred_bin[:, :C]
        labels = labels[:, :C]
        
        pred_flat = pred_bin.reshape(C, -1)
        label_flat = labels.reshape(C, -1)
        
        dice, _, _ = self.calculate_dice_split(pred_flat, label_flat, C)
        
        class_present = torch.max(label_flat, dim=1)[0].cpu().numpy()
        
        if self.calculate_surface_distance is not None:
            spacing = self._normalize_spacing(spacing)
            
            asd = np.zeros(C)
            hd = np.zeros(C)
            
            for c in range(C):
                if class_present[c]:
                    try:
                        surface_dist = self.calculate_surface_distance(
                            pred_bin[0, c].cpu().numpy(),
                            labels[0, c].cpu().numpy(),
                            spacing
                        )
                        
                        asd[c] = surface_dist['mean_surface_distance']
                        hd[c] = surface_dist['hausdorff_distance_95']
                    except Exception as e:
                        logging.warning(f"Error calculating surface distance for class {c}: {e}")
                        asd[c] = float('nan')
                        hd[c] = float('nan')
                else:
                    # Surface metrics are undefined for absent classes.
                    asd[c] = float('nan')
                    hd[c] = float('nan')
            
            # Replace NaN with a large value
            asd = np.nan_to_num(asd, nan=500.0)
            hd = np.nan_to_num(hd, nan=500.0)
            
            # Clip to reasonable values
            asd = np.clip(asd, 0, 500.0)
            hd = np.clip(hd, 0, 500.0)
        else:
            asd = np.zeros(C)
            hd = np.zeros(C)
        
        return {
            'dice': dice.cpu().numpy(),
            'asd': asd,
            'hd': hd,
            'class_present': class_present
        }

    def _normalize_spacing(self, spacing: Union[List[float], torch.Tensor]) -> List[float]:
        """Normalize DataLoader-collated spacing to a 3-value list in array-axis order."""
        values = []

        if isinstance(spacing, torch.Tensor):
            values = np.asarray(spacing.detach().cpu()).reshape(-1).tolist()
        elif isinstance(spacing, np.ndarray):
            values = spacing.reshape(-1).tolist()
        elif isinstance(spacing, (list, tuple)):
            for value in spacing:
                if isinstance(value, torch.Tensor):
                    value = np.asarray(value.detach().cpu()).reshape(-1).tolist()
                elif isinstance(value, np.ndarray):
                    value = value.reshape(-1).tolist()
                elif isinstance(value, tuple):
                    value = list(value)

                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
        else:
            values = [spacing]

        values = [float(value) for value in values]

        if len(values) == 1:
            return [values[0]] * 3
        if len(values) >= 3:
            return values[:3]
        return [1.0, 1.0, 1.0]
    
    def _gather_metrics(self, metric_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Gather metrics from all distributed processes.
        
        Args:
            metric_list: List of metric arrays from the current process
            
        Returns:
            Aggregated list of metric arrays from all processes
        """
        if not is_distributed():
            return metric_list
        
        gathered_metrics = all_gather_object(metric_list)
        
        flattened_metrics = []
        for metrics in gathered_metrics:
            flattened_metrics.extend(metrics)
        
        return flattened_metrics
    
    def _compute_class_means(
        self,
        dice_list: List[np.ndarray],
        asd_list: List[np.ndarray],
        hd_list: List[np.ndarray],
        class_present_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean metrics for each class.
        
        Args:
            dice_list: List of Dice coefficients for each sample
            asd_list: List of average surface distances for each sample
            hd_list: List of Hausdorff distances for each sample
            class_present_list: List of boolean arrays indicating class presence
            
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        if not dice_list:
            return np.array([]), np.array([]), np.array([])
        
        num_classes = dice_list[0].shape[0]
        
        dice_means = np.zeros(num_classes)
        asd_means = np.zeros(num_classes)
        hd_means = np.zeros(num_classes)
        
        # Count how many samples have each class present
        class_counts = np.zeros(num_classes)
        
        # Sum metrics for each class
        for dice, asd, hd, present in zip(dice_list, asd_list, hd_list, class_present_list):
            for c in range(num_classes):
                if present[c]:
                    dice_means[c] += dice[c]
                    asd_means[c] += asd[c]
                    hd_means[c] += hd[c]
                    class_counts[c] += 1
        
        for c in range(num_classes):
            if class_counts[c] > 0:
                dice_means[c] /= class_counts[c]
                asd_means[c] /= class_counts[c]
                hd_means[c] /= class_counts[c]
            else:
                dice_means[c] = 0.0
                asd_means[c] = 500.0  # Large value for non-existent classes
                hd_means[c] = 500.0
        
        return dice_means, asd_means, hd_means
    
    def _average_task_embeddings(self, embeddings: List) -> List:
        """
        Norm-weighted average of task embeddings for ensemble.
        
        References that produce stronger (higher L2-norm) prior tokens
        are considered more reliable and get higher weight. This reduces
        noise from weak reference examples.
        
        Args:
            embeddings: List of task embeddings, each a list of tensors per scale
            
        Returns:
            Norm-weighted averaged task embeddings
        """
        if not embeddings:
            raise ValueError("No embeddings to average")
        
        num_levels = len(embeddings[0])
        num_refs = len(embeddings)
        
        if num_refs == 1:
            return embeddings[0]
        
        # Compute per-reference quality scores from the deepest-scale tokens.
        # Deepest scale has the most semantic information.
        scores = []
        for emb in embeddings:
            # emb[0] is the deepest scale: [B, N, M, C]
            # Use mean L2 norm across all token dimensions as quality score.
            token_norms = torch.norm(emb[0], dim=-1).mean()  # scalar
            scores.append(token_norms.item())
        
        # Normalize scores to sum to 1 for weighted averaging.
        scores_t = torch.tensor(scores, device=embeddings[0][0].device)
        weights = torch.softmax(scores_t / max(scores_t), dim=0)  # soft normalization
        
        logging.info(f'Ensemble weights: {weights.tolist()} (norms: {[f"{s:.4f}" for s in scores]})')
        
        avg_embeddings = []
        for level in range(num_levels):
            level_tensors = [emb[level] for emb in embeddings]
            
            shapes = [tensor.shape for tensor in level_tensors]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError(f"Embedding shapes don't match at level {level}: {shapes}")
            
            # Weighted average
            weighted_sum = sum(w * t for w, t in zip(weights, level_tensors))
            avg_embeddings.append(weighted_sum)
        
        return avg_embeddings

    def _save_sample(
        self, 
        image: torch.Tensor, 
        label: torch.Tensor, 
        prediction: torch.Tensor, 
        spacing: Union[List[float], torch.Tensor],
        idx: int
    ):
        """
        Save image, label, and prediction as nii.gz files.
        
        Args:
            image: Input image tensor [C, D, H, W]
            label: Ground truth label tensor [C, D, H, W]
            prediction: Predicted label tensor [C, D, H, W] - contains sigmoid probabilities
            spacing: Voxel spacing in array-axis order (z, y, x)
            idx: Sample index
        """
        image_np = image.cpu().numpy()
        label_np = label.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy().tolist()
        elif isinstance(spacing, np.ndarray):
            spacing = spacing.tolist()
        
        spacing_itk = spacing[::-1] if len(spacing) == 3 else [1.0, 1.0, 1.0]
        
        img_itk = self.sitk.GetImageFromArray(image_np[0].astype(np.float32))
        img_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(img_itk, str(self.save_dir / f'sample_{idx:04d}_image.nii.gz'))
        
        label_combined = np.zeros(label_np.shape[1:], dtype=np.uint8)
        pred_combined = np.zeros(pred_np.shape[1:], dtype=np.uint8)
        
        for c in range(label_np.shape[0]):
            label_combined[label_np[c] > 0] = c + 1
        
        for c in range(pred_np.shape[0]):
            pred_mask = pred_np[c] >= 0.5
            pred_combined[pred_mask] = c + 1
        
        label_itk = self.sitk.GetImageFromArray(label_combined)
        label_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(label_itk, str(self.save_dir / f'sample_{idx:04d}_label.nii.gz'))
        
        pred_itk = self.sitk.GetImageFromArray(pred_combined)
        pred_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(pred_itk, str(self.save_dir / f'sample_{idx:04d}_prediction.nii.gz'))

    def _save_reference_images(self, class_id, ref_img, ref_mask, ref_index, save_dir):
        """
        Save reference image and mask pair used for in-context evaluation.
        
        Args:
            class_id: Class ID of the reference
            ref_img: Reference image tensor 
            ref_mask: Reference mask tensor
            ref_index: Index of the reference (for multiple references per class)
            save_dir: Directory to save the reference images
        """
        ref_dir = save_dir / 'reference_images'
        ref_dir.mkdir(exist_ok=True)
        
        ref_img_np = ref_img.cpu().numpy()
        ref_mask_np = ref_mask.cpu().numpy()
        
        logging.debug(f"Class {class_id} ref {ref_index} - Original shapes: "
                    f"Image: {ref_img_np.shape}, Mask: {ref_mask_np.shape}")
        
        if ref_img_np.ndim == 4:
            ref_img_3d = ref_img_np[0] if ref_img_np.shape[0] == 1 or ref_img_np.shape[0] <= 4 else ref_img_np[0]
        elif ref_img_np.ndim == 3:
            ref_img_3d = ref_img_np
        else:
            ref_img_3d = np.squeeze(ref_img_np)
            if ref_img_3d.ndim != 3:
                raise ValueError(f"Cannot convert image to 3D. Shape: {ref_img_np.shape}")
        
        if ref_mask_np.ndim == 4:
            ref_mask_3d = ref_mask_np[0] if ref_mask_np.shape[0] == 1 else ref_mask_np[0]
        elif ref_mask_np.ndim == 3:
            ref_mask_3d = ref_mask_np
        else:
            ref_mask_3d = np.squeeze(ref_mask_np)
            if ref_mask_3d.ndim != 3:
                raise ValueError(f"Cannot convert mask to 3D. Shape: {ref_mask_np.shape}")
        
        if ref_img_3d.shape != ref_mask_3d.shape:
            logging.error(f"Shape mismatch after processing - Image: {ref_img_3d.shape}, Mask: {ref_mask_3d.shape}")
            ref_img_3d = np.squeeze(ref_img_3d)
            ref_mask_3d = np.squeeze(ref_mask_3d)
            
            if ref_img_3d.shape != ref_mask_3d.shape:
                raise ValueError(f"Cannot match dimensions - Image: {ref_img_3d.shape}, Mask: {ref_mask_3d.shape}")
        
        spacing = (1.0, 1.0, 1.0)
        
        img_itk = self.sitk.GetImageFromArray(ref_img_3d)
        img_itk.SetSpacing(spacing)
        img_path = ref_dir / f'class_{class_id:03d}_ref_{ref_index:02d}_image.nii.gz'
        self.sitk.WriteImage(img_itk, str(img_path))
        
        mask_itk = self.sitk.GetImageFromArray(ref_mask_3d.astype(np.uint8))
        mask_itk.SetSpacing(spacing)
        mask_path = ref_dir / f'class_{class_id:03d}_ref_{ref_index:02d}_mask.nii.gz'
        self.sitk.WriteImage(mask_itk, str(mask_path))
        
        logging.info(f"Saved reference for class {class_id} (ref {ref_index}) - Shape: {ref_img_3d.shape}")

def split_idx(stride, size, window_size, i):
    """Calculate start and end indices ensuring no voxels are missed."""
    start_idx = stride * i
    end_idx = start_idx + window_size
    
    if end_idx > size:
        start_idx = size - window_size
        end_idx = size
    
    return start_idx, end_idx




class RegularEvaluator:
    """
    Evaluator for downstream segmentation finetuning models.
    
    Simplified version without in-context learning or learnable priors.
    Supports:
    - Standard evaluation with sliding window inference
    - Distributed evaluation
    - Mixed precision evaluation
    - Per-class metrics reporting
    - Optional prediction saving
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        save_predictions: bool = False,
        save_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation
            config: Configuration dictionary
            device: Device to run evaluation on
            save_predictions: Whether to save predictions
            save_dir: Directory to save predictions
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device or torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        
        self.save_predictions = save_predictions
        self.save_dir = save_dir
        
        if self.save_predictions and self.save_dir is None:
            raise ValueError("save_dir must be provided if save_predictions is True")
        if self.save_predictions:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            import SimpleITK as sitk
            self.sitk = sitk
        
        self.eval_config = config.get('evaluation', {})
        self.use_amp = config.get('amp', {}).get('enabled', False)
        self.amp_dtype = config.get('amp', {}).get('dtype', 'float16')
        
        if isinstance(self.amp_dtype, str):
            if self.amp_dtype == 'float16':
                self.amp_dtype = torch.float16
            elif self.amp_dtype == 'bfloat16':
                self.amp_dtype = torch.bfloat16
        
        self.calculate_dice_split = calculate_dice_split
        self.calculate_surface_distance = (
            calculate_surface_distance
            if self.eval_config.get('calculate_surface_metrics', False)
            else None
        )
        
        # Sliding window parameters
        self.sliding_window = self.eval_config.get('sliding_window', False)
        if self.sliding_window:
            self.window_size = self.eval_config.get('window_size', [128, 128, 128])
            self.overlap = self.eval_config.get('overlap', 0.5)
    
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run evaluation.
        
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        self.model.eval()
        self.sample_idx = 0
        
        dice_list = []
        asd_list = []
        hd_list = []
        class_present = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating", disable=not is_master()):
                query_img = batch['query_img'].to(self.device)
                query_lab = batch['query_lab'].to(self.device)
                spacing = batch.get('spacing', [[1.0, 1.0, 1.0]])
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        predictions = self._sliding_window_inference(query_img)
                else:
                    predictions = self._sliding_window_inference(query_img)
                
                metrics = self._calculate_metrics(predictions, query_lab, spacing)
                
                if self.save_predictions:
                    sample_spacing = spacing[0] if isinstance(spacing, list) else spacing
                    self._save_sample(
                        query_img[0],
                        query_lab[0], 
                        predictions[0],
                        sample_spacing,
                        self.sample_idx
                    )
                    self.sample_idx += 1
                
                dice_list.append(metrics['dice'])
                asd_list.append(metrics['asd'])
                hd_list.append(metrics['hd'])
                class_present.append(metrics['class_present'])
        
        # Aggregate metrics across distributed processes
        if is_distributed():
            all_dice = self._gather_metrics(dice_list)
            all_asd = self._gather_metrics(asd_list)
            all_hd = self._gather_metrics(hd_list)
            all_present = self._gather_metrics(class_present)
        else:
            all_dice = dice_list
            all_asd = asd_list
            all_hd = hd_list
            all_present = class_present
        
        dice_means, asd_means, hd_means = self._compute_class_means(
            all_dice, all_asd, all_hd, all_present
        )
        
        return dice_means, asd_means, hd_means
    
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            images: Input images [B, C, D, H, W]
            
        Returns:
            Model predictions [B, num_classes, D, H, W] with sigmoid applied
        """
        output = self.model(images)

        return torch.sigmoid(output)
    
    def _sliding_window_inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform sliding window inference for large volumes.
        If the volume is smaller than the window size, the volume is padded to the window size.
        
        Args:
            images: Input images [B, C, D, H, W]
            
        Returns:
            Predictions for the entire volume [B, num_classes, D, H, W]
        """
        if not self.sliding_window:
            return self._forward(images)
        
        B, C, D, H, W = images.shape
        window_size = self.window_size
        overlap = self.overlap
        
        flag = False
        if D < window_size[0] or H < window_size[1] or W < window_size[2]:
            # Pad small volumes to the trained window size, then crop back.
            flag = True
            diff_D = max(0, window_size[0] - D)
            diff_H = max(0, window_size[1] - H)
            diff_W = max(0, window_size[2] - W)
            
            images = F.pad(images, (0, diff_W, 0, diff_H, 0, diff_D))
            origin_D, origin_H, origin_W = D, H, W
            B, C, D, H, W = images.shape
        
        stride_d = int(window_size[0] * (1 - overlap))
        stride_h = int(window_size[1] * (1 - overlap))
        stride_w = int(window_size[2] * (1 - overlap))
        
        half_win_d = stride_d
        half_win_h = stride_h
        half_win_w = stride_w
        
        with torch.no_grad():
            # Test forward pass with a small crop
            test_window = images[:, :, :window_size[0], :window_size[1], :window_size[2]]
            test_output = self._forward(test_window)
            n_classes = test_output.shape[1]
        
        output = torch.zeros((B, n_classes, D, H, W), device=images.device)
        count_map = torch.zeros((B, 1, D, H, W), device=images.device)
        
        n_windows_d = max(1, D // half_win_d) if half_win_d > 0 else 1
        n_windows_h = max(1, H // half_win_h) if half_win_h > 0 else 1
        n_windows_w = max(1, W // half_win_w) if half_win_w > 0 else 1
        
        with torch.no_grad():
            for d in range(n_windows_d):
                for h in range(n_windows_h):
                    for w in range(n_windows_w):
                        # Use split_idx to ensure no gaps
                        d_start, d_end = split_idx(half_win_d, D, window_size[0], d)
                        h_start, h_end = split_idx(half_win_h, H, window_size[1], h)
                        w_start, w_end = split_idx(half_win_w, W, window_size[2], w)
                        
                        window = images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        window_output = self._forward(window)
                        
                        output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += window_output
                        count_map[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        # Average overlapping windows instead of letting dense regions dominate.
        output = output / (count_map + 1e-8)
        
        if flag:
            output = output[:, :, :origin_D, :origin_H, :origin_W]
        
        return output
    
    def _calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        spacing: Union[List[float], torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions [B, num_classes, D, H, W]
            labels: Ground truth labels [B, num_classes, D, H, W]
            spacing: Voxel spacing in array-axis order (z, y, x)
            
        Returns:
            Dictionary of calculated metrics
        """
        pred_bin = (predictions >= 0.5).to(torch.int8)
        
        C = pred_bin.shape[1]
        
        pred_flat = pred_bin.reshape(C, -1)
        label_flat = labels.reshape(C, -1)
        
        dice, _, _ = self.calculate_dice_split(pred_flat, label_flat, C)
        
        class_present = torch.max(label_flat, dim=1)[0].cpu().numpy()
        
        if self.calculate_surface_distance is not None:
            spacing = self._normalize_spacing(spacing)
            
            asd = np.zeros(C)
            hd = np.zeros(C)
            
            for c in range(C):
                if class_present[c]:
                    try:
                        surface_dist = self.calculate_surface_distance(
                            pred_bin[0, c].cpu().numpy(),
                            labels[0, c].cpu().numpy(),
                            spacing
                        )
                        
                        asd[c] = surface_dist['mean_surface_distance']
                        hd[c] = surface_dist['hausdorff_distance_95']
                    except Exception as e:
                        logging.warning(f"Error calculating surface distance for class {c}: {e}")
                        asd[c] = float('nan')
                        hd[c] = float('nan')
                else:
                    # Surface metrics are undefined for absent classes.
                    asd[c] = float('nan')
                    hd[c] = float('nan')
            
            # Replace NaN with a large value
            asd = np.nan_to_num(asd, nan=500.0)
            hd = np.nan_to_num(hd, nan=500.0)
            
            # Clip to reasonable values
            asd = np.clip(asd, 0, 500.0)
            hd = np.clip(hd, 0, 500.0)
        else:
            asd = np.zeros(C)
            hd = np.zeros(C)
        
        return {
            'dice': dice.cpu().numpy(),
            'asd': asd,
            'hd': hd,
            'class_present': class_present
        }

    def _normalize_spacing(self, spacing: Union[List[float], torch.Tensor]) -> List[float]:
        """Normalize DataLoader-collated spacing to a 3-value list in array-axis order."""
        values = []

        if isinstance(spacing, torch.Tensor):
            values = np.asarray(spacing.detach().cpu()).reshape(-1).tolist()
        elif isinstance(spacing, np.ndarray):
            values = spacing.reshape(-1).tolist()
        elif isinstance(spacing, (list, tuple)):
            for value in spacing:
                if isinstance(value, torch.Tensor):
                    value = np.asarray(value.detach().cpu()).reshape(-1).tolist()
                elif isinstance(value, np.ndarray):
                    value = value.reshape(-1).tolist()
                elif isinstance(value, tuple):
                    value = list(value)

                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
        else:
            values = [spacing]

        values = [float(value) for value in values]

        if len(values) == 1:
            return [values[0]] * 3
        if len(values) >= 3:
            return values[:3]
        return [1.0, 1.0, 1.0]
    
    def _gather_metrics(self, metric_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Gather metrics from all distributed processes.
        
        Args:
            metric_list: List of metric arrays from the current process
            
        Returns:
            Aggregated list of metric arrays from all processes
        """
        if not is_distributed():
            return metric_list
        
        gathered_metrics = all_gather_object(metric_list)
        
        flattened_metrics = []
        for metrics in gathered_metrics:
            flattened_metrics.extend(metrics)
        
        return flattened_metrics
    
    def _compute_class_means(
        self,
        dice_list: List[np.ndarray],
        asd_list: List[np.ndarray],
        hd_list: List[np.ndarray],
        class_present_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean metrics for each class.
        
        Args:
            dice_list: List of Dice coefficients for each sample
            asd_list: List of average surface distances for each sample
            hd_list: List of Hausdorff distances for each sample
            class_present_list: List of boolean arrays indicating class presence
            
        Returns:
            Tuple of (dice_means, asd_means, hd_means) as numpy arrays
        """
        if not dice_list:
            return np.array([]), np.array([]), np.array([])
        
        num_classes = dice_list[0].shape[0]
        
        dice_means = np.zeros(num_classes)
        asd_means = np.zeros(num_classes)
        hd_means = np.zeros(num_classes)
        
        # Count how many samples have each class present
        class_counts = np.zeros(num_classes)
        
        # Sum metrics for each class
        for dice, asd, hd, present in zip(dice_list, asd_list, hd_list, class_present_list):
            for c in range(num_classes):
                if present[c]:
                    dice_means[c] += dice[c]
                    asd_means[c] += asd[c]
                    hd_means[c] += hd[c]
                    class_counts[c] += 1
        
        for c in range(num_classes):
            if class_counts[c] > 0:
                dice_means[c] /= class_counts[c]
                asd_means[c] /= class_counts[c]
                hd_means[c] /= class_counts[c]
            else:
                dice_means[c] = 0.0
                asd_means[c] = 500.0  # Large value for non-existent classes
                hd_means[c] = 500.0
        
        return dice_means, asd_means, hd_means
    
    def _save_sample(
        self, 
        image: torch.Tensor, 
        label: torch.Tensor, 
        prediction: torch.Tensor, 
        spacing: Union[List[float], torch.Tensor],
        idx: int
    ):
        """
        Save image, label, and prediction as nii.gz files.
        
        Args:
            image: Input image tensor [C, D, H, W]
            label: Ground truth label tensor [num_classes, D, H, W]
            prediction: Predicted label tensor [num_classes, D, H, W] - contains sigmoid probabilities
            spacing: Voxel spacing in array-axis order (z, y, x)
            idx: Sample index
        """
        image_np = image.cpu().numpy()
        label_np = label.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy().tolist()
        elif isinstance(spacing, np.ndarray):
            spacing = spacing.tolist()
        
        spacing_itk = spacing[::-1] if len(spacing) == 3 else [1.0, 1.0, 1.0]
        
        img_itk = self.sitk.GetImageFromArray(image_np[0].astype(np.float32))
        img_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(img_itk, str(self.save_dir / f'sample_{idx:04d}_image.nii.gz'))
        
        label_combined = np.zeros(label_np.shape[1:], dtype=np.uint8)
        pred_combined = np.zeros(pred_np.shape[1:], dtype=np.uint8)
        
        for c in range(label_np.shape[0]):
            label_combined[label_np[c] > 0] = c + 1
        
        for c in range(pred_np.shape[0]):
            pred_mask = pred_np[c] >= 0.5
            pred_combined[pred_mask] = c + 1
        
        label_itk = self.sitk.GetImageFromArray(label_combined)
        label_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(label_itk, str(self.save_dir / f'sample_{idx:04d}_label.nii.gz'))
        
        pred_itk = self.sitk.GetImageFromArray(pred_combined)
        pred_itk.SetSpacing(spacing_itk)
        self.sitk.WriteImage(pred_itk, str(self.save_dir / f'sample_{idx:04d}_prediction.nii.gz'))
