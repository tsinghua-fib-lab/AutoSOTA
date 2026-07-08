"""Core component."""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional, Tuple
from pathlib import Path
import csv


class ClusterStructureAnalyzer:
    """Handle init."""
    
    def __init__(
        self,
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        args: Dict,
    ):
        self.feature_extractor = feature_extractor
        self.device = device
        self.args = args
        self._cluster_samples: Dict[int, Dict[str, torch.Tensor]] = {}
        self._procrustes_distances: Dict[int, Dict[int, float]] = {}
    
    def save_task1_samples(
        self,
        dataset_loader: Callable[[], torch.utils.data.Dataset],
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        
        init_cls = self.args.get("init_cls", 10)
        task1_dataset = dataset_loader()
        task1_loader = DataLoader(
            task1_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        
        images_dict = {} 
        feats_dict = {} 
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(task1_loader):
                inputs = inputs.to(self.device)
                targets = targets.cpu().numpy()
                
                features = self.feature_extractor(inputs)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                
                for i in range(len(targets)):
                    cls = int(targets[i])
                    if cls not in images_dict:
                        images_dict[cls] = []
                        feats_dict[cls] = []
                    images_dict[cls].append(inputs[i].cpu())
                    feats_dict[cls].append(features[i])
        
        for cls in images_dict:
            images_tensor = torch.stack(images_dict[cls])
            feats_array = np.array(feats_dict[cls])
            
            # feats_array_norm = feats_array / (np.linalg.norm(feats_array, axis=1, keepdims=True) + 1e-8)
            
            self._cluster_samples[cls] = {
                'images': images_tensor,
                'feats_task1': feats_array.copy()  # Raw features (deep copy to prevent accidental modification)
            }
        
    
    def compute_procrustes_distances(self, cur_task: int) -> None:
        if len(self._cluster_samples) == 0:
            logging.warning("[Cluster Structure Analysis] No Task 1 samples found, skipping")
            return
        
        
        if cur_task not in self._procrustes_distances:
            self._procrustes_distances[cur_task] = {}
        
        distances_summary = []
        
        for cls, data in self._cluster_samples.items():
            images = data['images']
            feats_task1 = data['feats_task1']
            
            feats_current = []
            with torch.no_grad():
                batch_size = 32
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size].to(self.device)
                    batch_feats = self.feature_extractor(batch_images)
                    if isinstance(batch_feats, torch.Tensor):
                        batch_feats = batch_feats.cpu().numpy()
                    feats_current.append(batch_feats)
            
            feats_current = np.concatenate(feats_current, axis=0)
            
            if len(feats_task1) > 0 and len(feats_current) > 0:
                feat1_first_val = feats_task1[0, 0] if feats_task1.ndim > 1 else feats_task1[0]
                feat_curr_first_val = feats_current[0, 0] if feats_current.ndim > 1 else feats_current[0]
                feat_diff = np.abs(feat1_first_val - feat_curr_first_val)
                feat1_mean = np.mean(feats_task1)
                feat_curr_mean = np.mean(feats_current)
                feat_mean_diff = np.abs(feat1_mean - feat_curr_mean)
                
                are_identical = np.allclose(feats_task1, feats_current, rtol=1e-8, atol=1e-8)
                
                
                if are_identical:
                    logging.error(
                        f"[Cluster Structure Analysis ERROR] Class {cls}: "
                        f"feats_task1 and feats_current are IDENTICAL! "
                        f"This indicates a bug: either feats_task1 was incorrectly updated, "
                        f"or feature_extractor is not using the current task's model weights."
                    )
            
            # feats_current_norm = feats_current / (np.linalg.norm(feats_current, axis=1, keepdims=True) + 1e-8)
            
            procrustes_dist, scale, angle_rad = self._compute_procrustes_distance(feats_task1, feats_current)
            self._procrustes_distances[cur_task][cls] = procrustes_dist
            
            distances_summary.append({
                'task': cur_task + 1,
                'class': cls,
                'procrustes_distance': procrustes_dist,
                'num_samples': len(feats_task1),
                'scale': scale,
                'angle': angle_rad
            })
            
            logging.info(
                f"[Cluster Structure Analysis] T{cur_task + 1}, C{cls}: "
                f"Dist={procrustes_dist:.4f}, Scale={scale:.4f}, Angle={angle_rad:.4f}"
            )
        
        if len(distances_summary) > 0:
            avg_distance = np.mean([d['procrustes_distance'] for d in distances_summary])
            avg_scale = np.mean([d['scale'] for d in distances_summary])
            logging.info(
                f"[Cluster Structure Analysis] Task {cur_task + 1} Avg Dist: {avg_distance:.4f}, Avg Scale: {avg_scale:.4f}"
            )
        
        self._save_procrustes_results()
    
    def _compute_procrustes_distance(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float, float]:
        """Handle compute procrustes distance."""
        if X1.shape != X2.shape:
            raise ValueError(f"Shape mismatch: X1 {X1.shape} vs X2 {X2.shape}")
        if X1.shape[0] < 2:
            return 0.0, 1.0, 0.0
        
        are_inputs_identical = np.allclose(X1, X2, rtol=1e-8, atol=1e-8)
        
        mu1 = X1.mean(axis=0)
        mu2 = X2.mean(axis=0)
        X1_centered = X1 - mu1
        X2_centered = X2 - mu2
        
        
        M = np.dot(X1_centered.T, X2_centered)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = np.dot(U, Vt)
        det_R_before = np.linalg.det(R)
        if det_R_before < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
            det_R_after = np.linalg.det(R)
            
        # s = trace(X2_centered.T @ X1_centered @ R) / trace(X1_centered.T @ X1_centered)
        # trace(A @ B) = sum(elementwise_multiplication)
        numerator = np.sum(S) # trace(S) comes from SVD
        denominator = np.sum(X1_centered ** 2)
        s = numerator / (denominator + 1e-8)
        
        X1_transformed = s * np.dot(X1_centered, R)
        residual = X2_centered - X1_transformed
        
        norm_factor = np.sqrt(np.sum(X2_centered**2)) + 1e-8
        residual_norm = np.linalg.norm(residual, ord='fro')
        procrustes_dist = residual_norm / norm_factor
        
        if procrustes_dist < 1e-6:
            pass  # Debug logging removed
        
        trace_R = np.trace(R)
        D = X1.shape[1]
        # cos(theta) = (Tr(R) - (D-2)) / 2  [Approx for high dim]
        cos_theta = (trace_R - (D - 2)) / 2.0
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        return float(procrustes_dist), float(s), float(angle)
    
    def _save_procrustes_results(self) -> None:
        if len(self._procrustes_distances) == 0: return
        
        init_cls = self.args.get("init_cls", 10)
        increment = self.args.get("increment", 10)
        dataset = self.args.get("dataset", "unknown")
        model_name = self.args.get("model_name", "unknown")
        
        log_dir = Path(f"logs/{model_name}/{dataset}/{init_cls}/{increment}")
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "procrustes_distances.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Class', 'Procrustes_Dist', 'Num_Samples'])
            for task, class_distances in sorted(self._procrustes_distances.items()):
                for cls, dist in sorted(class_distances.items()):
                    num_samples = len(self._cluster_samples[cls]['feats_task1'])
                    writer.writerow([task + 1, cls, f"{dist:.6f}", num_samples])