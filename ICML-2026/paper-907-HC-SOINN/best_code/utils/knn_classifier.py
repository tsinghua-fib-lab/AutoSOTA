import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist


class KNNClassifier:
    """Handle init."""

    def __init__(
        self,
        metric: str = "euclidean",
        use_all_samples: bool = True,
        k_neighbors: Optional[int] = None,
    ) -> None:
        """Handle init."""
        self.metric = metric
        self.use_all_samples = use_all_samples
        self.k_neighbors = k_neighbors
        self.class_to_features: Dict[int, List[np.ndarray]] = {}
        self.bank_feats_norm = None
        self.usage_counts: Dict[int, np.ndarray] = {}

    def clear(self) -> None:
        """Handle clear."""
        self.class_to_features.clear()
        self.bank_feats_norm = None
        self.usage_counts.clear()

    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Handle add features."""
        unique_labels = np.unique(labels)
        for cls in unique_labels:
            mask = labels == cls
            cls_feats = features[mask]
            if cls not in self.class_to_features:
                self.class_to_features[cls] = []
            self.class_to_features[cls].append(cls_feats)
            if cls not in self.usage_counts:
                total_nodes = sum(chunk.shape[0] for chunk in self.class_to_features[cls])
                self.usage_counts[cls] = np.zeros(total_nodes, dtype=np.int64)
            else:
                total_nodes = sum(chunk.shape[0] for chunk in self.class_to_features[cls])
                old_size = len(self.usage_counts[cls])
                if total_nodes > old_size:
                    new_counts = np.zeros(total_nodes, dtype=np.int64)
                    new_counts[:old_size] = self.usage_counts[cls]
                    self.usage_counts[cls] = new_counts

    def add_from_loader(self, loader, feature_fn, device: torch.device) -> None:
        """Handle add from loader."""
        feats, lbs = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                batch_feats = feature_fn(inputs).detach().cpu().numpy()
                feats.append(batch_feats)
                lbs.append(targets.numpy())
        feats_np = np.concatenate(feats) if len(feats) else np.zeros((0, 0))
        lbs_np = np.concatenate(lbs) if len(lbs) else np.zeros((0,), dtype=np.int64)
        if feats_np.shape[0] > 0:
            self.add_features(feats_np, lbs_np)
        logging.info(
            f"KNN bank updated. Classes in bank: {sorted(list(self.class_to_features.keys()))}"
        )

    def prune_zero_usage(self, class_ids: Optional[List[int]] = None) -> None:
        """Handle prune zero usage."""
        if class_ids is None:
            class_ids = list(self.class_to_features.keys())

        pruned_total = 0
        remaining_total = 0

        for cls in class_ids:
            if cls not in self.class_to_features or cls not in self.usage_counts:
                continue

            chunks = self.class_to_features[cls]
            if len(chunks) == 0:
                continue

            feats = np.concatenate(chunks, axis=0)  # [Nc, D]
            usage = self.usage_counts[cls]          # [Nc]

            if usage.shape[0] != feats.shape[0]:
                logging.warning(
                    f"KNN prune_zero_usage: mismatch for class {cls}, "
                    f"usage len={usage.shape[0]}, feats len={feats.shape[0]}. "
                    f"Skip pruning for this class."
                )
                continue

            mask = usage > 0
            pruned = int((~mask).sum())
            kept = int(mask.sum())

            pruned_total += pruned
            remaining_total += kept

            if kept == 0:
                self.class_to_features[cls] = []
                del self.usage_counts[cls]
            else:
                self.class_to_features[cls] = [feats[mask]]
                self.usage_counts[cls] = usage[mask]

        if pruned_total > 0:
            self.bank_feats_norm = None
            logging.info(
                f"KNN prune_zero_usage: pruned {pruned_total} nodes, "
                f"remaining {remaining_total} nodes in classes {class_ids}"
            )

    def _gather_bank(self):
        """Handle gather bank."""
        all_features, all_labels = [], []
        for cls, chunks in self.class_to_features.items():
            if len(chunks) == 0:
                continue
            feats = np.concatenate(chunks)
            labels = np.full((feats.shape[0],), cls, dtype=np.int64)
            all_features.append(feats)
            all_labels.append(labels)
        if len(all_features) == 0:
            return None, None
        bank_feats = np.concatenate(all_features)
        bank_labels = np.concatenate(all_labels)
        
        if self.metric == "cosine":
            norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
            self.bank_feats_norm = bank_feats / (norms + 1e-8)
        else:
            self.bank_feats_norm = None
        
        return bank_feats, bank_labels

    def _update_usage_counts(self, used_indices: np.ndarray, bank_labels: np.ndarray) -> None:
        """Handle update usage counts."""
        if len(self.usage_counts) == 0:
            return
        
        class_start_indices = {}
        current_idx = 0
        for cls in sorted(self.class_to_features.keys()):
            class_start_indices[cls] = current_idx
            class_mask = bank_labels == cls
            class_size = np.sum(class_mask)
            current_idx += class_size
        
        for query_indices in used_indices:
            for global_idx in query_indices:
                if global_idx < len(bank_labels):
                    cls = bank_labels[global_idx]
                    if cls in self.usage_counts and cls in class_start_indices:
                        class_start = class_start_indices[cls]
                        class_mask = bank_labels == cls
                        class_global_indices = np.where(class_mask)[0]
                        if global_idx in class_global_indices:
                            local_idx = np.where(class_global_indices == global_idx)[0][0]
                            if local_idx < len(self.usage_counts[cls]):
                                self.usage_counts[cls][local_idx] += 1

    def predict_topk(self, query_features: np.ndarray, topk: int, total_classes: int, device: Optional[torch.device] = None, track_usage: bool = False) -> np.ndarray:
        """Handle predict topk."""
        bank_feats, bank_labels = self._gather_bank()
        if bank_feats is None or bank_feats.shape[0] == 0:
            raise ValueError(
                "KNN feature bank is empty. Please ensure features are added to the bank "
                "before calling predict_topk (e.g., call add_from_loader or rebuild_from_loader)."
            )

        query_features = np.asarray(query_features)
        N = query_features.shape[0]
        M = bank_feats.shape[0]
        
        use_gpu = device is not None and torch.cuda.is_available() and self.metric in ["euclidean", "cosine"]
        
        # if use_gpu:
        #     logging.info(f"KNN inference using GPU: {device} (query_samples={N}, bank_samples={M}, metric={self.metric})")
        # else:
        #     device_str = str(device) if device is not None else "CPU"
        #     reason = "CUDA not available" if device is not None and not torch.cuda.is_available() else \
        #              f"metric '{self.metric}' not GPU-accelerated" if device is not None and self.metric not in ["euclidean", "cosine"] else \
        #              "device not specified"
        #     logging.info(f"KNN inference using CPU (query_samples={N}, bank_samples={M}, metric={self.metric}, reason={reason})")
        
        if use_gpu:
            query_t = torch.from_numpy(query_features).float().to(device)  # [N, D]
            bank_labels_t = torch.from_numpy(bank_labels).long().to(device)  # [M]
            
            if self.metric == "cosine":
                if self.bank_feats_norm is None:
                    norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
                    bank_feats_norm = bank_feats / (norms + 1e-8)
                else:
                    bank_feats_norm = self.bank_feats_norm
                
                bank_feats_t = torch.from_numpy(bank_feats_norm).float().to(device)  # [M, D]
                query_norm_t = torch.nn.functional.normalize(query_t, p=2, dim=1)  # [N, D]
                cosine_sims = torch.mm(query_norm_t, bank_feats_t.t())  # [N, M]
                dists = 1.0 - cosine_sims  # [N, M]
            else:  # euclidean
                bank_feats_t = torch.from_numpy(bank_feats).float().to(device)  # [M, D]
                # dists[i, j] = ||query[i] - bank[j]||^2
                dists = torch.cdist(query_t, bank_feats_t, p=2)  # [N, M]
            
            if self.use_all_samples or self.k_neighbors is None:
                nn_labels = bank_labels_t.unsqueeze(0).expand(N, -1)  # [N, M]
                nn_dists = dists  # [N, M]
                weights = 1.0 / (dists + 1e-8)  # [N, M]
            else:
                K = max(self.k_neighbors, topk)
                k_dists, k_indices = torch.topk(dists, k=min(K, M), dim=1, largest=False)  # [N, K]
                nn_labels = bank_labels_t[k_indices]  # [N, K]
                nn_dists = k_dists  # [N, K]
                weights = 1.0 / (k_dists + 1e-8)  # [N, K]
            
            nn_labels_cpu = nn_labels.cpu().numpy()  # [N, M] or [N, K]
            nn_dists_cpu = nn_dists.cpu().numpy()  # [N, M] or [N, K]
            weights_cpu = weights.cpu().numpy()  # [N, M] or [N, K]
            
            used_indices_list = []
            
            preds = []
            for i in range(N):
                votes: Dict[int, float] = {}
                min_dist: Dict[int, float] = {}
                
                if track_usage:
                    if self.use_all_samples or self.k_neighbors is None:
                        used_indices = np.arange(M)
                    else:
                        used_indices = k_indices[i].cpu().numpy()
                    used_indices_list.append(used_indices)
                
                for lbl, dist, weight in zip(nn_labels_cpu[i], nn_dists_cpu[i], weights_cpu[i]):
                    votes[lbl] = votes.get(lbl, 0.0) + weight
                    if lbl not in min_dist:
                        min_dist[lbl] = dist
                    else:
                        min_dist[lbl] = min(min_dist[lbl], dist)
                
                sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                top = sorted_classes[:topk]
                
                if len(top) < topk:
                    remaining = [c for c in range(total_classes) if c not in top]
                    top.extend(remaining[: topk - len(top)])
                
                preds.append(top)
            
            if track_usage and len(used_indices_list) > 0:
                used_indices_array = np.array(used_indices_list)
                self._update_usage_counts(used_indices_array, bank_labels)
            
            return np.array(preds, dtype=np.int64)
        else:
            if self.metric == "cosine":
                if self.bank_feats_norm is None:
                    norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
                    bank_feats_norm = bank_feats / (norms + 1e-8)
                else:
                    bank_feats_norm = self.bank_feats_norm
                
                query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
                query_norm = query_features / (query_norms + 1e-8)  # [N, D]
                
                cosine_sims = np.dot(query_norm, bank_feats_norm.T)  # [N, M]
                dists = 1.0 - cosine_sims  # [N, M]
            else:  # euclidean or other metrics
                dists = cdist(query_features, bank_feats, metric=self.metric)  # [N, M]
            
            if self.use_all_samples or self.k_neighbors is None:
                nn_labels = bank_labels  # [M]
                nn_dists = dists  # [N, M]
                weights = 1.0 / (dists + 1e-8)  # [N, M]
                
                used_indices_list = []
                
                preds = []
                for i in range(N):
                    votes: Dict[int, float] = {}
                    min_dist: Dict[int, float] = {}
                    
                    if track_usage:
                        used_indices = np.arange(M)
                        used_indices_list.append(used_indices)
                    
                    for lbl, dist, weight in zip(nn_labels, nn_dists[i], weights[i]):
                        votes[lbl] = votes.get(lbl, 0.0) + weight
                        if lbl not in min_dist:
                            min_dist[lbl] = dist
                        else:
                            min_dist[lbl] = min(min_dist[lbl], dist)
                    
                    sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                    top = sorted_classes[:topk]
                    
                    if len(top) < topk:
                        remaining = [c for c in range(total_classes) if c not in top]
                        top.extend(remaining[: topk - len(top)])
                    
                    preds.append(top)
                
                if track_usage and len(used_indices_list) > 0:
                    used_indices_array = np.array(used_indices_list)
                    self._update_usage_counts(used_indices_array, bank_labels)
                
                return np.array(preds, dtype=np.int64)
            else:
                K = max(self.k_neighbors, topk)
                k_indices = np.argpartition(dists, K-1, axis=1)[:, :K]  # [N, K]
                for i in range(N):
                    k_indices[i] = k_indices[i][np.argsort(dists[i, k_indices[i]])]
                
                k_labels = bank_labels[k_indices]  # [N, K]
                k_dists = np.take_along_axis(dists, k_indices, axis=1)  # [N, K]
                weights = 1.0 / (k_dists + 1e-8)  # [N, K]
                
                used_indices_list = []
                
                preds = []
                for i in range(N):
                    votes: Dict[int, float] = {}
                    min_dist: Dict[int, float] = {}
                    
                    if track_usage:
                        used_indices = k_indices[i]
                        used_indices_list.append(used_indices)
                    
                    for lbl, dist, weight in zip(k_labels[i], k_dists[i], weights[i]):
                        votes[lbl] = votes.get(lbl, 0.0) + weight
                        if lbl not in min_dist:
                            min_dist[lbl] = dist
                        else:
                            min_dist[lbl] = min(min_dist[lbl], dist)
                    
                    sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                    top = sorted_classes[:topk]
                    
                    if len(top) < topk:
                        remaining = [c for c in range(total_classes) if c not in top]
                        top.extend(remaining[: topk - len(top)])
                    
                    preds.append(top)
                
                if track_usage and len(used_indices_list) > 0:
                    used_indices_array = np.array(used_indices_list)
                    self._update_usage_counts(used_indices_array, bank_labels)
                
                return np.array(preds, dtype=np.int64)


