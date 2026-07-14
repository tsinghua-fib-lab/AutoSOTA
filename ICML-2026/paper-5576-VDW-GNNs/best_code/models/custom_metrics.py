"""
Custom metric classes used by VDW models.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torchmetrics import Metric


def _clustering_accuracy(
    y_true: torch.Tensor | np.ndarray, 
    y_pred: torch.Tensor | np.ndarray,
) -> float:
    """
    Compute clustering accuracy by optimally matching predicted clusters to true labels.
    """
    
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import contingency_matrix

    if isinstance(y_pred, torch.Tensor):
        pred_np = y_pred.cpu().numpy()
    else:
        pred_np = y_pred

    if y_true.numel() == 0 or pred_np.size == 0:
        return float('nan')
    if isinstance(y_true, torch.Tensor):
        true_np = y_true.cpu().numpy()
    else:
        true_np = y_true
    cont = contingency_matrix(true_np, pred_np)
    if cont.size == 0:
        return float('nan')
    # Maximize trace via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cont)
    correct = cont[row_ind, col_ind].sum()
    total = cont.sum()
    if total == 0:
        return float('nan')
    return float(correct) / float(total)


def compute_svm_probe_accuracy(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    eval_X: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    C: float = 0.1,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    degree: int = 3,
) -> torch.Tensor:
    """
    Fit an SVM on train embeddings and return accuracy on eval embeddings.
    """
    if (
        train_X.numel() == 0
        or train_y.numel() == 0
        or eval_X.numel() == 0
        or eval_y.numel() == 0
    ):
        return torch.tensor(float('nan'), dtype=torch.float32)

    try:
        
        from sklearn.svm import SVC

        train_y_np = train_y.cpu().numpy()
        if np.unique(train_y_np).size < 2:
            return torch.tensor(float('nan'), dtype=torch.float32)

        train_X_np = train_X.cpu().numpy()
        eval_X_np = eval_X.cpu().numpy()
        eval_y_np = eval_y.cpu().numpy()

        clf = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)
        clf.fit(train_X_np, train_y_np)
        acc = float(clf.score(eval_X_np, eval_y_np))
        return torch.tensor(acc, dtype=torch.float32)
    except Exception:
        return torch.tensor(float('nan'), dtype=torch.float32)


def compute_logistic_probe_accuracy(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    eval_X: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    solver: str = 'lbfgs',
    max_iter: int = 1000,
    C: float = 1.0,
) -> torch.Tensor:
    """
    Fit a logistic regression classifier on train embeddings and evaluate accuracy.
    """
    if (
        train_X.numel() == 0
        or train_y.numel() == 0
        or eval_X.numel() == 0
        or eval_y.numel() == 0
    ):
        return torch.tensor(float('nan'), dtype=torch.float32)

    try:
        
        from sklearn.linear_model import LogisticRegression

        train_y_np = train_y.cpu().numpy()
        if np.unique(train_y_np).size < 2:
            return torch.tensor(float('nan'), dtype=torch.float32)

        train_X_np = train_X.cpu().numpy()
        eval_X_np = eval_X.cpu().numpy()
        eval_y_np = eval_y.cpu().numpy()

        clf = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=C,
            multi_class='auto',
        )
        clf.fit(train_X_np, train_y_np)
        acc = float(clf.score(eval_X_np, eval_y_np))
        return torch.tensor(acc, dtype=torch.float32)
    except Exception:
        return torch.tensor(float('nan'), dtype=torch.float32)


def compute_knn_probe_accuracy(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    eval_X: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    k: int = 1,
) -> torch.Tensor:
    """
    Fit a k-nearest neighbors classifier on train embeddings and evaluate accuracy.

    Ties are broken by selecting, among the tied classes, the one whose nearest
    neighbor (smallest distance) appears first in the neighbor ranking.
    """
    if (
        train_X.numel() == 0
        or train_y.numel() == 0
        or eval_X.numel() == 0
        or eval_y.numel() == 0
    ):
        return torch.tensor(float('nan'), dtype=torch.float32)

    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be at least 1 for k-NN; received k={k_int}.")

    try:
        
        from sklearn.neighbors import NearestNeighbors

        train_features = train_X.cpu().numpy()
        eval_features = eval_X.cpu().numpy()
        train_labels = train_y.view(-1).cpu().numpy()
        eval_labels = eval_y.view(-1).cpu().numpy()

        if train_features.shape[0] == 0 or eval_features.shape[0] == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        n_neighbors = min(k_int, train_features.shape[0])
        if n_neighbors == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        neighbor_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='euclidean',
        )
        neighbor_model.fit(train_features)
        distances, indices = neighbor_model.kneighbors(
            eval_features,
            n_neighbors=n_neighbors,
            return_distance=True,
        )

        preds = np.empty(eval_labels.shape[0], dtype=train_labels.dtype)

        def _resolve_label(
            neighbor_labels: np.ndarray,
            neighbor_distances: np.ndarray,
        ) -> int:
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            candidate_labels = unique_labels[counts == max_count]
            if candidate_labels.size == 1:
                return candidate_labels[0]

            best_label = candidate_labels[0]
            best_distance = np.inf
            # Pick the label whose closest neighbor distance is smallest.
            for label in candidate_labels:
                label_mask = neighbor_labels == label
                label_distance = neighbor_distances[label_mask].min()
                if label_distance < best_distance:
                    best_distance = label_distance
                    best_label = label
                elif label_distance == best_distance:
                    # Tie on distances: fall back to earliest occurrence in neighbor ranking.
                    first_idx = np.argmax(label_mask)  # first True index
                    best_idx = np.argmax(neighbor_labels == best_label)
                    if first_idx < best_idx:
                        best_label = label
                        best_distance = label_distance
            return best_label

        for idx in range(eval_labels.shape[0]):
            neighbor_labels = train_labels[indices[idx]]
            neighbor_distances = distances[idx]
            preds[idx] = _resolve_label(neighbor_labels, neighbor_distances)

        if eval_labels.size == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        acc = float(np.mean(preds == eval_labels))
        return torch.tensor(acc, dtype=torch.float32)
    except Exception:
        return torch.tensor(float('nan'), dtype=torch.float32)


def compute_kmeans_probe_accuracy(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    eval_X: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 200,
    random_state: Optional[int] = 123456,
) -> torch.Tensor:
    """
    Fit k-means on training embeddings and score accuracy on evaluation embeddings.
    """
    if (
        train_X.numel() == 0
        or train_y.numel() == 0
        or eval_X.numel() == 0
        or eval_y.numel() == 0
    ):
        return torch.tensor(float('nan'), dtype=torch.float32)
    try:
        
        from sklearn.cluster import KMeans

        eval_labels = eval_y.view(-1)
        eval_labels_np = eval_labels.cpu().numpy()
        if np.unique(eval_labels_np).size < 2:
            return torch.tensor(float('nan'), dtype=torch.float32)

        train_np = train_X.cpu().numpy()
        eval_np = eval_X.cpu().numpy()
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        kmeans.fit(train_np)
        preds_np = kmeans.predict(eval_np)
        preds = torch.from_numpy(preds_np)
        acc = _clustering_accuracy(eval_labels, preds)
        return torch.tensor(acc, dtype=torch.float32)
    except Exception:
        return torch.tensor(float('nan'), dtype=torch.float32)


def compute_spectral_probe_accuracy(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    eval_X: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    n_clusters: int,
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 16,
    assign_labels: str = 'cluster_qr',
    n_jobs: int = 1,
) -> torch.Tensor:
    """
    Fit spectral clustering on training embeddings and score eval embeddings by nearest centroid.
    """
    if (
        train_X.numel() == 0
        or train_y.numel() == 0
        or eval_X.numel() == 0
        or eval_y.numel() == 0
    ):
        print(f"WARNING: compute_spectral_probe_accuracy() called with empty tensors")
        return torch.tensor(float('nan'), dtype=torch.float32)
    try:
        
        from sklearn.cluster import SpectralClustering

        eval_labels = eval_y.view(-1)
        eval_labels_np = eval_labels.cpu().numpy()
        if np.unique(eval_labels_np).size < 2:
            print(f"WARNING: compute_spectral_probe_accuracy() called with less than 2 unique eval labels")
            return torch.tensor(float('nan'), dtype=torch.float32)

        train_np = train_X.cpu().numpy()
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            assign_labels=assign_labels,
            n_jobs=n_jobs,
        )
        train_cluster_np = clusterer.fit_predict(train_np)
        device = eval_X.device
        train_X = train_X.to(device)
        eval_X = eval_X.to(device)
        train_cluster = torch.from_numpy(train_cluster_np).to(device)

        unique_clusters = torch.unique(train_cluster)
        if unique_clusters.numel() == 0:
            print(f"WARNING: compute_spectral_probe_accuracy() called with 0 unique clusters")
            return torch.tensor(float('nan'), dtype=torch.float32)

        device = train_X.device
        centroids: List[torch.Tensor] = []
        centroid_ids: List[int] = []
        for cid in unique_clusters.tolist():
            mask = train_cluster == cid
            if mask.any():
                centroids.append(train_X[mask].mean(dim=0, keepdim=True))
                centroid_ids.append(cid)

        if len(centroids) == 0:
            print(f"WARNING: compute_spectral_probe_accuracy() called with 0 centroids")
            return torch.tensor(float('nan'), dtype=torch.float32)

        centroids_tensor = torch.cat(centroids, dim=0).to(device)
        centroid_ids_tensor = torch.tensor(centroid_ids, dtype=torch.long, device=device)

        dists = torch.cdist(eval_X, centroids_tensor)
        assign_idx = torch.argmin(dists, dim=1)
        preds = centroid_ids_tensor[assign_idx]

        acc = _clustering_accuracy(eval_labels, preds)
        return torch.tensor(acc, dtype=torch.float32)
    except Exception as e:
        print(f"WARNING: compute_spectral_probe_accuracy() raised an error: {e}")
        return torch.tensor(float('nan'), dtype=torch.float32)


class MultiTargetMSE(Metric):
    """
    A custom metric class for computing Mean Squared Error (MSE) separately for each target 
    (if mode == 'per_target') in a multi-target regression task, or for a vector target 
    (if mode == 'vector').

    This metric is particularly useful when you need to track the MSE for each output
    independently, rather than computing a single aggregated MSE across all targets.

    Note that the incoming batch_size could be number of nodes or number of graphs,
    depending on the task.

    Attributes:
        num_targets (int): The number of target variables in the regression task.
        mode (str): 'per_target' computes per-target MSE, 'vector' computes vector-norm MSE.
        sum_mse_across_graphs (torch.Tensor): Running sum of per-graph mean squared errors.
            Shape is (num_targets,) in 'per_target' mode, or (1,) in 'vector' mode.
        graph_count (torch.Tensor): Running total of graphs aggregated across updates.

    Example:
        >>> metric = MultiTargetMSE(num_targets=3)
        >>> preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> target = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
        >>> metric.update(preds, target)
        >>> mse_per_target = metric.compute()
        >>> metric.reset()
    """

    def __init__(
        self,
        num_targets: int,
        mode: str = 'per_target',
        dist_sync_on_step: bool = False,  # deprecated (more expensive)
        sync_on_compute: bool = True,
    ) -> None:
        # Enable automatic cross-process reduction when .compute() is called
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        self.num_targets = num_targets
        self.mode = mode
        if mode not in ['per_target', 'vector']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'per_target' or 'vector'.")
        # Running sum over graphs of per-graph mean squared error
        # Shape: (num_targets,) for per_target else (1,) for vector
        out_dim = num_targets if mode == 'per_target' else 1
        self.add_state(
            "sum_mse_across_graphs",
            default=torch.zeros(out_dim),
            dist_reduce_fx="sum",
        )
        # Total number of graphs aggregated
        self.add_state(
            "graph_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_index: torch.Tensor | None = None,
        node_counts: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with a new batch.

        For node-level tasks (when ``batch_index`` is provided and preds are node-aligned),
        computes per-graph MSEs (mean over nodes in each graph) and accumulates their sum
        along with the number of graphs. For single-graph masked updates, pass
        ``node_counts=torch.tensor([num_masked_nodes])`` and leave ``batch_index`` as None.

        For graph-level tasks (no ``batch_index`` and no ``node_counts``), falls back to
        treating each row as a separate "graph" (i.e., standard mean across rows).
        """
        if preds.shape != target.shape:
            raise AssertionError(
                f"Predictions and targets must have the same shape. "
                f"Got {preds.shape} and {target.shape}."
            )

        device = preds.device
        # Ensure internal state tensors live on the same device as inputs
        if self.sum_mse_across_graphs.device != device:
            self.sum_mse_across_graphs = self.sum_mse_across_graphs.to(device)
        if self.graph_count.device != device:
            self.graph_count = self.graph_count.to(device)

        # Determine per-node squared error tensor
        if self.mode == 'per_target':
            # Shape: (N, T)
            per_node_sq_err = (preds - target) ** 2
        else:  # 'vector'
            # Sum across coordinates -> (N,)
            per_node_sq_err = ((preds - target) ** 2).sum(dim=1)

        # Case 1: Multi-graph node-level batch with graph assignments
        if batch_index is not None:
            # Normalize per graph by its node count (mean over nodes), then sum across graphs
            unique_graphs = torch.unique(batch_index)
            num_graphs_batch = unique_graphs.numel()

            if self.mode == 'per_target':
                # Accumulate sum of per-graph mean (shape (T,)) across graphs in this batch
                batch_sum = torch.zeros(self.num_targets, device=device)
                for g in unique_graphs:
                    mask = (batch_index == g)
                    graph_mean = per_node_sq_err[mask].mean(dim=0)  # (T,)
                    batch_sum += graph_mean
            else:  # 'vector'
                batch_sum = torch.tensor(0.0, device=device)
                for g in unique_graphs:
                    mask = (batch_index == g)
                    graph_mean = per_node_sq_err[mask].mean()  # scalar
                    batch_sum = batch_sum + graph_mean

            # Update running totals
            if self.mode == 'per_target':
                self.sum_mse_across_graphs += batch_sum
            else:
                self.sum_mse_across_graphs += batch_sum.unsqueeze(0)
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)
            return

        # Case 2: Single-graph masked node-level batch (no batch_index but node_counts provided)
        if node_counts is not None and node_counts.numel() == 1:
            if self.mode == 'per_target':
                graph_mean = per_node_sq_err.mean(dim=0)  # (T,)
                self.sum_mse_across_graphs += graph_mean
            else:
                graph_mean = per_node_sq_err.mean()  # scalar
                self.sum_mse_across_graphs += graph_mean.unsqueeze(0)
            self.graph_count += torch.tensor(1, device=device, dtype=torch.long)
            return

        # Case 3: Graph-level batch (or fallback) — treat each row as a graph
        if self.mode == 'per_target':
            row_means = per_node_sq_err  # each row is one graph already
            # Sum across rows (graphs) -> (T,)
            batch_mean = row_means.mean(dim=0)  # mean across rows
            # To be consistent with per-graph averaging across batches, count number of rows as graphs
            num_graphs_batch = preds.shape[0]
            # Convert mean back to sum over graphs for accumulator:
            batch_sum = batch_mean * num_graphs_batch
            self.sum_mse_across_graphs += batch_sum
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)
        else:
            # vector mode: per_node_sq_err is (N,), each row treated as its own graph
            batch_mean = per_node_sq_err.mean()
            num_graphs_batch = preds.shape[0]
            batch_sum = batch_mean * num_graphs_batch
            self.sum_mse_across_graphs += batch_sum.unsqueeze(0)
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)

    def compute(self) -> torch.Tensor:
        """
        Compute the mean squared error across all graphs aggregated so far.
        """
        # Avoid division by zero
        denom = torch.clamp(self.graph_count.to(torch.float32), min=1.0)
        return self.sum_mse_across_graphs / denom

    def reset(self) -> None:
        """
        Reset internal accumulators.
        """
        self.sum_mse_across_graphs.zero_()
        self.graph_count.zero_()



class SilhouetteScore(Metric):
    """
    Clustering quality metric based on the Silhouette score.

    This metric wraps :func:`sklearn.metrics.silhouette_score` so it can be used
    with ``BaseModule`` in clustering tasks (e.g., supervised contrastive learning
    where the model outputs embeddings and "target" provides cluster labels).

    The Silhouette score ranges from -1 to 1, where higher is better. It measures
    how similar a sample is to its own cluster compared to other clusters.

    The ``metric`` argument controls the distance metric used internally by
    scikit-learn. It is passed directly to ``sklearn.metrics.silhouette_score``
    and defaults to ``'cosine'``. Other valid options include
    ``'l1'``, ``'l2'``, ``'euclidean'``, ``'haversine'``, and
    ``'nan_euclidean'`` (see the scikit-learn documentation for details).

    Notes:
        - This implementation collects all embeddings and labels in memory on
          the host (CPU) and computes a single Silhouette score at ``compute()``.
        - It is intended primarily for single-process evaluation; multi-process
          / DDP aggregation is not currently handled inside this metric.
    """

    def __init__(
        self,
        metric: str = 'cosine',
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        from sklearn.metrics import silhouette_score
        self.silhouette_score = silhouette_score
        self.metric = metric
        # Store embeddings/labels as CPU tensors; we intentionally do not register
        # them as torchmetrics states so we can keep them as Python lists.
        self._embeddings: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Accumulate a batch of embeddings and labels.

        Args:
            embeddings: Tensor of shape (N, D) containing embedding vectors.
            labels: Tensor of shape (N,) or (N, 1) containing integer cluster labels.
        """
        if embeddings is None or labels is None:
            return

        if embeddings.dim() < 2:
            raise ValueError(
                f"SilhouetteScore.update expects embeddings with shape (N, D); got {tuple(embeddings.shape)}"
            )

        # Flatten any trailing dimensions beyond the feature dimension
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        # Ensure 1D labels
        labels = labels.view(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings and labels must have the same number of samples. "
                f"Got {embeddings.shape[0]} and {labels.shape[0]}."
            )

        # Move to CPU and detach to avoid holding computation graph
        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Compute the Silhouette score over all accumulated embeddings and labels.

        Returns:
            A scalar tensor containing the Silhouette score.

        Raises:
            RuntimeError: If scikit-learn is not available.
        """
        if self.silhouette_score is None:
            raise RuntimeError(
                "SilhouetteScore metric requires scikit-learn. "
                "Please install it via 'pip install scikit-learn'."
            )

        if len(self._embeddings) == 0:
            # No data accumulated; return NaN tensor
            return torch.tensor(float('nan'), dtype=torch.float32)

        embeddings = torch.cat(self._embeddings, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0).numpy()

        # scikit-learn expects numeric labels; they can be ints or floats
        score = self.silhouette_score(embeddings, labels, metric=self.metric)
        return torch.tensor(float(score), dtype=torch.float32)

    def reset(self) -> None:
        """
        Clear accumulated embeddings and labels.
        """
        self._embeddings.clear()
        self._labels.clear()


class LogisticLinearAccuracy(Metric):
    """
    Clustering metric that measures linear separability of embeddings via a
    multinomial logistic regression classifier.

    This metric collects embeddings and labels, fits a linear logistic
    regression classifier (using scikit-learn) on all accumulated samples,
    and reports the classification accuracy on those same samples. It is
    suitable for multi-class problems (e.g., a 7-class trajectory-labeling
    task) and can be used as a clustering-quality proxy when the model is
    trained with (supervised) contrastive learning.

    Notes:
        - This implementation trains the classifier on the full dataset and
          reports the in-sample accuracy (no cross-validation or held-out split).
        - For large datasets, this can be more expensive than Dunn Index or
          Silhouette score, since it solves a full optimization problem.
        - It is intended primarily for single-process evaluation; multi-process
          / DDP aggregation is not handled inside this metric.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        # Import scikit-learn lazily so the dependency is only required when
        # this metric is actually used.
        from sklearn.linear_model import LogisticRegression

        self.LogisticRegression = LogisticRegression
        self.max_iter = max_iter
        self.C = C

        # Embeddings and labels are accumulated on CPU as plain tensors.
        self._embeddings: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Accumulate a batch of embeddings and labels.

        Args:
            embeddings: Tensor of shape (N, D) containing embedding vectors.
            labels: Tensor of shape (N,) or (N, 1) containing integer class labels.
        """
        if embeddings is None or labels is None:
            return

        if embeddings.dim() < 2:
            raise ValueError(
                f"LogisticLinearAccuracy.update expects embeddings with shape (N, D); "
                f"got {tuple(embeddings.shape)}"
            )

        # Flatten any trailing dimensions beyond the feature dimension
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        # Ensure 1D labels
        labels = labels.view(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings and labels must have the same number of samples. "
                f"Got {embeddings.shape[0]} and {labels.shape[0]}."
            )

        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Fit a logistic regression classifier on the accumulated embeddings and
        return the classification accuracy.

        Returns:
            A scalar tensor containing the classification accuracy in [0, 1].
        """
        if len(self._embeddings) == 0:
            # No data accumulated; return NaN tensor
            return torch.tensor(float('nan'), dtype=torch.float32)

        # Concatenate all samples
        embeddings = torch.cat(self._embeddings, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0).numpy()

        # Need at least two classes to train a classifier
        

        if np.unique(labels).size < 2:
            return torch.tensor(float('nan'), dtype=torch.float32)

        clf = self.LogisticRegression(
            # multi_class='multinomial',  # leave default to avoid future deprecation warning
            solver='lbfgs',
            max_iter=self.max_iter,
            C=self.C,
        )
        clf.fit(embeddings, labels)
        acc = float(clf.score(embeddings, labels))
        return torch.tensor(acc, dtype=torch.float32)

    def reset(self) -> None:
        """
        Clear accumulated embeddings and labels.
        """
        self._embeddings.clear()
        self._labels.clear()


class SVMAccuracy(Metric):
    """
    Clustering metric that measures linear separability using an SVM classifier.

    The classifier defaults to the same hyperparameters as MARBLE's SVM probe
    (C = 1.0, radial basis function kernel). Embeddings are accumulated on CPU,
    then a single SVM is trained and evaluated on those samples when ``compute`` 
    is called.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        from sklearn.svm import SVC

        self.SVC = SVC
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self._embeddings: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Accumulate embeddings and labels for later SVM fitting.
        """
        if embeddings is None or labels is None:
            return

        if embeddings.dim() < 2:
            raise ValueError(
                f"SVMAccuracy.update expects embeddings with shape (N, D); "
                f"got {tuple(embeddings.shape)}"
            )

        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        labels = labels.view(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings and labels must have the same number of samples. "
                f"Got {embeddings.shape[0]} and {labels.shape[0]}."
            )

        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Fit an SVM classifier on the accumulated embeddings and report accuracy.
        """
        if len(self._embeddings) == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        embeddings = torch.cat(self._embeddings, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0).numpy()

        

        if np.unique(labels).size < 2:
            return torch.tensor(float('nan'), dtype=torch.float32)

        clf = self.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        clf.fit(embeddings, labels)
        acc = float(clf.score(embeddings, labels))
        return torch.tensor(acc, dtype=torch.float32)

    def reset(self) -> None:
        """
        Clear accumulated embeddings and labels.
        """
        self._embeddings.clear()
        self._labels.clear()


class SpectralClusteringAccuracy(Metric):
    """
    Clustering metric using scikit-learn's SpectralClustering with nearest-neighbor affinity.
    """

    def __init__(
        self,
        n_clusters: int,
        affinity: str = 'nearest_neighbors',  # nearest_neighbors | rbf | precomputed
        n_neighbors: int = 10,
        assign_labels: str = 'cluster_qr',
        n_jobs: int = 1,
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        from sklearn.cluster import SpectralClustering

        self.SpectralClustering = SpectralClustering
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.n_jobs = n_jobs
        self.affinity = affinity
        self._embeddings: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if embeddings is None or labels is None:
            return
        if embeddings.dim() < 2:
            raise ValueError(
                f"SpectralClusteringAccuracy expects embeddings of shape (N, D); "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)
        labels = labels.view(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must have matching lengths.")
        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> torch.Tensor:
        if len(self._embeddings) == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        embeddings = torch.cat(self._embeddings, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0)

        if embeddings.shape[0] < self.n_clusters:
            return torch.tensor(float('nan'), dtype=torch.float32)

        try:
            clusterer = self.SpectralClustering(
                n_clusters=self.n_clusters,
                affinity=self.affinity,
                n_neighbors=self.n_neighbors,
                assign_labels=self.assign_labels,
                n_jobs=self.n_jobs,
            )
            pred = clusterer.fit_predict(embeddings)
            acc = _clustering_accuracy(labels, pred)
        except Exception:
            acc = float('nan')

        return torch.tensor(acc, dtype=torch.float32)

    def reset(self) -> None:
        self._embeddings.clear()
        self._labels.clear()


class KMeansAccuracy(Metric):
    """
    Clustering metric that runs k-means with a fixed number of clusters and scores accuracy
    via optimal label matching.
    """

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )
        from sklearn.cluster import KMeans

        self.KMeans = KMeans
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._embeddings: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if embeddings is None or labels is None:
            return
        if embeddings.dim() < 2:
            raise ValueError(
                f"KMeansAccuracy expects embeddings of shape (N, D); "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)
        labels = labels.view(-1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must have matching lengths.")
        self._embeddings.append(embeddings.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> torch.Tensor:
        if len(self._embeddings) == 0:
            return torch.tensor(float('nan'), dtype=torch.float32)

        embeddings = torch.cat(self._embeddings, dim=0).numpy()
        labels = torch.cat(self._labels, dim=0)

        if embeddings.shape[0] < self.n_clusters:
            return torch.tensor(float('nan'), dtype=torch.float32)

        try:
            kmeans = self.KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            pred = kmeans.fit_predict(embeddings)
            acc = _clustering_accuracy(labels, pred)
        except Exception:
            acc = float('nan')

        return torch.tensor(acc, dtype=torch.float32)

    def reset(self) -> None:
        self._embeddings.clear()
        self._labels.clear()
