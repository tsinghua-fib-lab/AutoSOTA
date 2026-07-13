"""
Original UniRouter Implementation

Implements the cluster-based routing from the UniRouter paper:
- Each LLM represented as prediction error vector on validation set
- Cluster prompts into K semantic groups
- Route queries based on cluster similarity

Author: Implementation based on UniRouter paper specification
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class OriginalUniRouter:
    """
    Original UniRouter: Routes queries to best LLM (always with unlimited tokens).

    Key differences from Uni-R2:
    - No token budget optimization (always uses unlimited tokens)
    - Simpler features: Ψ(h) ∈ R^K (per-cluster error rates)
    - Routing: Pick best LLM only
    """

    def __init__(self,
                 val_size: int = 500,
                 n_clusters: int = 100,
                 use_clustering: bool = True):
        """
        Args:
            val_size: Number of validation samples
            n_clusters: Number of clusters
            use_clustering: If True, cluster validation set
        """
        self.val_size = val_size
        self.n_clusters = n_clusters
        self.use_clustering = use_clustering

        self.val_indices = None
        self.val_embeddings = None
        self.cluster_assignments = None
        self.cluster_embeddings = None

        # Model features: model_name → Ψ(h) ∈ R^K
        self.model_features = {}

        # Model metadata
        self.model_sizes = {}  # model_name → size in billions
        self.model_costs = {}  # model_name → per-token cost

    def create_validation_set(self, embeddings: np.ndarray, train_idx: np.ndarray,
                              test_idx: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Create validation set from training data.

        Args:
            embeddings: All query embeddings [num_queries, embedding_dim]
            train_idx: Training indices
            test_idx: Test indices
            seed: Random seed

        Returns:
            val_idx: Validation indices (subset of train_idx)
        """
        np.random.seed(seed)

        # Sample from training set
        val_idx = np.random.choice(train_idx, size=min(self.val_size, len(train_idx)),
                                   replace=False)
        self.val_indices = val_idx
        self.val_embeddings = embeddings[val_idx]

        # Optionally cluster
        if self.use_clustering:
            from sklearn.cluster import KMeans
            print(f"\nClustering {len(val_idx)} validation samples into {self.n_clusters} clusters...")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed)
            self.cluster_assignments = kmeans.fit_predict(self.val_embeddings)
            self.cluster_embeddings = kmeans.cluster_centers_
            print(f"✓ Clustering complete")

        return val_idx

    def register_model(self,
                      model_name: str,
                      scores_df: pd.DataFrame,
                      val_idx: np.ndarray,
                      size: float,
                      per_token_cost: float = None):
        """
        Register a new model using UNLIMITED tokens only.

        Args:
            model_name: Name of the model
            scores_df: DataFrame with performance scores
            val_idx: Validation indices
            size: Model size in billions of parameters
            per_token_cost: Cost per token (default: size * 1e-4)
        """
        # Extract unlimited scores for validation set
        val_df = scores_df.iloc[val_idx]

        if 'unlimited_score' not in val_df.columns:
            raise ValueError(f"Column 'unlimited_score' not found in dataframe")

        unlimited_scores = val_df['unlimited_score'].fillna(0).values

        if not self.use_clustering:
            # Direct features: [val_size] - one error rate per validation sample
            # Error = 1 - score
            features = 1.0 - unlimited_scores
        else:
            # Cluster-based features: [n_clusters] - average error per cluster
            features = np.zeros(self.n_clusters)

            for k in range(self.n_clusters):
                cluster_mask = self.cluster_assignments == k
                if cluster_mask.sum() == 0:
                    continue

                cluster_scores = unlimited_scores[cluster_mask]
                features[k] = 1.0 - cluster_scores.mean()  # Error rate

        self.model_features[model_name] = features
        self.model_sizes[model_name] = size

        if per_token_cost is None:
            per_token_cost = size * 1e-4

        self.model_costs[model_name] = per_token_cost

        print(f"✓ Registered {model_name} (size={size}B, features shape={features.shape})")

    def compute_query_features(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute Φ(x) - query feature vector in K-dimensional space.

        Args:
            query_embedding: Query embedding [embedding_dim]

        Returns:
            phi_x: Query features [K] (normalized to sum to 1)
        """
        if not self.use_clustering:
            # Direct similarity to validation samples
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            val_norms = self.val_embeddings / (np.linalg.norm(self.val_embeddings, axis=1, keepdims=True) + 1e-8)
            phi_x = val_norms @ query_norm  # [val_size]
        else:
            # Similarity to cluster centroids
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            cluster_norms = self.cluster_embeddings / (np.linalg.norm(self.cluster_embeddings, axis=1, keepdims=True) + 1e-8)
            phi_x = cluster_norms @ query_norm  # [n_clusters]

        # Normalize to sum to 1 (convert to probability distribution)
        phi_x = np.exp(phi_x * 10)  # Scale before softmax
        phi_x = phi_x / phi_x.sum()

        return phi_x

    def route(self,
              query_embeddings: np.ndarray,
              lamb: float = 0.0) -> Dict:
        """
        Route queries to best LLM (always with unlimited tokens).

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            lamb: Cost-performance tradeoff parameter

        Returns:
            results: Dictionary with routing results
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]

        num_queries = query_embeddings.shape[0]
        models = list(self.model_features.keys())

        selected_models = []
        predicted_errors = []
        predicted_costs = []

        # Route each query
        for i in range(num_queries):
            query_emb = query_embeddings[i]

            # Compute Φ(x)
            phi_x = self.compute_query_features(query_emb)

            # Compute routing scores for all models
            # Maximize: (1-λ)*quality - λ*cost
            best_score = -np.inf
            best_model = None

            for model_name in models:
                psi_h = self.model_features[model_name]  # [K]
                per_token_cost = self.model_costs[model_name]

                # Predicted error: Φ(x)^T · Ψ(h)
                # This is a weighted average of cluster error rates
                pred_error = phi_x @ psi_h

                # Estimate unlimited tokens based on R2-Bench data average
                assumed_tokens = 3650  # Average unlimited_count from dataset
                cost = per_token_cost * assumed_tokens

                # Convert error to quality (quality = 1 - error)
                pred_quality = 1.0 - pred_error

                # Routing score: (1-λ)*quality - λ*cost (MAXIMIZE)
                # λ=0: pure quality, λ=1: pure cost minimization
                score = (1 - lamb) * pred_quality - lamb * cost

                if score > best_score:
                    best_score = score
                    best_model = model_name

            # Record selection
            selected_models.append(best_model)

            # Predicted error and cost
            psi_best = self.model_features[best_model]
            pred_error = phi_x @ psi_best
            pred_cost = self.model_costs[best_model] * 3650  # Estimated unlimited tokens

            predicted_errors.append(pred_error)
            predicted_costs.append(pred_cost)

        results = {
            'models': selected_models,
            'budgets': ['unlimited'] * num_queries,  # Always unlimited
            'predicted_errors': np.array(predicted_errors),
            'predicted_costs': np.array(predicted_costs)
        }

        return results

    def evaluate_routing(self,
                        query_embeddings: np.ndarray,
                        true_scores_dict: Dict[str, np.ndarray],
                        true_counts_dict: Dict[str, np.ndarray],
                        lamb_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate routing performance across lambda range.

        Args:
            query_embeddings: Test query embeddings [num_queries, embedding_dim]
            true_scores_dict: Ground truth scores {model_name: {budget: scores}}
            true_counts_dict: Ground truth token counts {model_name: {budget: counts}}
            lamb_range: Array of lambda values

        Returns:
            costs: Average costs [len(lamb_range)]
            performances: Average performances [len(lamb_range)]
        """
        num_queries = query_embeddings.shape[0]

        costs_out = []
        perfs_out = []

        for lamb in lamb_range:
            # Route all queries
            routing = self.route(query_embeddings, lamb=lamb)

            # Evaluate using ground truth (always unlimited)
            query_perfs = np.zeros(num_queries)
            query_costs = np.zeros(num_queries)

            for i in range(num_queries):
                model = routing['models'][i]

                # Always use unlimited budget
                score_key = 'unlimited_score'
                count_key = 'unlimited_count'

                if model in true_scores_dict and score_key in true_scores_dict[model]:
                    query_perfs[i] = true_scores_dict[model][score_key].iloc[i]
                    actual_tokens = true_counts_dict[model][count_key].iloc[i]
                    model_size = self.model_sizes[model]
                    query_costs[i] = actual_tokens * model_size

            costs_out.append(query_costs.mean())
            perfs_out.append(query_perfs.mean())

        return np.array(costs_out), np.array(perfs_out)

    def save(self, filepath: str):
        """Save to disk."""
        data = {
            'val_size': self.val_size,
            'n_clusters': self.n_clusters,
            'use_clustering': self.use_clustering,
            'val_indices': self.val_indices,
            'val_embeddings': self.val_embeddings,
            'cluster_assignments': self.cluster_assignments,
            'cluster_embeddings': self.cluster_embeddings,
            'model_features': self.model_features,
            'model_sizes': self.model_sizes,
            'model_costs': self.model_costs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        router = cls(
            val_size=data['val_size'],
            n_clusters=data['n_clusters'],
            use_clustering=data['use_clustering']
        )
        router.val_indices = data['val_indices']
        router.val_embeddings = data['val_embeddings']
        router.cluster_assignments = data['cluster_assignments']
        router.cluster_embeddings = data['cluster_embeddings']
        router.model_features = data['model_features']
        router.model_sizes = data['model_sizes']
        router.model_costs = data['model_costs']

        return router
