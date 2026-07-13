"""
Uni-R2: Universal Cost-aware Routing for Efficient LLM Inference

Combines:
- R2-Router: Multiple token budgets per model
- UniRouter: Dynamic model pool (add new LLMs without retraining)

Key idea: Represent each LLM as a feature matrix Ψ(h) ∈ R^(K×B)
where K = number of validation samples (or clusters), B = number of token budgets

Author: Implementation based on Uni-R2 specification
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib


class ValidationSetBuilder:
    """
    Builds validation set and computes model features for each (model, budget) pair.
    """

    def __init__(self,
                 val_size: int = 500,
                 token_budgets: List[int] = [50, 100, 200, 400, 800, 1600, 3200],
                 use_clustering: bool = False,
                 n_clusters: int = 50):
        """
        Args:
            val_size: Number of validation samples
            token_budgets: List of token budget limits
            use_clustering: If True, cluster validation set and use cluster averages
            n_clusters: Number of clusters (if use_clustering=True)
        """
        self.val_size = val_size
        self.token_budgets = token_budgets
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters

        self.val_indices = None
        self.val_embeddings = None
        self.cluster_assignments = None  # [val_size] → cluster_id
        self.cluster_embeddings = None   # [n_clusters, embedding_dim] centroids

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

    def compute_model_features(self,
                               model_name: str,
                               scores_df: pd.DataFrame,
                               val_idx: np.ndarray) -> np.ndarray:
        """
        Compute feature matrix Ψ(h) for a model.

        Args:
            model_name: Name of the model
            scores_df: DataFrame with performance scores
            val_idx: Validation indices

        Returns:
            features: Feature matrix [K, B] where K=val_size or n_clusters, B=num_budgets
        """
        # Map token budgets to column names
        budget_cols = [f"{b}_score" if b < 9999 else "unlimited_score"
                       for b in self.token_budgets]

        # Validate that all budget columns exist
        missing_cols = [col for col in budget_cols if col not in scores_df.columns]
        if missing_cols:
            available_budgets = [col.replace('_score', '') for col in scores_df.columns if col.endswith('_score')]
            raise ValueError(
                f"Missing columns for {model_name}: {missing_cols}\n"
                f"Available budgets in CSV: {available_budgets}\n"
                f"Requested budgets: {self.token_budgets}\n"
                f"Make sure TOKEN_BUDGETS matches your CSV file!"
            )

        # Extract scores for validation set
        val_df = scores_df.iloc[val_idx]

        if not self.use_clustering:
            # Direct features: [val_size, num_budgets]
            features = np.zeros((len(val_idx), len(self.token_budgets)))

            for b_idx, col in enumerate(budget_cols):
                if col in val_df.columns:
                    scores = val_df[col].fillna(0).values
                    features[:, b_idx] = scores
                else:
                    print(f"  ⚠️  Column {col} not found, using 0")

        else:
            # Cluster-based features: [n_clusters, num_budgets]
            features = np.zeros((self.n_clusters, len(self.token_budgets)))

            for k in range(self.n_clusters):
                cluster_mask = self.cluster_assignments == k
                if cluster_mask.sum() == 0:
                    continue

                cluster_df = val_df[cluster_mask]

                for b_idx, col in enumerate(budget_cols):
                    if col in cluster_df.columns:
                        scores = cluster_df[col].fillna(0).values
                        features[k, b_idx] = scores.mean()  # Average over cluster

        return features


class ModelFeatureMatrix:
    """
    Stores feature matrices for all registered models.

    For each model h, stores Ψ(h) ∈ R^(K×B)
    """

    def __init__(self, token_budgets: List[int], use_clustering: bool = False,
                 n_clusters: int = 50, val_size: int = 500):
        """
        Args:
            token_budgets: List of token budget limits
            use_clustering: Whether features are cluster-based
            n_clusters: Number of clusters (if use_clustering)
            val_size: Validation set size (if not use_clustering)
        """
        self.token_budgets = token_budgets
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.val_size = val_size

        self.K = n_clusters if use_clustering else val_size
        self.B = len(token_budgets)

        # Model features: model_name → Ψ(h) ∈ R^(K×B)
        self.features = {}

        # Model metadata
        self.model_sizes = {}  # model_name → size in billions
        self.model_costs = {}  # model_name → per-token cost

    def register_model(self, model_name: str, features: np.ndarray,
                       size: float, per_token_cost: float = None):
        """
        Register a new model.

        Args:
            model_name: Name of the model
            features: Feature matrix Ψ(h) ∈ R^(K×B)
            size: Model size in billions of parameters
            per_token_cost: Cost per token (default: size, following R2-Router cost model)
        """
        if features.shape != (self.K, self.B):
            raise ValueError(f"Features shape {features.shape} != ({self.K}, {self.B})")

        self.features[model_name] = features
        self.model_sizes[model_name] = size

        # Cost model: cost = actual_tokens × model_size (following R2-Router)
        if per_token_cost is None:
            per_token_cost = size  # Use model size directly

        self.model_costs[model_name] = per_token_cost

        print(f"✓ Registered {model_name} (size={size}B, cost_per_token={per_token_cost:.4f})")

    def get_models(self) -> List[str]:
        """Get list of registered models."""
        return list(self.features.keys())

    def get_features(self, model_name: str) -> np.ndarray:
        """Get feature matrix for a model."""
        return self.features.get(model_name)

    def save(self, filepath: str):
        """Save to disk."""
        data = {
            'token_budgets': self.token_budgets,
            'use_clustering': self.use_clustering,
            'n_clusters': self.n_clusters,
            'val_size': self.val_size,
            'features': self.features,
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

        matrix = cls(
            token_budgets=data['token_budgets'],
            use_clustering=data['use_clustering'],
            n_clusters=data['n_clusters'],
            val_size=data['val_size']
        )
        matrix.features = data['features']
        matrix.model_sizes = data['model_sizes']
        matrix.model_costs = data['model_costs']

        return matrix


class UniR2Router:
    """
    Uni-R2 router: Routes queries to (model, budget) pairs.

    Routing score: Φ(x)^T · Ψ(h, budget_b) - λ · cost(h, budget_b)
    """

    def __init__(self,
                 feature_matrix: ModelFeatureMatrix,
                 val_embeddings: np.ndarray = None,
                 cluster_embeddings: np.ndarray = None):
        """
        Args:
            feature_matrix: Model feature matrix
            val_embeddings: Validation embeddings [val_size, embedding_dim] (if not clustering)
            cluster_embeddings: Cluster centroids [n_clusters, embedding_dim] (if clustering)
        """
        self.feature_matrix = feature_matrix
        self.val_embeddings = val_embeddings
        self.cluster_embeddings = cluster_embeddings

        if feature_matrix.use_clustering:
            if cluster_embeddings is None:
                raise ValueError("cluster_embeddings required when use_clustering=True")
            self.K_embeddings = cluster_embeddings
        else:
            if val_embeddings is None:
                raise ValueError("val_embeddings required when use_clustering=False")
            self.K_embeddings = val_embeddings

    def compute_query_features(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute Φ(x) - query feature vector in K-dimensional space.

        Args:
            query_embedding: Query embedding [embedding_dim]

        Returns:
            phi_x: Query features [K] (normalized to sum to 1)
        """
        # Compute similarity to each validation sample/cluster
        # Use cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        K_norms = self.K_embeddings / (np.linalg.norm(self.K_embeddings, axis=1, keepdims=True) + 1e-8)

        # Φ(x) = cosine similarity to each validation sample/cluster
        phi_x = K_norms @ query_norm  # [K]

        # Normalize to sum to 1 (convert to probability distribution)
        # Use softmax to convert similarities to probabilities
        phi_x = np.exp(phi_x * 10)  # Scale before softmax for better discrimination
        phi_x = phi_x / phi_x.sum()

        return phi_x

    def route(self,
              query_embeddings: np.ndarray,
              lamb: float = 0.0,
              return_details: bool = False) -> Dict:
        """
        Route queries to (model, budget) pairs.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            lamb: Cost-performance tradeoff parameter (0 = quality only, high = cost-aware)
            return_details: If True, return detailed routing information

        Returns:
            results: Dictionary with routing results
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]

        num_queries = query_embeddings.shape[0]
        models = self.feature_matrix.get_models()

        # Storage for results
        selected_models = []
        selected_budgets = []
        predicted_scores = []
        predicted_costs = []

        # Route each query
        for i in range(num_queries):
            query_emb = query_embeddings[i]

            # Compute Φ(x)
            phi_x = self.compute_query_features(query_emb)  # [K]

            # Compute routing scores for all (model, budget) pairs
            best_score = -np.inf
            best_model = None
            best_budget_idx = None

            for model_name in models:
                psi_h = self.feature_matrix.get_features(model_name)  # [K, B]
                per_token_cost = self.feature_matrix.model_costs[model_name]

                for b_idx, budget in enumerate(self.feature_matrix.token_budgets):
                    # Predicted quality: Φ(x)^T · Ψ(h, budget_b)
                    pred_quality = phi_x @ psi_h[:, b_idx]

                    # Estimate actual token count (not budget limit)
                    # For unlimited (9999), use reasonable estimate based on data
                    if budget >= 9999:
                        estimated_tokens = 3650  # Average unlimited tokens from data
                    else:
                        # Budget limits are usually close to actual usage
                        estimated_tokens = budget

                    # Cost: estimated_tokens × model_size (R2-Router cost model)
                    # per_token_cost is actually model_size in our implementation
                    cost = per_token_cost * estimated_tokens

                    # Routing score: (1-λ)*quality - λ*cost
                    # λ=0: pure quality, λ=high: cost-aware
                    score = (1 - lamb) * pred_quality - lamb * cost

                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_budget_idx = b_idx

            # Record selection
            selected_models.append(best_model)
            selected_budgets.append(self.feature_matrix.token_budgets[best_budget_idx])

            # Predicted performance and cost
            psi_best = self.feature_matrix.get_features(best_model)
            pred_score = phi_x @ psi_best[:, best_budget_idx]

            # Use same token estimation as routing
            best_budget = self.feature_matrix.token_budgets[best_budget_idx]
            if best_budget >= 9999:
                estimated_tokens = 3650
            else:
                estimated_tokens = best_budget

            pred_cost = self.feature_matrix.model_costs[best_model] * estimated_tokens

            predicted_scores.append(pred_score)
            predicted_costs.append(pred_cost)

        results = {
            'models': selected_models,
            'budgets': selected_budgets,
            'predicted_scores': np.array(predicted_scores),
            'predicted_costs': np.array(predicted_costs)
        }

        if return_details:
            results['query_features'] = [self.compute_query_features(q)
                                         for q in query_embeddings]

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

            # Evaluate using ground truth
            query_perfs = np.zeros(num_queries)
            query_costs = np.zeros(num_queries)

            for i in range(num_queries):
                model = routing['models'][i]
                budget = routing['budgets'][i]

                # Look up true performance and cost
                budget_key = f"{budget}_score" if budget < 9999 else "unlimited_score"
                count_key = f"{budget}_count" if budget < 9999 else "unlimited_count"

                if model in true_scores_dict and budget_key in true_scores_dict[model]:
                    query_perfs[i] = true_scores_dict[model][budget_key].iloc[i]
                    actual_tokens = true_counts_dict[model][count_key].iloc[i]
                    model_size = self.feature_matrix.model_sizes[model]
                    query_costs[i] = actual_tokens * model_size

            costs_out.append(query_costs.mean())
            perfs_out.append(query_perfs.mean())

        return np.array(costs_out), np.array(perfs_out)
