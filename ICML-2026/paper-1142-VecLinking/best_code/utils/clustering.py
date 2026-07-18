from sklearn.cluster import KMeans
import numpy as np
import torch
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from scipy.sparse import csr_matrix, eye
from sklearn.metrics import v_measure_score
import random
from loguru import logger


class Clusterer:
    def __init__(self, method='kmeans', n_clusters=8, verbose=False, use_gpu=True, batch_size=100, n_neighbors=10, affinity='nearest_neighbors', labels=None):
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.model = None
        self.fitted = False
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.labels = labels

    def fit(self, data):
        if self.method == 'kmeans':
            labels = self._fit_kmeans(data)
        elif self.method == 'spectral':
            labels = self._fit_spectral(data)
        elif self.method == 'kmeans_sampled':
            labels = self._fit_kmeans_sampled(data, self.labels)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        self.fitted = True
        return labels

    def predict(self, data):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        if self.method == 'kmeans':
            return self._predict_kmeans(data)
        elif self.method == 'spectral':
            return self._predict_spectral(data)
        elif self.method == 'kmeans_sampled':
            return self._predict_kmeans_sampled(data)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

    def _predict_kmeans(self, data):
        if isinstance(self.model, dict) and self.model.get('type') == 'pytorch':
            # GPU prediction using PyTorch
            device = self.model['device']
            centroids = self.model['centroids']
            
            if isinstance(data, torch.Tensor):
                data_tensor = data.to(device).float()
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
            
            # Calculate distances to centroids
            distances = torch.cdist(data_tensor, centroids)
            labels = torch.argmin(distances, dim=1)
            return labels.cpu().numpy()
        else:
            # CPU prediction using scikit-learn
            data = self._maybe_torch_to_numpy(data)
            labels = self.model['sklearn_model'].predict(data)
            return labels

    def _predict_spectral(self, data):
        if not hasattr(self, '_spectral_method') or self._spectral_method is None:
            raise RuntimeError("Spectral model not fitted. Call fit() first.")
        
        # Removed gpu_cuml method - now handled by gpu_sklearn fallback
            
        if self._spectral_method == 'gpu_sklearn':
            # GPU path with sklearn k-means fallback
            
            # Convert to tensor and normalize
            if isinstance(data, torch.Tensor):
                data_tensor = data.to(self._device)
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32).to(self._device)
            
            data_normalized = torch.nn.functional.normalize(data_tensor, p=2, dim=1)
            
            # Project new data using saved eigenvectors
            spectral_projection = data_normalized.cpu().numpy() @ self._spectral_eigenvectors
            spectral_projection_normalized = spectral_projection / np.linalg.norm(spectral_projection, axis=1, keepdims=True)
            
            # Use sklearn k-means model to predict
            return self.model.predict(spectral_projection_normalized)
            
        elif self._spectral_method == 'cpu_sparse':
            # CPU path with sparse matrices
            # SpectralClustering doesn't have predict method, so we need to approximate
            # For sparse matrices, this is complex - we'll need to refit or use a different approach
            raise NotImplementedError("Prediction for sparse matrix spectral clustering is not implemented. "
                                     "Consider refitting the model with new data included.")
            
        elif self._spectral_method == 'cpu_dense':
            # CPU path with dense arrays - no scaling
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            from sklearn.cluster import KMeans
            
            temp_kmeans = KMeans(n_clusters=int(self.n_clusters))
            temp_kmeans.fit(data)  

            return temp_kmeans.predict(data)
        
        else:
            raise ValueError(f"Unknown spectral method: {self._spectral_method}")

    def _maybe_torch_to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _pytorch_kmeans(self, data, n_clusters, max_iters=100, tol=1e-4):
        """
        GPU-accelerated K-means implementation using PyTorch
        
        Args:
            data: torch.Tensor on GPU, shape (n_samples, n_features)
            n_clusters: number of clusters
            max_iters: maximum iterations
            tol: tolerance for convergence
        
        Returns:
            centroids: torch.Tensor, shape (n_clusters, n_features)
            labels: torch.Tensor, shape (n_samples,)
        """
        n_samples, n_features = data.shape
        device = data.device
        
        # Initialize centroids randomly
        
        # K-means++ initialization
        centroids = self._kmeans_plus_plus_init(data, n_clusters)
        
        prev_centroids = centroids.clone()
        
        for iteration in range(max_iters):
            # Calculate distances from each point to each centroid
            distances = torch.cdist(data, centroids)  # (n_samples, n_clusters)
            
            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)  # (n_samples,)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]
            
            centroids = new_centroids
            
            # Check for convergence
            centroid_shift = torch.norm(centroids - prev_centroids)
            if centroid_shift < tol:
                break
                
            prev_centroids = centroids.clone()
        
        return centroids, labels
    
    def _kmeans_plus_plus_init(self, data, n_clusters):
        """
        K-means++ initialization for better cluster initialization
        """
        n_samples, n_features = data.shape
        device = data.device
        
        centroids = torch.zeros(n_clusters, n_features, device=device)
        
        # Choose first centroid randomly
        first_idx = torch.randint(0, n_samples, (1,), device=device)
        centroids[0] = data[first_idx]
        
        for i in range(1, n_clusters):
            # Calculate distances to nearest existing centroid
            distances = torch.cdist(data, centroids[:i])  # (n_samples, i)
            min_distances = torch.min(distances, dim=1)[0]  # (n_samples,)
            
            # Choose next centroid with probability proportional to squared distance
            probs = min_distances ** 2
            probs = probs / probs.sum()
            
            # Sample according to probabilities
            next_idx = torch.multinomial(probs, 1)
            centroids[i] = data[next_idx]
        
        return centroids

    def _fit_kmeans(self, data):
        if self.use_gpu and torch.cuda.is_available():
            try:
                # GPU-accelerated K-means using PyTorch
                device = torch.device('cuda')
                
                # Convert data to torch tensor on GPU
                if isinstance(data, torch.Tensor):
                    data_tensor = data.to(device).float()
                else:
                    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
                
                # Run PyTorch K-means
                centroids, labels = self._pytorch_kmeans(data_tensor, int(self.n_clusters))
                
                # Store GPU model for prediction
                self.model = {'centroids': centroids, 'device': device, 'type': 'pytorch'}
                self.labels_ = labels.cpu().numpy()
                return self.labels_
                
            except Exception as e:
                print(f"GPU K-means failed: {e}, falling back to CPU implementation")
                self.use_gpu = False
        
        # CPU fallback using scikit-learn
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=int(self.n_clusters), verbose=self.verbose)
        labels = kmeans.fit_predict(data)
        self.model = {'sklearn_model': kmeans, 'type': 'sklearn'}
        self.labels_ = labels
        return labels
        
    def determine_n_clusters(self, data, max_clusters=100):
        if isinstance(data, csr_matrix):
            # If data is already an affinity matrix, use it directly
            affinity_matrix = data
        else:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            # Normalize data to unit norm
            data_norm = np.linalg.norm(data, axis=1, keepdims=True)
            data_norm[data_norm == 0] = 1  # Avoid division by zero
            data_normalized = data / data_norm
            
            # Create affinity matrix using nearest neighbors
            spectral = SpectralClustering(
                n_clusters=2,  # Dummy value
                affinity='nearest_neighbors',
                assign_labels='kmeans'
            )
            spectral.fit(data_normalized)
            affinity_matrix = spectral.affinity_matrix_
        
        # Compute normalized Laplacian
        degrees = affinity_matrix.sum(axis=1).A1
        # Add small epsilon to avoid division by zero
        degrees = np.maximum(degrees, 1e-10)
        D_inv_sqrt = csr_matrix((1/np.sqrt(degrees), (np.arange(len(degrees)), np.arange(len(degrees)))))
        L = eye(len(degrees)) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        eigenvalues = np.sort(eigenvalues)
        
        # Remove very small eigenvalues (numerical noise)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Normalize eigenvalues to [0,1] range
        if len(eigenvalues) > 0:
            eigenvalues = (eigenvalues - eigenvalues[0]) / (eigenvalues[-1] - eigenvalues[0])
        
        # Find the largest gap between consecutive eigenvalues
        if len(eigenvalues) > 1:
            gaps = np.diff(eigenvalues[:min(max_clusters+1, len(eigenvalues))])
            optimal_k = np.argmax(gaps) + 1
        else:
            optimal_k = 1
        
        # Ensure optimal_k is at least 2 and not too large
        optimal_k = max(2, min(optimal_k, min(max_clusters, len(data) // 2)))
        
        return optimal_k

    def _fit_spectral(self, data):
        if self.n_clusters is None:
            self.n_clusters = self.determine_n_clusters(data)
            print(f"Automatically determined number of clusters: {self.n_clusters}")
        
        # Initialize spectral fitting method tracker
        self._spectral_method = None
        
        if self.use_gpu:
            try:
                from torch_geometric.nn import knn_graph
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Convert data to torch tensor on GPU
                if isinstance(data, torch.Tensor):
                    data_tensor = data.to(device)
                else:
                    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                
                # Store original data shape and device for prediction
                self._original_data_shape = data_tensor.shape
                self._device = device
                
                # Normalize data
                data_normalized = torch.nn.functional.normalize(data_tensor, p=2, dim=1)
                
                # Build k-NN graph using PyG
                edge_index = knn_graph(data_normalized, k=self.n_neighbors, batch=None, loop=False)
                
                # Convert to adjacency matrix
                num_nodes = data_normalized.size(0)
                edge_weight = torch.ones(edge_index.size(1), device=device)
                
                # Create symmetric adjacency matrix
                adj_indices = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                adj_values = torch.cat([edge_weight, edge_weight])
                adj_matrix = torch.sparse_coo_tensor(adj_indices, adj_values, (num_nodes, num_nodes))
                adj_matrix = adj_matrix.coalesce()
                
                # Compute degree matrix
                degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
                degrees = torch.clamp(degrees, min=1e-10)  # Avoid division by zero
                
                # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
                deg_inv_sqrt = torch.pow(degrees, -0.5)
                deg_inv_sqrt = torch.diag(deg_inv_sqrt)
                
                # Convert to dense for eigendecomposition
                adj_dense = adj_matrix.to_dense()
                laplacian = torch.eye(num_nodes, device=device) - deg_inv_sqrt @ adj_dense @ deg_inv_sqrt
                
                # Compute eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
                
                # Take the smallest n_clusters eigenvectors (spectral embedding)
                spectral_embedding = eigenvectors[:, :int(self.n_clusters)]
                
                # Save eigenvectors for prediction (move to CPU to save memory)
                self._spectral_eigenvectors = spectral_embedding.cpu().numpy().astype(np.float32)
                
                # Normalize using PyTorch (L2 normalization)
                spectral_embedding_normalized = torch.nn.functional.normalize(spectral_embedding, p=2, dim=1)
                
                # Use scikit-learn k-means (removed cuML dependency)
                from sklearn.cluster import KMeans
                spectral_embedding_cpu = spectral_embedding_normalized.cpu().numpy()
                kmeans_sklearn = KMeans(n_clusters=int(self.n_clusters), n_init='auto')
                labels = kmeans_sklearn.fit_predict(spectral_embedding_cpu)
                
                # Save sklearn model and method for prediction
                self.model = kmeans_sklearn
                self._spectral_method = 'gpu_sklearn'
                
                self.labels_ = labels
                return labels
                
            except ImportError as e:
                print(f"GPU dependencies not available: {e}")
                print("Falling back to CPU implementation...")
                self.use_gpu = False
        
        if isinstance(data, csr_matrix):
            spectral = SpectralClustering(
                n_clusters=int(self.n_clusters),
                affinity=self.affinity,
                assign_labels='kmeans'
            )
            labels = spectral.fit_predict(data)
            
            # Save model and method for prediction
            self.model = spectral
            self._spectral_method = 'cpu_sparse'
            self.labels_ = labels
            
        else:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            # No scaling - use raw data
            spectral = SpectralClustering(
                n_clusters=int(self.n_clusters),
                n_neighbors=self.n_neighbors,
                affinity=self.affinity,
                assign_labels='kmeans'
            )
            
            labels = spectral.fit_predict(data)
            
            # Save model and method for prediction
            self.model = spectral
            self._spectral_method = 'cpu_dense'
            self.labels_ = labels
        
        return self.labels_

    def _fit_kmeans_sampled(self, data, labels=None):
        data = self._maybe_torch_to_numpy(data)
        # No scaling - use raw data
        
        n_embeddings = data.shape[0]
        v_measures = []

        clustering_model = MiniBatchKMeans(
            n_clusters=int(self.n_clusters),
            batch_size=self.batch_size,
            n_init="auto",
        )
        
        rng_state = random.Random()
        if labels is not None:
            # Add cluster_size attribute if not present
            if not hasattr(self, 'cluster_size'):
                self.cluster_size = min(1000, n_embeddings // 10)  # Default cluster size
                
            for _ in range(int(self.n_clusters)):
                cluster_indices = rng_state.choices(range(n_embeddings), k=self.cluster_size)
                _embeddings = data[cluster_indices]
                _labels = labels[cluster_indices]
                cluster_assignment = clustering_model.fit_predict(_embeddings)
                v_measure = v_measure_score(_labels, cluster_assignment)
                v_measures.append(v_measure)

            logger.debug(f"Cluster V-measure: {np.mean(v_measures)}, std: {np.std(v_measures)}")
        
        cluster_labels = clustering_model.fit_predict(data)
        self.model = clustering_model
        self.labels_ = cluster_labels
        return self.labels_

    def _predict_kmeans_sampled(self, data):
        data = self._maybe_torch_to_numpy(data)
        return self.model.predict(data)

