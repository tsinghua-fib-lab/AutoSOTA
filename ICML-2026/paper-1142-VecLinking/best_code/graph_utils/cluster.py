import numpy as np
from utils.clustering import Clusterer
import torch
from utils.graph_util import get_dists


class Cluster:
    def __init__(self, emb, ori_ind, n_clu, cluster_method, graph_method='fc', knn_k=10, sample=False, use_gpu=True):
        # Store original data format and device info
        self.is_tensor = isinstance(emb, torch.Tensor)
        if self.is_tensor:
            self.device = emb.device
            self.emb = emb
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
            self.emb = emb  # Keep as numpy for now, convert only when needed

        self.ori_ind = ori_ind
        self.n_clu = n_clu
        self.cluster_method = cluster_method
        self.graph_method = graph_method
        self.knn_k = knn_k
        self.sample = sample
        self.use_gpu = use_gpu
        self.labels = self.cluster(self.sample)
        if self.is_tensor:
            self.label_list = torch.unique(self.labels)
        else:
            self.label_list = np.unique(self.labels)

    def cluster(self, sample=False, ground_truth=None):
        # Always use numpy for clustering (sklearn requirement)
        emb_for_clustering = self.emb.cpu().numpy() if self.is_tensor else self.emb

        if sample:
            batch_size = 100 * self.n_clu
            if ground_truth is not None:
                self.clusterer = Clusterer(method='kmeans_sampled', n_clusters=self.n_clu, batch_size=batch_size, labels=ground_truth)
            else:
                self.clusterer = Clusterer(method='kmeans_sampled', n_clusters=self.n_clu, batch_size=batch_size)
        else:
            if self.cluster_method == "kmeans":
                self.clusterer = Clusterer(method='kmeans', n_clusters=self.n_clu, use_gpu=self.use_gpu)
            elif self.cluster_method == "spectral":
                self.clusterer = Clusterer(method='spectral', n_clusters=self.n_clu, use_gpu=self.use_gpu)
            else:
                raise ValueError(f"Invalid cluster method: {self.cluster_method}")

        labels = self.clusterer.fit(emb_for_clustering)

        # Return labels in same format as input
        if self.is_tensor:
            return torch.from_numpy(labels).to(self.device)
        else:
            return labels

    def get_affinity_matrix(self, distance_metric):
        emb_clu_affinity = {}

        # Get distance matrix
        if not hasattr(self, 'dist_matrix'):
            self.dist_matrix = get_dists(self.emb, metric=distance_metric)

        # Handle tensor vs numpy operations
        if self.is_tensor:
            dist_matrix_work = self.dist_matrix.cpu().numpy()  # Convert for numpy operations
            labels_work = self.labels.cpu().numpy()
        else:
            dist_matrix_work = self.dist_matrix
            labels_work = self.labels

        for emb_label in np.unique(labels_work):
            emb_cluster_indices = np.where(labels_work == emb_label)[0]

            if self.graph_method == 'fc':
                dist_matrix = dist_matrix_work[np.ix_(emb_cluster_indices, emb_cluster_indices)]
                sigma = np.median(dist_matrix[dist_matrix > 0])
                sigma = max(sigma, 1e-8)
                affinity_matrix = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
                np.fill_diagonal(affinity_matrix, 0)

            elif self.graph_method == 'knn':
                from sklearn.neighbors import kneighbors_graph

                # Get cluster embeddings
                if self.is_tensor:
                    emb_cluster_embs = self.emb[emb_cluster_indices].cpu().numpy()
                else:
                    emb_cluster_embs = self.emb[emb_cluster_indices]

                # Build symmetric k-NN graph
                knn_graph = kneighbors_graph(emb_cluster_embs, self.knn_k, mode='distance', include_self=False, metric=distance_metric)
                knn_graph = knn_graph.maximum(knn_graph.T).toarray()

                dist_matrix = knn_graph
                sigma = np.median(dist_matrix[dist_matrix > 0])
                sigma = max(sigma, 1e-8)

                affinity_matrix = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
                affinity_matrix[knn_graph == 0] = 0
                np.fill_diagonal(affinity_matrix, 0)

            else:
                raise ValueError(f"Invalid graph method: {self.graph_method}")

            emb_clu_affinity[emb_label] = affinity_matrix
        return emb_clu_affinity

    def get_dist_matrix(self, distance_metric):
        self.dist_matrix = get_dists(self.emb, metric=distance_metric)
        emb_clu_dist = {}

        # Handle tensor vs numpy operations
        if self.is_tensor:
            dist_matrix_work = self.dist_matrix.cpu().numpy()
            labels_work = self.labels.cpu().numpy()
        else:
            dist_matrix_work = self.dist_matrix
            labels_work = self.labels

        for emb_label in np.unique(labels_work):
            emb_cluster_indices = np.where(labels_work == emb_label)[0]
            dist_matrix = dist_matrix_work[np.ix_(emb_cluster_indices, emb_cluster_indices)]
            emb_clu_dist[emb_label] = dist_matrix
        return emb_clu_dist

    def get_ind(self, clu_label):
        if self.is_tensor:
            clu_ind = torch.where(self.labels == clu_label)[0]
            return clu_ind.cpu().numpy()  # Return numpy for indexing compatibility
        else:
            clu_ind = np.where(self.labels == clu_label)[0]
            return clu_ind

    def get_ori_ind(self, clu_label):
        clu_ind = self.get_ind(clu_label)
        if isinstance(self.ori_ind, torch.Tensor):
            return self.ori_ind[clu_ind].cpu().numpy()
        else:
            return np.array(self.ori_ind)[clu_ind]

    def get_clu_emb(self, clu_label):
        if self.is_tensor:
            clu_ind = torch.where(self.labels == clu_label)[0]
            return self.emb[clu_ind]
        else:
            clu_ind = np.where(self.labels == clu_label)[0]
            return self.emb[clu_ind]