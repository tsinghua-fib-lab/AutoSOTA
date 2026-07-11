import numpy as np
import math
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
from sklearn import cluster
from sklearn.preprocessing import Normalizer
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def normalize_multiview_data(data_views, row_normalized=True):
    '''The rows or columns of a matrix normalized '''
    norm2 = Normalizer(norm='l2')
    num_views = len(data_views)
    for idx in range(num_views):
        if row_normalized:
            data_views[idx] = norm2.fit_transform(data_views[idx])
        else:
            data_views[idx] = norm2.fit_transform(data_views[idx].T).T

    return data_views


def spectral_clustering(W, num_clusters):
    """
    Apply spectral clustering on W.
    # Arguments
    :param W: an affinity matrix
    :param num_clusters: the number of clusters
    :return: cluster labels.
    """
    # spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed',
    #                                       assign_labels='discretize')

    assign_labels='kmeans'
    spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(W)
    labels = spectral.fit_predict(W)

    return labels


def cal_spectral_embedding(W, num_clusters):

    D = np.diag(1 / np.sqrt(np.sum(W, axis=1) + math.e))
    # D1 = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    Z = np.dot(np.dot(D, W), D)
    U, _, _ = np.linalg.svd(Z)
    eigenvectors = U[:, 0 : num_clusters]

    return eigenvectors


def cal_spectral_embedding_1(W, num_clusters):
    D = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    L = np.eye(len(W)) - np.dot(np.dot(D, W), D)
    eigvals, eigvecs = np.linalg.eig(L)
    x_val = []
    x_vec = np.zeros((len(eigvecs[:, 0]), len(eigvecs[0])))
    for i in range(len(eigvecs[:, 0])):
        for j in range(len(eigvecs[0])):
            x_vec[i][j] = eigvecs[i][j].real
    for i in range(len(eigvals)):
        x_val.append(eigvals[i].real)
    # Select the first n smallest eigenvectors
    indices = np.argsort(x_val)[: num_clusters]
    eigenvectors = x_vec[:, indices[: num_clusters]]

    return eigenvectors


def cal_l2_distances(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sqrt(np.sum(np.square(data_view - data_view[i]), axis=1)).T
    return dists


def cal_l2_distances_1(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sqrt(np.sum(np.square(data_view[i]-data_view[j])))

    return dists


def cal_squared_l2_distances(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sum(np.square(data_view - data_view[i]), axis=1).T
    return dists


def cal_squared_l2_distances_1(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sum(np.square(data_view[i]-data_view[j]))

    return dists


def cal_similiarity_matrix(data_view, k):
    '''
    calculate similiarity matrix
    '''
    num_samples = data_view.shape[0]
    dist = cal_squared_l2_distances(data_view)

    W = np.zeros((num_samples,num_samples), dtype=float)

    idx_set = dist.argsort()[::1]
    for i in range(num_samples):
        idx_sub_set = idx_set[i, 1:(k+2)]
        di = dist[i, idx_sub_set]
        W[i, idx_sub_set] = (di[k]-di) / (di[k] - np.mean(di[0:(k-1)]) + math.e)

    W = (W + W.T) / 2

    return W
