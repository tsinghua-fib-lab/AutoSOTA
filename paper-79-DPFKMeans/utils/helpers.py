import argparse
import numpy as np
import os
import random
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from typing import List, Tuple

from pfl.data.dataset import Dataset
from pfl.metrics import StringMetricName
from pfl.model.base import EvaluatableModel
from pfl.privacy.privacy_mechanism import PrivacyMechanism

from privacy import compute_privacy_accounting


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def shorten_name(arg_name):
    return ''.join([s[0] for s in arg_name.split('_')])


def make_results_path(privacy_type:str, dataset: str):
    path = os.path.join('results', privacy_type, dataset)
    os.makedirs(path, exist_ok=True)
    return path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def project_subspace(V: np.array, U: np.array):
    """
    Projection of each row of U into the subspace spanned by the rows of V

    :param V: np.array, the rows are the orthogonal spanning vectors
    :param U: np.array, each row is a vector we wish to project
    :return: np.array, each row is a projected vector
    """
    return (V.T @ ((V @ U.T) / np.sum(V * V, axis=1, keepdims=True))).T


def compute_num_correct(y_pred: np.array, y_true: np.array):
    """
    Computes the number of correct cluster assignments. A point that has predicted cluster
    j is correctly clustered if it belongs to the majority true cluster among all points
    that have been assigned to j.

    :param y_pred: predicted cluster assignments
    :param y_true: true cluster assignments
    :return: num_correct: the number of correctly clustered points
    """
    unique_predicted_clusters = np.unique(y_pred)
    num_correct = 0
    cluster_preds = []
    for j in unique_predicted_clusters:
        i = np.argmax(np.unique(y_true[y_pred == j], return_counts=True)[1])
        cluster_preds.append(np.unique(y_true[y_pred == j], return_counts=True)[0][i])
        num_correct += np.max(np.unique(y_true[y_pred == j], return_counts=True)[1])
    # print(np.unique(cluster_preds, return_counts=True))
    return num_correct


def kmeans_cost(P, centers, y_pred=None):
    """

    :param P: np.array of shape (num_datapoints, dimension)
    :param centers: np.array of shape (num_centers, dimension)
    :param y_pred: Optional, np.array of shape (num_datapoints). Cluster assignments if already computed.
    :return: float, kmeans cost.
    """
    if y_pred is None:
        dist_matrix = pairwise_distances(P, centers)
        y_pred = np.argmin(dist_matrix, axis=1)

    return ((P - centers[y_pred]) ** 2).sum()


def awasthisheffet_kmeans(P: np.array, k: int, max_iter: int = 10, random_svd: bool = True, mult_margin: float=0.9):
    """
    Centralized kmeans clustering of P following [Awasthi Sheffet, 2012]

    :param P: array of shape (n, d), n number of points, d dimension.
    :param k: number of clusters to infer
    :param max_iter: maximum number of lloyds iterations to use
    :param random_svd: use randomized svd (fast) or exact svd (slow)
    :return: center_assignments, array of shape (n,), cluster assignment for each point
    """
    # Part 1a: Project P onto subspace spanned by top k singular vectors
    if random_svd:
        _, _, V = randomized_svd(P, n_components=k, random_state=None)
    else:
        _, _, V = svd(P)
        V = V[:k]
    P_proj = project_subspace(V, P)

    # Part 1b: Cluster the projected points
    projected_kmeans = KMeans(n_clusters=k).fit(P_proj)
    projected_centers = projected_kmeans.cluster_centers_

    # Part 2: Initialize centers in original space using 1/3 proximity condition
    dist_matrix = pairwise_distances(P_proj, projected_centers)
    center_assignments = -1 * np.ones(len(P_proj), dtype=int)
    smallest_two_distances = np.partition(dist_matrix, 1, axis=1)[:, :2]
    assignment_mask = smallest_two_distances[:, 0] <= (mult_margin * smallest_two_distances[:, 1])
    center_assignments[assignment_mask] = np.argmin(dist_matrix, axis=1)[assignment_mask]
    initial_centers = []
    for j in range(k):
        initial_centers.append(np.mean(P[center_assignments == j], axis=0))
    initial_centers = np.vstack(initial_centers)

    # Part 3: Run Lloyds for max_iters using computed initial centers
    original_space_kmeans = KMeans(n_clusters=k, init=initial_centers, max_iter=max_iter).fit(P)

    return original_space_kmeans.labels_, original_space_kmeans.cluster_centers_


def post_evaluation(results_dict: dict, args: argparse.Namespace, model: EvaluatableModel,
                    data: Tuple[Dataset], when: str, executed_privacy_mechanisms: List[PrivacyMechanism],
                    num_compositions: List[int], sampling_probs: List[float], verbose: bool):
    metrics_train = model.evaluate(data[0], lambda s: StringMetricName(s))
    metrics_val = model.evaluate(data[1], lambda s: StringMetricName(s))

    train_cost = metrics_train['kmeans-cost'].overall_value
    train_acc = metrics_train['kmeans-accuracy'].overall_value

    val_cost = metrics_val['kmeans-cost'].overall_value
    val_acc = metrics_val['kmeans-accuracy'].overall_value

    if verbose:
        print()
        print(f'Evaluating {when}...')
        print()
        print('Train Client Metrics:')
        print(f"Cost {train_cost:.4f}",
              f"Accuracy {train_acc:.4f}")
        print()
        print()
        print('Validation Client Metrics:')
        print(f"Cost {val_cost:.4f}",
              f"Accuracy {val_acc:.4f}")
        print()

    if when != 'Optimal':
        if when == 'Initialization':
            target_delta = args.initialization_target_delta
        elif when == 'Clustering':
            target_delta = args.overall_target_delta
        else:
            raise ValueError('When not recognized.')

        accountant_epsilon = compute_privacy_accounting(
            executed_privacy_mechanisms,
            target_delta,
            num_compositions=num_compositions,
            sampling_probs=sampling_probs
        )

        privacy_cost = (accountant_epsilon, target_delta)
        if verbose:
            print(f'{when} is ({accountant_epsilon}, {target_delta})-DP')
            print()
    else:
        privacy_cost = (0, 0)

    results_dict[when] = {
        'Train client cost': train_cost,
        'Train client accuracy': train_acc,
        'Val client cost': val_cost,
        'Val client accuracy': val_acc,
        'Privacy cost': privacy_cost,
    }


def generate_sphere_packing_centroids(a, r, d, k, num_trys):
    i = 0
    centroids = []
    while len(centroids) < k:
        new_point = np.random.uniform(-r, r, size=d)
        closest_corner = np.sign(new_point) * r
        dist_to_corner = np.sqrt(np.sum((new_point - closest_corner)**2))
        if dist_to_corner > a:
            if not centroids:
                centroids.append(new_point)
            else:
                centroids_numpy = np.array(centroids)
                dist_to_centroids = np.min(np.sqrt(np.sum((centroids_numpy - new_point)**2, axis=1)))
                if dist_to_centroids > 2 * a:
                    centroids.append(new_point)

        i += 1
        if i == num_trys:
            break

    return len(centroids) == k, centroids


def kmeans_initialise_sphere_packing(r, d, k, num_trys, tol, max_iters_binary_search):
    rad_low, rad_high = 0, r * np.sqrt(d)
    iter_count = 0
    success = False
    while (rad_high - rad_low > tol) and not success:
        rad_mid = (rad_low + rad_high) / 2
        success, centroids = generate_sphere_packing_centroids(rad_mid, r, d, k, num_trys)
        if success:
            rad_low = rad_mid
        else:
            rad_high = rad_mid

        iter_count += 1
        if iter_count == max_iters_binary_search:
            print(f'Binary search failed to converge after {max_iters_binary_search} iterations.')
            break

    return np.array(centroids)
