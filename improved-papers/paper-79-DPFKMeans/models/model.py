from dataclasses import dataclass
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Tuple, Callable, Optional


from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType, ModelHyperParams
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.model.base import EvaluatableModel, ModelType
from pfl.stats import StatisticsType, MappedVectorStatistics

from utils import kmeans_cost, compute_num_correct, awasthisheffet_kmeans


class KMeansModel(EvaluatableModel):
    def __init__(self, centers: np.array):
        super().__init__()
        self.centers = centers

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['KMeansModel', Metrics]:
        pass

    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn: Callable[[str], StringMetricName],
            eval_params: Optional[ModelHyperParamsType] = None) -> Metrics:

        if len(dataset.raw_data) == 2:
            X, Y = dataset.raw_data
        else:
            X = dataset.raw_data[0]
            Y = None

        dist_matrix = pairwise_distances(X, self.centers)
        Y_pred = np.argmin(dist_matrix, axis=1)
        cost = kmeans_cost(X, self.centers, y_pred=Y_pred)
        num_correct = 0
        if Y is not None:
            num_correct = compute_num_correct(Y_pred, Y)

        cost_metric = Weighted(cost, len(X))
        accuracy_metric = Weighted(num_correct, len(X))
        metrics = Metrics([(name_formatting_fn('kmeans-cost'), cost_metric),
                           (name_formatting_fn('kmeans-accuracy'), accuracy_metric)
                           ])

        return metrics


@dataclass(frozen=True)
class LloydsModelHyperParams(ModelHyperParams):
    K: int


class LloydsModel(KMeansModel):
    def __init__(self, centers: np.array):
        super().__init__(centers)

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['LloydsModel', Metrics]:
        if 'mean_points_per_component' in statistics.keys():
            # statistics.average()
            self.centers = statistics['mean_points_per_component'] / statistics['contributed_components'].reshape(-1, 1)
        else:
            mask = statistics['num_points_per_component'] == 0
            statistics['num_points_per_component'][mask] = 1
            self.centers = (statistics['sum_points_per_component']
                            / statistics['num_points_per_component'].reshape(-1, 1))

        return self, Metrics()


@dataclass(frozen=True)
class FedClusterInitModelHyperParams(ModelHyperParams):
    K: int


class FedClusterInitModel(EvaluatableModel):
    def __init__(self):
        super().__init__()
        self.accumulated_statistics = {}
        self.singular_vectors = None
        self.proj_X_server = None
        self.proj_server_point_centers = None
        self.initial_centers = None

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['FedClusterInitModel', Metrics]:
        for statistic_name, statistic_val in statistics.items():
            try:
                self.accumulated_statistics[statistic_name] += statistic_val
            except KeyError:
                self.accumulated_statistics[statistic_name] = statistic_val


        return self, Metrics()

    def compute_centers(self):
        if 'sum_points_per_component' in self.accumulated_statistics.keys():
            point_sums = self.accumulated_statistics['sum_points_per_component']
            point_counts = self.accumulated_statistics['num_points_per_component']
            self.initial_centers = point_sums / point_counts.reshape(-1, 1)
        else:
            # print(self.accumulated_statistics['contributed_components'])
            self.initial_centers = self.accumulated_statistics['mean_points_per_component'] / self.accumulated_statistics['contributed_components'].reshape(-1, 1)

    def compute_centers_server_data(self, X_server, algorithm_params):
        # In this case we compute initial centers using the weighted server data
        assert self.proj_server_point_centers is not None
        proj_dist_matrix = pairwise_distances(self.proj_X_server, self.proj_server_point_centers)
        center_assignments = -np.ones(len(self.proj_X_server), dtype=int)
        smallest_two_distances = np.partition(proj_dist_matrix, 1, axis=1)[:, :2]
        assignment_mask = (smallest_two_distances[:, 0] <=
                           (algorithm_params.multiplicative_margin * smallest_two_distances[:, 1]))
        center_assignments[assignment_mask] = np.argmin(proj_dist_matrix, axis=1)[assignment_mask]

        hint_weights = self.accumulated_statistics['server_point_weights'].reshape(-1, 1)
        initial_centers = []
        for k in range(algorithm_params.K):
            kth_mask = center_assignments == k
            initial_centers.append((X_server * hint_weights)[kth_mask].sum(axis=0)
                                   / hint_weights[kth_mask].sum())

        self.initial_centers = np.vstack(initial_centers)

    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn: Callable[[str], StringMetricName],
            eval_params: Optional[ModelHyperParamsType] = None) -> Metrics:

        if self.initial_centers is None:
            return Metrics()

        if len(dataset.raw_data) == 2:
            X, Y = dataset.raw_data
        else:
            X = dataset.raw_data[0]
            Y = None
        dist_matrix = pairwise_distances(X, self.initial_centers)
        Y_pred = np.argmin(dist_matrix, axis=1)
        cost = kmeans_cost(X, self.initial_centers, y_pred=Y_pred)
        num_correct = 0
        if Y is not None:
            num_correct = compute_num_correct(Y_pred, Y)

        cost_metric = Weighted(cost, len(X))
        accuracy_metric = Weighted(num_correct, len(X))
        metrics = Metrics([(name_formatting_fn('kmeans-cost'), cost_metric),
                           (name_formatting_fn('kmeans-accuracy'), accuracy_metric)
                           ])

        return metrics


@dataclass(frozen=True)
class KFedModelHyperParams(ModelHyperParams):
    K: int
    K_client: int


class KFedModel(KMeansModel):
    def __init__(self, centers, K, K_client):
        super().__init__(centers)
        self.client_local_centers = []
        self.K = K
        self.K_client = K_client

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['KFedModel', Metrics]:

        local_client_centers = statistics['local_client_centers']
        for client_idx in range(len(local_client_centers) // self.K_client):
            self.client_local_centers.append(
                local_client_centers[client_idx * self.K_client: (client_idx + 1) * self.K_client])

        return self, Metrics()

    def initialize_kfed_centers(self):
        assert len(self.client_local_centers)
        i = np.random.choice(range(len(self.client_local_centers)))
        initial_centers = self.client_local_centers.pop(i)
        remaining_centers = np.vstack(self.client_local_centers)

        dist_to_initial_centers = np.min(pairwise_distances(remaining_centers, initial_centers), axis=1)

        M = list(initial_centers)
        while len(M) < self.K:
            assert len(dist_to_initial_centers) == len(remaining_centers)
            i = np.argmax(dist_to_initial_centers)
            M.append(remaining_centers[i])
            dists_to_new_point = np.sqrt(np.sum((remaining_centers - remaining_centers[i])**2, axis=1))
            dist_to_initial_centers = np.min(np.array([dist_to_initial_centers, dists_to_new_point]), axis=0)

        self.centers = np.vstack(M)




