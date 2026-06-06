from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import AlgorithmHyperParamsType, ModelHyperParamsType, AlgorithmHyperParams
from pfl.metrics import Metrics, MetricName
from pfl.model.base import ModelType
from pfl.stats import MappedVectorStatistics, StatisticsType

from models import KFedModel
from utils import awasthisheffet_kmeans


@dataclass(frozen=True)
class KFedHyperParams(AlgorithmHyperParams):
    K: int
    K_client: int
    userid_to_idx: dict
    multiplicative_margin: float
    train_cohort_size: int
    val_cohort_size: Optional[int]


class KFed(FederatedAlgorithm):
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            algo_params = central_context.algorithm_params
            K_client = algo_params.K_client
            X, _ = user_dataset.raw_data
            dists = pairwise_distances(X, X)
            np.fill_diagonal(dists, 1)
            num_unique_points = np.isclose(dists, 0).sum() / 2

            if num_unique_points > K_client:
                try:
                    _, client_centers = awasthisheffet_kmeans(X, K_client, max_iter=100,
                                                              random_svd=False, mult_margin=algo_params.multiplicative_margin)
                except ValueError:
                    client_centers = np.vstack([X for _ in range(K_client // len(X) + 1)])[:K_client]

            else:
                client_centers = np.vstack([X for _ in range(K_client // len(X) + 1)])[:K_client]

            idx = algo_params.userid_to_idx[user_dataset.user_id]
            if (idx + 1) % 1000 == 0:
                print(f'Running user {idx+1}.')
            statistics['local_client_centers'] = np.zeros(
                (algo_params.train_cohort_size * K_client, client_centers.shape[1]))

            statistics['local_client_centers'][idx * K_client: (idx + 1) * K_client] = client_centers

            return statistics, Metrics()

        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: KFedModel,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
               Metrics]:

        if iteration == 1:
            model.initialize_kfed_centers()
            lloyds_single_step = KMeans(n_clusters=algorithm_params.K, init=model.centers, max_iter=1)
            lloyds_single_step.fit(np.vstack(model.client_local_centers))
            model.centers = lloyds_single_step.cluster_centers_
            return None, model, Metrics()

        context = CentralContext(
                current_central_iteration=iteration,
                do_evaluation=False,
                cohort_size=algorithm_params.train_cohort_size,
                population=Population.TRAIN,
                model_train_params=model_train_params.static_clone(),
                model_eval_params=model_eval_params.static_clone(),
                algorithm_params=algorithm_params.static_clone(),
                seed=self._get_seed())

        return tuple([context]), model, Metrics()
