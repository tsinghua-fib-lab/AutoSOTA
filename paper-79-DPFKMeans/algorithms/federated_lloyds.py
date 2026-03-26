from dataclasses import dataclass
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import Dataset
from pfl.hyperparam.base import AlgorithmHyperParams
from pfl.metrics import Metrics, Weighted, MetricName
from pfl.stats import MappedVectorStatistics

from utils import compute_num_correct, kmeans_cost
from models import LloydsModel, LloydsModelHyperParams


@dataclass(frozen=True)
class FederatedLloydsHyperParams(AlgorithmHyperParams):
    K: int
    send_sums_and_counts: int
    central_num_iterations: int
    evaluation_frequency: int
    train_cohort_size: int
    val_cohort_size: Optional[int]


class FederatedLloyds(FederatedAlgorithm):
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[FederatedLloydsHyperParams,
                                                  LloydsModelHyperParams],
            aggregate_metrics: Metrics, model: LloydsModel,
            statistics: MappedVectorStatistics) -> Tuple[LloydsModel, Metrics]:

        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: LloydsModel, user_dataset: Dataset,
        central_context: CentralContext[FederatedLloydsHyperParams,
                                        LloydsModelHyperParams]
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:

        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        algo_params = central_context.algorithm_params
        if central_context.population == Population.TRAIN:
            if len(user_dataset.raw_data) == 2:
                X, Y = user_dataset.raw_data
            else:
                X = user_dataset.raw_data[0]
                Y = None
            dist_matrix = pairwise_distances(X, model.centers)
            Y_pred = np.argmin(dist_matrix, axis=1)

            sum_points_per_component = []
            num_points_per_component = []
            for k in range(algo_params.K):
                component_mask = Y_pred == k
                sum_of_points = np.sum(X[component_mask], axis=0)
                assert len(sum_of_points) == X.shape[1]
                sum_points_per_component.append(sum_of_points)
                num_points_per_component.append(sum(component_mask))

            sum_points_per_component = np.array(sum_points_per_component)
            num_points_per_component = np.array(num_points_per_component)

            statistics = MappedVectorStatistics()
            if algo_params.send_sums_and_counts:
                statistics['sum_points_per_component'] = sum_points_per_component
                statistics['num_points_per_component'] = num_points_per_component
            else:
                statistics['contributed_components'] = (num_points_per_component > 0).astype(int)
                num_points_per_component[num_points_per_component == 0] = 1
                statistics['mean_points_per_component'] = (sum_points_per_component /
                                                           num_points_per_component.reshape(-1, 1))

            cost = kmeans_cost(X, model.centers, y_pred=Y_pred)
            num_correct = 0
            if Y is not None:
                num_correct = compute_num_correct(Y_pred, Y)

            cost_metric = Weighted(cost, len(X))
            accuracy_metric = Weighted(num_correct, len(X))
            metrics = Metrics([(name_formatting_fn('kmeans-cost'), cost_metric),
                               (name_formatting_fn('kmeans-accuracy'), accuracy_metric)
                               ])

            return statistics, metrics

        else:
            return None, model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: LloydsModel,
        iteration: int,
        algorithm_params: FederatedLloydsHyperParams,
        model_train_params: LloydsModelHyperParams,
        model_eval_params: Optional[LloydsModelHyperParams] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], LloydsModel,
               Metrics]:

        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        do_evaluation = iteration % algorithm_params.evaluation_frequency == 0

        configs = [
            CentralContext(current_central_iteration=iteration,
                           do_evaluation=do_evaluation,
                           cohort_size=algorithm_params.train_cohort_size,
                           population=Population.TRAIN,
                           model_train_params=model_train_params.static_clone(),
                           model_eval_params=model_eval_params.static_clone(),
                           algorithm_params=algorithm_params.static_clone(),
                           seed=self._get_seed())
        ]

        if do_evaluation and algorithm_params.val_cohort_size is not None:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    algorithm_params=algorithm_params.static_clone(),
                    model_train_params=model_train_params.static_clone(),
                    model_eval_params=model_eval_params.static_clone(),
                    seed=self._get_seed()))
        return tuple(configs), model, Metrics()
