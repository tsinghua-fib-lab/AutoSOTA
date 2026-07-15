import threading
from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future
from omegaconf import DictConfig
from appfl.algorithm.scheduler import BaseScheduler
from appfl.algorithm.aggregator import BaseAggregator
from appfl.misc.memory_utils import optimize_memory_cleanup


class SyncScheduler(BaseScheduler):
    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = {}
        self.aggregation_kwargs = {}
        self.future = {}
        self.num_clients = self.scheduler_configs.num_clients
        self._num_global_epochs = 0
        self._access_lock = threading.Lock()

        # Check for optimize_memory in scheduler_configs, default to True
        self.optimize_memory = getattr(scheduler_configs, "optimize_memory", True)

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        """
        Schedule a synchronous global aggregation for the local model from a client.
        The method will return a future object for the aggregated model, which will
        be set after all clients have submitted their local models for the global aggregation.
        :param client_id: the id of the client
        :param local_model: the local model from a client
        :param kwargs: additional keyword arguments for the scheduler
        :return: the future object for the aggregated model
        """
        with self._access_lock:
            future = Future()

            # Memory optimization: Store models efficiently
            if self.optimize_memory:
                # For memory efficiency, avoid creating unnecessary copies
                self.local_models[client_id] = local_model
                # Force garbage collection after storing each model for large models
                if len(self.local_models) > self.num_clients // 2:
                    optimize_memory_cleanup(force_gc=True)
            else:
                self.local_models[client_id] = local_model

            for key, value in kwargs.items():
                if key not in self.aggregation_kwargs:
                    self.aggregation_kwargs[key] = {}
                self.aggregation_kwargs[key][client_id] = value
            self.future[client_id] = future

            if len(self.local_models) == self.num_clients:
                is_final_chunk = True
                if (
                    "_chunk_idx" in self.aggregation_kwargs
                    and "_total_chunks" in self.aggregation_kwargs
                ):
                    # Extract from first client (all should be same)
                    chunk_idx = list(self.aggregation_kwargs["_chunk_idx"].values())[0]
                    total_chunks = list(
                        self.aggregation_kwargs["_total_chunks"].values()
                    )[0]
                    is_final_chunk = chunk_idx == total_chunks - 1

                # Memory optimization: Process aggregation with cleanup
                if self.optimize_memory:
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )

                    # Immediate cleanup of local models after aggregation
                    temp_futures = dict(self.future)  # Create copy before clearing
                    self.local_models.clear()  # Clear immediately after aggregation
                    self.aggregation_kwargs.clear()  # Clear aggregation kwargs
                    optimize_memory_cleanup(force_gc=True)

                    # Set results for all futures
                    while temp_futures:
                        client_id, client_future = temp_futures.popitem()
                        client_future.set_result(
                            self._parse_aggregated_model(aggregated_model, client_id)
                        )
                    self.future.clear()

                    # Final cleanup after setting all results
                    optimize_memory_cleanup(temp_futures, force_gc=True)
                else:
                    # Original behavior
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )
                    while self.future:
                        client_id, future = self.future.popitem()
                        future.set_result(
                            self._parse_aggregated_model(aggregated_model, client_id)
                        )
                    self.local_models.clear()
                if is_final_chunk:
                    self._num_global_epochs += 1
            return future

    def get_num_global_epochs(self) -> int:
        """
        Get the number of global epochs.
        :return: the number of global epochs
        """
        with self._access_lock:
            return self._num_global_epochs

    def _parse_aggregated_model(
        self, aggregated_model: Dict, client_id: Union[int, str]
    ) -> Dict:
        """
        Parse the aggregated model. Currently, this method is used to
        parse different client gradients for the vertical federated learning.
        :param aggregated_model: the aggregated model
        :return: the parsed aggregated model
        """
        if isinstance(aggregated_model, tuple):
            if client_id in aggregated_model[0]:
                return (
                    aggregated_model[0][client_id],
                    aggregated_model[1],
                )  # this is for vertical federated learning
            else:
                return aggregated_model
        else:
            if client_id in aggregated_model:
                return aggregated_model[
                    client_id
                ]  # this is for vertical federated learning
            else:
                return aggregated_model
