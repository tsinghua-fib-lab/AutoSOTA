import gc
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Optional, Any, Union, Dict, OrderedDict
from appfl.misc.memory_utils import (
    optimize_memory_cleanup,
    safe_inplace_operation,
    clone_state_dict_optimized,
)


class FedQueueAggregator(BaseAggregator):
    """
    FedQueue asynchronous federated learning algorithm.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.aggregator_configs = aggregator_configs
        self.logger = logger
        self.staleness_fn = self.__staleness_fn_factory(
            self.aggregator_configs.get("staleness_fn", "constant"),
            **self.aggregator_configs.get("staleness_fn_kwargs", {}),
        )
        self.global_state = None
        if model is not None:
            self.named_parameters = set()
            for name, _ in model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

    def aggregate(
        self,
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        staleness: Dict[Union[str, int], int],
        local_steps: Dict[Union[str, int], int],
    ) -> Dict:
        if self.global_state is None:
            first_model = list(local_models.values())[0]
            if self.model is not None:
                try:
                    with torch.no_grad():
                        self.global_state = {}
                        model_state = self.model.state_dict()
                        for name in first_model:
                            if name in model_state:
                                self.global_state[name] = (
                                    model_state[name].clone().detach()
                                )
                    gc.collect()
                except:  # noqa E722
                    with torch.no_grad():
                        self.global_state = {
                            name: tensor.detach().clone()
                            for name, tensor in first_model.items()
                        }
                    gc.collect()
            else:
                with torch.no_grad():
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in first_model.items()
                    }
            optimize_memory_cleanup(first_model, force_gc=False)

        gradient_based = self.aggregator_configs.get("gradient_based", False)

        aggregation_factors = self._get_aggregation_factors(local_models, staleness)
        aggregation_factor_sum = sum(aggregation_factors.values())
        aggregation_factors = {
            client_id: factor / aggregation_factor_sum
            for client_id, factor in aggregation_factors.items()
        }

        with torch.no_grad():
            if not gradient_based:
                global_state_cp = {}
                for name in self.global_state:
                    global_state_cp[name] = torch.zeros_like(self.global_state[name])

            num_clients = len(local_models)

            for i, client_id in enumerate(local_models):
                local_model = local_models[client_id]
                for name in self.global_state:
                    if (
                        self.named_parameters is not None
                        and name not in self.named_parameters
                    ) or (
                        self.global_state[name].dtype == torch.int64
                        or self.global_state[name].dtype == torch.int32
                    ):
                        if i == 0:
                            self.global_state[name] = torch.zeros_like(
                                local_model[name]
                            )
                        self.global_state[name] = safe_inplace_operation(
                            self.global_state[name], "add", local_model[name]
                        )
                        if i == num_clients - 1:
                            # Safe division with dtype preservation
                            self.global_state[name] = safe_inplace_operation(
                                self.global_state[name], "div", num_clients
                            )
                    else:
                        if gradient_based:
                            self.global_state[name] = safe_inplace_operation(
                                self.global_state[name],
                                "sub",
                                local_model[name] * aggregation_factors[client_id],
                            )
                        else:
                            global_state_cp[name] = safe_inplace_operation(
                                global_state_cp[name],
                                "add",
                                local_model[name] * aggregation_factors[client_id],
                            )
                            if i == num_clients - 1:
                                self.global_state[name] = global_state_cp[name]

                if not gradient_based:
                    optimize_memory_cleanup(global_state_cp, force_gc=True)
                else:
                    optimize_memory_cleanup(force_gc=True)

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        return clone_state_dict_optimized(self.global_state)

    def _get_aggregation_factors(
        self,
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        staleness: Dict[Union[str, int], int],
    ) -> Dict[Union[str, int], float]:
        """
        FedQueue uses data-size client weights and staleness decay. Local steps
        only affect client-side work and learning-rate scaling, not aggregation.
        """
        if hasattr(self, "client_sample_size") and all(
            client_id in self.client_sample_size for client_id in local_models
        ):
            total_sample_size = sum(
                self.client_sample_size[client_id] for client_id in local_models
            )
            if total_sample_size > 0:
                return {
                    client_id: self.staleness_fn(staleness[client_id])
                    * (self.client_sample_size[client_id] / total_sample_size)
                    for client_id in local_models
                }

        if self.logger is not None:
            self.logger.warning(
                "FedQueueAggregator did not receive sample sizes for all "
                "participating clients; falling back to equal client weights."
            )
        return {
            client_id: self.staleness_fn(staleness[client_id])
            * (1.0 / len(local_models))
            for client_id in local_models
        }

    def get_parameters(self, **kwargs) -> Dict:
        if self.global_state is None:
            if self.model is not None:
                return clone_state_dict_optimized(self.model.state_dict())
            else:
                raise ValueError("Model is not provided to the aggregator.")
        return clone_state_dict_optimized(self.global_state)

    def __staleness_fn_factory(
        self,
        staleness_fn_name,
        **kwargs,
    ):
        if staleness_fn_name == "constant":
            return lambda u: 1.0
        elif staleness_fn_name == "harmonic":
            beta = kwargs["beta"]
            return lambda u: 1.0 / (1.0 + beta * u)
        elif staleness_fn_name == "exponential":
            beta = kwargs["beta"]
            return lambda u: torch.exp(-beta * u)
        else:
            raise NotImplementedError
