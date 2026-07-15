import gc
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import FedAsyncAggregator
from typing import Union, Dict, OrderedDict, Any, Optional
from appfl.misc.memory_utils import clone_state_dict_optimized, optimize_memory_cleanup


class FedBuffAggregator(FedAsyncAggregator):
    """
    FedBuff Aggregator class for Federated Learning.
    For more details, check paper: https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(model, aggregator_configs, logger)
        self.buff_size = 0
        self.K = self.aggregator_configs.K

    def aggregate(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Dict:
        # Memory optimization: Efficient global state initialization
        if self.global_state is None:
            if self.model is not None:
                try:
                    if self.optimize_memory:
                        with torch.no_grad():
                            self.global_state = {
                                name: self.model.state_dict()[name]
                                for name in local_model
                            }
                        gc.collect()
                    else:
                        self.global_state = {
                            name: self.model.state_dict()[name] for name in local_model
                        }
                except:  # noqa E722
                    if self.optimize_memory:
                        with torch.no_grad():
                            self.global_state = {
                                name: tensor.detach().clone()
                                for name, tensor in local_model.items()
                            }
                        gc.collect()
                    else:
                        self.global_state = {
                            name: tensor.detach().clone()
                            for name, tensor in local_model.items()
                        }
            else:
                if self.optimize_memory:
                    with torch.no_grad():
                        self.global_state = {
                            name: tensor.detach().clone()
                            for name, tensor in local_model.items()
                        }
                    gc.collect()
                else:
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in local_model.items()
                    }

        self.compute_steps(client_id, local_model)
        self.buff_size += 1
        if self.buff_size == self.K:
            for name in self.global_state:
                if (
                    self.named_parameters is not None
                    and name not in self.named_parameters
                ) or (
                    self.global_state[name].dtype == torch.int64
                    or self.global_state[name].dtype == torch.int32
                ):
                    self.global_state[name] = torch.div(self.step[name], self.K).type(
                        self.global_state[name].dtype
                    )
                else:
                    self.global_state[name] = self.global_state[name] + self.step[name]

            self.global_step += 1
            self.buff_size = 0

            if self.model is not None:
                self.model.load_state_dict(self.global_state, strict=False)

            # Memory optimization: Clean up after buffer flush
            if self.optimize_memory:
                optimize_memory_cleanup(force_gc=True)

        self.client_step[client_id] = self.global_step
        # Memory optimization: Use efficient state dict cloning for return
        if self.optimize_memory:
            return clone_state_dict_optimized(self.global_state)
        else:
            return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(
        self,
        client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
    ):
        """
        Compute changes to the global model after the aggregation.
        """
        # Memory optimization: Efficient step initialization at buffer start
        if self.buff_size == 0:
            if self.optimize_memory:
                with torch.no_grad():
                    for name in self.global_state:
                        self.step[name] = torch.zeros_like(self.global_state[name])
                gc.collect()
            else:
                for name in self.global_state:
                    self.step[name] = torch.zeros_like(self.global_state[name])

        if client_id not in self.client_step:
            self.client_step[client_id] = 0
        gradient_based = self.aggregator_configs.get("gradient_based", False)
        if (
            self.client_weights_mode == "sample_size"
            and hasattr(self, "client_sample_size")
            and client_id in self.client_sample_size
        ):
            weight = self.client_sample_size[client_id] / sum(
                self.client_sample_size.values()
            )
        else:
            weight = 1.0 / self.aggregator_configs.get("num_clients", 1)
        alpha_t = (
            self.alpha
            * self.staleness_fn(self.global_step - self.client_step[client_id])
            * weight
        )

        for name in self.global_state:
            if self.named_parameters is not None and name not in self.named_parameters:
                self.step[name] += local_model[name]
            elif (
                self.global_state[name].dtype == torch.int64
                or self.global_state[name].dtype == torch.int32
            ):
                self.step[name] += local_model[name]
            else:
                self.step[name] += (
                    alpha_t * (-local_model[name])
                    if gradient_based
                    else alpha_t * (local_model[name] - self.global_state[name])
                )
