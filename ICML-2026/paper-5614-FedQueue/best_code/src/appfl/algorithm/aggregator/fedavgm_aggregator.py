import gc
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import FedAvgAggregator
from typing import Union, Dict, OrderedDict, Any, Optional
from appfl.misc.memory_utils import safe_inplace_operation, optimize_memory_cleanup


class FedAvgMAggregator(FedAvgAggregator):
    """
    FedAvgM Aggregator class for Federated Learning.
    For more details, check paper `Measuring the effects of non-identical data distribution for federated visual classification`
    at https://arxiv.org/pdf/1909.06335.pdf

    Required aggregator_configs fields:
        - server_momentum_param_1: `beta` in the paper
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(model, aggregator_configs, logger)
        self.v_vector = {}

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(local_models)

        # Memory optimization: Initialize vectors efficiently
        if len(self.v_vector) == 0:
            if self.optimize_memory:
                with torch.no_grad():
                    for name in self.step:
                        self.v_vector[name] = torch.zeros_like(self.step[name])
                    gc.collect()
            else:
                for name in self.step:
                    self.v_vector[name] = torch.zeros_like(self.step[name])

        # Memory optimization: Use safe in-place operations
        if self.optimize_memory:
            with torch.no_grad():
                for name in self.step:
                    # Momentum update with safe operations
                    momentum_term = (
                        self.v_vector[name]
                        * self.aggregator_configs.server_momentum_param_1
                    )
                    self.v_vector[name] = safe_inplace_operation(
                        momentum_term, "add", self.step[name]
                    )

                    # Use the momentum vector as the step
                    self.step[name] = self.v_vector[name].clone()

                    # Cleanup intermediate tensors
                    optimize_memory_cleanup(momentum_term, force_gc=False)

                optimize_memory_cleanup(force_gc=True)
        else:
            # Original behavior
            for name in self.step:
                self.v_vector[name] = (
                    self.aggregator_configs.server_momentum_param_1
                    * self.v_vector[name]
                    + self.step[name]
                )
                self.step[name] = self.v_vector[name]
