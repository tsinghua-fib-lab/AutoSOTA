import gc
import copy
import torch
import torch.nn as nn
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Any, Dict, Union
from appfl.algorithm.aggregator import BaseAggregator
from appfl.misc.memory_utils import clone_state_dict_optimized, optimize_memory_cleanup


class ICEADMMAggregator(BaseAggregator):
    """
    ICEADMM Aggregator class for Federated Learning.
    It has to be used with the ICEADMMTrainer.
    For more details, check paper: https://arxiv.org/pdf/2110.15318.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        aggregator_configs: DictConfig,
        logger: Any,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

        # Check for optimize_memory in aggregator_configs, default to True
        self.optimize_memory = getattr(aggregator_configs, "optimize_memory", True)

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)
        self.is_first_iter = True
        self.penalty = OrderedDict()
        self.prim_res = 0
        self.dual_res = 0
        self.global_state = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        self.device = (
            self.aggregator_configs.device
            if "device" in self.aggregator_configs
            else "cpu"
        )

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        if len(self.primal_states) == 0:
            self.num_clients = len(local_models)
            for i in local_models:
                self.primal_states[i] = OrderedDict()
                self.dual_states[i] = OrderedDict()
                self.primal_states_curr[i] = OrderedDict()
                self.primal_states_prev[i] = OrderedDict()

        if self.optimize_memory:
            global_state = clone_state_dict_optimized(self.model.state_dict())
        else:
            global_state = copy.deepcopy(self.model.state_dict())

        for client_id, model in local_models.items():
            if model is not None:
                self.primal_states[client_id] = model["primal"]
                self.dual_states[client_id] = model["dual"]
                self.penalty[client_id] = model["penalty"]

        # Calculate the primal residual
        primal_res = 0
        for client_id in local_models:
            for name in self.named_parameters:
                primal_res += torch.sum(
                    torch.square(
                        global_state[name].to(self.device)
                        - self.primal_states[client_id][name].to(self.device)
                    )
                )
        self.prim_res = torch.sqrt(primal_res).item()

        # Calculate the dual residual
        dual_res = 0
        if self.is_first_iter:
            if self.optimize_memory:
                with torch.no_grad():
                    for client_id in local_models:
                        for name in self.named_parameters:
                            self.primal_states_curr[client_id][name] = (
                                self.primal_states[client_id][name]
                                .to(self.device)
                                .clone()
                                .detach()
                            )
                    gc.collect()
            else:
                for client_id in local_models:
                    for name in self.named_parameters:
                        self.primal_states_curr[client_id][name] = copy.deepcopy(
                            self.primal_states[client_id][name].to(self.device)
                        )
            self.is_first_iter = False
        else:
            if self.optimize_memory:
                with torch.no_grad():
                    # Efficient copying of states
                    temp_prev = OrderedDict()
                    for client_id in self.primal_states_curr:
                        temp_prev[client_id] = OrderedDict()
                        for name in self.primal_states_curr[client_id]:
                            temp_prev[client_id][name] = (
                                self.primal_states_curr[client_id][name]
                                .clone()
                                .detach()
                            )
                    self.primal_states_prev = temp_prev

                    for client_id in local_models:
                        for name in self.named_parameters:
                            self.primal_states_curr[client_id][name] = (
                                self.primal_states[client_id][name]
                                .to(self.device)
                                .clone()
                                .detach()
                            )

                    optimize_memory_cleanup(temp_prev, force_gc=True)
            else:
                self.primal_states_prev = copy.deepcopy(self.primal_states_curr)
                for client_id in local_models:
                    for name in self.named_parameters:
                        self.primal_states_curr[client_id][name] = copy.deepcopy(
                            self.primal_states[client_id][name].to(self.device)
                        )
            for name in self.named_parameters:
                res = 0
                for client_id in local_models:
                    res += self.penalty[client_id] * (
                        self.primal_states_prev[client_id][name]
                        - self.primal_states_curr[client_id][name]
                    )
                dual_res += torch.sum(torch.square(res))
            self.dual_res = torch.sqrt(dual_res).item()

        total_penalty = 0
        for client_id in local_models:
            total_penalty += self.penalty[client_id]

        for name, param in self.model.named_parameters():
            state_param = torch.zeros_like(param)
            for client_id in local_models:
                self.primal_states[client_id][name] = self.primal_states[client_id][
                    name
                ].to(self.device)
                self.dual_states[client_id][name] = self.dual_states[client_id][
                    name
                ].to(self.device)

                state_param += (
                    self.penalty[client_id] / total_penalty
                ) * self.primal_states[client_id][name] + (
                    1.0 / total_penalty
                ) * self.dual_states[client_id][name]
            global_state[name] = state_param

        self.model.load_state_dict(global_state)
        return global_state

    def get_parameters(self, **kwargs) -> Dict:
        if self.optimize_memory:
            return clone_state_dict_optimized(self.model.state_dict())
        else:
            return copy.deepcopy(self.model.state_dict())
