import abc
import copy
import torch
import logging
from .fl_base import BaseServer
from collections import OrderedDict


class FedServer(BaseServer):
    """
    FedServer:
        Abstract server class of general FL algorithms that aggregates and updates model parameters.
    Args:
        weights (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients
        device (str): device for computation
    """

    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super().__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)
        self.step = OrderedDict()
        self.list_named_parameters = []
        self.pseudo_grad = OrderedDict()
        for name, _ in self.model.named_parameters():
            self.list_named_parameters.append(name)
        if hasattr(self, "server_momentum_param_1"):
            self.m_vector = OrderedDict()
            for name, _ in self.model.named_parameters():
                self.list_named_parameters.append(name)
                self.m_vector[name] = torch.zeros_like(
                    self.model.state_dict()[name], device=device
                )
        if hasattr(self, "server_adapt_param"):
            self.v_vector = OrderedDict()
            for name, _ in self.model.named_parameters():
                self.v_vector[name] = (
                    torch.zeros_like(self.model.state_dict()[name], device=device)
                    + self.server_adapt_param**2
                )

    def update_m_vector(self):
        """Update the `m_vector` in adaptive federated optimization."""
        for name, _ in self.model.named_parameters():
            self.m_vector[name] = (
                self.server_momentum_param_1 * self.m_vector[name]
                - (1.0 - self.server_momentum_param_1) * self.pseudo_grad[name]
            )

    def compute_pseudo_gradient(self):
        """Compute the gradient from the client local updates, where gradient is the difference between the old model and the new model."""
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
            for i in range(self.num_clients):
                self.pseudo_grad[name] += self.weights[i] * (
                    self.global_state[name] - self.primal_states[i][name]
                )

    @abc.abstractmethod
    def compute_step(self):
        """Compute the step for global model gradient descend."""
        pass

    def update(self, local_states: list):
        self.global_state = copy.deepcopy(self.model.state_dict())
        ## Load the client local states
        for sid, states in enumerate(local_states):
            if states is not None:
                self.primal_states[sid] = states
        for i in range(self.num_clients):
            for name in self.model.state_dict():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )

        ## Update global state
        self.compute_step()
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                self.global_state[name] += self.step[name]
            else:
                tmpsum = torch.zeros_like(self.global_state[name], device=self.device)
                for i in range(self.num_clients):
                    tmpsum += self.primal_states[i][name]
                self.global_state[name] = torch.div(tmpsum, self.num_clients)

        self.model.load_state_dict(self.global_state)

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super().log_title()
            logger.info(title)
        contents = super().log_contents(cfg, t)
        logger.info(contents)
