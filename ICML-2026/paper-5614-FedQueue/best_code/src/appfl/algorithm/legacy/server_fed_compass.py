import logging
import copy
import torch
from collections import OrderedDict
from .server_federated import FedServer
from appfl.misc.deprecation import deprecated

logger = logging.getLogger(__name__)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ServerFedCompass(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        self.counter = 0
        self.global_step = 0
        # weights = [1.0 / num_clients for _ in range(num_clients)]
        weights = (
            [1.0 / num_clients for _ in range(num_clients)]
            if weights is None
            else weights
        )
        super().__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        self.staleness = self.__staleness_func_factory(
            stalness_func_name=self.staleness_func["name"],
            **self.staleness_func["args"],
        )
        self.group_pseudo_grad = OrderedDict()
        self.general_buffer = OrderedDict()
        for name in self.model.state_dict():
            self.general_buffer[name] = torch.zeros_like(self.model.state_dict()[name])
        self.general_buffer_size = 0
        self.list_named_parameters = []
        for name, _ in self.model.named_parameters():
            self.list_named_parameters.append(name)

    def update(self, local_gradient: dict, init_step: int, client_idx: int):
        """Update the model directly using the client local gradient with staleness factor applied."""
        self.global_state = copy.deepcopy(self.model.state_dict())
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                self.global_state[name] -= (
                    local_gradient[name] * self.weights[client_idx] * alpha_t
                )
            else:
                self.global_state[name] = local_gradient[name]
        self.model.load_state_dict(self.global_state)
        self.global_step += 1

    def buffer(
        self, local_gradient: dict, init_step: int, client_idx: int, group_idx: int
    ):
        """Buffer the local gradient from the client of a certain group."""
        if group_idx not in self.group_pseudo_grad:
            self.group_pseudo_grad[group_idx] = OrderedDict()
            for name in self.model.state_dict():
                self.group_pseudo_grad[group_idx][name] = torch.zeros_like(
                    self.model.state_dict()[name]
                )
            self.group_pseudo_grad[group_idx]["_counter"] = 0
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                self.group_pseudo_grad[group_idx][name] += (
                    local_gradient[name] * self.weights[client_idx] * alpha_t
                )
            else:
                self.group_pseudo_grad[group_idx][name] += local_gradient[name]
        self.group_pseudo_grad[group_idx]["_counter"] += 1

    def single_buffer(self, local_gradient: dict, init_step: int, client_idx: int):
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                self.general_buffer[name] += (
                    local_gradient[name] * self.weights[client_idx] * alpha_t
                )
            else:
                self.general_buffer[name] += local_gradient[name]
        self.general_buffer_size += 1

    def update_group(self, group_idx: int):
        """Update the model using all the buffered gradients for a certain group."""
        if group_idx in self.group_pseudo_grad:
            self.global_state = copy.deepcopy(self.model.state_dict())
            for name in self.model.state_dict():
                if name in self.list_named_parameters:
                    self.global_state[name] -= (
                        self.group_pseudo_grad[group_idx][name]
                        + self.general_buffer[name]
                    )
                    self.general_buffer[name] = torch.zeros_like(
                        self.model.state_dict()[name]
                    )
                else:
                    self.global_state[name] = torch.div(
                        self.group_pseudo_grad[group_idx][name]
                        + self.general_buffer[name],
                        self.group_pseudo_grad[group_idx]["_counter"]
                        + self.general_buffer_size,
                    )
                    self.general_buffer[name] = torch.zeros_like(
                        self.model.state_dict()[name]
                    )
            self.model.load_state_dict(self.global_state)
            self.global_step += 1
            self.general_buffer_size = 0
            del self.group_pseudo_grad[group_idx]

    def update_all(self):
        """Update the model using all the buffered gradients for a group."""
        self.global_state = copy.deepcopy(self.model.state_dict())
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                for group_idx in self.group_pseudo_grad:
                    self.global_state[name] -= self.group_pseudo_grad[group_idx][name]
            # We may just skip this
            else:
                tmpsum = torch.zeros_like(self.model.state_dict()[name])
                counter = 0
                for group_idx in self.group_pseudo_grad:
                    tmpsum += self.group_pseudo_grad[group_idx][name]
                    counter += self.group_pseudo_grad[group_idx]["_counter"]
                if counter > 0:
                    self.global_state[name] = torch.div(tmpsum, counter)
        self.group_pseudo_grad = OrderedDict()
        self.model.load_state_dict(self.global_state)
        self.global_step += 1

    def __staleness_func_factory(self, stalness_func_name, **kwargs):
        if stalness_func_name == "constant":
            return lambda u: 1
        elif stalness_func_name == "polynomial":
            a = kwargs["a"]
            return lambda u: (u + 1) ** a
        elif stalness_func_name == "hinge":
            a = kwargs["a"]
            b = kwargs["b"]
            return lambda u: 1 if u <= b else 1.0 / (a * (u - b) + 1.0)
        else:
            raise NotImplementedError

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate=%s " % (cfg.fed.args.optim_args.lr))
        logger.info("model_mixing_parameter=%s " % (cfg.fed.args.alpha))
        logger.info("staleness_func=%s" % (cfg.fed.args.staleness_func.name))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " ServerFedCompass ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " ServerFedCompass Alpha "
                    + str(cfg.fed.args.alpha)
                    + " ServerFedCompass Staleness Function"
                    + str(cfg.fed.args.staleness_func.name)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
