import logging
import copy
import torch
from appfl.misc.deprecation import deprecated
from .server_federated import FedServer

logger = logging.getLogger(__name__)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ServerFedBuffer(FedServer):
    """
    ServerFedBuffer
        FedBuffer - Federated Learning with Buffered Asynchronous Aggregation: https://arxiv.org/abs/2106.06639
    Args:
        weights: weight for each client
        model: PyTorch model
        loss_fn: loss function
        num_clients: number of clients
        device: server device (TODO: do we really need this, server aggregation is on CPU by default)
    """

    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        self.counter = 0
        self.global_step = 0
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
        for name in self.model.state_dict():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
        self.list_named_parameters = []
        for name, _ in self.model.named_parameters():
            self.list_named_parameters.append(name)

    def update_gradient(self, local_gradient: dict, init_step: int, client_idx: int):
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name in self.model.state_dict():
            if name in self.list_named_parameters:
                self.pseudo_grad[name] += (
                    local_gradient[name] * self.weights[client_idx] * alpha_t
                )
            else:
                self.pseudo_grad[name] += local_gradient[name]
        self.counter += 1

    def update(self, local_gradient: dict, init_step: int, client_idx: int):
        self.update_gradient(local_gradient, init_step, client_idx)
        if self.counter == self.K:
            self.global_state = copy.deepcopy(self.model.state_dict())
            for name in self.model.state_dict():
                if name in self.list_named_parameters:
                    self.global_state[name] -= self.pseudo_grad[name]
                else:
                    self.global_state[name] = torch.div(
                        self.pseudo_grad[name], self.counter
                    )
            self.model.load_state_dict(self.global_state)
            self.global_step += 1
            self.counter = 0
            for name in self.model.state_dict():
                self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])

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
        logger.info("buffer_size=%s" % (cfg.fed.args.K))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedBuffer ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " FedBuffer Alpha "
                    + str(cfg.fed.args.alpha)
                    + " FedBuffer Staleness Function"
                    + str(cfg.fed.args.staleness_func.name)
                    + " FedBuffer Buffer Size"
                    + str(cfg.fed.args.K)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
