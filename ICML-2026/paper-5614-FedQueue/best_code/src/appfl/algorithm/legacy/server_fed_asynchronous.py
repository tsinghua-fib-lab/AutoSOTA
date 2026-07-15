import copy
import logging
from .server_federated import FedServer
from appfl.misc.deprecation import deprecated

logger = logging.getLogger(__name__)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ServerFedAsynchronous(FedServer):
    """
    ServerFedAsynchronous:
        FedAsync - Asynchronous Federated Optimization: http://arxiv.org/abs/1903.03934
    Args:
        weights: weight for each client
        model: PyTorch model
        loss_fn: loss function
        num_clients: number of clients
        device: server device (TODO: do we really need this, server aggregation is on CPU by default)
    """

    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        weights = (
            [1.0 / num_clients for _ in range(num_clients)]
            if weights is None
            else weights
        )
        super().__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        self.global_step = 0
        self.staleness = self.__staleness_func_factory(
            stalness_func_name=self.staleness_func["name"],
            **self.staleness_func["args"],
        )
        self.list_named_parameters = []
        for name, _ in self.model.named_parameters():
            self.list_named_parameters.append(name)

    def update(self, local_gradient: dict, init_step: int, client_idx: int):
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
        logger.info("gradient_based=%s" % (cfg.fed.args.gradient_based))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAsync ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " FedAsync Alpha "
                    + str(cfg.fed.args.alpha)
                    + " FedAsync Staleness Function"
                    + str(cfg.fed.args.staleness_func.name)
                    + " FedAsync Gradient-based"
                    + str(cfg.fed.args.gradient_based)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
