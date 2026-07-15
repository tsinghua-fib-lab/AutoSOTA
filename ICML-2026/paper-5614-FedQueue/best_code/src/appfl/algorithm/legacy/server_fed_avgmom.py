from appfl.misc.deprecation import deprecated
from .server_federated import FedServer


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ServerFedAvgMomentum(FedServer):
    def update_m_vector(self):
        for name, _ in self.model.named_parameters():
            self.m_vector[name] = (
                self.server_momentum_param_1 * self.m_vector[name]
                + self.pseudo_grad[name]
            )

    def compute_step(self):
        super().compute_pseudo_gradient()
        self.update_m_vector()
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.m_vector[name]

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))
        logger.info(
            "server_momentum_param_1 = %s " % (cfg.fed.args.server_momentum_param_1)
        )

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvgM ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " MParam1 "
                    + str(cfg.fed.args.server_momentum_param_1)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
