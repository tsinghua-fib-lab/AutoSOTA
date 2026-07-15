from appfl.misc.deprecation import deprecated
from .server_federated import FedServer


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ServerFedAvg(FedServer):
    def compute_step(self):
        super().compute_pseudo_gradient()
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))
        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
