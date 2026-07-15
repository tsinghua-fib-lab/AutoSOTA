import abc
import copy
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader


class BaseServer:
    """
    BaseServer:
        Abstract base FL server class.

    Args:
        weights (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients
        device (str): device for computation
    """

    def __init__(
        self,
        weights: OrderedDict,
        model: nn.Module,
        loss_fn: nn.Module,
        num_clients: int,
        device: str,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.device = device
        self.weights = copy.deepcopy(weights)
        self.global_state = OrderedDict()
        self.primal_states = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()

    def get_model(self) -> nn.Module:
        return copy.deepcopy(self.model)

    def set_weights(self, weights: OrderedDict):
        for key, value in weights.items():
            self.weights[key] = value

    def log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s %10s" % (
            "Iter",
            "Local(s)",
            "Global(s)",
            "Valid(s)",
            "PerIter(s)",
            "Elapsed(s)",
            "TestLoss",
            "TestAccu",
        )
        return title

    def log_contents(self, cfg, t):
        contents = "%10d %10.2f %10.2f %10.2f %10.2f %10.2f %10.4f %10.2f" % (
            t + 1,
            cfg["logginginfo"]["LocalUpdate_time"],
            cfg["logginginfo"]["GlobalUpdate_time"],
            cfg["logginginfo"]["Validation_time"],
            cfg["logginginfo"]["PerIter_time"],
            cfg["logginginfo"]["Elapsed_time"],
            cfg["logginginfo"]["test_loss"],
            cfg["logginginfo"]["test_accuracy"],
        )
        return contents

    def log_summary(self, cfg: DictConfig, logger):
        logger.info("Device=%s" % (cfg.device))
        logger.info("#Processors=%s" % (cfg["logginginfo"]["comm_size"]))
        logger.info("#Clients=%s" % (self.num_clients))
        logger.info("Server=%s" % (cfg.fed.servername))
        logger.info("Clients=%s" % (cfg.fed.clientname))
        logger.info("Comm_Rounds=%s" % (cfg.num_epochs))
        logger.info("Local_Rounds=%s" % (cfg.fed.args.num_local_epochs))
        logger.info(
            "DP_Eps=%s" % (cfg.fed.args.epsilon if cfg.fed.args.use_dp else False)
        )
        logger.info(
            "Clipping=%s"
            % (
                cfg.fed.args.clip_value
                if (cfg.fed.args.clip_grad or cfg.fed.args.use_dp)
                else False
            )
        )
        logger.info("Elapsed_time=%s" % (round(cfg["logginginfo"]["Elapsed_time"], 2)))
        logger.info("BestAccuracy=%s" % (round(cfg["logginginfo"]["BestAccuracy"], 2)))
        logger.info("Data_Readiness_Metric=%s" % (cfg.dr_metrics))


class BaseClient:
    """
    BaseClient:
        Abstract base FL client class.

    Args:
        id: unique ID for each client
        weight: aggregation weight assigned to each client
        model: torch neural network model to train
        loss_fn: loss function
        dataloader: training data loader
        cfg: configurations
        outfile: logging file
        test_dataloader: test dataloader
        metric: performance evaluation metric
    """

    def __init__(
        self,
        id: int,
        weight: Dict,
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: DataLoader,
        cfg: DictConfig,
        outfile: str,
        test_dataloader: Optional[DataLoader] = None,
        metric: Any = None,
    ):
        self.round = 0
        self.id = id
        self.weight = weight
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.cfg = cfg
        self.test_dataloader = test_dataloader
        self.metric = metric if metric is not None else self._default_metric
        self.primal_state = OrderedDict()
        self.outfile = outfile
        if type(outfile) is str:
            self.outfile = open(outfile, "a")

    @abc.abstractmethod
    def update(self):
        """Update local model parameters"""
        pass

    def get_model(self):
        return self.model.state_dict()

    def client_log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s \n" % (
            "Round",
            "LocalEpoch",
            "PerIter[s]",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        )
        self.outfile.write(title)
        self.outfile.flush()

    def client_log_content(
        self, t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
    ):
        contents = "%10s %10s %10.2f %10.4f %10.4f %10.4f %10.4f \n" % (
            self.round,
            t,
            per_iter_time,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
        )
        self.outfile.write(contents)
        self.outfile.flush()

    def client_validation(self):
        if self.loss_fn is None or self.test_dataloader is None:
            return 0.0, 0.0
        if self.metric is None:
            self.metric = self._default_metric

        device = self.cfg.device
        self.model.to(device)
        self.model.eval()

        loss, tmpcnt = 0, 0
        with torch.no_grad():
            target_pred_final = []
            target_true_final = []
            for data, target in self.test_dataloader:
                tmpcnt += 1
                data = data.to(device)
                target = target.to(device)
                output = self.model(data)
                loss += self.loss_fn(output, target).item()
                target_pred_final.append(output.detach().cpu().numpy())
                target_true_final.append(target.detach().cpu().numpy())
            loss = loss / tmpcnt
            target_true_final = np.concatenate(target_true_final)
            target_pred_final = np.concatenate(target_pred_final)
            accuracy = float(self.metric(target_true_final, target_pred_final))
        self.model.train()
        return loss, accuracy

    def _default_metric(self, y_true, y_pred):
        if len(y_pred.shape) == 1:
            y_pred = np.round(y_pred)
        else:
            y_pred = y_pred.argmax(axis=1)
        return 100 * np.sum(y_pred == y_true) / y_pred.shape[0]

    def laplace_mechanism_output_perturb(self, scale_value):
        """
        laplace_mechanism_output_perturb:
            Differential privacy for output perturbation based on Laplacian distribution.This output perturbation adds Laplace noise to ``primal_state``. Variance is equal to `2*(scale_value)^2`, and `scale_value = sensitivity/epsilon`, where `sensitivity` is determined by data, algorithm.

        Args:
            scale_value: scaling vector to control the variance of Laplacian distribution
        """
        for name, param in self.model.named_parameters():
            mean = torch.zeros_like(param.data)
            scale = torch.zeros_like(param.data) + scale_value
            m = torch.distributions.laplace.Laplace(mean, scale)
            self.primal_state[name] += m.sample()

    def laplace_mechanism_output_perturb_personalized(self, scale_value):
        """
        laplace_mechanism_output_perturb:
            Differential privacy for output perturbation based on Laplacian distribution. This output perturbation adds Laplace noise to ``primal_state``. Variance is equal to `2*(scale_value)^2`, and `scale_value = sensitivity/epsilon`, where `sensitivity` is determined by data, algorithm.

        Args:
            scale_value: scaling vector to control the variance of Laplacian distribution
        """
        for name, param in self.model.named_parameters():
            if name not in self.cfg.p_layers:
                mean = torch.zeros_like(param.data)
                scale = torch.zeros_like(param.data) + scale_value
                m = torch.distributions.laplace.Laplace(mean, scale)
                self.primal_state[name] += m.sample()
