import copy
import time
import warnings
from typing import Optional, Any
from omegaconf import DictConfig
from appfl.algorithm.trainer.base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
from monai.fl.client.monai_algo import MonaiAlgo  # noqa: E402
from monai.fl.utils.exchange_object import ExchangeObject  # noqa: E402
from monai.fl.utils.constants import ExtraItems, WeightType  # noqa: E402


class MonaiTrainer(BaseTrainer):
    def __init__(
        self,
        client_id: str,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.round = 0
        self.logger = logger
        self.train_configs = train_configs

        assert hasattr(train_configs, "bundle_root"), (
            "bundle_root not found in train_configs"
        )
        assert not train_configs.get("send_gradient", False), (
            "send_gradient=True is not supported in the beta version of MonaiTrainer. It will be supported soon."
        )

        self.monai_algo = MonaiAlgo(
            bundle_root=train_configs.bundle_root,
            local_epochs=train_configs.get("num_local_epochs", 1),
            send_weight_diff=train_configs.get("send_gradient", False),
        )
        self.monai_algo.logger.setLevel("WARNING")  # suppress logging
        self.monai_algo.initialize(
            extra={
                ExtraItems.CLIENT_NAME: client_id,
            }
        )

        # load initial model parameters
        self.monai_algo.send_weight_diff = False
        init_model = self.monai_algo.get_weights()
        self.load_parameters(init_model.weights)
        self.monai_algo.send_weight_diff = train_configs.get("send_gradient", False)

        # set initial model parameters
        self.model_state = copy.deepcopy(init_model.weights)

    def get_parameters(self):
        return (
            (self.model_state, self.metrics)
            if hasattr(self, "metrics")
            else self.model_state
        )

    def load_parameters(self, params):
        self._loaded_model = ExchangeObject(
            weights=params, weight_type=WeightType.WEIGHTS
        )

    def train(self, **kwargs):
        self.metrics = {"round": self.round + 1}
        do_validation = self.train_configs.get("do_validation", False)
        do_pre_validation = self.train_configs.get("do_pre_validation", False)
        title = (
            ["Round", "Time"]
            if (not do_validation) and (not do_pre_validation)
            else (
                ["Round", "Pre Val?", "Time", "Metrics"]
                if do_pre_validation
                else ["Round", "Time", "Metrics"]
            )
        )
        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        # Validation pre training
        if do_pre_validation:
            metric = self.monai_algo.evaluate(self._loaded_model).metrics
            for k, v in metric.items():
                self.metrics[k + "_before_train"] = v
            content = [self.round, "Y", "N/A", metric]
            self.logger.log_content(content)

        # Start training
        start_time = time.time()
        self.monai_algo.train(self._loaded_model)
        end_time = time.time()

        # Post training validation
        if do_validation:
            self.monai_algo.send_weight_diff = False
            new_model = self.monai_algo.get_weights()
            self.monai_algo.send_weight_diff = self.train_configs.get(
                "send_gradient", False
            )
            metric = self.monai_algo.evaluate(new_model).metrics
            for k, v in metric.items():
                self.metrics[k] = v
            content = (
                [self.round, "N", end_time - start_time, metric]
                if do_pre_validation
                else [self.round, end_time - start_time, metric]
            )
        else:
            content = (
                [self.round, end_time - start_time]
                if not do_pre_validation
                else [self.round, "N", end_time - start_time, "N/A"]
            )

        self.logger.log_content(content)

        # Update model state
        model = self.monai_algo.get_weights()
        self.model_state = copy.deepcopy(model.weights)

        self.round += 1
