"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import abc
import time
import uuid
import traceback
import torch.nn as nn
from typing import Any
from .algorithm import SchedulerCompassGlobusCompute
from omegaconf import DictConfig
from collections import OrderedDict
from globus_compute_sdk import Client
from .comm.globus_compute import GlobusComputeCommunicator
from .comm.utils.utils import get_dataloader
from .comm.utils.s3_storage import LargeObjectWrapper
from .comm.globus_compute.utils.logging import GlobusComputeServerLogger
from .comm.globus_compute import (
    client_validate_data,
    client_testing,
    client_training,
    client_model_saving,
)
from appfl.misc.data import Dataset
from appfl.misc.utils import validation, get_appfl_algorithm


class APPFLGlobusComputeServer(abc.ABC):
    def __init__(self, cfg: DictConfig, gcc: Client):
        self.cfg = cfg
        self.gcc = gcc
        self.cfg.num_clients = len(self.cfg.clients)

        # Logger for a server
        self.logger = GlobusComputeServerLogger.get_logger()
        self.eval_logger = GlobusComputeServerLogger.get_eval_logger()

        # Globus Compute communicator
        self.communicator = GlobusComputeCommunicator(self.cfg, gcc, self.logger)

        # Using tensorboard to visualize the test loss
        if cfg.use_tensorboard:
            self.writer = GlobusComputeServerLogger.get_tensorboard_writer()

        # Runtime variables
        self.best_accuracy = 0.0
        self.data_info_at_client = None

    def _initialize_training(
        self, model: nn.Module, loss_fn: nn.Module, val_metric: Any
    ):
        """Set up model, loss function, and validation metric."""
        self.model = model
        self.loss_fn = loss_fn
        self.val_metric = val_metric

    def _validate_clients_data(self):
        """Validate the dataloader provided by the clients."""
        mode = ["train", "val", "test"]
        self.communicator.send_task_to_all_clients(client_validate_data, mode)
        data_info_at_client, _ = self.communicator.receive_sync_endpoints_updates()
        assert len(data_info_at_client) > 0, (
            "Number of clients need to be larger than 0"
        )
        GlobusComputeServerLogger.log_client_data_info(self.cfg, data_info_at_client)
        self.data_info_at_client = data_info_at_client

    def _set_client_weights(self):
        """Set the aggregation weights for clients."""
        assert self.data_info_at_client is not None, (
            "Please call the validate clients' data first"
        )
        mode = self.cfg.fed.args.client_weights
        if mode == "sample_size":
            total_num_data = 0
            for k in range(self.cfg.num_clients):
                total_num_data += self.data_info_at_client[k]["train"]
            weights = {}
            for k in range(self.cfg.num_clients):
                weights[k] = self.data_info_at_client[k]["train"] / total_num_data
        elif mode == "equal":
            weights = {k: 1 / self.cfg.num_clients for k in range(self.cfg.num_clients)}
        else:
            raise NotImplementedError
        self.weights = weights

    def _initialize_fl_server(self):
        """Initialize federated learning server."""
        self.server = get_appfl_algorithm(
            algorithm_name=self.cfg.fed.servername,
            args=(
                self.client_weights,
                self.model,
                self.loss_fn,
                self.num_clients,
                "cpu",
            ),
            kwargs=self.cfg.fed.args,
        )
        # Server model should stay on CPU for serialization
        self.server.model.to("cpu")

    @abc.abstractmethod
    def _do_training(self):
        pass

    def _do_client_testing(self):
        """Perform tesing at clients."""
        if self.cfg.client_do_testing:
            global_state = self.server.model.state_dict()
            server_model_basename = str(uuid.uuid4()) + "_server_state"
            self.communicator.send_task_to_all_clients(
                client_testing,
                self.weights,
                LargeObjectWrapper(global_state, server_model_basename),
            )
            testing_results, _ = self.communicator.receive_sync_endpoints_updates()
            testing_results = self.__parse_client_testing_results(testing_results)
            self.eval_logger.log_client_testing(testing_results)

    def __parse_client_testing_results(self, testing_results):
        ret = OrderedDict()
        for res in testing_results:
            client_idx = res["client_idx"]
            ret[client_idx] = {k: v for k, v in res.items() if k != "client_idx"}
        return ret

    def _do_server_testing(self):
        """Perform testing at server."""
        if self.cfg.server_do_testing:
            test_loss, test_accuracy = validation(
                self.server, self.server_testing_dataloader, self.val_metric
            )
            self.eval_logger.log_server_testing(
                {"acc": test_accuracy, "loss": test_loss}
            )

    def _do_server_validation(self, step: int):
        val_loss = 0.0
        val_accuracy = 0.0
        if self.cfg.server_do_validation:
            # Move server model to GPU (if available) for validation inference
            val_loss, val_accuracy = validation(
                self.server, self.server_validation_dataloader, self.val_metric
            )
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                self.writer.add_scalar("server_test_accuracy", val_accuracy, step)
                self.writer.add_scalar("server_test_loss", val_loss, step)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
            self.eval_logger.log_server_validation(
                {"val_loss": val_loss, "val_acc": val_accuracy}, step
            )

    def _parse_client_logs(self, step: int, client_logs):
        """Parse validation results at clients from client logs"""
        if self.cfg.client_do_validation:
            validation_results = self.__get_eval_results_from_logs(client_logs)
            self.eval_logger.log_client_validation(validation_results, step)
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                for client_idx in validation_results:
                    client_name = self.cfg.clients[client_idx].name
                    for val_k in validation_results[client_idx]:
                        self.writer.add_scalar(
                            f"{client_name}-{val_k}",
                            validation_results[client_idx][val_k],
                            step,
                        )

    def __get_eval_results_from_logs(self, logs):
        val_results = {}
        for client_idx in logs:
            val_results[client_idx] = {**logs[client_idx]["info"]["Validation"]}
        return val_results

    def _save_checkpoint(self, step):
        """Saving model"""
        if step % self.cfg.checkpoints_interval == 0 or step == self.cfg.num_epochs:
            if self.cfg.save_model:
                GlobusComputeServerLogger.save_checkpoint(
                    step, self.server.model.state_dict()
                )

    def _lr_step(self, step):
        """Perform learning rate decay."""
        if step != 0:
            self.communicator.decay_learning_rate()

    def _send_final_model(self):
        """Send the final model to the clients for saving."""
        global_state = self.server.model.state_dict()
        server_model_basename = f"final_model_{str(uuid.uuid4())}"
        self.communicator.send_task_to_all_clients(
            client_model_saving, LargeObjectWrapper(global_state, server_model_basename)
        )

    def set_server_dataset(self, validation_dataset=None, testing_dataset=None):
        """Set validation and testing dataset at the server side if given."""
        val_loader, test_loader = None, None
        val_size, test_size = 0, 0
        if self.cfg.server_do_validation:
            val_loader = get_dataloader(self.cfg, validation_dataset, mode="val")
            val_size = len(validation_dataset) if val_loader is not None else 0
        if self.cfg.server_do_testing:
            test_loader = get_dataloader(self.cfg, testing_dataset, mode="test")
            test_size = len(testing_dataset) if test_loader is not None else 0
        if val_loader is None:
            self.cfg.server_do_validation = False
            self.logger.info("Validation dataset at server is empty")
        if test_loader is None:
            self.cfg.server_do_testing = False
            self.logger.info("Testing dataset at server is empty")

        GlobusComputeServerLogger.log_server_data_info(
            {"val": val_size, "test": test_size}
        )
        self.server_testing_dataloader = test_loader
        self.server_validation_dataloader = val_loader

    def run(self, model: nn.Module, loss_fn: nn.Module, val_metric: Any):
        self._initialize_training(model, loss_fn, val_metric)
        self._validate_clients_data()
        self._set_client_weights()
        self._initialize_fl_server()
        self._do_training()
        self._do_client_testing()
        self._do_server_testing()
        if self.cfg.send_final_model:
            self._send_final_model()
        GlobusComputeServerLogger.save_globus_compute_log(self.cfg)
        self.communicator.shutdown_all_clients()

    def cleanup(self):
        self.communicator.shutdown_all_clients()


class APPFLGlobusComputeSyncServer(APPFLGlobusComputeServer):
    def __init__(self, cfg: DictConfig, gcc: Client):
        super().__init__(cfg, gcc)

    def _do_training(self):
        start_time = time.time()
        server_model_basename = str(uuid.uuid4()) + "_server_state"
        for t in range(self.cfg.num_epochs):
            self.logger.info(
                " ======================== Epoch [%d/%d] ======================== "
                % (t + 1, self.cfg.num_epochs)
            )
            per_iter_start = time.time()
            global_state = self.server.model.state_dict()
            self._lr_step(t)
            self.communicator.send_task_to_all_clients(
                client_training,
                self.weights,
                LargeObjectWrapper(global_state, f"{server_model_basename}_{t}"),
                do_validation=self.cfg.client_do_validation,
                global_epoch=t + 1,
            )
            local_states, client_logs = (
                self.communicator.receive_sync_endpoints_updates()
            )
            self._parse_client_logs(t + 1, client_logs)

            # Perform global update
            self.server.update(local_states)

            # Server validation and saving checkpoint
            if (t + 1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t + 1)
            self._save_checkpoint(t + 1)

            self.logger.info(f"Total training time: {time.time() - start_time:.3f}")
            self.logger.info(
                f"Training time for epoch {t + 1}: {time.time() - per_iter_start:.3f}"
            )


class APPFLGlobusComputeAsyncServer(APPFLGlobusComputeServer):
    def __init__(self, cfg: DictConfig, gcc: Client):
        super().__init__(cfg, gcc)
        self.global_epoch = 0
        self.client_model_timestamp = {i: 0 for i in range(self.cfg.num_clients)}

    def _do_training(self):
        start_time = time.time()
        server_model_basename = str(uuid.uuid4()) + "_server_state"
        # Broadcast the global model to the clients at the beginning
        global_state = self.server.model.state_dict()
        self.communicator.send_task_to_all_clients(
            client_training,
            self.weights,
            LargeObjectWrapper(global_state, f"{server_model_basename}_{0}"),
            do_validation=self.cfg.client_do_validation,
            global_epoch=0,
        )
        for t in range(self.cfg.num_epochs):
            self.logger.info(
                " ======================== Epoch [%d/%d] ======================== "
                % (t + 1, self.cfg.num_epochs)
            )
            client_idx, local_update, client_log = (
                self.communicator.receive_async_endpoint_update()
            )
            self._parse_client_logs(t + 1, client_log)

            # Perform asynchronous global update
            prev_global_step = self.server.global_step
            self.server.update(
                local_update, self.client_model_timestamp[client_idx], client_idx
            )
            self.client_model_timestamp[client_idx] = self.server.global_step
            if prev_global_step != self.server.global_step:
                self._lr_step(t + 1)

            # Send new model to the client
            if (t + 1) < self.cfg.num_epochs:
                self.communicator.send_task_to_one_client(
                    client_idx,
                    client_training,
                    self.weights,
                    LargeObjectWrapper(global_state, f"{server_model_basename}_{t}"),
                    do_validation=self.cfg.client_do_validation,
                    global_epoch=t + 1,
                )

            # Server validation and saving checkpoint
            if (t + 1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t + 1)
            self._save_checkpoint(t + 1)

            self.logger.info(f"Total training time: {time.time() - start_time:.3f}")
        self.communicator.cancel_all_tasks()


class APPFLGlobusComputeCompassServer(APPFLGlobusComputeServer):
    def __init__(self, cfg: DictConfig, gcc: Client):
        super().__init__(cfg, gcc)
        self.global_epoch = 0
        self.client_model_timestamp = {i: 0 for i in range(self.cfg.num_clients)}

    def _do_training(self):
        start_time = time.time()
        server_model_basename = str(uuid.uuid4()) + "_server_state"
        # Initialize the compass scheduler
        self.scheduler = SchedulerCompassGlobusCompute(
            self.communicator,
            self.server,
            self.cfg.fed.args.num_local_steps,
            self.cfg.num_clients,
            self.cfg.num_epochs,
            self.cfg.fed.args.optim_args.lr,
            self.logger,
            self.cfg.fed.servername == "ServerFedCompassNova",
            self.cfg.fed.args.q_ratio,
            self.cfg.fed.args.lambda_val,
            do_validation=self.cfg.client_do_validation,
        )
        # Broadcast the global model to the clients at the beginning
        global_state = self.server.model.state_dict()
        self.communicator.send_task_to_all_clients(
            client_training,
            self.weights,
            LargeObjectWrapper(global_state, f"{server_model_basename}_{0}"),
            do_validation=self.cfg.client_do_validation,
            global_epoch=0,
        )

        for t in range(self.cfg.num_epochs):
            self.logger.info(
                " ======================== Epoch [%d/%d] ======================== "
                % (t + 1, self.cfg.num_epochs)
            )
            client_log = self.scheduler.update()
            self._parse_client_logs(t + 1, client_log)

            # Server validation and saving checkpoint
            if (t + 1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t + 1)
            self._save_checkpoint(t + 1)

            self.logger.info(f"Total training time: {time.time() - start_time:.3f}")
        self.communicator.cancel_all_tasks()


def run_server(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    val_metric: Any,
    gcc: Client,
    test_data: Dataset = Dataset(),
    val_data: Dataset = Dataset(),
):
    if cfg.fed.args.use_compass:
        server = APPFLGlobusComputeCompassServer(cfg, gcc)
    elif cfg.fed.args.is_async:
        server = APPFLGlobusComputeAsyncServer(cfg, gcc)
    else:
        server = APPFLGlobusComputeSyncServer(cfg, gcc)
    try:
        server.set_server_dataset(
            validation_dataset=val_data, testing_dataset=test_data
        )
        server.run(model, loss_fn, val_metric)
    except Exception:
        logger = GlobusComputeServerLogger.get_logger()
        logger.error("Training failed with the exception.")
        logger.error(traceback.format_exc())
        server.cleanup()
