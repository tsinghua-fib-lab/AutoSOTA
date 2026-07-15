import copy
import torch
import logging
import numpy as np
from appfl.algorithm import BaseServer
from collections import OrderedDict
from .grpc_communicator_old_pb2 import Job
from torch.utils.data import DataLoader
from appfl.misc.utils import (
    create_custom_logger,
    load_model,
    validation,
    save_model_iteration,
    get_appfl_algorithm,
)


class APPFLgRPCServer:
    def __init__(self, cfg, model, loss_fn, test_dataset, num_clients, metric):
        self.logger1 = create_custom_logger(logging.getLogger(__name__), cfg)
        cfg["logginginfo"]["comm_size"] = 1

        self.logger = logging.getLogger(__name__)
        self.operator_id = cfg.operator.id
        self.cfg = cfg
        self.num_clients = num_clients
        self.num_epochs = cfg.num_epochs
        self.round_number = 1
        self.best_accuracy = 0.0
        self.device = "cpu"
        self.model = copy.deepcopy(model)
        self.loss_fn = loss_fn
        self.metric = metric
        """ Loading Model """
        if cfg.load_model:
            self.model = load_model(cfg)
        self.client_training_size = OrderedDict()
        self.client_training_size_received = OrderedDict()
        self.client_weights = OrderedDict()
        self.client_states = OrderedDict()
        for c in range(num_clients):
            self.client_states[c] = OrderedDict()
            self.client_states[c]["penalty"] = OrderedDict()
        self.client_learning_status = OrderedDict()

        self.dataloader = None
        if self.cfg.validation and len(test_dataset) > 0:
            self.dataloader = DataLoader(
                test_dataset,
                num_workers=0,
                batch_size=cfg.test_data_batch_size,
                shuffle=cfg.test_data_shuffle,
            )
        else:
            self.cfg.validation = False

        self.fed_server: BaseServer = get_appfl_algorithm(
            algorithm_name=self.cfg.fed.servername,
            args=(
                self.client_weights,
                self.model,
                self.loss_fn,
                self.num_clients,
                self.device,
            ),
            kwargs=self.cfg.fed.args,
        )

    """
    Return the tensor record of a global model requested by its name.
    """

    def get_tensor(self, name):
        return (
            np.array(self.fed_server.model.state_dict()[name])
            if name in self.fed_server.model.state_dict()
            else None
        )

    """
    Return the job status indicating the next job a client is supposed to do.
    """

    def get_job(self):
        job_todo = Job.WEIGHT
        self.logger.debug(
            f"[Round: {self.round_number: 04}] client_training_size_received: {self.client_training_size_received}"
        )
        if all(
            c in self.client_training_size_received for c in range(self.num_clients)
        ):
            job_todo = Job.TRAIN
        if self.round_number > self.num_epochs:
            job_todo = Job.QUIT
        return min(self.round_number, self.num_epochs), job_todo

    """
    Compute weights of clients based on their training data size.
    """

    def get_weight(self, client_id, training_size) -> float:
        self.client_training_size[client_id] = training_size
        self.client_training_size_received[client_id] = True
        self.logger.debug(
            f"[Round: {self.round_number: 04}] client_training_size_received: {self.client_training_size_received}"
        )

        # If we have received training size from all clients
        if all(
            c in self.client_training_size_received for c in range(self.num_clients)
        ):
            # Instantiate a fed_server if not instantiated yet.
            total_training_size = sum(
                self.client_training_size[c] for c in range(self.num_clients)
            )
            for c in range(self.num_clients):
                self.client_weights[c] = (
                    self.client_training_size[c] / total_training_size
                )

            self.fed_server.set_weights(self.client_weights)
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.client_weights: {self.client_weights}"
            )
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.client_training_size: {self.client_training_size}"
            )
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
            )
            return self.client_weights[client_id]
        else:
            return -1.0

    """
    Update model weights of a global model. After updating, we increment the round number.
    """

    def update_model_weights(self):
        self.logger.info(f"[Round: {self.round_number: 04}] Updating model weights")
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )
        client_states_list = []
        for i in self.client_states:
            if len(self.client_states[i]["dual"]) == 0:
                client_states_list.append(self.client_states[i]["primal"])
            else:
                client_states_list.append(self.client_states[i])
        self.fed_server.update(client_states_list)

        if self.cfg.validation:
            test_loss, accuracy = validation(
                self.fed_server, self.dataloader, self.metric
            )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

            self.logger.info(
                f"[Round: {self.round_number: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Best Accuracy: {self.best_accuracy:.2f}%"
            )

        if (
            self.round_number % self.cfg.checkpoints_interval == 0
            or self.round_number == self.cfg.num_epochs
        ):
            """Saving model"""
            if self.cfg.save_model:
                save_model_iteration(self.round_number, self.model, self.cfg)

        self.round_number += 1

    """
    Check if we have received model weights from all clients for this round.
    """

    def is_round_finished(self):
        return all(
            (c, self.round_number) in self.client_learning_status
            for c in range(0, self.num_clients)
        )

    """
    Receive model weights from a client. When we have received weights from all clients,
    it will trigger a global model update.
    """

    def send_learning_results(self, client_id, round_number, penalty, primal, dual):
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )
        primal_tensors = OrderedDict()
        dual_tensors = OrderedDict()
        for tensor in primal:
            name = tensor.name
            shape = tuple(tensor.data_shape)
            flat = np.frombuffer(tensor.data_bytes, dtype=eval(tensor.data_dtype))
            nparray = np.reshape(flat, newshape=shape, order="C")
            primal_tensors[name] = torch.from_numpy(nparray)
        for tensor in dual:
            name = tensor.name
            shape = tuple(tensor.data_shape)
            flat = np.frombuffer(tensor.data_bytes, dtype=eval(tensor.data_dtype))
            nparray = np.reshape(flat, newshape=shape, order="C")
            dual_tensors[name] = torch.from_numpy(nparray)
        self.client_states[client_id]["primal"] = primal_tensors
        self.client_states[client_id]["dual"] = dual_tensors
        self.client_states[client_id]["penalty"][client_id] = penalty
        self.client_learning_status[(client_id, round_number)] = True
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )

        # Round is finished when we have received model weights from all clients.
        if self.is_round_finished():
            self.logger.info(
                f"[Round: {self.round_number: 04}] Finished; all clients have sent their results."
            )
            self.update_model_weights()
