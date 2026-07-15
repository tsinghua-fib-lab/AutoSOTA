import abc
import copy
import math
import time
import uuid
import threading
from logging import Logger
from typing import Any
from collections import OrderedDict
from appfl.misc.deprecation import deprecated


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.scheduler instead."
)
class SchedulerCompass(abc.ABC):
    def __init__(
        self,
        communicator,
        server: Any,
        max_local_steps: int,
        num_clients: int,
        num_global_epochs: int,
        lr: float,
        logger: Logger,
        use_nova: bool,
        q_ratio: float = 0.2,
        lambda_val: float = 1.5,
        **kwargs,
    ):
        self.iter = 0
        self.lr = lr
        self.communicator = communicator
        self.server = server
        self.logger = logger
        self.num_clients = num_clients
        self.num_global_epochs = num_global_epochs
        self.group_counter = 0
        self.max_local_steps = max_local_steps
        self.min_local_steps = max(math.floor(q_ratio * self.max_local_steps), 1)
        self.max_local_steps_bound = math.floor(1.2 * self.max_local_steps)
        self.SPEED_MOMENTUM = 0.9
        self.LATEST_TIME_FACTOR = lambda_val
        self.LR_DECAY = 0.995
        self.client_info = {}
        self.group_of_arrival = OrderedDict()
        self.use_nova = use_nova  # whether to use normalized averaging
        self.start_time = time.time()
        self.__dict__.update(kwargs)

    @abc.abstractmethod
    def _recv_local_model_from_client(self):
        pass

    @abc.abstractmethod
    def _send_global_model_to_client(self, client_idx, client_steps, client_lr):
        pass

    def update(self):
        """Schedule update when receive information from one client."""
        client_idx, local_model, local_log = self._recv_local_model_from_client()
        self._record_info(client_idx)
        self._update(local_model, client_idx)
        return local_log

    def _update(self, local_model: dict, client_idx: int):
        self.iter += 1
        self.validation_flag = False
        # Get the client group
        group_idx = (
            self.client_info[client_idx]["goa"]
            if "goa" in self.client_info[client_idx]
            else -1
        )
        if group_idx == -1:
            self._single_update(local_model, client_idx, buffer=False)
        else:
            self._group_update(local_model, client_idx, group_idx)

    def _single_update(self, local_model: dict, client_idx: int, buffer: bool = True):
        """Update the global model using the local model itself."""
        # Update the global model
        self.server.model.to("cpu")
        # if buffer and False:
        if buffer:
            if self.use_nova:
                self.server.single_buffer(
                    local_model,
                    self.client_info[client_idx]["step"],
                    client_idx,
                    self.client_info[client_idx]["local_steps"],
                )
            else:
                self.server.single_buffer(
                    local_model, self.client_info[client_idx]["step"], client_idx
                )
            self.validation_flag = True
        else:
            self.server.update(
                local_model, self.client_info[client_idx]["step"], client_idx
            )
            self.validation_flag = True
        self.client_info[client_idx]["step"] = self.server.global_step
        # Assign the client to a group of arrival
        self._assign_group(client_idx)
        if self.iter < self.num_global_epochs:
            self._send_model(client_idx)
        else:
            self.server.update_all()

    def _group_update(self, local_model: dict, client_idx: int, group_idx: int):
        curr_time = time.time() - self.start_time
        # Update directly if the client arrives late
        if curr_time >= self.group_of_arrival[group_idx]["latest_arrival_time"]:
            self.group_of_arrival[group_idx]["clients"].remove(client_idx)
            # self.logger.info(f"Client {client_idx} arrived at group {group_idx} at time {curr_time}")
            if len(self.group_of_arrival[group_idx]["clients"]) == 0:
                del self.group_of_arrival[group_idx]
            self._single_update(local_model, client_idx)
        # Add the new coming model to the buffer and wait until group timer event gets triggered
        else:
            self.group_of_arrival[group_idx]["clients"].remove(client_idx)
            self.group_of_arrival[group_idx]["arrived_clients"].append(client_idx)
            # self.logger.info(f"Client {client_idx} arrived at group {group_idx} at time {curr_time}")
            self.server.model.to("cpu")
            if self.use_nova:
                self.server.buffer(
                    local_model,
                    self.client_info[client_idx]["step"],
                    client_idx,
                    group_idx,
                    self.client_info[client_idx]["local_steps"],
                )
            else:
                self.server.buffer(
                    local_model,
                    self.client_info[client_idx]["step"],
                    client_idx,
                    group_idx,
                )
            if len(self.group_of_arrival[group_idx]["clients"]) == 0:
                self._group_aggregation(group_idx)

    def _assign_group(self, client_idx: int):
        """Assign a group to the client or create a new group for it when no suitable one exists."""
        curr_time = time.time() - self.start_time
        # Create a new group if no group exists at all
        if len(self.group_of_arrival) == 0:
            self.group_of_arrival[self.group_counter] = {
                "clients": [client_idx],
                "arrived_clients": [],
                "expected_arrival_time": curr_time
                + self.max_local_steps * self.client_info[client_idx]["speed"],
                "latest_arrival_time": curr_time
                + self.max_local_steps
                * self.client_info[client_idx]["speed"]
                * self.LATEST_TIME_FACTOR,
            }
            # self.logger.info(f"Group {self.group_counter} created at {curr_time} with expected_arrival_time: {self.group_of_arrival[self.group_counter]['expected_arrival_time']}, latest_arrival_time: {self.group_of_arrival[self.group_counter]['latest_arrival_time']}")
            # self.logger.info(f"Client {client_idx} joinded group {self.group_counter} at time {curr_time}")
            # Add a timer event
            timer = threading.Timer(
                self.group_of_arrival[self.group_counter]["latest_arrival_time"]
                - curr_time,
                self._group_aggregation,
                args=(self.group_counter,),
            )
            timer.start()
            self.client_info[client_idx]["goa"] = self.group_counter
            self.client_info[client_idx]["local_steps"] = self.max_local_steps
            self.client_info[client_idx]["start_time"] = curr_time
            # self.logger.info(f"Client {client_idx} - Create GOA {self.group_counter} - Local steps {self.max_local_steps}")
            self.group_counter += 1
        # Assign the client to a group or create one for it
        else:
            if not self._join_group(client_idx):
                self._create_group(client_idx)

    def _join_group(self, client_idx: int):
        curr_time = time.time() - self.start_time
        assigned_group = -1  # assigned group for the client
        assigned_steps = -1  # assigned local training steps for the client
        for group in self.group_of_arrival:
            remaining_time = (
                self.group_of_arrival[group]["expected_arrival_time"] - curr_time
            )
            local_steps = math.floor(
                remaining_time / self.client_info[client_idx]["speed"]
            )
            if (
                local_steps < self.min_local_steps
                or local_steps < assigned_steps
                or local_steps > self.max_local_steps_bound
            ):
                continue
            else:
                assigned_steps = local_steps
                assigned_group = group
        if assigned_group != -1:
            self.client_info[client_idx]["goa"] = assigned_group
            self.client_info[client_idx]["local_steps"] = assigned_steps
            self.client_info[client_idx]["start_time"] = curr_time
            self.group_of_arrival[assigned_group]["clients"].append(client_idx)
            # self.logger.info(f"Client {client_idx} joinded group {assigned_group} at time {curr_time}")
            # self.logger.info(f"Client {client_idx} - Join GOA {assigned_group} - Local steps {assigned_steps}")
            return True
        else:
            return False

    def _create_group(self, client_idx: int):
        curr_time = time.time() - self.start_time
        # Calculate the assigned steps for the client
        assigned_steps = -1
        for group in self.group_of_arrival:
            if curr_time < self.group_of_arrival[group]["latest_arrival_time"]:
                # Find the client with the fastest speed
                fastest_speed = float("inf")
                group_clients = (
                    self.group_of_arrival[group]["clients"]
                    + self.group_of_arrival[group]["arrived_clients"]
                )
                for client in group_clients:
                    fastest_speed = min(
                        fastest_speed, self.client_info[client]["speed"]
                    )
                est_arrival_time = (
                    self.group_of_arrival[group]["latest_arrival_time"]
                    + fastest_speed * self.max_local_steps
                )
                local_steps = math.floor(
                    (est_arrival_time - curr_time)
                    / self.client_info[client_idx]["speed"]
                )
                if local_steps <= self.max_local_steps:
                    assigned_steps = max(assigned_steps, local_steps)
        assigned_steps = (
            self.min_local_steps
            if assigned_steps >= 0 and assigned_steps < self.min_local_steps
            else assigned_steps
        )
        assigned_steps = self.max_local_steps if assigned_steps < 0 else assigned_steps
        # Create a group for the client
        self.group_of_arrival[self.group_counter] = {
            "clients": [client_idx],
            "arrived_clients": [],
            "expected_arrival_time": curr_time
            + assigned_steps * self.client_info[client_idx]["speed"],
            "latest_arrival_time": curr_time
            + assigned_steps
            * self.client_info[client_idx]["speed"]
            * self.LATEST_TIME_FACTOR,
        }
        # self.logger.info(f"Group {self.group_counter} created at {curr_time} with expected_arrival_time: {self.group_of_arrival[self.group_counter]['expected_arrival_time']}, latest_arrival_time: {self.group_of_arrival[self.group_counter]['latest_arrival_time']}")
        # self.logger.info(f"Client {client_idx} joinded group {self.group_counter} at time {curr_time}")
        # Add a timer event
        timer = threading.Timer(
            self.group_of_arrival[self.group_counter]["latest_arrival_time"]
            - curr_time,
            self._group_aggregation,
            args=(self.group_counter,),
        )
        timer.start()
        self.client_info[client_idx]["goa"] = self.group_counter
        self.client_info[client_idx]["local_steps"] = assigned_steps
        self.client_info[client_idx]["start_time"] = curr_time
        # self.logger.info(f"Client {client_idx} - Create GOA {self.group_counter} - Local steps {assigned_steps}")
        self.group_counter += 1

    def _record_info(self, client_idx: int):
        """Record/update the client information for the coming client."""
        curr_time = time.time() - self.start_time
        local_start_time = (
            self.client_info[client_idx]["start_time"]
            if client_idx in self.client_info
            else 0
        )
        local_update_time = curr_time - local_start_time
        local_steps = (
            self.client_info[client_idx]["local_steps"]
            if client_idx in self.client_info
            else self.max_local_steps
        )
        local_speed = local_update_time / local_steps
        if client_idx not in self.client_info:
            self.client_info[client_idx] = {
                "speed": local_speed,
                "step": 0,
                "total_steps": self.min_local_steps,
            }
        else:
            self.client_info[client_idx]["speed"] = (
                1 - self.SPEED_MOMENTUM
            ) * self.client_info[client_idx][
                "speed"
            ] + self.SPEED_MOMENTUM * local_speed

    def _send_model(self, client_idx: int):
        # Record total steps and decay the learning rate
        self.client_info[client_idx]["total_steps"] += self.client_info[client_idx][
            "local_steps"
        ]
        client_lr = self.lr * (self.LR_DECAY) ** (
            math.floor(
                self.client_info[client_idx]["total_steps"] / self.max_local_steps
            )
        )
        # self.logger.info(f"Total number of steps for client{client_idx} is {self.client_info[client_idx]['total_steps']}")
        # self.logger.info(f"Learning rate for client{client_idx} is {client_lr}")
        client_steps = self.client_info[client_idx]["local_steps"]
        self._send_global_model_to_client(client_idx, client_steps, client_lr)

    def _group_aggregation(self, group_idx: int):
        if group_idx in self.group_of_arrival:
            """Aggregate all the local gradients from a certain group."""
            # TODO: Do we need to add some lock?
            self.validation_flag = True
            self.server.model.to("cpu")
            self.server.update_group(group_idx)
            client_speed = []
            for client in self.group_of_arrival[group_idx]["arrived_clients"]:
                self.client_info[client]["step"] = self.server.global_step
                client_speed.append((client, self.client_info[client]["speed"]))
            # sort clients in reverse order of speed, and assign group to clients (TODO: Check this)
            sorted_client_speed = sorted(
                client_speed, key=lambda x: x[1], reverse=False
            )
            self.group_of_arrival[group_idx]["expected_arrival_time"] = 0
            self.group_of_arrival[group_idx]["latest_arrival_time"] = 0
            for client, _ in sorted_client_speed:
                self._assign_group(client)
            # delete the group is not waiting any client
            if len(self.group_of_arrival[group_idx]["clients"]) == 0:
                del self.group_of_arrival[group_idx]
                # self.logger.info(f"Group {group_idx} is deleted at {time.time() - self.start_time}")
            # Send the model if required
            if self.iter < self.num_global_epochs:
                for client, _ in sorted_client_speed:
                    self._send_model(client)
            else:
                self.server.model.to("cpu")
                self.server.update_all()


class SchedulerCompassMPI(SchedulerCompass):
    def _recv_local_model_from_client(self):
        client_idx, local_model = self.communicator.recv_local_model_from_client(
            copy.deepcopy(self.server.model)
        )
        return client_idx, local_model, None

    def _send_global_model_to_client(self, client_idx, client_steps, client_lr):
        self.communicator.send_global_model_to_client(
            self.server.model.state_dict(),
            {"done": False, "steps": client_steps, "lr": client_lr},
            client_idx,
        )


class SchedulerCompassGlobusCompute(SchedulerCompass):
    def _recv_local_model_from_client(self):
        return self.communicator.receive_async_endpoint_update()

    def _send_global_model_to_client(self, client_idx, client_steps, client_lr):
        from appfl.comm.globus_compute import client_training
        from appfl.comm.utils.s3_storage import LargeObjectWrapper

        if not hasattr(self, "server_model_basename"):
            self.server_model_basename = str(uuid.uuid4()) + "_server_state"
        self.communicator.set_learning_rate(client_lr, client_idx=client_idx)
        self.communicator.set_local_steps(client_steps, client_idx=client_idx)
        self.communicator.send_task_to_one_client(
            client_idx,
            client_training,
            self.server.weights,
            LargeObjectWrapper(
                self.server.model.state_dict(),
                f"{self.server_model_basename}_{self.iter + 1}",
            ),
            do_validation=self.do_validation,
            global_epoch=self.iter + 1,
        )
