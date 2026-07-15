import gc
import time
import math
import threading
from omegaconf import DictConfig
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Union, Dict, Tuple
from appfl.algorithm.scheduler import BaseScheduler
from appfl.algorithm.aggregator import BaseAggregator
from appfl.misc.memory_utils import optimize_memory_cleanup


class CompassScheduler(BaseScheduler):
    """
    Scheduler for `FedCompass` asynchronous federated learning algorithm.
    Paper reference: https://arxiv.org/abs/2309.14675
    """

    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        self.client_info = {}
        self.group_counter = 0
        self.arrival_group = {}
        self.group_buffer = {}
        self.general_buffer = {
            "local_models": {},
            "local_steps": {},
            "timestamp": {},
        }
        self.future_record = {}
        self.global_timestamp = 0
        self._num_global_epochs = 0
        self._access_lock = threading.Lock()  # handle client requests as a queue
        self._group_aggregation_lock = (
            threading.Lock()
        )  # ensure only one group aggregation at a time
        self._timer_record = {}
        self.start_time = time.time()

        # Check for optimize_memory in scheduler_configs, default to True
        self.optimize_memory = getattr(scheduler_configs, "optimize_memory", True)

        super().__init__(scheduler_configs, aggregator, logger)

    def get_parameters(
        self, **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model parameters for the clients.
        The `Compass` scheduler requires all clients to get the initial model at the same
        time to record a consistent start time for the clients. So we add a wrapper to the
        `get_parameters` method of the `BaseScheduler` class to record the start time.
        """
        with self._access_lock:
            if kwargs.get("init_model", True):
                init_model_requests = (
                    self.init_model_requests
                    if hasattr(self, "init_model_requests")
                    else 0
                )
                if init_model_requests == 0:
                    self.start_time = time.time()
            parameters = super().get_parameters(**kwargs)
            return parameters

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Schedule an asynchronous global aggregation for the local model from a client
        using the `Compass` algorithm. The method will either return the current global model
        directly, or a `Future` object for the global model.
        :param `client_id`: the id of the client
        :param `local_model`: the local model from the client
        :param `kwargs`: additional keyword arguments for the scheduler
        :return: `global_model`: the global model and the number of local steps for the client
            in next round or a `Future` object for the global model
        """
        with self._access_lock:
            self._record_info(client_id)
            arrival_group_idx = (
                self.client_info[client_id]["goa"]
                if "goa" in self.client_info[client_id]
                else -1
            )
            global_model = (
                self._single_update(client_id, local_model, buffer=False, **kwargs)
                if arrival_group_idx == -1
                else self._group_update(
                    client_id, local_model, arrival_group_idx, **kwargs
                )
            )
            return global_model

    def get_num_global_epochs(self) -> int:
        """Return the total number of global epochs for federated learning."""
        with self._access_lock:
            return self._num_global_epochs

    def clean_up(self) -> None:
        """Optional function to clean up the scheduler states."""
        for group_idx in self._timer_record:
            self._timer_record[group_idx].cancel()

        # Memory optimization: Clean up all buffers and state
        if self.optimize_memory:
            optimize_memory_cleanup(
                self.client_info,
                self.arrival_group,
                self.group_buffer,
                self.general_buffer,
                self.future_record,
                force_gc=True,
            )

    def _record_info(self, client_id: Union[int, str]) -> None:
        """
        Record/update the client information for the coming client, including the client's
        - `timestamp`: the timestamp of the local model
        - `speed`: the estimated speed of the client
        - `local_steps`: the number of local steps for the client in current round
        :param `client_id`: the id of the client
        """
        curr_time = time.time() - self.start_time
        client_start_time = (
            self.client_info[client_id]["start_time"]
            if client_id in self.client_info
            else 0
        )
        client_update_time = curr_time - client_start_time
        client_steps = (
            self.client_info[client_id]["local_steps"]
            if client_id in self.client_info
            else self.scheduler_configs.max_local_steps
        )
        client_speed = client_update_time / client_steps
        if client_id not in self.client_info:
            self.client_info[client_id] = {
                "timestamp": 0,
                "speed": client_speed,
                "local_steps": self.scheduler_configs.max_local_steps,
            }
        else:
            self.client_info[client_id]["speed"] = (
                1 - self.scheduler_configs.speed_momentum
            ) * self.client_info[client_id][
                "speed"
            ] + self.scheduler_configs.speed_momentum * client_speed

    def _single_update(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        buffer: bool = True,
        **kwargs,
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Perform global update for the local model from a single client.
        :param `client_id`: the id of the client
        :param `local_model`: the local model from the client
        :param `buffer`: whether to buffer the local model or not
        :return: `global_model`: up-to-date global model
        :return: `local_steps`: the number of local steps for the client in next round
        """
        if not buffer:
            global_model = self.aggregator.aggregate(
                client_id,
                local_model,
                staleness=self.global_timestamp
                - self.client_info[client_id]["timestamp"],
                local_steps=self.client_info[client_id]["local_steps"],
                **kwargs,
            )
            self.global_timestamp += 1

            # Memory optimization: Clean up after single update
            if self.optimize_memory:
                optimize_memory_cleanup(local_model, force_gc=True)
        else:
            self.general_buffer["local_models"][client_id] = local_model
            self.general_buffer["local_steps"][client_id] = self.client_info[client_id][
                "local_steps"
            ]
            self.general_buffer["timestamp"][client_id] = self.client_info[client_id][
                "timestamp"
            ]
            global_model = self.aggregator.get_parameters(**kwargs)

            # Memory optimization: Clean up after buffering
            if self.optimize_memory:
                optimize_memory_cleanup(force_gc=False)
        self.client_info[client_id]["timestamp"] = self.global_timestamp
        self._assign_group(client_id, **kwargs)
        local_steps = self.client_info[client_id]["local_steps"]
        self._num_global_epochs += 1
        return global_model, {"local_steps": local_steps}

    def _group_update(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        group_idx: int,
        **kwargs,
    ) -> Union[Future, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Perform global update using local models from a certain arrival group. The function
        may return the global model directly, or a `Future` object for the global model.
        :param `client_id`: the id of the client
        :param `local_model`: the local model from the client
        :param `group_idx`: the index of the client arrival group
        :param `kwargs`: additional keyword arguments for the scheduler
        :return: `global_model`: current global model or a `Future` object for the global model
        :return: `local_steps`: the number of local steps for the client in next round
        """
        curr_time = time.time() - self.start_time
        if curr_time > self.arrival_group[group_idx]["latest_arrival_time"]:
            with self._group_aggregation_lock:
                self.arrival_group[group_idx]["clients"].remove(client_id)
                if len(self.arrival_group[group_idx]["clients"]) == 0:
                    # self.logger.debug(f'[Debug]: Group {group_idx} has no clients left after a late arrival, cleaning up group state.')
                    del self.arrival_group[group_idx]
            return self._single_update(
                client_id=client_id, local_model=local_model, buffer=True, **kwargs
            )
        else:
            self.arrival_group[group_idx]["clients"].remove(client_id)
            self.arrival_group[group_idx]["arrived_clients"].append(client_id)
            if group_idx not in self.group_buffer:
                self.group_buffer[group_idx] = {
                    "local_models": {},
                    "local_steps": {},
                    "timestamp": {},
                }
            self.group_buffer[group_idx]["local_models"][client_id] = local_model
            self.group_buffer[group_idx]["local_steps"][client_id] = self.client_info[
                client_id
            ]["local_steps"]
            self.group_buffer[group_idx]["timestamp"][client_id] = self.client_info[
                client_id
            ]["timestamp"]
            future = Future()
            self.future_record[client_id] = future

            # Memory optimization: Clean up after group buffering
            if self.optimize_memory:
                optimize_memory_cleanup(force_gc=False)
            if len(self.arrival_group[group_idx]["clients"]) == 0:
                self._group_aggregation(group_idx, **kwargs)
            return future

    def _group_aggregation(self, group_idx: int, **kwargs) -> None:
        """
        Perform the global aggregation using local models from a certain arrival group,
        as well as the general buffer, and assign new groups to the clients.
        :param `group_idx`: the index of the client arrival group
        :param `kwargs`: additional keyword arguments for the scheduler
        """
        with self._group_aggregation_lock:
            if group_idx in self.arrival_group and group_idx in self.group_buffer:
                # self.logger.debug(f'[Debug]: Starting group aggregation for group {group_idx} with clients {self.arrival_group[group_idx]["arrived_clients"]}.')
                if group_idx in self._timer_record:
                    del self._timer_record[group_idx]
                # merge the general buffer and group buffer
                local_models = {
                    **self.general_buffer["local_models"],
                    **self.group_buffer[group_idx]["local_models"],
                }
                local_steps = {
                    **self.general_buffer["local_steps"],
                    **self.group_buffer[group_idx]["local_steps"],
                }
                timestamp = {
                    **self.general_buffer["timestamp"],
                    **self.group_buffer[group_idx]["timestamp"],
                }
                staleness = {
                    client_id: self.global_timestamp - timestamp[client_id]
                    for client_id in timestamp
                }
                self.general_buffer = {
                    "local_models": {},
                    "local_steps": {},
                    "timestamp": {},
                }
                global_model = self.aggregator.aggregate(
                    local_models=local_models,
                    staleness=staleness,
                    local_steps=local_steps,
                    **kwargs,
                )
                self.global_timestamp += 1
                self._num_global_epochs += len(local_models)

                # Memory optimization: Clean up after group aggregation
                if self.optimize_memory:
                    optimize_memory_cleanup(
                        local_models, local_steps, timestamp, staleness, force_gc=True
                    )
                    # Clean up group buffer after aggregation
                    # self.logger.debug(f'[Debug]: Cleaning up group buffer for group {group_idx} after aggregation.')
                    del self.group_buffer[group_idx]
                    gc.collect()
                client_speeds = []
                for client_id in self.arrival_group[group_idx]["arrived_clients"]:
                    self.client_info[client_id]["timestamp"] = self.global_timestamp
                    client_speeds.append(
                        (client_id, self.client_info[client_id]["speed"])
                    )
                sorted_client_speeds = sorted(
                    client_speeds, key=lambda x: x[1], reverse=False
                )
                self.arrival_group[group_idx]["expected_arrival_time"] = 0
                self.arrival_group[group_idx]["latest_arrival_time"] = 0
                for client_id, _ in sorted_client_speeds:
                    self._assign_group(client_id, **kwargs)
                    self.future_record[client_id].set_result(
                        (
                            global_model,
                            {"local_steps": self.client_info[client_id]["local_steps"]},
                        )
                    )
                    del self.future_record[client_id]
                if len(self.arrival_group[group_idx]["clients"]) == 0:
                    # self.logger.debug(f'[Debug]: Group {group_idx} has no clients left after aggregation, cleaning up group state.')
                    del self.arrival_group[group_idx]

                # Memory optimization: Final cleanup after group processing
                if self.optimize_memory:
                    optimize_memory_cleanup(force_gc=True)

    def _assign_group(self, client_id: Union[int, str], **kwargs) -> None:
        """
        Assign the client to an arrival group based on the client estimated speed.
        :param `client_id`: the id of the client
        """
        curr_time = time.time() - self.start_time
        if len(self.arrival_group) == 0:
            self.arrival_group[self.group_counter] = {
                "clients": [client_id],
                "arrived_clients": [],
                "expected_arrival_time": (
                    curr_time
                    + self.scheduler_configs.max_local_steps
                    * self.client_info[client_id]["speed"]
                ),
                "latest_arrival_time": (
                    curr_time
                    + self.scheduler_configs.max_local_steps
                    * self.client_info[client_id]["speed"]
                    * self.scheduler_configs.latest_time_factor
                ),
            }
            group_timer = threading.Timer(
                self.arrival_group[self.group_counter]["latest_arrival_time"]
                - curr_time,
                self._group_aggregation,
                args=(self.group_counter,),
                kwargs=kwargs,
            )
            group_timer.start()
            self._timer_record[self.group_counter] = group_timer
            self.client_info[client_id]["goa"] = self.group_counter
            self.client_info[client_id]["local_steps"] = (
                self.scheduler_configs.max_local_steps
            )
            self.client_info[client_id]["start_time"] = curr_time
            self.group_counter += 1

            # Memory optimization: Clean up after group assignment
            if self.optimize_memory:
                optimize_memory_cleanup(force_gc=False)
        else:
            if not self._join_group(client_id):
                self._create_group(client_id, **kwargs)

    def _join_group(self, client_id: Union[int, str]) -> bool:
        """
        Try to join the client to an existing arrival group.
        :return: whether the client can join an existing group or not
        """
        curr_time = time.time() - self.start_time
        assigned_group = -1
        assigned_steps = -1
        for group in self.arrival_group:
            remaining_time = (
                self.arrival_group[group]["expected_arrival_time"] - curr_time
            )
            local_steps = math.floor(
                remaining_time / self.client_info[client_id]["speed"]
            )
            if (
                local_steps < self.scheduler_configs.min_local_steps
                or local_steps < assigned_steps
                or local_steps > self.scheduler_configs.max_local_steps
            ):
                continue
            else:
                assigned_group = group
                assigned_steps = local_steps
        if assigned_group == -1:
            return False
        else:
            self.arrival_group[assigned_group]["clients"].append(client_id)
            self.client_info[client_id]["goa"] = assigned_group
            self.client_info[client_id]["local_steps"] = assigned_steps
            self.client_info[client_id]["start_time"] = curr_time
            return True

    def _create_group(self, client_id: Union[int, str], **kwargs):
        """
        Create a new group for the client.
        :param `client_id`: the id of the client
        """
        curr_time = time.time() - self.start_time
        assigned_steps = -1
        for group in self.arrival_group:
            if curr_time < self.arrival_group[group]["latest_arrival_time"]:
                fastest_speed = float("inf")
                group_clients = (
                    self.arrival_group[group]["clients"]
                    + self.arrival_group[group]["arrived_clients"]
                )
                for client in group_clients:
                    fastest_speed = min(
                        fastest_speed, self.client_info[client]["speed"]
                    )
                est_arrival_time = (
                    self.arrival_group[group]["latest_arrival_time"]
                    + fastest_speed * self.scheduler_configs.max_local_steps
                )
                local_steps = math.floor(
                    (est_arrival_time - curr_time)
                    / self.client_info[client_id]["speed"]
                )
                if local_steps <= self.scheduler_configs.max_local_steps:
                    assigned_steps = max(assigned_steps, local_steps)
        assigned_steps = (
            self.scheduler_configs.min_local_steps
            if assigned_steps >= 0
            and assigned_steps < self.scheduler_configs.min_local_steps
            else assigned_steps
        )
        assigned_steps = (
            self.scheduler_configs.max_local_steps
            if assigned_steps < 0
            else assigned_steps
        )
        self.arrival_group[self.group_counter] = {
            "clients": [client_id],
            "arrived_clients": [],
            "expected_arrival_time": (
                curr_time + assigned_steps * self.client_info[client_id]["speed"]
            ),
            "latest_arrival_time": (
                curr_time
                + assigned_steps
                * self.client_info[client_id]["speed"]
                * self.scheduler_configs.latest_time_factor
            ),
        }
        group_timer = threading.Timer(
            self.arrival_group[self.group_counter]["latest_arrival_time"] - curr_time,
            self._group_aggregation,
            args=(self.group_counter,),
            kwargs=kwargs,
        )
        group_timer.start()
        self._timer_record[self.group_counter] = group_timer
        self.client_info[client_id]["goa"] = self.group_counter
        self.client_info[client_id]["local_steps"] = assigned_steps
        self.client_info[client_id]["start_time"] = curr_time
        self.group_counter += 1

        # Memory optimization: Clean up after group creation
        if self.optimize_memory:
            optimize_memory_cleanup(force_gc=False)
