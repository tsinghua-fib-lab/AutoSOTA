import time
import math
import csv
import os
import random
import threading
from omegaconf import DictConfig
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Union, Dict, Tuple, List
from appfl.algorithm.scheduler import BaseScheduler
from appfl.algorithm.aggregator import BaseAggregator


class QueueScheduler(BaseScheduler):
    """
    Scheduler for `FedQueue` asynchronous federated learning algorithm.
    """

    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.global_round = 0
        self._num_global_epochs = 0
        self.num_clients = self.scheduler_configs.num_clients
        self.t_sync = self.scheduler_configs.get("t_sync", 3600)
        self.lr_base = self.scheduler_configs.get("lr_base", 0.01)
        self.warm_up_steps = self.scheduler_configs.get("warm_up_steps", 20)
        self.alpha_queue = self.scheduler_configs.get("alpha_queue", 0.5)
        self.alpha_compute = self.scheduler_configs.get("alpha_compute", 0.5)
        self.safety_buffer = self.scheduler_configs.get("safety_buffer", 60)
        self.admission_tolerance = self.scheduler_configs.get(
            "admission_tolerance", 0.15
        )
        self.simulate_queue = self.scheduler_configs.get("simulate_queue", False)
        self.log_fedqueue_metrics = self.scheduler_configs.get(
            "log_fedqueue_metrics", False
        )
        self.q_init = self.scheduler_configs.get("q_init", 2.0)
        self.alpha_admission = self.scheduler_configs.get("alpha_admission", 0.2)
        self.beta_queue = self.scheduler_configs.get("beta_queue", self.alpha_queue)
        self.theta = self.scheduler_configs.get("theta", self.safety_buffer)
        self.queue_mode = str(
            self.scheduler_configs.get("queue_mode", "lognormal")
        ).lower()
        self.queue_sigma = self.scheduler_configs.get("queue_sigma", 0.4)
        self.queue_means = self._parse_float_list(
            self.scheduler_configs.get("queue_means", self.q_init)
        )
        self.queue_fixed = self._parse_float_list(
            self.scheduler_configs.get("queue_fixed", self.q_init)
        )
        self.slowdown = self._parse_float_list(
            self.scheduler_configs.get("slowdown", 1.0)
        )
        self._queue_rng = random.Random(self.scheduler_configs.get("queue_seed", 0))
        self._access_lock = threading.Lock()
        self.future_record = {}
        self.timer_record = None
        self.client_round_record = {}
        self.client_steps_record = {}
        self.queue_time_estimation = {}
        self.compute_time_estimation = {}
        self.client_deadline_record = {}
        self.round_start_record = {}
        self.late_client_model_buffer = []
        self._fedqueue_log_paths = None
        self._reset_client_model_buffer()
        self._init_fedqueue_logs()

    def get_parameters(
        self, **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        with self._access_lock:
            kwargs["record_time"] = True
            return super().get_parameters(**kwargs)

    def get_num_global_epochs(self) -> int:
        """Return the total number of global epochs for federated learning."""
        with self._access_lock:
            return self._num_global_epochs

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        with self._access_lock:
            queue_prior = self.queue_time_estimation.get(client_id, self.q_init)
            self._update_queue_estimation(client_id, kwargs)
            self._update_compute_estimation(client_id, kwargs)
            future = Future()
            self.future_record[client_id] = future
            kwargs["queue_prior"] = queue_prior
            self._schedule_update_with_admission(client_id, local_model, kwargs)
            if self._should_aggregate(lock_acquired=True):
                self._aggregate_global_model(lock_acquired=True)
            return future

    def _schedule_update_with_admission(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        kwargs: Dict,
    ) -> None:
        curr_round = self._get_curr_round(client_id)
        q_delay = kwargs.get("queue_delay", kwargs.get("queue_time", 0.0))
        q_hat = kwargs.get("queue_prior", self.queue_time_estimation[client_id])
        deadline = self.client_deadline_record.get(
            client_id,
            q_hat + self.alpha_admission * self.t_sync,
        )
        admitted = q_delay <= deadline
        arrival_offset = time.time() - self.round_start_record.get(
            client_id, time.time()
        )
        entry = {
            "client_id": client_id,
            "local_model": local_model,
            "local_steps": self._get_local_steps(client_id),
            "curr_round": curr_round,
            "q_delay": q_delay,
            "q_hat": q_hat,
            "deadline": deadline,
            "Jk": self._get_job_budget_from_queue_estimate(q_hat),
            "arrival_offset": arrival_offset,
        }
        if admitted:
            self.client_model_buffer["local_models"][client_id] = local_model
            self.client_model_buffer["local_steps"][client_id] = entry["local_steps"]
            self.client_model_buffer["curr_round"][client_id] = curr_round
            self.client_model_buffer["q_delay"][client_id] = q_delay
            self.client_model_buffer["q_hat"][client_id] = entry["q_hat"]
            self.client_model_buffer["deadline"][client_id] = deadline
            self.client_model_buffer["arrival_offset"][client_id] = arrival_offset
        else:
            self.late_client_model_buffer.append(entry)
        self._log_round_row(client_id, admitted, entry)
        if self.timer_record is None:
            self._start_aggregation_timer()

    def _aggregate_global_model(
        self,
        lock_acquired: bool = False,
    ) -> None:
        # Stop current timer if all clients arrived
        if lock_acquired:
            if self.timer_record is not None:
                self.timer_record.cancel()

        if not lock_acquired:
            self._access_lock.acquire()

        self._merge_carry_over_clients()

        all_clients = list(self.client_model_buffer["local_models"].keys())
        late_clients = [entry["client_id"] for entry in self.late_client_model_buffer]

        if len(all_clients) > 0 or len(self.late_client_model_buffer) > 0:
            if len(all_clients) == 0:
                self._use_late_updates_if_no_admitted_updates()
                all_clients = list(self.client_model_buffer["local_models"].keys())

            self.logger.info(
                f"[FedQueue Scheduler][Global Round {self.global_round + 1}] Aggregating global model for clients: {list(self.client_model_buffer['local_models'].keys())} {'(Triggered by timer)' if not lock_acquired else ''}"
            )
            staleness = {
                client_id: self.global_round
                - self.client_model_buffer["curr_round"][client_id]
                for client_id in all_clients
            }
            global_model = self.aggregator.aggregate(
                local_models=self.client_model_buffer["local_models"],
                staleness=staleness,
                local_steps=self.client_model_buffer["local_steps"],
            )
            self._log_applied_rows(all_clients, staleness)
            self.global_round += 1
            self._num_global_epochs += 1
            for client_id in all_clients:
                client_metadata = self._get_client_metadata(client_id, all_clients)
                self.future_record[client_id].set_result(
                    (global_model, client_metadata)
                )
                self.client_round_record[client_id] = self.global_round
                del self.future_record[client_id]
            self._reset_client_model_buffer()
            self._log_detail(
                f"[Round {self.global_round}] admitted={all_clients}; "
                f"late_buffer={late_clients}"
            )

            self._start_aggregation_timer()

        if not lock_acquired:
            self._access_lock.release()

    def _should_aggregate(self, lock_acquired: bool) -> bool:
        admitted_count = len(self.client_model_buffer["local_models"])
        arrived_count = admitted_count + len(self.late_client_model_buffer)
        return lock_acquired and arrived_count >= self.num_clients

    def _use_late_updates_if_no_admitted_updates(self) -> None:
        for entry in self.late_client_model_buffer:
            client_id = entry["client_id"]
            self.client_model_buffer["local_models"][client_id] = entry["local_model"]
            self.client_model_buffer["local_steps"][client_id] = entry["local_steps"]
            self.client_model_buffer["curr_round"][client_id] = entry["curr_round"]
            self.client_model_buffer["q_delay"][client_id] = entry["q_delay"]
            self.client_model_buffer["q_hat"][client_id] = entry["q_hat"]
            self.client_model_buffer["deadline"][client_id] = entry["deadline"]
            self.client_model_buffer["arrival_offset"][client_id] = entry[
                "arrival_offset"
            ]
        self.late_client_model_buffer = []

    def _merge_carry_over_clients(self) -> None:
        retained_late_clients = []
        for entry in self.late_client_model_buffer:
            if entry["curr_round"] >= self.global_round:
                retained_late_clients.append(entry)
                continue
            client_id = entry["client_id"]
            self.client_model_buffer["local_models"][client_id] = entry["local_model"]
            self.client_model_buffer["local_steps"][client_id] = entry["local_steps"]
            self.client_model_buffer["curr_round"][client_id] = entry["curr_round"]
            self.client_model_buffer["q_delay"][client_id] = entry["q_delay"]
            self.client_model_buffer["q_hat"][client_id] = entry["q_hat"]
            self.client_model_buffer["deadline"][client_id] = entry["deadline"]
            self.client_model_buffer["arrival_offset"][client_id] = entry[
                "arrival_offset"
            ]
        self.late_client_model_buffer = retained_late_clients

    def _start_aggregation_timer(self) -> None:
        if self.timer_record is not None:
            self.timer_record.cancel()
        self.timer_record = threading.Timer(
            (1 + self.admission_tolerance) * self.t_sync,
            self._aggregate_global_model,
            kwargs={"lock_acquired": False},
        )
        self.timer_record.start()

    def _update_queue_estimation(
        self,
        client_id: Union[int, str],
        kwargs: Dict,
    ) -> None:
        assert "queue_time" in kwargs, "QueueScheduler requires `queue_time` in kwargs."
        queue_time = kwargs.get("queue_delay", kwargs["queue_time"])
        if client_id not in self.queue_time_estimation:
            self.queue_time_estimation[client_id] = self.q_init
        self.queue_time_estimation[client_id] = (
            self.beta_queue * queue_time
            + (1 - self.beta_queue) * self.queue_time_estimation[client_id]
        )
        self.logger.debug(
            f"Updated queue time estimation for client {client_id}: {self.queue_time_estimation[client_id]}"
        )

    def _update_compute_estimation(
        self,
        client_id: Union[int, str],
        kwargs: Dict,
    ) -> None:
        assert "compute_second_per_step" in kwargs, (
            "QueueScheduler requires `compute_second_per_step` in kwargs."
        )
        compute_second_per_step = kwargs["compute_second_per_step"]
        if client_id not in self.compute_time_estimation:
            self.compute_time_estimation[client_id] = compute_second_per_step
        self.compute_time_estimation[client_id] = (
            self.alpha_compute * compute_second_per_step
            + (1 - self.alpha_compute) * self.compute_time_estimation[client_id]
        )

    def _get_local_steps(self, client_id: Union[int, str]) -> int:
        if client_id not in self.client_steps_record:
            self.client_steps_record[client_id] = self.warm_up_steps
        return self.client_steps_record[client_id]

    def _get_curr_round(self, client_id: Union[int, str]) -> int:
        if client_id not in self.client_round_record:
            self.client_round_record[client_id] = 0
        return self.client_round_record[client_id]

    def _reset_client_model_buffer(self):
        self.client_model_buffer = {
            "local_models": {},
            "local_steps": {},
            "curr_round": {},
            "q_delay": {},
            "q_hat": {},
            "deadline": {},
            "arrival_offset": {},
        }

    def _get_client_metadata(
        self, client_id: Union[int, str], all_clients: List[Union[int, str]]
    ) -> Dict[str, Any]:
        q_hat = self.queue_time_estimation.get(client_id, self.q_init)
        job_budget = self._get_job_budget_from_queue_estimate(q_hat)
        local_steps = max(
            math.floor(job_budget / self.compute_time_estimation[client_id]),
            self.warm_up_steps,
        )
        step_clients = list(self.compute_time_estimation.keys())
        all_local_steps = [
            max(
                math.floor(
                    self._get_job_budget_from_queue_estimate(
                        self.queue_time_estimation.get(cid, self.q_init)
                    )
                    / self.compute_time_estimation[cid]
                ),
                self.warm_up_steps,
            )
            for cid in step_clients
        ]
        min_local_steps = min(all_local_steps)
        learning_rate = self.lr_base * (min_local_steps / local_steps)
        client_metadata = {
            "job_budget": job_budget,
            "local_steps": local_steps,
            "learning_rate": learning_rate,
            "start_time": time.time(),
        }
        deadline = q_hat + self.alpha_admission * self.t_sync
        self.client_deadline_record[client_id] = deadline
        self.round_start_record[client_id] = client_metadata["start_time"]
        if self.simulate_queue:
            queue_delay = self._sample_queue_delay(client_id)
            client_metadata.update(
                {
                    "queue_delay": queue_delay,
                }
            )
        client_metadata.update(
            {
                "slowdown": self._get_indexed_value(self.slowdown, client_id, 1.0),
                "origin_round": self.global_round,
            }
        )
        self._log_detail(
            f"client={client_id} q_hat={q_hat:.6f} "
            f"deadline={deadline:.6f} "
            f"local_steps={local_steps} learning_rate={learning_rate:.8f}"
        )
        self.logger.debug(f"Client {client_id} metadata: {client_metadata}")
        self.client_steps_record[client_id] = local_steps
        return client_metadata

    def _get_job_budget_from_queue_estimate(self, q_hat: float) -> float:
        return max(0.5, self.t_sync - q_hat - self.theta)

    def _sample_queue_delay(self, client_id: Union[int, str]) -> float:
        if self.queue_mode == "off":
            return 0.0
        if self.queue_mode == "fixed":
            return max(0.0, self._get_indexed_value(self.queue_fixed, client_id, 0.0))
        mean = max(
            1e-6, self._get_indexed_value(self.queue_means, client_id, self.q_init)
        )
        sigma = float(self.queue_sigma)
        mu = math.log(mean) - 0.5 * sigma * sigma
        return max(0.0, self._queue_rng.lognormvariate(mu, sigma))

    def _get_indexed_value(
        self, values: List[float], client_id: Union[int, str], default: float
    ) -> float:
        if len(values) == 0:
            return default
        idx = self._client_index(client_id)
        if idx < len(values):
            return values[idx]
        return values[-1]

    def _client_index(self, client_id: Union[int, str]) -> int:
        try:
            return int(client_id) - 1
        except (TypeError, ValueError):
            digits = "".join(ch for ch in str(client_id) if ch.isdigit())
            if digits:
                return max(0, int(digits) - 1)
        return 0

    def _parse_float_list(self, value: Any) -> List[float]:
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(v.strip()) for v in str(value).split(",") if v.strip()]

    def _init_fedqueue_logs(self) -> None:
        if not self.log_fedqueue_metrics:
            return
        outdir = self.scheduler_configs.get("fedqueue_log_dir", "./output")
        log_dir = os.path.join(str(outdir), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._fedqueue_log_paths = {
            "rounds": os.path.join(log_dir, "fedqueue_rounds.csv"),
            "applied": os.path.join(log_dir, "fedqueue_applied.csv"),
            "detail": os.path.join(log_dir, "fedqueue.log"),
        }
        with open(self._fedqueue_log_paths["rounds"], "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "round",
                    "cid",
                    "admitted",
                    "q_delay",
                    "q_hat",
                    "deadline",
                    "Jk",
                    "Ek",
                    "Emin",
                    "eta_k",
                    "arrival_offset",
                ]
            )
        with open(self._fedqueue_log_paths["applied"], "w", newline="") as f:
            csv.writer(f).writerow(
                ["round_applied", "cid", "origin_round", "staleness", "p_k"]
            )
        open(self._fedqueue_log_paths["detail"], "w").close()

    def _log_round_row(
        self, client_id: Union[int, str], admitted: bool, entry: Dict
    ) -> None:
        if self._fedqueue_log_paths is None:
            return
        local_steps = entry["local_steps"]
        all_steps = list(self.client_steps_record.values()) or [local_steps]
        min_steps = min(all_steps)
        eta = self.lr_base * (min_steps / local_steps)
        with open(self._fedqueue_log_paths["rounds"], "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.global_round + 1,
                    client_id,
                    int(admitted),
                    round(float(entry["q_delay"]), 6),
                    round(float(entry["q_hat"]), 6),
                    round(float(entry["deadline"]), 6),
                    round(float(entry["Jk"]), 6),
                    int(local_steps),
                    int(min_steps),
                    round(float(eta), 8),
                    round(float(entry["arrival_offset"]), 6),
                ]
            )

    def _log_applied_rows(
        self, all_clients: List[Union[int, str]], staleness: Dict[Union[str, int], int]
    ) -> None:
        if self._fedqueue_log_paths is None:
            return
        sample_sizes = getattr(self.aggregator, "client_sample_size", {})
        total_sample_size = sum(sample_sizes.values()) if sample_sizes else 0
        with open(self._fedqueue_log_paths["applied"], "a", newline="") as f:
            writer = csv.writer(f)
            for client_id in all_clients:
                p_k = (
                    sample_sizes.get(client_id, 0) / total_sample_size
                    if total_sample_size > 0
                    else 1.0 / len(all_clients)
                )
                writer.writerow(
                    [
                        self.global_round + 1,
                        client_id,
                        self.client_model_buffer["curr_round"][client_id],
                        staleness[client_id],
                        round(float(p_k), 8),
                    ]
                )

    def _log_detail(self, message: str) -> None:
        if self._fedqueue_log_paths is None:
            return
        with open(self._fedqueue_log_paths["detail"], "a") as f:
            f.write(message + "\n")

    def clean_up(self) -> None:
        if self.timer_record is not None:
            self.timer_record.cancel()
            self.timer_record = None
