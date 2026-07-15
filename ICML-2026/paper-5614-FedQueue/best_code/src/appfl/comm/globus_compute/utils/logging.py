"""
[Deprecated] This module is deprecated and will be removed in the future.
Loggers for supporting Globus Compute-based federated learning experiments.
"""

import os
import csv
import json
import time
import torch
import logging
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

TIME_STR = "%m%d%y_%H%M%S"


class GlobusComputeEvalLogger:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.test_results = {"clients": None, "server": None}
        self.val_results = {"clients": [], "server": []}
        self.main_logger = GlobusComputeServerLogger.get_logger()

    def __format(self, key, val):
        """Pretty format for the key-value pair."""
        c = "%12s:" % key
        if type(val) is int:
            c += "%8s" % val
        else:
            c += "%10.3f" % val
        return c

    def log_info_client_results(self, results, result_type=""):
        """
        Log the client results into console and the output file.

        Inputs:
            - results: dict(dict): `{client_name: {result_name: result}}`
            - result_type: str: type of the results
        """
        for cli_name in results:
            c = "[%8s] %15s" % (result_type, cli_name)
            for k in results[cli_name]:
                c += self.__format(k, results[cli_name][k])
            self.main_logger.info(c)

    def log_info_server_results(self, results, result_type=""):
        """
        Log the server results into console and the output file.

        Inputs:
            - results: dict: `{result_name: result}`
            - result_type: str: type of the results
        """
        c = "[%8s] %20s" % (result_type, "fl_server")
        for k in results:
            c += self.__format(k, results[k])
        self.main_logger.info(c)

    def log_client_testing(self, results):
        """Log the clients testing results."""
        rs = {
            self.cfg.clients[client_idx].name: results[client_idx]
            for client_idx in results
        }
        self.test_results["clients"] = rs
        self.log_info_client_results(rs, "CLI-TEST")

    def log_server_testing(self, results):
        """Log the server testing results."""
        self.test_results["server"] = results
        self.log_info_server_results(results, "SER-TEST")

    def log_client_validation(self, results, step):
        """Log the clients validation results."""
        rs = {
            self.cfg.clients[client_idx].name: {
                "step": step,
                **results[client_idx],
            }
            for client_idx in results
        }
        self.val_results["clients"].append(rs)
        self.log_info_client_results(rs, "CLI-VAL")

    def log_server_validation(self, results, step):
        """Log the server validation results."""
        rs = {"step": step, **results}
        self.val_results["server"].append(rs)
        self.log_info_server_results(rs, "SER-VAL")

    def save_log(self, output_file):
        """Save the validation results and testing results in an output file for further analysis or visualization."""
        out_dict = {"val": self.val_results, "test": self.test_results}
        if self.cfg.load_model:
            out_dict["checkpoint_dirname"] = self.cfg.load_model_dirname
            out_dict["checkpoint_filename"] = self.cfg.load_model_filename
        with open(output_file, "w") as f:
            json.dump(out_dict, f, indent=2)


class GlobusComputeServerLogger:
    """A global and singleton logger for a globus compute-based federated learning server."""

    __logger = None

    @classmethod
    def config_logger(cls, cfg: DictConfig):
        """Configure and create a singleton server logger for all logging tasks for the federated learning server."""
        # Prepare for the logging directory
        dir = cfg.server.output_dir
        cls.__cleanup_dir(dir)
        tb_dir = os.path.join(dir, "tensorboard")
        cls.__cleanup_dir(tb_dir)
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

        # Logging format such as [2023-02-20 12:12:12,643 INFO]: message
        fmt = logging.Formatter("[%(asctime)s %(levelname)-4s]: %(message)s")
        log_fname = os.path.join(dir, "log_globus_compute_server.log")

        # Prepare the logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_fname)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        c_handler.setFormatter(fmt)
        f_handler.setFormatter(fmt)
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        # Instantiate method
        class_logger = GlobusComputeServerLogger()
        class_logger.logger = logger
        class_logger.dir = dir
        if cfg.use_tensorboard:
            from tensorboardX import SummaryWriter

            class_logger.writer = SummaryWriter(tb_dir)

        # Initialize eval logger
        cls.__logger = class_logger
        class_logger.eval_logger = GlobusComputeEvalLogger(cfg)

    @classmethod
    def get_logger(cls):
        """Get the singleton server logger."""
        if cls.__logger is None:
            raise RuntimeError("Need to configure logger first")
        return cls.__logger.logger

    @classmethod
    def get_eval_logger(cls):
        """Get the logger for logging the evaluation or testing results."""
        if cls.__logger is None:
            raise RuntimeError("Need to configure logger first")
        return cls.__logger.eval_logger

    @classmethod
    def get_tensorboard_writer(cls):
        """Get the tensorboard writer."""
        if cls.__logger.writer is None:
            raise Exception("Tensorboard X writer need to be configured first")
        return cls.__logger.writer

    @classmethod
    def save_globus_compute_log(cls, cfg):
        """Save the globus compute task information into a csv file."""
        header = ["timestamp", "task_name", "client_name", "status", "execution_time"]
        lgg = cls.__logger
        # Save csv file
        with open(os.path.join(lgg.dir, "log_glosbus_compute.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for tlog in cfg.logging_tasks:
                writer.writerow(
                    [
                        datetime.fromtimestamp(tlog.start_time),
                        tlog.task_name,
                        cfg.clients[tlog.client_idx].name,
                        "success" if tlog.success else "failed",
                        "%.02f" % (tlog.end_time - tlog.start_time),
                    ]
                )

        # Save json file
        with open(os.path.join(lgg.dir, "log_globus_compute.yaml"), "w") as f:
            log_tasks = []
            for tlog in cfg.logging_tasks:
                l_tsk = {}
                l_tsk["task_name"] = tlog.task_name
                l_tsk["endpoint"] = cfg.clients[tlog.client_idx].name
                l_tsk["start_at"] = str(datetime.fromtimestamp(tlog.start_time))
                l_tsk["end_at"] = str(datetime.fromtimestamp(tlog.end_time))
                if "events" in tlog.log:
                    l_tsk["events"] = dict(tlog.log["events"])
                if "timing" in tlog.log:
                    l_tsk["timing"] = dict(tlog.log["timing"])
                if "info" in tlog.log:
                    l_tsk["info"] = dict(tlog.log["info"])
                log_tasks.append(l_tsk)
            f.write(OmegaConf.to_yaml(log_tasks))

        # Save evaluation and testing log
        lgg.eval_logger.save_log(os.path.join(lgg.dir, "log_eval.json"))

    @classmethod
    def log_client_data_info(cls, cfg, data_info_at_client):
        """
        Log the information about the client data distribution in a table.

        Inputs:
            - data_info_at_client: dict(dict): `{client_idx: {mode: num_data}}`
        """
        mode = list(data_info_at_client[0].keys())
        logger = cls.get_logger()
        table_line = "|" + "-" * 25 + "|" + ("-" * 10 + "|") * len(mode)
        logger.info(table_line)
        title_line = ("|%25s|" + "%10s|" * len(mode)) % ("client name ", *mode)
        logger.info(title_line)
        logger.info(table_line)
        for client_idx in range(cfg.num_clients):
            client_line = "|%25s|" % cfg.clients[client_idx].name
            for mode in data_info_at_client[client_idx]:
                client_line += "%10s|" % (data_info_at_client[client_idx][mode])
            logger.info(client_line)
        logger.info(table_line)

    @classmethod
    def log_server_data_info(cls, data_info_at_server):
        """
        Log the information about the server data distribution (if any) in a table.

        Inputs:
            - data_info_at_server: dict: `{mode: num_data}`
        """
        mode = list(data_info_at_server.keys())
        logger = cls.get_logger()
        table_line = "|" + "-" * 10 + "|" + ("-" * 10 + "|") * len(mode)
        logger.info(table_line)
        title_line = ("|%10s|" + "%10s|" * len(mode)) % (" ", *mode)
        logger.info(title_line)
        logger.info(table_line)
        server_line = "|%10s|" % "server"
        for k in data_info_at_server:
            server_line += "%10s|" % (data_info_at_server[k])
        logger.info(server_line)
        logger.info(table_line)

    @classmethod
    def save_checkpoint(cls, step, state_dict):
        """Save the model check point for a certain global step."""
        lgg = cls.__logger
        file = os.path.join(lgg.dir, "checkpoint_%d.pt" % step)
        torch.save(state_dict, file)

    @staticmethod
    def __cleanup_dir(dir):
        """Clean up a certain directory if it exists."""
        if os.path.exists(dir) and os.path.isdir(dir):
            files = os.listdir(dir)
            for file in files:
                file_path = os.path.join(dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)


class GlobusComputeClientLogger:
    """A logger for globus compute-based federated learning client."""

    def __init__(self) -> None:
        self.info = {}
        self.events = {}
        self.timing = {}
        self.timer_stack = []  # A stack (first-in-last-out) for timing events

    def to_dict(self):
        return {"events": self.events, "timing": self.timing, "info": self.info}

    def __get_step_str(self, step):
        return "Epoch %d" % (step + 1)

    def add_info(self, name: str, value, step=None):
        if step is None:
            self.info[name] = value
        else:
            step = self.__get_step_str(step)
            if step not in self.info:
                self.info[step] = {}
            self.info[step][name] = value

    def mark_event(self, name: str, step=None):
        """Record the time for a certain event."""
        time_stp = str(datetime.now())
        if step is None:
            self.events[name] = time_stp
        else:
            step = self.__get_step_str(step)
            if step not in self.events:
                self.events[step] = {}
            self.events[step][name] = time_stp

    def start_timer(self, name: str, step=None):
        """Start timing a certain event `name`."""
        if step is not None:
            step = self.__get_step_str(step)
            name = f"{step} {name}"
        if len(self.timer_stack) == 0:
            self.timing[name] = time.time()
        else:
            event_dict = self.timing
            for i, event in enumerate(self.timer_stack):
                if i == len(self.timer_stack) - 1:
                    if type(event_dict[event]) is not dict:
                        event_dict[event] = {"total_time": event_dict[event]}
                    event_dict[event][name] = time.time()
                else:
                    event_dict = event_dict[event]
        self.timer_stack.append(name)

    def stop_timer(self, name: str, step=None):
        """End timing a certain event `name`."""
        if step is not None:
            step = self.__get_step_str(step)
            name = f"{step} {name}"
        assert name == self.timer_stack[-1], (
            f"Please stop timing event in order! {name} {self.timer_stack}"
        )
        event_dict = self.timing
        for i, event in enumerate(self.timer_stack):
            if i == len(self.timer_stack) - 1:
                if type(event_dict[event]) is not dict:
                    event_dict[event] = round(time.time() - event_dict[event], 3)
                else:
                    event_dict[event]["total_time"] = round(
                        time.time() - event_dict[event]["total_time"], 3
                    )
            else:
                event_dict = event_dict[event]
        self.timer_stack.pop()
