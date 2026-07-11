# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class Logger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.console = Console()

        handler = RichHandler(console=self.console, rich_tracebacks=True, show_time=False)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M")
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def log_metric(self, name: str, value: float, **kwargs):
        from .globals import get_experiment_config

        experiment_config = get_experiment_config()
        if experiment_config is None:
            raise ValueError("Experiment config not set. Call set_experiment_config() first.")
        log_dir = Path(".logs") / experiment_config.id
        log_dir.mkdir(parents=True, exist_ok=True)

        config_file = log_dir / "config.json"
        if not config_file.exists():
            with open(config_file, "w") as f:
                json.dump(experiment_config.to_dict(), f, indent=2)

        metrics_file = log_dir / "metrics.jsonl"
        metric_entry = {"name": name, "value": value, **kwargs}
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")

    def log_done(self):
        from .globals import get_experiment_config

        experiment_config = get_experiment_config()
        if experiment_config is None:
            raise ValueError("Experiment config not set. Call set_experiment_config() first.")
        log_dir = Path(".logs") / experiment_config.id
        log_dir.mkdir(parents=True, exist_ok=True)
        done_file = log_dir / "done"
        done_file.touch()


def get_logger(name: str) -> Logger:
    return Logger(name)
