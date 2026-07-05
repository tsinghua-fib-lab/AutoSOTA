"""File-I/O logger for training runs.

This module only handles persistence of the recorded metrics.
Metric computation lives in the solver classes (`algorithm.base_solver`).
"""

import os
from glob import glob

import pandas as pd


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.records = {}

    def record(self, iteration, metrics):
        self.records[iteration] = metrics

    def _to_dataframe(self):
        return pd.DataFrame.from_dict(self.records, orient="index")

    def checkpoint(self, iteration):
        for old in glob(f"{self.output_dir}/exploitability_log_checkpoint_*.csv"):
            os.remove(old)
        self._to_dataframe().to_csv(f"{self.output_dir}/exploitability_log_checkpoint_{iteration}.csv")

    def save_metrics(self):
        self._to_dataframe().to_csv(f"{self.output_dir}/exploitability_log.csv")
        for old in glob(f"{self.output_dir}/exploitability_log_checkpoint_*.csv"):
            os.remove(old)
