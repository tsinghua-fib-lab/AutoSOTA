"""Tests."""
import os
import shutil
from os import path
import subprocess
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile


def test_rq2_script():
    command = ["python", "main-ase.py", "--method", "baro", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_basic():
    # You can put the code here to a file named test.py
    from RCAEval.e2e.baro import baro
    from RCAEval.utility import download_data, read_data

    # download a sample data to data.csv
    download_data()

    # read data from data.csv
    data = read_data("data.csv")
    anomaly_detected_timestamp = 1692569339

    # perform root cause analysis
    root_causes = baro(data, anomaly_detected_timestamp)["ranks"]

    # print the top 5 root causes
    print("Top 5 root causes:", root_causes[:5])