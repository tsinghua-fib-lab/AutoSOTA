"""Tests."""
import os
import shutil
from os import path

import numpy as np
import pandas as pd
import pytest
import tempfile

from baro.anomaly_detection import nsigma, bocpd
from baro.root_cause_analysis import robust_scorer
from baro.utility import (
    visualize_metrics,
    download_data,
    drop_constant,
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)

def test_download_dataset():
    """Test download dataset."""
    local_path = tempfile.NamedTemporaryFile().name
    download_online_boutique_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)
    
    local_path = tempfile.NamedTemporaryFile().name
    download_sock_shop_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)    
    
    local_path = tempfile.NamedTemporaryFile().name
    download_train_ticket_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)


def test_nsigma_basic():
    """Test nsigma basic."""
    time_col = np.arange(0, 1000, 1)
    normal_latency = np.random.normal(3, 1, 500)
    normal_latency = np.clip(normal_latency, 1, 5)    
    
    abnormal_latency = np.random.normal(50, 1, 500)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # make df from time_col and latency
    df = pd.DataFrame({'time': time_col, 'latency': latency})
    anomalies = nsigma(df, startsfrom=300)
    assert abs(anomalies[0] - 500) < 10, anomalies

     
def test_bocpd_basic():
    """Test bocpd basic."""
    time_col = np.arange(0, 200, 1)
    normal_latency = np.random.normal(3, 1, 100)
    normal_latency = np.clip(normal_latency, 1, 5)    
    
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # make df from time_col and latency
    df = pd.DataFrame({'time': time_col, 'latency': latency})
    
    anomalies = bocpd(df)
    assert abs(anomalies[0] - 100) < 10, anomalies

def test_baro():
    """Test BARO end-to-end"""
    local_path = tempfile.NamedTemporaryFile().name
    download_data(local_path=local_path)
    df = pd.read_csv(local_path)
    df = df[60:660].reset_index(drop=True)

    # select latency and error rate
    time_col = pd.Series(range(df.shape[0])) # df["time"]
    selected_cols = [c for c in df.columns if "latency-50" in c or "error" in c]
    selected_df = drop_constant(df[selected_cols])
    selected_df.insert(0, "time", time_col)
    
    # anomaly detection
    anomalies = bocpd(selected_df)
    
    # root cause analysis
    ranks = robust_scorer(df, anomalies=anomalies)["ranks"]
    
    # check if cartservice is in the top 5
    service_ranks = [r.split("_")[0] for r in ranks]
    assert "cartservice" in service_ranks[:5]
    