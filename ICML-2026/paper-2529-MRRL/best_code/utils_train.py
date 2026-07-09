import csv
import json
import os
import platform
import socket
import subprocess

import lightning as L
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import psutil
import statsmodels.api as sm
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback


def random_orthogonal_qr(n, device=None, dtype=torch.float32, generator=None):
    """
    Generate random orthogonal matrix using Householder reflections
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random normal matrix
    H = torch.randn(n, n, device=device, dtype=dtype, generator=generator)

    # QR decomposition
    Q, R = torch.linalg.qr(H)

    # Ensure determinant is positive (optional)
    d = torch.diag(R)
    signs = torch.sign(d)
    Q = Q * signs.unsqueeze(0)

    return Q


def random_orthogonal_cayley(
    n, max_norm=1.0, device=None, dtype=torch.float32, generator=None
):
    """
    Generate orthogonal matrix with bounded entries using Cayley transform
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate skew-symmetric matrix
    A = torch.randn(n, n, device=device, dtype=dtype, generator=generator)
    A = (A - A.T) / 2

    # Scale to control the norm
    A = A * max_norm / torch.norm(A)

    # Cayley transform: Q = (I - A) @ inv(I + A)
    D = torch.eye(n, device=device, dtype=dtype)
    Q = torch.linalg.solve(D + A, D - A)

    return Q


def random_normal(n, device=None, dtype=torch.float32, generator=None):
    """
    Generate a random invertible matrix using uniform sampling Unif(0,1).
    Checks for invertibility and regenerates if determinant is too close to zero.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        Q = torch.randn(n, n, device=device, dtype=dtype, generator=generator)
        if torch.abs(torch.linalg.det(Q)) > 1e-6:
            return Q


def random_normal_clamped(
    n,
    max_cond=5.0,
    method="clip",
    device=None,
    dtype=torch.float32,
    generator=None,
):
    """
    Generates a random matrix where the ratio of largest to smallest
    singular value is exactly or approximately 'max_cond'.
    method: 'exact' for exact conditioning, 'clip' to only limit bad values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate Random Matrix
    A = torch.randn(n, n, device=device, dtype=dtype, generator=generator)

    # 2. Decompose (SVD)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # 3. Force the Condition Number

    # Option A: Linearly space singular values (Uniform scaling across dims)
    if method == "exact":
        S_new = torch.linspace(max_cond, 1.0, steps=n, device=device)

    # Option B: Only clip the bad values
    elif method == "clip":
        S_new = torch.clamp(S, min=S[0] / max_cond)

    # 4. Reconstruct
    Q = U @ torch.diag(S_new) @ Vh

    return Q


class IVTrackingCallback(L.Callback):
    def __init__(
        self, dataloader, iv_wrapper_func, z_regex="^hW", every_n_epochs=1
    ):
        super().__init__()
        self.dataloader = dataloader
        self.iv_wrapper = iv_wrapper_func
        self.z_regex = z_regex
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        pl_module.eval()

        try:
            with torch.no_grad():
                dfs = pl_module.encode_dataset(self.dataloader)

            pl_module.train()

            df = pd.concat(dfs, axis=0)

            # 1. Prepare Data
            Y = df["Y"]
            D = df.filter(regex="^D")
            Z = df.filter(regex=self.z_regex)

            # 2. Run IV Estimation (Get Theta)
            theta_est, _, _ = self.iv_wrapper(
                Y=Y, D=D, Z=Z, method="2sls", fit_intercept=True
            )

            # 3. Calculate First-Stage F-Statistic (Strength)
            f_stats = []

            # Iterate over each treatment column in D (e.g., D_0, D_1)
            for col in D.columns:
                Z_with_const = sm.add_constant(Z)
                model_first_stage = sm.OLS(D[col], Z_with_const).fit()
                f_stats.append(model_first_stage.fvalue)

            # 4. Logging
            theta_est = (
                theta_est.ravel()
                if hasattr(theta_est, "ravel")
                else [theta_est]
            )
            for i, val in enumerate(theta_est):
                pl_module.log(f"iv/est_theta_{i}", val, on_epoch=True)

            # Log F-Statistics (Strength)
            for i, f_val in enumerate(f_stats):
                pl_module.log(f"iv/f_stat_D{i}", f_val, on_epoch=True)

        except Exception as e:
            print(f"[IV Callback Error] Epoch {trainer.current_epoch}: {e}")
            pl_module.train()


class SaveValMetricsCallback(Callback):
    def __init__(self, out_dir: str, metric_prefix: str = "val"):
        """
        out_dir: directory to save CSV (created if missing).
        metric_prefix: only metrics with keys containing this substring are saved.
        """
        self.out_dir = out_dir
        self.metric_prefix = metric_prefix
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(
            self.out_dir, f"{self.metric_prefix}_metrics.csv"
        )
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["epoch", "metric", "value", "timestamp"])

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = dict(trainer.callback_metrics or {})
        epoch = (
            int(trainer.current_epoch)
            if hasattr(trainer, "current_epoch")
            else -1
        )
        import time

        ts = time.time()
        rows = []
        for k, v in metrics.items():
            if self.metric_prefix in str(k):  # allow "val", "val_loss", etc.
                try:
                    if hasattr(v, "item"):
                        val = v.item()
                    else:
                        val = float(v)
                except Exception:
                    val = str(v)
                rows.append([epoch, k, val, ts])

        if rows:
            with open(self.csv_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerows(rows)


class RuntimeInfoCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        out_dir = HydraConfig.get().runtime.output_dir
        info = {
            "num_nodes": trainer.num_nodes,
            "devices": trainer.num_devices,
            "accelerator": trainer.accelerator.__class__.__name__,
            "strategy": trainer.strategy.__class__.__name__,
            "precision": trainer.precision,
        }
        with open(os.path.join(out_dir, "runtime_info.json"), "w") as f:
            json.dump(info, f, indent=2)


def get_hardware_info():
    info = {}

    # Node / system
    info["hostname"] = socket.gethostname()
    info["platform"] = platform.platform()
    info["python_version"] = platform.python_version()

    # CPU
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["cpu_model"] = platform.processor()

    # Memory
    mem = psutil.virtual_memory()
    info["memory_gb"] = round(mem.total / 1e9, 2)

    # GPUs
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count()

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    info["gpus"] = gpus

    # CUDA / driver
    info["cuda_version"] = torch.version.cuda
    try:
        info["nvidia_driver"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
    except Exception:
        info["nvidia_driver"] = None

    return info
