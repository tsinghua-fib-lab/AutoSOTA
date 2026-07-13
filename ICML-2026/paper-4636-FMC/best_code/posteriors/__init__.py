import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict, Any
import json

import ot
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.nn as nn
from lampe.inference import NPE
from pyro.distributions import Distribution
from torch import Tensor
from zuko.distributions import DiagNormal
from zuko.flows import MAF, NSF, Flow, UnconditionalDistribution
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows.neural import UMNN

from flow_matching.torch_flow import FlowMatching
from simulator.base import Simulator
from utils.metrics import (
    MMD,
    acauc,
    acauc_rope,
    classifier_two_samples_test,
    classifier_two_samples_test_torch,
    mse,
    stein_discrepancy,
)
from utils.networks import get_embedding_network
from utils.timing import HierarchicalTimer


def _timed_flow_sample(flow, source, cond, device, timer, label, **kwargs):
    """Sample from a FlowMatching model, optionally timing the operation."""
    if timer:
        with timer.time_operation("sampling", label) as ctx:
            samples, nfe = flow.sample(source, cond, device, return_nfe=True, **kwargs)
            ctx.add_nfe(nfe)
    else:
        samples = flow.sample(source, cond, device, **kwargs)
    return samples


# =============================================================================
# Model Configuration (for clean serialization)
# =============================================================================

@dataclass
class EstimatorConfig:
    """Configuration for Estimator model.

    This dataclass captures all parameters needed to reconstruct an Estimator.
    It can be serialized to JSON for checkpoint storage.
    """
    task_name: str
    theta_dim: Tuple[int, ...]
    cond_dim: Tuple[int, ...]
    density_estimator: str = "nsf"
    embedding_net: Dict[str, Any] = field(default_factory=dict)
    npe_params: Dict[str, Any] = field(default_factory=dict)
    rescale_mode: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "task_name": self.task_name,
            "theta_dim": list(self.theta_dim),
            "cond_dim": list(self.cond_dim),
            "density_estimator": self.density_estimator,
            "embedding_net": self.embedding_net,
            "npe_params": self.npe_params,
            "rescale_mode": self.rescale_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstimatorConfig":
        """Create config from dictionary."""
        return cls(
            task_name=data["task_name"],
            theta_dim=tuple(data["theta_dim"]),
            cond_dim=tuple(data["cond_dim"]),
            density_estimator=data.get("density_estimator", "nsf"),
            embedding_net=data.get("embedding_net", {}),
            npe_params=data.get("npe_params", {}),
            rescale_mode=data.get("rescale_mode", "none"),
        )

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EstimatorConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class Posterior(nn.Module, ABC):
    def __init__(self, name, theta_dim, obs_dim):
        super().__init__()
        self.name = name
        self._theta_dim = theta_dim
        self._obs_dim = obs_dim

    @property
    def theta_dim(self):
        return self._theta_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the log probability of theta given x."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, x: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        """
        Sample from the posterior given x.
        Args:
            x: Observed data, shape (batch_size, dim).
            nsamples: Number of samples to draw.

        Returns:
            samples: Samples from the posterior, shape (nsamples, batch_size, dim).
        """
        raise NotImplementedError

    def evaluate_metrics(
        self,
        y: torch.Tensor,
        theta: torch.Tensor,
        device: torch.device,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> dict:
        """
        Evaluate the metrics for the posterior.

        Args:
            y: Observed data, shape (nobs, dim).
            theta: GT Parameters, shape (nobs, dim) ie one sample per observation.
            save_path: Path to save the results.
        """
        ntest = theta.shape[0]
        compute_lpp = kwargs.get("compute_lpp", True)
        compute_c2st = kwargs.get("compute_c2st", True)
        compute_wasserstein = kwargs.get("compute_wasserstein", True)
        compute_mmd = kwargs.get("compute_mmd", True)
        compute_acauc = kwargs.get("compute_acauc", True)
        compute_mse = kwargs.get("compute_mse", False)
        simulator: Optional[Simulator] = kwargs.get("simulator", None)
        with torch.no_grad():
            # embedding_network.to(device)
            # y_emb = embedding_network(y.to(device)).cpu()  # shape (nobs, emb_dim)
            # embedding_network.to("cpu")  # Move back to CPU to save memory
            y_emb = y.flatten(start_dim=1)
        if simulator is not None and simulator.name in ["wind_tunnel", "pendulum"]:
            out_dim = 10
            emb_net = "conv1d_v2" if simulator.name == "wind_tunnel" else "conv1d"
            model_kwargs = {
                "emb_net": emb_net,
                "theta_dim": theta.shape[1],
                "x_dim": y_emb.shape[1:],
                "out_dim": out_dim,
            }
            training_kwargs = {
                "epochs": 150,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
            }
        elif simulator is not None and simulator.name == "light_tunnel":
            out_dim = 10
            emb_net = "conv2d"
            image_size = simulator.obs_dim
            model_kwargs = {
                "emb_net": emb_net,
                "theta_dim": theta.shape[1],
                "x_dim": image_size,
                "out_dim": out_dim,
                "image_size": image_size,
            }
            training_kwargs = {
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
            }
        else:
            model_kwargs = None
            training_kwargs = {
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
            }
        m = kwargs.get("m", 100)  # number of samples for acauc
        metrics = {}
        if save_path is not None and kwargs.get("load", False):
            try:
                theta_pred = torch.load(save_path / f"theta_pred_{ntest}.pt")
            except FileNotFoundError:
                theta_pred = self.sample(y, 1, device, **kwargs).squeeze(0)
                torch.save(theta_pred, save_path / f"theta_pred_{ntest}.pt")
        elif save_path is not None:
            theta_pred = self.sample(y, 1, device, **kwargs).squeeze(0)
            torch.save(theta_pred, save_path / f"theta_pred_{ntest}.pt")
        else:
            theta_pred = self.sample(y, 1, device, **kwargs).squeeze(0)

        # Compute LPP before moving to CPU (needs model on device)
        if compute_lpp:
            print("\tComputing LPP")
            try:
                metrics["lpp"] = self.log_prob(theta.to(device), y.to(device), **kwargs).mean().item()
            except NotImplementedError:
                print("\t  log_prob not implemented, skipping LPP")
                metrics["lpp"] = float("nan")

        # Move all tensors to CPU for remaining metrics computation
        theta_pred = theta_pred.cpu()
        theta = theta.cpu()
        y_emb = y_emb.cpu()

        if False:  # LPP already computed above
            pass
        # if compute_c2st:
        #     print("\tComputing C2ST")
        #     metrics["c2st"] = classifier_two_samples_test(
        #         theta, theta_pred, n_folds=3, scoring="accuracy"
        #     )
        if compute_c2st:
            print("\tComputing Joint C2ST")
            theta_y_pred = torch.cat([theta_pred, y_emb], dim=1)
            theta_true_y = torch.cat([theta, y_emb], dim=1)
            metrics["joint_c2st"] = classifier_two_samples_test_torch(
                theta_true_y,
                theta_y_pred,
                z_score=True,
                n_folds=3,
                scoring="accuracy",
                model="mlp",
                model_kwargs=model_kwargs,
                training_kwargs=training_kwargs,
            )
            # metrics["joint_c2st"] = classifier_two_samples_test(
            #     theta_true_y, theta_y_pred, z_score=True, n_folds=3, scoring="accuracy"
            # )
        if compute_wasserstein:
            print("\tComputing Wasserstein")
            a, b = (
                torch.ones((theta.shape[0],)) / theta.shape[0],
                torch.ones((theta_pred.shape[0],)) / theta_pred.shape[0],
            )
            M = ot.dist(theta, theta_pred)
            metrics["wasserstein"] = torch.sqrt(ot.emd2(a, b, M)).item()

            print("\tComputing joint Wasserstein")
            theta_y_pred = torch.cat([theta_pred, y_emb], dim=1)
            theta_true_y = torch.cat([theta, y_emb], dim=1)
            a, b = (
                torch.ones((theta_true_y.shape[0],)) / theta_true_y.shape[0],
                torch.ones((theta_y_pred.shape[0],)) / theta_y_pred.shape[0],
            )
            M = ot.dist(theta_true_y, theta_y_pred)
            metrics["joint_wasserstein"] = torch.sqrt(ot.emd2(a, b, M)).item()
        if compute_mmd:
            print("\tComputing MMD")
            metrics["mmd"] = MMD(theta, theta_pred, kernel="rbf").item()
            print("\tComputing joint MMD")
            theta_y_pred = torch.cat([theta_pred, y_emb], dim=1)
            theta_true_y = torch.cat([theta, y_emb], dim=1)
            metrics["joint_mmd"] = MMD(theta_true_y, theta_y_pred, kernel="rbf").item()

        if compute_acauc:
            print("\tComputing ACAUC")
            if save_path is not None:
                try:
                    theta_pred_m = torch.load(save_path / "theta_pred_m.pt")
                except FileNotFoundError:
                    theta_pred_m = self.sample(y, m, device, **kwargs).to("cpu")
                    torch.save(theta_pred_m, save_path / "theta_pred_m.pt")
            else:
                theta_pred_m = self.sample(y, m, device, **kwargs).to(
                    "cpu"
                )  # shape (100, nobs, dim)
            metrics["acauc"] = acauc(theta, theta_pred_m)
            print("\tComputing ACAUC v2")
            metrics["acauc_v2"] = acauc_rope(theta, theta_pred_m)

        if compute_mse:
            print("\tComputing MSE")
            if save_path is not None and kwargs.get("load", False):
                try:
                    theta_pred_m = torch.load(save_path / f"theta_pred_m_{m}_{ntest}.pt")
                except FileNotFoundError:
                    theta_pred_m = self.sample(y, m, device, **kwargs).squeeze(0).cpu()
                    torch.save(theta_pred_m, save_path / f"theta_pred_m_{m}_{ntest}.pt")
            elif save_path is not None:
                theta_pred_m = self.sample(y, m, device, **kwargs).squeeze(0).cpu()
                torch.save(theta_pred, save_path / f"theta_pred_m_{m}_{ntest}.pt")
            else:
                theta_pred_m = self.sample(y, m, device, **kwargs).squeeze(0).cpu()
            # rescale samples for light tunnel
            if simulator is not None and simulator.name == "light_tunnel":
                a = simulator.prior_params["low"]
                b = simulator.prior_params["high"]
                theta_pred_m = (theta_pred_m - a[None, None, :]) / (
                    b[None, None, :] - a[None, None, :]
                )
            metrics["mse"] = mse(theta, theta_pred_m)

        return metrics

    def evaluate_conditional_metrics(
        self,
        y: torch.Tensor,
        theta: torch.Tensor,
        device: torch.device,
        true_dist: Optional[Callable[[Tensor], Distribution]] = None,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> dict:
        """Evaluate the conditional metrics for the posterior.
        Args:
            y: Observed data, shape (nobs,dim).
            theta: GT Parameters, shape (nsamples, nobs ,dim).
            metrics_to_compute: List of metrics to compute.
            simulator: Simulator object to compute ground_truth pdf.
        returns:
            metrics: Dictionary of computed metrics.
        """
        compute_lpp = kwargs.get("compute_lpp", True)
        compute_c2st = kwargs.get("compute_c2st", True)
        compute_stein_discrepancy = kwargs.get("compute_stein_discrepancy", True)
        compute_wasserstein = kwargs.get("compute_wasserstein", True)
        compute_mmd = kwargs.get("compute_mmd", True)

        metrics = {}
        ntest = theta.shape[0]
        nobs = theta.shape[1]
        dim = theta.shape[2]

        if save_path is not None and kwargs.get("load", False):
            try:
                theta_pred = torch.load(save_path / "theta_pred.pt")
            except FileNotFoundError:
                theta_pred = self.sample(y, ntest, device, **kwargs)
                torch.save(theta_pred, save_path / "theta_pred.pt")
        elif save_path is not None:
            theta_pred = self.sample(y, ntest, device, **kwargs)
            torch.save(theta_pred, save_path / "theta_pred.pt")
        else:
            theta_pred = self.sample(y, ntest, device)

        if compute_lpp:
            print("\tComputing LPP")
            lpp = self.log_prob(
                theta.reshape(-1, dim),
                theta_pred.reshape(-1, dim),
            )
            lpps = [lpp[i * ntest : (i + 1) * ntest].mean().item() for i in range(nobs)]
            metrics["lpp"] = lpps

        if compute_c2st:
            print("\tComputing C2ST")
            c2st = [
                classifier_two_samples_test(
                    theta[:, i, :], theta_pred[:, i, :], n_folds=3, scoring="accuracy"
                )
                for i in range(nobs)
            ]
            metrics["c2st"] = c2st

        if compute_mmd:
            print("\tComputing MMD")
            mmd = []
            for i in range(nobs):
                theta_pred_i = theta_pred[:, i, :]
                gt_theta_i = theta[:, i, :]
                mmd.append(MMD(gt_theta_i, theta_pred_i, kernel="rbf").item())
            metrics["mmd"] = mmd

        if compute_wasserstein:
            print("\tComputing Wasserstein")
            ws = []
            for i in range(nobs):
                theta_pred_i = theta_pred[:, i, :]
                gt_theta_i = theta[:, i, :]
                a, b = (
                    torch.ones((ntest,)) / ntest,
                    torch.ones((ntest,)) / ntest,
                )  # uniform distribution on samples
                M = ot.dist(theta_pred_i, gt_theta_i)
                ws.append(torch.sqrt(ot.emd2(a, b, M)).item())
            metrics["wasserstein"] = ws

        if compute_stein_discrepancy:
            print("\tComputing Stein discrepancy")
            sds = []
            if true_dist is None:
                # skip stein discrepancy computation
                print("true_dist must be provided for Stein discrepancy computation")
                return metrics
            for i in range(nobs):
                theta_pred_i = theta_pred[:, i, :]
                y_i = y[i, :].reshape(1, -1).repeat_interleave(ntest, 0)

                def score_fn(theta):
                    log_prob = true_dist(y_i).log_prob(theta)  # log p(θ | y)
                    log_prob.sum().backward()  # Compute gradients
                    theta_grad = theta.grad  # ∇θ log p(θ | y)
                    return theta_grad

                sd = stein_discrepancy(theta_pred_i, score_fn)
                sds.append(sd)
            metrics["stein_discrepancy"] = sds

        return metrics


class UMNNFlow(Flow):
    def __init__(self, features, context, **kwargs):
        integrand_params = kwargs.get("integrand_params", {})
        conditoner_params = kwargs.get("conditioner_params", {})
        embedding_dim = kwargs.get("embedding_dim", None)
        transforms = kwargs.get("ntransform", None)
        neural_nets = [UMNN(embedding_dim, **integrand_params) for _ in range(transforms)]

        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]
        orders = list(map(torch.LongTensor, orders))
        transform = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=orders[i % 2],
                univariate=neural_nets[i],
                shapes=((embedding_dim,), ()),
                **conditoner_params,
            )
            for i in range(transforms)
        ]
        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )
        super().__init__(transform, base)
        pass


class Estimator(nn.Module):
    """Neural Posterior Estimator (NPE) with rescaling support.

    This class wraps a neural posterior estimator with embedding networks and
    data rescaling functionality. Rescaling parameters are stored as part of
    the model state and automatically saved/loaded with checkpoints.

    Important: The forward() method expects ALREADY RESCALED inputs. Rescaling
    should be done explicitly in training loops before calling forward().

    Attributes:
        npe: Neural posterior estimator (normalizing flow)
        embedding_net: Embedding network for observations
        data_rescaler: Rescaler for theta (parameters)
        cond_rescaler: Rescaler for x (observations)
        theta_dim: Dimension of parameters
        cond_dim: Dimension of observations
        obs_dim: Dimension of embedded observations

    Example:
        >>> estimator = Estimator("gaussian", theta.shape[1:], x.shape[1:])
        >>> estimator.set_scales(theta, x, "z_score")
        >>> # In training loop:
        >>> theta_batch, x_batch = estimator.rescale(theta_batch, x_batch)
        >>> loss = criterion(estimator.forward(theta_batch, x_batch))
    """

    def __init__(self, task_name: str, dim: torch.Size, cond_dim: torch.Size, **kwargs):
        super(Estimator, self).__init__()
        density_estimator = kwargs.get("density_estimator", "nsf")
        embedding_net_config = kwargs.get("embedding_net", {})
        npe_params = kwargs.get("npe_params", {})
        rescale_mode = kwargs.get("rescale_mode", "none")

        if len(dim) != 1:
            raise ValueError("dim must be a 1D tensor representing the parameter dimension.")

        # Store configuration for serialization
        self._config = EstimatorConfig(
            task_name=task_name,
            theta_dim=tuple(dim),
            cond_dim=tuple(cond_dim),
            density_estimator=density_estimator,
            embedding_net=embedding_net_config,
            npe_params=npe_params,
            rescale_mode=rescale_mode,
        )

        self.theta_dim = dim
        self.cond_dim = cond_dim
        self.obs_dim = embedding_net_config.get("output_dim", cond_dim[0])
        self.npe = NPE(
            dim[0],
            self.obs_dim,
            build=get_build_fn(task_name, density_estimator, **npe_params),
        )
        self.embedding_net = get_embedding_network(task_name, **embedding_net_config)

        # Initialize rescalers based on mode (will be fitted via set_scales)
        from utils.rescaling import create_rescaler
        self.data_rescaler = create_rescaler(rescale_mode)
        self.cond_rescaler = create_rescaler(rescale_mode)

    def set_rescalers(self, data_rescaler, cond_rescaler):
        """Set rescalers for data and conditioning variables.

        Args:
            data_rescaler: DataRescaler for theta/data
            cond_rescaler: DataRescaler for x/conditioning variables
        """
        from utils.rescaling import DataRescaler

        if not isinstance(data_rescaler, DataRescaler):
            raise TypeError(f"data_rescaler must be DataRescaler, got {type(data_rescaler)}")
        if not isinstance(cond_rescaler, DataRescaler):
            raise TypeError(f"cond_rescaler must be DataRescaler, got {type(cond_rescaler)}")

        self.data_rescaler = data_rescaler
        self.cond_rescaler = cond_rescaler

    def set_scales(self, data: Tensor, cond: Tensor, rescale_name: str):
        """Create and fit rescalers for data and conditioning variables.

        Args:
            data: Training data (theta)
            cond: Conditioning variables (x)
            rescale_name: Type of rescaling ('none', 'z_score', 'whiten')
        """
        from utils.rescaling import create_rescaler

        # Create and fit rescalers
        if rescale_name == "z_score":
            print("Using z_score rescaling.")
        data_rescaler = create_rescaler(rescale_name)
        cond_rescaler = create_rescaler(rescale_name)

        data_rescaler.fit(data)
        cond_rescaler.fit(cond)

        self.set_rescalers(data_rescaler, cond_rescaler)

        # Update config with rescale mode
        self._config.rescale_mode = rescale_name

    # =========================================================================
    # Serialization Methods (Clean API)
    # =========================================================================

    def get_config(self) -> EstimatorConfig:
        """Get model configuration for serialization."""
        return self._config

    @classmethod
    def from_config(cls, config: EstimatorConfig) -> "Estimator":
        """Create Estimator from configuration.

        Args:
            config: EstimatorConfig instance

        Returns:
            New Estimator with architecture matching config
        """
        return cls(
            task_name=config.task_name,
            dim=torch.Size(config.theta_dim),
            cond_dim=torch.Size(config.cond_dim),
            density_estimator=config.density_estimator,
            embedding_net=config.embedding_net,
            npe_params=config.npe_params,
            rescale_mode=config.rescale_mode,
        )

    def save(self, path: Path) -> None:
        """Save model to directory.

        Creates:
            path/config.json - Model configuration
            path/weights.pth - Model weights (state_dict)

        Args:
            path: Directory to save model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self._config.save(path / "config.json")

        # Save weights (state_dict with weights_only compatible format)
        torch.save(self.state_dict(), path / "weights.pth")

    @classmethod
    def load(cls, path: Path, device: torch.device = torch.device("cpu")) -> "Estimator":
        """Load model from directory.

        Args:
            path: Directory containing config.json and weights.pth
            device: Device to load model to

        Returns:
            Loaded Estimator
        """
        path = Path(path)

        # Load config and create model
        config = EstimatorConfig.load(path / "config.json")
        model = cls.from_config(config)

        # Load weights
        state_dict = torch.load(path / "weights.pth", map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)

        return model

    def _auto_rescale(self, data: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rescaling transformations automatically.

        Args:
            data: Target data (theta)
            cond: Conditioning data (x/y)

        Returns:
            (rescaled_data, rescaled_cond)
        """
        data = self.data_rescaler.transform(data)
        cond = self.cond_rescaler.transform(cond)
        return data, cond

    def rescale(self, data: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rescaling transformation to data and condition.

        Deprecated: Use compute_loss() which handles rescaling internally.
        """
        return self._auto_rescale(data, cond)

    def scale(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply inverse rescaling transformation to data and condition.

        Uses new DataRescaler infrastructure.
        """
        x = self.data_rescaler.inverse_transform(x)
        cond = self.cond_rescaler.inverse_transform(cond)
        return x, cond

    def embedding(self, x: Tensor) -> Tensor:
        """Embed the input data."""
        if x.shape[1] == self.obs_dim:
            return x
        else:
            _, x = self.rescale(torch.zeros(1, self.theta_dim), x)
            return self.embedding_net(x)

    def forward(self, theta, x):
        """Forward pass through the estimator.

        Note: This method expects ALREADY RESCALED inputs. For training,
        use compute_loss() which handles rescaling automatically.

        Args:
            theta: Parameter samples (already rescaled)
            x: Observations (already rescaled)

        Returns:
            Output from the neural posterior estimator
        """
        return self.npe(theta, self.embedding_net(x))

    def compute_loss(self, theta: Tensor, x: Tensor) -> Tensor:
        """Compute NPE loss with automatic rescaling.

        Implements: -E[log p(θ|x)]

        Args:
            theta: Parameter samples, shape (batch, ...)
            x: Observations, shape (batch, ...)

        Returns:
            Scalar loss (negative log likelihood mean)
        """
        # Automatic rescaling
        theta_scaled, x_scaled = self._auto_rescale(theta, x)

        # Compute negative log likelihood
        return -self.forward(theta_scaled, x_scaled).mean()

    def _flow(self, x):
        """Get the flow conditioned on x.

        IMPORTANT: x should be already rescaled before calling this method.

        Args:
            x: Observations (already rescaled)

        Returns:
            Conditional flow for sampling theta
        """
        return self.npe.flow(self.embedding_net(x))

    def sample(self, x, nsamples, device, **kwargs):
        self.to(device)  # Move to device
        _, x = self.rescale(torch.zeros(1, *self.theta_dim).to(x.device), x)
        space = kwargs.get("space", "data")
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)

        timer_ctx = timer.time_operation("sampling", "npe_flow_sample") if timer else nullcontext()

        if kwargs.get("batch_size", None) is not None:
            samples = []
            bs = kwargs["batch_size"]
            N = x.shape[0]  # number of observations
            for start in range(0, N, bs):
                with torch.no_grad():
                    end = min(start + bs, N)
                    x_batch = x[start:end].to(device)
                    with timer_ctx:
                        if space == "latent":
                            samples_batch = self.npe.flow(x_batch).sample((nsamples,))
                        elif space == "data":
                            samples_batch = self._flow(x_batch).sample(
                                (nsamples,)
                            )  # shape (nsamples, end-start, *self.theta_dim)
                    samples.append(samples_batch.cpu())
            samples = torch.cat(samples, dim=1)  # shape (nsamples, N ,*self.theta_dim)
        else:
            with torch.no_grad():
                # Sample from the flow
                # x is shape (N, *self.obs_dim)
                # nsamples is the number of samples to draw
                # samples will be of shape (nsamples, N, *self.theta_dim)
                with timer_ctx:
                    if space == "latent":
                        samples = self.npe.flow(x.to(device)).sample((nsamples,)).to("cpu")
                    elif space == "data":
                        samples = (
                            self._flow(x.to(device)).sample((nsamples,)).to("cpu")
                        )  # shape (nsamples, N, *self.theta_dim)
        samples, _ = self.scale(samples.to(x.device), x)
        # print("Samples shape :", samples.shape)
        return samples

    def log_prob(self, theta, x, **kwargs):
        theta, x = self.rescale(theta, x)
        return self.npe.flow(self.embedding_net(x)).log_prob(theta)

    def state_dict(self, *args, **kwargs):
        """Get model state dict.

        Note: For new code, prefer using save()/load() methods which handle
        both config and weights properly.

        The state_dict includes rescaler state as submodules (automatically
        handled by PyTorch since rescalers are nn.Module instances).
        """
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict.

        Note: For new code, prefer using save()/load() methods which handle
        both config and weights properly.
        """
        super().load_state_dict(state_dict, strict=strict)


class DirectPosteriorEstimator(Posterior):
    def __init__(self, name: str, posterior: Estimator):
        super().__init__(name, posterior.theta_dim, posterior.obs_dim)
        self.posterior = posterior

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)
        return self.posterior.sample(
            y, nsamples, device, batch_size=kwargs.get("batch_size", None), timer=timer
        )

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.posterior.log_prob(theta, y)


class DirectFlowMatchingPosterior(Posterior):
    def __init__(
        self,
        name: str,
        flow: FlowMatching,
    ):
        super().__init__(name, flow.dim, flow.cond_dim)
        self.flow = flow

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)

        self.flow.to(device)  # Move to device
        source = self.flow.sample_base(y, nsamples)  # shape (nsamples, N, *obs_dim)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)

        num_steps = kwargs.get('num_steps', self.flow.num_steps)
        source_flat = source.reshape(-1, *source.shape[2:])  # shape (nsamples * N, *obs_dim)
        cond_flat = cond.reshape(-1, *cond.shape[2:])  # shape (nsamples * N, *obs_dim)
        samples = _timed_flow_sample(
            self.flow, source_flat, cond_flat, device, timer, "flow_integration",
            only_last=True, num_steps=num_steps,
        )

        return samples.reshape(nsamples, y.shape[0], *self.theta_dim)

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(theta|y) via instantaneous change of variables.

        Delegates to FlowMatching.log_prob which solves the reverse ODE
        from t=1 to t=0 while tracking divergence.

        Args:
            theta: Parameters, shape (batch, theta_dim)
            y: Observations, shape (batch, obs_dim)
            **kwargs: Passed to FlowMatching.log_prob (num_steps, exact_trace, etc.)

        Returns:
            log_probs: shape (batch,)
        """
        device = next(self.flow.parameters()).device
        return self.flow.log_prob(
            theta.to(device),
            y.to(device),
            device,
            num_steps=kwargs.get("num_steps", None),
            exact_trace=kwargs.get("exact_trace", True),
            n_hutchinson_probes=kwargs.get("n_hutchinson_probes", 1),
        )


class FlowMatchingPosterior(Posterior):
    def __init__(
        self,
        name: str,
        posterior: Union[Estimator, DirectFlowMatchingPosterior],
        flow: FlowMatching,
        theta_dim,
        obs_dim,
    ):
        super().__init__(name, theta_dim, obs_dim)
        self.denoiser = flow
        self.npe = posterior

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)

        self.denoiser.to(device)  # Move to device
        source = self.denoiser.sample_base(y, nsamples)  # shape (nsamples, N, obs_dim)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)

        # Stage 1: Denoiser flow sampling
        num_steps = kwargs.get('num_steps', self.denoiser.num_steps)
        source_flat = source.reshape(-1, *source.shape[2:])  # shape (nsamples * N, *obs_dim)
        cond_flat = cond.reshape(-1, *cond.shape[2:])  # shape (nsamples * N, *obs_dim)
        x_tilde = _timed_flow_sample(
            self.denoiser, source_flat, cond_flat, device, timer, "denoiser_flow",
            only_last=True, num_steps=num_steps,
        )

        # Stage 2: NPE sampling (pass timer to support recursive timing)
        npe_ctx = timer.time_operation("sampling", "npe_sample") if timer else nullcontext()
        with npe_ctx:
            samples = self.npe.sample(x_tilde, 1, device, timer=timer)

        return samples.squeeze(0).reshape(nsamples, y.shape[0], *self.theta_dim)

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        ntraj = kwargs.get("ntraj", 1)
        device = y.device
        source = (
            self.denoiser.sample_base(y, ntraj).transpose(0, 1).to(device)
        )  # shape (ntraj, N ,obs_dim)
        source = source.reshape(-1, source.shape[-1])  # shape (ntraj * N, obs_dim)
        cond = y.repeat_interleave(ntraj, dim=0)  # shape (ntraj * N, obs_dim)
        x_tilde = self.denoiser.sample(
            source, cond, device, only_last=True
        )
        theta = theta.repeat_interleave(ntraj, 0)
        self.npe.to("cpu")  # To avoid GPU memory issues
        pdf = torch.exp(self.npe.log_prob(theta.cpu(), x_tilde))
        self.npe.to(device)  # Move back to the original device
        # resize to (ntraj, N)
        lpp = torch.log(pdf.reshape(ntraj, -1).mean(dim=0))
        return lpp


class DualFlowPosteriorEstimator(Posterior):
    def __init__(
        self,
        name: str,
        base_dist: DirectPosteriorEstimator | DirectFlowMatchingPosterior,
        flow_theta: FlowMatching,
        flow_x: Optional[FlowMatching] = None,
        embedding_net: Optional[nn.Module] = None,
    ):
        super().__init__(name, flow_theta.dim, flow_theta.cond_dim)
        self.posterior_transform = flow_theta
        self.proposal = base_dist
        self.denoiser = flow_x
        self.embedding_net = embedding_net
        self.space = "data" if embedding_net is None else "latent"

    def y_to_x(
        self,
        y: torch.Tensor,
        nsamples: int,
        device,
        timer: Optional[HierarchicalTimer] = None,
    ) -> torch.Tensor:
        # Convert y to x using the flow matching model
        if self.denoiser is not None:
            source = self.denoiser.sample_base(y, nsamples)
            broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
            cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)

            source_flat = source.reshape(-1, *source.shape[2:])  # shape (nsamples * N, *obs_dim)
            cond_flat = cond.reshape(-1, *cond.shape[2:])  # shape (nsamples * N, *obs_dim)
            x_tilde = _timed_flow_sample(
                self.denoiser, source_flat, cond_flat, device, timer, "y_to_x_flow",
                only_last=True, num_steps=20,
            )

            return x_tilde.reshape(nsamples, y.shape[0], *self.denoiser.dim)
        else:
            broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
            cond = (
                y.unsqueeze(0).expand(*broadcast_shape).to(device)
            )  # shape (nsamples, N, *obs_dim)
            return cond

    def sample(self, y: torch.Tensor, nsamples: int, device, **kwargs) -> torch.Tensor:
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)

        # Stage 1: y→x transformation (optional denoiser flow)
        x = self.y_to_x(y, nsamples, device, timer=timer)
        x = x.reshape(-1, *x.shape[2:])

        # Stage 2: Proposal sampling
        proposal_ctx = timer.time_operation("sampling", "proposal_sample") if timer else nullcontext()
        with proposal_ctx:
            source = self.proposal.sample(
                x, 1, device, space=self.space, timer=timer
            ).squeeze(0)  # shape (nsamples* N, obs_dim)

        source, _ = self.posterior_transform.rescale(source, y)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)

        # Stage 3: Posterior transform flow
        cond_flat = cond.reshape(-1, *cond.shape[2:])  # shape (nsamples * N, *obs_dim)
        samples = _timed_flow_sample(
            self.posterior_transform, source, cond_flat, device, timer,
            "posterior_transform_flow", only_last=True, num_steps=20,
        )

        return samples.reshape(nsamples, y.shape[0], *self.theta_dim)

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log p(theta|y) for the 3-stage pipeline.

        Decomposes as:
            log p(θ|y) = log_det_ode + log E_{x̃~denoiser(y)}[p_proposal(θ₀|x̃)]

        where θ₀ = T⁻¹(θ|y) via the posterior transform backward ODE, and
        log_det_ode is the change-of-variables Jacobian.

        The expectation over x̃ is estimated via Monte Carlo with ntraj samples.

        Args:
            theta: Parameters, shape (batch, theta_dim)
            y: Observations, shape (batch, obs_dim)
            **kwargs: ntraj (int), num_steps, exact_trace, n_hutchinson_probes

        Returns:
            log_probs: shape (batch,)
        """
        ntraj = kwargs.get("ntraj", 1)
        device = next(self.posterior_transform.parameters()).device

        # Stage 1: Backward ODE through flow_theta to get θ₀ in rescaled space
        x0_rescaled, _, log_det = self.posterior_transform.reverse_ode(
            theta.to(device),
            y.to(device),
            device,
            num_steps=kwargs.get("num_steps", None),
            exact_trace=kwargs.get("exact_trace", True),
            n_hutchinson_probes=kwargs.get("n_hutchinson_probes", 1),
        )

        # Unscale to data space for the proposal
        theta_0 = self.posterior_transform.target_rescaler.inverse_transform(x0_rescaled)

        # Stage 2: Sample x̃ from denoiser (Monte Carlo over y→x)
        if self.denoiser is not None:
            source = self.denoiser.sample_base(y, ntraj).transpose(0, 1).to(device)
            source = source.reshape(-1, source.shape[-1])
            cond = y.repeat_interleave(ntraj, dim=0).to(device)
            x_tilde = self.denoiser.sample(source, cond, device, only_last=True)

            # Repeat theta_0 for each trajectory
            theta_0_rep = theta_0.repeat_interleave(ntraj, dim=0)
        else:
            x_tilde = y.to(device)
            theta_0_rep = theta_0

        # Stage 3: Evaluate proposal log_prob
        proposal_lp = self.proposal.log_prob(theta_0_rep, x_tilde)

        if self.denoiser is not None and ntraj > 1:
            # Average over trajectories: log(1/K Σ exp(lp_k))
            batch_size = theta.shape[0]
            proposal_lp = proposal_lp.reshape(batch_size, ntraj)
            proposal_lp = torch.logsumexp(proposal_lp, dim=1) - math.log(ntraj)

        return proposal_lp + log_det


class RopePosterior(Posterior):
    """
    ROPE posterior: mixture of conditional posteriors weighted by OT between
    embeddings of observed and simulated data.
    """

    def __init__(
        self,
        name: str,
        theta_dim: int,
        obs_dim: int,
        embedding_net: nn.Module,
        fine_tuned_embedding_net: nn.Module,
        flow_model: Estimator,
    ):
        super().__init__(name, theta_dim, obs_dim)
        self.embedding_net = embedding_net
        self.fine_tuned_embedding_net = fine_tuned_embedding_net
        self.flow_model = flow_model

        self.sim_x = None
        self.sim_theta = None
        self.sim_emb = None
        self.ot_mat = None
        self.is_identity = isinstance(embedding_net, nn.Identity)

    # ----------------------------
    #  Set simulation data
    # ----------------------------
    def set_sim_data(self, sim_x: torch.Tensor, sim_theta: torch.Tensor):
        self.sim_x = sim_x
        self.sim_theta = sim_theta
        device = next(self.flow_model.parameters()).device
        with torch.no_grad():
            rescaled_sim = self.flow_model.data_rescaler.transform(sim_x.to(device))
            if self.is_identity:
                self.sim_emb = rescaled_sim.flatten(start_dim=1).cpu()
            else:
                self.sim_emb = (
                    self.embedding_net(rescaled_sim.to(device)).detach().cpu().flatten(start_dim=1)
                )

    # ----------------------------
    #  Compute OT plan
    # ----------------------------
    def _compute_ot_mat(
        self,
        obs: torch.Tensor,
        reg: float = 1.0,
    ):
        if self.sim_emb is None:
            raise ValueError("Simulated embeddings not set. Call set_sim_data() first.")

        n_sim_total = self.sim_emb.shape[0]
        sim_emb_np = self.sim_emb.numpy()
        # Get device from flow_model which always has parameters
        device = next(self.flow_model.parameters()).device

        with torch.no_grad():
            obs_rescaled = self.flow_model.cond_rescaler.transform(obs.to(device))
            if self.is_identity:
                obs_emb = obs_rescaled.flatten(start_dim=1).cpu()
            else:
                obs_emb = (
                    self.fine_tuned_embedding_net(obs_rescaled.to(device))
                    .detach()
                    .cpu()
                    .flatten(start_dim=1)
                )
        obs_emb_np = obs_emb.numpy()

        # Rescale using sklearn
        sc = StandardScaler()
        x_s = sc.fit_transform(sim_emb_np)
        x_o = sc.fit_transform(obs_emb.numpy())

        M = ot.dist(x_o, x_s, metric="euclidean")
        a = np.ones((x_o.shape[0],)) / x_o.shape[0]
        b = np.ones((x_s.shape[0],)) / x_s.shape[0]
        P = ot.sinkhorn(
            a, b, M, reg=reg, method="sinkhorn_stabilized", numItermax=5000, stopThr=1e-9
        )
        self.ot_mat = torch.from_numpy(P.astype(np.float32))

    # ----------------------------
    #  Flow sampling helper
    # ----------------------------
    def _sample_from_flow(
        self, context: torch.Tensor, nsamples: int, device: torch.device
    ) -> torch.Tensor:
        """
        Uses Estimator.sample-like interface:
        flow_model.sample(x=context, nsamples, device)
        Returns (nsamples, N, theta_dim)
        """
        return self.flow_model.sample(context, nsamples, device)

    def _sample_per_obs(self, weights, nsamples, device):
        """Sample from the ROPE posterior for each observation."""
        n_obs = weights.shape[0]
        all_samples = []
        for i in range(n_obs):
            w_i = weights[i].numpy()
            # Ensure weights sum to 1 (handle numerical errors from OT)
            w_i = np.clip(w_i, 0, None)  # Remove any negative values
            w_sum = w_i.sum()
            if w_sum > 0:
                w_i = w_i / w_sum
            else:
                # If all weights are zero, use uniform distribution
                w_i = np.ones_like(w_i) / len(w_i)

            idxs = np.random.choice(
                self.sim_emb.shape[0], size=nsamples, replace=True, p=w_i
            )
            # Use original sim_x for conditioning the flow, not embeddings
            # The flow was trained on (theta, x) pairs, not (theta, embedding(x))
            x_chosen = self.sim_x[idxs].to(device)  # (nsamples, obs_dim)

            # The flow expects shape (N, obs_dim), but we have nsamples contexts.
            # We sample one theta per context by calling with nsamples=1
            thetas = self._sample_from_flow(x_chosen, 1, device)  # (1, nsamples, theta_dim)
            thetas = thetas.squeeze(0).cpu()  # (nsamples, theta_dim)
            all_samples.append(thetas)
        return all_samples

    # ----------------------------
    #  Sample from ROPE posterior
    # ----------------------------
    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device = torch.device("cpu"), **kwargs
    ) -> torch.Tensor:
        # Extract RoPE-specific arguments from kwargs
        simulations = kwargs.get("simulations", None)
        ot_reg = kwargs.get("ot_reg", 1.0)
        subsample_sim = kwargs.get("subsample_sim", None)
        random_state = kwargs.get("random_state", None)
        timer: Optional[HierarchicalTimer] = kwargs.get("timer", None)

        if self.sim_emb is None:
            if simulations is None:
                raise ValueError(
                    "Simulated data not cached. Provide `simulations` or call set_sim_data()."
                )
            self.set_sim_data(simulations["x"], simulations["theta"])

        # Compute OT plan between y (real obs) and sim embeddings
        ot_ctx = timer.time_operation("sampling", "ot_computation") if timer else nullcontext()
        with ot_ctx:
            self._compute_ot_mat(y, reg=ot_reg)

        weights = self.ot_mat
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        # Per-observation sampling loop
        flow_ctx = timer.time_operation("sampling", "npe_flow_per_obs") if timer else nullcontext()
        with flow_ctx:
            all_samples = self._sample_per_obs(weights, nsamples, device)

        return torch.stack(all_samples, dim=1)  # (nsamples, n_obs, theta_dim)

    # ----------------------------
    #  Log probability
    # ----------------------------
    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute log p_ROPE(theta|y) = log sum_j n_obs * P_ij * p_flow(theta|z_j)

        Args:
            theta: Parameters, shape (n_obs, theta_dim)
            y: Observations, shape (n_obs, obs_dim)

        Returns:
            log_probs: Log probabilities, shape (n_obs,)
        """
        # Extract RoPE-specific arguments from kwargs
        simulations = kwargs.get("simulations", None)
        ot_reg = kwargs.get("ot_reg", 1.0)
        subsample_sim = kwargs.get("subsample_sim", None)
        random_state = kwargs.get("random_state", None)

        if self.sim_emb is None:
            if simulations is None:
                raise ValueError(
                    "Simulated data not cached. Provide `simulations` or call set_sim_data()."
                )
            self.set_sim_data(simulations["x"], simulations["theta"])

        # Compute OT plan between y and sim embeddings
        self._compute_ot_mat(y, reg=ot_reg)
        weights = self.ot_mat  # (n_obs, n_sim)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        n_obs = y.shape[0]
        device = next(self.flow_model.parameters()).device

        # Get the simulation observations to use as context for the flow
        # The flow was trained on (theta, x) pairs, not (theta, embedding(x))
        sim_x_pool = self.sim_x

        log_probs_list = []

        with torch.no_grad():
            for i in range(n_obs):
                theta_i = theta[i : i + 1]  # (1, theta_dim)
                w_i = weights[i]  # (n_sim,)

                # Compute log( w_ij * p(theta_i | x_j) ) in chunks
                log_weighted_terms = []

                n_sim = sim_x_pool.shape[0]
                chunk_size = 1000  # Adjust chunk size based on memory constraints
                for start in range(0, n_sim, chunk_size):
                    end = min(start + chunk_size, n_sim)

                    # slice chunk of simulation contexts
                    x_chunk = sim_x_pool[start:end]  # (chunk, obs_dim)
                    w_chunk = w_i[start:end]  # (chunk,)

                    # expand θ_i to match chunk
                    theta_chunk = theta_i.expand(end - start, -1)  # (chunk, theta_dim)

                    # compute log p(θ | x) for this chunk
                    logp_chunk = self.flow_model.log_prob(
                        theta_chunk.to(device), x_chunk.to(device)
                    )  # (chunk,)

                    # compute log( w * p )
                    log_w_chunk = torch.log(w_chunk + 1e-12).to(device)
                    log_weighted_terms.append((log_w_chunk + logp_chunk).cpu())

                # concatenate all chunks
                log_weighted = torch.cat(log_weighted_terms, dim=0)  # (n_sim,)

                # final mixture
                log_prob_i = torch.logsumexp(log_weighted, dim=0)  # scalar

                log_probs_list.append(log_prob_i)

        return torch.stack(log_probs_list)


class NPEPFNPosterior(Posterior):
    """Wrapper adapting NPE_PFN.sample_batched to Posterior interface.

    NPE_PFN uses in-context learning via TabPFN instead of gradient descent.
    The calibration data is stored as context and used at inference time.
    """

    def __init__(self, name: str, model):
        # Get dimensions from stored calibration data
        theta_dim = model._theta_train.shape[1:] if model._theta_train is not None else None
        obs_dim = model._x_train.shape[1:] if model._x_train is not None else None
        super().__init__(name, theta_dim, obs_dim)
        self.model = model
        self._force_cpu()

    def _force_cpu(self):
        """Force the underlying TabPFN model to use CPU."""
        if hasattr(self, 'model'):
            if hasattr(self.model, '_model') and self.model._model is not None:
                self.model._model.device = "cpu"
            if hasattr(self.model, '_theta_train') and self.model._theta_train is not None:
                self.model._theta_train = self.model._theta_train.cpu()
            if hasattr(self.model, '_x_train') and self.model._x_train is not None:
                self.model._x_train = self.model._x_train.cpu()

    def to(self, *args, **kwargs):
        """NPE_PFN / TabPFN runs on CPU only; force CPU instead of requested device."""
        self._force_cpu()
        return self

    def sample(
        self, x: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        """Sample from the posterior.

        Args:
            x: Observations, shape (batch_size, *obs_dim)
            nsamples: Number of samples to draw per observation
            device: Device (not used, NPE_PFN runs on CPU)

        Returns:
            samples: Shape (nsamples, batch_size, theta_dim)
        """
        # Flatten if multi-dimensional
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)

        # NPE_PFN returns [num_obs, num_samples, theta_dim]
        samples = self.model.sample_batched(
            x.cpu(),
            sample_shape=torch.Size([nsamples]),
            show_progress_bars=False,
        )

        # Transpose to [nsamples, num_obs, theta_dim]
        return samples.permute(1, 0, 2)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log probability of theta given x.

        Uses the underlying NPE_PFN autoregressive log_prob, looping over
        observations since each x_i requires a separate context fit.

        Args:
            theta: Parameters, shape (batch_size, theta_dim)
            x: Observations, shape (batch_size, *obs_dim)

        Returns:
            log_probs: Shape (batch_size,)
        """
        # Flatten if multi-dimensional
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)

        # NPE_PFN / TabPFN runs on CPU. Ensure stored training data is on CPU
        # to avoid device mismatch with CPU inputs in autoregressive log_prob.
        if self.model._theta_train is not None:
            self.model._theta_train = self.model._theta_train.cpu()
        if self.model._x_train is not None:
            self.model._x_train = self.model._x_train.cpu()

        log_probs = []
        for i in range(x.shape[0]):
            # npe_pfn.log_prob expects x as (1, dim_x) and theta as (n, dim_theta)
            lp = self.model.log_prob(
                theta[i:i+1].cpu(),
                x[i:i+1].cpu(),
            )
            log_probs.append(lp)
        return torch.cat(log_probs, dim=0)


def get_build_fn(task: str, density_estimator: str, **kwargs) -> Callable[[int, int], Flow]:
    if task in ["pendulum", "no_misspec_pendulum", "light_tunnel", "wind_tunnel", "no_misspec_wind_tunnel"]:
        embedding_dim = kwargs.get("embedding_dim", None)
        transforms = kwargs.get("ntransform", None)
        if embedding_dim is None or transforms is None:
            raise ValueError("embedding_dim and ntransform must be provided")
        return lambda f, c: UMNNFlow(f, c, **kwargs)
    elif task == "gaussian":
        return lambda f, c: MAF(f, c, **kwargs)
    else:
        if density_estimator == "nsf":
            return lambda f, c: NSF(f, c, **kwargs)
        elif density_estimator == "maf":
            return lambda f, c: MAF(f, c, **kwargs)


try:
    torch.serialization.add_safe_globals([Estimator, UMNNFlow])
except AttributeError:
    pass
