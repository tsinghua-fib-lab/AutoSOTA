from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from cellbridge.data.flow_matching_loaders import process_coupling
from cellbridge.dynamics.flow_matching import FlowMatchingLitModule
from cellbridge.dynamics.geopath import GeoPathBridge, load_or_train_geopath
from cellbridge.utils.io import get_run_dir, resolve_path

log = logging.getLogger(__name__)


@torch.no_grad()
def ode_integrate(
    v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t0: float = 0.0,
    t_final: float = 1.0,
    steps: int = 100,
    method: str = "rk4",
) -> torch.Tensor:
    """Integrate dx/dt = v(x,t) from t0 to t1.

    Args:
        v: function (x, t_tensor[B]) -> velocity
        x0: (B, D)
        steps: number of solver steps
        method: "euler" or "rk4"
    """
    x = x0.clone()
    B = x.shape[0]
    ts = torch.linspace(t0, t_final, steps + 1, device=x.device)
    dt = (t_final - t0) / steps
    for k in range(steps):
        t = ts[k].expand(B)
        if method == "euler":
            x = x + dt * v(x, t)
        else:  # rk4
            k1 = v(x, t)

            if k1.shape != x.shape:
                raise ValueError(f"Shape mismatch: {k1.shape} vs {x.shape}")

            k2 = v(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def ode_integrate_with_path_length(
    v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t0: float = 0.0,
    t_final: float = 1.0,
    steps: int = 100,
    method: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate dx/dt = v(x,t) from t0 to t1 and track path length.

    Args:
        v: function (x, t_tensor[B]) -> velocity
        x0: (B, D)
        steps: number of solver steps
        method: "euler" or "rk4"

    Returns:
        x_final: (B, D) final positions
        path_lengths: (B,) integrated path lengths for each trajectory
    """
    x = x0.clone()
    B = x.shape[0]
    ts = torch.linspace(t0, t_final, steps + 1, device=x.device)
    dt = (t_final - t0) / steps

    # Track cumulative path length (integral of ||v|| dt)
    path_lengths = torch.zeros(B, device=x.device)

    for k in range(steps):
        t = ts[k].expand(B)

        if method == "euler":
            vel = v(x, t)
            x = x + dt * vel
            path_lengths += torch.norm(vel, dim=1) * dt
        else:  # rk4
            k1 = v(x, t)
            if k1.shape != x.shape:
                raise ValueError(f"Shape mismatch: {k1.shape} vs {x.shape}")

            k2 = v(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v(x + dt * k3, t + dt)

            # Average velocity for RK4
            avg_vel = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            x = x + dt * avg_vel

            path_lengths += torch.norm(avg_vel, dim=1) * dt

    return x, path_lengths


@torch.no_grad()
def pushforward(
    net, X: torch.Tensor, steps: int = 100, t_final: float = 1.0, method="rk4"
) -> torch.Tensor:
    """Evolve a batch of X ~ p0 to t=1 using learned velocity.

    Returns: X_1 of shape (B,D)
    """

    def v(x, t):
        return net(x, t)

    return ode_integrate(v, X, t0=0.0, t_final=t_final, steps=steps, method=method)


@torch.no_grad()
def compute_trajectory_straightness(
    net: torch.nn.Module,
    X: torch.Tensor,
    steps: int = 100,
    t_final: float = 1.0,
    method: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute trajectory straightness ratio.

    The straightness S is defined as:
        S = ||z_1 - z_0|| / (integral of ||v(z, t)|| dt)

    A value close to 1 indicates a straight trajectory.
    A value < 1 indicates a curved path.

    Args:
        net: velocity network
        X: (B, D) initial positions
        steps: number of ODE solver steps
        t_final: final time
        method: "euler" or "rk4"

    Returns:
        straightness: (B,) straightness ratio for each trajectory
        x_final: (B, D) final positions
    """

    def v(x, t):
        return net(x, t)

    x_final, path_lengths = ode_integrate_with_path_length(
        v, X, t0=0.0, t_final=t_final, steps=steps, method=method
    )

    euclidean_dist = torch.norm(x_final - X, dim=1)
    straightness = euclidean_dist / (path_lengths + 1e-12)

    return straightness, x_final


@torch.no_grad()
def sde_integrate(
    velocity_net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    score_net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    sigma_sample: float,
    generator: torch.Generator | None = None,
    *,
    t0: float = 0.0,
    t_final: float = 1.0,
    steps: int = 100,
) -> torch.Tensor:
    """Euler-Maruyama simulation of dx = u(x,t) dt + g dW with constant g."""
    if steps <= 0:
        raise ValueError("Number of steps must be positive for SDE integration.")
    x = x0.clone()
    B = x.shape[0]
    ts = torch.linspace(t0, t_final, steps + 1, device=x.device)
    dt = (t_final - t0) / steps
    sqrt_dt = math.sqrt(max(float(dt), 0.0))
    diff = float(sigma_sample)
    for k in range(steps):
        t = ts[k].expand(B)
        drift = velocity_net(x, t) + 0.5 * (diff**2) * score_net(x, t)
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        x = x + drift * dt + diff * sqrt_dt * noise
    return x


def sample_from_velocity_model(
    velocity_model: torch.nn.Module,
    X_start: torch.Tensor,
    steps: int,
    t_final: float,
    *,
    method: str = "rk4",
    score_model: torch.nn.Module | None = None,
    sampling_mode: str = "ode",
    sigma_sample: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample trajectories starting at X_start by integrating the learned dynamics.
    """
    if velocity_model is None:
        raise RuntimeError("velocity_model is not initialized yet")

    sampling_mode = sampling_mode.lower()
    if sampling_mode == "ode":
        X_push = pushforward(
            velocity_model, X_start, steps=steps, t_final=t_final, method=method
        )
    elif sampling_mode == "sde":
        if score_model is None:
            raise ValueError("score_model must be provided for stochastic sampling.")
        X_push = sde_integrate(
            velocity_model,
            score_model,
            X_start,
            sigma_sample=float(sigma_sample),
            generator=generator,
            t0=0.0,
            t_final=t_final,
            steps=steps,
        )
    else:
        raise ValueError(
            f"Unsupported sampling_mode '{sampling_mode}'. Use 'ode' or 'sde'."
        )
    if X_push.shape != X_start.shape:
        raise ValueError(
            f"Shape mismatch after pushforward: {X_push.shape} vs {X_start.shape}"
        )
    return X_push


class Sampler(ABC):
    """Base sampler with metrics and output helpers."""

    def __init__(self, cfg: DictConfig, *, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or log

        try:
            self.root = Path(get_original_cwd())
        except Exception:
            self.root = Path.cwd()

    @abstractmethod
    def _load_data(self): ...

    def evaluate(
        self,
        X_push: torch.Tensor | np.ndarray,
        X_eval: torch.Tensor | np.ndarray,
        w_push: torch.Tensor | np.ndarray | None = None,
        w_eval: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate generated samples against target samples."""
        metric_collection = instantiate(self.cfg.metrics)
        return metric_collection(X_push, X_eval, w_push=w_push, w_eval=w_eval)

    def _get_output_dir(self) -> Path:
        # Respect explicit outputs.dir, else use Hydra's run dir helper
        out_cfg = getattr(self.cfg, "outputs", {})
        if isinstance(out_cfg, DictConfig) and "dir" in out_cfg and out_cfg.dir:
            out = Path(out_cfg.dir)
            return out if out.is_absolute() else (self.root / out)
        return get_run_dir()

    def save_outputs(
        self,
        metric_results: dict[str, dict[str, float]],
        X_push: torch.Tensor | np.ndarray,
        w_push: torch.Tensor | np.ndarray | None,
        X_eval: torch.Tensor | np.ndarray,
        out_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Save metrics and sampled arrays."""
        if out_dir is None:
            out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metric_results, f)

        X_push = X_push.cpu() if isinstance(X_push, torch.Tensor) else X_push
        X_eval = X_eval.cpu() if isinstance(X_eval, torch.Tensor) else X_eval
        w_push = w_push.cpu() if isinstance(w_push, torch.Tensor) else w_push

        samples_path = out_dir / "samples.pt"
        with open(samples_path, "wb") as f:
            torch.save({"X_push": X_push, "X_eval": X_eval, "w_push": w_push}, f)

        return {"metrics": metrics_path, "samples": samples_path}


class MarginalSampler(Sampler):
    """This class does not require a velocity model. It simply gets the marginal at a
    given time t based on a coupling"""

    def __init__(self, cfg: DictConfig, *, logger: logging.Logger | None = None):
        super().__init__(cfg, logger=logger)

        self.X_start: np.ndarray | None = None
        self.X_end: np.ndarray | None = None
        self.X_eval: np.ndarray | None = None

        self.G: np.ndarray | None = None
        self.geopath_bridge: GeoPathBridge | None = None

    def run(self):
        """Sample interpolation points from the stored coupling."""
        self._load_data()

        assert self.X_start is not None
        assert self.X_end is not None
        assert self.G is not None
        rows, cols, w_push = process_coupling(
            self.G,
            topk=self.cfg.coupling.topk,
            tau=self.cfg.coupling.tau,
            keep_marginals=self.cfg.coupling.keep_marginals,
        )

        interpolation = self._interpolate_with_coupling(rows, cols)

        assert self.X_eval is not None

        self.logger.info(
            "Interpolated samples: %s, Weights: %s",
            tuple(interpolation.shape),
            tuple(w_push.shape),
        )

        metrics_interpolation = self._filter_metrics_for_plotting(
            self.evaluate(interpolation, self.X_eval, w_push, None)
        )

        dic_metrics = {
            "interpolation": metrics_interpolation,
        }

        if self.cfg.evaluate_start_end:
            self.logger.info("Evaluating start and end distributions as well.")
            metrics_start = self._filter_metrics_for_plotting(
                self.evaluate(self.X_start, self.X_eval, None, None)
            )
            metrics_end = self._filter_metrics_for_plotting(
                self.evaluate(self.X_end, self.X_eval, None, None)
            )
            dic_metrics["start"] = metrics_start
            dic_metrics["end"] = metrics_end

        out_dir = self._get_output_dir()

        _ = self.save_outputs(
            dic_metrics,
            interpolation,
            w_push,
            self.X_eval,
            out_dir=out_dir,
        )

        self.logger.info("Outputs saved to: %s", out_dir)

    @staticmethod
    def _filter_metrics_for_plotting(metrics: dict[str, float]) -> dict[str, float]:
        keep = {"wasserstein_1", "wasserstein_2"}
        return {k: v for k, v in metrics.items() if k in keep}

    def _load_data(self) -> None:
        cfg = self.cfg

        def _to_abs(p: Path | str) -> Path:
            return resolve_path(p, self.root)

        self.logger.info("Loading data (mode=%s)", cfg.inputs.mode)

        if cfg.inputs.mode == "from_h5ad":
            A = ad.read_h5ad(_to_abs(cfg.inputs.h5ad_path))
            X_start = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.start_label].obsm[
                cfg.rep.obsm_key
            ]
            X_end = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.end_label].obsm[
                cfg.rep.obsm_key
            ]
            X_eval = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.eval_label].obsm[
                cfg.rep.obsm_key
            ]
        else:
            A_start = ad.read_h5ad(_to_abs(cfg.inputs.start_path))
            A_end = ad.read_h5ad(_to_abs(cfg.inputs.end_path))
            A_eval = ad.read_h5ad(_to_abs(cfg.inputs.eval_path))
            X_start = A_start.obsm[cfg.rep.obsm_key]
            X_end = A_end.obsm[cfg.rep.obsm_key]
            X_eval = A_eval.obsm[cfg.rep.obsm_key]

        self.X_start = np.asarray(X_start)
        self.X_end = np.asarray(X_end)
        self.X_eval = np.asarray(X_eval)

        if self.cfg.inputs.get("rep_transform", None):
            rep_transform = instantiate(self.cfg.inputs.rep_transform)
            self.X_start = rep_transform(self.X_start)
            self.X_end = rep_transform(self.X_end)
            self.X_eval = rep_transform(self.X_eval)
            self.logger.info("Applied representation transform")

        self.logger.info(
            "X_start: %s, X_end: %s, X_eval: %s",
            tuple(X_start.shape),
            tuple(X_end.shape),
            tuple(X_eval.shape),
        )

        if hasattr(cfg, "subsample") and cfg.inputs.subsample:
            self.logger.info("Subsampling to max %d cells", cfg.inputs.subsampling_size)
            seed = cfg.get("seed", 42)
            rng_start = np.random.Generator(np.random.PCG64(seed))
            rng_end = np.random.Generator(np.random.PCG64(seed))
            rng_eval = np.random.Generator(np.random.PCG64(seed))
            self.logger.info(
                "Created separate RNGs with seeds %d, %d, %d for start/end/eval",
                seed,
                seed,
                seed,
            )

            if self.X_start.shape[0] > cfg.inputs.subsampling_size:
                indices = rng_start.choice(
                    self.X_start.shape[0],
                    size=cfg.inputs.subsampling_size,
                    replace=False,
                )
                self.X_start = self.X_start[indices]
                self.logger.info("Subsampled X_start to %s", self.X_start.shape)
                self.logger.info(f"First few indices: {indices[:10]} for  X_start")

            if self.X_end.shape[0] > cfg.inputs.subsampling_size:
                indices = rng_end.choice(
                    self.X_end.shape[0], size=cfg.inputs.subsampling_size, replace=False
                )
                self.X_end = self.X_end[indices]
                self.logger.info("Subsampled X_end to %s", self.X_end.shape)

            if self.X_eval.shape[0] > cfg.inputs.subsampling_size:
                indices = rng_eval.choice(
                    self.X_eval.shape[0],
                    size=cfg.inputs.subsampling_size,
                    replace=False,
                )
                self.X_eval = self.X_eval[indices]
                self.logger.info("Subsampled X_eval to %s", self.X_eval.shape)

        self.G = np.load(Path(cfg.folder_artifacts) / "coupling.npy")
        self.geopath_bridge = self._build_geopath_bridge(self.X_start.shape[1])

    def _build_geopath_bridge(self, data_dim: int) -> GeoPathBridge | None:
        cfg_geopath = getattr(self.cfg, "geopath", None)
        if cfg_geopath is None or not cfg_geopath.get("enabled", False):
            return None

        geopath_net = load_or_train_geopath(
            geopath_cfg=cfg_geopath,
            data_dim=data_dim,
            artifacts_folder=Path(self.cfg.folder_artifacts),
            X=self.X_start,
            Y=self.X_end,
            G=self.G,
            logger=self.logger,
            save_trained_checkpoint=True,
            output_dir=self._get_output_dir() / "..",
        )

        return GeoPathBridge(
            geopath_net,
            alpha=cfg_geopath.get("alpha", 1.0),
            trainable=False,
        )

    def _interpolate_with_coupling(
        self, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        t_eval = self.cfg.inputs.t_eval
        if self.geopath_bridge is None:
            sigma_bridge = float(getattr(self.cfg, "sigma_bridge", 0.0))
            X_interp = (1 - t_eval) * self.X_start[rows] + t_eval * self.X_end[cols]

            if sigma_bridge > 0:
                # Add Brownian bridge noise: sqrt(t(1-t)) * sigma * Z
                noise_scale = np.sqrt(t_eval * (1 - t_eval)) * sigma_bridge
                noise = np.random.randn(*X_interp.shape) * noise_scale
                X_interp = X_interp + noise
                self.logger.info(
                    "Using Gaussian (Brownian bridge) interpolation with sigma=%.3f",
                    sigma_bridge,
                )
            else:
                self.logger.info("GeoPath disabled; using linear OT interpolation.")

            return X_interp

        self.logger.info("Generating interpolation via GeoPath bridge (t=%.3f)", t_eval)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xi = torch.from_numpy(self.X_start[rows]).float().to(device)
        yj = torch.from_numpy(self.X_end[cols]).float().to(device)
        t = torch.full((xi.shape[0],), float(t_eval), device=device)

        self.geopath_bridge.geopath_net.to(device)
        xt, _ = self.geopath_bridge.sample(xi, yj, t, detach_outputs=True)

        # Log deviation from linear interpolation.
        t_reshaped = t[:, None]  # Reshape for broadcasting
        correction = xt - ((1 - t_reshaped) * xi + t_reshaped * yj)
        correction_norm = torch.norm(correction, dim=1).mean().item()
        self.logger.info("Average correction norm from GeoPath: %.6f", correction_norm)
        return xt.cpu().numpy()


class VelocitySampler(Sampler):
    """Push forward samples with a trained velocity model."""

    def __init__(self, cfg: DictConfig, *, logger: logging.Logger | None = None):
        super().__init__(cfg, logger=logger)

        # Device
        want = getattr(cfg, "device", None)
        if want == "auto" or want is None:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(want)

        # Lazily populated
        self.velocity_model: torch.nn.Module | None = None
        self.score_model: torch.nn.Module | None = None
        self.X_start: torch.Tensor | None = None
        self.X_eval: torch.Tensor | None = None

    def run(self):
        """Load model + data, sample, evaluate, save artifacts for best checkpoint."""
        best_ckpt_info = self._load_best_model()
        if best_ckpt_info is None:
            self.logger.error("No checkpoints found in the specified folder")
            return []

        ckpt_path, wandb_run, val_loss = best_ckpt_info
        self.logger.info(
            f"Using best checkpoint: {ckpt_path} "
            f"(wandb_run={wandb_run}, val_loss={val_loss})"
        )

        self._load_data()

        assert self.X_start is not None, "X_start is not loaded"
        assert self.X_eval is not None, "X_eval is not loaded"

        self.logger.info(f"Loading checkpoint: {ckpt_path}")
        flow = FlowMatchingLitModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), strict=False
        )
        flow.eval().to(self.device)
        self.velocity_model = flow.net
        self.score_model = getattr(flow, "score_net", None)

        if hasattr(self.cfg, "rep_transform") and self.cfg.rep_transform is not None:
            rep_transform = instantiate(self.cfg.rep_transform)
            X_start_np = rep_transform(self.X_start.cpu().numpy())
            X_eval_np = rep_transform(self.X_eval.cpu().numpy())
            self.X_start = torch.tensor(X_start_np).to(self.device)
            self.X_eval = torch.tensor(X_eval_np).to(self.device)
            self.logger.info("Applied representation transform to sampling data")

        sampling_mode = str(getattr(self.cfg, "sampling_mode", "ode")).lower()
        sigma_sample = float(getattr(self.cfg, "sigma_sample", 0.0))
        if sampling_mode == "sde" and self.score_model is None:
            raise ValueError(
                "Stochastic sampling requested but score network was not found in "
                "the loaded checkpoint."
            )
        self.logger.info(
            f"Sampling with {self.cfg.steps} steps, t_final={self.cfg.t_final}, "
            f"mode={sampling_mode}"
        )
        rng_gen = None
        if sampling_mode == "sde":
            seed = int(getattr(self.cfg, "seed", 42))
            gen_device = "cuda" if self.X_start.is_cuda else "cpu"
            rng_gen = torch.Generator(device=gen_device).manual_seed(seed)
            self.logger.info("Using seeded SDE generator with seed=%d", seed)
        X_push = sample_from_velocity_model(
            self.velocity_model,
            self.X_start,
            steps=self.cfg.steps,
            t_final=self.cfg.t_final,
            method=getattr(self.cfg, "method", "rk4"),
            score_model=self.score_model,
            sampling_mode=sampling_mode,
            sigma_sample=sigma_sample,
            generator=rng_gen,
        )

        metrics_push = self.evaluate(X_push, self.X_eval)
        metrics_start = self.evaluate(self.X_start, self.X_eval)
        metrics = {
            "pushforward": metrics_push,
            "start": metrics_start,
            "checkpoint_info": {
                "path": str(ckpt_path),
                "wandb_run": wandb_run,
                "val_loss": val_loss,
            },
        }

        out_dir = self._get_output_dir()
        paths = self.save_outputs(metrics, X_push, None, self.X_eval, out_dir=out_dir)

        self.logger.info(f"Sampling complete. Metrics: {metrics}")

        return [
            {
                "wandb_run": wandb_run,
                "checkpoint": ckpt_path,
                "val_loss": val_loss,
                **paths,
                "metrics": metrics,
            }
        ]

    def compute_straightness(
        self,
        steps: int | None = None,
        t_final: float | None = None,
        method: str | None = None,
    ) -> dict[str, float]:
        """Compute trajectory straightness metrics.

        Returns:
            Dictionary with mean, std, min, max straightness values
        """
        if self.velocity_model is None:
            raise RuntimeError("velocity_model is not initialized yet")
        if self.X_start is None:
            raise RuntimeError("X_start is not initialized yet")

        _steps = steps if steps is not None else self.cfg.steps
        _t_final = t_final if t_final is not None else self.cfg.t_final
        _method = method if method is not None else getattr(self.cfg, "method", "rk4")

        self.logger.info(
            f"Computing trajectory straightness with {_steps} steps, "
            f"t_final={_t_final}, method={_method}"
        )

        straightness, _ = compute_trajectory_straightness(
            self.velocity_model,
            self.X_start,
            steps=_steps,
            t_final=_t_final,
            method=_method,
        )

        straightness_np = straightness.cpu().numpy()
        results = {
            "mean": float(np.mean(straightness_np)),
            "std": float(np.std(straightness_np)),
            "min": float(np.min(straightness_np)),
            "max": float(np.max(straightness_np)),
            "median": float(np.median(straightness_np)),
        }

        self.logger.info(f"Straightness statistics: {results}")
        return results

    def _load_best_model(self):
        """Load the best checkpoint based on validation loss."""
        cfg = self.cfg

        ckpt_folder = Path(cfg.flow.folder)
        if not ckpt_folder.is_absolute():
            ckpt_folder = self.root / ckpt_folder

        from cellbridge.utils.extraction import get_best_checkpoint

        best_ckpt_info = get_best_checkpoint(ckpt_folder)
        if best_ckpt_info is None:
            self.logger.error(f"No checkpoints found in {ckpt_folder}")
            return None

        ckpt_path, wandb_run, val_loss = best_ckpt_info
        self.logger.info(f"Found best checkpoint with val_loss={val_loss}: {ckpt_path}")
        return best_ckpt_info

    def _load_data(self) -> None:
        cfg = self.cfg

        def _to_abs(p: Path | str) -> Path:
            return resolve_path(p, self.root)

        self.logger.info("Loading data (mode=%s)", cfg.inputs.mode)

        if cfg.inputs.mode == "from_h5ad":
            A = ad.read_h5ad(_to_abs(cfg.inputs.h5ad_path))
            X_start = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.start_label].obsm[
                cfg.rep.obsm_key
            ]
            X_eval = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.eval_label].obsm[
                cfg.rep.obsm_key
            ]
        else:
            A_start = ad.read_h5ad(_to_abs(cfg.inputs.start_path))
            A_eval = ad.read_h5ad(_to_abs(cfg.inputs.eval_path))
            X_start = A_start.obsm[cfg.rep.obsm_key]
            X_eval = A_eval.obsm[cfg.rep.obsm_key]

        # Convert to numpy arrays first for rep_transform compatibility
        X_start = np.asarray(X_start)
        X_eval = np.asarray(X_eval)

        if self.cfg.inputs.get("rep_transform", None):
            rep_transform = instantiate(self.cfg.inputs.rep_transform)
            X_start = rep_transform(X_start)
            X_eval = rep_transform(X_eval)
            self.logger.info("Applied representation transform")

        self.X_start = torch.as_tensor(X_start, dtype=torch.float32, device=self.device)
        self.X_eval = torch.as_tensor(X_eval, dtype=torch.float32, device=self.device)

        self.logger.info(
            "X_start: %s, X_eval: %s",
            tuple(self.X_start.shape),
            tuple(self.X_eval.shape),
        )

        if hasattr(cfg, "subsample") and cfg.inputs.subsample:
            self.logger.info("Subsampling to max %d cells", cfg.inputs.subsampling_size)
            seed = cfg.get("seed", 42)
            rng_start = np.random.Generator(np.random.PCG64(seed))
            rng_eval = np.random.Generator(np.random.PCG64(seed))
            self.logger.info(
                "Created separate RNGs with seeds %d, %d for start/eval", seed, seed
            )

            if self.X_start.shape[0] > cfg.inputs.subsampling_size:
                indices = rng_start.choice(
                    self.X_start.shape[0],
                    size=cfg.inputs.subsampling_size,
                    replace=False,
                )
                self.X_start = self.X_start[indices]
                self.logger.info("Subsampled X_start to %s", self.X_start.shape)
                self.logger.info(f"First few indices: {indices[:10]} for  X_start")

            if self.X_eval.shape[0] > cfg.inputs.subsampling_size:
                indices = rng_eval.choice(
                    self.X_eval.shape[0],
                    size=cfg.inputs.subsampling_size,
                    replace=False,
                )
                self.X_eval = self.X_eval[indices]
                self.logger.info("Subsampled X_eval to %s", self.X_eval.shape)

    @staticmethod
    def _require(value, name: str):
        if value is None:
            raise RuntimeError(f"{name} is not initialized yet")
