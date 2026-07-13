"""Mechanism design helpers built on ``FiniteModel`` with logging and training utilities."""

import glob
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

import tools.feedback as feedback
from tools.feedback import logger
from tools.utils import (
    IR_constraint,
    attach_nan_hooks,
    model_max_constraint,
    model_min_constraint,
    hash_dict,
)

from models.finite_model import FiniteModel

__all__ = ["Mechanism", "Trainer", "run", "barrier"]


def barrier(
    c: torch.Tensor,
    epsilon: float = 1e-2,
    reduction: str = "none",
) -> torch.Tensor:
    """Hinge-squared penalty for constraints where c >= 0 is feasible."""
    pen = -(epsilon - c).clamp_min(0.0).pow(2)
    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    return pen


class Mechanism(FiniteModel):
    """
    Wrapper around ``FiniteModel`` that exposes the same interface used by the
    mechanism-design training loop (``compute_mechanism`` and optional hooks).
    """

    def __init__(
        self,
        npoints: int = 1000,
        kernel: Callable[..., torch.Tensor] = None,
        y_dim: int = 1,
        cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        temp: float = 50.0,
        is_Y_parameter: bool = False,
        is_there_default: bool = False,
        default_intercept: float = 0.0,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        original_distance_to_bounds: float = 1e-1,
        kernel_batch_size: Optional[int] = None,
        sorted_model: bool = False,
    ) -> None:
        if kernel is None:
            raise ValueError("kernel must be provided")
        super().__init__(
            num_candidates=npoints,
            num_dims=y_dim,
            kernel=kernel,
            mode="convex",
            temp=temp,
            is_y_parameter=is_Y_parameter,
            y_min=y_min,
            y_max=y_max,
            original_dist_to_bounds=original_distance_to_bounds,
            is_there_default=is_there_default,
            default_intercept=default_intercept,
            sorted_model=sorted_model,
        )
        self.cost_fn = cost_fn
        self.kernel = kernel
        self.converged = None
        self.kernel_batch_size = kernel_batch_size
        self.cost_fn_name = self._sanitize_cost_name(cost_fn)
        self.kernel_name = self._sanitize_kernel_name(kernel)
        self.default_intercept = default_intercept

    @staticmethod
    def _sanitize_cost_name(cost_fn: Optional[Callable[..., Any]]) -> str:
        if cost_fn is None:
            return "no_cost"
        name = getattr(cost_fn, "__name__", None)
        if not name:
            name = type(cost_fn).__name__
        sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
        return sanitized or "cost_fn"

    @staticmethod
    def _sanitize_kernel_name(kernel_fn: Optional[Callable[..., Any]]) -> str:
        if kernel_fn is None:
            return "no_kernel"
        name = getattr(kernel_fn, "__name__", None)
        if not name:
            name = type(kernel_fn).__name__
        sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
        return sanitized or "kernel_fn"

    @classmethod
    def with_hooks(cls, *args: Any, **kwargs: Any) -> "Mechanism":
        model = cls(*args, **kwargs)
        attach_nan_hooks(model)
        logger.info("✅ NaN hooks attached to the mechanism.")
        return model

    def compute_mechanism(self, sample: torch.Tensor, mode: str = "soft", already_sorted: bool = False) -> Dict[str, Any]:
        """Compute choice, utility, and profit statistics for a sample batch."""
        choice, v = self.forward(sample, selection_mode=mode, already_sorted=already_sorted)
        ker = self.kernel_fn(sample, choice)
        revenue = ker - v
        cost = self.cost_fn(choice) if self.cost_fn is not None else torch.zeros_like(revenue)
        profits = revenue - cost
        return {
            "sample": sample,
            "choice": choice,
            "v": v,
            "revenue": revenue,
            "kernel": ker,
            "cost": cost,
            "profits": profits,
            "y": self.full_Y(),
            "intercept": self.full_intercept(),
        }

    @staticmethod
    def _glob_pattern(path: str) -> str:
        return os.path.join(path, "*")

    @staticmethod
    def _load_final_if_exists(
        writing_dir: str,
        run_id: Optional[str] = None,
        run_hash: Optional[str] = None,
    ) -> Optional[nn.Module]:
        """Return the latest saved final mechanism matching ID/hash when available."""
        pattern = os.path.join(writing_dir, "*final*")
        finals = glob.glob(pattern)
        if run_hash:
            short = run_hash[:8]
            finals = [p for p in finals if short in os.path.basename(p)]
        elif run_id:
            finals = [p for p in finals if run_id in os.path.basename(p)]
        if finals:
            latest_final = max(finals, key=os.path.getmtime)
            logger.info(f"Loading final snapshot: {latest_final}")
            mech = torch.load(latest_final, map_location="cpu", weights_only=False)
            return mech
        return None

    @staticmethod
    def _load_latest_checkpoint(writing_dir: str) -> Optional[Tuple[Any, Any, Any, int, str]]:
        """Return most recent checkpoint (mechanism, optimizer, scheduler, epoch, run_id)."""
        pattern = os.path.join(writing_dir, "*epoch_*.pt")
        snapshots = glob.glob(pattern)
        if not snapshots:
            return None

        def extract_epoch(path: str) -> int:
            match = re.search(r"epoch_(\d+)\.pt", path)
            return int(match.group(1)) if match else -1

        latest_snapshot = max(snapshots, key=extract_epoch)
        checkpoint = torch.load(latest_snapshot, map_location="cpu", weights_only=False)

        m = re.search(r"([^/]+)_epoch_\d+\.pt", os.path.basename(latest_snapshot))
        run_id = m.group(1) if m else ""
        return (
            checkpoint.get("mechanism", None),
            checkpoint.get("optimizer", None),
            checkpoint.get("scheduler", None),
            extract_epoch(latest_snapshot),
            run_id,
        )

    @staticmethod
    def _save_snapshot(prefix: str, epoch: int, mechanism: nn.Module, optimizer: Any, scheduler: Any) -> None:
        """Persist mechanism/optimizer/scheduler state for resume."""
        snapshot_path = f"{prefix}epoch_{epoch}.pt"
        torch.save({"mechanism": mechanism, "optimizer": optimizer, "scheduler": scheduler}, snapshot_path)
        logger.info(f"Epoch {epoch}: Saved snapshot to {snapshot_path}")

    def fit(
        self,
        sample: torch.Tensor,
        modes: List[str] = ["soft"],
        compile: bool = False,
        optimizers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        schedulers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        already_sorted: bool = False,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Train the mechanism via the shared Trainer helper.

        Returns
        -------
        Tuple[nn.Module, Dict[str, Any]]
            The trained mechanism and a batch of computed mechanism outputs for logging.
        """
        optimizers_kwargs_dict = optimizers_kwargs_dict or {}
        schedulers_kwargs_dict = schedulers_kwargs_dict or {}
        train_kwargs = dict(train_kwargs) if train_kwargs is not None else {}

        for mode in modes:
            if mode not in optimizers_kwargs_dict:
                raise KeyError(f"Mode '{mode}' not found in optimizers_kwargs_dict keys: {list(optimizers_kwargs_dict.keys())}")
            if mode not in schedulers_kwargs_dict:
                raise KeyError(f"Mode '{mode}' not found in schedulers_kwargs_dict keys: {list(schedulers_kwargs_dict.keys())}")

        compiled_mechanism = torch.compile(self) if compile else self

        optimizers = train_kwargs.pop("optimizers", None)
        if optimizers is None:
            optimizers = {}
            for mode in modes:
                if mode in {"soft", "hard"}:
                    optimizers[mode] = torch.optim.AdamW(
                        compiled_mechanism.parameters(),
                        **optimizers_kwargs_dict.get(mode, {}),
                    )
                if mode == "ste":
                    optimizers[mode] = torch.optim.SGD(
                        compiled_mechanism.parameters(),
                        **optimizers_kwargs_dict.get(mode, {}),
                    )

        schedulers = train_kwargs.pop("schedulers", None)
        if schedulers is None:
            schedulers = {}
            for mode in modes:
                scheduler_kwargs = schedulers_kwargs_dict.get(mode, {})
                if mode in {"soft", "hard"}:
                    schedulers[mode] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizers[mode],
                        mode="max",
                        **scheduler_kwargs,
                    )
                if mode == "ste":
                    schedulers[mode] = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizers[mode],
                        **scheduler_kwargs,
                    )

        constraint_fns = train_kwargs.pop("constraint_fns", [])
        if self.y_min is not None and model_min_constraint not in constraint_fns:
            constraint_fns.append(model_min_constraint)
        if self.y_max is not None and model_max_constraint not in constraint_fns:
            constraint_fns.append(model_max_constraint)
        if IR_constraint not in constraint_fns:
            constraint_fns.append(IR_constraint)

        temp_warmup_steps = train_kwargs.pop("temp_warmup_steps", None)
        temp_schedule_initial = train_kwargs.pop("temp_schedule_initial", 0.0)
        temp_high = train_kwargs.pop("temp_high", None)

        train_kwargs_for_hash = {
            k: v
            for k, v in train_kwargs.items()
            if k not in {"optimizers", "schedulers", "writing_dir", "nsteps"}
        }
        run_hash = self._make_run_hash(sample, modes, optimizers_kwargs_dict, schedulers_kwargs_dict, train_kwargs_for_hash)

        trainer = Trainer(
            sample=sample,
            mechanism=compiled_mechanism,
            optimizers=optimizers,
            schedulers=schedulers,
            modes=modes,
            constraint_fns=constraint_fns,
            temp_warmup_steps=temp_warmup_steps,
            temp_schedule_initial=temp_schedule_initial,
            temp_high=temp_high,
            **train_kwargs,
            already_sorted=already_sorted,
            run_hash=run_hash,
        )
        return trainer.run()

    def _make_run_hash(
        self,
        sample: torch.Tensor,
        modes: List[str],
        optimizers_kwargs_dict: Dict[str, Any],
        schedulers_kwargs_dict: Dict[str, Any],
        train_kwargs: Dict[str, Any],
    ) -> str:
        """Hash training configuration for checkpoint identification."""
        sample_cpu = sample.detach().cpu()
        info = {
            "kernel": self.kernel,
            "cost_fn": self.cost_fn,
            "kernel_batch_size": self.kernel_batch_size,
            "modes": modes,
            "y_dim": self.num_dims,
            "npoints": self.num_candidates,
            "default_intercept": self.default_intercept,
            "sample": sample_cpu,
            "optimizers_kwargs": optimizers_kwargs_dict,
            "schedulers_kwargs": schedulers_kwargs_dict,
            "train_kwargs": train_kwargs,
        }
        return hash_dict(info)


class Trainer:
    """Encapsulate training loop state and behavior for mechanism optimization."""

    def __init__(
        self,
        sample: torch.Tensor,
        mechanism: nn.Module,
        optimizers: Dict[str, Any],
        schedulers: Dict[str, Any],
        modes: List[str],
        constraint_fns: Optional[List[Callable[[nn.Module, Dict[str, Any]], torch.Tensor]]] = None,
        initial_penalty_factor: float = 1.0,
        nsteps: int = 10000,
        steps_per_snapshot: int = 200,
        steps_per_update: int = 50,
        window: int = 500,
        detect_anomaly: bool = False,
        epsilon: float = 1e-2,
        use_wandb: bool = True,
        wandb_project: str = "mechanism-design",
        writing_dir: str = "tmp",
        run_hash: Optional[str] = None,
        convergence_tolerance: float = 1e-5,
        switch_threshold: float = 0.98,
        switch_patience: int = 200,
        already_sorted: bool = False,
        max_clipping_norm: float = 5.0,
        temp_warmup_steps: Optional[int] = None,
        temp_schedule_initial: float = 0.0,
        temp_high: Optional[float] = None,
    ) -> None:
        """Prepare optimizer/scheduler state, logging, and checkpoint metadata."""
        self.sample = sample
        self.mechanism = mechanism
        self.already_sorted = already_sorted
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.modes = modes
        self.constraint_fns = constraint_fns or []
        self.initial_penalty_factor = initial_penalty_factor
        self.nsteps = nsteps
        self.steps_per_snapshot = steps_per_snapshot
        self.steps_per_update = steps_per_update
        self.window = window
        self.detect_anomaly = detect_anomaly
        self.epsilon = epsilon
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        base_writing_dir = os.fspath(writing_dir)
        self.writing_dir = base_writing_dir
        self.convergence_tolerance = convergence_tolerance
        self.switch_threshold = switch_threshold
        self.switch_patience = switch_patience
        self.max_clipping_norm = max_clipping_norm
        self.temp_warmup_steps = temp_warmup_steps
        self.temp_schedule_initial = temp_schedule_initial
        self.temp_high = temp_high
        self.base_temp = float(self.mechanism.temp)

        self.mode = modes[0]
        self.optimizer = optimizers[self.mode]
        self.scheduler = schedulers[self.mode]
        self.nmodes = len(modes)
        self.above_thresh_streak = 0

        self.nconstraints = len(self.constraint_fns)
        self.base_penalty = torch.full((self.nconstraints,), initial_penalty_factor)
        self.penalty_factors = self.base_penalty.clone()
        self.max_penalty = 2000.0 * self.base_penalty
        self.min_penalty = 0.1 * self.base_penalty
        self.alpha = 2.0
        self.beta_ema = 0.9
        self.violation2_ema = torch.zeros(self.nconstraints)

        self.ls: List[float] = []
        self.best_profit_per_good = -float("inf")
        self.best_state = None
        self.start_epoch = 0
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{run_hash[:8]}" if run_hash else ""
        self.run_id = f"{timestamp}{suffix}"
        self.run_hash = run_hash
        self.dim_tag = f"dim{self.mechanism.num_dims}"
        os.makedirs(self.writing_dir, exist_ok=True)

        self.wandb_project_url = None
        self.wandb_run_url = None
        self.live_wandb_dict: Dict[str, Any] = {}
        if self.use_wandb:
            import wandb

            wandb_dir = os.path.join(self.writing_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(project=self.wandb_project, name=f"run_{self.run_id}", dir=wandb_dir, config={"nsteps": self.nsteps})
            wandb.watch(self.mechanism, log="all", log_freq=self.steps_per_snapshot)
            self.wandb_project_url = f"https://wandb.ai/{wandb.run.entity}/{self.wandb_project}"
            self.wandb_run_url = wandb.run.url
            self._wandb = wandb
            self.live_wandb_dict = {
                "wandb_project_url": self.wandb_project_url,
                "wandb_run_url": self.wandb_run_url,
            }
            logger.info(f"🚀 WandB Project: {self.wandb_project_url}")
            logger.info(f"📊 WandB Run: {self.wandb_run_url}")

    def _load_final_if_exists(self) -> Optional[nn.Module]:
        return Mechanism._load_final_if_exists(
            self.writing_dir,
            run_id=self.run_id,
            run_hash=self.run_hash,
        )

    def _load_latest_checkpoint(self) -> Optional[Tuple[Any, Any, Any]]:
        result = Mechanism._load_latest_checkpoint(self.writing_dir)
        if result is None:
            return None
        mech, opt, sch, epoch, run_id = result
        self.start_epoch = epoch
        if run_id:
            self.run_id = run_id + "R"
        return mech, opt, sch

    def _save_snapshot(self, prefix: str, epoch: int) -> None:
        Mechanism._save_snapshot(prefix, epoch, self.mechanism, self.optimizer, self.scheduler)
        if self.use_wandb:
            try:
                self._wandb.log({"checkpoint_saved": epoch}, step=epoch)
            except Exception:
                logger.debug("wandb.log failed during checkpoint save")

    def _compute_temp(self, epoch: int) -> float:
        """Temperature schedule supporting three modes:
        1. Full cosine annealing (temp_high set, warmup_steps=0): temp_high -> base_temp
        2. Two-phase (temp_high set, warmup_steps>0): warmup: init->high, then high->base
        3. Legacy warmup-only (temp_high not set): init -> base over warmup_steps
        """
        if self.temp_high is not None:
            if self.temp_warmup_steps is None or self.temp_warmup_steps <= 0:
                # Full cosine annealing from temp_high down to base_temp
                progress = min(float(epoch) / self.nsteps, 1.0)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
                return self.base_temp + (self.temp_high - self.base_temp) * cosine
            else:
                # Two-phase: warmup then anneal
                if epoch < self.temp_warmup_steps:
                    progress = float(epoch) / self.temp_warmup_steps
                    cosine = 0.5 * (1 - math.cos(math.pi * progress))
                    return self.temp_schedule_initial + (self.temp_high - self.temp_schedule_initial) * cosine
                else:
                    remaining = float(self.nsteps - self.temp_warmup_steps)
                    if remaining <= 0:
                        return self.base_temp
                    progress = float(epoch - self.temp_warmup_steps) / remaining
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    return self.base_temp + (self.temp_high - self.base_temp) * cosine
        # Legacy: warmup-only, then fixed base_temp
        if self.temp_warmup_steps is None or self.temp_warmup_steps == 0:
            return self.base_temp
        progress = min(float(epoch) / self.temp_warmup_steps, 1.0)
        cosine = 0.5 * (1 - math.cos(math.pi * progress))
        return self.temp_schedule_initial + (self.base_temp - self.temp_schedule_initial) * cosine

    def run(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Execute the training loop, returning the trained mechanism and rollout data."""
        final = self._load_final_if_exists()
        if final is not None:
            logger.info("Final mechanism found; returning it.")
            mech_data = final.compute_mechanism(self.sample, already_sorted=self.already_sorted)
            return final, mech_data

        ck = self._load_latest_checkpoint()
        if ck is not None:
            mech, opt, sch = ck
            if mech is not None:
                self.mechanism = mech
            if opt is not None:
                self.optimizer = opt
            if sch is not None:
                self.scheduler = sch
            logger.info(f"Starting from epoch {self.start_epoch}")

        kernel_tag = getattr(self.mechanism, "kernel_name", "kernel")
        cost_tag = getattr(self.mechanism, "cost_fn_name", "cost_fn")
        prefix = os.path.join(
            self.writing_dir, f"{self.dim_tag}_{self.run_id}_{kernel_tag}_{cost_tag}_"
        )
        remaining_epochs = self.nsteps - self.start_epoch

        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        with feedback.LiveOrJupyter() as live:
            for epoch in range(remaining_epochs):
                epoch_global = epoch + self.start_epoch
                self.mechanism.temp = self._compute_temp(epoch_global)
                self.optimizer.zero_grad()
                mechanism_data = self.mechanism.compute_mechanism(self.sample, mode=self.mode, already_sorted=self.already_sorted)

                # Evaluate all constraint functions on the current batch.
                constraint_vals = [fn(self.mechanism, mechanism_data).reshape(-1) for fn in self.constraint_fns]
                violations = [(-c).clamp_min(0.0) for c in constraint_vals]
                penalties = torch.stack([barrier(c, epsilon=self.epsilon, reduction="sum") for c in constraint_vals]) if constraint_vals else torch.tensor([])

                with torch.no_grad():
                    mean_violation2 = torch.stack([viol.pow(2).mean() for viol in violations]) if violations else torch.tensor([])
                    if mean_violation2.numel() > 0:
                        self.violation2_ema = self.beta_ema * self.violation2_ema + (1 - self.beta_ema) * mean_violation2
                        scale = 1.0 + self.alpha * (self.violation2_ema / (self.epsilon**2 + 1e-12))
                        target = self.base_penalty * scale
                        self.penalty_factors = self.beta_ema * self.penalty_factors + (1.0 - self.beta_ema) * target
                        self.penalty_factors.clamp_(min=self.min_penalty, max=self.max_penalty)
                        minimum_constraint_value = [v.min().item() for v in constraint_vals]
                    else:
                        minimum_constraint_value = []

                # Compute the aggregate penalty-weighted gain to bias optimization.
                weighted_penalties = (penalties * self.penalty_factors).sum() if penalties.numel() > 0 else torch.tensor(0.0)
                if penalties.numel() > 0 and penalties.isnan().any():
                    logger.error(
                        f"Epoch {epoch_global}: penalties became nan {penalties.detach().cpu().numpy()} with constraints being {constraint_vals.detach().cpu().numpy() if constraint_vals.numel()>0 else 'N/A'}"
                    )

                mean_profit = mechanism_data["profits"].mean()
                mean_profit_per_good = mean_profit / self.mechanism.num_dims
                # Track best mechanism by profit per good
                profit_val = mean_profit_per_good.item()
                if profit_val > self.best_profit_per_good:
                    self.best_profit_per_good = profit_val
                    self.best_state = {k: v.detach().clone() for k, v in self.mechanism.state_dict().items()}
                g = mean_profit + weighted_penalties
                g_per_good = g / self.mechanism.num_dims
                l = -g
                l.backward()
                l_scaler = l.item()
                self.ls.append(l_scaler)

                # Track gradient norm for logging/clipping purposes.
                total_gradient_norm = 0.0
                for p in self.mechanism.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_gradient_norm += param_norm.item() ** 2
                total_gradient_norm = total_gradient_norm ** 0.5
                max_clipping_norm = getattr(self, "max_clipping_norm", 5)
                max_logging_norm = 100
                torch.nn.utils.clip_grad_norm_(self.mechanism.parameters(), max_clipping_norm)
                if total_gradient_norm > max_logging_norm:
                    logger.warning(f"Epoch {epoch_global}: Gradient norm exceeded {max_logging_norm}: {total_gradient_norm:.4f}")

                self.optimizer.step()
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        try:
                            self.scheduler.step(g_per_good.item())
                        except Exception:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()

                last_mean_max_weight = getattr(self.mechanism, "_last_mean_max_weight", float("nan"))
                if (self.nmodes > 1) and (self.mode == self.modes[0]) and (not math.isnan(last_mean_max_weight)):
                    if last_mean_max_weight >= self.switch_threshold:
                        self.above_thresh_streak += 1
                    else:
                        self.above_thresh_streak = 0

                    if self.above_thresh_streak >= self.switch_patience:
                        self.mode = self.modes[1]
                        self.optimizer = self.optimizers[self.mode]
                        self.scheduler = self.schedulers[self.mode]
                        logger.info(f"🔀 Switched to {self.mode} at epoch {epoch_global} (mean_max≈{last_mean_max_weight:.3f}) based on max_weight")

                optimization_data = {
                    "mean_violation2": mean_violation2 if "mean_violation2" in locals() else torch.tensor([]),
                    "minimum_constraint_value": minimum_constraint_value,
                    "penalty": penalties,
                    "penalty_factors": self.penalty_factors,
                    "penalized gain": g,
                    "l_scaler": l_scaler,
                    "epoch": epoch_global,
                    "total_gradient_norm": total_gradient_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "last_mean_max_weight": last_mean_max_weight,
                }
                if self.use_wandb:
                    log_data = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in optimization_data.items()}
                    try:
                        self._wandb.log(log_data, step=epoch_global)
                    except Exception:
                        logger.debug("wandb.log failed during training step")

                # Check for convergence by monitoring stabilization of loss signal.
                if epoch >= self.window:
                    recent = self.ls[-self.window:]
                    max_diff = max(abs(recent[i] - recent[i - 1]) for i in range(1, self.window))
                    relative_max_diff = max_diff / abs(recent[-1])
                    if relative_max_diff < self.convergence_tolerance:
                        if self.mode == self.modes[-1]:
                            logger.info(f"Converged at epoch {epoch_global} (loss stable over last {self.window} epochs)")
                            if self.use_wandb:
                                try:
                                    self._wandb.log({"convergence_epoch": epoch_global}, step=epoch_global)
                                except Exception:
                                    logger.debug("wandb.log failed during convergence logging")
                        elif (self.nmodes > 1) and (self.mode == self.modes[0]):
                            self.ls[-1] = 0
                            self.mode = self.modes[1]
                            self.optimizer = self.optimizers[self.mode]
                            self.scheduler = self.schedulers[self.mode]
                            logger.info(f"🔀 Switched to {self.mode} at epoch {epoch_global} (mean_max≈{last_mean_max_weight:.3f}) based on convergence")
                        break

                if epoch > 0 and (epoch % self.steps_per_snapshot == 0 or torch.isnan(l).any()):
                    self._save_snapshot(prefix, epoch_global)

                if epoch % self.steps_per_update == 0:
                    panel_data = {
                        **mechanism_data,
                        **{k: v for k, v in optimization_data.items() if k not in {"lr", "l_scaler"}},
                        **self.live_wandb_dict,
                        "dimensionality": self.mechanism.num_dims,
                        "temperature": float(self.mechanism.temp),
                        "profit_per_good": torch.round(mean_profit_per_good, decimals=4).item(),
                        "desc": self.writing_dir,
                    }
                    panel = feedback.make_status_panel(panel_data)
                    live.update(panel)
            else:
                logger.warning(f"Finished training for {self.nsteps} epochs without convergence.")
                self.mechanism.converged = False

        self.mechanism.steps_taken = epoch
        self.mechanism.max_steps = self.nsteps
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                logger.debug("wandb.finish failed")
        torch.save(self.mechanism, f"{prefix}final.pt")
        # Save and restore best checkpoint
        if self.best_state is not None:
            torch.save({"state_dict": self.best_state, "best_profit_per_good": self.best_profit_per_good}, f"{prefix}best.pt")
            logger.info(f"Best checkpoint saved with profit_per_good={self.best_profit_per_good:.6f}")
            self.mechanism.load_state_dict(self.best_state)
        mech_data = self.mechanism.compute_mechanism(self.sample, already_sorted=self.already_sorted)
        avg_profit = float(mech_data["profits"].mean().item())
        avg_revenue = float(mech_data["revenue"].mean().item())
        avg_v = float(mech_data["v"].mean().item())
        q = max(self.mechanism.num_dims, 1)
        summary = {
            "run_id": self.run_id,
            "kernel": getattr(self.mechanism, "kernel_name", "kernel"),
            "cost": getattr(self.mechanism, "cost_fn_name", "cost_fn"),
            "dims": self.mechanism.num_dims,
            "mean_profit_per_good": avg_profit / q,
            "revenue_per_good": avg_revenue / q,
            "utility_per_good": avg_v / q,
        }
        summary_path = os.path.join(self.writing_dir, f"{self.run_id}_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info(f"Saved summary JSON to {summary_path}")
        aggregate_path = os.path.join(self.writing_dir, "summary_all.json")
        if os.path.exists(aggregate_path):
            with open(aggregate_path, "r") as fh:
                aggregate = json.load(fh)
        else:
            aggregate = []
        aggregate.append(summary)
        with open(aggregate_path, "w") as fh:
            json.dump(aggregate, fh, indent=2)
        logger.info(f"Appended summary to {aggregate_path}")
        root_dir = os.path.dirname(os.path.abspath(self.writing_dir))
        global_path = os.path.join(root_dir, "summary_all.json")
        if os.path.exists(global_path):
            with open(global_path, "r") as fh:
                global_aggregate = json.load(fh)
        else:
            global_aggregate = []
        global_aggregate.append(summary)
        with open(global_path, "w") as fh:
            json.dump(global_aggregate, fh, indent=2)
        logger.info(f"Appended summary to {global_path}")
        return self.mechanism, mech_data

def run(
    sample: torch.Tensor,
    modes: List[str] = ["soft"],
    compile: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    optimizers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    schedulers_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    with_hook: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Utility entry point that builds a mechanism and trains it via ``Mechanism.fit``."""
    if model_kwargs is None:
        model_kwargs = {}
    mechanism = Mechanism.with_hooks(**model_kwargs) if with_hook else Mechanism(**model_kwargs)
    return mechanism.fit(
        sample,
        modes=modes,
        compile=compile,
        optimizers_kwargs_dict=optimizers_kwargs_dict,
        schedulers_kwargs_dict=schedulers_kwargs_dict,
        train_kwargs=train_kwargs,
    )
