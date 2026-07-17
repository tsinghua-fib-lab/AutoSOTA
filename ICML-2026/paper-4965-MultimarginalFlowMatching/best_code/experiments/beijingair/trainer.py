"""
Beijing air quality specific trainer.

Author(s): Raghav Kansal
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments import Trainer
from experiments.beijingair import plotting
from experiments.beijingair.data import DEFAULT_HOLDOUT_TIMES, DEFAULT_TRAIN_TIMES
from experiments.evaluation import compute_w2_distance
from experiments.plotting import plot_target_vs_learned


class BeijingTrainer(Trainer):
    """Trainer for Beijing air quality PM2.5 experiments."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
        lr: float,
        # Training parameters
        epochs: int = 10,
        optimizer: str = "adam",
        grad_clip: float = 0.0,
        weight_decay: float = 0.0,
        lr_schedule: str | None = None,
        do_otp: bool = True,
        # Progressive loss weighting
        otp_alpha_type: str = "sigmoid",
        otp_alpha_slope: float = 6.0,
        otp_alpha_mean_scale: float = 1.0,
        # Sampling/evaluation
        sampling_steps: int = 50,
        ema_eval: bool = True,
        # Model
        potentials: OrderedDict | None = None,
        device: str = "cpu",
        # Beijing-specific
        scaler=None,
        reshuffle_each_epoch: bool = True,
        marginals: dict[int, Tensor] | None = None,
        train_times: list[int] | None = None,
        holdout_times: list[int] | None = None,
        eval_n_steps: int = 20,
        eval_num_samples: int = 1000,
        traj_skips: int | None = None,
        animation_duration: int = 500,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            lr=lr,
            epochs=epochs,
            optimizer=optimizer,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            do_otp=do_otp,
            otp_alpha_type=otp_alpha_type,
            otp_alpha_slope=otp_alpha_slope,
            otp_alpha_mean_scale=otp_alpha_mean_scale,
            sampling_steps=sampling_steps,
            ema_eval=ema_eval,
            potentials=potentials,
            device=device,
        )

        self.scaler = scaler
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.marginals = marginals
        self.train_times = train_times or DEFAULT_TRAIN_TIMES
        self.holdout_times = holdout_times or DEFAULT_HOLDOUT_TIMES
        self.eval_n_steps = eval_n_steps
        self.eval_num_samples = eval_num_samples
        self.animation_duration = animation_duration

        if traj_skips is None:
            self.traj_skips = 1 if epochs <= 30 else max(1, math.ceil(epochs / 30))
        else:
            self.traj_skips = traj_skips

        self.epoch_trajectories: list[np.ndarray] = []
        self.trajectory_t_eval: np.ndarray | None = None

        self.xtk_plot_dir = self.save_dir / "xtk_plots"
        self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

        self._init_metrics()

        # Log configuration
        self.logger.info("Beijing Trainer initialized")
        self.logger.info(f"  Training times: {self.train_times}")
        self.logger.info(f"  Holdout times: {self.holdout_times}")
        if reshuffle_each_epoch:
            self.logger.info("  Reshuffling each epoch")

    def _get_underlying_dataset(self):
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def _init_metrics(self):
        """Initialize W2 tracking."""
        all_times = sorted(set(self.train_times + self.holdout_times))
        for t in all_times:
            if t != 0:
                self.losses[f"w2_t{t}"] = []
        self.losses["w2_rest"] = []
        self.losses["metric_epochs"] = []

    def on_train_start(self):
        self._save_epoch_trajectories(epoch=0)
        self._compute_metrics(epoch=0)
        self._plot_xtk_comparison(epoch=0)

    def on_epoch_start(self, epoch: int, batch: Tensor | None = None):
        """Reshuffle sample pairings at the start of each epoch if enabled."""
        if self.reshuffle_each_epoch:
            dataset = self._get_underlying_dataset()
            if hasattr(dataset, "reshuffle"):
                dataset.reshuffle()

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None):
        if ((epoch + 1) % self.traj_skips == 0) or (epoch == self.epochs - 1):
            self._save_epoch_trajectories(epoch=epoch + 1)
            self._compute_metrics(epoch=epoch + 1)
            self._plot_xtk_comparison(epoch=epoch + 1)

    @torch.no_grad()
    def _save_epoch_trajectories(self, epoch: int):
        if self.marginals is None or self.train_times is None:
            return

        self.model.eval()
        source_time = min(self.train_times)
        source = self.marginals[source_time]
        num_samples = min(self.eval_num_samples, len(source))
        x0 = source[:num_samples].to(self.device)

        trajectories, t_eval = self.model.sample(x0, n_steps=self.eval_n_steps, ema=self.ema_eval)

        self.epoch_trajectories.append(trajectories.cpu().numpy())
        if self.trajectory_t_eval is None:
            self.trajectory_t_eval = t_eval.cpu().numpy()

        traj_dir = self.save_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            traj_dir / f"trajectories_epoch{epoch:04d}.npz",
            trajectories=trajectories.cpu().numpy(),
            t_eval=t_eval.cpu().numpy(),
        )

    @torch.no_grad()
    def _plot_xtk_comparison(self, epoch: int):
        if self.potentials is None or len(self.potentials) == 0:
            return

        dataset = self._get_underlying_dataset()
        if hasattr(dataset, "get_ot_aligned_samples"):
            try:
                batch = dataset.get_ot_aligned_samples(n_samples=5)
            except ValueError:
                batch = self._get_random_samples(5)
        else:
            batch = self._get_random_samples(5)

        otp_alpha = self.curriculum(epoch * len(self.train_loader))

        plot_target_vs_learned(
            model=self.model,
            batch=batch,
            potentials=self.potentials,
            otp_alpha=otp_alpha,
            n_samples=5,
            name=f"xtk_epoch{epoch:04d}",
            plot_dir=self.xtk_plot_dir,
            show=False,
            device=self.device,
        )

    def _get_random_samples(self, n_samples: int) -> torch.Tensor:
        samples = []
        for t in sorted(self.train_times):
            marginal = self.marginals[t]
            indices = np.random.choice(len(marginal), size=n_samples, replace=False)
            samples.append(marginal[indices].cpu())
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def _compute_metrics(self, epoch: int):
        """Compute W2 metrics at all time points."""
        if self.marginals is None or self.train_times is None:
            return

        if not self.epoch_trajectories or self.trajectory_t_eval is None:
            return

        trajectories = self.epoch_trajectories[-1]
        t_eval = self.trajectory_t_eval

        # Time normalization
        time_min = min(self.train_times)
        time_max = max(self.train_times)

        def normalize_time(t: int) -> float:
            return (t - time_min) / (time_max - time_min)

        all_times = sorted(self.marginals.keys())

        for t in all_times:
            if t == 0:
                continue

            t_norm = normalize_time(t)
            ground_truth = self.marginals[t]
            if self.scaler is not None:
                ground_truth = torch.from_numpy(
                    self.scaler.inverse_transform(ground_truth.cpu().numpy())
                )

            target_idx = np.argmin(np.abs(t_eval - t_norm))
            generated = torch.from_numpy(trajectories[target_idx])
            if self.scaler is not None:
                generated = torch.from_numpy(self.scaler.inverse_transform(generated.numpy()))

            w2 = compute_w2_distance(generated, ground_truth)
            self.losses[f"w2_t{t}"].append(w2)

        # Combined metric on training times (excluding t=0)
        train_w2 = [
            self.losses[f"w2_t{t}"][-1]
            for t in self.train_times
            if t != 0 and self.losses.get(f"w2_t{t}")
        ]
        if train_w2:
            self.losses["w2_rest"].append(np.mean(train_w2))

        self.losses["metric_epochs"].append(epoch)

        # Log summary for holdout times
        log_parts = [f"Epoch {epoch} W2:"]
        for t in self.holdout_times:
            if self.losses.get(f"w2_t{t}") and self.losses[f"w2_t{t}"]:
                log_parts.append(f"t{t}={self.losses[f'w2_t{t}'][-1]:.4f}")
        self.logger.info(" ".join(log_parts))

    def post_training(self, show: bool = False, create_animation: bool = True) -> Path:
        save_path = super().post_training(show=show)

        if self.marginals is not None and self.epoch_trajectories:
            self._plot_final_trajectories()

        if create_animation and self.epoch_trajectories:
            self.create_animations()

        self.append_to_master_csv()

        return save_path

    def _plot_final_trajectories(self):
        if not self.epoch_trajectories or self.trajectory_t_eval is None:
            return

        trajectories = torch.from_numpy(self.epoch_trajectories[-1])
        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.plot_trajectories(
            trajectories=trajectories,
            t_eval=self.trajectory_t_eval,
            ground_truth_marginals=gt_marginals,
            save_path=self.save_dir / "trajectories_1d.pdf",
            show=False,
        )

    def create_animations(self, num_trajectories: int | None = None):
        """Create animated GIF showing trajectory evolution across training."""
        if not self.epoch_trajectories or self.marginals is None or self.trajectory_t_eval is None:
            self.logger.warning("Skipping animation: missing data")
            return

        num_trajectories = num_trajectories or self.eval_num_samples
        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.create_trajectory_animation(
            epoch_trajectories=self.epoch_trajectories,
            ground_truth_marginals=gt_marginals,
            trajectory_t_eval=self.trajectory_t_eval,
            save_path=self.save_dir / "trajectories_animation.gif",
            traj_skips=self.traj_skips,
            num_trajectories=num_trajectories,
            duration=self.animation_duration,
        )

    def plot_losses(self, log: bool = False, show: bool = False):
        """Override to use Beijing-specific loss plotting."""
        plotting.plot_losses(
            self.losses,
            name="losses" + ("_log" if log else ""),
            plot_dir=self.save_dir,
            log=log,
            show=show,
        )

    def append_to_master_csv(self, master_csv_path: Path | None = None):
        """Append best epoch metrics to master results CSV."""
        if master_csv_path is None:
            master_csv_path = self.save_dir.parent.parent / "master_results.csv"

        model_dir = f"{self.save_dir.parent.name}/{self.save_dir.name}"

        # Find best epoch (lowest average W2 across holdout times)
        holdout_time_keys = [f"t{t}" for t in self.holdout_times]
        w2_cols = [f"w2_{t}" for t in holdout_time_keys]

        if not all(self.losses.get(col) for col in w2_cols):
            self.logger.warning("Cannot append to master CSV: missing W2 metrics")
            return

        # Calculate average W2 across holdout times for each epoch
        n_epochs = len(self.losses[w2_cols[0]])
        avg_w2_per_epoch = []
        for i in range(n_epochs):
            avg = np.mean([self.losses[col][i] for col in w2_cols if self.losses.get(col)])
            avg_w2_per_epoch.append(avg)

        best_idx = int(np.argmin(avg_w2_per_epoch))
        best_epoch = self.losses.get("metric_epochs", list(range(n_epochs)))[best_idx]

        row = {
            "model_dir": model_dir,
            "best_epoch": best_epoch,
            "avg_w2_holdout": avg_w2_per_epoch[best_idx],
        }

        # Add individual W2 scores at best epoch
        all_times = sorted(set(self.train_times + self.holdout_times))
        for t in all_times:
            if t == 0:
                continue
            col = f"w2_t{t}"
            if col in self.losses and self.losses[col]:
                row[col] = self.losses[col][best_idx]

        df_row = pd.DataFrame([row])

        if master_csv_path.exists():
            existing_df = pd.read_csv(master_csv_path)
            combined_df = pd.concat([existing_df, df_row], ignore_index=True)
            combined_df.to_csv(master_csv_path, index=False)
        else:
            master_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(master_csv_path, index=False)

        self.logger.info(f"Appended results to {master_csv_path}")
