"""
Trainer for OTP-FM on Gaussian marginals.

Author(s): Raghav Kansal
"""

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments import Trainer
from experiments.gaussian import plotting
from experiments.plotting import plot_target_vs_learned


class GaussianTrainer(Trainer):
    """
    Trainer for Gaussian experiments with specialized trajectory plotting.

    Extends the base Trainer with:
    - Trajectory visualization in 1D Gaussian space
    - X_tk comparison plots
    - Training progression animations
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
        lr: float,
        x0s_for_trajectories: Tensor,  # Gaussian-specific
        # Training parameters
        epochs: int = 10,
        optimizer: str = "adam",
        grad_clip: float = 0.0,
        do_otp: bool = True,
        # Progressive loss weighting
        otp_alpha_type: str = "sigmoid",
        otp_alpha_slope: float = 6.0,
        otp_alpha_mean_scale: float = 1.0,
        # Sampling/evaluation
        sampling_steps: int = 50,
        ema_eval: bool = True,
        traj_skips: int | None = None,
        # Model
        potentials: OrderedDict | None = None,
        device: str = "cpu",
        # Gaussian-specific
        unnormalize_fn: Callable[[Tensor], Tensor] | None = None,
        x0s_transform_fn: Callable[[Tensor], Tensor] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        eval_num_steps: list[int] | None = None,
        eval_samples: Tensor | None = None,
        ot_coupling: bool = False,
    ):
        """
        Initialize the Gaussian trainer.

        Gaussian-specific Args:
            x0s_for_trajectories: Initial points for trajectory visualization (standard normal samples)
            unnormalize_fn: Function to unnormalize trajectory data for plotting
            x0s_transform_fn: Function to transform standard normal x0s to model input space
            plot_kwargs: Kwargs for plotting.plot_trajectories_middle_marginal_1d
            eval_num_steps: List of # of steps for evaluation plots
            eval_samples: Samples for evaluation
            ot_coupling: Whether using OT coupling
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            lr=lr,
            epochs=epochs,
            optimizer=optimizer,
            grad_clip=grad_clip,
            do_otp=do_otp,
            otp_alpha_type=otp_alpha_type,
            otp_alpha_slope=otp_alpha_slope,
            otp_alpha_mean_scale=otp_alpha_mean_scale,
            sampling_steps=sampling_steps,
            ema_eval=ema_eval,
            potentials=potentials,
            device=device,
        )

        self.x0s_for_trajectories = x0s_for_trajectories
        self.unnormalize_fn = unnormalize_fn
        self.x0s_transform_fn = x0s_transform_fn
        self.plot_kwargs = plot_kwargs or {}
        self.eval_samples = eval_samples.permute(1, 0, 2) if eval_samples is not None else None
        self.ot_coupling = ot_coupling

        # Gaussian-specific parameters
        self.eval_num_steps = eval_num_steps or [1, 2, 5, 10, 50]

        # Auto-compute traj_skips if not provided
        if traj_skips is None:
            if epochs <= 30:
                self.traj_skips = 1
            else:
                self.traj_skips = max(1, round(epochs / 30))
        else:
            self.traj_skips = traj_skips

        # X_tk plotting directory
        self.xtk_plot_dir: Path | None = None
        if self.potentials:
            self.xtk_plot_dir = self.save_dir / "xtk_plots"
            self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory plotting directory
        self.traj_plot_dir: Path | None = None
        if self.plot_kwargs:
            self.traj_plot_dir = self.save_dir / "trajectory_plots"
            self.traj_plot_dir.mkdir(parents=True, exist_ok=True)

            # Build plot_kwargs from potentials
            if self.potentials:
                tks = list(self.potentials.keys())
                potentials_list = list(self.potentials.values())

                lambda_type = potentials_list[0].lambda_type if potentials_list else None
                lambda_widths = [p.width for p in potentials_list]
                strengths = [p.strength for p in potentials_list]

                if len(tks) == 1:
                    tks = tks[0]
                    lambda_widths = lambda_widths[0]
                    strengths = strengths[0]

                self.plot_kwargs = {
                    **self.plot_kwargs,
                    "lambda_type": lambda_type,
                    "lambda_width": lambda_widths,
                    "wks": strengths,
                    "t_k": tks,
                }

        # Trajectory storage for animations
        self.epoch_trajectories_vcorr: list[np.ndarray] = []

    def _get_potential_name(self, potential) -> str:
        """Get the name of the potential, with OT coupling handling."""
        name = potential.__class__.__name__
        if name in ("IndependentPotential", "W2InfPotential"):
            return "W2" if self.ot_coupling else "W2inf"
        return name

    def on_train_start(self) -> None:
        """Save initial trajectories before training."""
        self.save_trajectories(self.x0s_for_trajectories, self.unnormalize_fn, epoch=0)

        # Get initial batch for X_tk comparison
        batch = next(iter(self.val_loader))
        batch = self._process_batch(batch).to(self.device)
        self.plot_xtk_comparison(0, batch)

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None) -> None:
        """Save trajectories and X_tk comparison after each epoch."""
        if (((epoch + 1) % self.traj_skips) == 0) or (epoch == self.epochs - 1):
            self.save_trajectories(self.x0s_for_trajectories, self.unnormalize_fn, epoch=epoch + 1)

        self.plot_xtk_comparison(epoch + 1, batch)

    @torch.no_grad()
    def save_trajectories(
        self,
        x0s: Tensor,
        unnormalize_fn: Callable[[Tensor], Tensor] | None = None,
        num_samples: int = 200,
        epoch: int = 0,
    ) -> tuple[np.ndarray, Tensor]:
        """
        Sample, save, and plot trajectories for visualization.

        Args:
            x0s: Initial points (standard normal samples)
            unnormalize_fn: Optional function to unnormalize the data
            num_samples: Number of samples to plot
            epoch: Current epoch number

        Returns:
            Tuple of (trajectories, t_eval)
        """
        self.model.eval()

        # Transform standard normal x0s to model input space
        x0s_model = self.x0s_transform_fn(x0s) if self.x0s_transform_fn else x0s

        norm_xs_vcorr, t_eval = self.model.sample(x0s_model, self.sampling_steps, ema=self.ema_eval)
        xs_vcorr = unnormalize_fn(norm_xs_vcorr) if unnormalize_fn else norm_xs_vcorr
        self.epoch_trajectories_vcorr.append(xs_vcorr.cpu().numpy())

        # Save trajectory plot as PDF
        if self.plot_kwargs:
            xs_for_plot = xs_vcorr.permute(1, 0, 2)[:num_samples].squeeze(-1).cpu().numpy()
            t_eval_np = t_eval.cpu().numpy() if hasattr(t_eval, "cpu") else np.asarray(t_eval)
            pname = self._get_potential_name(list(self.potentials.values())[0])
            rd = plotting.distD_labels.get(pname, pname)
            plotting.plot_trajectories_middle_marginal_1d(
                **self.plot_kwargs,
                x0s=x0s[:num_samples].squeeze().cpu().numpy(),
                xs=xs_for_plot,
                t_eval=t_eval_np,
                title=rd,
                plot_dir=self.traj_plot_dir,
                name=f"traj_epoch_{epoch:03d}",
                show=False,
                close=True,
            )
            # Save trajectories
            torch.save(xs_vcorr.cpu(), self.traj_plot_dir / f"xs_vcorr_epoch_{epoch:03d}.pt")

            if epoch == 0:
                torch.save(t_eval.cpu(), self.traj_plot_dir / "t_eval.pt")

        return xs_vcorr, t_eval

    @torch.no_grad()
    def plot_xtk_comparison(
        self, epoch: int, batch: Tensor | None = None, show: bool = False
    ) -> None:
        """
        Plot learned vs base X_tk comparison for current epoch.

        Args:
            epoch: Current epoch number
            batch: Batch tensor of shape (batch_size, num_marginals, d)
            show: Whether to display the plot
        """
        if not self.potentials:
            return

        otp_alpha = self.curriculum(epoch * len(self.train_loader))

        if batch is None:
            batch = self.eval_samples
        elif self.eval_samples is not None:
            batch = torch.cat([self.eval_samples, batch], dim=0)

        if batch is None:
            return

        plot_target_vs_learned(
            model=self.model,
            batch=batch,
            potentials=self.potentials,
            otp_alpha=otp_alpha,
            n_samples=len(self.eval_samples) if self.eval_samples is not None else 5,
            name=f"xtk_epoch_{epoch:03d}",
            plot_dir=self.xtk_plot_dir,
            show=show,
            close=not show,
            device=self.device,
        )

    @torch.no_grad()
    def plot_trajectories(self, num_samples: int = 200, show: bool = False) -> None:
        """
        Evaluate model by plotting trajectories at various step counts.

        Args:
            num_samples: Number of samples to use for visualization
            show: Whether to display the plot
        """
        if not self.plot_kwargs:
            self.logger.info("No plot_kwargs provided, skipping evaluation plots")
            return

        self.model.eval()
        x0s = self.x0s_for_trajectories[:num_samples]

        # Transform standard normal x0s to model input space
        x0s_model = self.x0s_transform_fn(x0s) if self.x0s_transform_fn else x0s

        for num_steps in self.eval_num_steps:
            norm_xs, t_eval = self.model.sample(x0s_model, num_steps, ema=self.ema_eval)
            xs = self.unnormalize_fn(norm_xs) if self.unnormalize_fn else norm_xs

            xs_for_plot = xs.permute(1, 0, 2).squeeze(-1).cpu().numpy()
            t_eval_np = t_eval.cpu().numpy() if hasattr(t_eval, "cpu") else np.asarray(t_eval)

            pname = self._get_potential_name(list(self.potentials.values())[0])
            rd = plotting.distD_labels.get(pname, pname)
            plotting.plot_trajectories_middle_marginal_1d(
                **self.plot_kwargs,
                x0s=x0s.squeeze().cpu().numpy(),
                xs=xs_for_plot,
                t_eval=t_eval_np,
                name=f"trajectories_numsteps_{num_steps}",
                title=rd,
                plot_dir=self.save_dir,
                show=show and (num_steps in [1, 50]),
            )

    def create_animations(self, num_samples: int = 200, duration: int = 500) -> None:
        """
        Create animated GIFs from stored epoch trajectories.

        Args:
            num_samples: Number of samples to use in animation
            duration: Duration per frame in milliseconds
        """
        if not self.plot_kwargs:
            self.logger.info("No plot_kwargs provided, skipping animations")
            return

        # Get t_eval for animations
        x0s_model = (
            self.x0s_transform_fn(self.x0s_for_trajectories[:1])
            if self.x0s_transform_fn
            else self.x0s_for_trajectories[:1]
        )
        _, stored_t_eval = self.model.sample(x0s_model, self.sampling_steps, ema=self.ema_eval)
        t_eval_np = (
            stored_t_eval.cpu().numpy()
            if hasattr(stored_t_eval, "cpu")
            else np.asarray(stored_t_eval)
        )

        x0s_np = self.x0s_for_trajectories[:num_samples].squeeze().cpu().numpy()
        plot_kwargs_with_t_and_x0s = {**self.plot_kwargs, "t_eval": t_eval_np, "x0s": x0s_np}

        # Animation for model.sample
        if self.epoch_trajectories_vcorr:
            plotting.create_trajectory_animation(
                plotting.plot_trajectories_middle_marginal_1d,
                plot_kwargs_with_t_and_x0s,
                [
                    np.transpose(x, (1, 0, 2))[:num_samples].squeeze(-1)
                    for x in self.epoch_trajectories_vcorr
                ],
                skip_epochs=self.traj_skips,
                plot_dir=self.save_dir,
                name="trajectories_animation.gif",
                duration=duration,
            )

    def post_training(
        self,
        num_samples: int = 200,
        duration: int = 500,
        show: bool = False,
        create_animation: bool = True,
    ) -> Path:
        """
        Run all post-training tasks.

        Args:
            num_samples: Number of samples for evaluation/animation
            duration: Duration per frame in animations (ms)
            show: Whether to display plots
            create_animation: Whether to create animations

        Returns:
            Path to saved model checkpoint
        """
        self.plot_losses(show=show)
        self.save_losses_csv()
        save_path = self.save_checkpoint("model.pt")
        self.logger.info(f"Model saved to {save_path}")

        self.plot_trajectories(num_samples=num_samples, show=show)

        if create_animation:
            self.create_animations(num_samples=num_samples, duration=duration)

        return save_path
