import torch
import numpy as np
from torch.utils.data import DataLoader
import plotly.graph_objs as go

from cleanup_ssps.dataset import SSPDataset, _renorm
from cleanup_ssps.cleanup_methods import FlowMatching
from utils.evaluation_utils import compute_cleanup_baseline, compute_ssp_mean, make_unitary
from utils.wandb_utils import log_metrics


def _ode_init_from_noise(z_noise: torch.Tensor, z1: torch.Tensor, mix: float) -> torch.Tensor:
    """Build the tensor passed to the flow / FF. ``z_noise`` is always pure noise (from :class:`SSPDataset`); ``mix`` moves the start toward ``z1`` for eval sweeps only."""
    s = float(mix)
    if s <= 0.0:
        return z_noise
    if s >= 1.0:
        return z1
    return _renorm(s * z1 + (1.0 - s) * z_noise)

class EvaluationManager:
    def __init__(
        self,
        training_results,
        test_dir,
        device="cpu",
        signal_strengths=None,
        eval_steps=None,
        repeats=5,
        *,
        noise_type="uniform_hypersphere",
        target_type="coordinate",
    ):
        self.results           = training_results
        self.test_dir          = test_dir
        self.device            = device
        self.signal_strengths  = signal_strengths or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.eval_steps        = eval_steps or [1, 2, 5, 10, 50]
        self.repeats           = repeats
        self.noise_type        = noise_type
        self.target_type       = target_type

    # labels consistent with your taxonomy
    def _label(self, name, mode):
        if name.endswith("_FF"):         return "FeedForward"
        if mode == "geo_det":            return "GeoDetFM"
        if mode == "geo_amb_const":      return "GeoAmbConst (exact OT)"
        if mode == "geo_tan_const":      return "GeoTanConst (exact OT)"
        if mode == "geo_amb_sb":         return "GeoAmbSB (Sinkhorn)"
        if mode == "geo_tan_sb":         return "GeoTanSB (Sinkhorn)"
        if mode == "euc_det":            return "Det_CFM"
        if mode == "euc_ot":             return "OT_CFM (exact OT)"
        if mode == "euc_sb":             return "SB_CFM (Sinkhorn)"
        return f"{name} ({mode})"

    def evaluate_model(self, name, mode, model_obj, dataset, batch_size=128, N=10, *, init_mix: float = 0.0):
        """
        Returns per-sample mean and std of cosine similarities over the test set.
        Uses ODE sampling only (deterministic flows). No eval-time OT pairing.

        The dataloader yields pure-noise ``z0`` and target ``z1``. ``init_mix`` in
        ``[0, 1]`` blends them into the initial state for the model only
        (training-style starts use ``init_mix=0``).
        """
        if name.endswith("_FF"):
            ff_model = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
            ff_model = ff_model.to(self.device).eval()
        else:
            flow_model = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
            flow_model = flow_model.to(self.device).eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        sims_list = []

        use_sphere = mode.startswith("geo_")  # geodesic integration flag

        with torch.no_grad():
            for inputs, targets in loader:
                z_noise = inputs.squeeze(1).to(self.device)
                z1 = targets.squeeze(1).to(self.device)
                z_init = _ode_init_from_noise(z_noise, z1, init_mix)

                if name.endswith("_FF"):
                    preds = ff_model(z_init)
                else:
                    fm = FlowMatching(
                        model=flow_model,
                        sampling=mode,
                        num_steps=N,
                        device=self.device,
                        sigma_min=getattr(flow_model, "sigma_min", 0.1),  # harmless if unused
                    )
                    preds = fm.sample_ode(z_init=z_init, N=N, use_sphere=use_sphere)[-1]

                preds = make_unitary(preds)
                preds = preds / preds.norm(dim=1, keepdim=True)

                sims = torch.sum(preds * z1, dim=1)
                sims_list.append(sims.cpu())

        sims_all = torch.cat(sims_list)
        return sims_all.mean().item(), sims_all.std().item()

    def evaluate_noise_levels(self, ssp_space, N, batch_size=128):
        bl = compute_cleanup_baseline(
            ssp_space,
            ssp_dim=ssp_space.ssp_dim,
            snr=0.0,
            grid_resolution=64,
            method='sobol',
            num_trials=2000,
            device=self.device,
        )
        baseline_mean = bl["mean_cosine"]
        baseline_std = bl["std_cosine"]

        fig = go.Figure()
        fig.add_hline(
            y=baseline_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline ±1σ ({baseline_std:.3f})",
            annotation_position="bottom left"
        )

        for (name, mode), (model_obj, *_) in self.results.items():
            means, stds = [], []
            ds = SSPDataset(
                data_dir        = self.test_dir,
                ssp_dim         = ssp_space.ssp_dim,
                target_type     = self.target_type,
                noise_type      = self.noise_type,
                signal_strength = 0.0,
                mode            = 'test'
            )
            for sf in self.signal_strengths:
                steps = 1 if name.endswith("_FF") else N
                mean, std = self.evaluate_model(
                    name, mode, model_obj, ds, batch_size, N=steps, init_mix=sf
                )
                means.append(mean)
                stds.append(std)

            fig.add_trace(go.Scatter(
                x=self.signal_strengths,
                y=means,
                error_y=dict(type='data', array=stds),
                mode='lines+markers',
                name=self._label(name, mode)
            ))

        fig.update_layout(
            title=f"Avg Cosine vs Signal Strength @ N={N}",
            xaxis_title="Signal Strength",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )
        log_metrics({f"NoiseEval_N{N}": fig})

    def evaluate_steps(self, ssp_space, signal_strength, batch_size=128):
        fig = go.Figure()
        for (name, mode), (model_obj, *_) in self.results.items():
            means, stds = [], []
            ds = SSPDataset(
                data_dir        = self.test_dir,
                ssp_dim         = ssp_space.ssp_dim,
                target_type     = self.target_type,
                noise_type      = self.noise_type,
                signal_strength = 0.0,
                mode            = 'test'
            )
            for N in self.eval_steps:
                if name.endswith("_FF") and N > 1:
                    means.append(None)
                    stds.append(None)
                    continue

                steps = 1 if name.endswith("_FF") else N
                mean, std = self.evaluate_model(
                    name, mode, model_obj, ds, batch_size, N=steps, init_mix=signal_strength
                )
                means.append(mean)
                stds.append(std)

            xs, ys, errs = [], [], []
            for s, m, e in zip(self.eval_steps, means, stds):
                if m is not None:
                    xs.append(s); ys.append(m); errs.append(e)

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                error_y=dict(type='data', array=errs),
                mode='lines+markers',
                name=self._label(name, mode)
            ))

        fig.update_layout(
            title=f"Avg Cosine vs Steps @ Signal Strength={signal_strength}",
            xaxis_title="Number of Steps",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )
        log_metrics({f"StepsEval_Signal{signal_strength}": fig})

    def run_all(self, ssp_space, batch_size=128):
        for N in self.eval_steps:
            self.evaluate_noise_levels(ssp_space, N=N, batch_size=batch_size)

        for sf in self.signal_strengths:
            self.evaluate_steps(ssp_space, signal_strength=sf, batch_size=batch_size)

        grid_resolutions = [16, 32, 64, 128, 256]
        fig_base = go.Figure()

        for gr in grid_resolutions:
            means, stds = [], []
            for sf in self.signal_strengths:
                bl = compute_cleanup_baseline(
                    ssp_space,
                    ssp_dim         = ssp_space.ssp_dim,
                    snr             = sf,
                    grid_resolution = gr,
                    method          = 'sobol',
                    num_trials      = 100,
                    device          = self.device,
                )
                means.append(bl["mean_cosine"])
                stds.append(bl["std_cosine"])

            fig_base.add_trace(go.Scatter(
                x=self.signal_strengths,
                y=means,
                error_y=dict(type='data', array=stds),
                mode='lines+markers',
                name=f'Grid {gr}×{gr}'
            ))

        fig_base.update_layout(
            title="Cleanup Baseline: Avg Cosine vs Signal Strength",
            xaxis_title="Signal Strength",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )

        sims_euc = compute_ssp_mean(
            ssp_space,
            num_samples=2000,
            device=self.device
        )
        fig_base.add_hline(
            y=sims_euc,
            line_dash="dash",
            line_color="gray",
            annotation_text="Euclid‐mean",
            annotation_position="bottom left"
        )

        log_metrics({"Baseline_Cosine_by_Grid": fig_base})
