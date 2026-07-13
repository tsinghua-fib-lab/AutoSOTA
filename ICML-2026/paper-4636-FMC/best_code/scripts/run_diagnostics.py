#!/usr/bin/env python3
"""Diagnostic script for fm_post_transform component analysis.

Loads existing trained models and runs diagnostics:
1. Component ablation (FMPE@y, denoiser+NPE, plug_y, full fm_pt)
2. NPE-PFN preprocessing analysis
3. Simulation model quality checks
4. Rescaling verification
5. Plots (pairplots, training losses, ablation bars, simulation quality)

Usage:
    python scripts/run_diagnostics.py --experiment diagnostic_benchmark
    python scripts/run_diagnostics.py --experiment diagnostic_benchmark --tasks high_dim_gaussian
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpointing import CheckpointManager, ResultsPath  # noqa: E402


class DenoiserProposalWrapper:
    """Wraps denoiser + proposal for sampling without theta flow."""

    def __init__(self, denoiser, proposal):
        self.denoiser = denoiser
        self.proposal = proposal

    def sample(self, y, nsamples, device, **kwargs):
        source = self.denoiser.sample_base(y, nsamples)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)

        x_tilde = self.denoiser.sample(
            source.reshape(-1, *source.shape[2:]),
            cond.reshape(-1, *cond.shape[2:]),
            device,
            only_last=True,
            num_steps=20,
        )
        x_tilde = x_tilde.reshape(nsamples, y.shape[0], *self.denoiser.dim)
        x_flat = x_tilde.reshape(-1, x_tilde.shape[-1])

        samples = self.proposal.sample(x_flat, 1, device).squeeze(0)
        return samples.reshape(nsamples, y.shape[0], -1)


class DiagnosticRunner:
    """Runs diagnostic analysis on trained models."""

    def __init__(self, experiment_name: str, results_dir: str = "results"):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.paths = ResultsPath(self.results_dir, experiment_name)
        self.checkpoint_manager = CheckpointManager(self.paths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output directory
        self.diagnostics_dir = self.paths.base_path / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Simulator cache
        self._simulator_cache = {}

    def _get_simulator(self, task: str):
        if task in self._simulator_cache:
            return self._simulator_cache[task]
        try:
            from omegaconf import OmegaConf
            from simulator import get_simulator

            config_path = Path("configs/task") / f"{task}.yaml"
            if config_path.exists():
                task_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
                sim_cfg = task_cfg.get("simulator", {})
                simulator = get_simulator(sim_cfg.get("name", task), **sim_cfg.get("params", {}))
                self._simulator_cache[task] = simulator
                return simulator
        except Exception as e:
            print(f"  Warning: Could not load simulator for {task}: {e}")
        return None

    def _load_experiment_config(self) -> Dict[str, Any]:
        cfg = self.checkpoint_manager.load_experiment_config()
        if cfg is None:
            # Fallback: load from Hydra config
            try:
                from omegaconf import OmegaConf
                config_path = Path("configs/experiment") / f"{self.experiment_name}.yaml"
                if config_path.exists():
                    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
            except Exception:
                pass
            return {}
        return cfg

    def run_all(self, tasks: Optional[List[str]] = None):
        """Run all diagnostics."""
        cfg = self._load_experiment_config()
        if tasks is None:
            tasks = cfg.get("tasks", ["high_dim_gaussian", "pendulum"])

        seeds = cfg.get("seeds", [33, 43, 53])
        ncals = cfg.get("data", {}).get("calibration_sizes", [10, 50, 200, 1000])

        print(f"Running diagnostics for experiment: {self.experiment_name}")
        print(f"Tasks: {tasks}, Seeds: {seeds}, NCals: {ncals}")
        print(f"Device: {self.device}")
        print(f"Output: {self.diagnostics_dir}")
        print()

        for task in tasks:
            task_dir = self.diagnostics_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "plots").mkdir(exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}")

            # Diagnostic 1: Component ablation
            try:
                self.run_component_ablation(task, seeds, ncals)
            except Exception as e:
                print(f"  Component ablation failed: {e}")
                import traceback
                traceback.print_exc()

            # Diagnostic 2: NPE-PFN preprocessing check
            try:
                self.check_npe_pfn_preprocessing(task)
            except Exception as e:
                print(f"  NPE-PFN preprocessing check failed: {e}")

            # Diagnostic 3: Simulation model quality
            try:
                self.evaluate_simulation_quality(task)
            except Exception as e:
                print(f"  Simulation quality check failed: {e}")
                import traceback
                traceback.print_exc()

            # Diagnostic 4: Rescaling verification
            try:
                seed = seeds[0] if seeds else 33
                ncal = ncals[-1] if ncals else 200
                self.verify_rescaling(task, seed, ncal)
            except Exception as e:
                print(f"  Rescaling verification failed: {e}")

            # Diagnostic 5: Plots
            try:
                self.plot_training_losses(task)
            except Exception as e:
                print(f"  Training loss plots failed: {e}")

            try:
                self.generate_sample_plots(task, seeds, ncals)
            except Exception as e:
                print(f"  Sample plots failed: {e}")
                import traceback
                traceback.print_exc()

        # Summary report
        self.generate_report(tasks)

    def run_component_ablation(self, task: str, seeds: List[int], ncals: List[int]):
        """Evaluate 4 ablation components of fm_post_transform."""
        print(f"\n  --- Component Ablation for {task} ---")

        simulator = self._get_simulator(task)
        has_reference = simulator is not None and hasattr(simulator, "sample_reference_posterior")
        has_analytical = simulator is not None and hasattr(simulator, "post_dist")

        all_results = {}

        for seed in seeds:
            for ncal in ncals:
                key = f"seed{seed}_ncal{ncal}"
                results = {}

                # Component A: FMPE@y -- shared FMPE model, pass y as conditioning
                fmpe_model = self.checkpoint_manager.load_shared_simulation_model(task, "fmpe", self.device)
                if fmpe_model is not None:
                    results["fmpe_at_y"] = self._evaluate_component(
                        fmpe_model, task, simulator, has_reference, has_analytical, label="FMPE@y"
                    )

                # Component D: Full fm_pt
                fm_pt = self.checkpoint_manager.load_variant_calibration_model(
                    "fm_pt_base", task, seed, "fm_post_transform", ncal, self.device
                )
                if fm_pt is not None:
                    results["fm_pt"] = self._evaluate_component(
                        fm_pt, task, simulator, has_reference, has_analytical, label="fm_pt"
                    )

                    # Component B: Denoiser+NPE (no theta flow)
                    if hasattr(fm_pt, "denoiser") and fm_pt.denoiser is not None:
                        wrapper = DenoiserProposalWrapper(fm_pt.denoiser, fm_pt.proposal)
                        results["denoiser_npe"] = self._evaluate_component(
                            wrapper, task, simulator, has_reference, has_analytical, label="denoiser+NPE"
                        )

                # Component C: plug_y variant
                plug_y = self.checkpoint_manager.load_variant_calibration_model(
                    "fm_pt_plug_y", task, seed, "fm_post_transform_plug_y", ncal, self.device
                )
                if plug_y is not None:
                    results["plug_y"] = self._evaluate_component(
                        plug_y, task, simulator, has_reference, has_analytical, label="plug_y"
                    )

                if results:
                    all_results[key] = results
                    print(f"    {key}: {list(results.keys())}")

        # Save results
        output_path = self.diagnostics_dir / task / "ablation_metrics.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved ablation metrics to {output_path}")

        # Generate ablation bar plot
        self._plot_ablation_summary(task, all_results, ncals)

    def _evaluate_component(
        self,
        model,
        task: str,
        simulator,
        has_reference: bool,
        has_analytical: bool,
        label: str,
        n_obs: int = 10,
        n_samples: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate a single component model."""
        results = {}
        test_data = self.checkpoint_manager.load_shared_test_data(task)
        if test_data is None:
            print(f"    No test data for {task}")
            return results

        y_obs = test_data["y"][:n_obs]
        theta_true = test_data["theta"][:n_obs]

        # Sample from model
        try:
            with torch.no_grad():
                theta_samples = []
                for i in range(n_obs):
                    samples = model.sample(
                        y_obs[i:i+1].to(self.device), n_samples, self.device
                    ).cpu()
                    if samples.ndim == 3:
                        if samples.shape[0] == n_samples:
                            samples = samples[:, 0, :]
                        else:
                            samples = samples[0]
                    theta_samples.append(samples)
                theta_samples = torch.stack(theta_samples, dim=0)  # (n_obs, n_samples, theta_dim)
        except Exception as e:
            print(f"      {label}: sampling failed: {e}")
            return {"error": str(e)}

        # C2ST against reference posterior
        if has_reference:
            try:
                from sklearn.neural_network import MLPClassifier
                from sklearn.model_selection import cross_val_score

                c2st_values = []
                for i in range(n_obs):
                    ref = simulator.sample_reference_posterior(
                        num_samples=n_samples, observations=y_obs[i:i+1], misspecified=False
                    )
                    if ref.ndim == 3:
                        ref = ref[:, 0, :]

                    # Apply transform if available
                    if hasattr(simulator, "transform"):
                        ref = simulator.transform(ref)

                    model_s = theta_samples[i][:n_samples]
                    ref_s = ref[:n_samples]

                    if torch.isnan(model_s).any() or torch.isnan(ref_s).any():
                        continue

                    X = torch.cat([model_s, ref_s], dim=0).numpy()
                    y_labels = np.concatenate([np.zeros(len(model_s)), np.ones(len(ref_s))])
                    clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=42)
                    scores = cross_val_score(clf, X, y_labels, cv=3, scoring="accuracy")
                    c2st_values.append(scores.mean())

                if c2st_values:
                    results["cond_c2st"] = float(np.mean(c2st_values))
            except Exception as e:
                print(f"      {label}: C2ST failed: {e}")

        # true_lpp: true posterior log-prob on generated samples
        if has_analytical:
            try:
                true_lpp_values = []
                for i in range(n_obs):
                    true_post = simulator.post_dist(y_obs[i:i+1], misspecified=False)
                    method_samples_i = theta_samples[i]
                    log_probs = true_post.log_prob(method_samples_i)
                    true_lpp_values.append(log_probs.mean().item())
                results["true_lpp"] = float(np.mean(true_lpp_values))
            except Exception as e:
                print(f"      {label}: true_lpp failed: {e}")

        # lpp: model's own log_prob
        if hasattr(model, "log_prob"):
            try:
                lpp_values = []
                for i in range(n_obs):
                    lp = model.log_prob(
                        theta_true[i:i+1].to(self.device),
                        y_obs[i:i+1].to(self.device),
                    )
                    lpp_values.append(lp.mean().item())
                results["lpp"] = float(np.mean(lpp_values))
            except Exception as e:
                print(f"      {label}: lpp failed: {e}")

        # Conditional MMD
        try:
            mmd_values = []
            for i in range(n_obs):
                samples = theta_samples[i]
                true_theta = theta_true[i:i+1].expand(len(samples), -1)
                dist_sq = torch.cdist(samples, true_theta) ** 2
                k_ss = torch.exp(-torch.cdist(samples, samples) ** 2 / 2).mean()
                k_tt = torch.exp(-dist_sq / 2).mean()  # degenerate since all same
                k_st = torch.exp(-dist_sq / 2).mean()
                mmd = float(k_ss + k_tt - 2 * k_st)
                mmd_values.append(mmd)
            results["cond_mmd"] = float(np.mean(mmd_values))
        except Exception as e:
            print(f"      {label}: MMD failed: {e}")

        print(f"      {label}: {results}")
        return results

    def check_npe_pfn_preprocessing(self, task: str):
        """Report on NPE-PFN preprocessing differences."""
        print(f"\n  --- NPE-PFN Preprocessing Check for {task} ---")

        report = {
            "task": task,
            "notes": [],
        }

        # Check if NPE-PFN uses any preprocessing
        report["notes"].append(
            "NPE-PFN uses TabPFN internal preprocessing (standardization)."
        )
        report["notes"].append(
            "Other methods use configurable rescaling (z_score, whiten, none)."
        )
        report["notes"].append(
            "fm_post_transform variants use rescale='none' by default."
        )

        # Check test data dimensions
        test_data = self.checkpoint_manager.load_shared_test_data(task)
        if test_data is not None:
            report["theta_dim"] = test_data["theta"].shape[-1]
            report["obs_dim"] = list(test_data["y"].shape[1:])
            report["n_test"] = len(test_data["y"])

        output_path = self.diagnostics_dir / task / "npe_pfn_preprocessing.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved to {output_path}")

    def evaluate_simulation_quality(self, task: str):
        """Evaluate NPE/FMPE quality on both x and y data."""
        print(f"\n  --- Simulation Quality Check for {task} ---")

        simulator = self._get_simulator(task)
        has_reference = simulator is not None and hasattr(simulator, "sample_reference_posterior")
        has_analytical = simulator is not None and hasattr(simulator, "post_dist")

        if not has_reference and not has_analytical:
            print(f"    No reference posterior available for {task}, skipping quality check")
            return

        test_data = self.checkpoint_manager.load_shared_test_data(task)
        if test_data is None:
            print(f"    No test data for {task}")
            return

        n_obs = 5  # Use fewer obs since MCMC can be slow
        n_samples = 500
        results = {}

        for model_type in ["npe", "fmpe"]:
            model = self.checkpoint_manager.load_shared_simulation_model(task, model_type, self.device)
            if model is None:
                print(f"    {model_type.upper()} not found, skipping")
                continue

            for data_key, misspecified, label in [
                ("x", True, f"{model_type}_on_x"),
                ("y", False, f"{model_type}_on_y"),
            ]:
                obs = test_data[data_key][:n_obs]

                try:
                    with torch.no_grad():
                        theta_samples = []
                        for i in range(n_obs):
                            samples = model.sample(
                                obs[i:i+1].to(self.device), n_samples, self.device
                            ).cpu()
                            if samples.ndim == 3:
                                if samples.shape[0] == n_samples:
                                    samples = samples[:, 0, :]
                                else:
                                    samples = samples[0]
                            theta_samples.append(samples)
                        theta_samples = torch.stack(theta_samples, dim=0)
                except Exception as e:
                    print(f"    {label}: sampling failed: {e}")
                    continue

                entry = {}

                # C2ST against reference
                if has_reference:
                    try:
                        from sklearn.neural_network import MLPClassifier
                        from sklearn.model_selection import cross_val_score

                        c2st_values = []
                        for i in range(n_obs):
                            ref = simulator.sample_reference_posterior(
                                num_samples=n_samples,
                                observations=obs[i:i+1],
                                misspecified=misspecified,
                            )
                            if ref.ndim == 3:
                                ref = ref[:, 0, :]
                            if hasattr(simulator, "transform"):
                                ref = simulator.transform(ref)

                            model_s = theta_samples[i][:n_samples]
                            ref_s = ref[:n_samples]

                            if torch.isnan(model_s).any() or torch.isnan(ref_s).any():
                                continue

                            X = torch.cat([model_s, ref_s], dim=0).numpy()
                            y_labels = np.concatenate([np.zeros(len(model_s)), np.ones(len(ref_s))])
                            clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=42)
                            scores = cross_val_score(clf, X, y_labels, cv=3, scoring="accuracy")
                            c2st_values.append(scores.mean())

                        if c2st_values:
                            entry["cond_c2st"] = float(np.mean(c2st_values))
                    except Exception as e:
                        print(f"    {label}: C2ST failed: {e}")

                # true_lpp
                if has_analytical:
                    try:
                        true_lpp_values = []
                        for i in range(n_obs):
                            true_post = simulator.post_dist(obs[i:i+1], misspecified=misspecified)
                            log_probs = true_post.log_prob(theta_samples[i])
                            true_lpp_values.append(log_probs.mean().item())
                        entry["true_lpp"] = float(np.mean(true_lpp_values))
                    except Exception as e:
                        print(f"    {label}: true_lpp failed: {e}")

                results[label] = entry
                print(f"    {label}: {entry}")

        output_path = self.diagnostics_dir / task / "simulation_quality.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {output_path}")

        # Plot
        self._plot_simulation_quality(task, results)

    def verify_rescaling(self, task: str, seed: int, ncal: int):
        """Inspect rescaler modes across pipeline components."""
        print(f"\n  --- Rescaling Verification for {task} (seed={seed}, ncal={ncal}) ---")

        report = {"task": task, "seed": seed, "ncal": ncal, "components": {}}

        # Load fm_pt model
        fm_pt = self.checkpoint_manager.load_variant_calibration_model(
            "fm_pt_base", task, seed, "fm_post_transform", ncal, self.device
        )
        if fm_pt is not None:
            report["components"]["flow_theta.target_rescaler"] = type(
                fm_pt.posterior_transform.target_rescaler
            ).__name__
            report["components"]["flow_theta.cond_rescaler"] = type(
                fm_pt.posterior_transform.cond_rescaler
            ).__name__

            if fm_pt.denoiser is not None:
                report["components"]["flow_x.target_rescaler"] = type(
                    fm_pt.denoiser.target_rescaler
                ).__name__
                report["components"]["flow_x.cond_rescaler"] = type(
                    fm_pt.denoiser.cond_rescaler
                ).__name__
            else:
                report["components"]["flow_x.target_rescaler"] = "N/A (no denoiser)"
                report["components"]["flow_x.cond_rescaler"] = "N/A (no denoiser)"

            if hasattr(fm_pt.proposal, "model") and hasattr(fm_pt.proposal.model, "cond_rescaler"):
                report["components"]["proposal_rescaler"] = type(
                    fm_pt.proposal.model.cond_rescaler
                ).__name__
            elif hasattr(fm_pt.proposal, "cond_rescaler"):
                report["components"]["proposal_rescaler"] = type(
                    fm_pt.proposal.cond_rescaler
                ).__name__

            report["notes"] = [
                "When rescale='none': all rescalers should be NoOpRescaler.",
                "Inference chain in DualFlowPosteriorEstimator.sample():",
                "  1. y_to_x: denoiser samples x_tilde from p(x|y)",
                "  2. proposal.sample(x_tilde): NPE samples source theta from p(theta|x_tilde)",
                "  3. posterior_transform.rescale(source, y): rescales source before theta flow",
                "  4. posterior_transform.sample(source, y): theta flow refines source",
                "When rescale!='none': training uses unrescaled source in compute_loss_with_source,",
                "  but inference rescales source at step 3 -- potential mismatch.",
            ]
        else:
            report["error"] = "fm_pt model not found"

        output_path = self.diagnostics_dir / task / "rescaling_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved to {output_path}")
        for k, v in report.get("components", {}).items():
            print(f"    {k}: {v}")

    def plot_training_losses(self, task: str):
        """Plot training loss curves from training_log.json files."""
        print(f"\n  --- Training Loss Plots for {task} ---")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loss_data = {}

        # Simulation models
        for model_type in ["npe", "fmpe"]:
            log_path = self.paths.shared_simulation_model_path(task, model_type) / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                lh = log.get("loss_history", {})
                if lh.get("train") and lh.get("val"):
                    loss_data[model_type.upper()] = lh

        # Baselines (first seed, largest ncal)
        cfg = self._load_experiment_config()
        seeds = cfg.get("seeds", [33])
        ncals = cfg.get("data", {}).get("calibration_sizes", [200])
        seed = seeds[0]
        ncal = ncals[-1] if ncals else 200

        for baseline in ["dpe", "mf_npe"]:
            log_path = (
                self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
                / "training_log.json"
            )
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                lh = log.get("loss_history", {})
                if lh.get("train") and lh.get("val"):
                    loss_data[baseline.upper()] = lh

        # Variants
        for variant_name, method in [
            ("fm_pt_base", "fm_post_transform"),
            ("fm_pt_plug_y", "fm_post_transform_plug_y"),
        ]:
            model_dir = self.paths.variant_calibration_model_path(
                variant_name, task, seed, method, ncal
            )
            log_path = model_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                lh = log.get("loss_history", {})
                if lh.get("train") and lh.get("val"):
                    loss_data[variant_name] = lh

        if not loss_data:
            print("    No loss history data found")
            return

        n_plots = len(loss_data)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        axs_flat = axs.flatten()

        for idx, (name, lh) in enumerate(loss_data.items()):
            ax = axs_flat[idx]
            epochs = list(range(1, len(lh["train"]) + 1))
            ax.plot(epochs, lh["train"], label="Train", linewidth=0.8)
            ax.plot(epochs, lh["val"], label="Val", linewidth=0.8)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax.tick_params(labelsize=7)

        # Legend on last used axis
        axs_flat[n_plots - 1].legend(fontsize=7)

        # Hide unused axes
        for idx in range(n_plots, len(axs_flat)):
            axs_flat[idx].set_visible(False)

        fig.suptitle(f"Training Losses - {task}", fontsize=12)
        plt.tight_layout()
        output_path = self.diagnostics_dir / task / "plots" / "training_losses.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {output_path}")

    def generate_sample_plots(self, task: str, seeds: List[int], ncals: List[int]):
        """Generate pairplots comparing method samples vs reference posterior."""
        print(f"\n  --- Sample Plots for {task} ---")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        simulator = self._get_simulator(task)
        has_reference = simulator is not None and hasattr(simulator, "sample_reference_posterior")

        test_data = self.checkpoint_manager.load_shared_test_data(task)
        if test_data is None:
            print("    No test data")
            return

        seed = seeds[0]
        ncal = ncals[-1] if ncals else 200
        n_samples = 1000
        obs_idx = 0
        y_obs = test_data["y"][obs_idx:obs_idx+1]

        # Collect samples from different methods
        method_samples = {}

        # Reference posterior
        if has_reference:
            try:
                ref = simulator.sample_reference_posterior(
                    num_samples=n_samples, observations=y_obs, misspecified=False
                )
                if ref.ndim == 3:
                    ref = ref[:, 0, :]
                if hasattr(simulator, "transform"):
                    ref = simulator.transform(ref)
                method_samples["Reference"] = ref.numpy()
            except Exception as e:
                print(f"    Reference sampling failed: {e}")

        # fm_pt
        fm_pt = self.checkpoint_manager.load_variant_calibration_model(
            "fm_pt_base", task, seed, "fm_post_transform", ncal, self.device
        )
        if fm_pt is not None:
            try:
                with torch.no_grad():
                    s = fm_pt.sample(y_obs.to(self.device), n_samples, self.device).cpu()
                    if s.ndim == 3:
                        s = s[:, 0, :] if s.shape[0] == n_samples else s[0]
                    method_samples["fm_pt"] = s.numpy()
            except Exception as e:
                print(f"    fm_pt sampling failed: {e}")

        # plug_y
        plug_y = self.checkpoint_manager.load_variant_calibration_model(
            "fm_pt_plug_y", task, seed, "fm_post_transform_plug_y", ncal, self.device
        )
        if plug_y is not None:
            try:
                with torch.no_grad():
                    s = plug_y.sample(y_obs.to(self.device), n_samples, self.device).cpu()
                    if s.ndim == 3:
                        s = s[:, 0, :] if s.shape[0] == n_samples else s[0]
                    method_samples["plug_y"] = s.numpy()
            except Exception as e:
                print(f"    plug_y sampling failed: {e}")

        # FMPE@y
        fmpe = self.checkpoint_manager.load_shared_simulation_model(task, "fmpe", self.device)
        if fmpe is not None:
            try:
                with torch.no_grad():
                    s = fmpe.sample(y_obs.to(self.device), n_samples, self.device).cpu()
                    if s.ndim == 3:
                        s = s[:, 0, :] if s.shape[0] == n_samples else s[0]
                    method_samples["FMPE@y"] = s.numpy()
            except Exception as e:
                print(f"    FMPE@y sampling failed: {e}")

        # Baselines
        for baseline in ["dpe", "mf_npe"]:
            model = self.checkpoint_manager.load_shared_baseline(task, seed, baseline, ncal, self.device)
            if model is not None:
                try:
                    with torch.no_grad():
                        s = model.sample(y_obs.to(self.device), n_samples, self.device).cpu()
                        if s.ndim == 3:
                            s = s[:, 0, :] if s.shape[0] == n_samples else s[0]
                        method_samples[baseline.upper()] = s.numpy()
                except Exception as e:
                    print(f"    {baseline} sampling failed: {e}")

        if not method_samples:
            print("    No samples collected")
            return

        # Create pairplot
        theta_dim = list(method_samples.values())[0].shape[-1]
        n_methods = len(method_samples)

        if theta_dim <= 3:
            fig, axs = plt.subplots(
                theta_dim, theta_dim, figsize=(3 * theta_dim, 3 * theta_dim), squeeze=False
            )

            colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

            for mi, (name, samples) in enumerate(method_samples.items()):
                for i in range(theta_dim):
                    for j in range(theta_dim):
                        ax = axs[i][j]
                        if i == j:
                            ax.hist(
                                samples[:, i], bins=30, alpha=0.4,
                                color=colors[mi], label=name, density=True
                            )
                        elif i > j:
                            ax.scatter(
                                samples[:, j], samples[:, i],
                                alpha=0.15, s=2, color=colors[mi], label=name
                            )
                        else:
                            ax.set_visible(False)

                        if j == 0 and i > 0:
                            ax.set_ylabel(f"θ_{i}", fontsize=8)
                        if i == theta_dim - 1:
                            ax.set_xlabel(f"θ_{j}", fontsize=8)

            # Legend
            handles, labels = axs[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", fontsize=7)
            fig.suptitle(f"{task} - seed{seed} ncal{ncal} obs{obs_idx}", fontsize=10)
            plt.tight_layout()

            output_path = (
                self.diagnostics_dir / task / "plots"
                / f"pairplot_seed{seed}_ncal{ncal}_obs{obs_idx}.pdf"
            )
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved pairplot to {output_path}")
        else:
            # For high-dim theta, just plot marginals
            fig, axs = plt.subplots(1, min(theta_dim, 6), figsize=(12, 3), squeeze=False)
            colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
            for di in range(min(theta_dim, 6)):
                ax = axs[0][di]
                for mi, (name, samples) in enumerate(method_samples.items()):
                    ax.hist(samples[:, di], bins=30, alpha=0.4, color=colors[mi], label=name, density=True)
                ax.set_xlabel(f"θ_{di}", fontsize=8)
            axs[0][-1].legend(fontsize=6)
            fig.suptitle(f"{task} - Marginals", fontsize=10)
            plt.tight_layout()

            output_path = (
                self.diagnostics_dir / task / "plots"
                / f"pairplot_seed{seed}_ncal{ncal}_obs{obs_idx}.pdf"
            )
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved marginals plot to {output_path}")

    def _plot_ablation_summary(self, task: str, all_results: Dict, ncals: List[int]):
        """Generate ablation summary bar plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Extract metric to plot (prefer cond_c2st, fallback to cond_mmd)
        metric_key = None
        for key in ["cond_c2st", "cond_mmd", "true_lpp"]:
            for _, components in all_results.items():
                for _, metrics in components.items():
                    if key in metrics:
                        metric_key = key
                        break
                if metric_key:
                    break
            if metric_key:
                break

        if metric_key is None:
            print("    No ablation metrics to plot")
            return

        # Aggregate over seeds for each ncal
        component_names = ["fmpe_at_y", "denoiser_npe", "plug_y", "fm_pt"]
        component_labels = ["FMPE@y", "Denoiser+NPE", "plug_y", "Full fm_pt"]

        ncal_data = {}
        for ncal in ncals:
            ncal_data[ncal] = {c: [] for c in component_names}
            for key, components in all_results.items():
                if f"ncal{ncal}" in key:
                    for comp in component_names:
                        if comp in components and metric_key in components[comp]:
                            ncal_data[ncal][comp].append(components[comp][metric_key])

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(ncals))
        width = 0.18
        colors = plt.cm.Set2(np.linspace(0, 1, len(component_names)))

        for i, (comp, label) in enumerate(zip(component_names, component_labels)):
            means = []
            stds = []
            for ncal in ncals:
                vals = ncal_data[ncal][comp]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if len(vals) > 1 else 0)
            ax.bar(x + i * width, means, width, yerr=stds, label=label, color=colors[i], capsize=2)

        ax.set_xlabel("n_cal")
        ax.set_ylabel(metric_key)
        ax.set_title(f"Component Ablation - {task}")
        ax.set_xticks(x + width * (len(component_names) - 1) / 2)
        ax.set_xticklabels([str(n) for n in ncals])
        ax.legend(fontsize=8)
        plt.tight_layout()

        output_path = self.diagnostics_dir / task / "plots" / "ablation_summary.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ablation summary to {output_path}")

    def _plot_simulation_quality(self, task: str, results: Dict):
        """Generate simulation quality bar plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not results:
            return

        metric_key = None
        for key in ["cond_c2st", "true_lpp"]:
            for _, metrics in results.items():
                if key in metrics:
                    metric_key = key
                    break
            if metric_key:
                break

        if metric_key is None:
            print("    No simulation quality metrics to plot")
            return

        labels = list(results.keys())
        values = [results[label].get(metric_key, 0) for label in labels]

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric_key)
        ax.set_title(f"Simulation Quality - {task}")
        plt.tight_layout()

        output_path = self.diagnostics_dir / task / "plots" / "simulation_quality.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved simulation quality plot to {output_path}")

    def generate_report(self, tasks: List[str]):
        """Generate a text summary report."""
        print("\n  --- Generating Summary Report ---")

        lines = [
            f"Diagnostic Report: {self.experiment_name}",
            "=" * 60,
            "",
        ]

        for task in tasks:
            task_dir = self.diagnostics_dir / task
            lines.append(f"Task: {task}")
            lines.append("-" * 40)

            # Ablation metrics
            ablation_path = task_dir / "ablation_metrics.json"
            if ablation_path.exists():
                with open(ablation_path) as f:
                    ablation = json.load(f)
                lines.append("  Component Ablation:")
                for config_key, components in ablation.items():
                    lines.append(f"    {config_key}:")
                    for comp, metrics in components.items():
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
                        lines.append(f"      {comp}: {metrics_str}")

            # Simulation quality
            sim_path = task_dir / "simulation_quality.json"
            if sim_path.exists():
                with open(sim_path) as f:
                    sim_quality = json.load(f)
                lines.append("  Simulation Quality:")
                for label, metrics in sim_quality.items():
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
                    lines.append(f"    {label}: {metrics_str}")

            # Rescaling
            rescale_path = task_dir / "rescaling_report.json"
            if rescale_path.exists():
                with open(rescale_path) as f:
                    rescale = json.load(f)
                lines.append("  Rescaling:")
                for k, v in rescale.get("components", {}).items():
                    lines.append(f"    {k}: {v}")

            lines.append("")

        report_text = "\n".join(lines)
        report_path = self.diagnostics_dir / "summary_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics for fm_post_transform")
    parser.add_argument("--experiment", type=str, default="diagnostic_benchmark",
                        help="Experiment name")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to diagnose (default: all from experiment config)")
    args = parser.parse_args()

    runner = DiagnosticRunner(args.experiment, args.results_dir)
    runner.run_all(args.tasks)


if __name__ == "__main__":
    main()
