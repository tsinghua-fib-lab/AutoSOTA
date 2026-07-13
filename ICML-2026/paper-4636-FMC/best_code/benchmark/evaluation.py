"""Evaluation module for benchmark experiments (Phase 3).

This module handles selective evaluation of methods and variants,
computing metrics on the shared test set.

Usage:
    evaluator = BenchmarkEvaluator.from_config(cfg)

    # Evaluate a specific variant
    results = evaluator.evaluate_variant("fm_pt_fmpe_lr1e3", tasks=["pendulum"])

    # Evaluate all variants
    results = evaluator.evaluate_all(tasks=["pendulum", "gaussian"])

    # Evaluate baselines
    results = evaluator.evaluate_baselines(tasks=["pendulum"])
"""

import gc
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from core.checkpointing import CheckpointManager, ResultsPath
from utils.data_split import BenchmarkSplitter, DEFAULT_CALIBRATION_SIZES
from utils.transform import IdentityTransform


@dataclass
class EvaluationConfig:
    """Configuration for evaluation.

    Metric Types:
    - Joint metrics: Use many observations (num_joint_samples), 1 theta sample per observation.
      Compare (theta_sampled_i, y_i) to (theta_true_i, y_i).

    - Conditional metrics: Use few observations (num_conditional_obs),
      many samples per observation (num_posterior_samples).
      Compute metric per observation and average.
      Used for tasks with sample_reference_posterior implemented.
    """

    # Default metrics (joint) for tasks without reference posterior
    metrics: List[str] = field(default_factory=lambda: [
        "joint_c2st", "joint_mmd", "joint_wasserstein"
    ])

    # Conditional metrics for tasks with sample_reference_posterior
    conditional_metrics: List[str] = field(default_factory=lambda: [
        "cond_c2st", "cond_mmd"
    ])

    # Tasks that should use conditional metrics
    conditional_tasks: List[str] = field(default_factory=list)

    # Joint metric parameters
    # - Use many observations, 1 sample per observation
    num_joint_samples: int = 3000  # Number of (y, theta) pairs to compare

    # Conditional metric parameters (for HDM tasks)
    # - Use few observations, many samples per observation
    num_conditional_obs: int = 5  # Number of y observations for conditional metrics
    num_posterior_samples: int = 1000  # Samples per observation for conditional metrics

    batch_size: int = 100

    # Data configuration
    calibration_sizes: List[int] = field(default_factory=lambda: DEFAULT_CALIBRATION_SIZES.copy())
    seeds: List[int] = field(default_factory=lambda: [33, 43, 53])

    # Paths
    results_dir: Path = Path("results")
    experiment_name: str = "comparison_benchmark"
    data_dir: Optional[str] = None  # Base directory for file-based simulators

    # Skip existing
    skip_existing: bool = True

    # Tasks that should use embedding-based C2ST for joint metrics
    # Useful for high-dimensional observations (e.g., light_tunnel with 3x64x64 images)
    # where concatenating raw (theta, y) gives too much weight to y dimensions
    embedding_c2st_tasks: List[str] = field(default_factory=list)

    def use_embedding_c2st(self, task: str) -> bool:
        """Check if task should use embedding-based C2ST."""
        return task in self.embedding_c2st_tasks

    def get_metrics_for_task(self, task: str) -> List[str]:
        """Get the appropriate metrics for a task.

        All tasks get joint metrics. Tasks in conditional_tasks additionally
        get conditional metrics (cond_c2st, true_lpp, etc.).
        """
        if task in self.conditional_tasks:
            return self.metrics + self.conditional_metrics
        return self.metrics

    @classmethod
    def from_config(cls, cfg) -> "EvaluationConfig":
        """Create from Hydra config."""
        eval_cfg = cfg.get("evaluation", {})

        return cls(
            metrics=list(eval_cfg.get("metrics", ["joint_c2st", "joint_mmd", "joint_wasserstein"])),
            conditional_metrics=list(eval_cfg.get("conditional_metrics", ["cond_c2st", "cond_mmd"])),
            conditional_tasks=list(eval_cfg.get("conditional_tasks", [])),
            num_joint_samples=eval_cfg.get("num_joint_samples", 1000),
            num_conditional_obs=eval_cfg.get("num_conditional_obs", 5),
            num_posterior_samples=eval_cfg.get("num_posterior_samples", 1000),
            batch_size=eval_cfg.get("batch_size", 100),
            calibration_sizes=list(cfg.get("data", {}).get(
                "calibration_sizes", DEFAULT_CALIBRATION_SIZES
            )),
            # Seeds from experiment config (no fallback to config.yaml)
            seeds=list(cfg.get("seeds", [33, 43, 53])),
            results_dir=Path(cfg.get("results_dir", "results")),
            experiment_name=cfg.get("experiment", {}).get("name", "comparison_benchmark"),
            skip_existing=eval_cfg.get("skip_existing", True),
            embedding_c2st_tasks=list(eval_cfg.get("embedding_c2st_tasks", [])),
            data_dir=cfg.get("data_dir", None),
        )


class BenchmarkEvaluator:
    """Handles Phase 3: Evaluation of methods and variants."""

    def __init__(
        self,
        config: EvaluationConfig,
    ):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Setup paths and checkpoint manager
        self.paths = ResultsPath(config.results_dir, config.experiment_name)
        self.checkpoint_manager = CheckpointManager(self.paths)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache for loaded data, models, and simulators
        self._test_data_cache = {}
        self._model_cache = {}
        self._simulator_cache = {}

    def _get_simulator(self, task: str):
        """Get simulator for a task (with caching)."""
        if task in self._simulator_cache:
            return self._simulator_cache[task]

        try:
            from omegaconf import OmegaConf
            from pathlib import Path
            from simulator import get_simulator

            config_path = Path("configs/task") / f"{task}.yaml"
            if config_path.exists():
                task_cfg = OmegaConf.load(config_path)
                task_cfg = OmegaConf.to_container(task_cfg, resolve=False)
                simulator_config = task_cfg.get("simulator", {})
                sim_kwargs = dict(simulator_config.get("params", {}))
                if self.config.data_dir is not None:
                    # Only add data_dir for file-based simulators; try without first
                    sim_cls_name = simulator_config.get("name", task)
                    try:
                        simulator = get_simulator(sim_cls_name, **sim_kwargs)
                        self._simulator_cache[task] = simulator
                        return simulator
                    except TypeError:
                        sim_kwargs["data_dir"] = self.config.data_dir
                simulator = get_simulator(
                    simulator_config.get("name", task),
                    **sim_kwargs,
                )
                self._simulator_cache[task] = simulator
                return simulator
        except Exception as e:
            print(f"    Warning: Could not load simulator for {task}: {e}")

        return None

    @classmethod
    def from_config(cls, cfg) -> "BenchmarkEvaluator":
        """Create evaluator from Hydra config."""
        return cls(EvaluationConfig.from_config(cfg))

    def evaluate_variant(
        self,
        variant_name: str,
        tasks: List[str],
        seeds: Optional[List[int]] = None,
        ncals: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a variant on specified tasks.

        Args:
            variant_name: Name of the variant to evaluate
            tasks: List of tasks to evaluate on
            seeds: Optional list of seeds (defaults to config)
            ncals: Optional list of ncal values (defaults to config)
            metrics: Optional list of metrics (defaults to config)
            skip_existing: Skip if metrics already exist

        Returns:
            Evaluation results
        """
        seeds = seeds or self.config.seeds
        ncals = ncals or self.config.calibration_sizes

        results = {}

        for task in tasks:
            # Get task-specific metrics (conditional vs joint)
            task_metrics = metrics or self.config.get_metrics_for_task(task)

            print(f"\n{'='*60}")
            print(f"Evaluating variant: {variant_name} on {task}")
            print(f"Metrics: {task_metrics}")
            print(f"{'='*60}")

            task_results = {}

            for seed in seeds:
                task_results[seed] = {}

                for ncal in ncals:
                    # Load variant config to get method name
                    variant_cfg = self.checkpoint_manager.load_variant_config(variant_name)
                    if variant_cfg is None:
                        print(f"  Variant config not found for {variant_name}, skipping...")
                        continue

                    method = variant_cfg.get("method", "fm_post_transform")

                    # Check if metrics already exist
                    if skip_existing and self.checkpoint_manager.variant_metrics_exist(
                        variant_name, task, seed, method, ncal
                    ):
                        print(f"  Metrics exist for {method} (seed={seed}, ncal={ncal}), loading...")
                        existing = self.checkpoint_manager.load_variant_metrics(
                            variant_name, task, seed, method, ncal
                        )
                        task_results[seed][ncal] = existing
                        continue

                    # Evaluate
                    result = self._evaluate_variant_model(
                        variant_name, task, seed, ncal, method, task_metrics
                    )
                    task_results[seed][ncal] = result

            results[task] = task_results

        return results

    def evaluate_baselines(
        self,
        tasks: List[str],
        baselines: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        ncals: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate shared baselines.

        Args:
            tasks: List of tasks to evaluate on
            baselines: Optional list of baselines (defaults to discovering from shared/)
            seeds: Optional list of seeds (defaults to config)
            ncals: Optional list of ncal values (defaults to config)
            metrics: Optional list of metrics (defaults to config)
            skip_existing: Skip if metrics already exist

        Returns:
            Evaluation results
        """
        seeds = seeds or self.config.seeds
        ncals = ncals or self.config.calibration_sizes

        results = {}

        for task in tasks:
            # Get task-specific metrics (conditional vs joint)
            task_metrics = metrics or self.config.get_metrics_for_task(task)

            print(f"\n{'='*60}")
            print(f"Evaluating baselines on {task}")
            print(f"Metrics: {task_metrics}")
            print(f"{'='*60}")

            task_results = {}

            # Discover baselines if not specified
            if baselines is None:
                if seeds:
                    baselines = self.checkpoint_manager.list_shared_baselines(task, seeds[0])
                else:
                    baselines = []

            for seed in seeds:
                task_results[seed] = {}

                for baseline in baselines:
                    for ncal in ncals:
                        # Check if model exists
                        if not self.checkpoint_manager.shared_baseline_exists(
                            task, seed, baseline, ncal
                        ):
                            print(f"  {baseline} (seed={seed}, ncal={ncal}) not found, skipping...")
                            continue

                        # Check if metrics already exist
                        existing_metrics = self.checkpoint_manager.load_shared_baseline_metrics(
                            task, seed, baseline, ncal
                        )
                        if skip_existing and existing_metrics is not None:
                            print(f"  Metrics exist for {baseline} (seed={seed}, ncal={ncal}), loading...")
                            key = f"{baseline}_ncal{ncal}"
                            task_results[seed][key] = existing_metrics
                            continue

                        # Evaluate
                        result = self._evaluate_baseline_model(
                            task, seed, baseline, ncal, task_metrics
                        )
                        key = f"{baseline}_ncal{ncal}"
                        task_results[seed][key] = result

            results[task] = task_results

        return results

    def evaluate_simulation_models(
        self,
        tasks: List[str],
        models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate shared simulation models (NPE, FMPE).

        Args:
            tasks: List of tasks to evaluate on
            models: Optional list of models (defaults to ["npe", "fmpe"])
            metrics: Optional list of metrics (defaults to config)

        Returns:
            Evaluation results
        """
        models = models or ["npe", "fmpe"]

        results = {}

        for task in tasks:
            # Get task-specific metrics (conditional vs joint)
            task_metrics = metrics or self.config.get_metrics_for_task(task)

            print(f"\n{'='*60}")
            print(f"Evaluating simulation models on {task}")
            print(f"Metrics: {task_metrics}")
            print(f"{'='*60}")

            task_results = {}

            for model_type in models:
                # Check if model exists
                if not self.checkpoint_manager.shared_simulation_model_exists(task, model_type):
                    print(f"  {model_type.upper()} not found for {task}, skipping...")
                    continue

                result = self._evaluate_simulation_model(task, model_type, task_metrics)
                task_results[model_type] = result

            results[task] = task_results

        return results

    def evaluate_all(
        self,
        tasks: List[str],
        include_baselines: bool = True,
        include_simulation_models: bool = True,
        variants: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate all methods and variants.

        Args:
            tasks: List of tasks to evaluate
            include_baselines: Include shared baselines
            include_simulation_models: Include simulation models (NPE, FMPE)
            variants: List of variants to evaluate (defaults to all)
            skip_existing: Skip if metrics already exist

        Returns:
            Combined evaluation results
        """
        results = {
            "simulation_models": {},
            "baselines": {},
            "variants": {},
        }

        if include_simulation_models:
            results["simulation_models"] = self.evaluate_simulation_models(tasks)

        if include_baselines:
            results["baselines"] = self.evaluate_baselines(
                tasks, skip_existing=skip_existing
            )

        # Discover or use specified variants
        if variants is None:
            variants = self.checkpoint_manager.list_variants()

        for variant_name in variants:
            results["variants"][variant_name] = self.evaluate_variant(
                variant_name, tasks, skip_existing=skip_existing
            )

        return results

    def _evaluate_variant_model(
        self,
        variant_name: str,
        task: str,
        seed: int,
        ncal: int,
        method: str,
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a variant model."""
        print(f"  Evaluating {method} (seed={seed}, ncal={ncal})...")

        # Load model
        model = self.checkpoint_manager.load_variant_calibration_model(
            variant_name, task, seed, method, ncal, self.device
        )
        if model is None:
            return {"status": "not_found", "metrics": {}}

        # Load test data
        test_data = self._get_test_data(task)
        if test_data is None:
            return {"status": "no_test_data", "metrics": {}}

        # Compute metrics
        computed_metrics = self._compute_metrics(model, test_data, metrics, task)

        # Free GPU memory from this model before evaluating the next one
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save results
        result = {
            "status": "evaluated",
            "metrics": computed_metrics,
            "timestamp": datetime.now().isoformat(),
            "num_test_observations": len(test_data["y"]),
        }

        self.checkpoint_manager.save_variant_metrics(
            result, variant_name, task, seed, method, ncal
        )

        return result

    def _evaluate_baseline_model(
        self,
        task: str,
        seed: int,
        baseline: str,
        ncal: int,
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a baseline model."""
        from posteriors import RopePosterior

        print(f"  Evaluating {baseline} (seed={seed}, ncal={ncal})...")

        # Load model
        model = self.checkpoint_manager.load_shared_baseline(
            task, seed, baseline, ncal, self.device
        )
        if model is None:
            return {"status": "not_found", "metrics": {}}

        # Load test data
        test_data = self._get_test_data(task)
        if test_data is None:
            return {"status": "no_test_data", "metrics": {}}

        # RopePosterior needs simulation data for OT-weighted sampling
        # Subsample to test set size to keep Sinkhorn tractable
        if isinstance(model, RopePosterior):
            sim_data = self.checkpoint_manager.load_shared_simulations(task)
            if sim_data is None:
                print(f"  Warning: No simulation data found for {task}, ROPE evaluation will fail")
                return {"status": "no_sim_data", "metrics": {}}
            n_test = len(test_data["y"])
            n_sim = sim_data["x"].shape[0]
            if n_sim > n_test:
                idx = torch.randperm(n_sim)[:n_test]
                model.set_sim_data(sim_data["x"][idx], sim_data["theta"][idx])
            else:
                model.set_sim_data(sim_data["x"], sim_data["theta"])

        # Compute metrics
        computed_metrics = self._compute_metrics(model, test_data, metrics, task)

        # Free GPU memory from this model before evaluating the next one
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save results (update the baseline's metrics.json)
        result = {
            "status": "evaluated",
            "metrics": computed_metrics,
            "timestamp": datetime.now().isoformat(),
            "num_test_observations": len(test_data["y"]),
        }

        baseline_path = self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
        with open(baseline_path / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _evaluate_simulation_model(
        self,
        task: str,
        model_type: str,
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a simulation model (NPE/FMPE)."""
        print(f"  Evaluating {model_type.upper()}...")

        # Load model
        model = self.checkpoint_manager.load_shared_simulation_model(
            task, model_type, self.device
        )
        if model is None:
            return {"status": "not_found", "metrics": {}}

        # Load test data
        test_data = self._get_test_data(task)
        if test_data is None:
            return {"status": "no_test_data", "metrics": {}}

        # For simulation models, we evaluate p(theta|x) not p(theta|y)
        # So we use x as the conditioning variable
        sim_test_data = {
            "theta": test_data["theta"],
            "y": test_data["x"],  # Simulation models condition on x
        }

        # Compute metrics with misspecified=True since we're evaluating on x data
        computed_metrics = self._compute_metrics(model, sim_test_data, metrics, task, misspecified=True)

        # Free GPU memory from this model before evaluating the next one
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save results
        result = {
            "status": "evaluated",
            "metrics": computed_metrics,
            "timestamp": datetime.now().isoformat(),
            "num_test_observations": len(test_data["x"]),
        }

        model_path = self.paths.shared_simulation_model_path(task, model_type)
        with open(model_path / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _get_test_data(self, task: str) -> Optional[Dict[str, Tensor]]:
        """Get test data for a task (with caching)."""
        if task in self._test_data_cache:
            return self._test_data_cache[task]

        test_data = self.checkpoint_manager.load_shared_test_data(task)
        if test_data is not None:
            self._test_data_cache[task] = test_data

        return test_data

    def _compute_metrics(
        self,
        model: Any,
        test_data: Dict[str, Tensor],
        metrics: List[str],
        task: str,
        misspecified: bool = False,
    ) -> Dict[str, float]:
        """Compute specified metrics for a model.

        Metric computation depends on metric type:
        - Joint metrics (joint_*): Use many observations, 1 sample per observation.
          Compare sets (theta_sampled_i, y_i) to (theta_true_i, y_i).
        - Conditional metrics (cond_*): Use few observations, many samples per observation.
          Compute metric per observation and average.

        Args:
            model: Posterior model (must have .sample() method)
            test_data: Dict with 'theta' (ground truth) and 'y' (observations)
            metrics: List of metric names to compute
            task: Task name (for accessing simulator in conditional metrics)
            misspecified: If True, use p(θ|x) for reference (simulation models on x data).
                         If False, use p(θ|y) for reference (baselines/variants on y data).

        Returns:
            Dict mapping metric names to values
        """
        results = {}

        theta_true = test_data["theta"]
        y_obs = test_data["y"]

        # Get the simulator's parameter transform for mapping to original space.
        # Models are trained on transformed theta (e.g. logit-mapped to R^n),
        # but metrics should be computed in the original parameter space.
        simulator = self._get_simulator(task)
        transform = None
        if simulator is not None:
            t = simulator.transform
            if not isinstance(t, IdentityTransform):
                transform = t

        # Separate joint and conditional metrics
        # lpp is a per-pair metric (needs one (theta, y) pair) so it routes through joint
        # true_lpp needs many samples per obs so it stays conditional
        joint_metrics = [m for m in metrics if m.startswith("joint_") or m == "lpp"]
        conditional_metrics = [m for m in metrics if m.startswith("cond_") or m == "true_lpp"]
        other_metrics = [m for m in metrics if m not in joint_metrics and m not in conditional_metrics]

        # Compute joint metrics: many observations, 1 sample per observation
        if joint_metrics:
            joint_results = self._compute_joint_metrics(model, theta_true, y_obs, joint_metrics, task, transform=transform)
            results.update(joint_results)

        # Compute conditional metrics: few observations, many samples per observation
        if conditional_metrics:
            cond_results = self._compute_conditional_metrics(model, theta_true, y_obs, conditional_metrics, task, misspecified, original_model=model, transform=transform)
            results.update(cond_results)

        # Compute other metrics (e.g., coverage) - use conditional-style sampling
        if other_metrics:
            other_results = self._compute_conditional_metrics(model, theta_true, y_obs, other_metrics, task, misspecified, original_model=model, transform=transform)
            results.update(other_results)

        return results

    def _compute_lpp(
        self,
        model: Any,
        theta_true: Tensor,
        y_obs: Tensor,
    ) -> float:
        """Compute mean log posterior probability.

        Calls model.log_prob(theta, y) in batches. Returns NaN if the
        model does not implement log_prob.

        Args:
            model: Posterior model (must have .log_prob() method)
            theta_true: Ground truth theta (n_total, theta_dim)
            y_obs: Observations (n_total, obs_dim)

        Returns:
            Mean log posterior probability (float)
        """
        n_obs = min(len(y_obs), self.config.num_joint_samples)
        theta_true = theta_true[:n_obs]
        y_obs = y_obs[:n_obs]

        print(f"    Computing LPP with {n_obs} observations...")

        try:
            log_probs = []
            batch_size = self.config.batch_size
            for start in range(0, n_obs, batch_size):
                end = min(start + batch_size, n_obs)
                theta_batch = theta_true[start:end].to(self.device)
                y_batch = y_obs[start:end].to(self.device)

                lp = model.log_prob(theta_batch, y_batch)
                log_probs.append(lp.detach().cpu())

            all_log_probs = torch.cat(log_probs, dim=0)
            # Filter out NaN/Inf before computing mean
            valid = torch.isfinite(all_log_probs)
            if valid.sum() == 0:
                return float('nan')
            return float(all_log_probs[valid].mean().item())
        except NotImplementedError:
            print("      log_prob not implemented, skipping LPP")
            return float('nan')
        except Exception as e:
            print(f"      Error computing LPP: {e}")
            return float('nan')

    def _compute_joint_metrics(
        self,
        model: Any,
        theta_true: Tensor,
        y_obs: Tensor,
        metrics: List[str],
        task: str,
        transform=None,
    ) -> Dict[str, float]:
        """Compute joint metrics.

        Joint metrics compare the joint distribution:
        - Sample 1 theta per observation from p(theta|y)
        - Compare (theta_sampled, y) to (theta_true, y)

        Args:
            model: Posterior model
            theta_true: Ground truth theta (n_total, theta_dim)
            y_obs: Observations (n_total, obs_dim)
            metrics: List of joint metric names
            task: Task name (used for embedding-based C2ST)

        Returns:
            Dict mapping metric names to values
        """
        results = {}

        # Use num_joint_samples observations
        n_obs = min(len(y_obs), self.config.num_joint_samples)
        theta_true = theta_true[:n_obs]
        y_obs = y_obs[:n_obs]

        print(f"    Computing joint metrics with {n_obs} observations, 1 sample per obs...")

        # Sample 1 theta per observation using batched sampling
        # This is much more efficient than looping, especially for methods like NPE-PFN
        batch_size = self.config.batch_size
        try:
            with torch.no_grad():
                theta_samples_list = []
                for start in range(0, n_obs, batch_size):
                    end = min(start + batch_size, n_obs)
                    y_batch = y_obs[start:end].to(self.device)

                    # Sample 1 theta per observation in batch
                    # Returns shape (nsamples=1, batch_size, theta_dim)
                    samples = model.sample(y_batch, 1, self.device).cpu()

                    # Squeeze nsamples dimension: (1, batch_size, theta_dim) -> (batch_size, theta_dim)
                    samples = samples.squeeze(0)

                    # Handle edge case where theta_dim=1 and squeeze removes it
                    if samples.dim() == 1:
                        samples = samples.unsqueeze(-1)

                    theta_samples_list.append(samples)

                theta_samples = torch.cat(theta_samples_list, dim=0)  # (n_obs, theta_dim)

                # Map to original parameter space
                if transform is not None:
                    theta_samples = transform.inv(theta_samples)
        except Exception as e:
            print(f"    Error sampling from model: {e}")
            import traceback
            traceback.print_exc()
            return {m: float('nan') for m in metrics}

        # Map theta_true to original space for sample-based metrics
        theta_true_orig = transform.inv(theta_true) if transform is not None else theta_true

        # Separate special joint metrics from standard ones
        standard_metrics = [m for m in metrics if m not in ("joint_coverage", "lpp")]
        has_coverage = "joint_coverage" in metrics
        has_lpp = "lpp" in metrics

        # Compute each standard joint metric (uses 1 sample per obs)
        for metric_name in standard_metrics:
            try:
                metric_value = self._compute_single_joint_metric(
                    metric_name, theta_true_orig, theta_samples, y_obs, task
                )
                results[metric_name] = metric_value
                print(f"    {metric_name}: {metric_value:.4f}")
            except Exception as e:
                print(f"    Error computing {metric_name}: {e}")
                results[f"{metric_name}_error"] = str(e)

        # Compute joint coverage: many samples per observation across all test obs
        if has_coverage:
            try:
                n_samples = 100
                print(f"    Computing joint_coverage with {n_obs} observations, {n_samples} samples per obs...")
                coverage_samples_list = []
                with torch.no_grad():
                    for start in range(0, n_obs, batch_size):
                        end = min(start + batch_size, n_obs)
                        y_batch = y_obs[start:end].to(self.device)
                        # (n_samples, batch_size, theta_dim)
                        samples = model.sample(y_batch, n_samples, self.device).cpu()
                        # -> (batch_size, n_samples, theta_dim)
                        samples = samples.permute(1, 0, 2)
                        coverage_samples_list.append(samples)
                coverage_samples = torch.cat(coverage_samples_list, dim=0)  # (n_obs, n_samples, theta_dim)
                # Map to original parameter space
                if transform is not None:
                    coverage_samples = transform.inv(coverage_samples)
                coverage_value = self._coverage(theta_true_orig, coverage_samples)
                results["joint_coverage"] = coverage_value
                print(f"    joint_coverage: {coverage_value:.4f}")
            except Exception as e:
                print(f"    Error computing joint_coverage: {e}")
                results["joint_coverage_error"] = str(e)

        # Compute lpp: model log_prob evaluated on all test (theta, y) pairs
        # NOTE: no torch.no_grad() here — log_prob may need gradients for
        # divergence computation (e.g. reverse ODE in flow-based models)
        if has_lpp:
            if hasattr(model, 'log_prob'):
                try:
                    print(f"    Computing lpp with {n_obs} observations...")
                    lpp_values = []
                    for start in range(0, n_obs, batch_size):
                        end = min(start + batch_size, n_obs)
                        lp = model.log_prob(
                            theta_true[start:end].to(self.device),
                            y_obs[start:end].to(self.device),
                        )
                        lpp_values.append(lp.detach().cpu())
                    lpp_all = torch.cat(lpp_values, dim=0)
                    # Jacobian correction: log p(θ|y) = log p(θ'|y) + log |dθ'/dθ|
                    # where θ = original params, θ' = transformed (what model was trained on)
                    if transform is not None:
                        log_jac = transform.log_abs_det_jacobian(theta_true_orig, None)
                        lpp_all = lpp_all + log_jac
                    lpp_mean = float(lpp_all.mean().item())
                    results["lpp"] = lpp_mean
                    print(f"    lpp: {lpp_mean:.4f}")
                except Exception as e:
                    print(f"    Error computing lpp: {e}")
                    results["lpp"] = float('nan')
            else:
                print(f"    Model does not support log_prob, skipping lpp")
                results["lpp"] = float('nan')

        return results

    def _compute_conditional_metrics(
        self,
        model: Any,
        theta_true: Tensor,
        y_obs: Tensor,
        metrics: List[str],
        task: str,
        misspecified: bool = False,
        original_model: Any = None,
        transform=None,
    ) -> Dict[str, float]:
        """Compute conditional metrics.

        Conditional metrics evaluate p(theta|y) for individual observations:
        - Use few observations (num_conditional_obs)
        - Sample many thetas per observation (num_posterior_samples)
        - Compute metric per observation and average

        For cond_c2st: Compare model samples to reference posterior samples
        (obtained via simulator.sample_reference_posterior).

        Args:
            model: Posterior model
            theta_true: Ground truth theta (n_total, theta_dim)
            y_obs: Observations (n_total, obs_dim)
            metrics: List of conditional metric names
            task: Task name for accessing simulator
            misspecified: If True, use p(θ|x) for reference (simulation models).
                         If False, use p(θ|y) for reference (baselines/variants).

        Returns:
            Dict mapping metric names to values
        """
        results = {}

        # Use num_conditional_obs observations
        n_obs = min(len(y_obs), self.config.num_conditional_obs)
        theta_true = theta_true[:n_obs]
        y_obs = y_obs[:n_obs]

        print(f"    Computing conditional metrics with {n_obs} observations, {self.config.num_posterior_samples} samples per obs...")

        # Sample many thetas per observation from the model
        try:
            with torch.no_grad():
                theta_samples = []
                for i in range(n_obs):
                    # Most posteriors use: sample(y, nsamples, device)
                    # Returns shape (nsamples, batch, theta_dim) for SBI posteriors
                    samples = model.sample(
                        y_obs[i:i+1].to(self.device),
                        self.config.num_posterior_samples,
                        self.device,
                    ).cpu()
                    # Handle various output shapes:
                    # SBI: (n_samples, batch, theta_dim) -> (n_samples, theta_dim) via [:, 0, :]
                    # FlowMatching: (batch, n_samples, theta_dim) -> (n_samples, theta_dim) via [0, :, :]
                    # (n_samples, theta_dim) -> keep as is
                    if samples.ndim == 3:
                        # Determine which is the batch dimension (should be 1 for single obs)
                        if samples.shape[0] == self.config.num_posterior_samples:
                            # Shape: (n_samples, batch=1, theta_dim) - SBI format
                            samples = samples[:, 0, :]
                        else:
                            # Shape: (batch=1, n_samples, theta_dim) - FlowMatching format
                            samples = samples[0]
                    theta_samples.append(samples)
                # Shape: (n_obs, n_samples, theta_dim)
                theta_samples = torch.stack(theta_samples, dim=0)

                # Map to original parameter space
                if transform is not None:
                    theta_samples = transform.inv(theta_samples)
        except Exception as e:
            print(f"    Error sampling from model: {e}")
            return {m: float('nan') for m in metrics}

        # Map theta_true to original space for metrics
        theta_true_orig = transform.inv(theta_true) if transform is not None else theta_true

        # Get reference posterior samples if needed (for cond_c2st)
        reference_samples = None
        simulator = None
        if "cond_c2st" in metrics:
            simulator = self._get_simulator(task)
            if simulator is not None and hasattr(simulator, 'sample_reference_posterior'):
                try:
                    # Get reference samples for each observation
                    # sample_reference_posterior returns (n_samples, n_obs, theta_dim)
                    # or similar depending on implementation
                    ref_samples_list = []
                    for i in range(n_obs):
                        ref_samples = simulator.sample_reference_posterior(
                            num_samples=self.config.num_posterior_samples,
                            observations=y_obs[i:i+1],
                            misspecified=misspecified,  # True for simulation models (x data), False for baselines (y data)
                        )
                        # Handle shape: should be (n_samples, theta_dim) or (n_samples, 1, theta_dim)
                        if ref_samples.ndim == 3:
                            ref_samples = ref_samples[:, 0, :]  # (n_samples, theta_dim)
                        ref_samples_list.append(ref_samples)
                    # Shape: (n_obs, n_samples, theta_dim)
                    reference_samples = torch.stack(ref_samples_list, dim=0)
                    print(f"    Reference samples shape: {reference_samples.shape}")
                    print(f"    Model samples shape: {theta_samples.shape}")

                except Exception as e:
                    print(f"    Warning: Could not get reference posterior samples: {e}")
                    import traceback
                    traceback.print_exc()
                    reference_samples = None
            else:
                print(f"    Warning: Simulator for {task} does not have sample_reference_posterior")

        # Compute each conditional metric
        for metric_name in metrics:
            try:
                if metric_name == "cond_c2st" and reference_samples is not None:
                    # Use reference samples for C2ST comparison
                    metric_value = self._conditional_c2st(theta_samples, reference_samples)
                elif metric_name == "true_lpp":
                    # True posterior log-prob on method-generated samples
                    simulator = self._get_simulator(task)
                    if simulator is not None and hasattr(simulator, 'post_dist'):
                        try:
                            true_lpp_values = []
                            for i in range(n_obs):
                                true_post = simulator.post_dist(y_obs[i:i+1], misspecified=False)
                                method_samples_i = theta_samples[i]  # (n_samples, theta_dim)
                                log_probs = true_post.log_prob(method_samples_i)
                                true_lpp_values.append(log_probs.mean().item())
                            metric_value = float(np.mean(true_lpp_values))
                        except Exception as e:
                            print(f"    true_lpp failed: {e}")
                            metric_value = float('nan')
                    else:
                        metric_value = float('nan')
                else:
                    metric_value = self._compute_single_conditional_metric(
                        metric_name, theta_true_orig, theta_samples
                    )
                results[metric_name] = metric_value
                print(f"    {metric_name}: {metric_value:.4f}")
            except Exception as e:
                print(f"    Error computing {metric_name}: {e}")
                results[metric_name] = float('nan')

        return results

    def _compute_single_joint_metric(
        self,
        metric_name: str,
        theta_true: Tensor,
        theta_samples: Tensor,
        y_obs: Tensor,
        task: str,
    ) -> float:
        """Compute a single joint metric.

        Args:
            metric_name: Name of metric to compute
            theta_true: Ground truth theta (n_obs, theta_dim)
            theta_samples: Posterior samples, 1 per observation (n_obs, theta_dim)
            y_obs: Observations (n_obs, obs_dim)
            task: Task name (used for embedding-based C2ST)

        Returns:
            Metric value
        """
        if metric_name == "joint_c2st":
            if self.config.use_embedding_c2st(task):
                return self._joint_c2st_with_embedding(theta_true, theta_samples, y_obs, task)
            return self._joint_c2st(theta_true, theta_samples, y_obs)
        elif metric_name == "joint_mmd":
            return self._joint_mmd(theta_true, theta_samples, y_obs)
        elif metric_name == "joint_wasserstein":
            return self._joint_wasserstein(theta_true, theta_samples, y_obs)
        else:
            raise ValueError(f"Unknown joint metric: {metric_name}")

    def _compute_single_conditional_metric(
        self,
        metric_name: str,
        theta_true: Tensor,
        theta_samples: Tensor,
    ) -> float:
        """Compute a single conditional metric.

        Args:
            metric_name: Name of metric to compute
            theta_true: Ground truth theta (n_obs, theta_dim)
            theta_samples: Posterior samples (n_obs, n_samples, theta_dim)

        Returns:
            Metric value (averaged over observations)
        """
        if metric_name == "coverage":
            return self._coverage(theta_true, theta_samples)
        elif metric_name == "cond_c2st":
            return self._conditional_c2st(theta_true, theta_samples)
        elif metric_name == "cond_mmd":
            return self._conditional_mmd(theta_true, theta_samples)
        elif metric_name == "cond_wasserstein":
            return self._conditional_wasserstein(theta_true, theta_samples)
        else:
            raise ValueError(f"Unknown conditional metric: {metric_name}")

    def _joint_c2st(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
        y_obs: Tensor,
    ) -> float:
        """Compute joint C2ST metric.

        Compares joint distribution (theta, y) from posterior to ground truth.
        - theta_true, y_obs: ground truth pairs
        - theta_samples, y_obs: posterior samples (1 per observation)

        Perfect calibration gives C2ST = 0.5.
        """
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("scikit-learn required for C2ST metric")

        # Concatenate theta and y for joint comparison
        # True: (theta_true, y_obs)
        # Posterior: (theta_samples, y_obs)
        joint_true = torch.cat([theta_true, y_obs.reshape(len(theta_true), -1)], dim=1)
        joint_posterior = torch.cat([theta_samples, y_obs.reshape(len(theta_samples), -1)], dim=1)

        # Create labels: 0 = true, 1 = posterior
        n = len(theta_true)
        X = torch.cat([joint_true, joint_posterior], dim=0).numpy()
        y = torch.cat([torch.zeros(n), torch.ones(n)]).numpy()

        # Train classifier and compute accuracy
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        return float(scores.mean())

    def _joint_c2st_with_embedding(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
        y_obs: Tensor,
        task: str,
    ) -> float:
        """Compute joint C2ST with learned embedding for high-dimensional observations.

        For tasks with high-dimensional observations (e.g., light_tunnel with 3x64x64 images),
        concatenating raw (theta, y) gives too much weight to y dimensions. Instead, we:
        1. Load the NPE encoder pretrained on (theta, x) pairs
        2. Fine-tune it during C2ST training to adapt to y distribution
        3. Use (theta, embed(y)) for classification

        Args:
            theta_true: Ground truth theta (n_obs, theta_dim)
            theta_samples: Posterior samples, 1 per observation (n_obs, theta_dim)
            y_obs: Observations (n_obs, *obs_shape) - can be high-dimensional
            task: Task name (to load the corresponding NPE encoder)

        Returns:
            C2ST accuracy (0.5 = indistinguishable, 1.0 = perfectly separable)
        """
        # Load NPE model to extract encoder
        npe_model = self.checkpoint_manager.load_shared_simulation_model(task, "npe", self.device)
        if npe_model is None:
            print(f"    Warning: NPE model not found for {task}, falling back to standard C2ST")
            return self._joint_c2st(theta_true, theta_samples, y_obs)

        # Extract encoder and rescaler (deep copy to avoid modifying original)
        encoder = copy.deepcopy(npe_model.embedding_net).to(self.device)
        cond_rescaler = npe_model.cond_rescaler

        # Get dimensions
        theta_dim = theta_true.shape[-1]
        embed_dim = npe_model.obs_dim

        # Build C2ST classifier network
        class C2STClassifier(nn.Module):
            def __init__(self, encoder, theta_dim, embed_dim, cond_rescaler):
                super().__init__()
                self.encoder = encoder
                self.cond_rescaler = cond_rescaler
                self.classifier = nn.Sequential(
                    nn.Linear(theta_dim + embed_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )

            def forward(self, theta, y):
                # Rescale y before embedding (as NPE was trained)
                y_rescaled = self.cond_rescaler.transform(y)
                y_embed = self.encoder(y_rescaled)
                combined = torch.cat([theta, y_embed], dim=-1)
                return self.classifier(combined)

        classifier = C2STClassifier(encoder, theta_dim, embed_dim, cond_rescaler).to(self.device)

        # Prepare data: class 0 = true (theta_true, y), class 1 = posterior (theta_samples, y)
        n = len(theta_true)
        theta_all = torch.cat([theta_true, theta_samples], dim=0).to(self.device)
        y_all = torch.cat([y_obs, y_obs], dim=0).to(self.device)  # Same y for both
        labels = torch.cat([torch.zeros(n), torch.ones(n)]).to(self.device)

        # Shuffle data
        perm = torch.randperm(2 * n)
        theta_all = theta_all[perm]
        y_all = y_all[perm]
        labels = labels[perm]

        # Split into train/val (80/20)
        split_idx = int(0.8 * 2 * n)
        train_theta, val_theta = theta_all[:split_idx], theta_all[split_idx:]
        train_y, val_y = y_all[:split_idx], y_all[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # Training setup
        optimizer = torch.optim.Adam([
            {"params": classifier.encoder.parameters(), "lr": 1e-5},  # Lower LR for encoder
            {"params": classifier.classifier.parameters(), "lr": 1e-3},  # Higher LR for classifier
        ])
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        batch_size = min(128, split_idx // 4)
        n_epochs = 50
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0

        train_dataset = TensorDataset(train_theta, train_y, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        epoch = 0
        for epoch in range(n_epochs):
            # Train
            classifier.train()
            for batch_theta, batch_y, batch_labels in train_loader:
                optimizer.zero_grad()
                logits = classifier(batch_theta, batch_y).squeeze(-1)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            # Validate
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_theta, val_y).squeeze(-1)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_preds == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        print(f"    C2ST with embedding: val_acc={best_val_acc:.4f} (epochs={epoch+1})")
        return best_val_acc

    def _joint_mmd(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
        y_obs: Tensor,
    ) -> float:
        """Compute joint MMD metric.

        Maximum Mean Discrepancy between (theta_posterior, y) and (theta_true, y).
        Lower is better.
        """
        # Concatenate theta and y for joint comparison
        joint_true = torch.cat([theta_true, y_obs.reshape(len(theta_true), -1)], dim=1)
        joint_posterior = torch.cat([theta_samples, y_obs.reshape(len(theta_samples), -1)], dim=1)

        # Compute MMD with Gaussian kernel
        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y) ** 2
            return torch.exp(-dist / (2 * sigma ** 2))

        xx = gaussian_kernel(joint_true, joint_true)
        yy = gaussian_kernel(joint_posterior, joint_posterior)
        xy = gaussian_kernel(joint_true, joint_posterior)

        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return float(mmd)

    def _joint_wasserstein(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
        y_obs: Tensor,
    ) -> float:
        """Compute joint Wasserstein distance.

        Uses sliced Wasserstein approximation for computational efficiency.
        Computes on joint (theta, y) space.
        """
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            raise ImportError("scipy required for Wasserstein metric")

        # Concatenate theta and y for joint comparison
        joint_true = torch.cat([theta_true, y_obs.reshape(len(theta_true), -1)], dim=1).numpy()
        joint_posterior = torch.cat([theta_samples, y_obs.reshape(len(theta_samples), -1)], dim=1).numpy()

        # Compute 1D Wasserstein for each dimension and average
        n_dims = joint_true.shape[1]
        distances = []
        for d in range(n_dims):
            dist = wasserstein_distance(joint_true[:, d], joint_posterior[:, d])
            distances.append(dist)

        return float(sum(distances) / len(distances))

    def _conditional_c2st(
        self,
        model_samples: Tensor,
        reference_samples: Tensor,
    ) -> float:
        """Compute conditional C2ST metric (averaged over observations).

        Compares model posterior samples to reference posterior samples
        (from simulator.sample_reference_posterior) for each observation.

        C2ST = 0.5 means distributions are indistinguishable (ideal).
        C2ST = 1.0 means distributions are perfectly separable (bad).

        Args:
            model_samples: Samples from model posterior (n_obs, n_samples, theta_dim)
            reference_samples: Samples from true posterior (n_obs, n_samples, theta_dim)
                              Already transformed to match model output space.

        Returns:
            Average C2ST across observations
        """
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("scikit-learn required for C2ST metric")

        n_obs = model_samples.shape[0]
        c2st_values = []

        # Check for dimension mismatch
        if model_samples.shape[-1] != reference_samples.shape[-1]:
            print(f"      Warning: theta_dim mismatch - model: {model_samples.shape[-1]}, ref: {reference_samples.shape[-1]}")
            return float('nan')

        for i in range(n_obs):
            model_s = model_samples[i]  # (n_samples, theta_dim)
            ref_s = reference_samples[i]  # (n_samples, theta_dim)

            # Use equal number of samples from each distribution
            n = min(len(model_s), len(ref_s))
            model_s = model_s[:n]
            ref_s = ref_s[:n]

            # Check for NaN/Inf values
            if torch.isnan(model_s).any() or torch.isinf(model_s).any():
                print(f"      Warning: NaN/Inf in model samples for obs {i}")
                continue
            if torch.isnan(ref_s).any() or torch.isinf(ref_s).any():
                print(f"      Warning: NaN/Inf in reference samples for obs {i}")
                continue

            # Create dataset: label 0 for model, label 1 for reference
            X = torch.cat([model_s, ref_s], dim=0).numpy()
            y = torch.cat([torch.zeros(n), torch.ones(n)]).numpy()

            # Train classifier to distinguish the two distributions
            clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=42)
            try:
                scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
                c2st_values.append(scores.mean())
            except Exception as e:
                print(f"      Warning: C2ST classifier failed for obs {i}: {e}")
                continue

        if not c2st_values:
            print(f"      Warning: No valid C2ST values computed")

        return float(sum(c2st_values) / len(c2st_values)) if c2st_values else float('nan')

    def _conditional_mmd(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
    ) -> float:
        """Compute conditional MMD metric (averaged over observations).

        For each observation, compute MMD between posterior samples and
        samples centered around true theta. Average over observations.
        """
        n_obs = theta_samples.shape[0]
        mmd_values = []

        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y) ** 2
            return torch.exp(-dist / (2 * sigma ** 2))

        for i in range(n_obs):
            samples = theta_samples[i]  # (n_samples, theta_dim)
            true_theta = theta_true[i:i+1].expand(len(samples), -1)

            xx = gaussian_kernel(samples, samples)
            yy = gaussian_kernel(true_theta, true_theta)
            xy = gaussian_kernel(samples, true_theta)

            mmd = xx.mean() + yy.mean() - 2 * xy.mean()
            mmd_values.append(float(mmd))

        return float(sum(mmd_values) / len(mmd_values)) if mmd_values else float('nan')

    def _conditional_wasserstein(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
    ) -> float:
        """Compute conditional Wasserstein-2 metric (averaged over observations).

        For each observation, compute W2 between posterior samples and
        point mass at true theta.
        """
        import ot

        n_obs = theta_samples.shape[0]
        ws_values = []

        for i in range(n_obs):
            samples = theta_samples[i]  # (n_samples, theta_dim)
            true_theta = theta_true[i:i+1].expand(len(samples), -1)
            n = len(samples)
            a = torch.ones(n) / n
            b = torch.ones(n) / n
            M = ot.dist(samples, true_theta)
            ws_values.append(float(torch.sqrt(ot.emd2(a, b, M))))

        return float(sum(ws_values) / len(ws_values)) if ws_values else float('nan')

    def _coverage(
        self,
        theta_true: Tensor,
        theta_samples: Tensor,
    ) -> float:
        """Compute ACAUC (paper definition from Appendix J).

        For each (observation, dimension) pair, computes the empirical CDF
        rank of the true parameter under the posterior samples. Returns the
        mean deviation from ideal calibration.

        ACAUC = 0 for perfect calibration, >0 overconfident, <0 underconfident.

        Args:
            theta_true: Ground truth theta (n_obs, theta_dim)
            theta_samples: Posterior samples (n_obs, n_samples, theta_dim)
        """
        from utils.metrics import acauc

        # Transpose to (n_samples, n_obs, theta_dim) for acauc function
        samples_transposed = theta_samples.transpose(0, 1)
        return acauc(theta_true, samples_transposed)


# Convenience functions


def evaluate_variant(
    cfg,
    variant_name: str,
    tasks: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Evaluate a variant.

    Args:
        cfg: Hydra configuration
        variant_name: Name of variant to evaluate
        tasks: List of tasks (defaults to cfg.tasks)
        skip_existing: Skip if metrics exist

    Returns:
        Evaluation results
    """
    evaluator = BenchmarkEvaluator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    return evaluator.evaluate_variant(variant_name, tasks, skip_existing=skip_existing)


def evaluate_baselines(
    cfg,
    tasks: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Evaluate baselines.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks (defaults to cfg.tasks)
        skip_existing: Skip if metrics exist

    Returns:
        Evaluation results
    """
    evaluator = BenchmarkEvaluator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    return evaluator.evaluate_baselines(tasks, skip_existing=skip_existing)


def evaluate_all(
    cfg,
    tasks: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Evaluate all methods and variants.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks (defaults to cfg.tasks)
        skip_existing: Skip if metrics exist

    Returns:
        Evaluation results
    """
    evaluator = BenchmarkEvaluator.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    return evaluator.evaluate_all(tasks, skip_existing=skip_existing)
