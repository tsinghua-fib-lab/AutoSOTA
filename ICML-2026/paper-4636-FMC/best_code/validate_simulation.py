"""Main validation script for NPE/FMPE models trained on simulation data.

This script runs comprehensive validation checks on trained simulation models,
following best practices from top ML research and the sbi library.

Usage:
    # Run all validation checks on gaussian task
    python validate_simulation.py task=gaussian

    # Run specific checks
    python validate_simulation.py task=gaussian validation.checks=[sbc,coverage]

    # With sbi comparison
    python validate_simulation.py task=gaussian validation.sbi_comparison.enabled=true

    # For other tasks (no true posterior available for conditional check)
    python validate_simulation.py task=ou_process
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/omegaconf types to Python types for JSON serialization."""
    if isinstance(obj, DictConfig):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.checkpointing import CheckpointManager, ResultsPath
from ropefm.validation import (
    ConditionalMetricsCheck,
    CoverageCheck,
    LogProbCheck,
    PredictiveCheck,
    SBCCheck,
    SBIComparisonCheck,
    ValidationConfig,
    ValidationResult,
)
from simulator import get_simulator


def get_available_checks() -> Dict[str, type]:
    """Get mapping of check names to check classes."""
    return {
        "sbc": SBCCheck,
        "coverage": CoverageCheck,
        "predictive": PredictiveCheck,
        "log_prob": LogProbCheck,
        "conditional": ConditionalMetricsCheck,
        "sbi_comparison": SBIComparisonCheck,
    }


def run_validation(
    posterior,
    simulator,
    checks: List[str],
    config: ValidationConfig,
    model_type: str,
    **kwargs,
) -> Dict[str, ValidationResult]:
    """Run validation checks on a posterior.

    Args:
        posterior: Trained posterior model
        simulator: Simulator instance
        checks: List of check names to run
        config: Validation configuration
        model_type: "npe" or "fmpe" (for output organization)
        **kwargs: Additional arguments for specific checks

    Returns:
        Dictionary mapping check names to results
    """
    available_checks = get_available_checks()
    results = {}

    for check_name in checks:
        if check_name not in available_checks:
            print(f"Warning: Unknown check '{check_name}', skipping")
            continue

        # Create check-specific output directory
        check_config = ValidationConfig(
            num_simulations=config.num_simulations,
            num_posterior_samples=config.num_posterior_samples,
            n_obs_c2st=config.n_obs_c2st,
            alpha_levels=config.alpha_levels,
            seed=config.seed,
            device=config.device,
            batch_size=config.batch_size,
            save_plots=config.save_plots,
            output_dir=config.output_dir / model_type / check_name if config.output_dir else None,
        )

        check_cls = available_checks[check_name]
        check = check_cls(check_config)

        print(f"\n{'='*60}")
        print(f"Running {check_name} check for {model_type}...")
        print(f"{'='*60}")

        try:
            result = check.run(posterior, simulator, **kwargs)
            results[check_name] = result

            # Print summary
            status = "PASSED" if result.passed else "FAILED"
            print(f"\n{check_name}: {status}")

            if result.metrics:
                print("Metrics:")
                for k, v in result.metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    elif isinstance(v, dict) and len(v) <= 5:
                        print(f"  {k}: {v}")

            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  - {w}")

        except Exception as e:
            print(f"Error running {check_name}: {e}")
            import traceback
            traceback.print_exc()
            results[check_name] = ValidationResult(
                check_name=check_name,
                passed=False,
                details={"error": str(e)},
            )

    return results


def save_validation_results(
    results: Dict[str, ValidationResult],
    output_dir: Path,
    model_type: str,
):
    """Save validation results to JSON.

    Args:
        results: Dictionary of validation results
        output_dir: Base output directory
        model_type: "npe" or "fmpe"
    """
    model_dir = output_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to serializable format
    results_dict = {name: convert_to_serializable(result.to_dict()) for name, result in results.items()}

    # Add summary
    results_dict["_summary"] = {
        "total_checks": len(results),
        "passed": sum(1 for r in results.values() if r.passed),
        "failed": sum(1 for r in results.values() if not r.passed),
        "all_passed": bool(all(r.passed for r in results.values())),
    }

    output_path = model_dir / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main validation entry point."""
    print("="*60)
    print("Simulation Model Validation")
    print("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get task configuration
    task_cfg = cfg.task if hasattr(cfg, "task") else cfg
    task_name = task_cfg.name

    print(f"Task: {task_name}")

    # Setup paths
    results_dir = Path(cfg.get("results_dir", "./results"))
    experiment_name = cfg.get("experiment_name", "main_v1")
    paths = ResultsPath(results_dir, experiment_name)
    checkpoint_manager = CheckpointManager(paths)

    # Output directory for validation results
    output_dir = paths.base_path / "evaluation" / task_name / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get validation configuration
    val_cfg = cfg.get("validation", {})
    checks_to_run = val_cfg.get("checks", ["sbc", "coverage", "predictive", "log_prob", "conditional"])

    # Convert to list if needed
    if isinstance(checks_to_run, str):
        checks_to_run = [checks_to_run]
    elif hasattr(checks_to_run, "__iter__") and not isinstance(checks_to_run, list):
        checks_to_run = list(checks_to_run)

    # Add sbi comparison if enabled
    sbi_comparison_cfg = val_cfg.get("sbi_comparison", {})
    if sbi_comparison_cfg.get("enabled", False):
        checks_to_run.append("sbi_comparison")

    print(f"Checks to run: {checks_to_run}")

    # Create validation config
    validation_config = ValidationConfig(
        num_simulations=val_cfg.get("num_simulations", 1000),
        num_posterior_samples=val_cfg.get("num_posterior_samples", 1000),
        n_obs_c2st=val_cfg.get("n_obs_c2st", 5),
        alpha_levels=val_cfg.get("alpha_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        seed=cfg.get("seed", 42),
        device=device,
        batch_size=val_cfg.get("batch_size", 500),
        save_plots=val_cfg.get("save_plots", True),
        output_dir=output_dir,
    )

    # Create simulator
    sim_params = dict(task_cfg.simulator.params) if hasattr(task_cfg.simulator, "params") else {}
    simulator = get_simulator(task_cfg.simulator.name, **sim_params)
    print(f"Simulator: {simulator.name} (theta_dim={simulator.theta_dim}, obs_dim={simulator.obs_dim})")

    # Load training data for sbi comparison if needed
    training_data = None
    if "sbi_comparison" in checks_to_run:
        data = checkpoint_manager.load_shared_simulations(task_name)
        if data is not None:
            training_data = {"theta": data["theta"], "x": data["x"]}
            print(f"Loaded training data: {training_data['theta'].shape[0]} samples")

    # Get models to validate
    models_to_validate = val_cfg.get("models", ["npe", "fmpe"])
    if isinstance(models_to_validate, str):
        models_to_validate = [models_to_validate]

    all_results = {}

    for model_type in models_to_validate:
        print(f"\n{'#'*60}")
        print(f"# Validating {model_type.upper()}")
        print(f"{'#'*60}")

        # Load model
        posterior = checkpoint_manager.load_shared_simulation_model(task_name, model_type, device)

        if posterior is None:
            print(f"Warning: {model_type} model not found for task {task_name}")
            continue

        # Wrap in Posterior class if needed
        if not hasattr(posterior, "sample") or not callable(getattr(posterior, "sample", None)):
            print(f"Warning: {model_type} model doesn't have sample method, skipping")
            continue

        # Check model type and wrap appropriately
        posterior_name = f"{task_name}_{model_type}"
        if hasattr(posterior, "npe"):
            # It's an Estimator, wrap in DirectPosteriorEstimator
            from posteriors import DirectPosteriorEstimator
            wrapped_posterior = DirectPosteriorEstimator(posterior_name, posterior)
        elif hasattr(posterior, "drift"):
            # It's a FlowMatching model, wrap in DirectFlowMatchingPosterior
            from posteriors import DirectFlowMatchingPosterior
            wrapped_posterior = DirectFlowMatchingPosterior(posterior_name, posterior)
        else:
            # Assume it's already a Posterior
            wrapped_posterior = posterior

        # Run validation
        results = run_validation(
            wrapped_posterior,
            simulator,
            checks_to_run,
            validation_config,
            model_type,
            training_data=training_data,
            sbi_method=sbi_comparison_cfg.get("method", "snpe"),
        )

        all_results[model_type] = results

        # Save results
        save_validation_results(results, output_dir, model_type)

    # Print final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for model_type, results in all_results.items():
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        status = "ALL PASSED" if passed == total else f"{passed}/{total} PASSED"
        print(f"{model_type.upper()}: {status}")

        for check_name, result in results.items():
            status_icon = "✓" if result.passed else "✗"
            print(f"  {status_icon} {check_name}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
