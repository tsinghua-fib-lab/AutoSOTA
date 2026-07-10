"""
Run All Rebuttal Experiments
============================
Execute all listed experiments sequentially, or specify individual ones.

Usage:
    python run_all.py              # Run all experiments
    python run_all.py 1 3 5        # Run experiments 1, 3, and 5
"""

import sys
import subprocess
import os
import time

EXPERIMENTS = {
    1: ('exp1_epsilon_sensitivity.py', 'Epsilon Sensitivity Sweep'),
    2: ('exp2_computational_overhead.py', 'Computational Overhead Table'),
    3: ('exp3_piecewise_affine.py', 'Piecewise-Affine Function Test'),
    4: ('exp4_parameter_matched.py', 'Parameter-Matched Baselines'),
    5: ('exp5_convergence_rate.py', 'Convergence Rate vs Hidden Width'),
    6: ('exp6_multilayer_cauchy.py', 'Multi-Layer CauchyNet'),
    7: ('exp7_uci_tabular.py', 'UCI Tabular Datasets'),
    8: ('exp8_rational_nn.py', 'Rational Neural Networks Comparison'),
    9: ('exp9_delta_sweep.py', 'Near-Singularity Delta Sweep'),
    10: ('exp10_sample_efficiency.py', 'Sample Efficiency Sweep'),
    11: ('exp11_fixed_pole_ridge.py', 'Fixed-Pole Ridge on Mixed 1D Target'),
}


def run_experiment(exp_id):
    script, name = EXPERIMENTS[exp_id]
    script_path = os.path.join(os.path.dirname(__file__), script)
    print(f"\n{'#' * 60}")
    print(f"# Experiment {exp_id}: {name}")
    print(f"# Script: {script}")
    print(f"{'#' * 60}\n")

    start = time.time()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        env=env,
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n>>> Experiment {exp_id} finished: {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    if len(sys.argv) > 1:
        exp_ids = [int(x) for x in sys.argv[1:]]
    else:
        exp_ids = sorted(EXPERIMENTS.keys())

    print(f"Running experiments: {exp_ids}")
    results = {}
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_id}")
            continue
        results[exp_id] = run_experiment(exp_id)

    print(f"\n{'='*60}")
    print("Summary")
    print('=' * 60)
    for exp_id, success in results.items():
        _, name = EXPERIMENTS[exp_id]
        status = "PASSED" if success else "FAILED"
        print(f"  Experiment {exp_id} ({name}): {status}")


if __name__ == '__main__':
    main()
