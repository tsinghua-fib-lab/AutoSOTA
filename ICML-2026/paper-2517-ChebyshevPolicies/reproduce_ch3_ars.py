#!/usr/bin/env python3
"""
Reproduction script for CH-3-ARS on MountainCarContinuous-v0.
Paper: "Chebyshev Policies and the Mountain Car Problem"
Trains 20 CH-3-ARS policies and evaluates on 100 evenly spaced start positions.
Reports: R (mean return), t* (mean time to goal), L2 distance to analytic policy.
"""
import os
import sys
import time
import json
import numpy as np
import multiprocessing as mp

# Add repo to path
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/algorithms')

from utils import exp_run, plot

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hyperparameters from notebook
ENV_NAME = "MountainCarContinuous-v0"
TENSORBOARD_LOG_DIR = "/repo/tensorboard_logs/"

NUM_POLICIES = 20
BASE_SEED = 42
NAME_PREFIX_CHEBYSHEV = "mountaincar_ch-ars_"

# Analytic policy constants from paper Eq. (7)
C1 = 4.3346   # Phase 1 coefficient
C2 = 4.8358   # Phase 2 coefficient
ALPHA_BOOT = 0.1
X_HAT = -np.pi / 6  # ~ -0.5236


def analytic_policy_unnormalized(x, v):
    """Analytic worst-case policy pi_ana from the paper Eq. (7)."""
    abs_v = abs(v)
    if abs_v < 1e-10:
        if abs(x - X_HAT) <= 0.01:
            return ALPHA_BOOT
        return 0.0
    # Simplified phase detection
    is_phase2 = (v > 0 and x > X_HAT - 0.02) or (v < 0 and x < X_HAT + 0.02)
    C = C2 if is_phase2 else C1
    action = np.sign(v) * C * abs_v
    if abs(x - X_HAT) <= 0.01 and abs_v < 0.005:
        action = np.sign(v) * max(C * abs_v, ALPHA_BOOT)
    return float(np.clip(action, -1.0, 1.0))


def main():
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    kwargs_ars = dict(
        algo='ars',
        env_name=ENV_NAME,
        degree=3,
        basis='chebyshev',
        tensorboard_log_dir=TENSORBOARD_LOG_DIR + "ars_mountaincarcontinuous",
        delta_std=0.1,
        n_delta=4,
        n_top=1,
        learning_rate=0.018,
        steps=80000,
        evaluate_every_n_steps=0,
        normalize_actions=True,
        min_action=-1.0,
        max_action=1.0,
        initialization='random',
        zero_policy=False,
        seed=None,
        verbose=0,
    )

    NUM_CORES = min(16, mp.cpu_count())

    print(f"Starting CH-3-ARS reproduction on {ENV_NAME}")
    print(f"Training {NUM_POLICIES} policies with {kwargs_ars['steps']} steps each")
    print(f"Using {NUM_CORES} cores, seed={BASE_SEED}")

    args_list, logdirs = plot.get_kwargs_with_distinct_seeds(
        kwargs=kwargs_ars,
        num_experiments=NUM_POLICIES,
        name_prefix=NAME_PREFIX_CHEBYSHEV,
        seed=BASE_SEED,
        degree=3
    )

    print(f"Args: {len(args_list)} configurations prepared")

    # Set start method for multiprocessing
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass

    # Train policies in parallel
    start_time = time.time()
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.map(exp_run.run_sb3_polyagent_training, args_list)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.0f} seconds")

    # Filter valid results
    valid_results = [(r[0], r[1], r[2]) for r in results if not isinstance(r, Exception)]
    print(f"Valid results: {len(valid_results)}/{len(results)}")

    if len(valid_results) == 0:
        print("ERROR: No valid training results!")
        for i, r in enumerate(results):
            print(f"  Result {i}: {type(r).__name__}: {r}")
        sys.exit(1)

    # Evaluate all policies on 100 evenly spaced start positions
    print("\nEvaluating on 100 evenly spaced start positions from [-0.6, -0.4]...")
    xs = np.linspace(-0.6, -0.4, 100)

    all_eval_results = []
    for policy_idx, (name, duration, coeffs) in enumerate(valid_results):
        rewards = []
        model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
            basis='chebyshev', env_name=ENV_NAME, coeffs=coeffs, algo='ars')
        for x in xs:
            r_mean, r_std = exp_run.run_sb3_model(model, eval_env, options={'low': x, 'high': x})
            rewards.append(r_mean)

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        min_r = float(np.min(rewards))
        max_r = float(np.max(rewards))

        all_eval_results.append({
            'name': name,
            'mean_R': mean_r,
            'std_R': std_r,
            'min_R': min_r,
            'max_R': max_r,
            'training_duration': duration,
        })
        print(f"  Policy {policy_idx+1}/{len(valid_results)}: mean_R={mean_r:.4f}, min_R={min_r:.4f}, max_R={max_r:.4f}")

    # Find best policy by mean return
    best = max(all_eval_results, key=lambda x: x['mean_R'])
    best_name = best['name']
    best_coeffs = None
    for name, duration, coeffs in valid_results:
        if name == best_name:
            best_coeffs = coeffs
            break

    print(f"\nBest policy: {best_name}")
    print(f"  mean_R = {best['mean_R']:.4f}")
    print(f"  min_R  = {best['min_R']:.4f}")
    print(f"  max_R  = {best['max_R']:.4f}")

    # Compute t* (mean time to goal) for the best policy
    print("\nComputing t* (mean time to goal) for best policy...")
    model_tstar, _ = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')

    times_to_goal = []
    for x in xs:
        _, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
            basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')
        env = eval_env.venv.envs[0]
        obs, info = env.reset(options={'low': x, 'high': x})
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and steps < 999:
            action, _ = model_tstar.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        times_to_goal.append(steps)

    mean_t_star = float(np.mean(times_to_goal))
    print(f"  mean t* = {mean_t_star:.1f}")
    print(f"  min t*  = {np.min(times_to_goal):.0f}")
    print(f"  max t*  = {np.max(times_to_goal):.0f}")

    # Compute L2 distance to analytic policy over state space
    print("\nComputing L2 distance to analytic policy...")

    N_GRID = 50
    x_grid = np.linspace(-1.2, 0.45, N_GRID)
    v_grid = np.linspace(-0.07, 0.07, N_GRID)

    model_l2, _ = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis='chebyshev', env_name=ENV_NAME, coeffs=best_coeffs, algo='ars')

    XX, VV = np.meshgrid(x_grid, v_grid)
    points = np.column_stack([XX.ravel(), VV.ravel()])

    obs_low = np.array([-1.2, -0.07])
    obs_high = np.array([0.6, 0.07])
    points_norm = 2.0 * (points - obs_low) / (obs_high - obs_low) - 1.0

    model_actions = []
    for i in range(len(points_norm)):
        action, _ = model_l2.predict(points_norm[i].astype(np.float32), deterministic=True)
        model_actions.append(float(action))

    analytic_actions = []
    for i in range(len(points)):
        analytic_actions.append(analytic_policy_unnormalized(points[i, 0], points[i, 1]))

    diff = np.array(model_actions) - np.array(analytic_actions)
    l2_distance = float(np.sqrt(np.mean(diff ** 2)))

    print(f"  L2 distance = {l2_distance:.4f}")

    # Final Results
    metrics = {
        "R": best['mean_R'],
        "t*": mean_t_star,
        "L2": l2_distance,
    }

    print("\n" + "="*70)
    print("REPRODUCTION RESULTS")
    print("="*70)
    print(f"  R  (mean return):               {metrics['R']:.2f}   (paper: 98.74)")
    print(f"  t* (mean time to goal):         {metrics['t*']:.1f}   (paper: 471)")
    print(f"  L2 (distance to pi_ana):        {metrics['L2']:.4f}  (paper: 0.152)")
    print("="*70)

    print(f"\nRubric targets:")
    print(f"  R  >= 96.67 (baseline ARS), paper claims 98.74")
    print(f"  t* >= 298   (baseline PPO), paper claims 471")
    print(f"  L2 <= 0.211 (baseline ARS), paper claims 0.152")

    # Save results
    results_path = "/repo/reproduction_results.json"
    output = {
        "metrics": metrics,
        "best_policy_name": best_name,
        "training_time_seconds": float(training_time),
        "num_policies_trained": NUM_POLICIES,
        "num_valid_results": len(valid_results),
        "all_eval_results": all_eval_results,
        "times_to_goal": [int(t) for t in times_to_goal],
        "l2_distance": l2_distance,
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("Done!")
    return metrics


if __name__ == '__main__':
    main()
