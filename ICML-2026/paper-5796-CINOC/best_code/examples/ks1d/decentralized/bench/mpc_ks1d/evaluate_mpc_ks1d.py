#!/usr/bin/env python
"""
Evaluate CasADi MPC for 1D KS Equation with detailed timing.

Measures:
- Per-step MPC solve time
- Total rollout time
- Mean squared errors and stabilization performance

Usage:
    python evaluate_mpc_ks1d.py --num-samples 10
"""

import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt

from config import (N_grid, dt, L_domain, centers, sigma, horizon,
                    Q, R, u_min, u_max, T_sim, mpc_substeps, terminal_weight)
from ks_mpc import KSMPC
from ks_simulator import KSSolverJAX


def plot_comparison(x, trajectory, traj_uncontrolled, u_target, controls,
                    dt, sample_id, save_path):
    """
    Plots the spacetime heatmap of the controlled vs uncontrolled KS trajectory
    along with the control signals.
    """
    T = len(trajectory)
    t_end = T * dt
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Uncontrolled Trajectory
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(traj_uncontrolled, aspect='auto', origin='lower',
               extent=[0, L_domain, 0, t_end], cmap='RdBu_r')
    ax1.set_title("Uncontrolled KS Field")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    plt.colorbar(im1, ax=ax1)

    # 2. Controlled Trajectory
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(trajectory, aspect='auto', origin='lower',
               extent=[0, L_domain, 0, t_end], cmap='RdBu_r')
    ax2.set_title("MPC Controlled KS Field")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    plt.colorbar(im2, ax=ax2)
    
    # Add vertical lines for actuators
    for c in centers:
        ax2.axvline(x=c, color='black', linestyle='--', alpha=0.3)

    # 3. Tracking Error
    error = trajectory - u_target
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(error, aspect='auto', origin='lower',
               extent=[0, L_domain, 0, t_end], cmap='coolwarm')
    ax3.set_title("Tracking Error (u - u_target)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    plt.colorbar(im3, ax=ax3)

    # 4. Control Signals
    ax4 = plt.subplot(2, 2, 4)
    time_ax = np.arange(len(controls)) * dt * mpc_substeps
    for i in range(controls.shape[1]):
        ax4.plot(time_ax, controls[:, i], label=f"a_{i+1} (x={centers[i]:.1f})")
    ax4.set_title("Control Inputs")
    ax4.set_xlabel("t")
    ax4.set_ylabel("amplitude")
    ax4.axhline(0, color='black', alpha=0.3)
    ax4.set_ylim([u_min - 5, u_max + 5])
    # For many controls, legend gets cluttered, skip if necessary or place outside
    if controls.shape[1] <= 8:
        ax4.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def timed_mpc_rollout(mpc, solver, u0, u_hat0, u_target, T_sim, mpc_substeps=1, verbose=False):
    """
    Run MPC simulation with detailed timing measurements.
    """
    trajectory = [u0.copy()]
    controls = []
    
    solve_times = []
    step_times = []
    
    u = u0.copy()
    u_hat = u_hat0.copy()
    current_control = np.zeros(len(centers))
    
    total_start = time.perf_counter()
    
    for t in range(T_sim):
        step_start = time.perf_counter()
        
        # Solve MPC every mpc_substeps steps
        if t % mpc_substeps == 0:
            solve_start = time.perf_counter()
            current_control, _, _ = mpc.solve(u, u_target)
            solve_end = time.perf_counter()
            solve_times.append(solve_end - solve_start)
        
        # Apply control to get next state (using JAX solver)
        u_next, u_hat_next = solver.step(u, u_hat, current_control)
        
        step_end = time.perf_counter()
        step_times.append(step_end - step_start)
        
        trajectory.append(u_next.copy())
        controls.append(current_control.copy())
        
        u = u_next
        u_hat = u_hat_next
        
        if verbose and t % 50 == 0:
            print(f"  Step {t:3d}/{T_sim}, solve: {solve_times[-1]*1000:.2f} ms | u_max: {np.max(np.abs(u)):.2f}")
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    
    timing_stats = {
        'solve_times': np.array(solve_times),
        'step_times': np.array(step_times),
        'total_time': total_time,
        'rollout_steps': T_sim
    }
    
    return np.array(trajectory), np.array(controls), timing_stats


def simulate_uncontrolled(solver, u0, u_hat0, T_sim):
    """Roll out dynamics with zero control"""
    trajectory = [u0.copy()]
    u = u0.copy()
    u_hat = u_hat0.copy()
    zero_ctrl = np.zeros(len(centers))
    for t in range(T_sim):
        u, u_hat = solver.step(u, u_hat, zero_ctrl)
        trajectory.append(u.copy())
    return np.array(trajectory)


def main(args):
    np.random.seed(42)
    
    print("=" * 60)
    print("CasADi MPC Evaluation for KS1D")
    print("=" * 60)
    
    output_dir = 'results_mpc_ks1d'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print(f"Initializing solver (N={N_grid}, L={L_domain}, dt={dt})...")
    solver = KSSolverJAX(N_grid, L_domain, dt, sigma, centers)
    
    print(f"Building MPC NLP (H={horizon}, Controls={len(centers)})...")
    mpc = KSMPC(N_grid, L_domain, dt, centers, sigma, horizon, 
                Q=Q, R=R, u_min=u_min, u_max=u_max, terminal_weight=terminal_weight)
    
    num_samples = args.num_samples
    u_target = np.zeros(N_grid)
    
    all_solve_times = []
    all_step_times = []
    all_rollout_times = []
    all_errors = []
    
    for i in range(num_samples):
        print(f"\n[{i+1}/{num_samples}] Generating chaotic IC...")
        # Use different seed for each sample burn-in to get un-correlated states
        # The solver.generate_chaotic_ic() compiles jax.lax.scan so first time will be slightly slow
        u_init, u_hat_init = solver.generate_chaotic_ic(seed=42 + i * 13, steps=1000)
        
        # Reset warm start
        mpc.U_init = None
        mpc.A_init = None
        
        # Run controlled trajectory
        print(f"Running MPC rollout ({T_sim} steps)...")
        trajectory, controls, timing_stats = timed_mpc_rollout(
            mpc, solver, u_init, u_hat_init, u_target, T_sim,
            mpc_substeps=mpc_substeps, verbose=True
        )
        
        # Collect timing data
        all_solve_times.extend(timing_stats['solve_times'].tolist())
        all_step_times.extend(timing_stats['step_times'].tolist())
        all_rollout_times.append(timing_stats['total_time'])
        
        # Compute error (average over the last 50 steps to assess actual stabilization effectiveness)
        final_errors_window = np.mean((trajectory[-50:] - u_target)**2)
        all_errors.append(final_errors_window)
        
        # Calculate uncontrolled trajectory for baseline visualization
        traj_unc = simulate_uncontrolled(solver, u_init, u_hat_init, T_sim)
        
        save_path = os.path.join(output_dir, f'mpc_ks_sample_{i}.png')
        plot_comparison(solver.dx * np.arange(N_grid), trajectory, traj_unc, 
                        u_target, controls, dt, sample_id=i, save_path=save_path)
        
        print(f"Sample {i+1} | MSE (last 50 steps): {final_errors_window:.4e} | "
              f"Solve: {np.mean(timing_stats['solve_times'])*1000:.2f} ms/step")
    
    # Timing Statistics Summary
    all_solve_times = np.array(all_solve_times)
    all_step_times = np.array(all_step_times)
    all_rollout_times = np.array(all_rollout_times)
    all_errors = np.array(all_errors)
    
    summary = (
        "\n" + "=" * 60 + "\n"
        "MPC EVALUATION SUMMARY\n" +
        "=" * 60 + "\n"
        f"Samples evaluated: {num_samples}\n"
        f"Rollout steps per sample: {T_sim}\n"
        f"MPC horizon: {horizon}\n"
        f"MPC solve frequency: every {mpc_substeps} step(s)\n\n"
        "--- Control Performance ---\n"
        f"Final Window MSE:\n"
        f"  Mean: {np.mean(all_errors):.4e}\n"
        f"  Std:  {np.std(all_errors):.4e}\n\n"
        "--- Timing Statistics ---\n"
        f"Per-step MPC SOLVE time (optimization only):\n"
        f"  Mean: {np.mean(all_solve_times)*1000:.4f} ms\n"
        f"  Std:  {np.std(all_solve_times)*1000:.4f} ms\n"
        f"  Min:  {np.min(all_solve_times)*1000:.4f} ms\n"
        f"  Max:  {np.max(all_solve_times)*1000:.4f} ms\n\n"
        f"Full rollout time ({T_sim} steps):\n"
        f"  Mean: {np.mean(all_rollout_times)*1000:.2f} ms\n"
    )
    
    print(summary)
    
    summary_file = os.path.join(output_dir, 'timing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CasADi MPC on KS1D")
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of chaotic IC samples to evaluate')
    args = parser.parse_args()
    main(args)
