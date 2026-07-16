#!/usr/bin/env python
"""
Evaluate CasADi MPC for 2D KS Equation with detailed timing.
"""
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt

from config import (N_grid, dt, L_domain, centers, sigma, horizon,
                    Q, R, u_min, u_max, T_sim, mpc_substeps, terminal_weight)
from ks_mpc import KSMPC2D
from ks_simulator import KSSolverJAX2D

def plot_comparison_2d(trajectory, traj_uncontrolled, u_target, controls,
                       sample_id, save_path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    
    mid_idx = len(trajectory) // 2
    end_idx = len(trajectory) - 1
    
    indices = [0, mid_idx, end_idx]
    
    for i, idx in enumerate(indices):
        vmax = max(np.max(np.abs(traj_uncontrolled[idx])), np.max(np.abs(trajectory[idx])), 1e-3)
        
        # Uncontrolled
        im1 = axes[i, 0].imshow(traj_uncontrolled[idx], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 0].set_title(f'Uncontrolled t={idx*dt:.1f}')
        if i == 0: axes[i, 0].set_ylabel('Y')
        axes[i, 0].set_xlabel('X')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Controlled
        im2 = axes[i, 1].imshow(trajectory[idx], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f'Controlled t={idx*dt:.1f}')
        axes[i, 1].set_xlabel('X')
        plt.colorbar(im2, ax=axes[i, 1])
        
        for cx, cy in centers:
            axes[i, 1].plot(cy * (N_grid/L_domain), cx * (N_grid/L_domain), 'ko', markersize=2)
            
        # Error
        error = trajectory[idx] - u_target
        im3 = axes[i, 2].imshow(error, origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title(f'Error t={idx*dt:.1f}')
        axes[i, 2].set_xlabel('X')
        plt.colorbar(im3, ax=axes[i, 2])
        
    ax_ctrl = axes[0, 3] 
    time_ax = np.arange(len(controls)) * dt * mpc_substeps
    for j in range(controls.shape[1]):
        ax_ctrl.plot(time_ax, controls[:, j])
    ax_ctrl.set_title("Control Inputs")
    ax_ctrl.set_xlabel("Time")
    ax_ctrl.set_ylabel("Amplitude")
    ax_ctrl.set_ylim([-55, 55])
    
    axes[1, 3].axis('off')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def timed_mpc_rollout(mpc, solver, u0, u_hat0, u_target, T_sim, mpc_substeps=1, verbose=False):
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
        
        if t % mpc_substeps == 0:
            solve_start = time.perf_counter()
            current_control, _, _ = mpc.solve(u, u_target, warm_start=True)
            solve_end = time.perf_counter()
            solve_times.append(solve_end - solve_start)
        
        u_next, u_hat_next = solver.step(u, u_hat, current_control)
        step_end = time.perf_counter()
        step_times.append(step_end - step_start)
        
        trajectory.append(u_next.copy())
        controls.append(current_control.copy())
        
        u = u_next
        u_hat = u_hat_next
        
        if verbose and t % 10 == 0:
            print(f"  Step {t:3d}/{T_sim}, solve: {solve_times[-1]*1000:.2f} ms | u_max: {np.max(np.abs(u)):.2f}")
    
    total_time = time.perf_counter() - total_start
    timing_stats = {
        'solve_times': np.array(solve_times),
        'step_times': np.array(step_times),
        'total_time': total_time,
        'rollout_steps': T_sim
    }
    return np.array(trajectory), np.array(controls), timing_stats

def simulate_uncontrolled(solver, u0, u_hat0, T_sim):
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
    output_dir = 'results_mpc_ks2d'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing solver (N={N_grid}, dt={dt})...")
    solver = KSSolverJAX2D(N_grid, L_domain, dt, sigma, centers)
    
    print(f"Building MPC NLP Dual-Grid (H={horizon}, N_mpc=32)...")
    mpc = KSMPC2D(N_sim=N_grid, N_mpc=32, L=L_domain, dt=dt, centers=centers, sigma=sigma, horizon=horizon, 
                  Q=Q, R=R, u_min=u_min, u_max=u_max, terminal_weight=terminal_weight)
    
    num_samples = args.num_samples
    u_target = np.zeros((N_grid, N_grid))
    
    all_solve_times, all_step_times, all_rollout_times, all_errors = [], [], [], []
    
    for i in range(num_samples):
        print(f"\n[{i+1}/{num_samples}] Generating chaotic IC...")
        u_init, u_hat_init = solver.generate_chaotic_ic(seed=42 + i * 13, steps=1000)
        
        mpc.U_init = None; mpc.A_init = None
        
        print(f"Running MPC rollout ({T_sim} steps)...")
        trajectory, controls, timing_stats = timed_mpc_rollout(
            mpc, solver, u_init, u_hat_init, u_target, T_sim, mpc_substeps=mpc_substeps, verbose=True
        )
        
        all_solve_times.extend(timing_stats['solve_times'].tolist())
        all_step_times.extend(timing_stats['step_times'].tolist())
        all_rollout_times.append(timing_stats['total_time'])
        
        final_errors_window = np.mean((trajectory[-50:] - u_target)**2)
        all_errors.append(final_errors_window)
        
        traj_unc = simulate_uncontrolled(solver, u_init, u_hat_init, T_sim)
        
        save_path = os.path.join(output_dir, f'mpc_ks_sample_{i}.png')
        plot_comparison_2d(trajectory, traj_unc, u_target, controls, i, save_path=save_path)
        
        print(f"Sample {i+1} | MSE (last 50 steps): {final_errors_window:.4e}")
    
    all_solve_times = np.array(all_solve_times)
    all_rollout_times = np.array(all_rollout_times)
    all_errors = np.array(all_errors)
    
    summary = (
        f"Samples evaluated: {num_samples}\n"
        f"Grid size: {N_grid}x{N_grid}\n"
        f"Final Window MSE:\n  Mean: {np.mean(all_errors):.4e}\n"
        f"Per-step MPC SOLVE time:\n  Mean: {np.mean(all_solve_times)*1000:.4f} ms\n"
    )
    print(summary)
    with open(os.path.join(output_dir, 'timing_summary.txt'), 'w') as f: f.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    main(args)
