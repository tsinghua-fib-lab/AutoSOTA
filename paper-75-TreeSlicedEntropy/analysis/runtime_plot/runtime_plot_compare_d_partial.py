#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark several UOT solvers by varying dimension 'd'
and plot runtime / GPU-memory curves.

N is fixed at 10^4.
Ds = [1, 2, 8, 16, 32, 64, 128, 256]
Two warm-up runs are executed for every (method, d) combo
before the nrun timed repetitions, giving stabler timings.
"""
import os, time, math, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import ot  # POT
# --- Imports -----------------------------------------------------------------
from tsw import PartialTSW, generate_trees_frames
from baselines import sopt, spot, pawl, sliced_unbalanced_ot, unbalanced_sliced_ot
# -----------------------------------------------------------------------------

# ------------------------------ configuration --------------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.set_float32_matmul_precision('high')

nrun             = 10
N_fixed          = 10**4
Ds               = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] # Varying dimensions
results_dir      = f"benchmark_vs_D_{CURRENT_TIME_STR}_N{N_fixed}_nrun{nrun}"
os.makedirs(results_dir, exist_ok=True)

L_projections    = 10
tsw_ntrees, tsw_nlines = 5, 2
sopt_reg_val     = 1.0
pot_reg_val      = 0.1
pot_reg_m_kl_val = 1.0
pot_num_iter_max = 100
pot_stop_thr_val = 1e-5
pot_cost_metric_p_val = 2
uot_sliced_p_val = 2
uot_sliced_rho1_val, uot_sliced_rho2_val = 0.01, 1.0
uot_sliced_niter_val = 10
tsw_p_val, tsw_delta_val = 2, 2.0
tsw_mass_division_val = 'distance_based'
# -----------------------------------------------------------------------------

# --------------------------- helper functions --------------------------------
def warmup_system():
    print("System warm-up …")
    if DEVICE.type == 'cuda':
        # Use a size that's somewhat demanding but not excessive for warm-up
        warmup_size = min(2048, N_fixed) if N_fixed > 0 else 2048
        # Use a small dimension for general GPU warmup, d doesn't matter much here
        warmup_d = 2
        for _ in range(5):
            x = torch.randn(warmup_size, warmup_d, device=DEVICE) # Adjusted warmup size
            y = torch.randn(warmup_size, warmup_d, device=DEVICE)
            _ = x @ y.T # Matrix multiplication is a common GPU operation
        torch.cuda.synchronize()
    else: # CPU warmup
        warmup_size = min(2048, N_fixed) if N_fixed > 0 else 2048
        warmup_d = 2
        for _ in range(5):
            x = torch.randn(warmup_size, warmup_d)
            y = torch.randn(warmup_size, warmup_d)
            _ = x @ y.T
    print("Done.")

def run_and_measure_operation(op_func, is_gpu_op, *args, **kwargs):
    if is_gpu_op and DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    result = op_func(*args, **kwargs)
    if is_gpu_op and DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
        peak_mem = torch.cuda.max_memory_allocated(DEVICE)
    else:
        peak_mem = 0 # No GPU memory to track for CPU ops
    dt = time.perf_counter() - t0

    loss_val = 0.0
    if isinstance(result, torch.Tensor) and result.numel() == 1:
        loss_val = result.item()
    elif isinstance(result, (tuple, list)) and len(result) > 0 \
         and isinstance(result[0], torch.Tensor) and result[0].numel() == 1:
        loss_val = result[0].item()
    elif isinstance(result, (float, int)):
        loss_val = float(result)
    return dt, peak_mem, loss_val
# -----------------------------------------------------------------------------

# --------------------------- method registry ---------------------------------
method_definitions = {
    "SOPT": {
        "func": lambda X, Y: sopt(X.cpu(), Y.cpu(), n_proj=L_projections, reg=sopt_reg_val),
        "is_gpu_op": False},
    "SPOT": {
        "func": lambda X, Y: spot(X.cpu(), Y.cpu(), n_proj=L_projections),
        "is_gpu_op": False},
    "Sinkhorn": {
        "func": lambda X, Y, a, b: ot.unbalanced.sinkhorn_unbalanced2(
            a, b,
            ot.dist(X, Y, metric='sqeuclidean'), # Metric will be d-dimensional
            reg        = pot_reg_val,
            reg_m      = pot_reg_m_kl_val,
            method     = "sinkhorn_translation_invariant",
            numItermax = pot_num_iter_max,
            stopThr    = pot_stop_thr_val,
            log=True)[1]['cost'],
        "is_gpu_op": False, # POT typically runs on CPU for this setup
        "prepare_marginals": True},
    "USOT": {
        "func": lambda X, Y, mX, mY: unbalanced_sliced_ot(
            mX, mY, X, Y,
            num_projections=L_projections, p=uot_sliced_p_val,
            rho1=uot_sliced_rho1_val, rho2=uot_sliced_rho2_val,
            niter=uot_sliced_niter_val, mode='icdf')[0],
        "is_gpu_op": True,
        "prepare_marginals": True},
    "SUOT": {
        "func": lambda X, Y, mX, mY: sliced_unbalanced_ot(
            mX, mY, X, Y,
            num_projections=L_projections, p=uot_sliced_p_val,
            rho1=uot_sliced_rho1_val, rho2=uot_sliced_rho2_val,
            niter=uot_sliced_niter_val, mode='icdf')[0],
        "is_gpu_op": True,
        "prepare_mass_vectors": True}, # Note: uses mass vectors, not normalized marginals
    "PAWL": {
        "func": lambda X, Y: pawl(X.cpu(), Y.cpu(), n_proj=L_projections),
        "is_gpu_op": False},
    "PartialTSW": {
        # Object creation and compilation will happen inside the dimension loop
        "func": lambda obj, X, Y, theta, intercept: obj(X, Y, theta, intercept),
        "is_gpu_op": True,
        "uses_trees_frames": True,
        "is_compiled": True, # Signifies that the 'func' expects a compiled object
        "d_dependent_compilation": True # Custom flag
    }
}
method_names_ordered = ["SOPT", "SPOT", "Sinkhorn", "USOT",
                        "SUOT", "PAWL", "PartialTSW"]
# method_names_ordered = ["USOT", "SUOT", "PartialTSW"] # For quicker testing
# -----------------------------------------------------------------------------

# --------------------------- colour palette ----------------------------------
tab10  = list(plt.cm.tab10.colors)
colors = [tab10[0], tab10[1], tab10[3], tab10[4], tab10[5], tab10[6], tab10[2]]
# -----------------------------------------------------------------------------

# ------------------------------ main loop ------------------------------------
if __name__ == "__main__":
    warmup_system()

    num_methods = len(method_names_ordered)
    runtimes_s  = np.zeros((num_methods, len(Ds), nrun))
    mems        = np.zeros_like(runtimes_s)
    losses_log  = np.zeros_like(runtimes_s)

    for m_idx, name in enumerate(tqdm(method_names_ordered, desc="Methods")):
        info = method_definitions[name]

        for d_idx, d_val in enumerate(tqdm(Ds, desc=f"{name} (d)", leave=False)):
            # Handle d-dependent object creation/compilation
            current_method_obj = None
            if info.get("d_dependent_compilation"):
                if name == "PartialTSW": # Specific handling for PartialTSW
                    tsw_obj_uncompiled = PartialTSW(
                        ntrees=tsw_ntrees, nlines=tsw_nlines,
                        p=tsw_p_val, delta=tsw_delta_val,
                        mass_division=tsw_mass_division_val,
                        device=DEVICE)
                    current_method_obj = torch.compile(tsw_obj_uncompiled, mode="reduce-overhead")

            # --- Warm-up for this (method, d_val) ---
            # Generate data with current d_val and N_fixed
            Xc_w = torch.rand(N_fixed, d_val)
            Yc_w = torch.rand(N_fixed, d_val)

            # Move data to device if op is GPU based
            if info["is_gpu_op"] and DEVICE.type == 'cuda':
                Xw, Yw = Xc_w.to(DEVICE), Yc_w.to(DEVICE)
            else: # CPU op or CPU device
                Xw, Yw = Xc_w.cpu(), Yc_w.cpu()

            args_w = [Xw, Yw]
            if info.get("prepare_marginals"):
                mXw = torch.ones(N_fixed, device=Xw.device) / N_fixed
                mYw = torch.ones(N_fixed, device=Xw.device) / N_fixed
                args_w.extend([mXw, mYw])
            if info.get("prepare_mass_vectors"):
                mXw = torch.ones(N_fixed, device=Xw.device) # Unnormalized mass
                mYw = torch.ones(N_fixed, device=Xw.device) # Unnormalized mass
                args_w.extend([mXw, mYw])

            if info.get("is_compiled"): # If func expects a compiled object
                if current_method_obj is None:
                    raise ValueError(f"Method {name} is marked as compiled but no object was prepared.")
                args_w.insert(0, current_method_obj)

            if info.get("uses_trees_frames"):
                # Ensure d_val is used for generating frames
                theta_w, intercept_w = generate_trees_frames(
                    tsw_ntrees, tsw_nlines, d_val, device=Xw.device)
                args_w.extend([theta_w, intercept_w])

            # Two untimed warm-up calls
            for _ in range(2):
                run_and_measure_operation(info["func"], info["is_gpu_op"], *args_w)

            if info["is_gpu_op"] and DEVICE.type == 'cuda':
                torch.cuda.empty_cache() # Clear cache after warm-up specific to (method, d_val)

            # --- Timed repetitions ---
            for rep in range(nrun):
                Xc = torch.rand(N_fixed, d_val)
                Yc = torch.rand(N_fixed, d_val)

                if info["is_gpu_op"] and DEVICE.type == 'cuda':
                    X, Y = Xc.to(DEVICE), Yc.to(DEVICE)
                else:
                    X, Y = Xc.cpu(), Yc.cpu()

                args = [X, Y]

                if info.get("prepare_marginals"):
                    mX = torch.ones(N_fixed, device=X.device) / N_fixed
                    mY = torch.ones(N_fixed, device=X.device) / N_fixed
                    args.extend([mX, mY])
                if info.get("prepare_mass_vectors"):
                    mX = torch.ones(N_fixed, device=X.device)
                    mY = torch.ones(N_fixed, device=X.device)
                    args.extend([mX, mY])

                if info.get("is_compiled"):
                    args.insert(0, current_method_obj) # Use the same compiled obj from above

                if info.get("uses_trees_frames"):
                    theta, intercept = generate_trees_frames(
                        tsw_ntrees, tsw_nlines, d_val, device=X.device)
                    args.extend([theta, intercept])

                dt, mem, loss = run_and_measure_operation(
                    info["func"], info["is_gpu_op"], *args)
                runtimes_s[m_idx, d_idx, rep] = dt
                mems[m_idx, d_idx, rep] = mem
                losses_log[m_idx, d_idx, rep] = loss

                if info["is_gpu_op"] and DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()


    # ------------------------- post-processing -------------------------------
    avg_rt = np.nanmean(runtimes_s, axis=2)
    std_rt = np.nanstd(runtimes_s, axis=2)
    mem_mb = mems / (1024**2)
    avg_mb = np.nanmean(mem_mb, axis=2)
    std_mb = np.nanstd(mem_mb, axis=2)

    # Use string labels for Ds as they are not perfectly log-spaced
    xtick_labels = [str(d) for d in Ds]
    plt.style.use('seaborn-v0_8-whitegrid')

    # ----------------------------- runtime plot ------------------------------
    fig_rt, ax_rt = plt.subplots(figsize=(8, 6))
    for m_idx, name in enumerate(method_names_ordered):
        ax_rt.plot(Ds, avg_rt[m_idx], marker='o', color=colors[m_idx], label=name)

    ax_rt.set_xscale('log')
    ax_rt.set_yscale('log')
    ax_rt.set_xticks(Ds)
    ax_rt.set_xticklabels(xtick_labels, fontsize=10)
    ax_rt.set_xlabel(r"Dimension $d$", fontsize=13)
    ax_rt.set_ylabel("Runtime (s)", fontsize=13)

    # --- Improved Legend (Top) ---
    ax_rt.legend(
        loc='upper center',          # Anchor point on the legend box
        bbox_to_anchor=(0.5, 1.20),  # Position: 0.5 horizontal center, 1.20 means 20% of axes height above the top
        fontsize=10,
        ncol=min(num_methods, 4)     # Number of columns (e.g., 4 or adapt to num_methods)
    )
    ax_rt.grid(True, which='both', ls='--', lw=0.7)

    # Adjust layout to make space for the legend at the top and labels at bottom
    # rect = [left, bottom, right, top] in normalized figure coordinates
    fig_rt.tight_layout(rect=[0.05, 0.05, 0.95, 0.83]) # Made top smaller to give space for legend

    fig_rt.savefig(os.path.join(results_dir,
                    f"runtime_vs_D_N{N_fixed}_{CURRENT_TIME_STR}.pdf"), dpi=300, bbox_inches='tight')

    # ----------------------------- memory plot -------------------------------
    fig_mb, ax_mb = plt.subplots(figsize=(8, 6))
    has_gpu_data_to_plot = False
    for m_idx, name in enumerate(method_names_ordered):
        info = method_definitions[name]
        if info["is_gpu_op"] and DEVICE.type == 'cuda':
            ax_mb.plot(Ds, avg_mb[m_idx], marker='s', ls='--',
                       color=colors[m_idx], label=name)
            has_gpu_data_to_plot = True
        elif DEVICE.type == 'cuda':
             ax_mb.plot(Ds, np.zeros_like(avg_mb[m_idx]), marker='x', ls=':',
                       color=colors[m_idx], label=f"{name} (CPU)")
             if not has_gpu_data_to_plot and np.any(np.zeros_like(avg_mb[m_idx])):
                 has_gpu_data_to_plot = True

    if DEVICE.type == 'cuda' and has_gpu_data_to_plot:
        ax_mb.set_xscale('log')
        ax_mb.set_xticks(Ds)
        ax_mb.set_xticklabels(xtick_labels, fontsize=10)
        ax_mb.set_xlabel(r"Dimension $d$", fontsize=13)
        ax_mb.set_ylabel("Peak GPU Memory (MB)", fontsize=13)

        # --- Improved Legend (Top) ---
        ax_mb.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.20), # Positioned above the plot
            fontsize=10,
            ncol=min(num_methods, 4)    # Number of columns
        )
        ax_mb.grid(True, which='both', ls='--', lw=0.7)

        # Adjust layout to make space for the legend at the top and labels at bottom
        fig_mb.tight_layout(rect=[0.05, 0.05, 0.95, 0.83]) # Made top smaller

        fig_mb.savefig(os.path.join(results_dir,
                        f"memory_vs_D_N{N_fixed}_{CURRENT_TIME_STR}.pdf"), dpi=300, bbox_inches='tight')
    else:
        plt.close(fig_mb)
        if DEVICE.type == 'cuda':
             print("No GPU operations with memory data to plot for the memory graph.")
        else:
             print("No CUDA device found; GPU memory plot not generated.")


    plt.show()