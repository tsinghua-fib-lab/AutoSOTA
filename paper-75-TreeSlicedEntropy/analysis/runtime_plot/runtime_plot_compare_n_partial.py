#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark several UOT solvers and plot runtime / GPU-memory curves.

Changes vs. previous script
---------------------------
1.  Ns = [10**3, 10**4, 10**5, 10**6]  (unchanged)
2.  Even log-spaced x-axis with nice tick labels (unchanged)
3.  **NEW:** two warm-up runs are executed for every (method, N) combo
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
Ns               = [10**k for k in range(2, 6)]   # 10^2 … 10^5
d                = 2
results_dir      = f"benchmark_{CURRENT_TIME_STR}_d{d}_nrun{nrun}"
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
        for _ in range(5):
            x = torch.randn(2048, 2048, device=DEVICE)
            _ = x @ x
        torch.cuda.synchronize()
    else:
        for _ in range(5):
            x = torch.randn(2048, 2048)
            _ = x @ x
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
        peak_mem = 0
    dt = time.perf_counter() - t0
    # try to record a scalar loss value (optional)
    loss_val = 0.0
    if isinstance(result, torch.Tensor) and result.numel() == 1:
        loss_val = result.item()
    elif isinstance(result, (tuple, list)) and len(result) \
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
            ot.dist(X, Y, metric='sqeuclidean'),
            reg        = pot_reg_val,
            reg_m      = pot_reg_m_kl_val,
            method     = "sinkhorn_translation_invariant",
            numItermax = pot_num_iter_max,
            stopThr    = pot_stop_thr_val,
            log=True)[1]['cost'],
        "is_gpu_op": False,
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
        "prepare_mass_vectors": True},
    "PAWL": {
        "func": lambda X, Y: pawl(X.cpu(), Y.cpu(), n_proj=L_projections),
        "is_gpu_op": False},
    "PartialTSW": {
        "init_once": lambda d_val: torch.compile(
            PartialTSW(ntrees=tsw_ntrees, nlines=tsw_nlines,
                       p=tsw_p_val, delta=tsw_delta_val,
                       mass_division=tsw_mass_division_val,
                       device=DEVICE),
            mode="reduce-overhead"),
        "func": lambda obj, X, Y, theta, intercept: obj(X, Y, theta, intercept),
        "is_gpu_op": True,
        "uses_trees_frames": True,
        "is_compiled": True}
}
method_names_ordered = ["SOPT", "SPOT", "Sinkhorn", "USOT",
                        "SUOT", "PAWL", "PartialTSW"]
# method_names_ordered = ["USOT", "SUOT", "PartialTSW"]
# -----------------------------------------------------------------------------

# --------------------------- colour palette ----------------------------------
tab10  = list(plt.cm.tab10.colors)
colors = [tab10[0], tab10[1], tab10[3], tab10[4], tab10[5], tab10[6], tab10[2]]

# -----------------------------------------------------------------------------

# ------------------------------ main loop ------------------------------------
if __name__ == "__main__":
    warmup_system()

    num_methods = len(method_names_ordered)
    runtimes_s  = np.zeros((num_methods, len(Ns), nrun))
    mems        = np.zeros_like(runtimes_s)
    losses_log  = np.zeros_like(runtimes_s)

    # Initialise compiled objects once
    compiled_objs = {}
    for name in method_names_ordered:
        info = method_definitions[name]
        if "init_once" in info:
            compiled_objs[name] = info["init_once"](d)

    # Benchmark
    for m_idx, name in enumerate(tqdm(method_names_ordered, desc="Methods")):
        info = method_definitions[name]
        for n_idx, N in enumerate(tqdm(Ns, desc=name, leave=False)):
            # ---------------- warm-up for this (method, N) -------------------
            Xc_w = torch.rand(N, d)
            Yc_w = torch.rand(N, d)
            Xw   = Xc_w.to(DEVICE) if info["is_gpu_op"] and DEVICE.type=='cuda' else Xc_w
            Yw   = Yc_w.to(DEVICE) if info["is_gpu_op"] and DEVICE.type=='cuda' else Yc_w
            args_w = [Xw, Yw]
            if info.get("prepare_marginals"):
                mXw = torch.ones(N, device=Xw.device) / N
                mYw = torch.ones(N, device=Xw.device) / N
                args_w.extend([mXw, mYw])
            if info.get("prepare_mass_vectors"):
                mXw = torch.ones(N, device=Xw.device)
                mYw = torch.ones(N, device=Xw.device)
                args_w.extend([mXw, mYw])
            if "init_once" in info:
                args_w.insert(0, compiled_objs[name])
            if info.get("uses_trees_frames"):
                theta_w, intercept_w = generate_trees_frames(
                    tsw_ntrees, tsw_nlines, d, device=Xw.device)
                args_w.extend([theta_w, intercept_w])

            # two untimed warm-up calls
            for _ in range(2):
                run_and_measure_operation(info["func"], info["is_gpu_op"], *args_w)

            # ---------------- timed repetitions ------------------------------
            for rep in range(nrun):
                Xc = torch.rand(N, d);  Yc = torch.rand(N, d)
                X  = Xc.to(DEVICE) if info["is_gpu_op"] and DEVICE.type=='cuda' else Xc
                Y  = Yc.to(DEVICE) if info["is_gpu_op"] and DEVICE.type=='cuda' else Yc
                args = [X, Y]

                if info.get("prepare_marginals"):
                    mX = torch.ones(N, device=X.device) / N
                    mY = torch.ones(N, device=X.device) / N
                    args.extend([mX, mY])
                if info.get("prepare_mass_vectors"):
                    mX = torch.ones(N, device=X.device)
                    mY = torch.ones(N, device=X.device)
                    args.extend([mX, mY])
                if "init_once" in info:
                    args.insert(0, compiled_objs[name])
                if info.get("uses_trees_frames"):
                    theta, intercept = generate_trees_frames(
                        tsw_ntrees, tsw_nlines, d, device=X.device)
                    args.extend([theta, intercept])

                dt, mem, loss = run_and_measure_operation(
                    info["func"], info["is_gpu_op"], *args)
                runtimes_s[m_idx, n_idx, rep] = dt
                mems     [m_idx, n_idx, rep] = mem
                losses_log[m_idx, n_idx, rep] = loss

    # ------------------------- post-processing -------------------------------
    avg_rt = np.nanmean(runtimes_s, axis=2); std_rt = np.nanstd(runtimes_s, axis=2)
    mem_mb = mems / (1024**2)
    avg_mb = np.nanmean(mem_mb, axis=2); std_mb = np.nanstd(mem_mb, axis=2)

    xtick_labels = [fr"$10^{{{int(math.log10(n))}}}$" for n in Ns]
    plt.style.use('seaborn-v0_8-whitegrid')

    # ----------------------------- runtime plot ------------------------------
    fig_rt, ax_rt = plt.subplots(figsize=(8, 6))
    for m_idx, name in enumerate(method_names_ordered):
        ax_rt.plot(Ns, avg_rt[m_idx], marker='o', color=colors[m_idx], label=name)
    ax_rt.set_xscale('log'); ax_rt.set_yscale('log')
    ax_rt.set_xticks(Ns); ax_rt.set_xticklabels(xtick_labels, fontsize=11)
    ax_rt.set_xlim(min(Ns), max(Ns))
    ax_rt.set_xlabel(r"Number of Samples $n$", fontsize=13)
    ax_rt.set_ylabel("Runtime (s)",            fontsize=13)
    ax_rt.legend(fontsize=11)
    ax_rt.grid(True, which='both', ls='--', lw=0.7)
    fig_rt.tight_layout()
    fig_rt.savefig(os.path.join(results_dir,
                    f"runtime_vs_N_{CURRENT_TIME_STR}.pdf"), dpi=300)

    # ----------------------------- memory plot -------------------------------
    fig_mb, ax_mb = plt.subplots(figsize=(8, 6))
    for m_idx, name in enumerate(method_names_ordered):
        info = method_definitions[name]
        if info["is_gpu_op"] and DEVICE.type == 'cuda':
            ax_mb.plot(Ns, avg_mb[m_idx], marker='s', ls='--',
                       color=colors[m_idx], label=name)
        else:
            ax_mb.plot(Ns, np.zeros_like(avg_mb[m_idx]), marker='x', ls=':',
                       color=colors[m_idx], label=f"{name} (CPU)")
    ax_mb.set_xscale('log')
    ax_mb.set_xticks(Ns); ax_mb.set_xticklabels(xtick_labels, fontsize=11)
    ax_mb.set_xlim(min(Ns), max(Ns))
    ax_mb.set_xlabel(r"Number of Samples $n$", fontsize=13)
    ax_mb.set_ylabel("Peak GPU Memory (MB)",   fontsize=13)
    ax_mb.legend(fontsize=11)
    ax_mb.grid(True, which='both', ls='--', lw=0.7)
    fig_mb.tight_layout()
    fig_mb.savefig(os.path.join(results_dir,
                    f"memory_vs_N_{CURRENT_TIME_STR}.pdf"), dpi=300)

    plt.show()
