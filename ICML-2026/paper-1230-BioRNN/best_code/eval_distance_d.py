#!/usr/bin/env python3
"""
Canonical evaluation script for paper 1230 (BioRNN).
Computes the distance d metric (generalization error) for the heavy-tailed RNN
and baseline models on the MWG task.

This exactly replicates the Kruskal-Wallis evaluation section (lines 2066-2195)
of MWG_generalization.py, using the same array dimensions, parameters, and
computation. Pre-trained networks are loaded from MWG_NETS/ and
TrainedNets_Generalization/.

Metric: distance d = mean absolute difference between predicted output interval
        and linear target interval, averaged over test intervals and test trials,
        then averaged over 10 trained RNNs. Units: seconds. Lower is better.

Output: prints distance d for all five models.
        Saves results to eval_result.npz.

Usage:
    cd /repo && python3 eval_distance_d.py
"""
import os, sys, time as Time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load function/class definitions from MWG_generalization.py
# (skip the module-level execution block that does plotting)
# ---------------------------------------------------------------------------
with open("MWG_generalization.py", "r") as f:
    source = f.read()

exec_marker = "\ncls2 = set_plot()"
exec_start = source.find(exec_marker)
assert exec_start > 0, f"Cannot find execution block marker in MWG_generalization.py"

exec(source[:exec_start], globals())

# ---------------------------------------------------------------------------
# Replication of the Kruskal-Wallis evaluation (lines 2066-2195)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"hidden_size={hidden_size}, input_size={input_size}, output_size={output_size}", flush=True)
print(f"tss3: {len(tss3)} intervals from {tss3[0]:.0f} to {tss3[-1]:.0f} ms", flush=True)
print(f"Training range: {tss} ms", flush=True)
print(f"Models: BioRNN (heavy-tailed), FullRank (unconstrained), FullRank+L1, LowRank, E-I", flush=True)
print(f"Repeats: 10 trained RNNs per model", flush=True)

# Array dimensions match the original code: (10, 32, 10)
# Note: tss3 has only 30 values, so columns 30-31 remain zeros.
# This matches the original code's behavior.
gener_error_low = np.zeros((10, 32, 10))
gener_error_full = np.zeros((10, 32, 10))
gener_error_bio = np.zeros((10, 32, 10))
gener_error_ei = np.zeros((10, 32, 10))
gener_error_reg = np.zeros((10, 32, 10))

repeat = 10

t_start = Time.time()
for tr in range(repeat):
    A = np.load('TrainedNets/net_MWG1.npz')
    M = A['arr_0']
    N_arr = A['arr_1']
    Wo = A['arr_3']
    cond0 = A['arr_4']
    Wo = Wo / hidden_size
    if len(np.shape(Wo)) == 1:
        Wo = Wo[:, np.newaxis]

    dtype = torch.FloatTensor
    mrec_i = M
    nrec_i = N_arr
    mrec_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_I = torch.from_numpy(nrec_i).type(dtype)
    Is2 = np.zeros((hidden_size, input_size))
    Is2[:, input_size - 1] = N_arr[:, -2]
    inp_I = torch.from_numpy(Is2.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)
    h0_i = torch.from_numpy(cond0).type(dtype)

    net_low = OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                                     rank=rank, train_wi=True, train_wrec=True, train_wo=True, train_h0=True,
                                     wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init=h0_i)
    wrec_ei_dscosgd = create_wrec_init(hidden_size)
    net_EI = EIRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha, wrec_init=wrec_ei_dscosgd,
                   train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)
    net_fr_reg = FullRankRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha,
                             train_wi=True, train_wrec=True, train_wo=True, train_h0=True)
    net_fr = FullRankRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True)
    wrec_dscosgd = create_wrec_init(hidden_size)
    net_DScoSGD = BIORNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha, wrec_init=wrec_dscosgd,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)

    net_low.load_state_dict(torch.load(f"MWG_NETS/{tr}MWG_LowRank_Train_net.pt", map_location=device))
    net_fr.load_state_dict(torch.load(f"MWG_NETS/{tr}MWG_FullRank_Train_net.pt", map_location=device))
    net_DScoSGD.load_state_dict(torch.load(f"MWG_NETS/{tr}MWG_BioRNN_net.pt", map_location=device))
    net_EI.load_state_dict(torch.load(f"TrainedNets_Generalization/{tr}MWG_EIRNN_Train_net.pt", map_location=device))
    net_fr_reg.load_state_dict(torch.load(f"TrainedNets_Generalization/{tr}MWG_FullRankReg_Train_net.pt", map_location=device))

    net_low.to(device)
    net_fr.to(device)
    net_DScoSGD.to(device)
    net_EI.to(device)
    net_fr_reg.to(device)

    test_trials = 10
    for xx in range(len(tss3)):
        input_tr, output_tr, mask_tr, ct_tr, ct2_tr, ct3_tr = create_inp_out_MWG(
            test_trials, Nt, tss3 // dt, R_on + 100, 1, just=xx, perc=0., perc1=0.,
            fact=factor, align_set=True, delayF=100, inp_size=3)
        input_tr = input_tr.to(device)
        output_tr = output_tr.to(device)
        mask_tr = mask_tr.to(device)

        with torch.no_grad():
            outp = net_low.forward(input_tr, return_dynamics=False)
        outp_np = outp.detach().cpu().numpy()
        t0s_lr = time[np.argmin(np.abs(outp_np - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_lr_diff = np.abs(t0s_lr / 0.85 - tss3[xx])

        with torch.no_grad():
            outp = net_fr.forward(input_tr, return_dynamics=False)
        outp_np = outp.detach().cpu().numpy()
        t0s_fr = time[np.argmin(np.abs(outp_np - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_fr_diff = np.abs(t0s_fr / 0.85 - tss3[xx])

        with torch.no_grad():
            outp = net_EI.forward(input_tr, return_dynamics=False)
        outp_np = outp.detach().cpu().numpy()
        t0s_ei = time[np.argmin(np.abs(outp_np - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_ei_diff = np.abs(t0s_ei / 0.85 - tss3[xx])

        with torch.no_grad():
            outp = net_DScoSGD.forward(input_tr, return_dynamics=False)
        outp_np = outp.detach().cpu().numpy()
        t0s_bio = time[np.argmin(np.abs(outp_np - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_bio_diff = np.abs(t0s_bio / 0.85 - tss3[xx])

        with torch.no_grad():
            outp = net_fr_reg.forward(input_tr, return_dynamics=False)
        outp_np = outp.detach().cpu().numpy()
        t0s_reg = time[np.argmin(np.abs(outp_np - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_reg_diff = np.abs(t0s_reg / 0.85 - tss3[xx])

        gener_error_full[tr, xx] = np.squeeze(t0s_fr_diff)
        gener_error_low[tr, xx] = np.squeeze(t0s_lr_diff)
        gener_error_ei[tr, xx] = np.squeeze(t0s_ei_diff)
        gener_error_bio[tr, xx] = np.squeeze(t0s_bio_diff)
        gener_error_reg[tr, xx] = np.squeeze(t0s_reg_diff)

    elapsed = Time.time() - t_start
    print(f"  Run {tr+1}/{repeat} done ({elapsed:.1f}s elapsed)", flush=True)

# Compute distance d (lines 2178-2195 of original)
gener_error_full_arr = np.mean(np.mean(gener_error_full, 2), 1) / 1000
gener_error_low_arr = np.mean(np.mean(gener_error_low, 2), 1) / 1000
gener_error_ei_arr = np.mean(np.mean(gener_error_ei, 2), 1) / 1000
gener_error_bio_arr = np.mean(np.mean(gener_error_bio, 2), 1) / 1000
gener_error_reg_arr = np.mean(np.mean(gener_error_reg, 2), 1) / 1000

mean_full = np.mean(gener_error_full_arr)
mean_low = np.mean(gener_error_low_arr)
mean_ei = np.mean(gener_error_ei_arr)
mean_bio = np.mean(gener_error_bio_arr)
mean_reg = np.mean(gener_error_reg_arr)

sem_full = np.std(gener_error_full_arr) / np.sqrt(len(gener_error_full_arr))
sem_low = np.std(gener_error_low_arr) / np.sqrt(len(gener_error_low_arr))
sem_ei = np.std(gener_error_ei_arr) / np.sqrt(len(gener_error_ei_arr))
sem_bio = np.std(gener_error_bio_arr) / np.sqrt(len(gener_error_bio_arr))
sem_reg = np.std(gener_error_reg_arr) / np.sqrt(len(gener_error_reg_arr))

print("\n" + "="*70)
print("REPRODUCTION RESULT: DISTANCE d (GENERALIZATION ERROR)")
print("="*70)
print(f"{'Model':<25} {'Mean':>8} {'SEM':>8} {'Paper':>8}")
print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
print(f"{'heavy-tailed (BioRNN)':<25} {mean_bio:>8.4f} {sem_bio:>8.4f} {'~0.40':>8}")
print(f"{'unconstrained (FullRank)':<25} {mean_full:>8.4f} {sem_full:>8.4f} {'~0.55':>8}")
print(f"{'L1-regularization':<25} {mean_reg:>8.4f} {sem_reg:>8.4f} {'N/A':>8}")
print(f"{'low-rank':<25} {mean_low:>8.4f} {sem_low:>8.4f} {'N/A':>8}")
print(f"{'E-I':<25} {mean_ei:>8.4f} {sem_ei:>8.4f} {'N/A':>8}")
print("="*70)
print(f"\n  heavy-tailed per-run: {np.array2string(gener_error_bio_arr, precision=3, separator=', ')}")
print(f"\n  Metric direction: lower is better")
print(f"  Paper reports heavy-tailed d ~0.40, unconstrained d ~0.55")
print(f"  Rubin CI bounds for heavy-tailed: [0.385, 0.55]")
within_bounds = "BETTER THAN" if mean_bio < 0.385 else ("WITHIN" if mean_bio <= 0.55 else "WORSE THAN")
print(f"  Reproduced heavy-tailed d = {mean_bio:.4f} is {within_bounds} expected bounds")
print("="*70)

# Save results
np.savez("eval_result.npz",
         distance_d_bio=float(mean_bio), distance_d_bio_sem=float(sem_bio),
         distance_d_bio_per_run=gener_error_bio_arr,
         distance_d_full=float(mean_full), distance_d_full_sem=float(sem_full),
         distance_d_low=float(mean_low), distance_d_low_sem=float(sem_low),
         distance_d_ei=float(mean_ei), distance_d_ei_sem=float(sem_ei),
         distance_d_reg=float(mean_reg), distance_d_reg_sem=float(sem_reg))
print(f"\nResults saved to eval_result.npz")
