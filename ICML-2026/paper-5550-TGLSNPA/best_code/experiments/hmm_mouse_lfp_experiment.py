"""
Fit a TG-HMM to mouse sleep spindle LFPs.

"""
__date__ = "April - September 2025"


import argparse
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import morlet2, cwt
from tqdm import tqdm

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate_hmm import HMMPriorParams
from src.hmm import fit_hmm_em, forward_backward, batched_compute_scores
from src.hmm_eval import plot_z_history, plot_hmm_states_no_gt



if __name__ == '__main__':
    WRITE_DATA = False
    PLOT_PLV = True
    TRAIN = False
    APPLY = True

    parser = argparse.ArgumentParser(description="Fit HMM to spindles.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "data", "spindle_data")

    prior = HMMPriorParams()

    key = jax.random.PRNGKey(17)
    key1, key2, key3 = jax.random.split(key, 3)

    # number of regions
    n = 5
    F = 6
    FS = 250

    if WRITE_DATA:
        # Amy, MdThal, Nac, PRL, VTa
        d = jnp.load(os.path.join(data_dir, "spindles_refined_sorted.npz"))
        lfps = d["windows_raw"]
        invperm = np.argsort(d["perm"])
        lfps = lfps[invperm]
        N, T, C = lfps.shape
        all_cwt = jnp.zeros((N,C,F,T), dtype=jnp.float16)

        freqs = jnp.array([4,6,8,10,12,14]) # in Hz
        w = 5.0  # Morlet central frequency parameter
        # scale → width relation: f = w / (2π·scale)  ⇒ scale = w / (2π·f)
        scales = w * FS / (2 * jnp.pi * jnp.array(freqs))
        for i in tqdm(range(N)):
            for j in range(C):
                cwt_mat = cwt(lfps[i,:,j], lambda M, s: morlet2(M, s, w), scales) # [F,T]
                cwt_mat = jnp.angle(cwt_mat).astype(jnp.float16) # [F,T]
                all_cwt = all_cwt.at[i,j].set(cwt_mat)
        all_cwt = jnp.transpose(all_cwt, (0,3,1,2)) # (N,T,C,F)
        jnp.save(os.path.join(data_dir, "spindle_cwt.npy"), all_cwt)
        print("Saved data of shape:", all_cwt.shape)
        quit()


    # Load some contiguous chunk of LFP.
    full_data = jnp.load("spindle_cwt.npy") # (N,T,C,F)
    print("Original data shape:", full_data.shape)
    ts = np.linspace(-0.75, 0.75, full_data.shape[1])[::2] # match what follows
    full_data = full_data[:,::2]
    K = args.k

    if PLOT_PLV:
        N, T, C, F = full_data.shape
        full_data = full_data.reshape(N, T, -1).astype(jnp.float32)
        print("Modified data shape:", full_data.shape)
        mask = (np.abs(ts) < 0.2).astype(jnp.bool)
        phases_1 = full_data[:,mask].reshape(-1,C*F) # (N1, CF)
        print(phases_1.shape)
        phases_2 = full_data[:,~mask].reshape(-1,C*F) # (N2, CF)
        print(phases_2.shape)

        plv_1 = jnp.abs(jnp.mean(jnp.exp(1j * (phases_1[:,None] - phases_1[:,:,None])), axis=0))
        plv_2 = jnp.abs(jnp.mean(jnp.exp(1j * (phases_2[:,None] - phases_2[:,:,None])), axis=0))

        fig, axarr = plt.subplots(ncols=3, figsize=(8,3))
        plt.sca(axarr[0])
        plt.imshow(plv_1, vmin=0, vmax=1)
        plt.title("Spindle PLVs")
        plt.colorbar()
        plt.sca(axarr[1])
        plt.imshow(plv_2, vmin=0, vmax=1)
        plt.title("No Spindle PLVs")
        plt.colorbar()
        diff = plv_1 - plv_2
        vmax = np.max(np.abs(diff))
        plt.sca(axarr[2])
        plt.title("Spindle Minus No-Spindle")
        plt.imshow(diff, vmin=-vmax, vmax=vmax, cmap='bwr')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "spindle_plvs.pdf"))
        quit()


    if TRAIN:
        full_data = full_data[:100] # Just train on the first 100 spindles
        N, T, C, F = full_data.shape
        full_data = full_data.reshape(N, T, -1).astype(jnp.float32)
        print("Modified data shape:", full_data.shape)

        # Fit HMM with K states.
        info = fit_hmm_em(
            full_data,
            K,
            prior,
            num_em_iters=40,
            warmup_iterations=30,
            num_part_opt_steps=1000,
            tau_initial=20.0,
            lr=1e-2,
            phi_l2_reg=0.1,
            phi_l1_reg=0.03,
            cross_entropy_lambda=1e-1,
            beta=1.0,
            beta_min=0.1,
            beta_annealing="linear",
            init_method="zscore",
            n_init=100,
            num_trials=N,
        )

        try:
            joblib.dump(info, os.path.join(data_dir, f"hmm_spindle_info_{K}.joblib"))
        except:
            print("Unable to dump!")
            pass

        # Plot the hard state assignments over trials.
        info = joblib.load(os.path.join(data_dir, f"hmm_spindle_info_{K}.joblib"))
        zs = info["z_history"]
        zs = zs[-1].reshape(N,T)
        print("zs", zs.shape, N, T)
        plot_z_history(zs, K=K, fn=os.path.join(data_dir, f"hmm_all_spindle_zs_{K}.png"))

        # Plot the state probabilities over trials and within trials.
        unique_states = np.unique(zs)

        # Get colormap
        cmap = plt.get_cmap("Set1")
        colors = [cmap(i) for i in unique_states]

        # Average across N (over trials, at each time)
        occupancy_T = np.array([
            np.mean(zs == s, axis=0) for s in unique_states
        ])  # shape: (num_states, T)

        # Average across T (over time, for each trial)
        occupancy_N = np.array([
            np.mean(zs == s, axis=1) for s in unique_states
        ])  # shape: (num_states, N)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot over time
        for i, s in enumerate(unique_states):
            axes[0].plot(ts, occupancy_T[i], label=f"State {s}", color=colors[i])
        axes[0].set_title("Average state occupancy over N (per time step)")
        axes[0].set_xlabel("Time from Spindle Center")
        axes[0].set_ylabel("Occupancy probability")
        axes[0].legend()

        # Plot over trials
        for i, s in enumerate(unique_states):
            axes[1].plot(occupancy_N[i], label=f"State {s}", color=colors[i])
        axes[1].set_title("Average state occupancy over T (per trial)")
        axes[1].set_xlabel("Trial index")
        axes[1].set_ylabel("Occupancy probability")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(oa.path.join(data_dir, f"hmm_spindle_occupancy_{K}.png"))
        plt.close("all")

        # Plot the history of the hard state assignments over the iterations.
        plot_z_history(info["z_history"], K=K, fn=os.path.join(data_dir, f"hmm_spindle_zs_{K}.png"))

        # Plot the inferred states and transition matrix.
        plot_hmm_states_no_gt(K, info, fn=os.path.join(data_dir, f"hmm_spindle_{K}.png"))
        quit()

    if APPLY:
        # Apply the learned HMM to all of the spindle data.
        N, T, C, F = full_data.shape
        full_data = full_data.reshape(N, T, -1).astype(jnp.float32)
        print("Full data shape:", full_data.shape)

        # Load HMM pieces.
        d = joblib.load(os.path.join(data_dir, f"hmm_spindle_info_{K}.joblib"))
        log_pi = d["log_pi"]
        log_trans = d["log_trans"]
        log_part = d["log_part"]
        phis = d["phis"]
        del d

        @jax.jit
        def temp_func(xs):
            raw_scores    = batched_compute_scores(xs, phis, log_part)   # (T_total,K)
            log_emissions = raw_scores - jnp.max(raw_scores, axis=1, keepdims=True)
            log_emissions = log_emissions - logsumexp(log_emissions, axis=1, keepdims=True)
            gamma, _, _ = forward_backward(log_emissions, log_trans, log_pi)
            return jnp.argmax(gamma, axis=1)

        # Apply.
        all_sequences = jnp.zeros((N,T), dtype=jnp.int32)
        for i in tqdm(range(N)):
            out = temp_func(full_data[i])
            all_sequences = all_sequences.at[i].set(out)
        
        # Save state sequences.
        jnp.savez(os.path.join(data_dir, f"hmm_spindle_all_z_seq_{K}.npz"), all_seq=all_sequences)

        # Plot the state sequences with different permutations.
        zs = jnp.load(os.path.join(data_dir, f"hmm_spindle_all_z_seq_{K}.npz"))["all_seq"]
        print("zs", zs.shape)

        print("Applying permutation:")
        d = jnp.load(os.path.join(data_dir, "spindles_refined_sorted.npz"))
        zs = zs[d["perm"]]
        
        # Plot the state probabilities over trials and within trials.
        unique_states = np.unique(zs)
        print(
            "unique_states", unique_states,
        )

        # Get colormap
        cmap = plt.get_cmap("Set1")
        colors = [cmap(i) for i in unique_states]

        # Average across N (over trials, at each time)
        occupancy_T = np.array([
            np.mean(zs == s, axis=0) for s in unique_states
        ])  # shape: (num_states, T)

        # Average across T (over time, for each trial)
        occupancy_N = np.array([
            np.mean(zs == s, axis=1) for s in unique_states
        ])  # shape: (num_states, N)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot over time
        for i, s in enumerate(unique_states):
            axes[0].plot(ts, occupancy_T[i], label=f"State {s}", color=colors[i])
        axes[0].set_title("Average state occupancy over N (per time step)")
        axes[0].set_xlabel("Time from Spindle Center")
        axes[0].set_ylabel("Occupancy probability")
        axes[0].legend()

        # Plot over trials
        for i, s in enumerate(unique_states):
            axes[1].plot(occupancy_N[i], label=f"State {s}", color=colors[i])
        axes[1].set_title("Average state occupancy over T (per trial)")
        axes[1].set_xlabel("Trial index")
        axes[1].set_ylabel("Occupancy probability")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"hmm_inferred_spindle_occupancy_{K}.png"))
        plt.close("all")
    