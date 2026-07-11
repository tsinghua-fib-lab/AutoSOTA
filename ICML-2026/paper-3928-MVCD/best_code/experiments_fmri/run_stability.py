
#!/usr/bin/env python3
"""fMRI stability experiment for LiMVAM — mimics MEG protocol with fMRI data."""
import numpy as np
import pandas as pd
from pathlib import Path
import os, sys, time
from limvam.pairwise_limvam import pairwise_limvam

N_JOBS = 5
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"

# Params
n_runs = 30
n_subjects_batch = 5  # sample 5 out of 9 subjects
algo = "pairwise_limvam"
standardize = False  # match fMRI experiment default

# Load data
repo_dir = Path("/repo")
data_dir = repo_dir / "experiments_fmri" / "data"
df_numbers = ["001", "004", "005", "009", "010", "013", "014", "016", "017"]

X_list = []
for subj_id in df_numbers:
    filename = data_dir / f"sub{subj_id}.cent-table.9.dat.txt"
    df = pd.read_csv(filename, sep="	")
    data = df.to_numpy().T  # (9, 160)
    X_list.append(data)
X = np.array(X_list)  # (9, 9, 160)
cols = df.columns.values.tolist()
n_subjects_full = X.shape[0]
n_regions = X.shape[1]

print(f"Data shape: {X.shape}")
print(f"Variables: {cols}")
print(f"n_runs: {n_runs}, n_subjects_batch: {n_subjects_batch}")

# Run experiment
B_total = np.zeros((n_runs, n_subjects_batch, n_regions, n_regions))
P_total = np.zeros((n_runs, n_regions, n_regions))
T_total = np.zeros((n_runs, n_subjects_batch, n_regions, n_regions))

for i in range(n_runs):
    rng = np.random.RandomState(i)
    subjects_idx = rng.choice(n_subjects_full, size=n_subjects_batch, replace=False)
    X_subset = X[subjects_idx]
    
    t0 = time.time()
    B, T, P = pairwise_limvam(X_subset, standardize=standardize)
    elapsed = time.time() - t0
    
    B_total[i] = B
    T_total[i] = T
    P_total[i] = P
    print(f"Run {i+1}/{n_runs}: {elapsed:.1f}s, ordering={np.argmax(P, axis=1)}")

# Save
import sys; suffix = sys.argv[1] if len(sys.argv) > 1 else ""; out_dir = repo_dir / "experiments_fmri" / f"results_{n_subjects_batch}_subjects_{n_runs}_runs_{algo}{suffix}"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "B_total.npy", B_total)
np.save(out_dir / "T_total.npy", T_total)
np.save(out_dir / "P_total.npy", P_total)
print(f"Saved results to {out_dir}")
print(f"B_total shape: {B_total.shape}")
