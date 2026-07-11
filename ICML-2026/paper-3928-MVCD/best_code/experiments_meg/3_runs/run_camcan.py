import numpy as np
import pickle
from time import time
from pathlib import Path
import os
from limvam.ica_limvam import ica_limvam
from limvam.pairwise_limvam import pairwise_limvam
from limvam.direct_limvam import direct_limvam


# Limit the number of jobs
N_JOBS = 5
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=1"
)

# Parameters
algo = "direct_limvam"
random_state = 42

# Load data
expes_dir = Path("/storage/store4/work/aheurteb/LiMVAM/experiments_meg")
load_dir = expes_dir / f"2_data_envelopes/aparc_sub_152_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]  # shape (152, 10, 1760)

# Load labels
with open(load_dir / f"labels.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Get a list of all 38 labels
n_labels_total = max(len(set(x)) for x in labels_list)
labels = next(x for x in labels_list if len(set(x)) == n_labels_total)

# Predefined list of 10 labels/regions
selected_label_names = [
    'superiortemporal_3-lh',
    'superiortemporal_5-rh',
    'pericalcarine_1-lh',
    'pericalcarine_4-rh',
    'postcentral_6-lh',
    'postcentral_8-lh',
    'postcentral_7-rh',
    'postcentral_8-rh',
    'precentral_11-lh',
    'precentral_7-rh',
]

# Only keep the 98 subjects (out of 152) who have all these 10 regions available
X = []
for X_current, labels_current in zip(X_list, labels_list):
    label_names_current = {label.name for label in labels_current}
    if all(name in label_names_current for name in selected_label_names):
        label_to_row = {label.name: row for label, row in zip(labels_current, X_current)}
        filtered_X = np.array([label_to_row[name] for name in selected_label_names])
        X.append(filtered_X)
X = np.array(X)  # shape (98, 10, 1760)
labels = [label for label in labels if label.name in selected_label_names]
n_subjects = len(X)

# Apply causal discovery method
if algo == "ica_limvam":
    start = time()
    B, T, P = ica_limvam(
        X, ica_algo="shica_ml", random_state=random_state)
    execution_time = time() - start
elif algo == "pairwise_limvam":
    start = time()
    B, T, P = pairwise_limvam(X)
    execution_time = time() - start
elif algo == "direct_limvam":
    start = time()
    B, T, P = direct_limvam(X)
    execution_time = time() - start

print(f"\nThe method {algo} took {execution_time:.2f} s.\n")

# Save data
save_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects}_subjects_{algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "P.npy", P)
np.save(save_dir / "T.npy", T)
np.save(save_dir / "B.npy", B)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels, f)
