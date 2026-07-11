import numpy as np
import pickle
from utils import get_participants, process_data_one_subject
import os
from pathlib import Path


# Not sure if the following lines limit the number of jobs when using the functions
# apply_inverse_epochs and make_inverse_operator from mne.minimum_norm
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Parameters
n_subjects = 156
tmin, tmax = -1.5, 3.
baseline = (-1.5, -1.0)
fmin, fmax = 8, 27
parcellation = "aparc_sub"
normalize = True
orthogonalize = False
n_crop_edges = 20
moving_avg = True
sfreq_envelope = 10
n_batches = 40

# Get subjects
participants = get_participants()
subjects = participants[
    'participant_id'].head(n_subjects).str.replace('sub-', '', regex=False).tolist()

# Get data
X_list = []
labels_list = []
for i, subject in enumerate(subjects):
    print(f"Starting processing of subject {i}: {subject}")
    try:
        envelope, labels = process_data_one_subject(
            subject,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            fmin=fmin,
            fmax=fmax,
            parcellation=parcellation,
            normalize=normalize,
            orthogonalize=orthogonalize,
            n_crop_edges=n_crop_edges,
            moving_avg=moving_avg,
            sfreq_envelope=sfreq_envelope,
            n_batches=n_batches,
        )
        X_list.append(envelope)
        labels_list.append(labels)
    except ValueError as e:
        if str(e) == "There are less epochs than batches.":
            print(f"Skipping index {i} due to ValueError: {e}")
        else:
            raise

# Save data
n_subjects_found = len(X_list)  # may be lower than n_subjects
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/experiments_meg")
save_dir = expes_dir / f"2_data_envelopes/{parcellation}_{n_subjects_found}_subjects"
save_dir.mkdir(parents=True, exist_ok=True)

np.savez(save_dir / f"X.npz", *X_list)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels_list, f)
