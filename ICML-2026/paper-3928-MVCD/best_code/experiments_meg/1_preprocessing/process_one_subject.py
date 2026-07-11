from utils import get_participants, process_data_one_subject, plot_envelope


# Parameters
subject_idx = 6
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

# Get DataFrame of participants
participants = get_participants()
subject = participants.iloc[subject_idx]['participant_id'].replace('sub-', '')

# Get envelope of shape (n_labels, n_batches*(tmax-tmin)*sfreq_envelope)
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

# Plot
plot_envelope(
    envelope,
    n_batches=n_batches,
    tmin=tmin,
    tmax=tmax,
    plot_avg=True,
    labels=labels,
    save=True,
    subject=subject,
)
