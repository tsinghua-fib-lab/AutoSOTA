from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import mne
import mne_connectivity
from mne_bids import BIDSPath, read_raw_bids
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator


# Global variables
DATA_DIR = Path("/storage/store/data")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"
FREESURFER_DIR = DATA_DIR / "camcan-mne/freesurfer"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
SRC = FREESURFER_DIR / "fsaverage/bem/fsaverage-ico-5-src.fif"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"


def process_data_one_subject(
    subject,
    tmin=-1.5,
    tmax=3.,
    baseline=(-1.5, -1.0),
    fmin=None,
    fmax=None,
    parcellation="aparc",
    normalize=True,
    orthogonalize=False,
    n_crop_edges=None,
    moving_avg=True,
    sfreq_envelope=None,
    n_batches=1,
    verbose=False,
):
    """Process data of one subject.
    """
    # Read raw data of one participant
    bp = BIDSPath(
        root=BIDS_ROOT,
        subject=subject,
        task="smt",
        datatype="meg",
        extension=".fif",
        session="smt",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Unable to map the following column.*")
        raw = read_raw_bids(bp, verbose=verbose)

    # Pick gradiometers and magnetometers
    raw.pick(picks=["grad", "mag"])
    # Load data in memory to speed up computations
    raw.load_data(verbose=verbose)
    # Maxwell filtering using calibration files
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=SSS_CAL_FILE, cross_talk=CT_SPARSE_FILE, verbose=verbose)
    # Band-pass filter at alpha or beta waves range
    if fmin is not None or fmax is not None:
        raw.filter(fmin, fmax, n_jobs=4, verbose=verbose)

    # Get events
    sfreq = raw.info["sfreq"]
    all_events, all_event_id = mne.events_from_annotations(
        raw, verbose=verbose)
    metadata, events, event_id = mne.epochs.make_metadata(
        events=all_events, event_id=all_event_id, tmin=tmin, tmax=0., sfreq=sfreq,
        row_events="button", keep_last="audiovis")

    # Add a column to metadata which counts the number of button events within [tmin, tmax]
    button_times = [
        event[0] / sfreq for event in all_events if event[2] == all_event_id["button"]
    ]
    nb_buttons = []
    for row_event_time in button_times:
        count = sum(
            1
            for button_time in button_times
            if tmin <= button_time - row_event_time <= tmax
        )
        nb_buttons.append(count)
    metadata["nb_buttons"] = nb_buttons

    # Create epochs
    epochs = mne.Epochs(
        raw, events, event_id, metadata=metadata, tmin=tmin, tmax=tmax, baseline=baseline,
        reject=dict(grad=4000e-13, mag=4e-12), preload=True, verbose=verbose)

    # Only keep epochs matching the conditions: 
    # (1) no other button press should occur within [tmin, tmax]
    # and (2) an audiovis stimulus should occur within one second
    epochs = epochs["audiovis > -1. and nb_buttons == 1"]
    n_epochs = len(epochs)
    print(f"{n_epochs} epochs found for subject {subject}.")

    # Compute the forward and inverse operators
    trans = TRANS_DIR / f"sub-{subject}-trans.fif"
    bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"
    src = mne.read_source_spaces(SRC, verbose=verbose)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem, n_jobs=1, verbose=verbose)
    del src
    cov = mne.compute_raw_covariance(raw, n_jobs=1, verbose=verbose)
    inv = make_inverse_operator(raw.info, fwd, cov, verbose=verbose)
    del fwd

    # Get the source time courses 
    # (inside the cortex; each source corresponds to a dipole)
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal",
        return_generator=False, verbose=verbose)

    # Get labels from "fsaverage"
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=parcellation, subjects_dir=FREESURFER_DIR, verbose=verbose)

    # Remove labels that do not have any vertex in the source space inv["src"]
    filtered_labels = [
        label for label in labels
        if (
            (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
            or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
        )
    ]

    # Select some interesting labels
    if parcellation == "aparc":
        # 10 already chosen labels
        label_names = [
            'paracentral-lh', 'precentral-lh', 'postcentral-lh', 'transversetemporal-lh', 
            'lateraloccipital-lh', 'paracentral-rh', 'precentral-rh', 'postcentral-rh',
            'transversetemporal-rh', 'lateraloccipital-rh']
    elif parcellation == "aparc_sub":
        # 38 already chosen labels
        label_names = [
            'postcentral_3-lh', 'postcentral_5-rh', 'postcentral_6-lh', 'postcentral_6-rh',
            'postcentral_7-lh', 'postcentral_7-rh', 'postcentral_8-lh', 'postcentral_8-rh',
            'postcentral_9-lh', 'postcentral_9-rh', 'precentral_10-lh', 'precentral_10-rh',
            'precentral_11-lh', 'precentral_11-rh', 'precentral_7-lh', 'precentral_7-rh',
            'precentral_8-lh', 'precentral_8-rh', 'precentral_9-lh', 'precentral_9-rh',
            'lateraloccipital_1-lh', 'lateraloccipital_1-rh', 'lateraloccipital_2-lh',
            'lateraloccipital_2-rh', 'lateraloccipital_3-rh', 'pericalcarine_1-lh',
            'pericalcarine_2-lh', 'pericalcarine_2-rh', 'pericalcarine_3-rh', 'pericalcarine_4-rh',
            'transversetemporal_1-lh', 'transversetemporal_1-rh', 'transversetemporal_2-lh',
            'superiortemporal_1-rh', 'superiortemporal_3-lh', 'superiortemporal_4-lh',
            'superiortemporal_5-lh', 'superiortemporal_5-rh',
        ]
    selected_labels = [label for label in filtered_labels if label.name in label_names]

    # Compute average time course across all sources (dipoles) 
    # that belong to each label (region of interest, ROI),
    # so there is one time series per epoch and per label.
    label_ts = mne.extract_label_time_course(
        stcs, selected_labels, inv["src"], return_generator=False, verbose=verbose)
    label_ts = np.array(label_ts)

    # Eventually normalize the label time series
    if normalize:
        baseline_bool = (stcs[0].times >= baseline[0]) & (stcs[0].times <= baseline[1])
        baseline_idx = np.arange(label_ts.shape[-1])[baseline_bool]
        baseline_means = np.mean(label_ts[:, :, baseline_idx], axis=-1, keepdims=True)
        baseline_stds = np.std(label_ts[:, :, baseline_idx], axis=-1, keepdims=True)
        label_ts = (label_ts - baseline_means) / baseline_stds

    # Eventually orthogonalize
    if orthogonalize:
        label_ts = mne_connectivity.symmetric_orth(label_ts)

    # Compute the envelope
    hilbert_ts = hilbert(label_ts, axis=2)
    envelope = np.abs(hilbert_ts)

    # Crop envelope to remove edge effects
    if n_crop_edges is not None:
        envelope = envelope[:, :, n_crop_edges:-n_crop_edges]

    # Subtract mean
    envelope -= np.mean(envelope, axis=-1, keepdims=True)

    # Perform batch-averaging
    if n_epochs < n_batches:
        print(f"There are {n_epochs} epochs for {n_batches} batches.")
        raise ValueError("There are less epochs than batches.")
    envelope = batch_average(envelope, n_batches=n_batches)

    # Downsample time points
    if sfreq_envelope is not None:
        factor = int(sfreq // sfreq_envelope)  # window size
        if moving_avg:
            kernel = np.ones(factor) / factor
            envelope = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode='valid'), axis=2, arr=envelope)
        envelope = envelope[:, :, ::factor]

    # Concatenate batches
    n_labels = len(selected_labels)
    envelope = envelope.swapaxes(0, 1).reshape(n_labels, -1)

    return envelope, selected_labels


def batch_average(envelope, n_batches):
    """Perform batch-averaging.
    """
    n_epochs, n_labels, n_timepoints = envelope.shape
    batch_size = n_epochs // n_batches
    remainder = n_epochs % n_batches
    results = []
    start = 0
    for i in range(n_batches):
        extra = 1 if i < remainder else 0
        end = start + batch_size + extra
        results.append(np.mean(envelope[start:end], axis=0))
        start = end
    return np.array(results)


def plot_envelope(
    envelope,
    n_batches=1,
    tmin=-1.5,
    tmax=3.,
    plot_avg=False,
    labels=None,
    save=False,
    subject=None,
):
    """Visualize envelope.
    """
    n_components, n_times = envelope.shape
    batch_size = n_times // n_batches
    if plot_avg:
        plt.subplots(figsize=(10, 8))
        envelope_avg = envelope[:, :batch_size].copy()
        for i in range(1, n_batches):
            envelope_avg += envelope[:, batch_size*i:batch_size*(i+1)]
        envelope_avg /= n_batches
        times = np.linspace(tmin, tmax, batch_size)
        if labels is not None:
            for i in range(n_components):
                plt.plot(times, envelope_avg[i], label=f"{labels[i].name}")
            plt.legend()
        else:
            plt.plot(times, envelope_avg.T)
        plt.vlines(
            x=0, ymin=np.min(envelope_avg), ymax=np.max(envelope_avg),
            linestyles="--", colors="grey")
        plt.xlabel("Time (s)")
    else:
        plt.subplots(figsize=(12, 4))
        if labels is not None:
            for i in range(n_components):
                plt.plot(envelope[i], label=f"{labels[i].name}")
            plt.legend()
        else:
            plt.plot(envelope.T)
        tick_positions = batch_size * np.arange(n_batches) + batch_size // 2
        tick_labels = np.arange(n_batches)
        plt.xticks(tick_positions, tick_labels)
        ymin, ymax = np.min(envelope), np.max(envelope)
        for i in range(n_batches+1):
            plt.vlines(x=batch_size*i, ymin=ymin, ymax=ymax, linestyles="--", colors="black")
        plt.xlabel("Batch")
    plt.grid()
    plt.ylabel("Mean amplitude")
    plt.title(f"Envelope of shape {envelope.shape}")
    if save:
        if subject is None:
            raise ValueError("subject should not be None when save=True")
        save_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments_meg/figures/"
        plt.savefig(save_dir + f"envelope_sub_{subject}.pdf")
    plt.show()


def get_participants():
    """Read participants file and only keep a subset of participants.
    """
    # Get the list of the 354 participants who have a trans file
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
    participants_names = participants[
        'participant_id'].str.replace('sub-', '', regex=False).tolist()
    names_trans_available = [
        name for name in participants_names 
        if (TRANS_DIR / f"sub-{name}-trans.fif").exists()]
    participants_with_trans = participants[
        participants['participant_id'].str.replace('sub-', '', regex=False).isin(
            names_trans_available)]

    # Only keep participants with clean evoked data (156 out of the 354 with a trans file)
    goods = [
        'CC110033', 'CC120182', 'CC120313', 'CC120376', 'CC120550', 'CC120727',
        'CC120795', 'CC121106', 'CC121111', 'CC121428', 'CC210051',
        'CC210617', 'CC220151', 'CC220352', 'CC220506', 'CC220518', 'CC220843',
        'CC220901', 'CC221031', 'CC221107', 'CC221220', 'CC221324', 'CC221352', 'CC221565',
        'CC222264', 'CC310086', 'CC310129', 'CC310135', 'CC310361', 'CC310400', 'CC310450',
        'CC310473', 'CC320160', 'CC320206', 'CC320218', 'CC320379', 'CC320417', 'CC320478',
        'CC320621', 'CC320680', 'CC320686', 'CC320776', 'CC321025', 'CC321069', 'CC321137',
        'CC321203', 'CC321281', 'CC321464', 'CC321544', 'CC321557', 'CC321594', 'CC321595',
        'CC321880', 'CC321899', 'CC322186', 'CC410040', 'CC410084', 'CC410086', 'CC410091',
        'CC410113', 'CC410119', 'CC410243', 'CC410284', 'CC412004', 'CC420004', 'CC420075',
        'CC420148', 'CC420157', 'CC420162', 'CC420167', 'CC420173', 'CC420182', 'CC420202',
        'CC420231', 'CC420260', 'CC420322', 'CC420324', 'CC420356', 'CC420396', 'CC420402',
        'CC420454', 'CC420589', 'CC420776', 'CC510043', 'CC510050', 'CC510220', 'CC510284',
        'CC510329', 'CC510392', 'CC510438', 'CC510474', 'CC510486', 'CC520011', 'CC520013',
        'CC520055', 'CC520211', 'CC520239', 'CC520247', 'CC520279', 'CC520377', 'CC520390',
        'CC520391', 'CC520503', 'CC520517', 'CC520585', 'CC520597', 'CC520607', 'CC520673',
        'CC520868', 'CC520980', 'CC521040', 'CC610071', 'CC610099', 'CC610212', 'CC610227',
        'CC610285', 'CC610344', 'CC610462', 'CC610625', 'CC610653', 'CC620005', 'CC620026',
        'CC620152', 'CC620193', 'CC620354', 'CC620413', 'CC620429', 'CC620659', 'CC621118',
        'CC621184', 'CC621248', 'CC621284', 'CC710037', 'CC710088', 'CC710342', 'CC710350',
        'CC710382', 'CC710462', 'CC710486', 'CC710664', 'CC720023', 'CC720180', 'CC720188',
        'CC720238', 'CC720290', 'CC720329', 'CC720670', 'CC720941', 'CC721052', 'CC721224',
        'CC721418', 'CC721504', 'CC721519', 'CC721729', 'CC722421', 'CC723395']
    filtered_participants = participants_with_trans[
        participants_with_trans['participant_id'].str.replace('sub-', '', regex=False).isin(
            goods)]

    return filtered_participants
