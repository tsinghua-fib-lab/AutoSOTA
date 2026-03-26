"""
DSP algorithms code to get spatial audio cues for SAVVY pipeline stage1 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import numpy as np
from numpy.fft import fft, ifft
import librosa
import json

from collections import Counter
from scipy.optimize import least_squares



import numpy as np
from scipy.signal import stft


from scipy.signal import welch, csd


def estimate_cdr_spherical(
    signals, fs, mic_positions, tdoas,
    f_low=500, f_high=2000,
    nperseg=1536, noverlap=768
):
    n_mics = signals.shape[0]
    eps = np.finfo(float).eps
    cdr_vals = []

    f, _ = welch(signals[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (f >= f_low) & (f <= f_high)

    for i in range(n_mics):
        for j in range(i+1, n_mics):
            _, Pii = welch(signals[i], fs, nperseg=nperseg, noverlap=noverlap)
            _, Pjj = welch(signals[j], fs, nperseg=nperseg, noverlap=noverlap)
            _, Pij = csd(signals[i], signals[j], fs, nperseg=nperseg, noverlap=noverlap)
            
            gamma_x = np.real(Pij) / (np.sqrt(Pii * Pjj) + eps)

            d = np.linalg.norm(mic_positions[i]-mic_positions[j])
            gamma_inf = np.sinc(2 * f * d / 343.0)
            tdoa_ij =  tdoas[(i, j)]
            gamma_s = np.cos(2 * np.pi * f * tdoa_ij)

            cdr_f = (gamma_x - gamma_inf) / (gamma_s - gamma_inf + eps)
            cdr_f = np.clip(cdr_f, 0, None)

            cdr_vals.append(np.mean(cdr_f[mask]))

    return np.mean(cdr_vals)


def estimate_cdr_midband(
    signals, fs, mic_positions,
    f_low=500, f_high=2000,
    nperseg=1536, noverlap=768
):
    """
    Welch-based blind CDR estimation restricted to a mid-frequency band.

    signals: [n_mics, n_samples]
    fs: sampling rate
    mic_positions: [n_mics, 3]
    f_low, f_high: low and high cutoff frequencies for mid-band (Hz)
    nperseg, noverlap: Welch parameters
    """
    n_mics = signals.shape[0]
    eps = np.finfo(float).eps
    cdr_pairs = []

    # Compute PSD and CSD via Welch for each mic pair
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            f, Pii = welch(signals[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pjj = welch(signals[j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pij = csd(signals[i], signals[j], fs=fs, nperseg=nperseg, noverlap=noverlap)

            # Compute magnitude coherence
            coh = np.real(Pij) / (np.sqrt(Pii * Pjj) + eps)

            # Model diffuse-field coherence
            d = np.linalg.norm(mic_positions[i] - mic_positions[j])
            gamma_inf = np.sinc(2 * f * d / 343.0)

            # CDR per frequency
            cdr_f = (coh - gamma_inf) / (1 - coh + eps)
            cdr_f = np.clip(cdr_f, 0, None)

            # Restrict to mid-frequency band
            mask = (f >= f_low) & (f <= f_high)
            if np.any(mask):
                cdr_mid = np.mean(cdr_f[mask])
            else:
                cdr_mid = 0.0

            cdr_pairs.append(cdr_mid)

    # Average across all pairs
    return np.mean(cdr_pairs)



def estimate_cdr_welch(signals, fs, mic_positions, nperseg=1280, noverlap=640):
    """
    Improved blind CDR estimation using Welch's method (averaged periodograms)
    for reduced variance on short (e.g., 250 ms) segments.

    signals: [n_mics, n_samples] multichannel audio chunk
    fs: sampling rate
    mic_positions: [n_mics, 3] array of 3D coordinates (meters)
    nperseg: window length for Welch (samples)
    noverlap: overlap length for Welch (samples)

    Returns:
        cdr_mean: average CDR across mic pairs
    """
    n_mics = signals.shape[0]
    eps = np.finfo(float).eps
    cdr_pairs = []
    
    # For each mic pair
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            # Compute auto and cross PSD with Welch
            f, Pii = welch(signals[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pjj = welch(signals[j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, Pij = csd(signals[i], signals[j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # Coherence (real part)
            coh = np.real(Pij) / (np.sqrt(Pii * Pjj) + eps)
            
            # Diffuse-field coherence
            d = np.linalg.norm(mic_positions[i] - mic_positions[j])
            gamma_inf = np.sinc(2 * f * d / 343.0)
            
            # CDR per frequency bin (clamped ≥0)
            cdr_f = (coh - gamma_inf) / (1 - coh + eps)
            cdr_f = np.clip(cdr_f, 0, None)
            
            # Average across frequency
            cdr_pairs.append(np.mean(cdr_f))
    
    # Mean across all mic pairs
    cdr_mean = np.mean(cdr_pairs)
    return cdr_mean


def estimate_cdr_blind(signals, fs, mic_positions, n_fft=1024, hop_len=512):
    """
    Blind CDR-based DRR estimation without any TDOA alignment.

    signals: [n_mics, n_samples] multichannel waveform
    fs: sampling rate (Hz)
    mic_positions: [n_mics, 3] array of 3D coordinates (meters)
    n_fft: FFT size for STFT
    hop_len: hop size for STFT

    Returns:
        cdr_mean: average coherent-to-diffuse ratio (≈ DRR) across mic pairs
    """
    n_mics = signals.shape[0]
    # Compute STFT for each channel: shape (n_mics, freq_bins, time_frames)
    freqs, _, STFT = stft(signals, fs, nperseg=n_fft, noverlap=n_fft - hop_len, axis=1)
    
    pair_cdrs = []
    eps = np.finfo(float).eps
    
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            S1 = STFT[i]  # shape (freq_bins, time_frames)
            S2 = STFT[j]
            # PSD estimates (freq_bins,)
            P11 = np.mean(np.abs(S1) ** 2, axis=1)
            P22 = np.mean(np.abs(S2) ** 2, axis=1)
            P12 = np.mean(S1 * np.conj(S2), axis=1)
            # Measured coherence
            gamma_x = np.abs(P12) / (np.sqrt(P11 * P22) + eps)
            # Diffuse-field coherence for distance d between mics
            d = np.linalg.norm(mic_positions[i] - mic_positions[j])
            gamma_inf = np.sinc(2 * freqs * d / 343.0)  # sinc(x)=sin(pi x)/(pi x)
            # Assume direct coherence ≈ 1 for broadband source
            cdr_f = (1 - gamma_x) / (gamma_x - gamma_inf + eps)
            # Average over frequencies, ignore negative/inf
            cdr_f = np.clip(cdr_f, 0, None)
            pair_cdrs.append(np.mean(cdr_f))
    
    # Return mean CDR across all mic pairs
    cdr_mean = np.mean(pair_cdrs)
    return cdr_mean





def estimate_source_position(tdoas, mic_pairs, mic_positions, c=343.0):
    """
    Estimate 3D source position from TDOAs and mic positions.

    Parameters:
      tdoas: List of TDOA values (in seconds)
      mic_pairs: List of mic index pairs [(i, j), ...]
      mic_positions: Array of shape (num_mics, 3)

    Returns:
      source_position: Estimated 3D coordinates (x, y, z)
    """
    def residuals(s):
        res = []
        for (i, j) in mic_pairs:
            tau = tdoas[(i, j)]
            di = np.linalg.norm(s - mic_positions[i])
            dj = np.linalg.norm(s - mic_positions[j])
            res.append((di - dj) / c - tau)
        return res

    # Initial guess: origin
    x0 = np.mean(mic_positions, axis=0)
    result = least_squares(residuals, x0)
    return result.x


def gcc_phat_full(x, y, fs, max_tau=None):
    """
    Compute the full GCC-PHAT cross-correlation between signals x and y.
    
    Returns:
      cc: cross-correlation array.
      lags: corresponding lags in samples.
    """
    n = x.shape[0] + y.shape[0]
    X = fft(x, n=n)
    Y = fft(y, n=n)
    R = X * np.conj(Y)
    R /= np.abs(R) + np.finfo(float).eps  # PHAT normalization
    cc = np.real(ifft(R))
    
    max_shift = int(n // 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    lags = np.arange(-max_shift, max_shift+1)
    
    if max_tau is not None:
        max_shift_allowed = int(fs * max_tau)
        valid = np.abs(lags) <= max_shift_allowed
        cc = cc[valid]
        lags = lags[valid]
    return cc, lags


def compute_mic_pair_weights(mic_pairs, angle, lateral_sensitive_pairs=[(5,6)],alpha=0, beta=0):
    """
    alpha = 1.8   # left-right direction gain coefficient
    beta = 1.5    # fron-back direction gain coefficient
    Parameters:
      mic_pairs: List of microphone pairs (i,j)
      angle: Candidate azimuth angle in degrees
      lateral_sensitive_pairs: Pairs that are sensitive to lateral (left-right) directions
    """
    weights = np.ones(len(mic_pairs))
    rad = np.radians(angle)

    sin_component = abs(np.sin(rad))  # lateral sensitivity
    cos_component = abs(np.cos(rad))  # frontal sensitivity
    
    w_base = 1.0  # base weight
    
    for i, (mic_i, mic_j) in enumerate(mic_pairs):
        # if mic 5 6
        if (mic_i, mic_j) in lateral_sensitive_pairs or (mic_j, mic_i) in lateral_sensitive_pairs:
            weights[i] = w_base + alpha * sin_component
        else: # if others
            weights[i] = w_base + beta * cos_component
    return weights


def srp_phat_localization(audio_chunk, fs, mic_positions, candidate_angles, alpha, beta, c=343.0):
    """
    Estimate the source azimuth (in degrees) using SRP-PHAT.
    """
    num_mics = mic_positions.shape[0]
    pair_corr = {}
    pair_lags = {}
    mic_pairs = []
    
    # emumerate GCC-PHAT for mics(i,j)
    for i in range(num_mics):
        for j in range(i+1, num_mics):
            mic_pairs.append((i, j))
            x = audio_chunk[:, i]
            y = audio_chunk[:, j]
            cc, lags = gcc_phat_full(x, y, fs, max_tau=0.001)
            # cc, lags = gcc_phat_full(x, y, fs, max_tau=0.0006)
            pair_corr[(i, j)] = cc
            pair_lags[(i, j)] = lags

    scores = np.zeros(len(candidate_angles))
    # define laterally sensitive microphone pairs
    lateral_sensitive_pairs = [(5,6)]
    
    # evaluate every candidate angle
    for idx, angle in enumerate(candidate_angles):
        # WEIGHTS
        weights = compute_mic_pair_weights(mic_pairs, angle, lateral_sensitive_pairs, alpha, beta)

        rad = np.radians(angle)
        u = np.array([-np.sin(rad), 0, np.cos(rad)])
        score_sum = 0.0
        for pair_idx, (i, j) in enumerate(mic_pairs):
            predicted_tau = np.dot(mic_positions[j] - mic_positions[i], u) / c
            predicted_samples = predicted_tau * fs
            cc = pair_corr[(i, j)]
            lags = pair_lags[(i, j)]
            idx_nearest = np.argmin(np.abs(lags - predicted_samples))
            score_sum += weights[pair_idx] * cc[idx_nearest]
        scores[idx] = score_sum

    best_idx = np.argmax(scores)
    best_angle = candidate_angles[best_idx]

    rad = np.radians(best_angle)
    u = np.array([-np.sin(rad), 0, np.cos(rad)])
    tdoas = {}
    for (i, j) in mic_pairs:
        predicted_tau = np.dot(mic_positions[j] - mic_positions[i], u) / c
        # tdoas.append(predicted_tau)
        tdoas[(i, j)] = predicted_tau

    return best_angle, tdoas, mic_pairs

def map_angle_to_label(angle):
    """
    Map an azimuth angle (in degrees, within [-180, 180]) to one of:
      'front-left', 'front-right', 'back-left', 'back-right'.
    
    Assumes that angles with absolute value less than 90° are 'front' and
    those with absolute value >= 90° are 'back'. Within each half, non-negative angles 
    map to 'left' and negative to 'right'.
    """
    # Normalize angle to [-180, 180]
    
    if abs(angle) < 2:
        return "front"
    elif abs(angle) > 178:
        return "back"
    elif abs(angle) < 90:
        return "front-left" if angle >= 0 else "front-right"
    else:
        return "back-left" if angle >= 0 else "back-right"


def test():
    # alpha=1.8, beta = 1.5
    alpha = 0
    beta = 0
    return alpha, beta



def doa_estimation(audio_chunk, fs, mic_positions):
    """
    Compute the DoA for an audio chunk.
    
    Returns:
      doa_label: Direction label
      best_angle: Estimated azimuth in degrees
      distance: Distance from microphone array center
      source_pos: Estimated 3D source position
    """
    candidate_angles = np.linspace(-180, 180, 361)
    alpha,beta = test()
    best_angle, tdoas, mic_pairs = srp_phat_localization(audio_chunk, fs, mic_positions, candidate_angles,alpha,beta)
    doa_label = map_angle_to_label(best_angle)

    source_pos = estimate_source_position(tdoas, mic_pairs, mic_positions)
    center = np.mean(mic_positions, axis=0)
    distance = np.linalg.norm(source_pos - center)

    return doa_label, best_angle, distance, source_pos, tdoas


# -------------------------------
# Main integration: reading audio, chunking, and processing
# -------------------------------

def get_doa_srp_phat_aria(audio_path, sample_rate = 48000, selected_indices = [0, 1, 2, 3, 4, 5, 6], fps = 4, output_json_path="audio_track_results.json", vote_by_second=False):
    all_waveform, sr = librosa.load(audio_path, sr=48000, mono=False)
    all_waveform = all_waveform.T
    
    # Check that the audio is 7-channel.
    if all_waveform.ndim == 1 or all_waveform.shape[1] != 7:
        raise ValueError(f"Expected 7-channel audio but got shape {all_waveform.shape}")

    # Function to chunk the audio into 2-second pieces.
    def chunk_audio(audio, fs, chunk_duration):
        num_samples = int(fs * chunk_duration)
        return [audio[i:i+num_samples] for i in range(0, len(audio), num_samples)
                if len(audio[i:i+num_samples]) == num_samples]

    
    waveform_list = chunk_audio(all_waveform, sr, 1./fps)
    
    mic_positions = np.array([
        [ 0.05, -0.04, 0.00],   # mic0
        [-0.005,  0.00, 0.00],   # mic1
        [-0.05, -0.04, 0.00],   # mic2
        [-0.07,  0.00, 0.00],   # mic3
        [0.07,  0.00, 0.00],    # mic4
        [-0.07,  0.00, -0.10],  # mic5
        [0.07,  0.00, -0.10]    # mic6
    ])

    
    
    mic_positions_selected = mic_positions[selected_indices, :]

    audio_track = {}
    import time
    all_time_list = []
    # Process each 2-second chunk to determine the DoA using only selected mics.
    for idx, chunk in enumerate(waveform_list):
        # Select only the channels for indices 3, 4, 5, and 6
        chunk_selected = chunk[:, selected_indices]
        # start_time_cur = time.time()
        doa_label, angle, distance, source_pos, tdoas = doa_estimation(chunk_selected, sr, mic_positions_selected)
        # print(f"Chunk {idx}: DoA label: {doa_label}, estimated azimuth: {angle:.2f}°, distance: {distance:.2f}")
        clip = abs(chunk_selected).mean(-1)
        
        audio_track[idx] = {
            "angle": -angle,
            "direction": doa_label,
            "distance": distance,
            "clip_energy": ((clip**2).sum())**0.5,
            "cdr": estimate_cdr_spherical(chunk_selected.T, sr, mic_positions_selected, tdoas)
        }

    # Aggregate the results by seconds (assuming fps is defined as frames per second)
    audio_track_seconds = {}
    for idx in range(len(waveform_list)):
        second = idx // fps
        
        # Initialize if this second doesn't exist yet
        if second not in audio_track_seconds:
            audio_track_seconds[second] = []
        
        # Add this chunk's data to the appropriate second
        audio_track_seconds[second].append(audio_track[idx])

    if vote_by_second:
        # Process each second to get the most common direction and average angle/distance
        final_audio_track = {}
        for second, chunks in audio_track_seconds.items():
            # Count directions to find the most common one
            directions = [chunk["direction"] for chunk in chunks]
            most_common_direction = Counter(directions).most_common(1)[0][0]
            
            # Filter chunks that have the most common direction
            filtered_chunks = [chunk for chunk in chunks if chunk["direction"] == most_common_direction]
            
            # Calculate mean angle and distance for the most common direction
            angles = [chunk["angle"] for chunk in filtered_chunks]
            distances = [chunk["distance"] for chunk in filtered_chunks]
            final_audio_track[second] = {
                "direction": most_common_direction,
                "angle": np.mean(angles),
                "distance": np.mean(distances)
            }
        # Save results to JSON
        with open(output_json_path, "w") as f:
            json.dump(final_audio_track, f, indent=4)
        return final_audio_track
    else:
        with open(output_json_path, "w") as f:
            json.dump(audio_track_seconds, f, indent=4)
        return audio_track_seconds
