import os
import mne
import numpy as np
import pickle
import argparse
import re


parser = argparse.ArgumentParser(description='Process EEG data for specified patient.')
parser.add_argument('--patient_id', '-p', required=True, help='Patient ID (e.g., PN01)')
args = parser.parse_args()

# Path configurations
base_dir = "path/to/siena-scalp-eeg-database-1.0.0"
patient_id = args.patient_id
output_dir = f"path/to/processed_segments/{patient_id}"

# Standard channel order (found across all EDFs, 29 channels)
STANDARD_CHANNELS = [
    "EEG Fp1", "EEG F3", "EEG C3", "EEG P3", "EEG O1", "EEG F7", "EEG T3", "EEG T5", "EEG Fc1", "EEG Fc5",
    "EEG Cp1", "EEG Cp5", "EEG F9", "EEG Fz", "EEG Cz", "EEG Pz", "EEG Fp2", "EEG F4", "EEG C4", "EEG P4",
    "EEG O2", "EEG F8", "EEG T4", "EEG T6", "EEG Fc2", "EEG Fc6", "EEG Cp2", "EEG Cp6", "EEG F10"
]

seizure_records = [
    # PN00
    {'file': 'PN00-1.edf', 'reg_start': '19.39.33', 'start_time': '19.58.36', 'end_time': '19.59.46'},
    {'file': 'PN00-2.edf', 'reg_start': '02.18.17', 'start_time': '02.38.37', 'end_time': '02.39.31'},
    {'file': 'PN00-4.edf', 'reg_start': '20.51.43', 'start_time': '21.08.29', 'end_time': '21.09.43'},
    {'file': 'PN00-5.edf', 'reg_start': '22.22.04', 'start_time': '22.37.08', 'end_time': '22.38.15'},
    # PN01
    {'file': 'PN01-1.edf', 'reg_start': '19.00.44', 'start_time': '21.51.02', 'end_time': '21.51.56'},
    {'file': 'PN01-1.edf', 'reg_start': '19.00.44', 'start_time': '07.53.17', 'end_time': '07.54.31'},
    # PN03
    {'file': 'PN03-1.edf', 'reg_start': '22.44.37', 'start_time': '09.29.10', 'end_time': '09.31.01'},
    {'file': 'PN03-2.edf', 'reg_start': '21.31.04', 'start_time': '07.13.05', 'end_time': '07.15.18'},
    # PN05
    {'file': 'PN05-2.edf', 'reg_start': '06.46.02', 'start_time': '08.45.25', 'end_time': '08.46.00'},
    {'file': 'PN05-3.edf', 'reg_start': '06.01.23', 'start_time': '07.55.19', 'end_time': '07.55.49'},
    {'file': 'PN05-4.edf', 'reg_start': '06.38.35', 'start_time': '07.38.43', 'end_time': '07.39.22'},
    # PN06
    {'file': 'PNO6-1.edf', 'reg_start': '04.21.22', 'start_time': '05.54.25', 'end_time': '05.55.29'},
    {'file': 'PNO6-2.edf', 'reg_start': '21.11.29', 'start_time': '23.39.09', 'end_time': '23.40.18'},
    {'file': 'PN06-3.edf', 'reg_start': '06.25.51', 'start_time': '08.10.26', 'end_time': '08.11.08'},
    {'file': 'PNO6-4.edf', 'reg_start': '11.16.09', 'start_time': '12.55.08', 'end_time': '12.56.11'},
    {'file': 'PN06-5.edf', 'reg_start': '13.24.41', 'start_time': '14.44.24', 'end_time': '14.45.08'},
    # PN07
    {'file': 'PN07-1.edf', 'reg_start': '23.18.10', 'start_time': '05.25.49', 'end_time': '05.26.51'},
    # PN09
    {'file': 'PN09-1.edf', 'reg_start': '14.08.54', 'start_time': '16.09.43', 'end_time': '16.11.03'},
    {'file': 'PN09-2.edf', 'reg_start': '15.02.09', 'start_time': '17.00.56', 'end_time': '17.01.55'},
    {'file': 'PN09-3.edf', 'reg_start': '14.20.23', 'start_time': '16.20.44', 'end_time': '16.21.48'},
    # PN10
    {'file': 'PN10-1.edf', 'reg_start': '05.40.05', 'start_time': '07.45.50', 'end_time': '07.46.59'},
    {'file': 'PN10-2.edf', 'reg_start': '09.30.15', 'start_time': '11.40.13', 'end_time': '11.41.04'},
    {'file': 'PN10-3.edf', 'reg_start': '13.33.18', 'start_time': '15.43.53', 'end_time': '15.45.02'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '12.49.50', 'end_time': '12.49.55'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '14.00.25', 'end_time': '14.00.44'},
    {'file': 'PN10-4.5.6.edf', 'reg_start': '12.11.21', 'start_time': '15.18.26', 'end_time': '15.19.23'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '17.35.13', 'end_time': '17.36.01'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '18.20.24', 'end_time': '18.20.42'},
    {'file': 'PN10-7.8.9.edf', 'reg_start': '16.49.25', 'start_time': '20.24.48', 'end_time': '20.25.03'},
    {'file': 'PN10-10.edf', 'reg_start': '08.45.22', 'start_time': '10.58.19', 'end_time': '10.58.33'},
    # PN11
    {'file': 'PN11-1.edf', 'reg_start': '11.31.25', 'start_time': '13.37.19', 'end_time': '13.38.14'},
    # PN12
    {'file': 'PN12-1.2.edf', 'reg_start': '15.51.31', 'start_time': '16.13.23', 'end_time': '16.14.26'},
    {'file': 'PN12-1.2.edf', 'reg_start': '15.51.31', 'start_time': '18.31.01', 'end_time': '18.32.09'},
    {'file': 'PN12-3.edf', 'reg_start': '08.42.35', 'start_time': '08.55.27', 'end_time': '08.57.03'},
    {'file': 'PN12-4.edf', 'reg_start': '15.59.19', 'start_time': '18.42.51', 'end_time': '18.43.54'},
    # PN13
    {'file': 'PN13-1.edf', 'reg_start': '08.24.28', 'start_time': '10.22.10', 'end_time': '10.22.58'},
    {'file': 'PN13-2.edf', 'reg_start': '06.55.02', 'start_time': '08.55.51', 'end_time': '08.56.56'},
    {'file': 'PN13-3.edf', 'reg_start': '12.00.01', 'start_time': '14.05.54', 'end_time': '14.08.25'},
    # PN14
    {'file': 'PN14-1.edf', 'reg_start': '11.44.58', 'start_time': '13.46.00', 'end_time': '13.46.27'},
    {'file': 'PN14-2.edf', 'reg_start': '15.50.13', 'start_time': '17.54.52', 'end_time': '17.55.04'},
    {'file': 'PN14-3.edf', 'reg_start': '16.17.45', 'start_time': '21.10.05', 'end_time': '21.10.46'},
    {'file': 'PN14-4.edf', 'reg_start': '14.18.30', 'start_time': '15.49.33', 'end_time': '15.50.56'},
    # PN16
    {'file': 'PN16-1.edf', 'reg_start': '20.45.21', 'start_time': '22.45.05', 'end_time': '22.47.08'},
    {'file': 'PN16-2.edf', 'reg_start': '00.53.55', 'start_time': '03.16.49', 'end_time': '03.18.36'},
    # PN17
    {'file': 'PN17-1.edf', 'reg_start': '20.14.28', 'start_time': '22.34.48', 'end_time': '22.35.58'},
    {'file': 'PN17-2.edf', 'reg_start': '13.52.18', 'start_time': '16.01.09', 'end_time': '16.02.32'}
]


def time_to_samples(time_str, start_time_str, sampling_rate):
    """Convert HH:MM:SS or HH.MM.SS time string to samples."""
    time_str = time_str.replace(':', '.')
    start_time_str = start_time_str.replace(':', '.')

    h, m, s = map(int, time_str.split('.'))
    start_h, start_m, start_s = map(int, start_time_str.split('.'))

    # Handle cases spanning across midnight
    if h < start_h:
        h += 24

    delta = (h - start_h) * 3600 + (m - start_m) * 60 + (s - start_s)
    return delta * sampling_rate


def process_edf(edf_path, seizure_records, output_dir, sampling_rate=512):
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        # Band-pass filter (0.1-75Hz)
        raw.filter(l_freq=0.1, h_freq=75.0)

        # Notch filter (50Hz power line noise)
        raw.notch_filter(freqs=50)

        # Sampling rate check
        if raw.info['sfreq'] != float(sampling_rate):
            return False, (f"Sampling rate mismatch in {os.path.basename(edf_path)}: "
                            f"expected {sampling_rate} Hz, got {raw.info['sfreq']} Hz")

        existing_channels = raw.ch_names
        # Create lowercase mapping: lowercase name -> original name
        existing_map = {ch.lower(): ch for ch in existing_channels}
        # Match channels, preserving original casing
        selected_channels = []
        reordered_channels = []
        missing_channels = []
        for ch in STANDARD_CHANNELS:
            ch_lower = ch.lower()
            if ch_lower in existing_map:
                real_name = existing_map[ch_lower]
                selected_channels.append(real_name)
                reordered_channels.append(real_name)
            else:
                missing_channels.append(ch)
        print(selected_channels)
        print(reordered_channels)
        # Print missing channels
        if missing_channels:
            print(
                f"Missing channels in {raw.filenames[0] if hasattr(raw, 'filenames') else 'EDF'}: {missing_channels}")
        # Pick and reorder channels
        raw.pick_channels(selected_channels)
        raw.reorder_channels(reordered_channels)

        # Get data
        data, _ = raw[:, :]

        # Get seizure events for the current file
        current_file = os.path.basename(edf_path)
        file_seizures = [sz for sz in seizure_records if sz["file"] == current_file]

        # Convert to samples
        seizure_samples = []
        for sz in file_seizures:
            start = time_to_samples(sz["start_time"], sz["reg_start"], sampling_rate)
            end = time_to_samples(sz["end_time"], sz["reg_start"], sampling_rate)
            seizure_samples.append((start, end))
        print(seizure_samples)

        # Generate all segments
        segments = []
        segment_length = 10 * sampling_rate

        # 1. Regular segmentation
        for i in range(0, data.shape[1], segment_length):
            if i + segment_length <= data.shape[1]:
                seg = data[:, i:i + segment_length]
                label = 0
                for sz_start, sz_end in seizure_samples:
                    if (i < sz_start < i + segment_length) or (i < sz_end < i + segment_length):
                        label = 1
                        break
                segments.append((seg, label))

        # 2. Seizure-enhanced segmentation
        for sz_start, sz_end in seizure_samples:
            start = max(0, sz_start - sampling_rate)
            end = min(data.shape[1], sz_end + sampling_rate)

            for i in range(start, end, 5 * sampling_rate):
                seg = data[:, i:i + segment_length]
                segments.append((seg, 1))

        # Save all segments
        base_name = os.path.splitext(current_file)[0]
        for idx, (seg, label) in enumerate(segments):
            output_data = {"X": seg, "Y": label}
            output_file = os.path.join(output_dir, f"{base_name}_{idx}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(output_data, f)

        return True, (f"Processed {current_file}, generated {len(segments)} segments (with 0.1-75Hz bandpass + 50Hz notch)")

    except Exception as e:
        return False, f"Error processing {os.path.basename(edf_path)}: {str(e)}"


def main():
    os.makedirs(output_dir, exist_ok=True)
    # Get all EDF files for the patient
    edf_files = sorted([f for f in os.listdir(os.path.join(base_dir, patient_id))
                        if f.endswith('.edf') and f.startswith(patient_id)])

    # Process each EDF file
    for edf_file in edf_files:
        edf_path = os.path.join(base_dir, patient_id, edf_file)
        print(f"Processing {edf_file}...")
        success, message = process_edf(edf_path, seizure_records, output_dir, sampling_rate=512)
        print(message)
    print(f"\nAll processing completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()