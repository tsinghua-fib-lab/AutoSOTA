import os
import pandas as pd
import mne
import numpy as np
import pickle

# File path configuration
edf_base_path = "path/to/tuh_eeg_slowing/v2.0.1/edf"
csv_file_path = "path/to/tuh_eeg_slowing/v2.0.1/json_process/detail.csv"
processed_data_path = "path/to/tuh_eeg_slowing/v2.0.1/processed_data"

# Channels to be renamed for consistency
rename_channels = {
    'EEG FP1-REF': 'FP1', 'EEG FP2-REF': 'FP2', 'EEG F3-REF': 'F3',
    'EEG F4-REF': 'F4', 'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4',
    'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4', 'EEG O1-REF': 'O1',
    'EEG O2-REF': 'O2', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8',
    'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5',
    'EEG T6-REF': 'T6', 'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2',
    'EEG FZ-REF': 'FZ', 'EEG CZ-REF': 'CZ', 'EEG PZ-REF': 'PZ',
    'EEG T1-REF': 'T1', 'EEG T2-REF': 'T2',
    'EEG FP1-LE': 'FP1', 'EEG FP2-LE': 'FP2', 'EEG F3-LE': 'F3',
    'EEG F4-LE': 'F4', 'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4',
    'EEG P3-LE': 'P3', 'EEG P4-LE': 'P4', 'EEG O1-LE': 'O1',
    'EEG O2-LE': 'O2', 'EEG F7-LE': 'F7', 'EEG F8-LE': 'F8',
    'EEG T3-LE': 'T3', 'EEG T4-LE': 'T4', 'EEG T5-LE': 'T5',
    'EEG T6-LE': 'T6', 'EEG A1-LE': 'A1', 'EEG A2-LE': 'A2',
    'EEG FZ-LE': 'FZ', 'EEG CZ-LE': 'CZ', 'EEG PZ-LE': 'PZ',
    'EEG T1-LE': 'T1', 'EEG T2-LE': 'T2'
}

def load_csv_labels(csv_path):
    return pd.read_csv(csv_path)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_edf_file(edf_file, labels_df):
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    current_channels = raw.info['ch_names']
    print(f"Before picking channels in {edf_file}:")
    print(f"Number of Channels: {len(current_channels)}")
    print(f"Channels: {current_channels}")

    # Retain and rename only channels present in `rename_channels`
    valid_channels = {ch: rename_channels[ch] for ch in rename_channels if ch in current_channels}
    raw.rename_channels(valid_channels)
    
    # Keep only the required channels
    raw.pick_channels(list(valid_channels.values()))

    filtered_channels = raw.info['ch_names']
    print(f"After picking channels in {edf_file}:")
    print(f"Number of Channels: {len(filtered_channels)}")
    print(f"Channels: {filtered_channels}")
    
    # Apply band-pass filter
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
    # Apply notch filter
    raw.notch_filter(freqs=60.0, verbose=False)

    edf_name = os.path.basename(edf_file).replace('.edf', '')
    edf_labels = labels_df[labels_df['filename'] == edf_name]

    for _, row in edf_labels.iterrows():
        start, stop, label = row['start_time'], row['stop_time'], row['label']
        save_slices(raw, edf_name, start, stop, label)

def save_slices(raw, edf_name, start_time, stop_time, label):
    labels_map = {'bckg': 0, 'seiz': 1, 'slow': 2}
    label_value = labels_map[label]
    
    start_sample = int(start_time * raw.info['sfreq'])
    stop_sample = int(stop_time * raw.info['sfreq'])
    segment_length = int(10 * raw.info['sfreq']) # 10-second segments
    folder_name = edf_name[:8]
    output_folder = os.path.join(processed_data_path, folder_name)
    create_folder(output_folder)
    
    for i, start in enumerate(range(start_sample, stop_sample, segment_length)):
        end = start + segment_length
        if end > stop_sample:
            break
        
        segment_data = raw.get_data(start=start, stop=end) 
        raw_segment = mne.io.RawArray(segment_data, info=raw.info)
        raw_segment.resample(250, npad="auto", verbose=False)

        output_file = os.path.join(output_folder, f"{edf_name}_{label}_{i}.pkl")
        
        # Ensure unique file names
        index = 0
        while os.path.exists(output_file):
            index += 1
            output_file = os.path.join(output_folder, f"{edf_name}_{label}_{i}_{index}.pkl")
            
        with open(output_file, 'wb') as f:
            pickle.dump({'X': raw_segment.get_data(), 'Y': label_value}, f)
        
        print(f"Saved: {output_file}")

def process_all_edf_files():
    labels_df = load_csv_labels(csv_file_path)
    for root, _, files in os.walk(edf_base_path):
        for file in files:
            if file.endswith('.edf'):
                edf_file = os.path.join(root, file)
                print(f"Processing: {edf_file}")
                process_edf_file(edf_file, labels_df)

if __name__ == "__main__":
    process_all_edf_files()