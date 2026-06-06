import os
import pandas as pd
import mne
import math
import pickle
import pdb

# Define input and output paths
edf_path = 'path/to/edf/recordings'
csv_folder = 'path/to/csv/recordings/'
output_path = 'path/to/output/HMC_processed_pkl_256Hz/'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

edf_files = [f for f in os.listdir(edf_path) if ((f.endswith('.edf')) and ("sleepscoring" not in f))]
fs = 256

used_label = ['Sleep stage W,', 'Sleep stage N1,', 'Sleep stage N2,', 'Sleep stage N3,', 'Sleep stage R,']
label_to_number = {label: index for index, label in enumerate(used_label)}

file_num = 0

for idx, file in enumerate(edf_files):
    print(f"Processing {idx+1}/{len(edf_files)}...") 
    file_name = file.split("/")[-1].split(".")[0]
    file_path = os.path.join(edf_path, file)
    csv_path = csv_folder + file_name + '_sleepscoring.csv'
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
    except Exception as e:
        print(f"Error processing file {edf_path}: {e}")
    
    droped_channels = [ch for ch in raw.info['ch_names'] if 'EEG' not in ch]
    raw.drop_channels(droped_channels)

    # Band-pass filter 0.1-75.0 Hz
    raw = raw.filter(l_freq=0.1, h_freq=75)
    # Notch filter at 50Hz
    raw = raw.notch_filter(50.0)
    # Convert units to uV
    data = raw.get_data(units='uV')

    if data.shape[0] != 4:
        pdb.set_trace()

    annotation_df = pd.read_csv(csv_path)

    idxx = 1
    for _, row in annotation_df.iterrows():
        start_time = row['Recording onset']
        duration = row['Duration']
        label = row['Annotation'] # e.g., 'Sleep stage W,'

        if label in used_label:
            start_point = math.ceil(start_time * fs)
            per_data = data[:, start_point : int(start_point + duration * fs)]
            per_label = label_to_number.get(label)

            save_file_path = output_path + file_name + '_' + str(idxx) + '.pkl'
            pickle.dump(
                    {"X": per_data, "Y":per_label},
                    open(save_file_path, "wb"),
                )
            print(f"{save_file_path} saved")
            idxx = idxx + 1
            file_num = file_num + 1

print(f"Total files processed: {file_num}")