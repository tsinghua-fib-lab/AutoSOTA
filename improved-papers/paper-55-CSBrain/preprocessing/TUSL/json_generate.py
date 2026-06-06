import os
import json
import random
import pickle
import numpy as np
from natsort import natsorted

# Root directory
base_folder = "path/to/tuh_eeg_slowing/v2.0.1/processed_data"

# Output paths
save_folder_train = 'path/to/tuh_eeg_slowing/v2.0.1/cross_subject_split/train.json'
save_folder_test = 'path/to/tuh_eeg_slowing/v2.0.1/cross_subject_split/test.json'
save_folder_val = 'path/to/tuh_eeg_slowing/v2.0.1/cross_subject_split/val.json'

# Dataset basic information
sampling_rate = 250
ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

# Get all folders
folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
random.shuffle(folders) # Randomly shuffle folder order

# Allocate folders to training, validation, and test sets
total_folders = len(folders)
num_train = int(total_folders * 0.6)
num_val = int(total_folders * 0.2)
train_folders = folders[:num_train]
val_folders = folders[num_train:num_train+num_val]
test_folders = folders[num_train+num_val:]

# Store file information
tuples_list_train = []
tuples_list_test = []
tuples_list_val = []
error_list = []

# Map file_id to subject_id
file_id_to_subject_id = {}
subject_id_counter = 0 # Initialize subject_id from 0

# Process each folder
for folder in train_folders + val_folders + test_folders:
    folder_path = os.path.join(base_folder, folder)
    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    
    pkl_files = natsorted(pkl_files)

    # Iterate through each pkl file
    for file in pkl_files:
        try:
            data = pickle.load(open(file, "rb"))
            label = data['Y']
            eeg_data = data['X']
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            error_list.append(file)
            continue
        
        file_id = folder # Folder name as file_id
        
        # Assign a new subject_id if file_id hasn't been assigned one
        if file_id not in file_id_to_subject_id:
            file_id_to_subject_id[file_id] = subject_id_counter
            subject_id_counter += 1
        
        subject_id = file_id_to_subject_id[file_id]

        num_channels = eeg_data.shape[0]

        # Resize total_mean and total_std arrays if necessary
        total_mean = np.resize(total_mean, num_channels)
        total_std = np.resize(total_std, num_channels)

        # Calculate mean and standard deviation
        for j in range(num_channels):
            total_mean[j] += eeg_data[j].mean()
            total_std[j] += eeg_data[j].std()

            # Calculate max and min values, avoiding NaN
            per_max_value = np.nanmax(eeg_data[j])
            per_min_value = np.nanmin(eeg_data[j])

            if per_max_value > max_value:
                max_value = per_max_value

            if per_min_value < min_value:
                min_value = per_min_value

        num_all += 1

        data_entry = {
            "subject_id": subject_id,
            "subject_name": file_id,
            "file": file,
            "label": label
        }

        # Assign file to train, val, or test set
        if folder in train_folders:
            tuples_list_train.append(data_entry)
        elif folder in val_folders:
            tuples_list_val.append(data_entry)
        else:
            tuples_list_test.append(data_entry)

# Calculate overall mean and standard deviation
data_mean = (total_mean / num_all).tolist()
data_std = (total_std / num_all).tolist()

# Build dataset_info
dataset_info = {
    "sampling_rate": sampling_rate,
    "ch_names": ch_names,
    "min": min_value,
    "max": max_value,
    "mean": data_mean,
    "std": data_std
}

# Build final JSON data structure
train_dataset = {
    "subject_data": tuples_list_train,
    "dataset_info": dataset_info
}

val_dataset = {
    "subject_data": tuples_list_val,
    "dataset_info": dataset_info
}

test_dataset = {
    "subject_data": tuples_list_test,
    "dataset_info": dataset_info
}

# Save as JSON files
formatted_json_train = json.dumps(train_dataset, indent=2)
with open(save_folder_train, 'w') as f:
    f.write(formatted_json_train)

formatted_json_val = json.dumps(val_dataset, indent=2)
with open(save_folder_val, 'w') as f:
    f.write(formatted_json_val)

formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_folder_test, 'w') as f:
    f.write(formatted_json_test)

# Print error list
print("Error list: ", error_list)