import json
import os
import pickle
import numpy as np
from natsort import natsorted
from collections import defaultdict
import random

# Set paths
data_folder = "path/to/Siena_Scalp_EEG_Database/processed_segments"
save_folder_train = 'path/to/Siena_Scalp_EEG_Database/cross_subject_json/train.json'
save_folder_val = 'path/to/Siena_Scalp_EEG_Database/cross_subject_json/val.json'
save_folder_test = 'path/to/Siena_Scalp_EEG_Database/cross_subject_json/test.json'

# Base parameters
sampling_rate = 512
ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2',
            'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10']
num_channels = len(ch_names)
random.seed(42) # Fix random seed

def load_subject_data(subject_folder):
    """Loads all data for a single subject."""
    subject_data = []
    subject_num = int(os.path.basename(subject_folder)[2:])
    subject_name = f"PN{subject_num:02d}"

    for file in natsorted(f for f in os.listdir(subject_folder) if f.endswith('.pkl')):
        try:
            with open(os.path.join(subject_folder, file), 'rb') as f:
                eeg_data = pickle.load(f)
            subject_data.append({
                "subject_id": subject_num,
                "subject_name": subject_name,
                "file": os.path.join(subject_folder, file),
                "label": eeg_data['Y'],
                "eeg_data": eeg_data['X'] # Cache data
            })
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    return subject_data

def split_subject_data(subject_data, val_ratio=0.2):
    """Splits a single subject's data into validation and training sets, balanced by class."""
    label_to_data = defaultdict(list)
    for data in subject_data:
        label_to_data[data["label"]].append(data)

    train_data, val_data = [], []
    for label, data_list in label_to_data.items():
        random.shuffle(data_list)
        split_idx = int(len(data_list) * (1 - val_ratio))
        train_data.extend(data_list[:split_idx])
        val_data.extend(data_list[split_idx:])

    return train_data, val_data

def compute_normalization_params(data_list):
    """Computes normalization parameters (mean, std, min, max) from a list of data."""
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    max_val, min_val = -np.inf, np.inf

    for data in data_list:
        eeg = data["eeg_data"]
        max_val = max(max_val, eeg.max())
        min_val = min(min_val, eeg.min())
        for j in range(num_channels):
            total_mean[j] += eeg[j].mean()
            total_std[j] += eeg[j].std()

    mean = (total_mean / len(data_list)).tolist()
    std = (total_std / len(data_list)).tolist()
    return mean, std, max_val, min_val

def save_dataset(data_list, save_path, norm_params=None):
    """Saves the dataset to a JSON file."""
    if norm_params is None:
        mean, std, max_val, min_val = compute_normalization_params(data_list)
    else:
        mean, std, max_val, min_val = norm_params

    dataset = {
        "subject_data": [{k: v for k, v in d.items() if k != "eeg_data"} for d in data_list],
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std
        }
    }

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    # Get all subject folders
    subject_folders = natsorted(
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.startswith("PN") and os.path.isdir(os.path.join(data_folder, f))
    )

    # Split subjects into training and testing sets
    train_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) <= 12]
    test_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) > 12]

    # Process training subjects (split validation set within each subject)
    all_train_data, all_val_data = [], []
    for subject in train_subjects:
        subject_data = load_subject_data(subject)
        train_data, val_data = split_subject_data(subject_data)
        all_train_data.extend(train_data)
        all_val_data.extend(val_data)

    # Process test subjects
    all_test_data = []
    for subject in test_subjects:
        all_test_data.extend(load_subject_data(subject))

    # Compute normalization parameters from the training set
    norm_params = compute_normalization_params(all_train_data)
    print(f"Total training samples: {len(all_train_data)}")
    print(f"Total validation samples: {len(all_val_data)}")
    print(f"Total test samples: {len(all_test_data)}")

    # Save datasets
    save_dataset(all_train_data, save_folder_train)
    save_dataset(all_val_data, save_folder_val, norm_params)
    save_dataset(all_test_data, save_folder_test, norm_params)

if __name__ == "__main__":
    main()