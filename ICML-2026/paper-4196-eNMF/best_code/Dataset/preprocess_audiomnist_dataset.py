import h5py
import numpy as np
import os


# Create a data matrix each column is a 8000 audio vector.
def read_file_lists(train_fn):
    with open(train_fn) as file:
        lines = [line.rstrip() for line in file]
    print(f"Read in {len(lines)} lines!")
    return lines


def fetch_audio_vector_label(fn):
    f1 = h5py.File(fn, "r")
    return f1["data"][0, 0].T, np.array(f1["label"]).T


if __name__ == "__main__":
    project_dir = os.getcwd()
    train_fn = os.path.join(
        project_dir,
        "Dataset",
        "AudioMNIST/preprocessed_data/AudioNet_digit_0_train.txt",
    )
    filename_list = read_file_lists(train_fn)
    data = []
    labels = []
    for fn in filename_list:
        cur_data, cur_label = fetch_audio_vector_label(fn)
        data.append(cur_data)
        labels.append(cur_label)

    final_data = np.concatenate(data, axis=-1).T
    # First dim for digital number, second dim (0 from male,  1 from female)
    final_labels = np.concatenate(labels, axis=-1).T
    print(final_data.shape)
    print(final_labels.shape)
    print("min value of current data:", np.min(final_data))
    final_data = final_data - np.min(final_data)
    np.savez("Dataset/audiomnist.npy", data=final_data, label=final_labels)
