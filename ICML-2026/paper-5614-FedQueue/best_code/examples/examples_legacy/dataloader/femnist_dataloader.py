import os
import json
import numpy as np
import torch
from appfl.misc.data import Dataset

# Please download and preprocess the FEMNIST data before calling this dataloader
# Reference: https://github.com/APPFL/APPFL/tree/main/examples/datasets/RawData


def get_femnist(num_pixel, num_channel, pretrained):
    # test data for a server
    dir = os.getcwd() + "/datasets/RawData/FEMNIST"
    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    for idx in range(36):
        with open(f"{dir}/test/all_data_{idx}_niid_05_keep_0_test_9.json") as f:
            test_data_raw[idx] = json.load(f)
        for client in test_data_raw[idx]["users"]:
            for data_input in test_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(num_pixel, num_pixel)
                # Repeating 1 channel data to use pretrained weight that based on 3 channels data
                if num_channel == 1 and pretrained > 0:
                    test_data_input.append([data_input, data_input, data_input])
                else:
                    test_data_input.append([data_input])

            for data_label in test_data_raw[idx]["user_data"][client]["y"]:
                test_data_label.append(data_label)
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = {}
    train_datasets = []
    for idx in range(36):
        with open(f"{dir}/train/all_data_{idx}_niid_05_keep_0_train_9.json") as f:
            train_data_raw[idx] = json.load(f)

        for client in train_data_raw[idx]["users"]:
            train_data_input_resize = []
            for data_input in train_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(num_pixel, num_pixel)
                # Repeating 1 channel data to use pretrained weight that based on 3 channels data
                if num_channel == 1 and pretrained > 0:
                    train_data_input_resize.append([data_input, data_input, data_input])
                else:
                    train_data_input_resize.append([data_input])

            train_datasets.append(
                Dataset(
                    torch.FloatTensor(train_data_input_resize),
                    torch.tensor(train_data_raw[idx]["user_data"][client]["y"]),
                )
            )

    return train_datasets, test_dataset
