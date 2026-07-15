import os
import json
import numpy as np
import torch
from appfl.misc.data import Dataset
from PIL import Image
from PIL import ImageOps

# Please download and preprocess the CELEBA data before calling this dataloader
# Reference: https://github.com/APPFL/APPFL/tree/main/examples/datasets/RawData


dir = os.getcwd() + "/datasets/RawData/CELEBA"


def get_celeba(num_pixel):
    # test data for a server
    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    imagedir = dir + "/raw/img_align_celeba/"
    with open("%s/test/all_data_niid_05_keep_0_test_9.json" % (dir)) as f:
        test_data_raw = json.load(f)
    for client in test_data_raw["users"]:
        for imagename in test_data_raw["user_data"][client]["x"]:
            imagepath = imagedir + imagename
            img = Image.open(imagepath)
            img = ImageOps.pad(img, [num_pixel, num_pixel])
            rgb = img.convert("RGB")
            arr = np.asarray(rgb).copy()
            arr = np.moveaxis(arr, -1, 0)
            arr = arr / 255  # scale all pixel values to between 0 and 1
            test_data_input.append(arr)
        for data_label in test_data_raw["user_data"][client]["y"]:
            test_data_label.append(data_label)
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.LongTensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = {}
    train_datasets = []
    with open("%s/train/all_data_niid_05_keep_0_train_9.json" % (dir)) as f:
        train_data_raw = json.load(f)

    for client in train_data_raw["users"]:
        train_data_input_resize = []
        for data_input in train_data_raw["user_data"][client]["x"]:
            imagepath = imagedir + imagename
            img = Image.open(imagepath)
            img = ImageOps.pad(img, [num_pixel, num_pixel])
            rgb = img.convert("RGB")
            arr = np.asarray(rgb).copy()
            arr = np.moveaxis(arr, -1, 0)
            arr = arr / 255  # scale all pixel values to between 0 and 1
            train_data_input_resize.append(arr)
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input_resize),
                torch.LongTensor(train_data_raw["user_data"][client]["y"]),
            )
        )

    return train_datasets, test_dataset
