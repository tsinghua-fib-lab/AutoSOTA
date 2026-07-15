import os
import json
import torch
import numpy as np
from PIL import Image
from PIL import ImageOps
from appfl.misc.data import Dataset

# Please download and preprocess the CELEBA data before calling this dataloader
# Reference: https://github.com/APPFL/APPFL/tree/main/examples/datasets/RawData


def get_celeba(
    num_pixel: int,
    client_id: int,
    **kwargs,
):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(cur_dir, "RawData/CELEBA")

    # test data
    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    imagedir = dir + "/raw/img_align_celeba/"
    with open(os.path.join(dir, "test/all_data_niid_05_keep_0_test_9.json")) as f:
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

    train_data_raw = {}
    with open("%s/train/all_data_niid_05_keep_0_train_9.json" % (dir)) as f:
        train_data_raw = json.load(f)

    keys = train_data_raw["users"].keys()
    keys = list(keys)
    keys.sort()
    client = keys[client_id]

    train_data_input_resize = []
    for _ in train_data_raw["user_data"][client]["x"]:
        imagepath = imagedir + imagename
        img = Image.open(imagepath)
        img = ImageOps.pad(img, [num_pixel, num_pixel])
        rgb = img.convert("RGB")
        arr = np.asarray(rgb).copy()
        arr = np.moveaxis(arr, -1, 0)
        arr = arr / 255  # scale all pixel values to between 0 and 1
        train_data_input_resize.append(arr)
    train_dataset = Dataset(
        torch.FloatTensor(train_data_input_resize),
        torch.LongTensor(train_data_raw["user_data"][client]["y"]),
    )

    return train_dataset, test_dataset
