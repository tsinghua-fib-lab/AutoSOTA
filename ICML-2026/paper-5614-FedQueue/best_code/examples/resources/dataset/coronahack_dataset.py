import os
import csv
import cv2
import json
import torch
import numpy as np
from appfl.misc.data import Dataset


class Coronahack:
    def __init__(self, dir, pixel, is_train):
        if is_train:
            self.imgs_path = (
                dir
                + "/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"
            )
        else:
            self.imgs_path = (
                dir
                + "/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
            )

        self.data = []
        with open(dir + "/Chest_xray_Corona_Metadata.csv") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                if is_train:
                    if row[3] == "TRAIN":
                        img_path = self.imgs_path + row[1]
                        class_name = row[2] + row[4] + row[5]
                        self.data.append([img_path, class_name])
                else:
                    if row[3] == "TEST":
                        img_path = self.imgs_path + row[1]
                        class_name = row[2] + row[4] + row[5]
                        self.data.append([img_path, class_name])

        class_name_list = []
        for img_path, class_name in self.data:
            if class_name not in class_name_list:
                class_name_list.append(class_name)

        self.class_map = {}
        tmpcnt = 0
        for class_name in class_name_list:
            self.class_map[class_name] = tmpcnt
            tmpcnt += 1

        self.img_dim = (pixel, pixel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img) / 256
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, class_id


def get_corona(
    num_clients: int,
    client_id: int,
    processed: bool = True,
    num_pixel: int = 32,
    **kwargs,
):
    """
    Get the Coronahack dataset.
    :param `num_clients`: Total number of clients.
    :param `client_id`: The client id, range from 0 to `num_clients-1`.
    :param `processed`: [Default is `True`] Whether the you have preprocessed the Corona data.
    :param `num_pixel`: [Optional, only if `processed=True`] The size of the image for resizing.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if processed:
        """If you have already preprocessed the data, you can use the preprocessed data."""
        dir = os.path.join(
            cur_dir, f"PreprocessedData/Coronahack_Clients_{num_clients}"
        )

        with open(os.path.join(dir, "all_test_data.json")) as f:
            test_data_raw = json.load(f)
        test_dataset = Dataset(
            torch.FloatTensor(test_data_raw["x"]), torch.tensor(test_data_raw["y"])
        )

        with open(f"{dir}/all_train_data_client_{client_id}.json") as f:
            train_data_raw = json.load(f)
        train_dataset = Dataset(
            torch.FloatTensor(train_data_raw["x"]),
            torch.tensor(train_data_raw["y"]),
        )
    else:
        """If you have not preprocessed the data, you can use the raw data."""
        dir = os.path.join(cur_dir, "RawData/Coronahack/archive")
        test_data_raw = Coronahack(dir, num_pixel, is_train=False)

        test_data_input = []
        test_data_label = []
        for idx in range(len(test_data_raw)):
            test_data_input.append(test_data_raw[idx][0].tolist())
            test_data_label.append(test_data_raw[idx][1])

        test_dataset = Dataset(
            torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
        )

        train_data_raw = Coronahack(dir, num_pixel, is_train=True)
        split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[client_id]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_dataset = Dataset(
            torch.FloatTensor(train_data_input),
            torch.tensor(train_data_label),
        )

    return train_dataset, test_dataset
