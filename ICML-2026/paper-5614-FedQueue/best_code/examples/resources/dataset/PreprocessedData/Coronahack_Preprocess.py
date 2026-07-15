import os
import csv
import cv2
import json
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_pixel", type=int, default=32, help="size of the image for resizing"
)
parser.add_argument(
    "--num_clients",
    type=int,
    default=4,
    help="how many client chunks should be split into from the original dataset",
)
args = parser.parse_args()


class CoronahackTrain:
    def __init__(self, pixel):
        dir = "../RawData/Coronahack/archive"

        self.imgs_path = (
            dir + "/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"
        )
        self.data = []
        with open(dir + "/Chest_xray_Corona_Metadata.csv") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                if row[3] == "TRAIN":
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


class CoronahackTest:
    def __init__(self, pixel):
        dir = "../RawData/Coronahack/archive"
        self.imgs_path = (
            dir + "/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
        )
        self.data = []
        with open(dir + "/Chest_xray_Corona_Metadata.csv") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
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


train_data_raw = CoronahackTrain(args.num_pixel)
test_data_raw = CoronahackTest(args.num_pixel)

## Output Directories
dir = f"./Coronahack_Clients_{args.num_clients}"
if not os.path.isdir(dir):
    os.mkdir(dir)

## Testing Data (Server)
out_test_file = dir + "/all_test_data.json"
all_test_data = {}
all_test_data["x"] = []
all_test_data["y"] = []
for idx in range(len(test_data_raw)):
    all_test_data["x"].append(test_data_raw[idx][0].tolist())
    all_test_data["y"].append(test_data_raw[idx][1])

with open(out_test_file, "w") as outfile:
    json.dump(all_test_data, outfile)

## Training Data (Clients)
split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
for i in range(args.num_clients):
    print("--client ", i)

    user_data = {}
    x = []
    y = []
    for idx in split_train_data_raw[i]:
        x.append(train_data_raw[idx][0].tolist())
        y.append(train_data_raw[idx][1])
    user_data["x"] = x
    user_data["y"] = y

    out_train_file = dir + f"/all_train_data_client_{i}.json"

    with open(out_train_file, "w") as outfile:
        json.dump(user_data, outfile)
