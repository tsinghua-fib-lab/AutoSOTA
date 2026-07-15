"""
Proxy dataset for DIMAT aggregator with CoronaHack.
Uses a subset of the CoronaHack test set for activation statistics and BN reset.
"""

import os
import csv
import cv2
import torch
from appfl.misc.data import Dataset


def get_coronahack_proxy(num_samples: int = 500, num_pixel: int = 32):
    """
    Return a small proxy dataset from CoronaHack test set.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "RawData/Coronahack/archive")

    imgs_path = (
        dir + "/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
    )

    data = []
    class_name_list = []
    with open(dir + "/Chest_xray_Corona_Metadata.csv") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            if row[3] == "TEST":
                img_path = imgs_path + row[1]
                class_name = row[2] + row[4] + row[5]
                data.append([img_path, class_name])
                if class_name not in class_name_list:
                    class_name_list.append(class_name)

    class_map = {name: i for i, name in enumerate(class_name_list)}

    num_samples = min(num_samples, len(data))
    data_input = []
    data_label = []
    for idx in range(num_samples):
        img_path, class_name = data[idx]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (num_pixel, num_pixel))
        img_tensor = torch.from_numpy(img).float() / 256
        img_tensor = img_tensor.permute(2, 0, 1)
        data_input.append(img_tensor.tolist())
        data_label.append(class_map[class_name])

    return Dataset(torch.FloatTensor(data_input), torch.tensor(data_label))
