"""
Script for Processing the FEMNIST Dataset

This script automatically downloads EMNIST dataset using torchvision and
reorganizes it by writer (similar to FEMNIST from LEAF project).

No manual download required! The script will:
   1. Auto-download EMNIST 'byclass' split from torchvision
   2. Reorganize images by writer ID
   3. Generate images_by_writer.pkl for federated learning

Usage:
   python generate_femnist.py <niid/iid> <balance/unbalance> <partition> [num_clients] [config_name]

   Example:
      python generate_femnist.py noniid balance pat 20 noniid_pat_20
      python generate_femnist.py noniid unbalance - 50 natural_noniid_50

   This will create processed data at:
      <PFLLIB_DATA_DIR>/femnist/<config_name>/

Note: First run will download ~300MB EMNIST data to rawdata/downloads/
"""

import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.dataset_utils import check, separate_data, split_data, save_file
import pickle
from tqdm import tqdm


random.seed(1)
np.random.seed(1)

# 默认值，将从命令行参数覆盖
num_clients = 100  # FEMNIST 默认
config_name = "default"
meta_file_name = 'images_by_writer.pkl'

# 确定数据根目录
if "PFLLIB_DATA_DIR" in os.environ:
    data_root = os.environ["PFLLIB_DATA_DIR"]
else:
    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = script_dir

# rawdata 路径（所有配置共享，存放从 LEAF 下载的原始数据）
rawdata_path = os.path.join(data_root, "femnist", "rawdata")

# 配置数据路径（将在参数解析后设置）
dir_path = None 

def relabel(c):
    """
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    """

    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61  
    
class FemnistDataset(Dataset):
    def __init__(self, data, transform):
        # super(FemnistNiidDataset, self).__init__()

        self.data = data
        self.transforms = transform

        self.transform = transform
        self.images = self.get_images()
        self.targets = self.get_targets()

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def get_images(self):
        pixel_values = []
        for i in range(len(self.data)):
            # 支持两种数据格式：
            # 1. (file_path, label) - 从文件加载（原始FEMNIST格式）
            # 2. (numpy_array, label) - 直接使用数组（EMNIST自动下载格式）
            data_item = self.data[i][0]

            if isinstance(data_item, str):
                # 格式1：文件路径
                path = os.path.join(rawdata_path, data_item)
                image = Image.open(path).convert("L")
            elif isinstance(data_item, np.ndarray):
                # 格式2：numpy数组
                image = Image.fromarray(data_item, mode='L')
            else:
                raise ValueError(f"Unsupported data format: {type(data_item)}")

            pixel_value = self.transform(image)
            pixel_values.append(pixel_value)
        return pixel_values

    def get_targets(self):
        labels = []
        for i in range(len(self.data)):
            label = torch.tensor(relabel(self.data[i][1]))
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.data)
    
 
def download_and_prepare_emnist(rawdata_path):
    """
    自动下载 EMNIST 数据集并重组为 FEMNIST 格式（按 writer 分组）
    优化：直接在 pickle 中存储图像数组，避免保存 80 万个小文件
    """
    print("=" * 80)
    print("首次运行：自动下载并预处理 EMNIST 数据集")
    print("这可能需要几分钟时间（下载 ~300MB + 重组数据）...")
    print("=" * 80)

    downloads_path = os.path.join(rawdata_path, "downloads")
    intermediate_path = os.path.join(rawdata_path, "intermediate")

    os.makedirs(downloads_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)

    # 1. 下载 EMNIST byclass 数据集
    print("\n[1/3] 下载 EMNIST 'byclass' 数据集...")
    train_dataset = torchvision.datasets.EMNIST(
        root=downloads_path,
        split='byclass',
        train=True,
        download=True
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=downloads_path,
        split='byclass',
        train=False,
        download=True
    )

    # 2. 合并训练和测试数据
    print("\n[2/3] 重组数据（按 writer 分组）...")
    all_images = torch.cat([train_dataset.data, test_dataset.data])
    all_labels = torch.cat([train_dataset.targets, test_dataset.targets])

    # EMNIST byclass 有 62 个类别（0-9数字 + 26大写 + 26小写）
    # 我们模拟 writer ID（因为 EMNIST 不直接提供 writer 信息）
    # 策略：将相同类别的样本分散到多个 "虚拟 writer"
    num_classes = 62
    samples_per_writer = 500  # 每个 writer 平均样本数

    # 按类别组织数据
    class_indices = {c: (all_labels == c).nonzero(as_tuple=True)[0] for c in range(num_classes)}

    # 为每个类别创建多个 writer
    images_by_writer = []
    writer_id = 0

    print("正在重组数据（直接存储图像数组，避免保存大量小文件）...")
    for class_id in tqdm(range(num_classes), desc="处理类别"):
        indices = class_indices[class_id]
        num_samples = len(indices)
        num_writers_for_class = max(1, num_samples // samples_per_writer)

        # 随机打乱并分配给不同 writer
        perm = torch.randperm(num_samples)
        split_indices = torch.chunk(perm, num_writers_for_class)

        for chunk in split_indices:
            writer_data = []
            for idx in chunk:
                global_idx = indices[idx].item()
                img = all_images[global_idx]
                label = all_labels[global_idx].item()

                # EMNIST 图像需要转置和镜像
                img_array = img.numpy().T  # 转置
                img_array = np.fliplr(img_array)  # 水平翻转

                # 存储标签（转换为16进制格式，兼容原始 FEMNIST）
                # label 0-9 -> '30'-'39' (数字)
                # label 10-35 -> '41'-'5a' (大写字母)
                # label 36-61 -> '61'-'7a' (小写字母)
                if label < 10:
                    label_hex = format(label + 30, 'x')
                elif label < 36:
                    label_hex = format(label + 55, 'x')
                else:
                    label_hex = format(label + 61, 'x')

                # 直接存储图像数组，而不是文件路径（格式：(numpy_array, label_hex)）
                writer_data.append((img_array, label_hex))

            if writer_data:
                images_by_writer.append((writer_id, writer_data))
                writer_id += 1

    # 3. 保存 images_by_writer.pkl
    print(f"\n[3/3] 保存元数据文件...")
    meta_path = os.path.join(intermediate_path, meta_file_name)
    with open(meta_path, 'wb') as f:
        pickle.dump(images_by_writer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ 预处理完成！")
    print(f"  - 总 writer 数量: {len(images_by_writer)}")
    print(f"  - 总样本数量: {len(all_images)}")
    print(f"  - 元数据保存至: {meta_path}")
    print(f"  - 注意：图像数据直接存储在 pickle 中（无需单独图片文件）")
    print("=" * 80)

    return meta_path


def get_writer_id(data, num_clients):
        images_per_writer = [(row[0],len(row[1])) for row in data]
        images_per_writer.sort(key = lambda x : x[1], reverse=True)

        writers = images_per_writer[:num_clients]
        user_ids = [w_id for w_id, count in writers]

        return user_ids
    
# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    num_classes = 62

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get FEMNIST data
    meta_path = os.path.join(rawdata_path, 'intermediate', meta_file_name)

    # 如果 rawdata 不存在，自动下载并预处理
    if not os.path.exists(meta_path):
        print(f"未找到预处理数据: {meta_path}")
        print("开始自动下载并预处理 EMNIST 数据集...")
        meta_path = download_and_prepare_emnist(rawdata_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    data = pd.read_pickle(meta_path)
    writer_ids = get_writer_id(data, num_clients)
    
 
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    for i in range(len(writer_ids)):
        
        user_data = [row[1] for row in data if row[0] == writer_ids[i]][0]
        dataset = FemnistDataset(data = user_data, transform = transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset.data), shuffle=False)
  
        for _, train_data in enumerate(dataloader, 0):
            dataset.images, dataset.targets = train_data

        # FEMNIST 保持单通道（灰度图），shape: [N, 1, 28, 28]
        X[i] = dataset.images.cpu().detach().numpy()
        y[i] = dataset.targets.cpu().detach().numpy()
        assert len(X[i]) == len(y[i]) , 'the length of images must be equal to the length of targets '
        statistic[i] = (int(i), len(X[i]))
        
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # 解析额外参数：num_clients 和 config_name
    if len(sys.argv) > 4:
        num_clients = int(sys.argv[4])
    if len(sys.argv) > 5:
        config_name = sys.argv[5]

    # 设置配置特定的数据路径
    dir_path = os.path.join(data_root, "femnist", config_name) + "/"

    print(f"Configuration: {config_name}")
    print(f"Number of clients: {num_clients}")
    print(f"Data path: {dir_path}")
    print(f"Shared rawdata path: {rawdata_path}")
    print(f"Meta file: {meta_file_name}")
    print("-" * 80)

    generate_dataset(dir_path, num_clients, niid, balance, partition)