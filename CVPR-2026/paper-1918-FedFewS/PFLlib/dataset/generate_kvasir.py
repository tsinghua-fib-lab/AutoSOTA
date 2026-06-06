import sys
import pandas as pd
import numpy as np
import os
import random
import torchvision.transforms as transforms
from sklearn.utils import resample
from sklearn.utils import shuffle
from utils.dataset_utils import check, separate_data, split_data, save_file, ImageDataset
from torch.utils.data import DataLoader


random.seed(1)
np.random.seed(1)

# 默认值，将从命令行参数覆盖
num_clients = 50  # Kvasir 默认
config_name = "default"
img_size = 64

# 确定数据根目录
if "PFLLIB_DATA_DIR" in os.environ:
    data_root = os.environ["PFLLIB_DATA_DIR"]
else:
    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = script_dir

# rawdata 路径（所有配置共享）
rawdata_path = os.path.join(data_root, "kvasir", "rawdata")

# 配置数据路径（将在参数解析后设置）
dir_path = None

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get data - auto download if not exists
    data_dir = os.path.join(rawdata_path, 'kvasir-dataset-v2/')
    if not os.path.exists(data_dir):
        print(f"Kvasir rawdata not found at {rawdata_path}, downloading...")
        os.makedirs(rawdata_path, exist_ok=True)

        # Download from official source
        url = 'https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip'
        zip_path = os.path.join(rawdata_path, 'kvasir-dataset-v2.zip')

        print(f"Downloading Kvasir dataset from {url}...")
        print("This may take a few minutes (approx. 2.6 GB)...")

        # Use curl for download (cross-platform)
        import subprocess
        result = subprocess.run(['curl', '-L', '-o', zip_path, url],
                              capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Kvasir dataset: {result.stderr}")

        print(f"Download complete. Extracting to {rawdata_path}...")

        # Unzip the file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)

        print(f"Extraction complete. Dataset ready at {data_dir}")

        # Clean up zip file
        os.remove(zip_path)
        print("Cleaned up zip file.")

    class_names = os.listdir(data_dir)
    print('All class names:', class_names)
    num_classes = len(class_names)

    file_names = []
    labels = []
    for dir in os.listdir(data_dir):
        if dir in class_names:
            label = class_names.index(dir)
            for file_name in os.listdir(os.path.join(data_dir, dir)):
                file_names.append(os.path.join(dir, file_name))
                labels.append(label)
    df = pd.DataFrame({'file_name': file_names, 'class': labels})
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    dataset = ImageDataset(df, data_dir, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=False, 
    )
    x, y = next(iter(dataloader))
    dataset_image = x.numpy()
    dataset_label = y.numpy()
    
    print('Total data amount', len(dataset_image), len(dataset_label))

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
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
    dir_path = os.path.join(data_root, "kvasir", config_name) + "/"

    print(f"Configuration: {config_name}")
    print(f"Number of clients: {num_clients}")
    print(f"Data path: {dir_path}")
    print(f"Shared rawdata path: {rawdata_path}")
    print("-" * 80)

    generate_dataset(dir_path, num_clients, niid, balance, partition)