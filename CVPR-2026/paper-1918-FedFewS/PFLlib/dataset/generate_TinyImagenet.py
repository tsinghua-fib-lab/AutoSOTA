import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
import urllib.request
import zipfile

random.seed(1)
np.random.seed(1)

# 默认值，将从命令行参数覆盖
num_clients = 20
config_name = "default"

# 确定数据根目录
if "PFLLIB_DATA_DIR" in os.environ:
    data_root = os.environ["PFLLIB_DATA_DIR"]
else:
    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = script_dir

# rawdata 路径（所有配置共享）
rawdata_path = os.path.join(data_root, "TinyImagenet", "rawdata")

# 配置数据路径（将在参数解析后设置）
dir_path = None


def download_and_extract_tinyimagenet(rawdata_path):
    """
    下载并解压 TinyImageNet 数据集
    兼容 Windows (不依赖 wget)
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(rawdata_path, "tiny-imagenet-200.zip")
    extract_path = rawdata_path

    # 检查是否已存在
    if os.path.exists(os.path.join(rawdata_path, "tiny-imagenet-200")):
        print("TinyImageNet dataset already exists.")
        return True

    print(f"Downloading TinyImageNet from {url}...")
    print(f"This may take a while (237 MB)...")

    try:
        # 下载文件（带进度条）
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}% ({count * block_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\n✅ Download complete!")

        # 解压文件
        print(f"Extracting to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete!")

        # 删除 zip 文件
        os.remove(zip_path)
        print("✅ Cleaned up zip file")

        return True

    except Exception as e:
        print(f"\n❌ Error downloading/extracting TinyImageNet: {e}")
        print("Please manually download from http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print(f"And extract to {rawdata_path}")
        return False


# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# 重组 TinyImageNet 验证集目录结构
def reorganize_val_folder(val_dir):
    """
    TinyImageNet 的验证集图片都在 val/images/ 目录下，
    需要根据 val_annotations.txt 重组为类别目录结构
    """
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.exists(val_annotations_file):
        print("Warning: val_annotations.txt not found, skipping validation set")
        return False

    # 检查是否已经重组过
    if not os.path.exists(val_img_dir):
        print("Validation set already reorganized")
        return True

    # 读取 annotations 文件
    with open(val_annotations_file, 'r') as f:
        val_annotations = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                val_annotations[img_name] = class_id

    # 为每个类别创建目录并移动图片
    print("Reorganizing validation set...")
    for img_name, class_id in val_annotations.items():
        # 创建类别目录
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        # 移动图片
        src = os.path.join(val_img_dir, img_name)
        dst = os.path.join(class_dir, img_name)
        if os.path.exists(src):
            os.rename(src, dst)

    # 删除空的 images 目录
    if os.path.exists(val_img_dir) and not os.listdir(val_img_dir):
        os.rmdir(val_img_dir)

    print("Validation set reorganized successfully")
    return True


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

    # Get data (共享 rawdata 目录)
    os.makedirs(rawdata_path, exist_ok=True)
    if not download_and_extract_tinyimagenet(rawdata_path):
        raise RuntimeError("Failed to download TinyImageNet dataset")

    # 重组验证集目录
    val_dir = os.path.join(rawdata_path, 'tiny-imagenet-200/val/')
    reorganize_val_folder(val_dir)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载训练集
    trainset = ImageFolder_custom(root=os.path.join(rawdata_path, 'tiny-imagenet-200/train/'), transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data

    # 加载验证集
    valset = ImageFolder_custom(root=val_dir, transform=transform)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=len(valset), shuffle=False)

    for _, val_data in enumerate(valloader, 0):
        valset.data, valset.targets = val_data

    dataset_image = []
    dataset_label = []

    # 合并训练集和验证集（与 CIFAR-10 保持一致）
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(valset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(valset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition, use_pickle=True)


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
    dir_path = os.path.join(data_root, "TinyImagenet", config_name) + "/"

    print(f"Configuration: {config_name}")
    print(f"Number of clients: {num_clients}")
    print(f"Data path: {dir_path}")
    print(f"Shared rawdata path: {rawdata_path}")
    print("-" * 80)

    generate_dataset(dir_path, num_clients, niid, balance, partition)