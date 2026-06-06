import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from PIL import Image


random.seed(1)
np.random.seed(1)

# 默认值，将从命令行参数覆盖
num_clients = 6  # Fed-ISIC2019 固定为 6 个中心
num_classes = 8  # 皮肤病变 8 分类
config_name = "default"

# 确定数据根目录
if "PFLLIB_DATA_DIR" in os.environ:
    data_root = os.environ["PFLLIB_DATA_DIR"]
else:
    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = script_dir

# rawdata 路径（所有配置共享）
rawdata_path = os.path.join(data_root, "FedISIC2019", "rawdata")

# 配置数据路径（将在参数解析后设置）
dir_path = None


# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get data from Hugging Face
    try:
        from datasets import load_dataset

        print(f"Loading Fed-ISIC2019 dataset from Hugging Face...")
        print(f"This may take a few minutes on first run (downloads ~3GB)...")

        # Load dataset from Hugging Face
        # Dataset URL: https://huggingface.co/datasets/flwrlabs/fed-isic2019
        dataset = load_dataset("flwrlabs/fed-isic2019", cache_dir=rawdata_path)

        # Initialize data structures
        X = [[] for _ in range(num_clients)]
        y = [[] for _ in range(num_clients)]
        statistic = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ISIC2019 图像标准尺寸
            transforms.ToTensor(),
        ])

        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Process each split (train and test)
        for split_name in ['train', 'test']:
            if split_name not in dataset:
                print(f"Warning: Split '{split_name}' not found in dataset")
                continue

            split_data_hf = dataset[split_name]
            print(f"\nProcessing {split_name} split ({len(split_data_hf)} samples)...")

            for idx, example in enumerate(split_data_hf):
                # Get image and label
                img = example['image']  # PIL Image
                label = example['label']  # integer label
                center_id = example['center']  # center ID (0-5)

                # Convert PIL image to tensor
                img_tensor = transform(img)

                # Add to corresponding center
                X[center_id].append(img_tensor.cpu().numpy())
                y[center_id].append(label)

                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(split_data_hf)} samples...")

        print(f'\nNumber of classes: {num_classes}')

        # Calculate statistics
        for i in range(num_clients):
            statistic.append([])
            y_arr = np.array(y[i])
            for yc in sorted(np.unique(y_arr)):
                statistic[-1].append((int(yc), int(sum(y_arr == yc))))

        # Print statistics
        for i in range(num_clients):
            print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
            print(f"\t\t Samples of labels: ", [item for item in statistic[i]])
            print("-" * 50)

        # Split and save
        train_data, test_data = split_data(X, y)
        # 大数据集使用 pickle 格式（加载快 3-4x）
        save_file(config_path, train_path, test_path, train_data, test_data,
            num_clients, num_classes, statistic, None, None, None, use_pickle=True)

        print(f"\n✅ Dataset generation completed successfully!")
        print(f"Data saved to: {dir_path}")

    except ImportError as e:
        print("❌ Error: Hugging Face datasets library not found!")
        print("Please install it using: uv add datasets")
        print(f"Import error details: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ Error generating dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 解析额外参数：config_name（num_clients 固定为 6）
    import sys
    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    # 设置配置特定的数据路径
    dir_path = os.path.join(data_root, "FedISIC2019", config_name) + "/"

    print(f"Configuration: {config_name}")
    print(f"Number of clients: {num_clients} (6 centers)")
    print(f"Number of classes: {num_classes} (8 skin lesion types)")
    print(f"Data path: {dir_path}")
    print(f"Shared rawdata path: {rawdata_path}")
    print("-" * 80)

    generate_dataset(dir_path)
