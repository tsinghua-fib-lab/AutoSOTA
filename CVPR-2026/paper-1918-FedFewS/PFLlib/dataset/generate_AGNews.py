"""
Script for Processing the AG News Dataset

Dataset will be automatically downloaded via Hugging Face datasets.
(Using HF datasets instead of torchtext to avoid Windows compatibility issues)

Usage:
   python generate_AGNews.py <niid/iid> <balance/unbalance> <partition> [num_clients] [config_name]

   Example:
      python generate_AGNews.py iid unbalance - 10 iid_10
      python generate_AGNews.py noniid unbalance dir 10 noniid_dir_10_a0p5

   This will create processed data at:
      <PFLLIB_DATA_DIR>/agnews/<config_name>/
"""

import numpy as np
import os
import sys
import random
from datasets import load_dataset
from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.language_utils import tokenizer_without_torchtext as tokenizer


random.seed(1)
np.random.seed(1)

# 默认值，将从命令行参数覆盖
num_clients = 10  # AG News 默认
config_name = "default"
max_len = 200
max_tokens = 32000

# 确定数据根目录
if "PFLLIB_DATA_DIR" in os.environ:
    data_root = os.environ["PFLLIB_DATA_DIR"]
else:
    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = script_dir

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

    # Get AG_News data from Hugging Face - all configs share the same cache
    print("Loading AG News dataset from Hugging Face...")
    cache_dir = os.path.join(data_root, "agnews", "rawdata", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset("ag_news", cache_dir=cache_dir)

    # Extract train and test data
    # HF ag_news format: {'text': str, 'label': int (0-3)}
    trainset = dataset['train']
    testset = dataset['test']

    print(f"Train size: {len(trainset)}, Test size: {len(testset)}")

    # Combine train and test data
    dataset_text = []
    dataset_label = []

    # Add training data
    for item in trainset:
        dataset_text.append(item['text'])
        dataset_label.append(item['label'])

    # Add test data
    for item in testset:
        dataset_text.append(item['text'])
        dataset_label.append(item['label'])

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    print(f'Total samples: {len(dataset_text)}')

    # Tokenize text (using custom tokenizer without torchtext)
    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)

    # Labels are already 0-3 (no need to subtract 1 like torchtext)
    label_list = np.array(dataset_label)

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)

    # dataset = []
    # for i in range(num_classes):
    #     idx = label_list == i
    #     dataset.append(text_list[idx])

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
            statistic, niid, balance, partition)

    print("The size of vocabulary:", len(vocab))


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
    dir_path = os.path.join(data_root, "agnews", config_name) + "/"

    print(f"Configuration: {config_name}")
    print(f"Number of clients: {num_clients}")
    print(f"Data path: {dir_path}")
    print(f"Max sequence length: {max_len}")
    print(f"Max vocabulary tokens: {max_tokens}")
    print("-" * 80)

    generate_dataset(dir_path, num_clients, niid, balance, partition)