import numpy as np
import os
import torch
from collections import defaultdict


def _get_dataset_root():
    """
    智能查找 dataset 目录
    支持从 system/ 目录或项目根目录运行
    """
    # 首先检查环境变量
    if "PFLLIB_DATA_DIR" in os.environ:
        return os.environ["PFLLIB_DATA_DIR"]

    # 方法1: 从当前文件位置向上查找 (适用于从 system/ 运行)
    current_file = os.path.abspath(__file__)
    system_dir = os.path.dirname(os.path.dirname(current_file))  # utils -> system
    dataset_dir_v1 = os.path.join(os.path.dirname(system_dir), 'dataset')  # system -> PFLlib -> dataset

    if os.path.exists(dataset_dir_v1):
        return dataset_dir_v1

    # 方法2: 从工作目录查找 (适用于从项目根目录运行)
    cwd = os.getcwd()
    # 情况1: cwd 是 code/，dataset 在 code/external/PFLlib/dataset/
    dataset_dir_v2 = os.path.join(cwd, 'external', 'PFLlib', 'dataset')
    if os.path.exists(dataset_dir_v2):
        return dataset_dir_v2

    # 情况2: cwd 是 system/，dataset 在 ../dataset/
    dataset_dir_v3 = os.path.join(cwd, '..', 'dataset')
    if os.path.exists(dataset_dir_v3):
        return os.path.abspath(dataset_dir_v3)

    # 如果都找不到，返回默认相对路径（保持向后兼容）
    return os.path.join('..', 'dataset')


def read_data(dataset, idx, is_train=True):
    """
    读取客户端数据，支持 pickle 和 npz 两种格式

    优先尝试读取 .pkl（大数据集，加载快），回退到 .npz（小数据集，文件小）
    """
    dataset_root = _get_dataset_root()

    if is_train:
        data_dir = os.path.join(dataset_root, dataset, 'train/')
    else:
        data_dir = os.path.join(dataset_root, dataset, 'test/')

    # 优先尝试 pickle 格式（大数据集：Camelyon17, DomainNet）
    pkl_file = os.path.join(data_dir, str(idx) + '.pkl')
    if os.path.exists(pkl_file):
        import pickle
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        return data

    # 回退到 npz 格式（小数据集：CIFAR-10/100, MNIST, Kvasir, Digit5）
    npz_file = os.path.join(data_dir, str(idx) + '.npz')
    if os.path.exists(npz_file):
        with open(npz_file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
        return data

    # 如果两种格式都不存在，报错
    raise FileNotFoundError(f"Data file not found for client {idx} in {data_dir} (tried .pkl and .npz)")


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    dataset_lower = dataset.lower()
    if "news" in dataset_lower:
        data_list = process_text(data)
    elif "shakespeare" in dataset_lower:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    # 修复性能问题：处理 list of numpy arrays（如 Camelyon17）
    # 直接 torch.Tensor(list_of_arrays) 会极慢 (PyTorch 警告)
    if isinstance(data['x'], list):
        # 使用 torch.stack 避免创建大的临时 numpy array
        # 将每个 numpy array 转为 tensor 再 stack
        X = torch.stack([torch.from_numpy(x) for x in data['x']]).type(torch.float32)
    else:
        X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

