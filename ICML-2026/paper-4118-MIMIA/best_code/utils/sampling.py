import numpy as np
from numpy import random
import numpy as np
import torch
from torch.utils.data import Subset


np.random.seed(1)

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


import numpy as np


def cifar_iid_MIA(dataset, num_users):
    """
    为联邦学习（I.I.D.数据分布）准备数据集，并为成员推理攻击（MIA）生成成员/非成员数据。
    """

    # 计算每个客户端应分到的样本数量
    num_items = int(len(dataset) / num_users)

    # dict_users 存储每个客户端的样本索引
    # all_idxs 包含所有样本的索引
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # 备份所有样本的原始索引，用于生成非成员数据集
    all_idx0 = all_idxs

    # 存储每个客户端的训练集（成员数据）和非成员数据索引
    train_idxs = []
    val_idxs = []

    # 循环为每个客户端分配数据
    for i in range(num_users):
        # 从剩余样本中随机选择，分配给当前客户端
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))

        # 将当前客户端的索引列表添加到训练集列表中
        train_idxs.append(list(dict_users[i]))

        # 从总索引池中移除已分配给当前客户端的索引
        all_idxs = list(set(all_idxs) - dict_users[i])

        # 生成非成员数据集：从原始总索引中移除当前客户端的训练集索引
        val_idxs.append(list(set(all_idx0) - dict_users[i]))

    # 返回三种类型的索引：字典格式、训练集列表、非成员集列表
    return dict_users, train_idxs, val_idxs


import numpy as np
from torch.utils.data import Dataset, Subset


def cifar_beta(dataset, beta, n_clients):
    """
    基于狄利克雷分布（beta参数）将数据集划分为非独立同分布（Non-I.I.D.）的数据集。
    每个客户端将获得类别分布不均的数据。

    Args:
        dataset: 完整的训练数据集，例如CIFAR100。
        beta: 狄利克雷分布的参数，beta值越小，数据非独立同分布程度越高。
        n_clients: 客户端（用户）的数量。

    Returns:
        一个包含三个元素的元组：
        - client_datasets: 包含每个客户端的Subset数据集对象的列表。
        - train_idxs: 包含每个客户端训练样本索引的列表（用于攻击）。
        - val_idxs: 包含每个客户端非成员样本索引的列表（用于攻击）。
    """

    print("The dataset is splited with non-iid param ", beta)

    # 1. 生成每个客户端的类别分布比例
    # 为每个类别生成一个狄利克雷分布的随机向量，该向量的维度等于客户端数量。
    # 向量中的每个元素代表该客户端在该类别中所占的样本比例。
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    # 2. 计算每个客户端、每个类别的样本数量
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    client_idx_map = {i: {} for i in range(n_clients)}  # 存储最终的索引
    client_size_map = {i: {} for i in range(n_clients)}  # 存储每个客户端、每个类别的样本数

    for y in range(len(dataset.dataset.classes)):
        # 找出当前类别（y）的所有样本索引
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)

        # 根据狄利克雷分布的比例，计算每个客户端应该分配到多少个当前类别的样本
        sample_size = (label_distributions[y] * label_y_size).astype(np.int32)

        # 修正因浮点数舍入导致的样本数总和不等于原始总数的问题
        # 将多余或不足的样本数加到最后一个客户端
        sample_size[n_clients - 1] += label_y_size - np.sum(sample_size)

        # 将计算好的样本数量存入映射中
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

    # 3. 分配样本索引
    for y in range(len(dataset.dataset.classes)):
        # 找出当前类别（y）的所有样本索引
        label_y_idx = np.where(labels == y)[0]
        # 打乱这些索引，确保随机性
        np.random.shuffle(label_y_idx)

        # 计算累积和，用于切分索引数组
        sample_interval = np.cumsum([client_size_map[i][y] for i in range(n_clients)])

        # 将当前类别的样本索引分配给每个客户端
        for i in range(n_clients):
            start_index = sample_interval[i - 1] if i > 0 else 0
            end_index = sample_interval[i]
            client_idx_map[i][y] = label_y_idx[start_index:end_index]

    # 4. 组装最终的数据集和索引
    train_idxs = []
    val_idxs = []
    client_datasets = []
    all_idxs = [i for i in range(len(dataset))]  # 所有样本的索引

    for i in range(n_clients):
        # 将客户端i的所有类别的索引合并成一个列表
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        # 再次打乱，确保客户端内的数据是随机的
        np.random.shuffle(client_i_idx)

        # 使用Subset类创建客户端i的数据子集
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)

        # 存储训练集索引（成员数据）
        train_idxs.append(client_i_idx)
        # 存储非成员数据索引（用于攻击），即所有样本减去成员数据
        val_idxs.append(list(set(all_idxs) - set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

