
import random

import torchvision
from PIL import Image
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
import csv
import os
import pickle
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import torch
from torch.utils.data import ConcatDataset, random_split
import json
from torch.utils.data import Dataset, Subset

np.random.seed(1)



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


import pandas as pd
import numpy as np
import torch


def cifar_iid_MIA_by_ratio(train_set_mia, num_users,
                           excel_path=r'E:\machang\code\FedMIA-main\FedMIA-main\cremad_final_data_with_ratio.xlsx'):
    """
    重写逻辑：
    1. 获取 train_set 的原始索引映射。
    2. 读取 Excel 并计算 ratio。
    3. 将 Excel 中的全局索引转为相对于 train_set 的本地索引 (0, 1, 2...)。
    4. 按 ratio 排序后分配。
    """
    # 1. 读取 Excel
    df = pd.read_excel(excel_path)
    if 'ratio' not in df.columns:
        df['ratio'] = df['video_scored'] / df['audio_score']

    # 2. 获取 train_set (90%子集) 在原始数据集中的绝对位置
    # train_set.indices 是一个列表，包含了这 90% 数据在原图中的 ID
    if hasattr(train_set_mia, 'indices'):
        abs_indices = list(train_set_mia.indices)
    else:
        abs_indices = list(range(len(train_set_mia)))

    # --- 关键步骤：建立【绝对索引】到【相对索引】的映射表 ---
    # 比如：abs_indices 是 [10, 25, 30]，那么映射就是 {10:0, 25:1, 30:2}
    abs_to_rel_map = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(abs_indices)}

    # 3. 筛选 Excel：只保留属于训练集的行
    df_filtered = df[df['index'].isin(abs_indices)].copy()

    # 4. 按照 ratio 排序 (ascending=True 是从小到大，False 是从大到小)
    df_sorted = df_filtered.sort_values(by='ratio', ascending=True)

    # 5. 获取排序后的【相对索引】
    # 我们不直接用 Excel 的 index，而是通过映射表转成 train_set 认识的 0, 1, 2...
    sorted_rel_idxs = [abs_to_rel_map[abs_idx] for abs_idx in df_sorted['index'].tolist()]

    # 6. 开始分配
    num_items = int(len(sorted_rel_idxs) / num_users)
    dict_users = {}
    train_idxs = []
    val_idxs = []

    all_rel_idx_set = set(sorted_rel_idxs)

    for i in range(num_users):
        start_ptr = i * num_items
        # 最后一个用户拿走剩余所有
        if i == num_users - 1:
            user_data_list = sorted_rel_idxs[start_ptr:]
        else:
            user_data_list = sorted_rel_idxs[start_ptr:start_ptr + num_items]

        # 存储为 list，确保 DataLoader 能读取
        dict_users[i] = list(user_data_list)
        train_idxs.append(list(user_data_list))

        # 生成非成员索引 (在本训练集内除了自己的其他索引)
        current_user_set = set(user_data_list)
        non_member_list = list(all_rel_idx_set - current_user_set)
        val_idxs.append(non_member_list)

    # 验证分配
    print(f">>> 数据映射完成。训练集大小: {len(abs_indices)}")
    print(f">>> 用户 0 (低Ratio) 包含样本数: {len(train_idxs[0])}")

    return dict_users, train_idxs, val_idxs



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






































def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class Balanced(Dataset):
    def __init__(
            self,
            args=None,
            test_data_transform=None,
            test_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
    ):
        super().__init__()
        self.image = []
        self.audio = []
        self.label = []
        classes = []
        self.name = []
        activities = ['playing piano', 'playing violin', 'playing flute', 'playing guitar', 'playing cello', 'singing',
                      'finger snapping', 'hair cutting', 'sneezing', 'whistling', 'shaving', 'hair drying', 'bowling',
                      'chopping wood', 'cutting pineapple', 'beat boxing', 'swimming', 'cheering',
                      'writing on blackboard', 'marching', 'eating', 'motorcycling', 'clapping', 'lawn mowing',
                      'cleaning floor', 'shot football', 'crying', 'laughing', 'tractor digging', 'gargling']
        self.data_root = './data/'
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}
        activity_labels = {activity: i for i, activity in enumerate(activities)}

        self.visual_feature_path = r'/data1/workspace_01/machang/dataset/balanced'
        self.audio_feature_path = r'/data1/workspace_01/machang/dataset/balanced/audio'

        # Convert the labels to one-hot encoding

        with open('/data1/workspace_01/machang/dataset/balanced/Imbalance_train_test_val_final.json', 'r',
                  encoding='utf-8') as json_file:
            self.data = json.load(json_file)

        # 遍历 JSON 文件的键值对
        for key, value in self.data.items():
            self.name.append(key)
            audio_path = os.path.join(self.audio_feature_path, key + '.pkl')
            self.audio.append(audio_path)
            visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), key)
            self.label.append(activity_labels[value['label']])
            self.image.append(visual_path)

        self.targets = torch.Tensor(self.label)
        self.classes = set(self.label)
        self.test_data_transform = test_data_transform
        self.test_target_transform = test_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.targets)

    def __getitem__(self, index):
        spectrogram = pickle.load(open(self.audio[index], 'rb'))
        image_samples = os.listdir(self.image[index])
        images = torch.zeros((1, 3, 224, 224))
        img = Image.open(os.path.join(self.image[index], image_samples[0])).convert('RGB')
        targets = torch.Tensor(self.targets[index]).long()
        train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4406, 0.3998, 0.3761], [0.2862, 0.2763, 0.2757])
        ])

        images[0] = train_data_transform(img)
        images = torch.permute(images, (1, 0, 2, 3)).float()
        spectrogram = torch.from_numpy(spectrogram).float()

        return index,spectrogram, images, targets


class AVDataset(Dataset):

    def __init__(self, mode='train'):

        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = r'E:\machang\code\dataset\AVE\AVE_Dataset'
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = r'E:\machang\code\dataset\AVE\AVE_Dataset'
        self.audio_feature_path = r'E:\machang\code\dataset\AVE\AVE_Dataset\Audio-1004-SE'

        self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        self.test_txt = os.path.join(self.data_root, 'testSet.txt')
        self.val_txt = os.path.join(self.data_root, 'valSet.txt')

        txt_files_to_load = [self.train_txt, self.val_txt, self.test_txt]
        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        for txt_file in txt_files_to_load:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    item = line.strip().split('&')
                    if len(item) < 2: continue

                    # 构建路径
                    # item[0]: 类别, item[1]: 文件名ID, item[2]: 时间段(如果有)
                    audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(1), item[1])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        # 简单的去重（可选，防止同一个文件在不同txt里出现）
                        if audio_path not in self.audio:
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[0]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # # audio
        # samples, rate = librosa.load(self.audio[idx], sr=22050)
        # resamples = np.tile(samples, 3)[:22050*3]
        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.
        #
        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # #mean = np.mean(spectrogram)
        # #std = np.std(spectrogram)
        # #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        # select_index.sort()
        images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return idx,spectrogram, images, label
class CramedDataset(Dataset):

    def __init__(self, mode='train', data_root=None):
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        if data_root is None:
            data_root = os.environ.get('CREMAD_DATA_ROOT', '/datasets/CREMA-D')
        self.data_root = data_root
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = data_root
        self.audio_feature_path = os.path.join(data_root, 'AudioWAV')

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        csv_files = [self.train_csv, self.test_csv]
        for file_path in csv_files:
            with open(file_path, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)
                for item in csv_reader:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), item[0])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=1, replace=False)
        select_index.sort()
        images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return idx,spectrogram, images, label
class CramedDataset3(Dataset):

    def __init__(self, mode='train', data_root=None):
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        if data_root is None:
            data_root = os.environ.get('CREMAD_DATA_ROOT', '/datasets/CREMA-D')
        self.data_root = data_root
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = data_root
        self.audio_feature_path = data_root

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        csv_files = [self.train_csv, self.test_csv]
        for file_path in csv_files:
            with open(file_path, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)
                for item in csv_reader:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(3), item[0])
                    image_samples = os.listdir(visual_path)
                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        if len(image_samples) < 3:
                            print(f"图片不足：{visual_path}，只有 {len(image_samples)} 张，需要 {3} 张")
                            continue
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=3, replace=False)
        select_index.sort()
        images = torch.zeros((3, 3, 224, 224))
        for i in range(3):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return idx,spectrogram, images, label
def get_data(dataset, data_root, iid, num_users,data_aug, noniid_beta):
    ds = dataset 
    if ds == 'cremad' :
        # train_set = CramedDataset(mode='train')
        # test_set = CramedDataset(mode='test')
        # train_set_mia =  CramedDataset(mode='train')
        # test_set_mia = CramedDataset(mode='test')


        full_data_set = CramedDataset(data_root=data_root)
        total_size = len(full_data_set)
        train_ratio = 0.9
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size
        train_set, test_set = random_split(
            dataset=full_data_set,
            lengths=[train_size, test_size],
            # 💡 可选：设置一个生成器来控制随机种子，以实现结果复现
            # generator=torch.Generator().manual_seed(42)
        )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'cremad3' :
        # train_set = CramedDataset(mode='train')
        # test_set = CramedDataset(mode='test')
        # train_set_mia =  CramedDataset(mode='train')
        # test_set_mia = CramedDataset(mode='test')


        full_data_set = CramedDataset3()
        total_size = len(full_data_set)
        train_ratio = 0.9
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size
        train_set, test_set = random_split(
            dataset=full_data_set,
            lengths=[train_size, test_size],
            # 💡 可选：设置一个生成器来控制随机种子，以实现结果复现
            # generator=torch.Generator().manual_seed(42)
        )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'cremad1':
        full_data_set = CramedDataset()
        total_size = len(full_data_set)
        mid_point = total_size // 2

        first_half_indices = list(range(0, mid_point))
        half_size = len(first_half_indices)

        # 2. 在这半个数据里，按 9:1 顺序划分
        # 训练集取这半个里的前 90%
        train_split_limit = int(0.9 * half_size)

        train_indices = first_half_indices[:train_split_limit]
        test_indices = first_half_indices[train_split_limit:]

        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)

        train_set_mia, test_set_mia = train_set, test_set
    if ds == 'cremad_rario1':
        full_data_set = CramedDataset()
        total_size = len(full_data_set)
        excel_path = 'E:\machang\code\FedMIA-main\FedMIA-main\cremad_final_data_loss_new.xlsx'
        df = pd.read_excel(excel_path)
        df['ratio'] = df['audio_score'] / (df['video_scored'] + 1e-6)
        df_sorted = df.sort_values(by='ratio', ascending=True)
        sorted_indices = df_sorted['index'].tolist()
        mid_point = len(sorted_indices) // 2
        selected_indices = sorted_indices[mid_point:]
        half_size = len(selected_indices)
        train_split_limit = int(0.9 * half_size)

        # 获取最终的索引
        train_indices = selected_indices[:train_split_limit]
        test_indices = selected_indices[train_split_limit:]

        # 7. 使用 Subset 构建 PyTorch 数据集对象
        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)
        train_set_mia, test_set_mia = train_set, test_set

    if ds == 'cremad_rario2':
        full_data_set = CramedDataset()
        excel_path = 'E:\machang\code\FedMIA-main\FedMIA-main\cremad_final_data_loss_new.xlsx'
        df = pd.read_excel(excel_path)
        df['ratio'] = df['audio_score'] / (df['video_scored'] + 1e-6)
        df_sorted = df.sort_values(by='ratio', ascending=True)
        sorted_indices = df_sorted['index'].tolist()
        mid_point = len(sorted_indices) // 2
        selected_indices = sorted_indices[:mid_point]
        half_size = len(selected_indices)
        train_split_limit = int(0.9 * half_size)

        # 获取最终的索引
        train_indices = selected_indices[:train_split_limit]
        test_indices = selected_indices[train_split_limit:]

        # 7. 使用 Subset 构建 PyTorch 数据集对象
        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)
        train_set_mia, test_set_mia = train_set, test_set

    if ds == 'cremad2':
        full_data_set = CramedDataset()
        total_size = len(full_data_set)
        mid_point = total_size // 2

        second_half_indices = list(range(mid_point, total_size))
        half_size = len(second_half_indices)

        # 2. 在这半个数据里，按 9:1 顺序划分
        # 训练集取这半个里的前 90%
        train_split_limit = int(0.9 * half_size)

        train_indices = second_half_indices[:train_split_limit]
        test_indices = second_half_indices[train_split_limit:]

        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)

        train_set_mia, test_set_mia = train_set, test_set
    if ds == 'AVE':
        # train_set = AVDataset(mode='train')
        # test_set = AVDataset(mode='test')
        # train_set_mia =  AVDataset(mode='train')
        # test_set_mia = AVDataset(mode='test')

        full_data_set = AVDataset()
        total_size = len(full_data_set)
        train_ratio = 0.9
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size
        train_set, test_set = random_split(
            dataset=full_data_set,
            lengths=[train_size, test_size],
            # 💡 可选：设置一个生成器来控制随机种子，以实现结果复现
            # generator=torch.Generator().manual_seed(42)
        )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'AVE1':
        full_data_set = AVDataset()
        total_size = len(full_data_set)
        mid_point = total_size // 2

        first_half_indices = list(range(0, mid_point))
        half_size = len(first_half_indices)

        # 2. 在这半个数据里，按 9:1 顺序划分
        # 训练集取这半个里的前 90%
        train_split_limit = int(0.9 * half_size)

        train_indices = first_half_indices[:train_split_limit]
        test_indices = first_half_indices[train_split_limit:]

        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)

        train_set_mia, test_set_mia = train_set, test_set
    if ds == 'AVE2':
        full_data_set = AVDataset()
        total_size = len(full_data_set)
        mid_point = total_size // 2

        second_half_indices = list(range(mid_point, total_size))
        half_size = len(second_half_indices)

        # 2. 在这半个数据里，按 9:1 顺序划分
        # 训练集取这半个里的前 90%
        train_split_limit = int(0.9 * half_size)

        train_indices = second_half_indices[:train_split_limit]
        test_indices = second_half_indices[train_split_limit:]

        train_set = Subset(full_data_set, train_indices)
        test_set = Subset(full_data_set, test_indices)

        train_set_mia, test_set_mia = train_set, test_set
    if ds == 'balanced':
        full_data_set = Balanced()
        total_size = len(full_data_set)
        train_ratio = 0.9
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size
        train_set, test_set = random_split(
            dataset=full_data_set,
            lengths=[train_size, test_size],
            # 💡 可选：设置一个生成器来控制随机种子，以实现结果复现
            # generator=torch.Generator().manual_seed(42)
        )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'cifar10':

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )

    if ds == 'cifar100':
        if data_aug :
            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),#
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(45),
                                                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])

            transform_train_mia = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test_mia = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test_mia
                                                )

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    if ds == 'dermnet':
        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        print(total_set[0].shape) # 19559, 3, 64, 64
        print(total_set[1].shape) # 19559
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:15000],total_set[1][0:15000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-4000:],total_set[1][-4000:] )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'oct':
        data=torch.load(data_root+"/oct_ts.pt")
        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:20000],total_set[1][0:20000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-2000:],total_set[1][-2000:] )

    if iid:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)
    else:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA_by_ratio(train_set, num_users)

    return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # 1. 拿到分配给用户的相对索引 (例如: 0)
        relative_idx = self.idxs[item]

        # 2. 直接向 train_set (Subset) 请求数据
        # subset 会自动查表，去底层 dataset 拿对应的全局数据
        # 因为底层 dataset 已经返回了 (global_idx, ...)，这里直接透传即可
        return self.dataset[relative_idx]

