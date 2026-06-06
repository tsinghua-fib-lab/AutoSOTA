import json
import os
import pickle
from natsort import natsorted
import numpy as np
import random

data_folder = "/home/bingxing2/ailab/group/ai4neuro/BrainLLM/HMC/HMC_processed_pkl_256Hz"
save_folder_train = '/home/bingxing2/ailab/group/ai4neuro/BrainLLM/HMC/cross_json_csbrain/train.json'
save_folder_test = '/home/bingxing2/ailab/group/ai4neuro/BrainLLM/HMC/cross_json_csbrain/test.json'
save_folder_val = '/home/bingxing2/ailab/group/ai4neuro/BrainLLM/HMC/cross_json_csbrain/val.json'

# 数据集基本信息
sampling_rate = 256
ch_names = ["F4", "C4", "O2", "C3"]
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

tuples_list_train = []
tuples_list_test = []
tuples_list_val = []
error_list = []

subject_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
subject_folder = natsorted(subject_folder)

# 将 subject_folder 划分为训练集、验证集和测试集
train_folders = subject_folder[:100]
val_folders = subject_folder[100:100 + 25]
test_folders = subject_folder[100 + 25:]

# 对每个文件夹进行处理，划分到训练集、验证集和测试集
for folder_id, per_folder in enumerate(subject_folder):
    print("正在处理 ", folder_id, '/', len(subject_folder) - 1)
    pkl_files = [os.path.join(per_folder, f) for f in os.listdir(per_folder) if f.endswith('.pkl')]
    pkl_files = natsorted(pkl_files)

    for pkl_id, pkl_file in enumerate(pkl_files):
        subject_name = pkl_file.split("/")[-1].split(".")[0].split("_")[0]

        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            eeg = eeg_data['X']
            label = eeg_data['Y']
        except Exception as e:
            print(f"加载文件 {pkl_file} 时出错: {e}")
            error_list.append(pkl_file)
            continue  # 跳过该文件

        data = {
            "subject_id": int(per_folder.split("/")[-1]),
            "subject_name": subject_name,
            "file": pkl_file,
            "label": label
        }

        if per_folder in train_folders:
            if folder_id < len(train_folders):  # 归一化统计基于训练集
                per_max_value = max(eeg.reshape(-1))
                if per_max_value > max_value:
                    max_value = per_max_value
                per_min_value = min(eeg.reshape(-1))
                if per_min_value < min_value:
                    min_value = per_min_value
                for j in range(num_channels):
                    total_mean[j] += eeg[j].mean()
                    total_std[j] += eeg[j].std()
                num_all += 1
            tuples_list_train.append(data)
        elif per_folder in val_folders:
            tuples_list_val.append(data)
        elif per_folder in test_folders:
            tuples_list_test.append(data)

# 计算均值和标准差（基于训练集）
data_mean = (total_mean / num_all).tolist()
data_std = (total_std / num_all).tolist()

# 创建数据集字典
train_dataset = {
    "subject_data": tuples_list_train,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

test_dataset = {
    "subject_data": tuples_list_test,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

val_dataset = {
    "subject_data": tuples_list_val,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

# 将数据集保存为json文件
formatted_json_train = json.dumps(train_dataset, indent=2)
with open(save_folder_train, 'w') as f:
    f.write(formatted_json_train)

formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_folder_test, 'w') as f:
    f.write(formatted_json_test)

formatted_json_val = json.dumps(val_dataset, indent=2)
with open(save_folder_val, 'w') as f:
    f.write(formatted_json_val)

print("错误文件列表: ", error_list)
