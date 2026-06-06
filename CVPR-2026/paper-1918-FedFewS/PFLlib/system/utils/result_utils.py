import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, result_dir=None):
    """
    计算多次运行的平均准确率

    Args:
        algorithm: 算法名称
        dataset: 数据集名称
        goal: 实验目标
        times: 运行次数
        result_dir: 结果保存目录（可选，默认为 results/）
    """
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, result_dir)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, result_dir=None):
    """
    获取一个算法的所有运行结果

    Args:
        algorithm: 算法名称
        dataset: 数据集名称（可以包含 /，如 AGNews/noniid_dir_20_a0p1）
        goal: 实验目标
        times: 运行次数
        result_dir: 结果保存目录（可选，默认为 results/）
    """
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        # 新格式：algorithm_goal_times.h5（dataset 信息在目录路径中）
        file_name = algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False, result_dir=result_dir, dataset=dataset)))

    return test_acc


def read_data_then_delete(file_name, delete=False, result_dir=None, dataset=None):
    """
    读取结果数据

    Args:
        file_name: 文件名（不含扩展名）
        delete: 是否读取后删除
        result_dir: 结果保存目录（可选，默认为 results/）
        dataset: 数据集名称（可以包含 /，如 AGNews/noniid_dir_20_a0p1）
    """
    # 使用可配置的 result_dir
    if result_dir is None:
        # 默认路径：优先使用当前目录的 results/，否则使用 ../results/
        if os.path.exists("results"):
            result_dir = "results"
        else:
            result_dir = "../results"

    # 如果提供了 dataset，解析为目录结构
    if dataset is not None:
        if "/" in dataset:
            # 新格式：results/dataset_name/config_name/file_name.h5
            dataset_name, config_name = dataset.split("/", 1)
            file_path = os.path.join(result_dir, dataset_name, config_name, file_name + ".h5")
        else:
            # 兼容格式：results/dataset_name/file_name.h5
            file_path = os.path.join(result_dir, dataset, file_name + ".h5")
    else:
        # 旧格式（兼容）：results/file_name.h5
        file_path = os.path.join(result_dir, file_name + ".h5")

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc