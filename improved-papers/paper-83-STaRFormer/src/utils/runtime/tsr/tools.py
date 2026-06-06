import multiprocessing
import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from torch import Tensor
from typing import Any, Dict



def initialise_multithread(num_cores=-1):
    """
    Initialise pool workers for multi processing
    :param num_cores:
    :return:
    """
    if (num_cores == -1) or (num_cores >= multiprocessing.cpu_count()):
        num_cores = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(num_cores)
    return p


def create_directory(directory_path):
    """
    Create a directory if path doesn't exists
    :param directory_path:
    :return:
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def save_train_duration(file_name, test_duration):
    """
    Save training time
    :param file_name:
    :param test_duration:
    :return:
    """
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float32), index=[0],
                       columns=['train_duration'])
    res['train_duration'] = test_duration
    res.to_csv(file_name, index=False)


def save_test_duration(file_name, test_duration):
    """
    Save test time
    :param file_name:
    :param test_duration:
    :return:
    """
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float32), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def get_synthetic_class_label(y_train, y_test, seed, kmeans_kwargs: Dict[str, Any]):
    targets_complete = np.concatenate([y_train, y_test])
    print(seed, kmeans_kwargs)
    kmeans = KMeans(random_state=seed, **kmeans_kwargs)
    kmeans_fitted = kmeans.fit(targets_complete.reshape(-1, 1))
    kmeans_labels_train, kmeans_labels_test = kmeans_fitted.labels_[:len(y_train)], kmeans_fitted.labels_[len(y_train):]
    assert len(kmeans_labels_train) == len(y_train)
    assert len(kmeans_labels_test) == len(y_test)
    return {'train': kmeans_labels_train, 'test': kmeans_labels_test}

def get_tsr_outlier_maks(target_values, upper_bound_scale: float=0.1, lower_bound_scale: float=0.1):
    y_max = np.max(target_values)
    y_min = np.min(target_values)

    max_bound = upper_bound_scale*y_max
    min_bound = lower_bound_scale*y_min

    upper_bound_mask = target_values <= (y_max - max_bound)
    lower_bound_mask = target_values >= (y_min - min_bound)

    mask = upper_bound_mask*lower_bound_mask
    return {'mask': mask, 'min_bound': min_bound, 'max_bound': max_bound}


def create_window_slices(sequence: Tensor, label: Tensor, target: Tensor, 
                        window_size: int, stride: int):
    num_windows = (sequence.shape[0] - window_size) // stride + 1 
    windows = [sequence]
    label = [label] + [label for _ in range(num_windows)]
    target = [target] + [target for _ in range(num_windows)]
    for i in range(0, num_windows*stride, stride):
        print(sequence.shape)
        print(sequence[i:i+window_size, ...].shape)
        windows.append(
            sequence[i:i+window_size, ...]
        )
    
    return windows, label, target