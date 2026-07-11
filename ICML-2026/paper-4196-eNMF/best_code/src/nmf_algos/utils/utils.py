import numpy as np
import json
from google.protobuf import json_format

# from ..Data import data_config_pb2
# (TODO) Change folder names
from nmf_algos.dataproto import data_config_pb2


def convert_dict_to_string(data_dict):
    info_str = ""
    for key, val in data_dict.items():
        info_str += f",  {key}: {val}"
    return info_str


def flush_dict_into_log(f_name, iter_n, info_dict):
    with open(f_name, "a+") as f:
        info_str = convert_dict_to_string(info_dict)
        f.write(f"{iter_n}: {info_str} \n")


def assert_shape(x, expected_shape: list):
    """Checking arr shape as expected"""
    assert len(x.shape) == len(expected_shape), (x.shape, expected_shape)
    for _a, _b in zip(x.shape, expected_shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, expected_shape)


def load_data_basedon_proto(f_path, mode="realDataset"):
    with open(f_path, "r") as jsonfile:
        data = json.load(jsonfile)
    if mode == "realDataset" or mode == "synDataset":
        res = json_format.Parse(
            json.dumps(data), data_config_pb2.Datasets.RealDataset()
        )
    elif mode == "exactDataset":
        res = json_format.Parse(
            json.dumps(data), data_config_pb2.Datasets.ExactDataset()
        )
    elif mode == "exactDatasets" or mode == "synDatasets":
        res = json_format.Parse(json.dumps(data), data_config_pb2.Datasets())
    else:
        raise NotImplementedError(f"Unsupported data loading mode: {mode}")
    return res


def load_data_matrix(f_path):
    data = np.load(f_path, allow_pickle=True)[()]
    return data


def load_audio_data(f_path):
    org_data = np.load(f_path, allow_pickle=True)
    return org_data["data"], org_data["label"]


def fetch_factors_from_result_path(f_path, f_type="algo", key_list=None):
    data = np.load(f_path, allow_pickle=True)[()]
    # single method stored
    if f_type == "algo":
        return data
    # mixed methods stored with multiple runs
    elif key_list is not None:
        for key in key_list:
            data = data[key]
    return data


def audio_preprocess(data, shift_min=0):
    # Ensure all negative elements are shifted to positive.
    print("min and max value of data:", np.min(data), np.max(data))
    data_new = data + shift_min
    return data_new


def generate_data_name(dataset, method_name, latent_dim, suffix="default"):
    return f"{method_name}_{dataset}_r_{latent_dim}_{suffix}.npy"


def generate_result_dir(dataset, method_name, latent_dim, iter=1, result_dir=None):
    # e.g.: Verb/ALS/latent_dim_10/1
    if result_dir is not None:
        return f"{result_dir}/{method_name}/latent_dim_{latent_dim}/{iter}"
    else:
        return f"{dataset}/{method_name}/latent_dim_{latent_dim}/{iter}"
