import os
import os
import sys

ROOT_PATH = "ABSOLUTE_PATH_TO_PROJECT_ROOT"
sys.path.append(os.path.join(ROOT_PATH, 'code'))

import json
import math
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from models.uncertainty.dist_match.tree import DistMatchQRF
from models.uncertainty.dist_match.utils import match_ks_stat

TOP_K = 10
PATCH_LEN = 100
TREE_DIR = os.path.join(ROOT_PATH, "models_save/uc/")
DATA_MAP = {
    "Elec": (
        "qrf_ks_stat<0.1_error_normal_electricelectricity-normalized_darts-forest|100.pkl",
        "darts_forest_dist_match_enbPI_electric_s20_260126_140743",
        "electricelectricity-normalized",
        0,
    ),
    "Solar": (
        "qrf_ks_stat<0.1_error_normal_solarSolar_Atl_data_aligned_darts-forest|100.pkl",
        "darts_forest_dist_match_enbPI_solar_atlanta_s20_270126_174558",
        "solarSolar_Atl_data_aligned",
        0,
        # 1,
    ),
    "Wind": (
        "qrf_ks_stat<0.1_error_normal_windWind_Hackberry_Generation_2019_2020_darts-forest|100.pkl",
        "darts_forest_dist_match_enbPI_wind_s20_260126_140743",
        "windWind_Hackberry_Generation_2019_2020",
        0,
    ),
    "META": (
        "qrf_ks_stat<0.1_error_normal_stockMETA_5m_darts-forest|100.pkl",
        "darts_forest_dist_match_stock_meta_5m_s20_260126_140743",
        "stockMETA_5m",
        0,
    ),
    "NVDA": (
        "qrf_ks_stat<0.1_error_normal_stockNVDA_5m_darts-forest|100.pkl",
        "darts_forest_dist_match_stock_nvda_5m_s20_260126_140743",
        "stockNVDA_5m",
        0,
    ),
    "rain": (
        "qrf_ks_stat<0.1_error_normal_raindaily_weather_darts-forest|100.pkl",
        "darts_forest_dist_match_rain_s20_280126_161125",
        "raindaily_weather",
        0
    )
}
# DATA_MAP = {
#     "Elec": (
#         "qrf_ks_stat<0.01_error_normal_electricelectricity-normalized_darts-forest|100.pkl",
#         "darts_forest_dist_match_enbPI_electric_s20_270126_170906",
#         "electricelectricity-normalized",
#         0,
#     ),
#     "Solar": (
#         "qrf_ks_stat<0.01_error_normal_solarSolar_Atl_data_aligned_darts-forest|100.pkl",
#         "darts_forest_dist_match_enbPI_solar_atlanta_s20_270126_174528",
#         "solarSolar_Atl_data_aligned",
#         0,
#         # 1,
#     ),
#     "Wind": (
#         "qrf_ks_stat<0.01_error_normal_windWind_Hackberry_Generation_2019_2020_darts-forest|100.pkl",
#         "darts_forest_dist_match_enbPI_wind_s20_270126_170906",
#         "windWind_Hackberry_Generation_2019_2020",
#         0,
#     ),
#     "META": (
#         "qrf_ks_stat<0.01_error_normal_stockMETA_5m_darts-forest|100.pkl",
#         "darts_forest_dist_match_stock_meta_5m_s20_270126_170906",
#         "stockMETA_5m",
#         0,
#     ),
#     "NVDA": (
#         "qrf_ks_stat<0.01_error_normal_stockNVDA_5m_darts-forest|100.pkl",
#         "darts_forest_dist_match_stock_nvda_5m_s20_270126_170905",
#         "stockNVDA_5m",
#         0,
#     ),
#     "rain": (
#         "qrf_ks_stat<0.01_error_normal_raindaily_weather_darts-forest|100.pkl",
#         "darts_forest_dist_match_rain_s20_280126_161121",
#         "raindaily_weather",
#         0
#     )
# }


def matcher(x1, x2):
    return match_ks_stat(x1, x2) < 0.1


def get_qrf(path: str) -> DistMatchQRF:
    qrf = DistMatchQRF(
        alpha=0.1,
        n_quantile_bins=10,
        feature_dim=-1,
        matcher=matcher,
        match_mask=None,
        n_trees=10,
        bagging_ratio=0.9,
        verbose=False,
    )
    qrf.load_trees(path)
    return qrf


def get_file_dir(data_key: str, table_key: str) -> str:
    return os.path.join(ROOT_PATH, f"outputs/{data_key}/wandb/latest-run/files/media/table/Eval_{table_key}_plots")


def read_json(path: str) -> pd.DataFrame:
    with open(path, "r") as file:
        data = json.load(file)
    cols = data["columns"]
    data = data["data"]
    data = [dict(zip(cols, datum)) for datum in data]
    return pd.DataFrame.from_records(data)


def get_data_df(data_key: str, table_key: str, file_idx: int = 0) -> pd.DataFrame:
    file_dir = get_file_dir(data_key, table_key)
    filename = os.listdir(file_dir)[file_idx]
    return read_json(os.path.join(file_dir, filename))


def get_tree_leafs_hits(data: np.ndarray, qrf_path: str, tree_id: int) -> list:
    qrf = get_qrf(qrf_path)
    tree = qrf.trees[tree_id]
    node_id_map = {node: idx for idx, node in enumerate(tree.leaf_nodes)}
    node_stats = [0 for _ in tree.leaf_nodes]

    for patch in tqdm(data, leave=False):
        node = tree.predict_single_node(patch)
        node_stats[node_id_map[node]] += 1

    return node_stats


def get_tree_leafs_hits_batched(
    data: np.ndarray, qrf_path: str, tree_id: int, n_processes: int = 8
):
    with multiprocessing.Pool(n_processes) as pool:
        callback = partial(get_tree_leafs_hits, qrf_path=qrf_path, tree_id=tree_id)

        batch_size = math.ceil(len(data) / n_processes)
        batches = [
            data[i * batch_size : (i + 1) * batch_size] for i in range(n_processes)
        ]
        tree_stats = pool.map(callback, batches)

    total_hits = [0 for _ in range(len(tree_stats[0]))]
    for cur_stats in tree_stats:
        for leaf_idx, stats in enumerate(cur_stats):
            total_hits[leaf_idx] += stats

    return total_hits


def get_calib_test_hit_ratios(
    data, qrf_path, k: int = 3, patch_len: int = 100, n_processes: int = 8
):
    qrf = get_qrf(qrf_path)
    test_ids = data["step"] >= 0
    test_data, calib_data = data[test_ids], data[~test_ids]
    total_k_hit_ratio = 0
    total_left_hit_ratio = 0
    
    total_major_leaf_size = 0
    total_left_leaf_size = 0

    n_trees = len(qrf.trees)

    x_data = (test_data["y_real"] - test_data["fc_y_hat"]).values
    x_data = np.lib.stride_tricks.sliding_window_view(x_data, patch_len)

    n_test_samples = len(x_data)

    qrf_hits = [
        get_tree_leafs_hits_batched(x_data, qrf_path, tree_id, n_processes)
        for tree_id in tqdm(range(n_trees))
    ]
    qrf_stats = []
    for tree_hits, tree in zip(qrf_hits, qrf.trees):
        cur_stats = [
            {"n_vals": len(node.get_values()[0]), "hits": node_hits}
            for node_hits, node in zip(tree_hits, tree.leaf_nodes)
        ]
        qrf_stats.append(cur_stats)

        total_left_hit_ratio += cur_stats[0]["hits"]
        top_stats = [*cur_stats]
        top_stats.sort(key=lambda x: x["n_vals"], reverse=True)
        top_stats = top_stats[:k]
        total_k_hit_ratio += sum((stat["hits"] for stat in top_stats))

        leaf_sizes = [stat['hits'] + stat['n_vals'] for stat in cur_stats]

        total_left_leaf_size += leaf_sizes[0]
        total_major_leaf_size += max(leaf_sizes[1:])
        

    total_k_hit_ratio /= n_trees * n_test_samples
    total_left_hit_ratio /= n_trees * n_test_samples

    return total_k_hit_ratio, total_left_hit_ratio, total_major_leaf_size, total_left_leaf_size


def app():
    n_processes = multiprocessing.cpu_count()
    for data_type, (qrf_path, data_key, table_key, file_idx) in DATA_MAP.items():
        qrf_path = os.path.join(TREE_DIR, qrf_path)
        data_df = get_data_df(data_key, table_key, file_idx)
        k_hit_ratio, left_hit_ratio, major_leaf_size, left_leaf_size = get_calib_test_hit_ratios(
            data_df, qrf_path, TOP_K, PATCH_LEN, n_processes
        )
        print(
            f"{data_type}: {k_hit_ratio:.2}, {left_hit_ratio:.2}, {major_leaf_size}, {left_leaf_size}, "
        )


if __name__ == "__main__":
    app()
