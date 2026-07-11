import os, sys
import numpy as np
from numpy import linalg as LA
import shutil

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from nmf_algos.utils.utils import generate_data_name, generate_result_dir

def check_permuatation_rate(mat1, mat2):
    # Check each column in mat1, 0-> same. 2-> opposite direction
    height, width = mat1.shape
    assert height >= width
    cosine_dist = cdist(mat1.T, mat2.T, 'cosine')
    row_ind, col_ind = linear_sum_assignment(cosine_dist)
    #print("bipartite Graph matching col id", col_ind)
    #print("bipartite Graph matching row id", row_ind)
    final_matching_cost = cosine_dist[row_ind, col_ind].sum()
    #print("final matching cost", final_matching_cost )
    return cosine_dist, final_matching_cost, col_ind


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

def compute_matching_percent(cosine_dist, col_ind, threshold=0.1):
    num_row, num_col = cosine_dist.shape
    dist_mat = cosine_dist[np.arange(num_row), col_ind]
    percent = float((dist_mat<threshold).sum())/num_col
    return percent
    

def compute_reconstruct_error(X, U, V):
    return LA.norm(X - U@V)
     
# A clean version for current data storage format:
def compare_method_pair_from_npy(dataset, latent_dim, method_name_pair, suffix_list=["default", "default"], threshold=0.1, factor="U", current_dir="./ENMF/Results", iter=1):
    data_pair = []
    for method_name, suffix in zip(method_name_pair, suffix_list):
        f_data_dir = generate_result_dir(dataset, method_name, latent_dim, iter)
        f_name = generate_data_name(dataset, method_name, latent_dim, suffix)
        f_path = os.path.join(current_dir, f_data_dir, f_name)
        method_data = fetch_factors_from_result_path(f_path, f_type="algo")
        data_pair.append(method_data)
        
    cosine_dist, final_matching_cost, col_ind = check_permuatation_rate(data_pair[0][factor], data_pair[1][factor])
    #error_diff = compute_reconstruct_error(data_list[0][factor])
    print(f"{factor} matching percentage", compute_matching_percent(cosine_dist, col_ind, threshold=threshold), "matching cost: ",  final_matching_cost )
    return data_pair

def compare_method_pair_from_config(dataset_config, method_ids, threshold=0.1, factor="U", current_dir="./ENMF",  data_store_type="algo"):
    data_list = []
    method_name_list = []
    factor_list = [factor]*len(method_ids)
    for method_id in method_ids:
        method_config = dataset_config.method_config[method_id]
        f_path = os.path.join(current_dir, method_config.result_dir, method_config.result_path)
        # how the data is stored
        f_type = method_config.result_storage if method_config.HasField("result_storage") else "algo"
        method_data = fetch_factors_from_result_path(f_path, f_type)
        method_name_list.append(method_config.method_name)
        data_list.append(method_data)
        #print(method_config.method_name, method_data[factor].shape)
    cosine_dist, final_matching_cost, col_ind = check_permuatation_rate(data_list[0][factor_list[0]], data_list[1][factor_list[1]])
    #error_diff = compute_reconstruct_error(data_list[0][factor])
    print(f"{factor} matching percentage", compute_matching_percent(cosine_dist, col_ind, threshold=threshold), "matching cost: ",  final_matching_cost )
    return data_list, factor_list, method_name_list


def compare_methods_from_config(org_data_mat, dataset_config, method_id_pairs, convergence_percent_threshold):
    # Option1: Load data with json config
    factor_name_pair = ["U", "V"]
    for method_ids in method_id_pairs:
        print("-------------------------------------------")
        print("latent_dim:", dataset_config.method_config[method_ids[0]].latent_dim)
        data_list, updated_left_factor_name_list, method_name_pair = compare_method_pair_from_config(dataset_config, method_ids=method_ids, threshold=convergence_percent_threshold, factor=factor_name_pair[0])
        _, updated_right_factor_name_list, _ = compare_method_pair_from_config(dataset_config, method_ids=method_ids, threshold=convergence_percent_threshold, factor=factor_name_pair[1])
        #print(updated_left_factor_name_list, updated_right_factor_name_list)
        error1 = compute_reconstruct_error(org_data_mat, data_list[0][updated_left_factor_name_list[0]], data_list[0][updated_right_factor_name_list[0]].T)
        error2 = compute_reconstruct_error(org_data_mat, data_list[1][updated_left_factor_name_list[1]], data_list[1][updated_right_factor_name_list[1]].T)
        print(f"Error diff {method_name_pair[0]} - {method_name_pair[1]}: ", error1 - error2)


def compare_one_dataset_over_latent_dims(org_data_mat, dataset_name, method_name_pair, method_suffix_list, latent_dims, matching_threshold=0.1):
    for latent_dim in latent_dims:
        print("latent dim:", latent_dim)
        print("-------------------------------------------")
        data_list = compare_method_pair_from_npy(dataset_name, latent_dim, method_name_pair, suffix_list=method_suffix_list, threshold=matching_threshold, factor="U", current_dir="./ENMF/Results")
        data_list = compare_method_pair_from_npy(dataset_name, latent_dim, method_name_pair, suffix_list=method_suffix_list, threshold=matching_threshold, factor="V", current_dir="./ENMF/Results")

        error1 = compute_reconstruct_error(org_data_mat, data_list[0]["U"], data_list[0]["V"].T)
        error2 = compute_reconstruct_error(org_data_mat, data_list[1]["U"], data_list[1]["V"].T)
        print(f"Error diff {method_name_pair[0]} - {method_name_pair[1]}: ", error1 - error2)

# Only for exact dataset where the gt factors are available
def dump_gt_factors_as_into_method(data_dir, dataset, dst_dir, fname):
    org_data_path = os.path.join(data_dir, dataset)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, fname)
    shutil.copy(org_data_path, dst_path)
