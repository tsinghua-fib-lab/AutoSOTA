import os
import numpy as np
from numpy import linalg as LA
from .permutation_comp_utils import *
from nmf_algos.utils.utils import  load_data_matrix

if __name__ == "__main__":
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir)

    dataset_name = "Verb"
    f_path = os.path.join(project_dir, "Dataset", "verb_matrix.npy")
    latent_dims = [10, 20, 40, 80, 100]

    dataset_name = "Face"
    f_path = os.path.join(project_dir, "Dataset", "face_id_4.npy")
    latent_dims = [5, 10, 15, 20, 25]

    org_data_mat = load_data_matrix(f_path)

    print(org_data_mat.shape)
    convergence_percent_threshold = 0.05
    method_name_pairs = [["HALS", "ENMF"], ["AOADMM", "ENMF"], ["ALS", "ENMF"]]
    # method_suffix_list = ["tc", "default"]
    method_suffix_list = ["tc", "tc"]

    # Option1: compare directly
    for method_name_pair in method_name_pairs:
        compare_one_dataset_over_latent_dims(
            org_data_mat,
            dataset_name,
            method_name_pair,
            method_suffix_list,
            latent_dims,
            matching_threshold=convergence_percent_threshold,
        )

    # Option2: compare via specifying a config storing results
    # idx of the method_configs
    # method_id_pairs = [[0, 2], [1, 2]]
    # compare_methods_from_config(org_data_mat, dataset_config, method_id_pairs, convergence_percent_threshold)
