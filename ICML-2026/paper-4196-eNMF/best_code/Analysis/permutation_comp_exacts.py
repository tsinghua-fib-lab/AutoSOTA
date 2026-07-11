import os, sys
import numpy as np
from numpy import linalg as LA
from .permutation_comp_utils import *
from nmf_algos.utils.utils import  load_data_matrix

if __name__ == "__main__":
    # Exact Dataset for different sparsities:
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir)
    sparsity_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    latent_dim = 50
    # TO compare VS GT factors, create a dummy method GT.
    for sparsity in sparsity_list:
        data_dir = os.path.join(project_dir, "Dataset/exact_dataset")
        fname = generate_data_name(
            dataset=f"exacts_{sparsity}",
            method_name="GT",
            latent_dim=latent_dim,
            suffix="tc",
        )
        dst_dir = os.path.join(
            project_dir,
            f"Results/exacts_{sparsity}",
            "GT",
            f"latent_dim_{latent_dim}",
            "1",
        )
        dump_gt_factors_as_into_method(
            data_dir, f"exacts_500_400_50_{sparsity}.npy", dst_dir, fname
        )

    method_names = ["ENMF", "HALS", "MUL", "AOADMM", "GRADMUL", "ALS", "GT"]

    matching_threshold = 0.05
    method_id_pairs = [[6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]]
    method_name_pairs = [
        [method_names[method_id_pair[0]], method_names[method_id_pair[1]]]
        for method_id_pair in method_id_pairs
    ]

    method_suffix_list = ["tc", "tc"]
    for sparsity in sparsity_list:
        for method_name_pair in method_name_pairs:
            print(
                f"-------------------sparsity: {sparsity}----{method_name_pair[0]}-VS-{method_name_pair[1]} ---------------------"
            )
            dataset_name = f"exacts_{sparsity}"
            org_data_path = os.path.join(
                project_dir,
                "Dataset/exact_dataset",
                f"exacts_500_400_50_{sparsity}.npy",
            )
            org_data_mat = load_data_matrix(org_data_path)["X"]
            compare_one_dataset_over_latent_dims(
                org_data_mat,
                dataset_name,
                method_name_pair,
                method_suffix_list,
                [latent_dim],
                matching_threshold=matching_threshold,
            )
