"""Demo cases"""

import os
from nmf_algos import NMF_ENMF, NMF_HALS
from nmf_algos.utils.utils import load_data_matrix


def run_single_algorithm(f_path):
    # Step 1: Load data
    # f_path = os.path.join(data_dir, "Dataset/verb/right_matrix.npy" )
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data with shape: ", org_data_mat.shape)
    # Step2: Specify method name
    method_name = "ENMF"
    # Step3: Specify method parameters.
    params = {"X": org_data_mat, "dataset_name": "Verb", "r": 20}
    nmf_enmf = NMF_ENMF(method_name=method_name, params=params)
    # Step4: Call NMF under specified modes:
    # Mode 1. Default run: saved to
    nmf_enmf.basic_run()
    # Mode 2. Run to target error
    # Mode 3. Run until the target time
    print(nmf_enmf.intermediate_result_dict)


def run_single_algorithm_multiple_times(f_path):
    # Step 1: Load data
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data with shape: ", org_data_mat.shape)
    # Step2: Specify method name
    method_name = "HALS"
    # Step3: Specify method parameters.
    params = {"X": org_data_mat, "dataset_name": "Verb", "r": 20, "rerun_times": 2}
    nmf_hals = NMF_HALS(method_name=method_name, params=params)
    # Step4: Call NMF under specified modes:
    # Mode 2. Run to target error
    # Mode 3. Run until the target time
    nmf_hals.run_within_fixed_time(target_run_time=6)


if __name__ == "__main__":
    project_dir = os.path.join(os.getcwd())
    suffix = "tc"
    # run_single_algorithm(project_dir)
    data_path = os.path.join(project_dir, "Dataset/verb/right_matrix.npy")
    run_single_algorithm_multiple_times(data_path)
    # for latent_dim in [5, 10, 15, 20, 25]:
    #     print(fetch_enmf_run_time(project_dir, "Face", latent_dim, suffix))
