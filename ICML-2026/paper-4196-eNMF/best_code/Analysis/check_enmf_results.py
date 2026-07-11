import os
import numpy as np
import pandas as pd
from nmf_algos.utils.utils import generate_result_dir, generate_data_name
from nmf_algos.utils.ENMF_utils import check_po_distance_from_SVD_factors
from nmf_algos.utils.utils import load_data_basedon_proto, fetch_factors_from_result_path


def fetch_enmf_run_time(
    project_dir,
    dataset_name,
    latent_dim,
    iter_num=1,
    method_suffix="",
    suffix="default",
    result_dir=None,
):
    method_name = "ENMF"
    f_data_dir = generate_result_dir(
        dataset_name,
        method_name=method_name + method_suffix,
        latent_dim=latent_dim,
        iter=iter_num,
        result_dir=result_dir,
    )
    f_name = generate_data_name(
        dataset_name,
        method_name=method_name + method_suffix,
        latent_dim=latent_dim,
        suffix=suffix,
    )
    data_path = os.path.join(project_dir, "Results", f_data_dir, f_name)
    exacts_data = np.load(data_path, allow_pickle=True).all()
    print(exacts_data["total_time"])
    import_key_list = [
        "svd_error",
        "t_rotate",
        "distance_po",
        "t_mp",
        "hitmp_error",
        "t_nmf",
        "enmf_error",
        "total_time",
    ]
    key_result_dict = {}

    for key in import_key_list:
        # print(key, exacts_data[key])
        key_result_dict[key] = exacts_data[key]

        if "data_scale" in exacts_data:
            # print("data_scale", exacts_data["data_scale"])
            if key == "svd_error" or key == "hitmp_error":
                key_result_dict[key] = key_result_dict[key] * exacts_data["data_scale"]
    # print("po dist to svd factors:", check_po_distance_from_SVD_factors(exacts_data['U_eig'], exacts_data['V_eig']))
    key_result_dict["svd_PO"] = check_po_distance_from_SVD_factors(
        exacts_data["U_eig"], exacts_data["V_eig"]
    )

    return exacts_data["total_time"], exacts_data["enmf_error"], key_result_dict


def show_dataset_result_varying_latent_dims(
    dataset_name, result_dir=None, method_suffix=""
):
    # the suffix for each saved factor
    running_mode_suffix = "tc"  # "default"
    if "Face" in dataset_name:
        # dataset_name = "Face"
        latent_dims = [5, 10, 15, 20, 25]
        # result_dir = "Face_no_shift"
    elif "Audiomnist" in dataset_name:
        # dataset_name = "Audiomnist"
        # latent_dims = [10, 20, 40]
        # latent_dims = [10, 20, 40, 60, 80, 100]
        latent_dims = [10, 20, 40, 80, 100]
        # latent_dims = [60, 80, 100]
        result_dir = dataset_name
        method_suffix = method_suffix
    elif "exacts" in dataset_name:
        latent_dims = [10]
        result_dir = dataset_name
        running_mode_suffix = "default"

    project_dir = os.path.join(os.getcwd())
    df_dict = {}
    result_dict_list = []

    for latent_dim in latent_dims:
        _, _, result_dict = fetch_enmf_run_time(
            project_dir,
            dataset_name,
            latent_dim,
            iter_num=1,
            method_suffix=method_suffix,
            suffix=running_mode_suffix,
            result_dir=result_dir,
        )
        result_dict[f"latent_dim"] = latent_dim
        result_dict_list.append(result_dict)

    result_pd = pd.DataFrame(result_dict_list).set_index("latent_dim")
    print(result_pd)


def show_dataset_result_varying_datasets(
    dataset_name_list, latent_dim_list, result_dir=None, method_suffix=""
):
    # the suffix for each saved factor
    running_mode_suffix = "default"  # "tc"
    project_dir = os.path.join(os.getcwd())
    df_dict = {}
    result_dict_list = []
    for dataset_name, latent_dim in zip(dataset_name_list, latent_dim_list):
        _, _, result_dict = fetch_enmf_run_time(
            project_dir,
            dataset_name,
            latent_dim,
            iter_num=1,
            method_suffix=method_suffix,
            suffix=running_mode_suffix,
            result_dir=result_dir,
        )
        result_dict[f"dataset_name"] = dataset_name
        result_dict_list.append(result_dict)

    result_pd = pd.DataFrame(result_dict_list).set_index("dataset_name")
    print(result_pd)


if __name__ == "__main__":

    ######/////--------------Experiment here----------------/////######
    # single dataset
    dataset_name_list = ["exacts_RSR_50_40_10_0.1"]
    latent_dim_list = [10]
    show_dataset_result_varying_datasets(dataset_name_list, latent_dim_list)

    ######/////--------------Experiment here----------------/////######
    # all exacts dataset
    project_dir = os.path.join(os.getcwd())
    proto_path = os.path.join(project_dir, "Data/exact_data_algo_RSR.json")
    dataset_configs = load_data_basedon_proto(
        proto_path, mode="exactDatasets"
    ).exact_dataset
    # print(dataset_configs)
    dataset_name_list = []
    latent_dim_list = []
    for dataset_config in dataset_configs:
        f_path = os.path.join(
            project_dir, dataset_config.data_dir, dataset_config.data_path
        )
        dataset_name_list.append(dataset_config.name)
        latent_dim_list.append(dataset_config.method_config[0].latent_dim)
        # print(dataset_config.data_path)
    show_dataset_result_varying_datasets(dataset_name_list, latent_dim_list)
