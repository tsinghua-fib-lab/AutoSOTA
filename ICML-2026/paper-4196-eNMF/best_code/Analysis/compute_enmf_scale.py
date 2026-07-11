import numpy as np
import pandas as pd
import os
from nmf_algos.utils.linalg_utils import get_l2_error
from nmf_algos.utils.utils import generate_result_dir, generate_data_name


def fetch_enmf_run_time(
    project_dir, dataset_name, latent_dim, iter_num=1, suffix="default", result_dir=None
):
    method_name = "ENMF"
    f_data_dir = generate_result_dir(
        dataset_name,
        method_name=method_name,
        latent_dim=latent_dim,
        iter=iter_num,
        result_dir=result_dir,
    )
    f_name = generate_data_name(
        dataset_name, method_name=method_name, latent_dim=latent_dim, suffix=suffix
    )
    data_path = os.path.join(project_dir, "Results", f_data_dir, f_name)
    exacts_data = np.load(data_path, allow_pickle=True).all()
    print(exacts_data["total_time"])
    return exacts_data["total_time"], exacts_data["enmf_error"]


def fetch_algos_re_error(
    project_dir,
    method_name,
    dataset_name,
    latent_dim,
    iter_num=1,
    suffix="default",
    result_dir=None,
):
    "Get reconstruction error for a given method (except ENMF)"
    # default result_dir is the same as dataset_name
    f_data_dir = generate_result_dir(
        dataset_name,
        method_name=method_name,
        latent_dim=latent_dim,
        iter=iter_num,
        result_dir=result_dir,
    )
    f_name = generate_data_name(
        dataset_name, method_name=method_name, latent_dim=latent_dim, suffix=suffix
    )
    data_path = os.path.join(project_dir, "Results", f_data_dir, f_name)
    exacts_data = np.load(data_path, allow_pickle=True).all()
    re_error = get_l2_error(exacts_data["X"], exacts_data["U"], exacts_data["V"])
    # print(exacts_data.keys())
    return re_error


def compute_avg_std(data_list):
    return np.mean(data_list), np.std(data_list)


def generate_stats_for_multiple_runs(
    project_dir,
    dataset_name,
    method_name,
    latent_dim,
    num_iters=1,
    suffix="default",
    enmf_suffix="default",
    result_dir=None,
):
    error_list = []
    for iter_num in range(1, num_iters + 1):
        if method_name == "ENMF":
            _, current_re_error = fetch_enmf_run_time(
                project_dir,
                dataset_name,
                latent_dim,
                iter_num=iter_num,
                suffix=enmf_suffix,
                result_dir=result_dir,
            )
        else:
            current_re_error = fetch_algos_re_error(
                project_dir,
                method_name,
                dataset_name,
                latent_dim,
                iter_num=iter_num,
                suffix=suffix,
                result_dir=result_dir,
            )
        error_list.append(current_re_error)
    return compute_avg_std(error_list)


def create_df_summary_from_result_dirs(
    result_dir_list,
    method_name,
    project_dir,
    latent_dims,
    num_iters=1,
    suffix="default",
    enmf_suffix="default",
):
    shown_avg_only = True
    result_dict = {}
    pd.options.display.float_format = "${:,.2f}".format
    for result_dir in result_dir_list:
        result_list = []
        for latent_dim in latent_dims:
            method_result = generate_stats_for_multiple_runs(
                project_dir,
                result_dir,
                method_name,
                latent_dim,
                num_iters=1,
                suffix=suffix,
                enmf_suffix=enmf_suffix,
                result_dir=result_dir,
            )
            if shown_avg_only:
                result_list.append(method_result[0])
        result_dict[result_dir] = result_list
        result_dict["latent_dim"] = latent_dims
    # each row is a latent_dim, each column is an algorithm
    result_pd = pd.DataFrame.from_dict(result_dict)
    result_pd = result_pd.set_index("latent_dim")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(result_pd)


if __name__ == "__main__":
    method_list = ["ENMF", "HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
    method_name = "ENMF"
    suffix = "tc"
    # dataset_name = "Face"
    dataset_name = "Audiomnist"
    project_dir = os.path.join(os.getcwd())

    # result_dir = "Audiomnist_scaled100"
    shift_scale = [60, 80, 120, 200]
    result_dir_list = [f"Audiomnist_shift_{shift}" for shift in shift_scale]
    latent_dims = [10, 20, 40]
    create_df_summary_from_result_dirs(
        result_dir_list,
        method_name,
        project_dir,
        latent_dims,
        suffix=suffix,
        enmf_suffix="default",
    )
