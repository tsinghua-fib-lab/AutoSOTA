import numpy as np
import pandas as pd
import os
from nmf_algos.utils.linalg_utils import get_l2_error
from nmf_algos.utils.utils import generate_result_dir, generate_data_name
from nmf_algos.utils.algo_utils import local_min


def fetch_method_result_dict(
    project_dir,
    method_name,
    dataset_name,
    latent_dim,
    iter_num=1,
    suffix="default",
    result_dir=None,
):
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
    return exacts_data


def fetch_enmf_run_time(
    project_dir, dataset_name, latent_dim, iter_num=1, suffix="default", result_dir=None
):
    method_name = "ENMF"
    exacts_data = fetch_method_result_dict(
        project_dir,
        method_name,
        dataset_name,
        latent_dim,
        iter_num=iter_num,
        suffix=suffix,
        result_dir=result_dir,
    )
    print(exacts_data["total_time"])
    return exacts_data["total_time"], exacts_data["enmf_error"]


def fetch_svd_from_enmf(
    project_dir, dataset_name, latent_dim, iter_num=1, suffix="default", result_dir=None
):
    method_name = "ENMF"
    exacts_data = fetch_method_result_dict(
        project_dir,
        method_name,
        dataset_name,
        latent_dim,
        iter_num=iter_num,
        suffix=suffix,
        result_dir=result_dir,
    )
    return exacts_data["svd_error"]


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
    exacts_data = fetch_method_result_dict(
        project_dir,
        method_name,
        dataset_name,
        latent_dim,
        iter_num=iter_num,
        suffix=suffix,
        result_dir=result_dir,
    )
    re_error = get_l2_error(exacts_data["X"], exacts_data["U"], exacts_data["V"])
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
    compute_relative_error=False,
):
    error_list = []
    if compute_relative_error:
        # fetch svd error from enmf first
        svd_baseline_error = fetch_svd_from_enmf(
            project_dir,
            dataset_name,
            latent_dim,
            iter_num=1,
            suffix=enmf_suffix,
            result_dir=result_dir,
        )
    else:
        svd_baseline_error = 1

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
        error_list.append(current_re_error / svd_baseline_error)
    return compute_avg_std(error_list)


def generate_kkt_condition_for_single_run(
    project_dir,
    dataset_name,
    method_name,
    latent_dim,
    iter_num=1,
    suffix="default",
    enmf_suffix="default",
    result_dir=None,
):
    if method_name == "ENMF":
        method_suffix = enmf_suffix
    else:
        method_suffix = suffix
    data_dict = fetch_method_result_dict(
        project_dir,
        method_name,
        dataset_name,
        latent_dim,
        iter_num=iter_num,
        suffix=method_suffix,
        result_dir=result_dir,
    )
    # result_dict = {}
    # result_dict["mat_abs_max"], result_dict["mat_abs_sum"], result_dict["grad_min"] = local_min(data_dict["X"], data_dict["U"], data_dict["V"])
    mat_abs_max, mat_abs_sum, grad_min = local_min(
        data_dict["X"], data_dict["U"], data_dict["V"]
    )
    return (mat_abs_max, mat_abs_sum, grad_min)


def convert_std_into_string(mean_std_tuple):
    return f"{mean_std_tuple[0]:.2f}  " + f"{mean_std_tuple[1]:.2f}"


def create_df_summary(
    method_list,
    project_dir,
    dataset_name,
    latent_dims,
    num_iters=1,
    suffix="default",
    enmf_suffix="default",
    result_dir=None,
    compute_relative_error=False,
):
    shown_avg_only = False  # True
    result_dict = {}
    pd.options.display.float_format = "${:,.2f}".format
    if method_list is None:
        print(os.listdir(os.path.join(project_dir, "Results", result_dir)))
        method_list = os.listdir(os.path.join(project_dir, "Results", result_dir))
    for method_name in method_list:
        result_list = []
        for latent_dim in latent_dims:
            method_result = generate_stats_for_multiple_runs(
                project_dir,
                dataset_name,
                method_name,
                latent_dim,
                num_iters=num_iters,
                suffix=suffix,
                enmf_suffix=enmf_suffix,
                result_dir=result_dir,
                compute_relative_error=compute_relative_error,
            )
            if shown_avg_only:
                result_list.append(f"method_result[0]:.2f")
            else:
                result_list.append(convert_std_into_string(method_result))
        result_dict[method_name] = result_list
        result_dict["latent_dim"] = latent_dims
    # each row is a latent_dim, each column is an algorithm
    result_pd = pd.DataFrame.from_dict(result_dict)
    result_pd = result_pd.set_index("latent_dim")
    # with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(result_pd)
    result_pd.to_csv(os.path.join(project_dir, "Results", dataset_name, "summary.csv"))


def create_df_summary_kkt(
    method_list,
    project_dir,
    dataset_name,
    latent_dims,
    iter_num=1,
    suffix="default",
    enmf_suffix="default",
    result_dir=None,
):
    "Display KKT condition for results from method_list."
    result_dict = {}
    pd.options.display.float_format = "${:,.2f}".format

    if method_list is None:
        print(os.listdir(os.path.join(project_dir, "Results", result_dir)))
        method_list = os.listdir(os.path.join(project_dir, "Results", result_dir))

    for method_name in method_list:
        result_list = []
        for latent_dim in latent_dims:
            kkt_tuple = generate_kkt_condition_for_single_run(
                project_dir,
                dataset_name,
                method_name,
                latent_dim,
                iter_num=iter_num,
                suffix=suffix,
                enmf_suffix=enmf_suffix,
                result_dir=result_dir,
            )
            result_list.append(kkt_tuple)
        result_dict[method_name] = result_list
        result_dict["latent_dim"] = latent_dims
    # each row is a latent_dim, each column is an algorithm
    result_pd = pd.DataFrame.from_dict(result_dict)
    result_pd = result_pd.set_index("latent_dim")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(result_pd)
    return result_pd


def display_reconstruction_error(method_list):
    # method_list = ["ENMF", "HALS", "MUL","AOADMM", "GRADMUL", "ALS"]
    suffix = "tc"
    # dataset_name = "Face"
    dataset_name = "Audiomnist"
    # dataset_name = "Audiomnist_shift_58.66517083673411"
    # dataset_name = "syn_snr_0.00143"
    project_dir = os.path.join(os.getcwd(), "ENMF")
    if "Face" in dataset_name:
        # result_dir = "Face_no_shift"
        result_dir = None
        latent_dims = [5, 10, 15, 20, 25]
        create_df_summary(
            method_list,
            project_dir,
            dataset_name,
            latent_dims,
            suffix=suffix,
            enmf_suffix="default",
        )
    elif "Audiomnist" in dataset_name:
        # result_dir = "Audiomnist_scaled100"
        # result_dir = "Audiomnist_shift_58.66517083673411"
        result_dir = None
        # latent_dims = [10, 20, 40, 60, 80, 100]
        latent_dims = [10, 20, 40, 80, 100]
        num_iters = 3
        enmf_suffix = "tc"  # "default"
        create_df_summary(
            method_list,
            project_dir,
            dataset_name,
            latent_dims,
            num_iters=num_iters,
            suffix=suffix,
            enmf_suffix=enmf_suffix,
            result_dir=result_dir,
        )
    elif "syn" in dataset_name:
        # result_dir = "Audiomnist_scaled100"
        noise_levels = ["0.00143", "0.01447", "0.05705", "0.09950"]
        for noise_level in noise_levels:
            result_dir = f"syn_snr_{noise_level}"
            print("result_dir:", result_dir)
            dataset_name = result_dir
            latent_dims = [100, 200, 300, 400, 500]
            method_list = None
            create_df_summary(
                method_list,
                project_dir,
                dataset_name,
                latent_dims,
                suffix=suffix,
                enmf_suffix="tc",
                result_dir=result_dir,
                compute_relative_error=True,
            )


def display_kkt_conditions():
    method_list = ["ENMF", "HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
    suffix = "tc"
    dataset_name = "Face"
    project_dir = os.path.join(os.getcwd(), "ENMF")
    if "Face" in dataset_name:
        # result_dir = "Face_no_shift"
        result_dir = None
        latent_dims = [5, 10, 15, 20, 25]
        result_pd = create_df_summary_kkt(
            method_list,
            project_dir,
            dataset_name,
            latent_dims,
            suffix=suffix,
            enmf_suffix="default",
        )
        if result_dir is not None:
            output_dir = os.path.join(project_dir, "Results", result_dir, "Stats")
        else:
            output_dir = os.path.join(project_dir, "Results", dataset_name, "Stats")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        result_pd.to_csv(os.path.join(output_dir, "KKT_condition.csv"))


if __name__ == "__main__":
    # display_kkt_conditions()
    # method_list = ["ENMF", "HALS", "MUL","AOADMM", "GRADMUL", "ALS"]
    # method_list = ["HALS", "MUL","AOADMM", "GRADMUL", "ALS"]
    method_list = ["ENMF"]
    display_reconstruction_error(method_list)
