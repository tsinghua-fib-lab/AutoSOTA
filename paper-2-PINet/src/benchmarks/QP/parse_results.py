"""Generate files from results of benchmarks."""

import argparse
import csv
import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def generate_bar_data(
    id: str, config: str, filename: str, opt_obj_test: float, plotting: bool = False
) -> None:
    """Generate bar plot data for RS and CV from saved results.

    Args:
        id (str): Identifier for the dataset.
        config (str): Configuration name.
        filename (str): Name of the file to save the results.
        opt_obj_test (float): Optimal objective value for the test set.
        plotting (bool): Whether to plot the results or not.
    """
    results_folder = pathlib.Path(__file__).parent / "results" / id / config
    subfolders = [folder for folder in results_folder.iterdir() if folder.is_dir()]
    # Generate bar plot data
    latest_folder = sorted(subfolders, key=lambda d: d.name)[-1]

    results_file = latest_folder / "results.npz"
    data = jnp.load(results_file)

    # Relative suboptimality
    rs = (data["obj_fun_test"] - opt_obj_test) / jnp.abs(opt_obj_test)

    # Constraint violation
    cv = jnp.maximum(data["eq_viol_test"], data["ineq_viol_test"])
    num_rows = int(rs.shape[0])
    ids = np.arange(1, num_rows + 1)
    performancesfolder = pathlib.Path(__file__).parent / "results" / id / "performances"
    performancesfolder.mkdir(parents=True, exist_ok=True)
    csv_file = performancesfolder / (filename + ".csv")

    data = np.column_stack((ids, rs[:, 0], cv[:, 0]))
    np.savetxt(
        csv_file,
        data,
        delimiter=",",
        header="id,rs,cv",
        comments="",
        fmt=("%d", "%.24f", "%.24f"),
    )

    # Plot bar graphs for relative suboptimality and constraint violation
    if plotting:
        # Convert to NumPy arrays and select the first column if needed
        rs_np = np.array(rs)[:, 0]
        cv_np = np.array(cv)[:, 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        indices = np.arange(len(ids))
        bar_width = 3.0
        ticks = [1e-4, 1e-3, 1e-2, 1e-1]

        # Plot Relative Suboptimality
        ax1.bar(indices, rs_np, bar_width, label="Relative Suboptimality")
        ax1.set_yscale("log")
        ax1.set_xlabel("Instance")
        ax1.set_ylabel("Relative Suboptimality (log scale)")
        ax1.set_title("Relative Suboptimality")
        ax1.set_xticks(indices)
        ax1.set_xticklabels(ids)
        ax1.set_yticks(ticks)
        ax1.set_ylim(1e-3, 5e-1)
        ax1.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        ax1.legend()

        # Plot Constraint Violation
        ax2.bar(indices, cv_np, bar_width, color="orange", label="Constraint Violation")
        ax2.set_yscale("log")
        ax2.set_xlabel("Instance")
        ax2.set_ylabel("Constraint Violation (log scale)")
        ax2.set_title("Constraint Violation")
        ax2.set_xticks(indices)
        ax2.set_xticklabels(ids)
        ax2.set_yticks(ticks)
        ax2.set_ylim(1e-10, 1e0)
        ax2.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.show()


# Generate training curves
def generate_learning_curves(
    id: str, config: str, filename: str, opt_obj_valid: float, plotting: bool = False
) -> None:
    """Parse results and saves learning curves for RS and CV.

    Args:
        id (str): Identifier for the dataset.
        config (str): Configuration name.
        filename (str): Name of the file to save the results.
        opt_obj_valid (float): Optimal objective value for the validation set.
        plotting (bool): Whether to plot the results or not.
    """
    results_folder = pathlib.Path(__file__).parent / "results" / id / config
    subfolders = [folder for folder in results_folder.iterdir() if folder.is_dir()]
    cv_columns = []
    optimal_objective_columns = []
    objective_columns = []
    train_time = []
    setup_time = []
    # Find all available runs with id and config
    for folder in subfolders:
        lc_file = folder / "learning_curves.npz"
        lc_data = jnp.load(lc_file)
        res_file = folder / "results.npz"
        res_data = jnp.load(res_file, allow_pickle=True)
        optimal_objective_columns.append(opt_obj_valid)
        objective_columns.append(lc_data["objective"])
        cv_columns.append(jnp.maximum(lc_data["eqcv"], lc_data["ineqcv"]))
        train_time.append(lc_data["train_time"].reshape(-1, 1))
        setup_time.append(
            res_data["setup_time"] if res_data["setup_time"].item() is not None else 0.0
        )

    # Create a matrix where each column corresponds to a subfolder's objective values
    optimal_objective_matrix = jnp.concatenate(optimal_objective_columns, axis=1)
    optimal_objective_matrix = jnp.tile(
        optimal_objective_matrix, (objective_columns[0].shape[0], 1, 1)
    )
    objective_matrix = jnp.concatenate(objective_columns, axis=2)
    cv_matrix = jnp.concatenate(cv_columns, axis=2)
    train_time_matrix = jnp.concatenate(train_time, axis=1)
    train_time_matrix = (
        jnp.mean(jnp.cumsum(train_time_matrix, axis=0), axis=1)
        + jnp.array(setup_time).mean()
    )
    # Compute relative suboptimality for each run and instance
    rs_runs = (objective_matrix - optimal_objective_matrix) / jnp.abs(
        optimal_objective_matrix
    )
    # Average rs over run dimension (axis=2) and over instances (axis=1)
    avg_rs = jnp.mean(jnp.mean(rs_runs, axis=1), axis=1)
    std_rs = jnp.std(jnp.mean(rs_runs, axis=1), axis=1)

    # Similar for constraint violation
    avg_cv = jnp.mean(jnp.mean(cv_matrix, axis=1), axis=1)
    std_cv = jnp.std(jnp.mean(cv_matrix, axis=1), axis=1)

    if plotting:
        plt.figure()
        # Plot relative suboptimality with shaded standard deviation area
        plt.semilogy(train_time_matrix, avg_rs, label="Relative suboptimality")
        plt.fill_between(train_time_matrix, avg_rs - std_rs, avg_rs + std_rs, alpha=0.3)

        # Plot constraint violation with shaded standard deviation area
        plt.semilogy(train_time_matrix, avg_cv, label="Constraint violation")
        plt.fill_between(train_time_matrix, avg_cv - std_cv, avg_cv + std_cv, alpha=0.3)

        plt.legend()
        plt.show()

    rs_folder = results_folder.parent / "learning_curves" / "RS"
    rs_folder.mkdir(parents=True, exist_ok=True)
    cv_folder = results_folder.parent / "learning_curves" / "CV"
    cv_folder.mkdir(parents=True, exist_ok=True)
    csv_file = rs_folder / (filename + ".csv")
    rs_to_save = np.column_stack(
        (np.asarray(train_time_matrix), np.asarray(avg_rs), np.asarray(std_rs))
    )
    np.savetxt(
        csv_file,
        rs_to_save,
        delimiter=",",
        header="t, mean, std",
        comments="",
        fmt=("%.24f", "%.24f", "%.24f"),
    )

    csv_file = cv_folder / (filename + ".csv")
    cv_to_save = np.column_stack(
        (np.asarray(train_time_matrix), np.asarray(avg_cv), np.asarray(std_cv))
    )
    np.savetxt(
        csv_file,
        cv_to_save,
        delimiter=",",
        header="t, mean, std",
        comments="",
        fmt=("%.24f", "%.24f", "%.24f"),
    )


def generate_time_data(id: str, config: str) -> None:
    """Parse results and save training and inference time data.

    Args:
        id (str): Identifier for the dataset.
        config (str): Configuration name.
    """
    single_inference_dict = {
        "ours": [],
        "dc3": [],
        "softMLP": [],
        "solver": [],
        "jaxopt": [],
    }
    inference_dict = {"ours": [], "dc3": [], "softMLP": [], "solver": [], "jaxopt": []}
    dataset_folder = pathlib.Path(__file__).parent / "results" / id
    dataset_subfolders = [
        folder
        for folder in dataset_folder.iterdir()
        if folder.is_dir()
        and folder.name not in ("learning_curves", "performances", "optimal_objectives")
    ]
    for subfolder in dataset_subfolders:
        if subfolder.name == "dc3":
            method_name = "dc3"
        elif subfolder.name == "softMLP":
            method_name = "softMLP"
        elif subfolder.name == config:
            method_name = "ours"
        elif subfolder.name == "solver":
            method_name = "solver"
        elif (
            subfolder.name == "benchmark_jaxopt_small"
            or subfolder.name == "benchmark_jaxopt_large"
        ):
            method_name = "jaxopt"
        for nested_folder in subfolder.iterdir():
            if nested_folder.is_dir():
                results_file = nested_folder / "results.npz"
                data = jnp.load(results_file)
                single_inference_dict[method_name].append(data["single_inference_time"])
                inference_dict[method_name].append(data["inference_time"])

    def compute_box_stats(values):
        arr = np.array(values)
        median = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        minimum = np.min(arr)
        maximum = np.max(arr)
        return median, q1, q3, minimum, maximum

    training_stats = {}
    inference_stats = {}

    for method in ["ours", "dc3", "solver", "jaxopt"]:
        # for method in training_dict:
        training_stats[method] = compute_box_stats(single_inference_dict[method])
        inference_stats[method] = compute_box_stats(inference_dict[method])

    print("Single inference time statistics (median, Q1, Q3, min, max):")
    for method in training_stats:
        print(training_stats[method])

    print("\nBatch Inference time statistics (median, Q1, Q3, min, max):")
    for method in inference_stats:
        print(inference_stats[method])

    methods_mapping = {
        "softMLP": ("mlpsoft", "SoftMLP"),
        "ours": ("ours", "Ours"),
        "dc3": ("dcthree", "DC3"),
        "solver": ("solver", "Solver"),
        "jaxopt": ("jaxopt", "JAXopt"),
    }

    for dict, name in zip(
        [single_inference_dict, inference_dict], ["single_inference", "inference"]
    ):
        all_stats = {}
        for method, (color, method_name) in methods_mapping.items():
            if method in dict and dict[method]:
                # compute_box_stats returns (median, q1, q3, minimum, maximum)
                stats = compute_box_stats(dict[method])
                median, q1, q3, minimum, maximum = stats
                # Reorder: median, uq (q3), lq (q1), uw (maximum), lw (minimum)
                all_stats[method] = (
                    color,
                    method_name,
                    median,
                    q3,
                    q1,
                    maximum,
                    minimum,
                )

        csv_path = dataset_folder / (name + ".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["color", "method", "median", "uq", "lq", "uw", "lw"])
            for method in ["softMLP", "ours", "dc3", "solver", "jaxopt"]:
                if method in all_stats:
                    writer.writerow(all_stats[method])


def save_optimal_objectives(id: str, config: str):
    """Save optimal objectives for the validation and test set.

    Args:
        id (str): Identifier for the dataset.
        config (str): Configuration name.
    """
    # Setup filenames and check if they exist
    optimal_objectives_folder = (
        pathlib.Path(__file__).parent / "results" / id / "optimal_objectives"
    )
    optimal_objectives_folder.mkdir(parents=True, exist_ok=True)
    optimal_objectives_file = optimal_objectives_folder / "optimal_objectives.npz"
    if optimal_objectives_file.exists():
        return
    results_folder = pathlib.Path(__file__).parent / "results" / id / config
    subfolders = [folder for folder in results_folder.iterdir() if folder.is_dir()]

    latest_folder = sorted(subfolders, key=lambda d: d.name)[-1]

    # Optimal objectives on the test set
    results_file = latest_folder / "results.npz"
    data_results = jnp.load(results_file)
    opt_obj_test = data_results["opt_obj_test"]

    # Optimal objectives on the validation set
    lc_file = latest_folder / "learning_curves.npz"
    data_lc = jnp.load(lc_file)
    # Drop the epoch dimension
    opt_obj_valid = data_lc["optimal_objective"][0]

    jnp.savez(
        optimal_objectives_file, opt_obj_test=opt_obj_test, opt_obj_valid=opt_obj_valid
    )


def load_optimal_objectives(id: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load optimal objectives for the validation and test set.

    Args:
        id (str): Identifier for the dataset.

    Returns:
        tuple: Optimal objectives for the test and validation sets.
    """
    optimal_objectives_folder = (
        pathlib.Path(__file__).parent / "results" / id / "optimal_objectives"
    )
    optimal_objectives_file = optimal_objectives_folder / "optimal_objectives.npz"
    data = jnp.load(optimal_objectives_file)
    opt_obj_test = data["opt_obj_test"]
    opt_obj_valid = data["opt_obj_valid"]

    return opt_obj_test, opt_obj_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for generating results")
    parser.add_argument(
        "--id", type=str, default="dc3_simple_1", help="Dataset identifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pinet",
        help="Configuration (default: pinet, options: dc3, softMLP, jaxopt, pinet)",
    )
    parser.add_argument(
        "--generate_bar_data",
        action="store_true",
        default=False,
        help="Generate bar plot data",
    )
    parser.add_argument(
        "--generate_learning_curves",
        action="store_true",
        default=False,
        help="Generate learning curves data",
    )
    parser.add_argument(
        "--generate_time_data",
        action="store_true",
        default=False,
        help="Generate training and inference time data",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plot results"
    )
    args = parser.parse_args()

    if args.config == "dc3":
        filename = "DC3"
    elif args.config == "softmlp":
        filename = "SoftMLP"
    elif "jaxopt" in args.config:
        filename = "JAXopt"
    elif args.config == "pinet":
        filename = "Ours"
        save_optimal_objectives(args.id, args.config)
    else:
        raise ValueError(
            f"Unknown configuration: {args.config}. "
            "Supported configurations are: dc3, softMLP, jaxopt, pinet."
        )

    opt_obj_test, opt_obj_valid = load_optimal_objectives(args.id)
    if args.generate_bar_data:
        print("Generating bar plot data for id:", args.id, "and config:", args.config)
        # Call bar plot generation functionality here
        generate_bar_data(
            args.id,
            args.config,
            filename=filename,
            plotting=args.plot,
            opt_obj_test=opt_obj_test,
        )

    if args.generate_learning_curves:
        print("Generating learning curves for id:", args.id, "and config:", args.config)
        # Call learning curves generation functionality here
        generate_learning_curves(
            args.id,
            args.config,
            filename=filename,
            plotting=args.plot,
            opt_obj_valid=opt_obj_valid,
        )

    if args.generate_time_data:
        print("Generating time data for id:", args.id, "and config:", args.config)
        # Call learning curves generation functionality here
        generate_time_data(args.id, args.config)
