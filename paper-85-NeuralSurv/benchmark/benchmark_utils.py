import os
import time
from datetime import date
import pandas as pd
import numpy as np

from postprocessing.evaluation_metrics import EvaluationMetrics


def get_benchmark_results_central_experiment(
    load_data_fn,
    fit_benchmark_fn,
    output_dir=None,
    subsample_n=125,
    jobid="sub_125_layers_2_hidden_16",
):

    # Store metrics for each fold
    metrics_list = []
    surv_new_times_list = []

    if output_dir is not None:
        today_str = date.today().isoformat()
        dated_folder = os.path.join(output_dir, today_str)
        os.makedirs(dated_folder, exist_ok=True)

    for chosen_fold in range(5):
        data = load_data_fn(subsample_n=subsample_n, chosen_fold=chosen_fold)

        data_train = data["train"]
        data_val = data["val"]
        data_test = data["test"]

        time_train = data_train["np.array"]["time"]
        event_train = data_train["np.array"]["event"]

        time_test = data_test["np.array"]["time"]
        event_test = data_test["np.array"]["event"]

        time_max = max(time_train.max(), time_test.max())
        delta_time = time_max / 20
        num = int(time_max // delta_time) + 1
        new_times = np.linspace(1e-6, time_max, num=num, dtype=np.float32)

        start_time = time.time()
        risk_test, surv_test, surv_new_times = fit_benchmark_fn(
            data_train, data_val, data_test, new_times
        )
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60

        evaluation_metrics = EvaluationMetrics(
            risk=risk_test,
            survival=surv_test,
            time_train=time_train,
            event_train=event_train,
            event_test=event_test,
            time_test=time_test,
            method="frequentist",
        )

        # append
        df = pd.DataFrame.from_dict(
            evaluation_metrics.metrics, orient="index", columns=["value"]
        )
        df["metric"] = df.index
        df["fold"] = chosen_fold
        metrics_list.append(df)
        surv_new_times_list.append(surv_new_times)

    # Combine all folds into a single DataFrame
    all_metrics_df = pd.concat(metrics_list, ignore_index=True)
    # surv_new_times = np.stack(surv_new_times_list)

    # Print summary
    summary = all_metrics_df.groupby("metric")["value"].agg(["mean", "std"])
    print(summary)

    # Save
    if output_dir is not None:

        method_name = fit_benchmark_fn.__name__.split("fit_")[-1]
        dataset_name = load_data_fn.__name__.split("load_")[-1]

        print(f"Fitting {method_name} took {elapsed_minutes:.4f} minutes")

        # Save run time
        filename = f"{dataset_name}_{jobid}_{method_name}_runtime_log.txt"
        log_file = os.path.join(dated_folder, filename)
        with open(log_file, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Runtime: {elapsed_minutes:.4f} minutes\n"
            )

        # Construct filename and save
        filename = f"{dataset_name}_{jobid}_{method_name}_evaluation_metrics.csv"
        filepath = os.path.join(dated_folder, filename)
        all_metrics_df.to_csv(filepath, index=False)

        # filename = f"{dataset_name}_{jobid}_{method_name}_surv_new_times.csv"
        # filepath = os.path.join(dated_folder, filename)
        # np.save(filepath, surv_new_times)


def get_benchmark_results_synthetic_experiment(
    load_data_fn, fit_benchmark_fn, output_dir=None
):

    subsample_n = 25
    jobid = "sub_25_layers_2_hidden_16"

    data = load_data_fn(subsample_n=subsample_n)

    data_train = data["train"]
    data_val = data["val"]
    data_test = data["test"]

    time_train = data_train["np.array"]["time"]
    event_train = data_train["np.array"]["event"]

    time_test = data_test["np.array"]["time"]
    event_test = data_test["np.array"]["event"]

    time_max = max(115.54582, time_test.max())
    delta_time = time_max / 40
    num = int(time_max // delta_time) + 1
    new_times = np.linspace(1e-6, time_max, num=num, dtype=np.float32)

    risk_test, surv_test, surv_new_times = fit_benchmark_fn(
        data_train, data_val, data_test, new_times
    )

    evaluation_metrics = EvaluationMetrics(
        risk=risk_test,
        survival=surv_test,
        time_train=time_train,
        event_train=event_train,
        event_test=event_test,
        time_test=time_test,
        method="frequentist",
    )

    df = pd.DataFrame.from_dict(
        evaluation_metrics.metrics, orient="index", columns=["value"]
    )
    df["metric"] = df.index
    print(df)

    # Save
    if output_dir is not None:
        today_str = date.today().isoformat()
        dated_folder = os.path.join(output_dir, today_str)
        os.makedirs(dated_folder, exist_ok=True)

        method_name = fit_benchmark_fn.__name__.split("fit_")[-1]
        dataset_name = load_data_fn.__name__.split("load_")[-1]

        # Construct filename and save
        filename = f"{dataset_name}_{jobid}_{method_name}_evaluation_metrics.csv"
        filepath = os.path.join(dated_folder, filename)
        df.to_csv(filepath, index=False)

        filename = f"{dataset_name}_{jobid}_{method_name}_surv_new_times"
        filepath = os.path.join(dated_folder, filename)
        np.save(filepath, surv_new_times)
