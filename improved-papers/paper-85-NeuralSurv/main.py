import os
import argparse
import json
import ml_collections
import time

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr

from sksurv.nonparametric import kaplan_meier_estimator

from config import get_config

from model.model import get_model
from neuralsurv import NeuralSurv
from postprocessing.plot import (
    plot_posterior_phi,
    plot_glin_function,
    plot_hazard_function,
    plot_survival_function_vs_km_curve,
)


def main(config: ml_collections.ConfigDict):

    # Create key
    rng = jr.PRNGKey(config.algorithm.seed_train)

    # Get some params
    alpha_prior = config.prior.p_phi_alpha
    beta_prior = config.prior.p_phi_beta
    rho = config.prior.rho
    num_points_integral_em = config.algorithm.num_points_integral_em
    num_points_integral_cavi = config.algorithm.num_points_integral_cavi
    batch_size = config.algorithm.batch_size
    num_samples = config.sample.num_samples
    max_iter_em = config.algorithm.max_iter_em
    max_iter_cavi = config.algorithm.max_iter_cavi
    overwrite_em = config.algorithm.overwrite_em
    overwrite_cavi = config.algorithm.overwrite_cavi

    # Directories
    output_dir = config.directories.outputs
    plot_dir = config.directories.plots

    # Get observed data from the train and test set
    data_train, data_val, data_test = (
        config.data_obs.data_train,
        config.data_obs.data_val,
        config.data_obs.data_test,
    )
    time_train, event_train, x_train = (
        data_train["time"],
        data_train["event"],
        data_train["x"],
    )
    time_val, event_val, x_val = data_val["time"], data_val["event"], data_val["x"]
    time_test, event_test, x_test = (
        data_test["time"],
        data_test["event"],
        data_test["x"],
    )

    # Get model
    model = get_model(config)

    # Instantiate parameters
    rng, step_rng = jr.split(rng)
    model_params_init = model.init(
        step_rng, jnp.array([0]), jnp.zeros(config.data_obs.dim_x)
    )

    # Instantiate neuralsurv
    neuralsurv = NeuralSurv.load_or_create(
        model,
        model_params_init,
        alpha_prior,
        beta_prior,
        rho,
        num_points_integral_em,
        num_points_integral_cavi,
        batch_size,
        max_iter_em,
        max_iter_cavi,
        output_dir,
        overwrite_em,
        overwrite_cavi,
    )

    # Fit neuralsurv
    start_time = time.time()
    neuralsurv.fit(
        time_train,
        event_train,
        x_train,
    )
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Fitting neuralsurv took {elapsed_minutes:.4f} minutes")

    # Save run time
    log_file = os.path.join(plot_dir, "runtime_log.txt")
    with open(log_file, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Runtime: {elapsed_minutes:.4f} minutes\n"
        )

    # Get prior and posterior samples
    rng, step_rng = jr.split(rng)
    neuralsurv.get_posterior_samples(step_rng, num_samples)

    rng, step_rng = jr.split(rng)
    neuralsurv.get_prior_samples(step_rng, num_samples)

    # Plot phi posterior distribution
    plot_posterior_phi(
        neuralsurv.posterior_params["alpha"],
        neuralsurv.posterior_params["beta"],
        plot_dir,
    )

    # Save posterior of gamma ~ Gamma(alpha, beta)
    df = pd.DataFrame(
        {
            "parameter": ["alpha", "beta"],
            "value": [
                neuralsurv.posterior_params["alpha"],
                neuralsurv.posterior_params["beta"],
            ],
        }
    )
    df.to_csv(plot_dir + "/posterior_gamma.csv", index=False)

    # Compute c-index, auc, brier score
    neuralsurv.compute_evaluation_metrics(
        time_train, event_train, time_test, event_test, x_test, plot_dir
    )

    # New times
    time_max = max(time_train.max(), time_test.max())
    delta_time = time_max / 20
    num = int(time_max // delta_time) + 1
    times = jnp.linspace(1e-6, time_max, num=num)

    # Plot hazard function
    plot_hazard_function(
        neuralsurv.predict_hazard_function(times, x_train, aggregate=(0, 2)),
        plot_dir,
        "train",
    )
    plot_hazard_function(
        neuralsurv.predict_hazard_function(times, x_test, aggregate=(0, 2)),
        plot_dir,
        "test",
    )

    # Plot survival function
    km_times_train, km_survival_train = kaplan_meier_estimator(
        event_train.astype(bool), time_train
    )
    km_times_test, km_survival_test = kaplan_meier_estimator(
        event_test.astype(bool), time_test
    )
    surv_train = neuralsurv.predict_survival_function(times, x_train, aggregate=(0, 2))
    surv_test = neuralsurv.predict_survival_function(times, x_test, aggregate=(0, 2))
    plot_survival_function_vs_km_curve(
        surv_train,
        km_times_train,
        km_survival_train,
        plot_dir,
        "train",
    )
    plot_survival_function_vs_km_curve(
        surv_test,
        km_times_test,
        km_survival_test,
        plot_dir,
        "test",
    )

    # Save survival function on test set
    surv_new_times_test = neuralsurv.predict_survival_function(times, x_test)
    file_dir = os.path.join(plot_dir, "survival_function_test")
    np.save(file_dir, surv_new_times_test)

    surv_new_times_train = neuralsurv.predict_survival_function(times, x_train)
    file_dir = os.path.join(plot_dir, "survival_function_train")
    np.save(file_dir, surv_new_times_train)

    # Plot glin prior and posterior distribution
    # plot_glin_function(
    #     neuralsurv.predict_glin_function(
    #         times,
    #         x_train,
    #         aggregate=(0, 2),
    #         dist="posterior",
    #     ),
    #     plot_dir,
    #     "posterior_train",
    # )
    # plot_glin_function(
    #     neuralsurv.predict_glin_function(
    #         times,
    #         x_train,
    #         aggregate=(0, 2),
    #         dist="prior",
    #     ),
    #     plot_dir,
    #     "prior_train",
    # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--workdir", type=str, help="Path to save the results")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = json.load(file)

    if not isinstance(config_data, list):
        config_data = [config_data]

    for config_set in config_data:
        config_set.pop("job_name", None)  # Remove the "job_name" key from the config
        config_set["workdir"] = args.workdir
        config = get_config(**config_set)

        # Print config
        print(config)

        # Get the list of available devices
        devices = jax.devices()

        # Check if there is at least one device and whether it is a CPU or GPU
        if not any(device.device_kind in config.devices.name for device in devices):
            raise ValueError("No valid device found.")
        else:
            print("Valid device found.")
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
            os.environ["XLA_FLAGS"] = (
                "--xla_gpu_triton_gemm_any=true "
                "--xla_gpu_enable_latency_hiding_scheduler=true "
            )

        # Select device
        if config.devices.index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.devices.index

        # run
        main(config)
