"""Run HCNN on toy MPC problem."""

import argparse
import datetime
import pathlib
import time
import timeit
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.serialization import to_bytes
from flax.training import train_state
from tqdm import tqdm

from benchmarks.toy_MPC.load_toy_MPC import JaxDataLoader, ToyMPCDataset, load_data
from benchmarks.toy_MPC.model import setup_model
from benchmarks.toy_MPC.plotting import generate_trajectories, plot_training
from src.tools.utils import GracefulShutdown, Logger, load_configuration

jax.config.update("jax_enable_x64", True)


def evaluate_hcnn(
    loader: ToyMPCDataset | JaxDataLoader,
    state: train_state.TrainState,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
    A: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    prefix: str,
    time_evals: int = 10,
    print_res: bool = True,
    cv_tol: float = 1e-3,
    single_instance: bool = True,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
    float,
    float,
]:
    """Evaluate the performance of HCNN.

    Args:
        loader (ToyMPCDataset | JaxDataLoader): DataLoader for the dataset.
        state (train_state.TrainState): The trained model state.
        batched_objective (Callable): Function to compute the objective.
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        lb (jnp.ndarray): Lower bounds for the decision variables.
        ub (jnp.ndarray): Upper bounds for the decision variables.
        prefix (str): Prefix for logging.
        time_evals (int, optional): Number of times to evaluate the model.
        print_res (bool, optional): Whether to print the results.
        cv_tol (float, optional): Tolerance for constraint violations.
        single_instance (bool, optional): Whether to evaluate a single instance or not.

    Returns:
        tuple: A tuple containing:
            - opt_obj (jnp.ndarray): Optimal objective values.
            - hcnn_obj (jnp.ndarray): HCNN objective values.
            - eq_cv (jnp.ndarray): Equality constraint violations.
            - ineq_cv (jnp.ndarray): Inequality constraint violations.
            - ineq_perc (float): Percentage of valid constraint violations.
            - eval_time (float): Average evaluation time.
            - eval_time_std (float): Standard deviation of evaluation time.
    """
    opt_obj = []
    hcnn_obj = []
    eq_cv = []
    ineq_cv = []
    for X, obj in loader:
        X_full = jnp.concatenate(
            (X, jnp.zeros((X.shape[0], A.shape[1] - X.shape[1], 1))), axis=1
        )
        predictions = state.apply_fn(
            {"params": state.params},
            X[:, :, 0],
            X_full,
            test=True,
        )
        opt_obj.append(obj)
        hcnn_obj.append(batched_objective(predictions))
        # Equality Constraint Violation
        eq_cv_batch = jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X_full,
        )
        eq_cv_batch = jnp.max(eq_cv_batch, axis=1)
        eq_cv.append(eq_cv_batch)
        # Inequality Constraint Violation
        ineq_cv_batch_ub = jnp.maximum(
            predictions.reshape(X.shape[0], A.shape[2], 1) - ub, 0
        )
        ineq_cv_batch_lb = jnp.maximum(
            lb - predictions.reshape(X.shape[0], A.shape[2], 1), 0
        )
        # Compute the maximum and normalize by the size
        ineq_cv_batch = jnp.maximum(ineq_cv_batch_ub, ineq_cv_batch_lb) / ub
        ineq_cv_batch = jnp.max(ineq_cv_batch, axis=1)
        ineq_cv.append(ineq_cv_batch)
    # Objectives
    opt_obj = jnp.concatenate(opt_obj, axis=0)
    opt_obj_mean = opt_obj.mean()
    hcnn_obj = jnp.concatenate(hcnn_obj, axis=0)
    hcnn_obj_mean = hcnn_obj.mean()
    # Equality Constraints
    eq_cv = jnp.concatenate(eq_cv, axis=0)
    eq_cv_mean = eq_cv.mean()
    eq_cv_max = eq_cv.max()
    # Inequality Constraints
    ineq_cv = jnp.concatenate(ineq_cv, axis=0)
    ineq_cv_mean = ineq_cv.mean()
    ineq_cv_max = ineq_cv.max()
    ineq_perc = (1 - jnp.mean(ineq_cv > cv_tol)) * 100
    # Inference time (assumes all the data in one batch)
    if single_instance:
        X_inf = X[:1, :, :]
        X_inf_full = jnp.concatenate(
            (X_inf, jnp.zeros((X_inf.shape[0], A.shape[1] - X_inf.shape[1], 1))), axis=1
        )
    else:
        X_inf = X
        X_inf_full = X_full
    times = timeit.repeat(
        lambda: state.apply_fn(
            {"params": state.params},
            X_inf[:, :, 0],
            X_inf_full,
            test=True,
        ).block_until_ready(),
        repeat=time_evals,
        number=1,
    )
    eval_time = np.mean(times)
    eval_time_std = np.std(times)
    if print_res:
        print(f"=========== {prefix} performance ===========")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max eq. cv               : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max normalized ineq. cv  : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print(
            "Perc of valid cv. tol.        : ",
            f"{ineq_perc:.3f}%",
        )
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj_mean:.5f}")

    return (opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, eval_time, eval_time_std)


def main(
    filepath: str,
    config_path: str,
    SEED: int,
    PLOT_TRAINING: bool,
    SAVE_RESULTS: bool,
    use_jax_loader: bool,
    run_name: str,
) -> train_state.TrainState:
    """Main for running toy MPC benchmarks.

    Args:
        filepath (str): Path to the dataset file.
        config_path (str): Path to the configuration file.
        SEED (int): Random seed for reproducibility.
        PLOT_TRAINING (bool): Whether to plot training curves.
        SAVE_RESULTS (bool): Whether to save the results.
        use_jax_loader (bool): Whether to use JAX DataLoader or PyTorch DataLoader.
        run_name (str): Name of the run for logging.

    Returns:
        train_state.TrainState: The trained model state.
    """
    hyperparameters = load_configuration(config_path)
    key = jax.random.PRNGKey(SEED)
    loader_key, key = jax.random.split(key, 2)
    # Parse data
    (
        As,
        lbxs,
        ubxs,
        lbus,
        ubus,
        xhat,
        alpha,
        T,
        base_dim,
        X,
        train_loader,
        valid_loader,
        test_loader,
        batched_objective,
    ) = load_data(
        filepath=filepath,
        rng_key=loader_key,
        val_split=hyperparameters["val_split"],
        test_split=hyperparameters["test_split"],
        batch_size=hyperparameters["batch_size"],
        use_jax_loader=use_jax_loader,
    )

    Y_DIM = As.shape[2]
    # The X contains only the initial conditions.
    # To properly define the equality constraints we need to append zeros
    Xfull = jnp.concatenate(
        (X, jnp.zeros((X.shape[0], As.shape[1] - X.shape[1], 1))), axis=1
    )
    lb = jnp.concatenate((lbxs, lbus), axis=1)
    ub = jnp.concatenate((ubxs, ubus), axis=1)
    # Setup projection layer
    LEARNING_RATE = hyperparameters["learning_rate"]
    # Setup the model
    model, params, train_step = setup_model(
        rng_key=key,
        hyperparameters=hyperparameters,
        A=As,
        X=X,
        b=Xfull,
        lb=lb,
        ub=ub,
        batched_objective=batched_objective,
    )
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    N_EPOCHS = hyperparameters["n_epochs"]
    eval_every = 1
    start = time.time()
    trainig_losses = []
    validation_losses = []
    eqcvs = []
    ineqcvs = []

    with (
        Logger(run_name=run_name, project_name="hcnn_toy_mpc") as data_logger,
        GracefulShutdown("Stop detected, finishing epoch...") as g,
    ):
        data_logger.run.config.update(hyperparameters)
        for step in (pbar := tqdm(range(N_EPOCHS))):
            if g.stop:
                break
            epoch_loss = []
            batch_sizes = []
            start_epoch_time = time.time()
            for batch in train_loader:
                X_batch, _ = batch
                X_batch_full = jnp.concatenate(
                    (
                        X_batch,
                        jnp.zeros(
                            (X_batch.shape[0], As.shape[1] - X_batch.shape[1], 1)
                        ),
                    ),
                    axis=1,
                )
                loss, state = train_step(
                    state,
                    X_batch[:, :, 0],
                    X_batch_full,
                )
                batch_sizes.append(X_batch.shape[0])
                epoch_loss.append(loss)
            weighted_epoch_loss = sum(
                el * bs for el, bs in zip(epoch_loss, batch_sizes)
            ) / sum(batch_sizes)
            trainig_losses.append(weighted_epoch_loss)
            pbar.set_description(f"Train Loss: {weighted_epoch_loss:.5f}")
            epoch_time = time.time() - start_epoch_time

            if step % eval_every == 0:
                start_evaluation_time = time.time()
                # TODO: Use some of the evaluate functions?
                for X_valid, valid_obj in valid_loader:
                    X_valid_full = jnp.concatenate(
                        (
                            X_valid,
                            jnp.zeros(
                                (X_valid.shape[0], As.shape[1] - X_valid.shape[1], 1)
                            ),
                        ),
                        axis=1,
                    )
                    predictions = state.apply_fn(
                        {"params": state.params},
                        X_valid[:, :, 0],
                        X_valid_full,
                        test=True,
                    )
                    validation_loss = batched_objective(predictions)
                    eqcv = jnp.abs(
                        As[0] @ predictions.reshape(-1, Y_DIM, 1) - X_valid_full
                    ).max()
                    ineqcvub = jnp.max(
                        jnp.maximum(predictions.reshape(-1, Y_DIM, 1) - ub, 0), axis=1
                    )
                    ineqcvlb = jnp.max(
                        jnp.maximum(lb - predictions.reshape(-1, Y_DIM, 1), 0), axis=1
                    )
                    ineqcv = jnp.maximum(ineqcvub, ineqcvlb).mean()
                    eqcvs.append(eqcv)
                    ineqcvs.append(ineqcv)
                    validation_losses.append(validation_loss.mean())
                    eval_time = time.time() - start_evaluation_time
                    pbar.set_postfix(
                        {
                            "eqcv": f"{eqcv:.5f}",
                            "ineqcv": f"{ineqcv:.5f}",
                            "Valid. Loss:": f"{validation_loss.mean():.5f}",
                        }
                    )
                    data_logger.log(
                        step,
                        {
                            "weighted_epoch_loss": weighted_epoch_loss,
                            "epoch_training_time": epoch_time,
                            "validation_objective_mean": validation_loss.mean(),
                            "validation_average_rs": (
                                (validation_loss - valid_obj) / jnp.abs(valid_obj)
                            ).mean(),
                            "validation_cv": jnp.maximum(ineqcv, eqcv),
                            "validation_time": eval_time,
                        },
                    )
        training_time = time.time() - start
        print(f"Training time: {training_time:.5f} seconds")

        if PLOT_TRAINING:
            plot_training(
                train_loader,
                valid_loader,
                trainig_losses,
                validation_losses,
                eqcvs,
                ineqcvs,
            )
        _ = evaluate_hcnn(
            loader=valid_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Validation",
            A=As,
            lb=lb,
            ub=ub,
            cv_tol=1e-3,
            single_instance=False,
        )
        opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, mean_inf_time, std_inf_time = (
            evaluate_hcnn(
                loader=test_loader,
                state=state,
                batched_objective=batched_objective,
                prefix="Test",
                A=As,
                lb=lb,
                ub=ub,
                cv_tol=1e-3,
                time_evals=10,
                single_instance=False,
            )
        )
        _, _, _, _, _, mean_inf_time_single, std_inf_time_single = evaluate_hcnn(
            loader=test_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Test",
            A=As,
            lb=lb,
            ub=ub,
            cv_tol=1e-3,
            time_evals=10,
            single_instance=True,
        )

        # Log summary metrics for wandb
        rs = (hcnn_obj - opt_obj) / jnp.abs(opt_obj)
        cv = jnp.maximum(eq_cv, ineq_cv)
        cvthres = 1e-3
        data_logger.run.summary.update(
            {
                "Average RS Test": jnp.mean(rs),
                "Max CV Test": jnp.max(cv),
                "Percentage CV < tol": (1 - jnp.mean(cv > cvthres)) * 100,
                "Average Single Inference Time": mean_inf_time_single,
                "Average Batch Inference Time": mean_inf_time,
            }
        )

    if SAVE_RESULTS:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = "results.npz"
        timestamp_folder = pathlib.Path(__file__).parent / "results" / current_timestamp
        timestamp_folder.mkdir(parents=True, exist_ok=True)
        results_path = timestamp_folder / results_filename
        # Save the inference time and trajectories
        jnp.savez(
            file=results_path,
            opt_obj=opt_obj,
            hcnn_obj=hcnn_obj,
            eq_cv=eq_cv,
            ineq_cv=ineq_cv,
            ineq_perc=ineq_perc,
            inference_time_mean=mean_inf_time,
            inference_time_std=std_inf_time,
            config_path=config_path,
            **hyperparameters,
        )
        # Save the network parameters for reusing
        params_filename = "params.msgpack"
        params_path = timestamp_folder / params_filename
        with open(params_path, "wb") as f:
            f.write(to_bytes(state.params))

    return state


if __name__ == "__main__":

    def parse_args():
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(description="Run HCNN on toy MPC problem.")
        parser.add_argument(
            "--filename",
            type=str,
            required=True,
            help="Filename of dataset.",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="toy_MPC.yaml",
            help="Configuration file for HCNN hyperparameters.",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for training HCNN."
        )
        parser.add_argument(
            "--plot-training",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Plot training curves.",
        )
        parser.add_argument(
            "--save-results", action="store_true", help="Save the results."
        )
        parser.add_argument(
            "--no-save-results",
            action="store_false",
            dest="save_results",
            help="Don't save the results.",
        )
        parser.add_argument(
            "--use-saved",
            action="store_true",
            help="Use saved network to plot trajectories and print results.",
        )
        parser.add_argument(
            "--results-folder",
            type=str,
            required=False,
            default=None,
            help="Name (suffix) of the results file and params file.",
        )
        parser.add_argument(
            "--jax-loader",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the jax loader or not. If not, use pytorch loader.",
        )
        parser.set_defaults(save_results=True)
        parser.set_defaults(use_saved=False)
        return parser.parse_args()

    # Parse arguments
    args = parse_args()
    filepath = pathlib.Path(__file__).parent.resolve() / "datasets" / args.filename
    config_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "configs"
        / (args.config + ".yaml")
    )
    SEED = args.seed
    torch.manual_seed(SEED)
    use_jax_loader = args.jax_loader
    run_name = f"toy_MPC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not args.use_saved:
        _ = main(
            filepath=filepath,
            config_path=config_path,
            SEED=SEED,
            PLOT_TRAINING=args.plot_training,
            SAVE_RESULTS=args.save_results,
            use_jax_loader=use_jax_loader,
            run_name=run_name,
        )
    else:
        if args.results_folder is None:
            raise ValueError("Please provide the name of the results file.")

        hyperparameters = load_configuration(config_path)
        key = jax.random.PRNGKey(SEED)
        loader_key, key = jax.random.split(key, 2)
        # Parse data
        (
            As,
            lbxs,
            ubxs,
            lbus,
            ubus,
            xhat,
            alpha,
            T,
            base_dim,
            X,
            train_loader,
            valid_loader,
            test_loader,
            batched_objective,
        ) = load_data(
            filepath=filepath,
            val_split=hyperparameters["val_split"],
            test_split=hyperparameters["test_split"],
            batch_size=hyperparameters["batch_size"],
            rng_key=loader_key,
            use_jax_loader=use_jax_loader,
        )
        Y_DIM = As.shape[2]
        # The X contains only the initial conditions.
        # To properly define the equality constraints we need to append zeros
        Xfull = jnp.concatenate(
            (X, jnp.zeros((X.shape[0], As.shape[1] - X.shape[1], 1))), axis=1
        )
        dimx = lbxs.shape[1]
        dimu = lbus.shape[1]
        lb = jnp.concatenate((lbxs, lbus), axis=1)
        ub = jnp.concatenate((ubxs, ubus), axis=1)
        model, params, train_step = setup_model(
            rng_key=key,
            hyperparameters=hyperparameters,
            A=As,
            X=X,
            b=Xfull,
            lb=lb,
            ub=ub,
            batched_objective=batched_objective,
        )

        params_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / ("params.msgpack")
        )
        # Load saved parameters.
        with open(params_filepath, "rb") as f:
            loaded_bytes = f.read()
        from flax.serialization import (  # Import here if not already imported.
            from_bytes,
        )

        restored_params = from_bytes(params["params"], loaded_bytes)

        # Create the optimizer and state.
        tx = optax.adam(learning_rate=hyperparameters["learning_rate"])
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=restored_params, tx=tx
        )

        trajectories_pred, trajectories_cp = generate_trajectories(
            state=state,
            As=As,
            lbxs=lbxs,
            ubxs=ubxs,
            lbus=lbus,
            ubus=ubus,
            alpha=alpha,
            base_dim=base_dim,
            Y_DIM=Y_DIM,
            dimx=dimx,
            xhat=xhat,
            T=T,
            lb=lb,
            ub=ub,
        )

        # Print results
        results_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / "results.npz"
        )
        results = jnp.load(results_filepath)
        print(
            f"Inference Time: {results['inference_time_mean']:.5f} ± "
            f"{results['inference_time_std']:.5f} s"
        )
        rel_suboptimality = (results["hcnn_obj"] - results["opt_obj"]) / results[
            "opt_obj"
        ]
        print(f"Average Relative Suboptimality: {rel_suboptimality.mean():.5%}")
        print(
            f"Percentage of ineq. constraint satisfaction: {results['ineq_perc']:.2f}%"
        )

        if True:
            trajectories_path = (
                pathlib.Path(__file__).parent.resolve()
                / "results"
                / args.results_folder
                / "trajectories"
            )
            trajectories_path.mkdir(parents=True, exist_ok=True)
            for ii in range(trajectories_pred.shape[0]):
                xpred = (
                    trajectories_pred[ii, :][:dimx].reshape((T + 1, base_dim)) / 20.0
                    + 0.5
                )
                xgt = (
                    trajectories_cp[ii, :][:dimx].reshape((T + 1, base_dim)) / 20.0
                    + 0.5
                )
                # Save trajectory to CSV file
                # Create output directory if not exists
                # Stack the columns:
                # x (xpred[:,0]), y (xpred[:,1]), xgt (xgt[:,0]), ygt (xgt[:,1])
                data = np.column_stack((xpred[:, 0], xpred[:, 1], xgt[:, 0], xgt[:, 1]))
                csv_filename = trajectories_path / f"trajectory_{ii+1}.csv"
                np.savetxt(
                    csv_filename,
                    data,
                    delimiter=",",
                    header="x,y,xgt,ygt",
                    comments="",
                    fmt="%.5f",
                )
