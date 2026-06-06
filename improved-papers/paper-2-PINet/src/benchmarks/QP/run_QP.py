"""Run HCNN on simple QP problem."""

import argparse
import datetime
import pathlib
import time
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import torch
import wandb
from flax.serialization import to_bytes
from flax.training import train_state
from tqdm import tqdm

from benchmarks.model import setup_model
from benchmarks.QP.load_QP import load_data
from benchmarks.QP.plotting import plot_inference_boxes, plot_rs_vs_cv
from src.tools.utils import GracefulShutdown, Logger, load_configuration

# jax.config.update("jax_enable_x64", True)  # Disabled for f32 speedup


class LoggingDict:
    """Class to log results during training."""

    def __init__(self):
        """Initialize the logging dictionary."""
        self.dict = {
            "optimal_objective": [],
            "objective": [],
            "eqcv": [],
            "ineqcv": [],
            "train_time": [],
            "inf_time": [],
        }

    def update(
        self,
        optimal_objective: jnp.ndarray,
        objective: jnp.ndarray,
        eqcv: jnp.ndarray,
        ineqcv: jnp.ndarray,
        train_time: jnp.ndarray,
        inf_time: jnp.ndarray,
    ) -> None:
        """Update the logging dictionary.

        Args:
            optimal_objective (jnp.ndarray): Optimal objective values.
            objective (jnp.ndarray): Objective values.
            eqcv (jnp.ndarray): Equality constraint violations.
            ineqcv (jnp.ndarray): Inequality constraint violations.
            train_time (jnp.ndarray): Training time for the epoch.
            inf_time (jnp.ndarray): Inference time for the epoch.
        """
        self.dict["optimal_objective"].append(optimal_objective)
        self.dict["objective"].append(objective)
        self.dict["eqcv"].append(eqcv)
        self.dict["ineqcv"].append(ineqcv)
        self.dict["train_time"].append(train_time)
        self.dict["inf_time"].append(inf_time)

    def as_array(self, label: str) -> jnp.ndarray:
        """Return the logging dictionary label as a jnp array.

        Args:
            label (str): The label to retrieve from the dictionary.
        """
        return jnp.array(self.dict[label])


# Evaluation function to clean up code
def evaluate_hcnn(
    loader,
    state: train_state.TrainState,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
    A: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    prefix: str,
    time_evals: int = 10,
    tol_cv: float = 1e-3,
    print_res: bool = True,
    single_instance: bool = False,
    instances: list = None,
    proj_method: str = "pinet",
) -> tuple[
    jnp.ndarray,  # Objective values
    jnp.ndarray,  # HCNN objective values
    jnp.ndarray,  # Equality constraint violations
    jnp.ndarray,  # Inequality constraint violations
    jnp.ndarray,  # Evaluation times
]:
    """Evaluate the performance of the HCNN.

    Args:
        loader: Data loader for the problem instances.
        state (train_state.TrainState): The trained model state.
        batched_objective (Callable): Function to compute the objective.
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        G (jnp.ndarray): Coefficient matrix for inequality constraints.
        h (jnp.ndarray): Right-hand side vector for inequality constraints.
        prefix (str): Prefix for logging.
        time_evals (int): Number of evaluations for inference time.
        tol_cv (float): Tolerance for constraint violation.
        print_res (bool): Whether to print the results.
        single_instance (bool): Whether to evaluate a single instance.
        instances (list): List of instances to evaluate if single_instance is True.
        proj_method (str): Projection method used in the model.

    Returns:
        tuple: A tuple containing the objective values, HCNN objective values,
               equality constraint violations, inequality constraint violations,
               and evaluation times.
    """

    def predict(xx):
        return state.apply_fn(
            {"params": state.params},
            x=xx[:, :, 0],
            b=xx,
            test=True,
        )

    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        predictions = predict(X)
    opt_obj = obj.mean()
    # HCNN objective
    hcnn_obj = batched_objective(predictions)
    rs = jnp.mean((hcnn_obj - obj) / jnp.abs(obj))
    # Equality constraint violation
    eq_cv = jnp.max(
        jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X
        ),
        axis=1,
    )
    # Average and max inequality constraint violation
    ineq_cv = jnp.max(
        jnp.maximum(
            G[0].reshape(1, G.shape[1], G.shape[2])
            @ predictions.reshape(X.shape[0], G.shape[2], 1)
            - h,
            0,
        ),
        axis=1,
    )
    # Percentage of constraint satisfaction at tolerance
    perc_cv = (1 - jnp.mean(ineq_cv > tol_cv)) * 100
    # Computation time
    if time_evals > 0:
        # Batch size 1 or full
        if single_instance:
            if instances is None:
                raise ValueError("Single instance evaluation requires instances.")

            eval_times = []
            for ii in instances:
                for rep in range(time_evals + 1):
                    Xtime = X[ii : ii + 1, :, :]
                    start = time.time()
                    predict(Xtime).block_until_ready()
                    # Drop first time cause it includes setups
                    if rep > 0:
                        eval_times.append(time.time() - start)
        else:
            Xtime = X
            eval_times = []
            for rep in range(time_evals + 1):
                start = time.time()
                predict(Xtime).block_until_ready()
                # Drop first time cause it includes setups
                if rep > 0:
                    eval_times.append(time.time() - start)

        eval_times = jnp.array(eval_times)
        eval_time = jnp.mean(eval_times)
    else:
        eval_time = -1
        eval_times = []
    if print_res:
        hcnn_obj_mean = hcnn_obj.mean()
        eq_cv_mean = eq_cv.mean()
        eq_cv_max = eq_cv.max()
        ineq_cv_mean = ineq_cv.mean()
        ineq_cv_max = ineq_cv.max()
        print(f"=========== {prefix} performance ===========")
        print("Mean Relative Suboptimality   : ", f"{rs:.5f}")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max equality violation   : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max inequality violation : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print("Percentage of ineq. cv < tol  : ", f"{perc_cv:.5f} %")
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj:.5f}")

    return (obj, hcnn_obj, eq_cv, ineq_cv, eval_times)


# Evaluate individual instance
def evaluate_instance(
    problem_idx: int,
    loader,
    state: train_state.TrainState,
    use_DC3_dataset: bool,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
    A: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    prefix: str,
    proj_method="pinet",
) -> None:
    """Evaluate performance on single problem instance.

    Args:
        problem_idx (int): Index of the problem instance to evaluate.
        loader: Data loader for the problem instances.
        state (train_state.TrainState): The trained model state.
        use_DC3_dataset (bool): Whether to use the DC3 dataset.
        batched_objective (Callable): Function to compute the objective.
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        G (jnp.ndarray): Coefficient matrix for inequality constraints.
        h (jnp.ndarray): Right-hand side vector for inequality constraints.
        prefix (str): Prefix for logging.
        proj_method (str): Projection method used in the model.
    """
    # Evaluate HCNN solution
    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        pass

    predictions = state.apply_fn(
        {"params": state.params},
        x=X[problem_idx, :, 0].reshape((1, X.shape[1])),
        b=X[problem_idx].reshape((1, X.shape[1], 1)),
        test=True,
    )

    objective_val_hcnn = batched_objective(predictions).item()
    eqcv_val_hcnn = jnp.abs(
        A[0] @ predictions.reshape(A.shape[2]) - X[problem_idx, :, 0]
    ).max()
    ineqcv_val_hcnn = jnp.maximum(
        G[0] @ predictions.reshape(G.shape[2]) - h[0, :, 0], 0
    ).max()
    print(f"=========== {prefix} individual performance ===========")
    print("HCNN")
    print(f"Objective:  \t{objective_val_hcnn:.5e}")
    print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")

    # Evaluate optimal solution
    if not use_DC3_dataset:
        objective_val = loader.dataset.dataset.objectives[
            loader.dataset.indices[problem_idx]
        ]
        eqcv_val = jnp.abs(
            A[0] @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]]
            - loader.dataset.dataset.X[loader.dataset.indices[problem_idx], :, :]
        ).max()
        ineqcv_val = jnp.maximum(
            G[0] @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]]
            - h[0, :, :],
            0,
        ).max()

    else:
        objective_val = loader.dataset.objectives[problem_idx].item()
        eqcv_val = jnp.abs(
            A[0] @ loader.dataset.Ystar[problem_idx]
            - loader.dataset.X[problem_idx, :, :]
        ).max()
        ineqcv_val = jnp.maximum(
            G[0] @ loader.dataset.Ystar[problem_idx] - h[0, :, :], 0
        ).max()

    print("Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")


def main(
    use_DC3_dataset: bool,
    use_convex: bool,
    problem_seed: int,
    problem_var: float,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    use_jax_loader: bool,
    config_path: str,
    SEED: int,
    proj_method: str,
    SAVE_RESULTS: bool,
    run_name: str,
    current_timestamp: str,
) -> train_state.TrainState:
    """Main for running simple QP benchmarks.

    Args:
        use_DC3_dataset (bool): Whether to use the DC3 dataset or not.
        use_convex (bool): Whether to use a convex problem or not.
        problem_seed (int): Seed for the problem generation.
        problem_var (float): Variance for the problem generation.
        problem_nineq (int): Number of inequality constraints.
        problem_neq (int): Number of equality constraints.
        problem_examples (int): Number of problem examples.
        use_jax_loader (bool): Whether to use the JAX loader or not.
        config_path (str): Path to the configuration file.
        SEED (int): Seed for training.
        proj_method (str): Projection method to use.
        SAVE_RESULTS (bool): Whether to save the results or not.
        run_name (str): Name of the run for logging.
        current_timestamp (str): Current timestamp for result folder naming.

    Returns:
        state (TrainState): The final state of the trained model.
    """
    # Load hyperparameter configuration
    hyperparameters = load_configuration(config_path)
    torch.manual_seed(SEED)
    key = jax.random.PRNGKey(SEED)
    loader_key, key = jax.random.split(key, 2)
    # Load problem data
    (
        A,
        G,
        h,
        X,
        batched_objective,
        train_loader,
        valid_loader,
        test_loader,
        batched_loss,
    ) = load_data(
        use_DC3_dataset=use_DC3_dataset,
        use_convex=use_convex,
        problem_seed=problem_seed,
        problem_var=problem_var,
        problem_nineq=problem_nineq,
        problem_neq=problem_neq,
        problem_examples=problem_examples,
        rng_key=loader_key,
        batch_size=hyperparameters.get("batch_size", 2048),
        use_jax_loader=use_jax_loader,
        penalty=hyperparameters.get("penalty", 0.0),
    )

    model, params, setup_time, train_step = setup_model(
        rng_key=key,
        hyperparameters=hyperparameters,
        proj_method=proj_method,
        A=A,
        X=X,
        G=G,
        h=h,
        batched_loss=batched_loss,
    )

    LEARNING_RATE = hyperparameters["learning_rate"]
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    if proj_method == "pinet":
        # Measure compilation time
        for batch in train_loader:
            X_batch, _ = batch
        start_compilation_time = time.time()
        _ = train_step.lower(
            state,
            X_batch[:, :, 0],
            X_batch,
        ).compile()
        # Note this also includes the time for one iteration
        compilation_time = time.time() - start_compilation_time

        print(f"Compilation time: {compilation_time:.5f} seconds")

    # Train the MLP
    N_EPOCHS = hyperparameters["n_epochs"]
    eval_every = 1
    start_training_time = time.time()
    trainig_losses = []
    validation_losses = []
    eqcvs = []
    ineqcvs = []
    logging_dict = LoggingDict()
    with (
        Logger(run_name=run_name, project_name="hcnn") as data_logger,
        GracefulShutdown("Stop detected, finish epoch...") as g,
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
                loss, state = train_step(
                    state,
                    X_batch[:, :, 0],
                    X_batch,
                )
                batch_sizes.append(X_batch.shape[0])
                epoch_loss.append(loss)
            weighted_epoch_loss = sum(
                el * bs for el, bs in zip(epoch_loss, batch_sizes)
            ) / sum(batch_sizes)
            trainig_losses.append(weighted_epoch_loss)
            pbar.set_description(f"Train Loss: {weighted_epoch_loss:.5f}")
            train_time = time.time() - start_epoch_time

            if step % eval_every == 0:
                start_evaluation_time = time.time()
                obj, hcnn_obj, eq_cv, ineq_cv, _ = evaluate_hcnn(
                    loader=valid_loader,
                    state=state,
                    batched_objective=batched_objective,
                    A=A,
                    G=G,
                    h=h,
                    prefix="Validation",
                    time_evals=0,
                    print_res=False,
                    proj_method=proj_method,
                )
                eqcvs.append(eq_cv.max())
                ineqcvs.append(ineq_cv.max())
                validation_losses.append(hcnn_obj.mean())
                pbar.set_postfix(
                    {
                        "eqcv": f"{eq_cv.mean():.5f}",
                        "ineqcv": f"{ineq_cv.mean():.5f}",
                        "Valid. Loss:": f"{hcnn_obj.mean():.5f}",
                    }
                )
                eval_time = time.time() - start_evaluation_time
                logging_dict.update(
                    obj,
                    hcnn_obj,
                    eq_cv,
                    ineq_cv,
                    train_time,
                    eval_time,
                )
                data_logger.log(
                    step,
                    {
                        "weighted_epoch_loss": weighted_epoch_loss,
                        "batch_training_time": train_time,
                        "validation_objective_mean": hcnn_obj.mean(),
                        "validation_average_rs": (
                            (hcnn_obj - obj) / jnp.abs(obj)
                        ).mean(),
                        "validation_eqcv_mean": eq_cv.mean(),
                        "validation_ineqcv_mean": ineq_cv.mean(),
                        "validation_time": eval_time,
                    },
                )

        training_time = time.time() - start_training_time
        print(f"Training time: {training_time:.5f} seconds")

        # Evaluate validation performance
        _ = evaluate_hcnn(
            loader=valid_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Validation",
            A=A,
            G=G,
            h=h,
            proj_method=proj_method,
        )
        # Solve some validation individual problem
        problem_idx = 4
        evaluate_instance(
            problem_idx=problem_idx,
            loader=valid_loader,
            state=state,
            use_DC3_dataset=use_DC3_dataset,
            batched_objective=batched_objective,
            A=A,
            G=G,
            h=h,
            prefix="Validation",
            proj_method=proj_method,
        )
        # Evaluate test performance
        (
            obj_test,
            obj_fun_test,
            eq_viol_test,
            ineq_viol_test,
            batch_inference_times,
        ) = evaluate_hcnn(
            loader=test_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Testing",
            A=A,
            G=G,
            h=h,
            proj_method=proj_method,
        )
        # Evaluate for single inference time
        (_, _, _, _, single_inference_times) = evaluate_hcnn(
            loader=test_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Testing",
            A=A,
            G=G,
            h=h,
            single_instance=True,
            instances=list(range(10)),
            proj_method=proj_method,
        )
        # Solve some test problems
        problem_idx = 0
        evaluate_instance(
            problem_idx=problem_idx,
            loader=test_loader,
            state=state,
            use_DC3_dataset=use_DC3_dataset,
            batched_objective=batched_objective,
            A=A,
            G=G,
            h=h,
            prefix="Testing",
            proj_method=proj_method,
        )

        # Log figures
        cvthres = 1e-3
        rsthres = 5e-2
        fig, rs, cv = plot_rs_vs_cv(
            obj_fun_test=obj_fun_test,
            obj_test=obj_test,
            eq_viol_test=eq_viol_test,
            ineq_viol_test=ineq_viol_test,
            cvthres=cvthres,
            rsthres=rsthres,
        )
        data_logger.run.log({"RS vs CV": wandb.Image(fig)})
        fig = plot_inference_boxes(single_inference_times, batch_inference_times)
        data_logger.run.log({"Inference Times": wandb.Image(fig)})

        # Log summary metrics for wandb
        data_logger.run.summary.update(
            {
                "Average RS Test": jnp.mean(rs),
                "Max CV Test": jnp.max(cv),
                "Percentage CV < tol": (1 - jnp.mean(cv > cvthres)) * 100,
                "Average Single Inference Time": jnp.mean(single_inference_times),
                "Average Batch Inference Time": jnp.mean(batch_inference_times),
            },
        )

        # Saving of overall results
        if SAVE_RESULTS:
            # Setup results path
            filename_results = "results.npz"
            results_folder = (
                pathlib.Path(__file__).parent
                / "results"
                / args.id
                / args.config
                / current_timestamp
            )
            results_folder.mkdir(parents=True, exist_ok=True)
            # Save final results
            jnp.savez(
                file=results_folder / filename_results,
                inference_time=batch_inference_times,
                single_inference_time=single_inference_times,
                setup_time=setup_time,
                compilation_time=compilation_time,
                training_time=training_time,
                eq_viol_test=eq_viol_test,
                ineq_viol_test=ineq_viol_test,
                obj_fun_test=obj_fun_test,
                opt_obj_test=obj_test,
                config_path=config_path,
                **hyperparameters,
            )
            # Save learning curve results
            jnp.savez(
                file=results_folder / "learning_curves.npz",
                optimal_objective=logging_dict.as_array("optimal_objective"),
                objective=logging_dict.as_array("objective"),
                eqcv=logging_dict.as_array("eqcv"),
                ineqcv=logging_dict.as_array("ineqcv"),
                train_time=logging_dict.as_array("train_time"),
                inf_time=logging_dict.as_array("inf_time"),
            )
            wandb.save(results_folder / filename_results)
            wandb.save(results_folder / "learning_curves.npz")
            # Save network parameters
            params_filename = "params.msgpack"
            params_path = results_folder / params_filename
            with open(params_path, "wb") as f:
                f.write(to_bytes(state.params))

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HCNN on simple QP or nonconvex problem."
    )
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Yaml file specifying the dataset, see the `ids` folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config_manual",
        help="Configuration file for hyperparameters.",
    )
    parser.add_argument(
        "--proj_method",
        type=str,
        default="pinet",
        help="Projection method. Options are: pinet, cvxpy, jaxopt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for training HCNN.")
    parser.add_argument("--save_results", action="store_true", help="Save the results.")
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Don't save the results.",
    )
    parser.add_argument(
        "--jax_loader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the jax loader or not. If not, use pytorch loader.",
    )
    parser.set_defaults(save_results=True)

    args = parser.parse_args()
    # Load the yaml file
    idpath = pathlib.Path(__file__).parent.resolve() / "ids" / (args.id + ".yaml")
    dataset = load_configuration(idpath)
    # Use existing DC3 Dataset or own dataset
    use_DC3_dataset = dataset["use_DC3_dataset"]
    use_convex = dataset["use_convex"]
    # Import dataset
    problem_seed = dataset["problem_seed"]
    problem_var = dataset["problem_var"]
    problem_nineq = dataset["problem_nineq"]
    problem_neq = dataset["problem_neq"]
    problem_examples = dataset["problem_examples"]
    use_jax_loader = args.jax_loader
    # Configs path
    config_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "configs"
        / (args.config + ".yaml")
    )
    SEED = args.seed
    proj_method = args.proj_method
    SAVE_RESULTS = args.save_results

    nowstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.config}_{'simple' if use_convex else 'nonconvex'}_{nowstamp}"

    main(
        use_DC3_dataset,
        use_convex,
        problem_seed,
        problem_var,
        problem_nineq,
        problem_neq,
        problem_examples,
        use_jax_loader,
        config_path,
        SEED,
        proj_method,
        SAVE_RESULTS,
        run_name,
        nowstamp,
    )
