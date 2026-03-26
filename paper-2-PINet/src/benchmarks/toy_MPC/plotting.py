"""Plotting functionalities for toy MPC."""

import cvxpy as cp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.training.train_state import TrainState

from .load_toy_MPC import JaxDataLoader, ToyMPCDataset


def plot_training(
    train_loader: ToyMPCDataset | JaxDataLoader,
    valid_loader: ToyMPCDataset | JaxDataLoader,
    training_losses: jnp.ndarray,
    validation_losses: jnp.ndarray,
    eqcvs: jnp.ndarray,
    ineqcvs: jnp.ndarray,
) -> None:
    """Plot training curves."""
    opt_train_loss = []
    for batch in train_loader:
        _, obj_batch = batch
        opt_train_loss.append(obj_batch)
    opt_train_loss = jnp.concatenate(opt_train_loss, axis=0).mean()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(training_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=opt_train_loss,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Training Objective",
    )
    plt.legend()

    opt_valid_loss = []
    for batch in valid_loader:
        _, obj_batch = batch
        opt_valid_loss.append(obj_batch)
    opt_valid_loss = jnp.array(opt_valid_loss).mean()
    plt.subplot(1, 4, 2)
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=opt_valid_loss,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Validation Objective",
    )
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.semilogy(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_trajectories(
    state: TrainState,
    As: jnp.ndarray,
    lbxs: jnp.ndarray,
    ubxs: jnp.ndarray,
    lbus: jnp.ndarray,
    ubus: jnp.ndarray,
    alpha: float,
    base_dim: int,
    Y_DIM: int,
    dimx: int,
    xhat: jnp.ndarray,
    T: int,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates trajectories from pinet and solver.

    Args:
        state (TrainState): The trained model state.
        As (jnp.ndarray): The equality constraint matrix.
        lbxs (jnp.ndarray): Lower bounds for state variables.
        ubxs (jnp.ndarray): Upper bounds for state variables.
        lbus (jnp.ndarray): Lower bounds for control inputs.
        ubus (jnp.ndarray): Upper bounds for control inputs.
        alpha (float): Regularization parameter.
        base_dim (int): Dimension of the base state.
        Y_DIM (int): Total dimension of the decision variable.
        dimx (int): Dimension of the state.
        xhat (jnp.ndarray): Reference state.
        T (int): Time horizon.
        lb (jnp.ndarray): Lower bounds for the state.
        ub (jnp.ndarray): Upper bounds for the state.

    Returns:
        tuple: A tuple containing:
            - trajectories (jnp.ndarray): Predicted trajectories from the model.
            - trajectories_cp (jnp.ndarray): Trajectories computed using cvxpy.
    """
    ntraj = 1
    xinit = jnp.array([[-7, -5]]).reshape(ntraj, base_dim, 1)
    # Evaluate the network on these initial points
    Xinitfull = jnp.concatenate(
        (xinit, jnp.zeros((xinit.shape[0], As.shape[1] - xinit.shape[1], 1))), axis=1
    )
    trajectories = state.apply_fn(
        {"params": state.params},
        xinit[:, :, 0],
        Xinitfull,
        test=True,
    )
    # Solve exact problems with cvxpy
    trajectories_cp = jnp.zeros((ntraj, Y_DIM, 1))
    for i in range(ntraj):
        xcp = cp.Variable(Y_DIM)
        xinitcp = cp.Parameter(base_dim.item())
        constraints = [
            As[0] @ xcp == cp.hstack([xinitcp, np.zeros(dimx - base_dim)]),
            xcp[:dimx] >= lbxs[0, :, 0],
            xcp[:dimx] <= ubxs[0, :, 0],
            xcp[dimx:] >= lbus[0, :, 0],
            xcp[dimx:] <= ubus[0, :, 0],
        ]
        objective = cp.Minimize(
            cp.sum_squares(xcp[:dimx] - jnp.tile(xhat[:, 0], T + 1))
            + alpha * cp.sum_squares(xcp[dimx:])
        )
        problem = cp.Problem(objective, constraints)
        # Setup problem parameter
        xinitcp.value = np.array(xinit[i, :, 0])
        problem.solve(verbose=False)
        trajectories_cp = trajectories_cp.at[i].set(
            jnp.asarray(xcp.value).reshape(-1, 1)
        )

    def plot_trajectory(trajectory_pred, trajectory_cp):
        """Plots the trajectory in z."""
        xpred = trajectory_pred[:dimx]
        xpred = xpred.reshape((T + 1, base_dim))
        # Ground truth trajectory
        xgt = trajectory_cp[:dimx]
        xgt = xgt.reshape((T + 1, base_dim))
        plt.plot(xpred[:, 0], xpred[:, 1], "-o", label="Prediction")
        plt.plot(xgt[:, 0], xgt[:, 1], "--*", label="Ground Truth")
        plt.plot(xhat[0], xhat[1], "rx", markersize=10, label="Goal")
        # Plot the bounds of x as a rectangle
        rect = plt.Rectangle(
            (lb[0, 0, 0], lb[0, 1, 0]),
            ub[0, 0, 0] - lb[0, 0, 0],
            ub[1, 0, 0] - lb[1, 0, 0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
            label="Bounds",
        )
        plt.gca().add_patch(rect)
        plt.legend()
        plt.show()

    for ii in range(ntraj):
        plot_trajectory(trajectories[ii, :], trajectories_cp[ii, :, 0])

    return trajectories, trajectories_cp
