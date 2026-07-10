import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_trajectory(traj, start=None, end=None, alpha=1.0):
    """
    Plots a single trajectory given an (n, 2) array of points.

    Parameters:
    - traj: np.ndarray of shape (n, 2), representing the trajectory.
    - start: Optional tuple (x, y) for the start point.
    - end: Optional tuple (x, y) for the end point.
    - alpha: Transparency level for the trajectory.
    """
    plt.plot(traj[:, 0], traj[:, 1], label='Trajectory', alpha=alpha * 0.2)

    if start is not None:
        plt.scatter(*start[:2], color='green', s=100, label='Start', edgecolors='black', alpha=alpha)
    
    if end is not None:
        plt.scatter(*end[:2], color='red', s=100, label='End', edgecolors='black', alpha=alpha)

    plt.gca().set_aspect('equal')

def plot_multiple_trajectories(trajs, alpha=0.5, savedir=None, show=True):
    """
    Plots multiple trajectories given a (B, N, 2) array of points.

    Parameters:
    - trajs: np.ndarray of shape (B, N, 2), where B is the number of trajectories.
    - alpha: Transparency level to visualize overlapping trajectories.
    """
    trajs = jnp.asarray(trajs)
    B = trajs.shape[0]

    plt.figure(figsize=(8, 6))

    for i in range(B):
        start, end = trajs[i, 0], trajs[i, -1]
        plot_trajectory(trajs[i], start=start, end=end, alpha=alpha)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Trajectory', 'Start', 'End'])
    plt.grid(True)
    plt.axis('equal')
    plt.gca().set_aspect('equal')
    if savedir is not None:
        plt.savefig(savedir)
    if show:
        plt.show()
    else:
        plt.close()

def plot_trajectories_with_background(trajs, background_data, alpha=0.5, savedir=None, show=True):
    """
    Plots multiple trajectories over a fixed potential background.
    
    - trajs: (B, N, 2) array.
    - background_data: dictionary containing X, Y, Z meshgrid and potential.
    - alpha: transparency of trajectory plots.
    """
    X, Y, Z = background_data["X"], background_data["Y"], background_data["Z"]
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=100, cmap="magma", alpha=0.3)

    B = trajs.shape[0]
    for i in range(B):
        traj = trajs[i]
        plt.plot(traj[:, 0], traj[:, 1], alpha=alpha * 0.3)
        plt.scatter(*traj[0], color='green', s=50, edgecolors='black', label='Start' if i == 0 else "")
        plt.scatter(*traj[-1], color='red', s=50, edgecolors='black', label='End' if i == 0 else "")

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.axis('equal')
    plt.legend()

    if savedir is not None:
        plt.savefig(savedir)
    if show:    
        plt.show()
    else:        
        plt.close()


def plot_trajectories_1d(trajs, alpha=0.5, savedir=None, show=True):
    trajs = jnp.asarray(trajs)
    if trajs.ndim == 3 and trajs.shape[-1] == 1:
        traj_vals = trajs[..., 0]
    elif trajs.ndim == 2:
        traj_vals = trajs
    else:
        raise ValueError("trajs must have shape (B, N, 1) or (B, N)")
    
    B, N = traj_vals.shape
    t = jnp.linspace(0, 1.0, N)
    
    for i in range(B):
        v_i = traj_vals[i]
        plt.plot(t, v_i, alpha=alpha * 0.8)
        # Start / End markers (labels only once for legend)
        plt.scatter(t[0], v_i[0], s=40, label='Start' if i == 0 else "")
        plt.scatter(t[-1], v_i[-1], s=40, marker='x', label='End' if i == 0 else "")
    
    plt.xlabel('Time step (scaled 0.0 to 1.0)')
    plt.ylabel('Value')
    # plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()
    if savedir is not None:
        plt.savefig(savedir)
    if show:
        plt.show()
    else:
        plt.close()


def plot_two_dim_trajectories(x_trajs, savedir=None, show=True):
    """
    Plot 64 trajectories with two dimensions each on the same graph.
    
    Args:
        x_trajs: array of shape (n_traj, n_time, 2)
    """
    time = jnp.linspace(0, 1.0, x_trajs.shape[1])

    plt.figure(figsize=(4, 3))
    for i, traj in enumerate(x_trajs):
        if i == 0:
            plt.plot(time, traj[:, 0], color='tab:blue', alpha=0.3, label='$X_{t,1}$')
            plt.plot(time, traj[:, 1], color='tab:orange', alpha=0.3, label='$X_{t,2}$')
        else:
            plt.plot(time, traj[:, 0], color='tab:blue', alpha=0.3)
            plt.plot(time, traj[:, 1], color='tab:orange', alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Trajectories, for dim 0 and dim 1")
    plt.legend()
    plt.tight_layout()
    if savedir is not None:
        plt.savefig(savedir)
    if show:
        plt.show()
    else:
        plt.close()

def plot_trajectories_with_background_1d(trajs, background_data, alpha=0.5, savedir=None, show=True):
    """
    Plots multiple 1D trajectories over a fixed 1D double-well potential background.
    
    - trajs: (B, N, 1) or (B, N) array. Time is along x-axis, value along y-axis.
    - background_data: dict with X, Y, Z from `make_1d_potential_background`.
    - alpha: transparency of trajectory lines.
    """
    X, Y, Z = background_data["X"], background_data["Y"], background_data["Z"]
    
    plt.figure(figsize=(9, 6))
    # Heatmap underlay (no explicit colormap set, to follow default styling)
    plt.pcolormesh(X, Y, Z, shading='auto', alpha=0.3)
    
    trajs = jnp.asarray(trajs)
    if trajs.ndim == 3 and trajs.shape[-1] == 1:
        traj_vals = trajs[..., 0]
    elif trajs.ndim == 2:
        traj_vals = trajs
    else:
        raise ValueError("trajs must have shape (B, N, 1) or (B, N)")
    
    B, N = traj_vals.shape
    t = jnp.linspace(0, 1.0, N)
    
    for i in range(B):
        v_i = traj_vals[i]
        plt.plot(t, v_i, alpha=alpha * 0.8)
        # Start / End markers (labels only once for legend)
        plt.scatter(t[0], v_i[0], s=40, label='Start' if i == 0 else "")
        plt.scatter(t[-1], v_i[-1], s=40, marker='x', label='End' if i == 0 else "")
    
    plt.xlabel('Time step (scaled 0.0 to 1.0)')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savedir is not None:
        plt.savefig(savedir)
    if show:
        plt.show()
    else:
        plt.close()