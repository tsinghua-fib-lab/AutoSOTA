"""
Ising Solvers - MiP-CRIM implementation
Local-Minima-Preserving Polynomial Relaxation of Ising Problems
Accepted at ICML 2026

Authors: Debraj Banerjee, Santanu Mahapatra, Kunal N. Chaudhury
"""

import networkx as nx
import pandas as pd
import numpy as np


def data2graph(path):
    """
    Convert G-set graph data file to graph and adjacency matrix.
    
    Parameters
    ----------
    path : str
        Path to the graph data file
        
    Returns
    -------
    G : networkx.Graph
        The graph object
    adj_matrix : numpy array
        Adjacency matrix as float32
    """
    # Read the data from the text file, skipping the first line
    data = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None)

    # Extract nodes and weights
    nodes = data.iloc[:, 0:2].values.astype(int)
    weights = data.iloc[:, 2].values

    # Create the graph
    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_edge(nodes[i, 0], nodes[i, 1], weight=weights[i])

    # Compute adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    return G, np.array(adj_matrix).astype(np.float32)

def eng2cut(energy, J_sum):
    """
    Compute cut value from Ising configuration and coupling matrix.
    
    Parameters
    ----------
    energy : float
        Energy of the spin configuration
    J_sum : float
        Sum of all elements in the coupling matrix J
        
    Returns
    -------
    cut_value : float
        The cut value
    """
    cut_value = -(J_sum / 4 + energy / 2)
    return cut_value


def MiP_CRIM(
    J, x_init, T=200, K=10,
    alpha=None, beta=None, lambda_=None,
    step=0.01,    # Adam learning rate
    beta1=0.9, beta2=0.999, eps=1e-8,
    sigma_noise=1e-3, sigma_noise_start=None, sigma_noise_end=None, n_trajectories=1, u=None,
    rng=None, return_all=False
):
    """
    MiP-CRIM: Minima Preserving Continuous Relaxation for Ising Model
    
    Minimizes over the box [-lambda_, lambda_]^n the energy
        H(x) = (beta/4) * ||x||_4^4 - (1/2) x^T (J + alpha I) x
    using Adam, then thresholds the final iterate to ±1 to get spin vector s.

    Parameters
    ----------
    J : (n, n) numpy array
        Symmetric coupling matrix with zero diagonal
    x_init : (n,) numpy array
        Initial point
    K : int
        Outer iterations (epochs)
    T : int
        Inner iterations (optimization steps per epoch)
    alpha : float
        Model parameter (must be provided)
    beta : float
        Model parameter (must be provided)
    lambda_ : float
        Model parameter (must be provided)
    step : float
        Adam learning rate
    beta1 : float
        Adam momentum parameter
    beta2 : float
        Adam momentum parameter
    eps : float
        Adam numerical stabilizer
    sigma_noise : float
        Variance for Gaussian sampling of next x_init
    n_trajectories : int
        Number of Adam trajectories per outer epoch (default 1)
    u : optional
        One-flip local minima test flag
    rng : np.random.Generator or None
        Random number generator
    return_all : bool
        If True, also return S_opt list

    Returns
    -------
    spin_star : (n,) int array in {-1, +1}
        Optimal spin configuration
    S_opt : list (optional)
        List of candidate spin vectors (only if return_all=True)
    """

    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    if x_init.shape[0] != n:
        raise ValueError("x_init length mismatch with J")

    if alpha is None or beta is None or lambda_ is None:
        raise ValueError("alpha, beta, lambda_ must be provided")

    if rng is None:
        rng = np.random.default_rng()

    A_alpha = J + alpha * np.eye(n)

    # Annealing schedule for sigma_noise
    if sigma_noise_start is not None and sigma_noise_end is not None:
        sigma_start = sigma_noise_start
        sigma_end = sigma_noise_end
        use_annealing = True
    else:
        sigma_start = sigma_noise
        sigma_end = sigma_noise
        use_annealing = False

    x_in = x_init.astype(float).copy()
    S_opt = []
    best_energy = -np.inf
    best_spin = None

    for outer in range(K):
        # Multi-trajectory: run M trajectories, keep best
        best_traj_spin = None
        best_traj_energy = -np.inf
        best_traj_y = None

        for traj in range(n_trajectories):
            # Perturb starting point differently for each trajectory
            if traj == 0:
                x = x_in.copy()
            else:
                # Add extra perturbation for diversity
                x = x_in + rng.normal(0, 0.1 * lambda_, size=n)

            # Adam state
            m = np.zeros_like(x)
            v = np.zeros_like(x)

            # ----- inner Adam loop -----
            for t in range(1, T + 1):
                # gradient: g(x) = beta * x^3 - (J + alpha I) x
                g_x = beta * (x**3) - A_alpha.dot(x)

                # Adam updates
                m = beta1 * m + (1.0 - beta1) * g_x
                v = beta2 * v + (1.0 - beta2) * (g_x * g_x)
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)

                # gradient step + projection onto [-lambda_, lambda_]^n
                x = x - step * m_hat / (np.sqrt(v_hat) + eps)
                x = np.clip(x, -lambda_, lambda_)

            # ----- threshold to +-1 spins at radius lambda -----
            s_T = np.where(x >= 0, 1, -1).astype(int)
            y_T = lambda_ * s_T

            # Always test one-flip local minima condition
            g_y = beta * (y_T.astype(float)**3) - A_alpha.dot(y_T.astype(float))
            if np.allclose(np.sign(g_y), -s_T, atol=1e-12):
                S_opt.append(s_T.copy())

            # energy of spin vector (original discrete Ising objective)
            energy = float(s_T.T.dot(J).dot(s_T))
            if energy > best_energy:
                best_energy = energy
                best_spin = s_T.copy()

            # Track best trajectory for next epoch
            if energy > best_traj_energy:
                best_traj_energy = energy
                best_traj_spin = s_T.copy()
                best_traj_y = y_T.copy()

        # Use best trajectory's result for next epoch
        s_T = best_traj_spin
        y_T = best_traj_y

        # sample next x_in ~ N(s_T, sigma*I) with annealing
        if use_annealing:
            frac = (outer + 1) / K
            sigma_k = sigma_start * (sigma_end / sigma_start) ** frac
        else:
            sigma_k = sigma_noise
        if sigma_k < 0:
            raise ValueError("sigma_noise must be non-negative")
        if sigma_k == 0:
            x_in = s_T.astype(float).copy()
        else:
            x_in = rng.normal(loc=y_T.astype(float),
                              scale=np.sqrt(sigma_k),
                              size=n)

    # Prefer sync-validated solutions (S_opt) over raw best energy
    if len(S_opt) == 0:
        spin_star = best_spin.copy()
    else:
        best_e = -np.inf
        best_s = None
        for s in S_opt:
            e = float(s.T.dot(J).dot(s))
            if e > best_e:
                best_e = e
                best_s = s.copy()
        spin_star = best_s.copy()  # Always prefer sync-validated
    if return_all:
        return spin_star, S_opt
    return spin_star
