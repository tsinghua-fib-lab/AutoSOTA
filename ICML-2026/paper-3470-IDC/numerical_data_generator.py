import utils_latent as ut
import numpy as np


# ---------- Utilities ----------

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def make_mlp(input_dim, hidden_dim=8, output_dim=1):
    params = {
        "W1": np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim),
        "b1": np.zeros(hidden_dim),
        "W2": np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim),
        "b2": np.zeros(output_dim),
    }
    return params


def mlp_forward(x, params, alpha=0.01):
    h = leaky_relu(x @ params["W1"] + params["b1"], alpha=alpha)
    y = h @ params["W2"] + params["b2"]
    return y.squeeze(-1)


def topological_order(adj):
    n = adj.shape[0]
    visited = set()
    order = []

    def dfs(i):
        if i in visited:
            return
        visited.add(i)
        for j in np.where(adj[i] == 1)[0]:
            dfs(j)
        order.append(i)

    for i in range(n):
        dfs(i)

    return order[::-1]


def generate_Z_nonlinear_nn(E_t, B, skeleton, hidden_dim=8, alpha=0.01):
    """
    Generates Z via nonlinear neural SEM:
        Z_j = MLP_j(Z_parents) + e_j
    """
    N, n = E_t.shape
    Z = np.zeros((N, n))

    # Build one MLP per node
    mlps = {}
    parents_list = []

    for j in range(n):
        parents = np.where(skeleton[:, j] == 1)[0]
        parents_list.append(parents)

        input_dim = len(parents) if len(parents) > 0 else 1
        mlps[j] = make_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)

    order = topological_order(skeleton)

    for j in order:
        parents = parents_list[j]

        # root node
        if len(parents) == 0:
            Z[:, j] = E_t[:, j]

        else:
            pa_vals = Z[:, parents]

            if pa_vals.ndim == 1:
                pa_vals = pa_vals[:, None]

            pred = mlp_forward(pa_vals, mlps[j], alpha=alpha)
            Z[:, j] = pred + E_t[:, j]

    return Z


# ---------- Main Data Generator ----------

def generate_data(
    T=4,
    n=2,
    xn=2,
    N=1000,
    graph_dense=1,
    graph_type="ER",
    noise_type="gauss",
    monomial=False,
    nn_nonlinear=False,
):
    """
    If nn_nonlinear=False -> linear SEM (original version)
    If nn_nonlinear=True  -> neural SCM over Z with LeakyReLU MLPs
    """

    # Different Ds for each domain
    Ds = [np.random.uniform(0.2, 1, size=n) for _ in range(T)]

    # Sigma diagonal with different entries
    sigma_vec = np.random.uniform(0.2, 1, size=n)
    sigma = np.diag(sigma_vec)

    # --- Simulate DAG + parameters ---
    skeleton = ut.simulate_dag(n, int(n * graph_dense), graph_type)  # adjacency mask
    B = ut.simulate_parameter(skeleton)

    I_B_inv = np.linalg.inv(np.eye(n) - B)

    # --- Mixing matrix A ---
    if monomial:
        A = make_sparse_full_rank(xn, n)
    else:
        if xn > n:
            A = np.random.randn(xn, n)
        else:
            U, _ = np.linalg.qr(np.random.randn(n, n))
            V, _ = np.linalg.qr(np.random.randn(n, n))
            s = np.random.uniform(1.0, 5.0, size=n)
            A = U @ np.diag(s) @ V.T

    X_list, Z_list = [], []

    for t in range(T):

        # -------- Noise --------
        if noise_type == "gauss":
            E_t = np.random.multivariate_normal(np.zeros(n), sigma @ sigma.T, size=N)

        elif noise_type == "exp":
            E_t = (np.random.exponential(scale=1.0, size=(N, n)) - 1.0) * sigma_vec

        elif noise_type == "gumbel":
            E_t = np.random.gumbel(loc=0.0, scale=1.0, size=(N, n)) * sigma_vec

        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

        # -------- Latent Z generation --------
        if nn_nonlinear:
            Z_t = generate_Z_nonlinear_nn(E_t, B, skeleton, hidden_dim=100, alpha=0.01)

        else:
            # linear SCM baseline (original)
            Z_t = (I_B_inv @ E_t.T).T

        # observed X
        X_t = (A @ np.diag(Ds[t]) @ Z_t.T).T

        Z_list.append(Z_t)
        X_list.append(X_t)

    return X_list, A, B, Ds, sigma_vec, Z_list, skeleton


def make_sparse_full_rank(xn,n):

    assert xn >= n, "Need at least xn >= n to get full column rank"

    A = np.zeros((xn, n))

    # Step 1: ensure each column has 1 non-zero entry (full rank)
    base_rows = np.random.choice(xn, n, replace=False)
    for j in range(n):
        A[j, j] = np.random.uniform(0, 1)

    # Step 2: fill remaining rows with exactly 1 non-zero entry
    remaining_rows = [r for r in range(xn) if r not in base_rows]
    for r in remaining_rows:
        col = np.random.randint(0, n)
        A[r, col] = np.random.uniform(0, 1)


    return A

