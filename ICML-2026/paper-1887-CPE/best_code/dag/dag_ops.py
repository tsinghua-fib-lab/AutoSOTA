import numpy as np

def adj_from_W(W: np.ndarray) -> np.ndarray:
    """Directed 0/1 adjacency from weights."""
    return (np.abs(W) > 1e-12).astype(int)

def skel_from_adj(A: np.ndarray) -> np.ndarray:
    """Undirected skeleton: 1 if either direction present."""
    return ((A + A.T) > 0).astype(int)


def random_dag(D: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a DAG by random permutation and forward edges."""
    order = rng.permutation(D)
    A = np.zeros((D, D), dtype=int)
    for a in range(D):
        for b in range(a + 1, D):
            i = order[a]
            j = order[b]
            if rng.random() < edge_prob:
                A[i, j] = 1
    return A


def reachable(A: np.ndarray) -> np.ndarray:
    """Transitive closure (Floyd-Warshall style, boolean)."""
    D = A.shape[0]
    R = (A > 0).astype(int).copy()
    for k in range(D):
        for i in range(D):
            if R[i, k]:
                R[i, :] = np.maximum(R[i, :], R[k, :])
    return R


def would_create_cycle(A: np.ndarray, i: int, j: int) -> bool:
    """Check if adding i->j creates a cycle."""
    if A[j, i] == 1:
        return True
    R = reachable(A)
    return bool(R[j, i])


def sample_weights(A: np.ndarray, w_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Assign real weights to existing edges."""
    D = A.shape[0]
    W = np.zeros((D, D), dtype=float)
    m = int(A.sum())
    if m > 0:
        W[A == 1] = rng.normal(0.0, w_scale, size=m)
        # nudge away from zero so log(eps+|W|) is informative
        W[A == 1] += np.sign(W[A == 1]) * 0.5
    return W


def is_acyclic(W):
    """Check if adjacency implied by W is acyclic (simple DFS).
    TODO: can use a more optimized version from somewhere else """
    A = (np.abs(W) > 1e-12).astype(int)
    D = A.shape[0]
    visited = [0]*D
    recstack = [0]*D

    def visit(v):
        visited[v] = 1
        recstack[v] = 1
        for u in range(D):
            if A[v,u]:
                if not visited[u] and visit(u):
                    return True
                elif recstack[u]:
                    return True
        recstack[v] = 0
        return False

    for node in range(D):
        if not visited[node]:
            if visit(node):
                return False
    return True

# def log_expert_likelihood(y, W, i, j, beta, beta0, lam):
#     """Return log p_beta(y | W, i, j)."""
#     s_i2j = beta * np.log(1e-12 + abs(W[i,j])) + lam * 0
#     s_j2i = beta * np.log(1e-12 + abs(W[j,i])) + lam * 0
#     s_none = beta0
#     logits = np.array([s_i2j, s_j2i, s_none])
#     # TODO: use logsumexp
#     logZ = np.log(np.sum(np.exp(logits)))
#     if y == 0: return s_i2j - logZ
#     if y == 1: return s_j2i - logZ
#     return s_none - logZ
